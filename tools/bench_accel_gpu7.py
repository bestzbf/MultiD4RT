"""GPU7 microbenchmark for D4RT training acceleration.

Runs the same forward+backward+optimizer step under several acceleration knobs
and reports mean/p50/p95 step times plus fwd/bwd/opt sub-timings.

No dataset, no DDP, no checkpoint loading. Random tensors with the same shapes
as the production 5-dataset config (clip_len=48, img_size=256, num_queries=2048,
patch_size=(2,14,14), bf16 AMP).

Usage:
    CUDA_VISIBLE_DEVICES=7 python tools/bench_accel_gpu7.py \
        --variant giant --batch-size 2 --steps 30 --warmup 5

Each configuration is run in a fresh CUDA context (re-creating model+optimizer).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import create_d4rt
from losses import D4RTLoss


@dataclass
class Config:
    name: str
    desc: str
    apply: Callable[[], contextlib.AbstractContextManager]


@dataclass
class StepStats:
    total_ms: list[float] = field(default_factory=list)
    fwd_ms: list[float] = field(default_factory=list)
    bwd_ms: list[float] = field(default_factory=list)
    opt_ms: list[float] = field(default_factory=list)
    peak_mem_gb: float = 0.0


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round(p / 100.0 * (len(ys) - 1)))))
    return ys[k]


def _mk_inputs(args, device):
    B, T, H, W = args.batch_size, args.num_frames, args.resolution, args.resolution
    Nq = args.num_queries
    video = torch.rand(B, T, H, W, 3, device=device, dtype=torch.float32)
    coords = torch.rand(B, Nq, 2, device=device, dtype=torch.float32)
    t_src = torch.randint(0, T, (B, Nq), device=device, dtype=torch.long)
    t_tgt = torch.randint(0, T, (B, Nq), device=device, dtype=torch.long)
    t_cam = torch.randint(0, T, (B, Nq), device=device, dtype=torch.long)
    aspect_ratio = torch.ones(B, 1, device=device, dtype=torch.float32)
    dataset_id = torch.zeros(B, dtype=torch.long, device=device)
    # Build positive non-degenerate 3D positions so log_transform / median
    # normalization don't blow up.  The benchmark is purely about wall-clock
    # compute, the actual loss values do not need to match real training.
    targets = {
        "pos_3d": torch.rand(B, Nq, 3, device=device) + 0.5,  # depth in [0.5, 1.5]
        "pos_2d": torch.rand(B, Nq, 2, device=device),
        "visibility": torch.ones(B, Nq, device=device),
        "displacement": torch.randn(B, Nq, 3, device=device) * 0.01,
        "mask_3d": torch.ones(B, Nq, dtype=torch.bool, device=device),
        "mask_2d": torch.ones(B, Nq, dtype=torch.bool, device=device),
        "mask_vis": torch.ones(B, Nq, dtype=torch.bool, device=device),
        "mask_disp": torch.ones(B, Nq, dtype=torch.bool, device=device),
    }
    return {
        "video": video,
        "coords": coords,
        "t_src": t_src,
        "t_tgt": t_tgt,
        "t_cam": t_cam,
        "aspect_ratio": aspect_ratio,
        "targets": targets,
        "dataset_id": dataset_id,
    }


def _build_model(args, device, *, fused_opt: bool):
    model = create_d4rt(
        variant=args.variant,
        img_size=args.resolution,
        temporal_size=args.num_frames,
        patch_size=(2, args.encoder_patch_size, args.encoder_patch_size),
        query_patch_size=9,
        videomae_model=None,
        use_videomae_v2=False,
        encoder_pretrained=False,
        patch_provider="sampled_resized",
    ).to(device)
    model.train()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        fused=fused_opt,
    )
    loss_fn = D4RTLoss(
        lambda_3d=1.0,
        lambda_2d=0.1,
        lambda_vis=0.1,
        lambda_disp=0.1,
        lambda_conf=0.1,
        lambda_normal=0.0,
        static_reprojection_weight=1.0,
    )
    return model, optim, loss_fn


def _run_one(args, cfg: Config, device) -> StepStats:
    """Apply config, build model fresh, run warmup+steps, return stats."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    fused_opt = (cfg.name in ("C_fused", "D_tf32", "E_flash", "F_cudnn", "G_compile"))
    use_compile = (cfg.name == "G_compile")
    use_cudnn_bench = (cfg.name in ("F_cudnn", "G_compile"))
    use_tf32 = (cfg.name in ("D_tf32", "E_flash", "F_cudnn", "G_compile"))
    force_flash = (cfg.name in ("E_flash", "F_cudnn", "G_compile"))
    keep_sync = (cfg.name == "A_baseline")

    if use_tf32:
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = bool(use_cudnn_bench)

    sdpa_ctx = contextlib.nullcontext()
    if force_flash:
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            sdpa_ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        except Exception:
            sdpa_ctx = contextlib.nullcontext()

    model, optim, loss_fn = _build_model(args, device, fused_opt=fused_opt)
    if use_compile:
        model = torch.compile(model, mode="default", dynamic=False)

    inputs = _mk_inputs(args, device)
    stats = StepStats()
    total_iters = args.warmup + args.steps

    with sdpa_ctx:
        for it in range(total_iters):
            if keep_sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                t_fs = time.perf_counter()
                outputs = model(
                    inputs["video"],
                    inputs["coords"],
                    inputs["t_src"],
                    inputs["t_tgt"],
                    inputs["t_cam"],
                    aspect_ratio=inputs["aspect_ratio"],
                )
                if keep_sync:
                    torch.cuda.synchronize()
                t_fwd_ms = (time.perf_counter() - t_fs) * 1000.0

                normalize_groups = inputs["dataset_id"].unsqueeze(1) * args.num_frames + inputs["t_cam"]
                loss_dict = loss_fn(outputs, inputs["targets"], normalize_groups=normalize_groups)
                loss = loss_dict["loss"]

            t_bs = time.perf_counter()
            loss.backward()
            if keep_sync:
                torch.cuda.synchronize()
            t_bwd_ms = (time.perf_counter() - t_bs) * 1000.0

            t_os = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            torch.cuda.synchronize()  # always sync at end so timing is accurate
            t_opt_ms = (time.perf_counter() - t_os) * 1000.0

            t_total_ms = (time.perf_counter() - t0) * 1000.0

            if it >= args.warmup:
                stats.total_ms.append(t_total_ms)
                stats.fwd_ms.append(t_fwd_ms)
                stats.bwd_ms.append(t_bwd_ms)
                stats.opt_ms.append(t_opt_ms)

            if it == 0:
                print(f"  [{cfg.name}] first iter: total={t_total_ms:.0f}ms (compile/warmup)", flush=True)

    stats.peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    del model, optim, loss_fn, inputs
    torch.cuda.empty_cache()
    return stats


CONFIGS: list[Config] = [
    Config("A_baseline", "no sync removed, no fused, no TF32, default SDPA", lambda: contextlib.nullcontext()),
    Config("B_nosync",   "skip per-step torch.cuda.synchronize calls",      lambda: contextlib.nullcontext()),
    Config("C_fused",    "+ fused AdamW",                                   lambda: contextlib.nullcontext()),
    Config("D_tf32",     "+ set_float32_matmul_precision('high')",           lambda: contextlib.nullcontext()),
    Config("E_flash",    "+ force SDPBackend.FLASH_ATTENTION",               lambda: contextlib.nullcontext()),
    Config("F_cudnn",    "+ cudnn.benchmark=True",                           lambda: contextlib.nullcontext()),
    Config("G_compile",  "+ torch.compile(mode=default)",                    lambda: contextlib.nullcontext()),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="giant", choices=["base", "large", "huge", "giant"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-frames", type=int, default=48)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num-queries", type=int, default=2048)
    p.add_argument("--encoder-patch-size", type=int, default=14)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--only", default="", help="comma list of config names to run")
    p.add_argument("--out", default="bench_accel_gpu7.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=7 maps GPU7 -> cuda:0

    cap = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)} cap={cap}", flush=True)
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}", flush=True)
    print(f"variant={args.variant} batch={args.batch_size} T={args.num_frames} "
          f"res={args.resolution} queries={args.num_queries}", flush=True)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    results = []
    for cfg in CONFIGS:
        if only and cfg.name not in only:
            continue
        print(f"\n=== {cfg.name}  ({cfg.desc}) ===", flush=True)
        try:
            s = _run_one(args, cfg, device)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({"name": cfg.name, "error": f"{type(e).__name__}: {e}"})
            continue
        row = {
            "name": cfg.name,
            "desc": cfg.desc,
            "mean_total_ms": statistics.mean(s.total_ms),
            "p50_total_ms": _percentile(s.total_ms, 50),
            "p95_total_ms": _percentile(s.total_ms, 95),
            "mean_fwd_ms": statistics.mean(s.fwd_ms),
            "mean_bwd_ms": statistics.mean(s.bwd_ms),
            "mean_opt_ms": statistics.mean(s.opt_ms),
            "peak_mem_gb": s.peak_mem_gb,
        }
        results.append(row)
        print(f"  total mean={row['mean_total_ms']:.0f}ms  p50={row['p50_total_ms']:.0f}ms  "
              f"p95={row['p95_total_ms']:.0f}ms", flush=True)
        print(f"  fwd={row['mean_fwd_ms']:.0f}ms  bwd={row['mean_bwd_ms']:.0f}ms  "
              f"opt={row['mean_opt_ms']:.0f}ms  peak_mem={row['peak_mem_gb']:.1f}GB",
              flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'config':<14}{'total_ms':>10}{'fwd':>8}{'bwd':>8}{'opt':>8}{'mem_GB':>9}{'speedup':>10}",
          flush=True)
    base = next((r for r in results if r.get("name") == "A_baseline" and "mean_total_ms" in r), None)
    base_t = base["mean_total_ms"] if base else None
    for r in results:
        if "error" in r:
            print(f"{r['name']:<14}  ERROR: {r['error']}", flush=True)
            continue
        sp = (base_t / r["mean_total_ms"]) if base_t else float("nan")
        print(f"{r['name']:<14}{r['mean_total_ms']:>10.0f}{r['mean_fwd_ms']:>8.0f}"
              f"{r['mean_bwd_ms']:>8.0f}{r['mean_opt_ms']:>8.0f}"
              f"{r['peak_mem_gb']:>9.1f}{sp:>10.2f}x", flush=True)

    out_path = os.path.join(ROOT, args.out)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
