#!/usr/bin/env python3
"""Benchmark PyTorch SDPA backends for D4RT attention shapes."""

from __future__ import annotations

import argparse
import gc
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class Case:
    q_shape: tuple[int, int, int, int]
    k_shape: tuple[int, int, int, int]
    v_shape: tuple[int, int, int, int]


CASES = {
    # large variant, batch_size=5, num_frames=48, patch=(2,16,16), resolution=256.
    # Local attention sees B * Tpatch = 5 * 24 rows and includes one local AR token.
    "encoder_local_large_b5": Case(
        q_shape=(120, 16, 257, 64),
        k_shape=(120, 16, 257, 64),
        v_shape=(120, 16, 257, 64),
    ),
    # B=1 keeps math-backend comparison reasonably fast while preserving sequence length.
    "encoder_global_large_b1": Case(
        q_shape=(1, 16, 6168, 64),
        k_shape=(1, 16, 6168, 64),
        v_shape=(1, 16, 6168, 64),
    ),
    "decoder_cross_large_b1": Case(
        q_shape=(1, 16, 2048, 64),
        k_shape=(1, 16, 6144, 64),
        v_shape=(1, 16, 6144, 64),
    ),
}


BACKENDS = {
    "default": None,
    "cudnn": [SDPBackend.CUDNN_ATTENTION],
    "flash": [SDPBackend.FLASH_ATTENTION],
    "efficient": [SDPBackend.EFFICIENT_ATTENTION],
    "no_cudnn": [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ],
    "math": [SDPBackend.MATH],
}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def _backend_context(name: str):
    backends = BACKENDS[name]
    if backends is None:
        return nullcontext()
    if len(backends) == 1:
        return sdpa_kernel(backends[0])
    return sdpa_kernel(backends)


def _clear_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def run_case(
    *,
    case: Case,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
    forward_only: bool,
) -> tuple[str, float | None, float | None]:
    q = torch.randn(case.q_shape, device=device, dtype=dtype, requires_grad=not forward_only)
    k = torch.randn(case.k_shape, device=device, dtype=dtype, requires_grad=not forward_only)
    v = torch.randn(case.v_shape, device=device, dtype=dtype, requires_grad=not forward_only)

    try:
        for _ in range(warmup):
            with _backend_context(backend):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                if not forward_only:
                    loss = out.float().square().mean()
            if not forward_only:
                loss.backward()
                _clear_grads(q, k, v)

        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            with _backend_context(backend):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                if not forward_only:
                    loss = out.float().square().mean()
            if not forward_only:
                loss.backward()
                _clear_grads(q, k, v)
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start.elapsed_time(end) / max(1, iters)
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        return "ok", elapsed_ms, peak_gib
    except Exception as exc:
        torch.cuda.synchronize(device)
        return f"{type(exc).__name__}: {str(exc).splitlines()[0]}", None, None
    finally:
        del q, k, v
        if "out" in locals():
            del out
        if "loss" in locals():
            del loss
        torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--cases",
        default="encoder_local_large_b5,encoder_global_large_b1,decoder_cross_large_b1",
        help=f"Comma-separated cases. Available: {','.join(CASES)}",
    )
    parser.add_argument(
        "--backends",
        default="flash,no_cudnn,math",
        help=(
            f"Comma-separated backends. Available: {','.join(BACKENDS)}. "
            "default/cudnn are useful diagnostics but may trigger cuDNN failures."
        ),
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    case_names = _parse_csv(args.cases)
    backend_names = _parse_csv(args.backends)

    unknown_cases = [name for name in case_names if name not in CASES]
    unknown_backends = [name for name in backend_names if name not in BACKENDS]
    if unknown_cases:
        raise ValueError(f"Unknown cases: {unknown_cases}. Available: {sorted(CASES)}")
    if unknown_backends:
        raise ValueError(f"Unknown backends: {unknown_backends}. Available: {sorted(BACKENDS)}")

    print(
        f"[sdpa-bench] torch={torch.__version__} cuda={torch.version.cuda} "
        f"cudnn={torch.backends.cudnn.version()}"
    )
    print(
        f"[sdpa-bench] device={device} name={torch.cuda.get_device_name(device)} "
        f"capability={torch.cuda.get_device_capability(device)} dtype={args.dtype} "
        f"mode={'forward' if args.forward_only else 'forward+backward'}"
    )
    print(f"{'case':28s} {'backend':10s} {'ms/iter':>10s} {'peak_gib':>10s} status")

    for case_name in case_names:
        case = CASES[case_name]
        for backend in backend_names:
            status, elapsed_ms, peak_gib = run_case(
                case=case,
                backend=backend,
                device=device,
                dtype=dtype,
                iters=args.iters,
                warmup=args.warmup,
                forward_only=args.forward_only,
            )
            ms_text = f"{elapsed_ms:10.2f}" if elapsed_ms is not None else f"{'-':>10s}"
            peak_text = f"{peak_gib:10.2f}" if peak_gib is not None else f"{'-':>10s}"
            print(f"{case_name:28s} {backend:10s} {ms_text} {peak_text} {status}", flush=True)


if __name__ == "__main__":
    main()
