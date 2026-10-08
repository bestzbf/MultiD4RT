"""Measure ADJACENT-frame covisibility cov(i, i+stride) per stride, across many scenes.

The stride policy controls the gap between *consecutive sampled frames*, which is
exactly `stride` raw frames apart -- NOT the clip endpoints (which are 47*stride
apart and get blown up by clip_len). This probe measures what stride actually
controls: how much two frames `stride` apart still co-see.

For each scene we sample several random (i, i+stride) pairs and compute cov_geom.
We report the per-scene median, then percentiles ACROSS scenes (p10..p90) so the
scene-to-scene spread is visible -- different scenes move the camera at different
speeds, so a single number hides the variance.

Usage:
    python tools/probe_adjacent_stride.py \
        --config configs/mixture_5datasets_cos_planned.yaml \
        --dataset scannet --num-scenes 150 --pairs-per-scene 8 \
        --strides 1,2,3,4,6
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.registry import create_adapter  # noqa: E402
from tools.probe_clip_overlap import (  # noqa: E402
    _PROBE_KWARGS_OVERRIDE,
    reproject_cov,
    two_frame_clip,
)


def pct(xs, p):
    return float(np.percentile(np.asarray(xs), p)) if xs else float("nan")


def probe_one(name, ds_cfg, cache_dir, strides, num_scenes, pairs_per_scene, seed):
    """Probe a single dataset; print an ADJACENT-cov table. Each call flushes
    its own table so a timeout/kill keeps already-finished datasets' output."""
    kwargs = dict(ds_cfg.get("adapter_kwargs", {}))
    kwargs.update(_PROBE_KWARGS_OVERRIDE.get(name, {}))
    if cache_dir and "cache_dir" not in kwargs:
        kwargs["cache_dir"] = cache_dir

    print(f"\n########## {name} ##########", flush=True)
    try:
        adapter = create_adapter(
            name=name, root=ds_cfg["root"],
            split=ds_cfg.get("split", "train"), **kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  create_adapter_failed: {type(exc).__name__}: {exc}", flush=True)
        return

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    try:
        seqs = adapter.list_sequences()
    except Exception as exc:  # noqa: BLE001
        print(f"  list_sequences_failed: {type(exc).__name__}: {exc}", flush=True)
        return
    n_total = len(seqs)
    py_rng.shuffle(seqs)
    seqs = seqs[:num_scenes]

    scene_med: dict[int, list[float]] = {s: [] for s in strides}
    fails: dict[str, int] = {}
    n_done = 0
    for seq in seqs:
        try:
            nf = adapter.get_num_frames(seq)
        except Exception:
            fails["num_frames"] = fails.get("num_frames", 0) + 1
            continue
        for s in strides:
            if nf < s + 1:
                continue
            vals = []
            for _ in range(pairs_per_scene):
                start = py_rng.randint(0, nf - 1 - s)
                pair, err = two_frame_clip(adapter, seq, start, start + s)
                if pair is None:
                    fails[err] = fails.get(err, 0) + 1
                    continue
                d0, K0, E0, d1, K1, E1 = pair
                cg, _ = reproject_cov(d0, K0, E0, d1, K1, E1, rng=np_rng)
                if cg is not None:
                    vals.append(cg)
            if vals:
                scene_med[s].append(float(np.median(vals)))
        n_done += 1
        if n_done % 10 == 0:
            print(f"  ...{n_done}/{len(seqs)} scenes", flush=True)

    print(f"\n=== {name}: ADJACENT-frame cov(i, i+stride), "
          f"{n_done} scenes (of {n_total} total) ===", flush=True)
    print("(per-scene median cov_geom, then percentiles ACROSS scenes)\n", flush=True)
    print(f"{'stride':>6} {'nscenes':>7} {'p10':>6} {'p25':>6} {'p50':>6} "
          f"{'p75':>6} {'p90':>6}", flush=True)
    for s in strides:
        v = scene_med[s]
        print(f"{s:>6} {len(v):>7} "
              f"{pct(v,10):>6.3f} {pct(v,25):>6.3f} {pct(v,50):>6.3f} "
              f"{pct(v,75):>6.3f} {pct(v,90):>6.3f}", flush=True)

    print("\nfraction of scenes with median cov < 0.30 (weak-overlap):", flush=True)
    for s in strides:
        v = scene_med[s]
        if v:
            frac = float(np.mean(np.asarray(v) < 0.30))
            print(f"  stride={s}: {frac*100:5.1f}%  "
                  f"({sum(1 for x in v if x<0.30)}/{len(v)})", flush=True)
    if fails:
        print(f"  fail_counters={fails}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True,
                    help="single name, or comma-separated list, or 'all'")
    ap.add_argument("--num-scenes", type=int, default=150)
    ap.add_argument("--pairs-per-scene", type=int, default=8)
    ap.add_argument("--strides", type=str, default="1,2,3,4,6")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    strides = [int(x) for x in args.strides.split(",")]
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir = cfg.get("index_cache_dir")

    by_name = {d["name"]: d for d in cfg["datasets"]}
    if args.dataset == "all":
        names = [d["name"] for d in cfg["datasets"]]
    else:
        names = [n.strip() for n in args.dataset.split(",") if n.strip()]

    for name in names:
        if name not in by_name:
            print(f"\n########## {name} ##########\n  not in config, skip", flush=True)
            continue
        probe_one(
            name=name, ds_cfg=by_name[name], cache_dir=cache_dir,
            strides=strides, num_scenes=args.num_scenes,
            pairs_per_scene=args.pairs_per_scene, seed=args.seed,
        )


if __name__ == "__main__":
    main()
