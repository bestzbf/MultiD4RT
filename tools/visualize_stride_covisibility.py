"""Visualize per-stride covisibility for forward-trajectory datasets (scannet/scannetpp).

For each scene and each stride s in {1,2,3,4}, sample a CLIP_LEN-frame clip
[start, start+s, ..., start+47*s] and render a row:

    [ first frame ] [ last frame ] [ covis heatmap ]

The covis heatmap reprojects frame-0 valid-depth pixels into the LAST frame:
green = lands in last-frame FOV with z>0 (co-visible), red = lost. The title of
each row reports cov_geom = green / total, i.e. how much of frame-0 the far end
of the clip still sees. This is the quantity the stride policy is tuned on.

Usage:
    python tools/visualize_stride_covisibility.py \
        --config configs/mixture_5datasets_cos_planned.yaml \
        --dataset scannet \
        --num-scenes 4 \
        --strides 1,2,3,4 \
        --output probe_results/scannet_stride_vis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.registry import create_adapter  # noqa: E402

CLIP_LEN = 48
N_PROBE_PIXELS = 4096


def reproject_mask(depth0, K0, E0, K1, E1, n_pixels=N_PROBE_PIXELS, rng=None):
    """Return (cov_geom, ys, xs, in_fov_bool) for frame-0 valid pixels.

    cov_geom = fraction of sampled frame-0 valid pixels that reproject into
    frame-1 image rect with z1 > 0. ys/xs are the sampled pixel coords in
    frame 0; in_fov_bool[i] says whether pixel i is co-visible in frame 1.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    H, W = depth0.shape[:2]
    valid = np.isfinite(depth0) & (depth0 > 0)
    if not valid.any():
        return None, None, None, None
    ys, xs = np.where(valid)
    if len(ys) > n_pixels:
        idx = rng.choice(len(ys), size=n_pixels, replace=False)
        ys, xs = ys[idx], xs[idx]
    z0 = depth0[ys, xs].astype(np.float64)

    K0_inv = np.linalg.inv(K0.astype(np.float64))
    pix0 = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones_like(z0)], axis=0)
    Xc0 = (K0_inv @ pix0) * z0[None, :]
    E0_inv = np.linalg.inv(E0.astype(np.float64))
    Xw = E0_inv[:3, :3] @ Xc0 + E0_inv[:3, 3:4]

    E1f = E1.astype(np.float64)
    Xc1 = E1f[:3, :3] @ Xw + E1f[:3, 3:4]
    z1 = Xc1[2]
    proj = K1.astype(np.float64) @ Xc1
    u1 = proj[0] / np.clip(proj[2], 1e-6, None)
    v1 = proj[1] / np.clip(proj[2], 1e-6, None)
    in_fov = (u1 >= 0) & (u1 < W) & (v1 >= 0) & (v1 < H) & (z1 > 1e-3)
    cov = float(in_fov.sum()) / max(1, len(z0))
    return cov, ys, xs, in_fov


def render_scene(adapter, seq, strides, np_rng, py_rng):
    num_frames = adapter.get_num_frames(seq)
    rows = []
    for s in strides:
        end_off = (CLIP_LEN - 1) * s
        if num_frames < 1 + end_off:
            rows.append((s, None, "too_short", num_frames))
            continue
        start = py_rng.randint(0, num_frames - 1 - end_off)
        idxs = [start + i * s for i in range(CLIP_LEN)]
        try:
            clip = adapter.load_clip(seq, [idxs[0], idxs[-1]])
        except Exception as exc:  # noqa: BLE001
            rows.append((s, None, f"load_fail:{type(exc).__name__}", num_frames))
            continue
        if clip.depths is None or clip.depths[0] is None or clip.depths[1] is None:
            rows.append((s, None, "no_depth", num_frames))
            continue
        cov, ys, xs, in_fov = reproject_mask(
            clip.depths[0], clip.intrinsics[0], clip.extrinsics[0],
            clip.intrinsics[1], clip.extrinsics[1], rng=np_rng,
        )
        rows.append((s, (clip, ys, xs, in_fov, idxs[0], idxs[-1]), cov, num_frames))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-scenes", type=int, default=4)
    ap.add_argument("--strides", type=str, default="1,2,3,4")
    ap.add_argument("--output", required=True, help="output dir for png files")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    strides = [int(x) for x in args.strides.split(",")]
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = next(d for d in cfg["datasets"] if d["name"] == args.dataset)
    kwargs = dict(ds_cfg.get("adapter_kwargs", {}))
    cache_dir = cfg.get("index_cache_dir")
    if cache_dir and "cache_dir" not in kwargs:
        kwargs["cache_dir"] = cache_dir

    adapter = create_adapter(
        name=args.dataset, root=ds_cfg["root"],
        split=ds_cfg.get("split", "train"), **kwargs,
    )

    import random
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    seqs = adapter.list_sequences()
    py_rng.shuffle(seqs)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    si = 0
    while rendered < args.num_scenes and si < len(seqs):
        seq = seqs[si]
        si += 1
        rows = render_scene(adapter, seq, strides, np_rng, py_rng)
        if not any(r[1] is not None for r in rows):
            continue

        ncol = 3
        nrow = len(rows)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
        if nrow == 1:
            axes = axes[None, :]
        for ri, (s, payload, cov, num_frames) in enumerate(rows):
            for c in range(ncol):
                axes[ri, c].axis("off")
            if payload is None:
                axes[ri, 0].set_title(f"stride={s}  {cov}  (frames={num_frames})", fontsize=9)
                continue
            clip, ys, xs, in_fov, f0, f1 = payload
            img0 = clip.images[0]
            img1 = clip.images[1]
            axes[ri, 0].imshow(img0)
            axes[ri, 0].set_title(f"stride={s}  first f#{f0}", fontsize=9)
            axes[ri, 1].imshow(img1)
            axes[ri, 1].set_title(f"last f#{f1} (span={(f1-f0)} frames)", fontsize=9)
            # covis heatmap over first frame
            axes[ri, 2].imshow(img0)
            covis_pct = 100.0 * cov if cov is not None else float("nan")
            if ys is not None:
                axes[ri, 2].scatter(xs[~in_fov], ys[~in_fov], s=1, c="red", alpha=0.35)
                axes[ri, 2].scatter(xs[in_fov], ys[in_fov], s=1, c="lime", alpha=0.35)
            axes[ri, 2].set_title(f"covis to last = {covis_pct:.0f}%  (green=seen)", fontsize=9)
        fig.suptitle(f"{args.dataset}  {seq}  ({num_frames} frames)", fontsize=11)
        fig.tight_layout()
        out_path = out_dir / f"{args.dataset}_{seq}_stride_covis.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}", flush=True)
        rendered += 1

    print(f"done: {rendered} scenes -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
