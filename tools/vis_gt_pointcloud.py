"""
Only visualize GT dense point cloud from a dataset config — no model required.

Usage:
    python tools/vis_gt_pointcloud.py --config configs/single_dynamic_replica.yaml --output-dir /tmp/gt_vis
    python tools/vis_gt_pointcloud.py --config configs/single_co3dv2.yaml --output-dir /tmp/gt_vis --split val
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.factory import create_training_dataset


def load_sample(dataset, idx: int, seed: int, allow_no_tracks: bool):
    sample_index = idx
    rng = random.Random(seed + sample_index)
    for attempt in range(20):
        dataset_idx, sequence_name, frame_indices = dataset.mixture_sampler.sample(rng)
        adapter = dataset.adapters[dataset_idx]
        try:
            clip = adapter.load_clip(sequence_name, frame_indices)
            result = dataset.transform(clip, rng=rng)
            if not allow_no_tracks and not bool(result.metadata.get("has_tracks", False)):
                raise RuntimeError("no tracks")
            return result, sequence_name
        except Exception as e:
            rng = random.Random(seed + idx + attempt + 1)
    raise RuntimeError(f"Could not load sample at idx={idx}")


def dense_axis_limits(pts_list, percentile=95.0):
    pts = np.concatenate([p for p in pts_list if len(p) > 0], axis=0)
    if len(pts) == 0:
        return np.zeros(3), 1.0
    lo = np.percentile(pts, 100.0 - percentile, axis=0)
    hi = np.percentile(pts, percentile, axis=0)
    center = (lo + hi) / 2.0
    radius = max(float((hi - lo).max()) / 2.0, 1e-3)
    return center.astype(np.float32), radius


def set_axes(ax, center, radius):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def point_colors_from_image(image, uv_norm, img_size):
    h, w = image.shape[:2]
    px = (uv_norm * np.array([max(w - 1, 1), max(h - 1, 1)])).astype(np.float32)
    xs = np.clip(np.round(px[:, 0]), 0, w - 1).astype(np.int32)
    ys = np.clip(np.round(px[:, 1]), 0, h - 1).astype(np.int32)
    return image[ys, xs].astype(np.float32)


def vis_gt_sample(result, out_dir: Path, tag: str, point_source: str, num_frames_plot: int = 6):
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.trajs_3d_world is None:
        print(f"  [{tag}] no trajs_3d_world, skip")
        return

    trajs_3d = np.asarray(result.trajs_3d_world, dtype=np.float32)  # [T, N, 3]
    trajs_2d = np.asarray(result.trajs_2d, dtype=np.float32)        # [T, N, 2]
    T = trajs_3d.shape[0]
    norm = np.array([max(result.crop.crop_w - 1, 1), max(result.crop.crop_h - 1, 1)], dtype=np.float32)

    # collect per-frame points
    frames_pts = []
    frames_colors = []
    for t in range(T):
        finite_3d = np.isfinite(trajs_3d[t]).all(axis=-1)
        if point_source == "visible":
            sel = result.visibs[t].astype(bool) & finite_3d
        else:
            sel = finite_3d
        pts = trajs_3d[t, sel]
        uv_norm = trajs_2d[t, sel] / norm[None]
        uv_norm = np.clip(uv_norm, 0.0, 1.0)
        colors = point_colors_from_image(result.images[t], uv_norm, result.img_size)
        frames_pts.append(pts)
        frames_colors.append(colors)

    # compute shared axis limits
    center, radius = dense_axis_limits(frames_pts)

    # static multi-frame plot
    frame_ids = np.linspace(0, T - 1, num=min(num_frames_plot, T), dtype=int).tolist()
    frame_ids = list(dict.fromkeys(frame_ids))
    ncols = len(frame_ids)
    nrows = 2  # front view + top view

    fig = plt.figure(figsize=(4.5 * ncols, 5.5 * nrows), constrained_layout=True)
    views = [(20, -62), (90, 0)]
    view_labels = ["front", "top"]

    for col, fid in enumerate(frame_ids):
        pts = frames_pts[fid]
        cols = frames_colors[fid]
        for row, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1, projection="3d")
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.8, alpha=0.9)
            ax.view_init(elev=elev, azim=azim)
            set_axes(ax, center, radius)
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            ax.set_zlabel("Z", fontsize=7)
            row_label = f"{view_labels[row]}\n" if col == 0 else ""
            ax.set_title(f"{row_label}frame {fid}  pts={len(pts)}", fontsize=9)

    fig.suptitle(f"{result.sequence_name}  GT dense point cloud ({point_source})", fontsize=13)
    out_path = out_dir / f"{tag}_gt_dense_static.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-search", type=int, default=100)
    parser.add_argument("--point-source", default="all_finite", choices=["visible", "all_finite"])
    parser.add_argument("--allow-no-tracks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = REPO_ROOT / config_path

    print(f"[gt_vis] Loading dataset: {config_path} split={args.split}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    dataset = create_training_dataset(config, split=args.split)
    print(f"[gt_vis] Dataset ready, searching for {args.num_samples} samples ...")

    found = 0
    idx = args.start_index
    checked = 0
    while found < args.num_samples and checked < args.max_search:
        try:
            result, seq_name = load_sample(dataset, idx, args.seed, args.allow_no_tracks)
            tag = f"s{found:02d}_idx{idx}_{seq_name}"
            sample_dir = out_dir / tag
            print(f"[gt_vis] sample {found}: {seq_name}")
            vis_gt_sample(result, sample_dir, tag, args.point_source)
            found += 1
        except Exception as e:
            print(f"[gt_vis]   skip idx={idx}: {e}")
        idx += 1
        checked += 1

    print(f"[gt_vis] Done. {found} samples saved to {out_dir}")


if __name__ == "__main__":
    main()
