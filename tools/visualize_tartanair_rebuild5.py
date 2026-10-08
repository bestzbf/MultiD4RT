#!/usr/bin/env python3
"""Rebuild and visualize a few TartanAir compact static-track clips.

This is a diagnostic script: it computes compact static points for selected
48-frame clips and writes visual checks under outputs/. It does not overwrite
the dataset-level precomputed caches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "datasets"))

from adapters.TartanAir import TartanAirAdapter, _project_static_points_to_clip
from computer.depth_to_tracks import compute_tracks


def _parse_clip_spec(raw: str) -> tuple[str, int]:
    seq, sep, start = raw.rpartition(":")
    if not sep or not seq:
        raise argparse.ArgumentTypeError(
            "clip must be formatted as sequence/name:PREFIX_START, "
            "for example abandonedfactory/Easy/P011:914"
        )
    return seq, int(start)


def _sanitize_name(sequence_name: str, start: int, clip_len: int) -> str:
    return f"{sequence_name.replace('/', '__')}__{start:06d}_{start + clip_len - 1:06d}"


def _select_display_frames(num_frames: int, count: int = 6) -> list[int]:
    count = min(count, num_frames)
    return sorted(set(np.linspace(0, num_frames - 1, count, dtype=np.int32).tolist()))


def _bounded_indices(mask: np.ndarray, limit: int, seed: int) -> np.ndarray:
    ids = np.flatnonzero(mask)
    if ids.size <= limit:
        return ids
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=limit, replace=False).astype(np.int64))


def _plot_image_grid(
    out_path: Path,
    images: list[np.ndarray],
    frame_indices: list[int],
    title: str,
    drawer,
) -> None:
    cols = min(3, len(frame_indices))
    rows = int(math.ceil(len(frame_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.8), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for panel_idx, frame_idx in enumerate(frame_indices):
        ax = axes.ravel()[panel_idx]
        ax.imshow(images[frame_idx])
        drawer(ax, frame_idx)
        ax.set_xlim(0, images[frame_idx].shape[1] - 1)
        ax.set_ylim(images[frame_idx].shape[0] - 1, 0)
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_reference_points(
    out_path: Path,
    images: list[np.ndarray],
    global_frame_indices: list[int],
    ref_frames: np.ndarray,
    ref_uv: np.ndarray,
) -> None:
    frames = _select_display_frames(len(images))

    def draw(ax, frame_idx: int) -> None:
        mask = ref_frames == frame_idx
        pts = ref_uv[mask]
        if pts.size:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=9,
                c="#00e5ff",
                alpha=0.8,
                edgecolors="black",
                linewidths=0.15,
            )
        ax.set_title(f"global {global_frame_indices[frame_idx]}  ref pts {pts.shape[0]}")

    _plot_image_grid(out_path, images, frames, "Reference points sampled per frame", draw)


def _plot_visible_points(
    out_path: Path,
    images: list[np.ndarray],
    global_frame_indices: list[int],
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: np.ndarray,
    ref_frames: np.ndarray,
) -> None:
    frames = _select_display_frames(len(images))
    cmap = cm.get_cmap("turbo")
    denom = max(1, len(images) - 1)

    def draw(ax, frame_idx: int) -> None:
        mask = valids[frame_idx] & visibs[frame_idx] & np.isfinite(trajs_2d[frame_idx]).all(axis=-1)
        ids = _bounded_indices(mask, limit=1200, seed=1009 + frame_idx)
        pts = trajs_2d[frame_idx, ids]
        if pts.size:
            colors = cmap(np.clip(ref_frames[ids] / denom, 0.0, 1.0))
            ax.scatter(pts[:, 0], pts[:, 1], s=4, c=colors, alpha=0.75, edgecolors="none")
        ax.set_title(f"global {global_frame_indices[frame_idx]}  visible {int(mask.sum())}")

    _plot_image_grid(out_path, images, frames, "Projected visible static points", draw)


def _plot_same_points(
    out_path: Path,
    images: list[np.ndarray],
    global_frame_indices: list[int],
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: np.ndarray,
) -> int:
    frames = _select_display_frames(len(images), count=5)
    common = np.ones((trajs_2d.shape[1],), dtype=bool)
    for frame_idx in frames:
        common &= valids[frame_idx] & visibs[frame_idx] & np.isfinite(trajs_2d[frame_idx]).all(axis=-1)
    ids = _bounded_indices(common, limit=140, seed=20260518)
    if ids.size == 0:
        any_visible = (valids & visibs).sum(axis=0) >= max(2, len(frames) // 2)
        ids = _bounded_indices(any_visible, limit=140, seed=20260519)

    colors = cm.get_cmap("hsv")(np.linspace(0, 1, max(1, ids.size), endpoint=False))

    def draw(ax, frame_idx: int) -> None:
        frame_visible = valids[frame_idx, ids] & visibs[frame_idx, ids]
        pts = trajs_2d[frame_idx, ids]
        if pts.size:
            ax.scatter(
                pts[frame_visible, 0],
                pts[frame_visible, 1],
                s=10,
                c=colors[frame_visible],
                alpha=0.9,
                edgecolors="black",
                linewidths=0.12,
            )
        ax.set_title(f"global {global_frame_indices[frame_idx]}  shared shown {int(frame_visible.sum())}")

    _plot_image_grid(out_path, images, frames, "Same static points across frames", draw)
    return int(ids.size)


def _plot_coverage(
    out_path: Path,
    global_frame_indices: list[int],
    ref_counts: np.ndarray,
    valids: np.ndarray,
    visibs: np.ndarray,
) -> None:
    visible_counts = (valids & visibs).sum(axis=1)
    valid_counts = valids.sum(axis=1)
    x = np.asarray(global_frame_indices, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(x, visible_counts, label="valid & visible", color="#0072b2", linewidth=2.0)
    ax.plot(x, valid_counts, label="valid projection", color="#d55e00", linewidth=1.5, alpha=0.85)
    ax.bar(x, ref_counts[: len(x)], label="new reference points", color="#009e73", alpha=0.28)
    ax.set_xlabel("global frame")
    ax.set_ylabel("points")
    ax.set_title(
        "Per-frame static point coverage "
        f"(visible min/mean/max = {visible_counts.min():.0f}/{visible_counts.mean():.1f}/{visible_counts.max():.0f})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_summary_montage(
    out_path: Path,
    images: list[np.ndarray],
    global_frame_indices: list[int],
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: np.ndarray,
    ref_frames: np.ndarray,
    summary: dict[str, Any],
) -> None:
    frames = _select_display_frames(len(images), count=4)
    cmap = cm.get_cmap("turbo")
    denom = max(1, len(images) - 1)
    fig, axes = plt.subplots(1, len(frames), figsize=(len(frames) * 4.2, 3.7), squeeze=False)
    for ax, frame_idx in zip(axes.ravel(), frames):
        ax.imshow(images[frame_idx])
        mask = valids[frame_idx] & visibs[frame_idx] & np.isfinite(trajs_2d[frame_idx]).all(axis=-1)
        ids = _bounded_indices(mask, limit=850, seed=3001 + frame_idx)
        pts = trajs_2d[frame_idx, ids]
        if pts.size:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=4,
                c=cmap(np.clip(ref_frames[ids] / denom, 0.0, 1.0)),
                alpha=0.75,
                edgecolors="none",
            )
        ax.set_title(f"f {global_frame_indices[frame_idx]}  vis {int(mask.sum())}")
        ax.axis("off")
    fig.suptitle(
        f"{summary['sequence']}  {summary['frames'][0]}..{summary['frames'][1]} | "
        f"pts {summary['dedup_points']} from {summary['pre_dedup_points']} | "
        f"visible min/mean/max {summary['valid_visible_min']}/"
        f"{summary['valid_visible_mean']:.1f}/{summary['valid_visible_max']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _select_clips(
    adapter: TartanAirAdapter,
    requested: list[tuple[str, int]],
    num_clips: int,
    clip_len: int,
) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    seen: set[str] = set()

    def add(seq: str, start: int) -> None:
        if len(selected) >= num_clips or seq in seen:
            return
        nf = adapter.get_num_frames(seq)
        if nf < clip_len:
            return
        start_clamped = max(0, min(int(start), nf - clip_len))
        selected.append((seq, start_clamped))
        seen.add(seq)

    for seq, start in requested:
        add(seq, start)

    if "abandonedfactory/Easy/P011" not in seen:
        add("abandonedfactory/Easy/P011", 914)

    eligible = [seq for seq in adapter.list_sequences() if adapter.get_num_frames(seq) >= clip_len and seq not in seen]
    need = num_clips - len(selected)
    if need <= 0:
        return selected[:num_clips]
    if not eligible:
        return selected

    picks = np.linspace(0, len(eligible) - 1, need, dtype=np.int32)
    for pick_idx, seq_idx in enumerate(picks.tolist()):
        seq = eligible[seq_idx]
        nf = adapter.get_num_frames(seq)
        span = max(0, nf - clip_len)
        start = int(round((pick_idx + 1) * span / (need + 1))) if span > 0 else 0
        add(seq, start)

    return selected[:num_clips]


def _write_index(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for item in summaries:
        sample_dir = Path(item["sample_dir"]).name
        rows.append(
            "<tr>"
            f"<td>{item['sample_id']}</td>"
            f"<td>{item['sequence']}</td>"
            f"<td>{item['frames'][0]}..{item['frames'][1]}</td>"
            f"<td>{item['dedup_points']}</td>"
            f"<td>{item['valid_visible_min']} / {item['valid_visible_mean']:.1f} / {item['valid_visible_max']}</td>"
            f"<td><a href='{sample_dir}/summary_montage.png'>montage</a></td>"
            "</tr>"
        )
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TartanAir static point rebuild diagnostics</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    table { border-collapse: collapse; }
    th, td { border: 1px solid #ccc; padding: 6px 9px; }
    th { background: #f3f3f3; }
    img { max-width: 100%; border: 1px solid #ddd; }
    section { margin-bottom: 28px; }
  </style>
</head>
<body>
<h1>TartanAir static point rebuild diagnostics</h1>
<table>
<thead><tr><th>id</th><th>sequence</th><th>frames</th><th>points</th><th>visible min/mean/max</th><th>link</th></tr></thead>
<tbody>
""" + "\n".join(rows) + """
</tbody>
</table>
"""
    for item in summaries:
        sample_dir = Path(item["sample_dir"]).name
        html += f"""
<section>
<h2>{item['sample_id']}: {item['sequence']} {item['frames'][0]}..{item['frames'][1]}</h2>
<img src="{sample_dir}/summary_montage.png">
<p>
<a href="{sample_dir}/reference_points_grid.png">reference points</a> |
<a href="{sample_dir}/visible_static_points_grid.png">visible points</a> |
<a href="{sample_dir}/same_static_points_across_frames.png">same points</a> |
<a href="{sample_dir}/coverage_counts.png">coverage</a> |
<a href="{sample_dir}/summary.json">summary.json</a>
</p>
</section>
"""
    html += "</body>\n</html>\n"
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data5/d4rt_dataset/tartanairv1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-clips", type=int, default=5)
    parser.add_argument("--clip-len", type=int, default=48)
    parser.add_argument("--camera", default="left", choices=["left", "right"])
    parser.add_argument("--clip", action="append", default=[], type=_parse_clip_spec)
    parser.add_argument("--points-per-frame", type=int, default=100)
    parser.add_argument("--ref-frame-stride", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--depth-max", type=float, default=200.0)
    parser.add_argument("--depth-bin-edges", default="0,5,10,20,40,80,200")
    parser.add_argument("--depth-bin-weights", default="6,6,4,2,1,1")
    parser.add_argument("--cache-dir", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / (
        "tartanair_static_rebuild5_" + time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    adapter = TartanAirAdapter(
        root=args.root,
        camera=args.camera,
        load_precomputed=False,
        verbose=True,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        max_static_track_points=None,
    )
    clips = _select_clips(adapter, args.clip, args.num_clips, args.clip_len)
    if not clips:
        raise RuntimeError("No clips selected. Check --root and --clip-len.")

    depth_bin_edges = [float(v) for v in args.depth_bin_edges.split(",") if v.strip()]
    depth_bin_weights = [float(v) for v in args.depth_bin_weights.split(",") if v.strip()]

    summaries: list[dict[str, Any]] = []
    for sample_idx, (seq_name, start) in enumerate(clips, start=1):
        sample_id = f"sample_{sample_idx:02d}"
        frame_indices = list(range(start, start + args.clip_len))
        sample_dir = out_dir / f"{sample_id}_{_sanitize_name(seq_name, start, args.clip_len)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{sample_id}] rebuild {seq_name} frames {frame_indices[0]}..{frame_indices[-1]}", flush=True)

        clip = adapter.load_clip(seq_name, frame_indices)
        tracks = compute_tracks(
            depths=clip.depths,
            intrinsics=clip.intrinsics,
            extrinsics=clip.extrinsics,
            num_points=8000,
            depth_max=args.depth_max,
            source_depth_sampling="log_balanced",
            source_depth_bin_edges=depth_bin_edges,
            source_depth_bin_weights=depth_bin_weights,
            points_per_ref_frame=args.points_per_frame,
            ref_frame_stride=args.ref_frame_stride,
            world_dedup_voxel_size=args.voxel_size,
            compact_static_tracks=True,
        )
        trajs_2d, valids, visibs = _project_static_points_to_clip(
            points_world=tracks["static_points_world"],
            depths=clip.depths,
            intrinsics=clip.intrinsics,
            extrinsics=clip.extrinsics,
            depth_consistency_thresh=float(tracks["track_depth_consistency_thresh"]),
            depth_max=args.depth_max,
        )

        ref_frames = np.asarray(tracks["static_point_ref_frames"], dtype=np.int32)
        ref_uv = np.asarray(tracks["static_point_ref_uv"], dtype=np.float32)
        ref_counts = np.asarray(tracks["static_point_ref_counts"], dtype=np.int32)
        visible_counts = (valids & visibs).sum(axis=1)
        shared_candidates = int(((valids & visibs).sum(axis=0) >= max(2, args.clip_len // 4)).sum())

        summary = {
            "sample_id": sample_id,
            "sequence": seq_name,
            "frames": [frame_indices[0], frame_indices[-1]],
            "clip_len": args.clip_len,
            "points_per_frame_requested": args.points_per_frame,
            "ref_frame_stride": args.ref_frame_stride,
            "voxel_size_m": args.voxel_size,
            "pre_dedup_points": int(np.asarray(tracks["pre_dedup_num_points"]).item()),
            "dedup_points": int(np.asarray(tracks["num_points"]).item()),
            "valid_visible_min": int(visible_counts.min()),
            "valid_visible_mean": float(visible_counts.mean()),
            "valid_visible_max": int(visible_counts.max()),
            "valid_visible_first_last": [int(visible_counts[0]), int(visible_counts[-1])],
            "shared_candidates_visible_in_quarter_clip": shared_candidates,
            "sample_dir": str(sample_dir),
        }

        np.savez_compressed(
            sample_dir / "clip_static_cache.npz",
            **tracks,
            trajs_2d=trajs_2d.astype(np.float32),
            valids=valids.astype(np.bool_),
            visibs=visibs.astype(np.bool_),
            global_frame_indices=np.asarray(frame_indices, dtype=np.int32),
        )

        _plot_reference_points(
            sample_dir / "reference_points_grid.png",
            clip.images,
            frame_indices,
            ref_frames,
            ref_uv,
        )
        _plot_visible_points(
            sample_dir / "visible_static_points_grid.png",
            clip.images,
            frame_indices,
            trajs_2d,
            valids,
            visibs,
            ref_frames,
        )
        summary["same_points_visualized"] = _plot_same_points(
            sample_dir / "same_static_points_across_frames.png",
            clip.images,
            frame_indices,
            trajs_2d,
            valids,
            visibs,
        )
        _plot_coverage(
            sample_dir / "coverage_counts.png",
            frame_indices,
            ref_counts,
            valids,
            visibs,
        )
        _plot_summary_montage(
            sample_dir / "summary_montage.png",
            clip.images,
            frame_indices,
            trajs_2d,
            valids,
            visibs,
            ref_frames,
            summary,
        )

        with open(sample_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        summaries.append(summary)
        print(
            f"[{sample_id}] points {summary['dedup_points']} "
            f"visible min/mean/max {summary['valid_visible_min']}/"
            f"{summary['valid_visible_mean']:.1f}/{summary['valid_visible_max']}",
            flush=True,
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    _write_index(out_dir, summaries)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
