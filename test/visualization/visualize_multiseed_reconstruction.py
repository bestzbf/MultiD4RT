#!/usr/bin/env python3
"""Multi-frame-seeded dense static-scene reconstruction for D4RT.

Problem with the original dense reconstruction (compute_dense_reference_sequence):
the query grid is seeded ONLY from the reference (middle) frame's pixels, so any
scene region not visible in that single frame is never reconstructed.

This script seeds the query grid from MULTIPLE keyframes (covering what each frame
uniquely sees), decodes every seed into a unified camera coordinate system, and
voxel-dedups overlapping points. Two modes:

  mode A (reproject):  build ONE canonical point set in the reference camera frame,
                       dedup once, then re-express that same point set in each of the
                       N camera frames (t_cam = 0..N-1). 48 frames share one point set
                       -> stable playback, fast. This is "reconstruct the full static
                       scene once, then fly the camera through it".

  mode B (per_frame):  for every camera frame k, seed from all keyframes, decode with
                       t_cam=k, dedup independently. Each frame uses its own visibility
                       -> more faithful per-frame coverage, but ~N x slower and frame
                       sets differ (can flicker).

Usage mirrors visualize_video_checkpoint_highconf.py; extra args:
  --seed-frames    comma list of clip frame indices to seed from (default: spread of 8)
  --voxel-size     dedup voxel edge in normalized 3D units (default 0.01)
  --recon-mode     A | B | both   (default both)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse all the heavy lifting from the existing highconf script.
VIS_DIR = Path(__file__).resolve().parent
if str(VIS_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_DIR))

import visualize_video_checkpoint_highconf as base  # noqa: E402
from utils.visualization import save_point_cloud_ply  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-frame-seeded dense static reconstruction.")
    p.add_argument("--input-path", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-variant", default="giant")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--patch-provider", default="sampled_resized")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num-frames", type=int, default=48)
    p.add_argument("--sampling-mode", default="contiguous", choices=["contiguous", "uniform"])
    p.add_argument("--start-index", type=int, default=-1)
    p.add_argument("--dense-point-cloud-stride", type=int, default=2)
    p.add_argument("--dense-vis-threshold", type=float, default=0.3)
    p.add_argument("--dense-confidence-percentile", type=float, default=10.0)
    p.add_argument("--dense-query-batch-size", type=int, default=16384)
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--flip-y-axis", action="store_true")
    p.add_argument("--normalize-3d", action="store_true")
    # Multi-seed specific
    p.add_argument(
        "--seed-frames",
        type=str,
        default="",
        help="逗号分隔的 clip 内帧索引作为播种帧。空表示自动在时序上均匀取 num-seed-frames 帧。",
    )
    p.add_argument("--num-seed-frames", type=int, default=8, help="自动播种帧数量（--seed-frames 为空时生效）。")
    p.add_argument("--voxel-size", type=float, default=0.01, help="体素去重边长（归一化 3D 单位）。")
    p.add_argument("--recon-mode", default="both", choices=["A", "B", "both"])
    return p.parse_args()


def build_seed_coords(stride: int, W: int, H: int, device: torch.device):
    """Return (coords_norm_torch [M,2], coords_px [M,2]) for a regular pixel grid."""
    x_pixels = np.arange(0, W, stride, dtype=np.int32)
    y_pixels = np.arange(0, H, stride, dtype=np.int32)
    gx, gy = np.meshgrid(x_pixels, y_pixels, indexing="xy")
    coords_px = np.stack([gx, gy], axis=-1).reshape(-1, 2)
    coords_norm = coords_px.astype(np.float32) / np.array(
        [max(W - 1, 1), max(H - 1, 1)], dtype=np.float32
    )[None]
    return torch.from_numpy(coords_norm).to(device=device, dtype=torch.float32), coords_px


def decode_seeds(
    model,
    encoder_features,
    frames_bcthw,
    coords_torch,
    t_src_val: int,
    t_tgt_val: int,
    t_cam_val: int,
    batch_size: int,
    device: torch.device,
):
    """Decode a grid of seeds with fixed (t_src,t_tgt,t_cam). Returns pos[M,3], vis[M], conf[M]."""
    total = int(coords_torch.shape[0])
    pos_list, vis_list, conf_list = [], [], []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = coords_torch[start:end].unsqueeze(0)
        n = end - start
        t_src = torch.full((1, n), t_src_val, device=device, dtype=torch.long)
        t_tgt = torch.full((1, n), t_tgt_val, device=device, dtype=torch.long)
        t_cam = torch.full((1, n), t_cam_val, device=device, dtype=torch.long)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model.decode(encoder_features, frames_bcthw, chunk, t_src, t_tgt, t_cam)
        pos_list.append(out["pos_3d"][0].detach().float().cpu().numpy())
        vis_list.append(torch.sigmoid(out["visibility"][0].squeeze(-1)).detach().float().cpu().numpy())
        ct = out.get("confidence_weight")
        if ct is None:
            conf_list.append(np.ones((n,), dtype=np.float32))
        else:
            conf_list.append(ct[0].squeeze(-1).detach().float().cpu().numpy())
    return (
        np.concatenate(pos_list, axis=0),
        np.concatenate(vis_list, axis=0),
        np.concatenate(conf_list, axis=0),
    )


def voxel_dedup(points: np.ndarray, colors: np.ndarray, conf: np.ndarray, voxel: float):
    """Keep one point per voxel: the highest-confidence one. Returns filtered arrays + keys."""
    if len(points) == 0:
        return points, colors, conf, np.empty((0, 3), dtype=np.int64)
    keys = np.floor(points / voxel).astype(np.int64)
    # order by confidence desc so the first occurrence per voxel is the best
    order = np.argsort(-conf, kind="stable")
    keys_o, pts_o, col_o, conf_o = keys[order], points[order], colors[order], conf[order]
    _, first_idx = np.unique(keys_o, axis=0, return_index=True)
    first_idx = np.sort(first_idx)
    return pts_o[first_idx], col_o[first_idx], conf_o[first_idx], keys_o[first_idx]


def collect_canonical_points(
    model, encoder_features, frames_bcthw, frames_resized, seed_frames,
    stride, vis_threshold, confidence_percentile, voxel_size, batch_size,
    ref_cam, device,
):
    """Mode A step 1: seed from many frames, decode all to t_cam=ref_cam, dedup -> canonical set.

    Returns dict with normalized seed coords + per-point (t_src) needed to re-decode later,
    plus the canonical points/colors in ref frame for PLY/preview.
    """
    H, W = frames_resized[0].shape[:2]
    coords_torch, coords_px = build_seed_coords(stride, W, H, device)

    all_coords = []      # normalized (u,v) of surviving seeds
    all_tsrc = []        # which frame each surviving seed came from
    all_pts = []         # 3D in ref cam frame
    all_col = []
    all_conf = []
    per_seed_kept = {}

    for sf in seed_frames:
        pos, vis, conf = decode_seeds(
            model, encoder_features, frames_bcthw, coords_torch,
            t_src_val=sf, t_tgt_val=sf, t_cam_val=ref_cam,
            batch_size=batch_size, device=device,
        )
        keep = (vis > vis_threshold) & np.isfinite(pos).all(axis=-1)
        if confidence_percentile > 0.0 and keep.sum() > 0:
            thr = float(np.percentile(conf[keep], confidence_percentile))
            keep &= conf >= thr
        per_seed_kept[sf] = int(keep.sum())
        if keep.sum() == 0:
            continue
        col = frames_resized[sf][coords_px[keep][:, 1], coords_px[keep][:, 0]].astype(np.float32)
        all_coords.append(coords_torch.cpu().numpy()[keep])
        all_tsrc.append(np.full((int(keep.sum()),), sf, dtype=np.int64))
        all_pts.append(pos[keep])
        all_col.append(col)
        all_conf.append(conf[keep])

    if not all_pts:
        raise RuntimeError("No surviving seed points across all seed frames.")

    coords_cat = np.concatenate(all_coords, axis=0)
    tsrc_cat = np.concatenate(all_tsrc, axis=0)
    pts_cat = np.concatenate(all_pts, axis=0)
    col_cat = np.concatenate(all_col, axis=0)
    conf_cat = np.concatenate(all_conf, axis=0)

    # voxel dedup using ref-frame positions; carry seed metadata so we can re-decode.
    if voxel_size > 0:
        keys = np.floor(pts_cat / voxel_size).astype(np.int64)
        order = np.argsort(-conf_cat, kind="stable")
        keys_o = keys[order]
        _, first_idx = np.unique(keys_o, axis=0, return_index=True)
        sel = order[np.sort(first_idx)]
    else:
        sel = np.arange(len(pts_cat))

    return {
        "coords_norm": coords_cat[sel],     # [K,2]
        "t_src": tsrc_cat[sel],             # [K]
        "points_ref": base.maybe_flip_y(pts_cat[sel], False),  # flip applied later if needed
        "colors": col_cat[sel],
        "conf": conf_cat[sel],
        "per_seed_kept": per_seed_kept,
        "before_dedup": int(len(pts_cat)),
        "after_dedup": int(len(sel)),
    }


def reproject_canonical_to_cams(
    model, encoder_features, frames_bcthw, canonical, num_cams, batch_size, device, flip_y,
):
    """Mode A step 2: re-express the canonical point set in each camera frame t_cam=k.

    For a static point, t_src/t_tgt are kept at the seed's own source frame and only
    t_cam varies; the model outputs the point's 3D coords in camera-k's frame.
    """
    coords = torch.from_numpy(canonical["coords_norm"]).to(device=device, dtype=torch.float32)
    tsrc = canonical["t_src"]
    colors = canonical["colors"]
    K = int(coords.shape[0])

    dense_frames = []
    for k in range(num_cams):
        pos_list = []
        for start in range(0, K, batch_size):
            end = min(start + batch_size, K)
            chunk = coords[start:end].unsqueeze(0)
            n = end - start
            t_src = torch.from_numpy(tsrc[start:end]).to(device).long().unsqueeze(0)
            t_tgt = t_src  # static: target = source
            t_cam = torch.full((1, n), k, device=device, dtype=torch.long)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model.decode(encoder_features, frames_bcthw, chunk, t_src, t_tgt, t_cam)
            pos_list.append(out["pos_3d"][0].detach().float().cpu().numpy())
        pts = np.concatenate(pos_list, axis=0)
        finite = np.isfinite(pts).all(axis=-1)
        dense_frames.append({
            "frame_id": k,
            "points_ref": base.maybe_flip_y(pts[finite], flip_y),
            "colors": colors[finite],
            "point_ids": np.flatnonzero(finite).astype(np.int32),
            "visible_points": int(finite.sum()),
            "visible_before_confidence": int(finite.sum()),
            "total_points": K,
            "confidence_threshold": math.nan,
            "mean_confidence": float(np.mean(canonical["conf"])),
        })
    return dense_frames


def per_frame_reconstruction(
    model, encoder_features, frames_bcthw, frames_resized, seed_frames,
    stride, vis_threshold, confidence_percentile, voxel_size, batch_size, num_cams, device, flip_y,
):
    """Mode B: for each camera k, seed from all keyframes, decode to t_cam=k, dedup per-frame."""
    H, W = frames_resized[0].shape[:2]
    coords_torch, coords_px = build_seed_coords(stride, W, H, device)
    coords_px_np = coords_px

    dense_frames = []
    for k in range(num_cams):
        pts_all, col_all, conf_all = [], [], []
        for sf in seed_frames:
            pos, vis, conf = decode_seeds(
                model, encoder_features, frames_bcthw, coords_torch,
                t_src_val=sf, t_tgt_val=sf, t_cam_val=k,
                batch_size=batch_size, device=device,
            )
            keep = (vis > vis_threshold) & np.isfinite(pos).all(axis=-1)
            if confidence_percentile > 0.0 and keep.sum() > 0:
                thr = float(np.percentile(conf[keep], confidence_percentile))
                keep &= conf >= thr
            if keep.sum() == 0:
                continue
            col = frames_resized[sf][coords_px_np[keep][:, 1], coords_px_np[keep][:, 0]].astype(np.float32)
            pts_all.append(pos[keep]); col_all.append(col); conf_all.append(conf[keep])
        if not pts_all:
            dense_frames.append({
                "frame_id": k, "points_ref": np.empty((0, 3), np.float32),
                "colors": np.empty((0, 3), np.float32), "point_ids": np.empty((0,), np.int32),
                "visible_points": 0, "visible_before_confidence": 0, "total_points": 0,
                "confidence_threshold": math.nan, "mean_confidence": math.nan,
            })
            continue
        P = np.concatenate(pts_all, 0); C = np.concatenate(col_all, 0); Cf = np.concatenate(conf_all, 0)
        before = len(P)
        if voxel_size > 0:
            P, C, Cf, _ = voxel_dedup(P, C, Cf, voxel_size)
        dense_frames.append({
            "frame_id": k,
            "points_ref": base.maybe_flip_y(P, flip_y),
            "colors": C,
            "point_ids": np.arange(len(P), dtype=np.int32),
            "visible_points": int(len(P)),
            "visible_before_confidence": int(before),
            "total_points": int(len(P)),
            "confidence_threshold": math.nan,
            "mean_confidence": float(np.mean(Cf)) if len(Cf) else math.nan,
        })
    return dense_frames


def main() -> None:
    args = parse_args()
    device = base.select_device(args.device)
    model = base.load_model(args, device)

    frames_all, fps, input_kind, source_names = base.read_input_frames(args.input_path)
    sampled, sampled_indices, padded = base.sample_clip_frames(
        frames_all, args.num_frames, args.sampling_mode, args.start_index
    )
    frames_resized = base.resize_frames_square(sampled, args.resolution)
    orig_h, orig_w = sampled[0].shape[:2]
    T = len(frames_resized)
    ref_cam = T // 2

    if args.seed_frames.strip():
        seed_frames = sorted({int(x) for x in args.seed_frames.split(",") if x.strip() != ""})
    else:
        seed_frames = sorted(set(np.linspace(0, T - 1, num=min(args.num_seed_frames, T), dtype=int).tolist()))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Encode once.
    video = base.to_video_tensor(frames_resized, device)
    aspect_ratio = base.make_aspect_ratio(orig_w, orig_h, device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoder_features = model.encode(video, aspect_ratio)
    frames_bcthw = video.permute(0, 1, 4, 2, 3)

    summary = {
        "input_path": args.input_path,
        "input_kind": input_kind,
        "num_frames": T,
        "sampling_mode": args.sampling_mode,
        "sampled_frame_indices": list(map(int, sampled_indices)),
        "seed_frames": seed_frames,
        "voxel_size": args.voxel_size,
        "stride": args.dense_point_cloud_stride,
        "ref_cam": ref_cam,
        "modes": {},
    }

    modes = ["A", "B"] if args.recon_mode == "both" else [args.recon_mode]

    if "A" in modes:
        print("[mode A] collecting canonical points ...", flush=True)
        canonical = collect_canonical_points(
            model, encoder_features, frames_bcthw, frames_resized, seed_frames,
            args.dense_point_cloud_stride, args.dense_vis_threshold,
            args.dense_confidence_percentile, args.voxel_size,
            args.dense_query_batch_size, ref_cam, device,
        )
        print(f"[mode A] before dedup={canonical['before_dedup']} after={canonical['after_dedup']}", flush=True)
        dense_A = reproject_canonical_to_cams(
            model, encoder_features, frames_bcthw, canonical,
            T, args.dense_query_batch_size, device, args.flip_y_axis,
        )
        base.plot_dense_reference_static(
            dense_frames=dense_A, output_path=out_dir / "modeA_static.png",
            title_prefix="Mode A: multi-seed canonical scene (camera fly-through)",
        )
        base.write_dense_reference_gif(
            dense_frames=dense_A, output_path=out_dir / "modeA_camera_flythrough.gif",
            fps=args.gif_fps, title_prefix="Mode A canonical scene",
        )
        save_point_cloud_ply(
            str(out_dir / "modeA_full_scene.ply"),
            base.maybe_flip_y(canonical["points_ref"], args.flip_y_axis),
            # clip to [0,1]: resize 出来的浮点偶尔 = 1.0000001，会让 save_point_cloud_ply
            # 的 `colors.max() <= 1.0` 判定失败 → 跳过 *255 → int() 全部截断为 0（黑）。
            colors=np.clip(canonical["colors"], 0.0, 1.0),
        )
        summary["modes"]["A"] = {
            "before_dedup": canonical["before_dedup"],
            "after_dedup": canonical["after_dedup"],
            "per_seed_kept": canonical["per_seed_kept"],
            "canonical_points": canonical["after_dedup"],
        }

    if "B" in modes:
        print("[mode B] per-frame reconstruction ...", flush=True)
        dense_B = per_frame_reconstruction(
            model, encoder_features, frames_bcthw, frames_resized, seed_frames,
            args.dense_point_cloud_stride, args.dense_vis_threshold,
            args.dense_confidence_percentile, args.voxel_size,
            args.dense_query_batch_size, T, device, args.flip_y_axis,
        )
        base.plot_dense_reference_static(
            dense_frames=dense_B, output_path=out_dir / "modeB_static.png",
            title_prefix="Mode B: per-frame multi-seed reconstruction",
        )
        base.write_dense_reference_gif(
            dense_frames=dense_B, output_path=out_dir / "modeB_per_frame.gif",
            fps=args.gif_fps, title_prefix="Mode B per-frame scene",
        )
        save_point_cloud_ply(
            str(out_dir / "modeB_refframe_scene.ply"),
            dense_B[ref_cam]["points_ref"],
            colors=np.clip(dense_B[ref_cam]["colors"], 0.0, 1.0),
        )
        counts = [f["visible_points"] for f in dense_B]
        summary["modes"]["B"] = {
            "mean_points_per_frame": float(np.mean(counts)),
            "max_points_per_frame": int(np.max(counts)),
            "ref_frame_points": int(dense_B[ref_cam]["visible_points"]),
        }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
