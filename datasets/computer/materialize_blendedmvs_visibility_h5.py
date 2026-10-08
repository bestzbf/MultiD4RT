#!/usr/bin/env python3
"""Materialize BlendedMVS compact static tracks into dense visibility H5 caches.

This mirrors the MVS-Synth workflow:

1. ``run_blendedmvs.py`` writes compact static 3D points into ``precomputed.npz``.
2. This script projects those points through every BlendedMVS frame with the
   selected visibility policy.
3. It writes dense ``precomputed.h5`` files containing ``trajs_2d``,
   ``trajs_3d_world``, ``valids``, and ``visibs`` for direct training reads.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.blendedmvs import BlendedMVSAdapter, _parse_cam_file
from computer.depth_to_tracks import TRACK_SEMANTICS_VERSION


PER_FRAME_KEYS = {"trajs_2d", "trajs_3d_world", "valids", "visibs", "normals"}
PER_POINT_KEYS = {
    "static_points_world",
    "static_point_ref_frames",
    "static_point_ref_uv",
    "static_point_ref_depth",
}


def _valid_mask_from_masked_rgb(
    image: np.ndarray,
    *,
    threshold: int,
    erosion_px: int,
) -> np.ndarray:
    """Return an eroded valid-pixel mask from a BlendedMVS *_masked.jpg."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected RGB masked image, got shape {arr.shape}")
    valid = arr.max(axis=-1) > int(threshold)
    erosion_px = max(int(erosion_px), 0)
    if erosion_px > 0:
        import cv2

        kernel = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), dtype=np.uint8)
        valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    return valid


def _read_masked_valid_mask(
    adapter: BlendedMVSAdapter,
    record: Any,
    frame_id: str,
    *,
    threshold: int,
    erosion_px: int,
) -> np.ndarray:
    image = adapter._read_image(record.rgb_path(frame_id, masked=True))
    return _valid_mask_from_masked_rgb(
        image,
        threshold=threshold,
        erosion_px=erosion_px,
    )


def _optional_positive_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and value > 0.0 else None


def _parse_sequences(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _write_dataset(
    handle: h5py.File,
    key: str,
    value: Any,
    compression: str | None,
) -> None:
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O"}:
        return
    if arr.ndim >= 2 and arr.shape[0] > 1 and key not in PER_POINT_KEYS:
        handle.create_dataset(
            key,
            data=arr,
            chunks=(1,) + arr.shape[1:],
            compression=compression,
        )
    else:
        handle.create_dataset(key, data=arr)


def _project_one_frame(
    points_world: np.ndarray,
    points_world_h: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    depth_max: float,
    depth_support_radius: int,
    depth_consistency_thresh: float,
    visible_abs_thresh: float,
    occlusion_abs_thresh: float,
    occlusion_rel_thresh: float,
    occlusion_min_support: int,
    visible_min_support: int,
    occlusion_far_depth_start: float | None,
    occlusion_far_depth_end: float | None,
    occlusion_far_abs_thresh: float | None,
    occlusion_far_rel_thresh: float | None,
    occlusion_far_min_support: int | None,
    far_uncertain_as_visible_depth: float | None,
    target_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project static world points into one frame."""
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(intrinsic, dtype=np.float32)
    E = np.asarray(extrinsic, dtype=np.float32)
    H, W = depth.shape[:2]
    N = points_world.shape[0]

    points_cam = (E @ points_world_h.T).T[:, :3]
    z = points_cam[:, 2].astype(np.float32)
    safe_z = np.where(z > 0, z, 1.0).astype(np.float32)
    uv = np.empty((N, 2), dtype=np.float32)
    uv[:, 0] = points_cam[:, 0] / safe_z * float(K[0, 0]) + float(K[0, 2])
    uv[:, 1] = points_cam[:, 1] / safe_z * float(K[1, 1]) + float(K[1, 2])

    uv_finite = np.isfinite(uv).all(axis=-1)
    world_finite = np.isfinite(points_world).all(axis=-1)
    in_front = (
        (z > 0)
        & np.isfinite(z)
        & (z < float(depth_max))
        & world_finite
    )
    in_bounds = (
        uv_finite
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < W)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < H)
    )

    sample_mask = in_bounds & in_front
    visible_t = np.zeros((N,), dtype=bool)
    occluded_t = np.zeros((N,), dtype=bool)
    if sample_mask.any():
        ids = np.flatnonzero(sample_mask)
        px = np.clip(np.round(uv[ids, 0]).astype(np.int32), 0, W - 1)
        py = np.clip(np.round(uv[ids, 1]).astype(np.int32), 0, H - 1)
        if target_valid_mask is not None:
            mask = np.asarray(target_valid_mask, dtype=bool)
            if mask.shape != depth.shape[:2]:
                raise ValueError(
                    f"target_valid_mask shape {mask.shape} does not match depth shape {depth.shape[:2]}"
                )
            mask_ok = mask[py, px]
            if not mask_ok.any():
                return uv, visible_t, visible_t
            ids = ids[mask_ok]
            px = px[mask_ok]
            py = py[mask_ok]
        z_sel = z[ids].astype(np.float32)

        radius = max(int(depth_support_radius), 0)
        min_support = max(int(occlusion_min_support), 1)
        vis_min_support = max(int(visible_min_support), 1)
        visible_margin = np.maximum(
            float(visible_abs_thresh),
            float(depth_consistency_thresh) * np.maximum(z_sel, 0.0),
        )
        occlusion_abs = np.full(
            z_sel.shape, float(occlusion_abs_thresh), dtype=np.float32
        )
        occlusion_rel = np.full(
            z_sel.shape, float(occlusion_rel_thresh), dtype=np.float32
        )
        support_required = np.full(z_sel.shape, min_support, dtype=np.int16)
        if occlusion_far_depth_start is not None:
            start = float(occlusion_far_depth_start)
            end = (
                float(occlusion_far_depth_end)
                if occlusion_far_depth_end is not None
                else start
            )
            denom = max(end - start, 1e-6)
            alpha = np.clip((z_sel - start) / denom, 0.0, 1.0).astype(np.float32)
            if occlusion_far_abs_thresh is not None:
                occlusion_abs = (
                    occlusion_abs
                    + alpha * (float(occlusion_far_abs_thresh) - occlusion_abs)
                )
            if occlusion_far_rel_thresh is not None:
                occlusion_rel = (
                    occlusion_rel
                    + alpha * (float(occlusion_far_rel_thresh) - occlusion_rel)
                )
            if occlusion_far_min_support is not None:
                support_required = np.ceil(
                    min_support
                    + alpha
                    * (max(int(occlusion_far_min_support), 1) - min_support)
                ).astype(np.int16)

        occlusion_margin = np.maximum(
            occlusion_abs,
            occlusion_rel * np.maximum(z_sel, 0.0),
        )
        visible_support = np.zeros((len(ids),), dtype=np.int16)
        nearer_support = np.zeros((len(ids),), dtype=np.int16)
        valid_depth_support = np.zeros((len(ids),), dtype=np.int16)
        for dy in range(-radius, radius + 1):
            yy = py + dy
            y_ok = (yy >= 0) & (yy < H)
            if not y_ok.any():
                continue
            yy = np.clip(yy, 0, H - 1)
            for dx in range(-radius, radius + 1):
                xx = px + dx
                ok = y_ok & (xx >= 0) & (xx < W)
                if not ok.any():
                    continue
                xx = np.clip(xx, 0, W - 1)
                sampled = depth[yy, xx].astype(np.float32)
                depth_ok = (
                    ok
                    & (sampled > 0)
                    & np.isfinite(sampled)
                    & (sampled < float(depth_max))
                )
                valid_depth_support += depth_ok.astype(np.int16)
                visible_support += (
                    depth_ok & (np.abs(sampled - z_sel) <= visible_margin)
                ).astype(np.int16)
                nearer_support += (
                    depth_ok & (sampled + occlusion_margin < z_sel)
                ).astype(np.int16)

        depth_occluded = nearer_support >= support_required
        far_uncertain_visible = np.zeros((len(ids),), dtype=bool)
        if far_uncertain_as_visible_depth is not None:
            far_uncertain_visible = (
                (z_sel >= float(far_uncertain_as_visible_depth))
                & (valid_depth_support > 0)
                & (~depth_occluded)
            )
        visible_t[ids] = (visible_support >= vis_min_support) | far_uncertain_visible
        occluded_t[ids] = (~visible_t[ids]) & depth_occluded

    valids_t = visible_t | occluded_t
    return uv, valids_t, visible_t


def _filter_source_points_by_mask(
    *,
    base: dict[str, Any],
    points_world: np.ndarray,
    adapter: BlendedMVSAdapter,
    record: Any,
    threshold: int,
    erosion_px: int,
) -> tuple[dict[str, Any], np.ndarray, int]:
    if "static_point_ref_frames" not in base or "static_point_ref_uv" not in base:
        return base, points_world, 0

    ref_frames = np.asarray(base["static_point_ref_frames"], dtype=np.int32)
    ref_uv = np.asarray(base["static_point_ref_uv"], dtype=np.float32)
    if ref_frames.shape[0] != points_world.shape[0] or ref_uv.shape[0] != points_world.shape[0]:
        return base, points_world, 0

    keep = np.zeros((points_world.shape[0],), dtype=bool)
    mask_cache: dict[int, np.ndarray] = {}
    for ref in np.unique(ref_frames):
        ref_i = int(ref)
        if ref_i < 0 or ref_i >= record.num_frames:
            continue
        if ref_i not in mask_cache:
            mask_cache[ref_i] = _read_masked_valid_mask(
                adapter,
                record,
                record.frame_ids[ref_i],
                threshold=threshold,
                erosion_px=erosion_px,
            )
        mask = mask_cache[ref_i]
        ids = np.flatnonzero(ref_frames == ref_i)
        xy = ref_uv[ids]
        finite = np.isfinite(xy).all(axis=1)
        px = np.clip(np.round(xy[:, 0]).astype(np.int32), 0, mask.shape[1] - 1)
        py = np.clip(np.round(xy[:, 1]).astype(np.int32), 0, mask.shape[0] - 1)
        in_bounds = (
            finite
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < mask.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < mask.shape[0])
        )
        keep[ids] = in_bounds & mask[py, px]

    dropped = int((~keep).sum())
    if dropped == 0:
        return base, points_world, 0

    filtered_base: dict[str, Any] = {}
    for key, value in base.items():
        arr = np.asarray(value)
        if key in PER_POINT_KEYS and arr.shape[:1] == keep.shape:
            filtered_base[key] = arr[keep]
        else:
            filtered_base[key] = value
    if "static_point_ref_frames" in filtered_base:
        filtered_base["static_point_ref_counts"] = np.bincount(
            np.asarray(filtered_base["static_point_ref_frames"], dtype=np.int32),
            minlength=record.num_frames,
        ).astype(np.int32)
    return filtered_base, points_world[keep], dropped


def _materialize_one(args: tuple) -> tuple[str, str | None, dict[str, Any] | None]:
    (
        split,
        seq_name,
        root,
        precompute_root,
        output_root,
        overwrite,
        compression,
        policy,
        mask_cfg,
    ) = args
    out_dir = Path(output_root) / seq_name
    out_path = out_dir / "precomputed.h5"
    tmp_path = out_dir / "precomputed.h5.tmp"
    if out_path.exists() and not overwrite:
        return seq_name, None, None
    if overwrite:
        out_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)

    try:
        adapter = BlendedMVSAdapter(
            root=root,
            split=split,
            load_precomputed=False,
            load_normals=False,
            verbose=False,
        )
        record = adapter._get_record(seq_name)
        npz_path = Path(precompute_root) / seq_name / "precomputed.npz"
        if not npz_path.exists():
            return seq_name, f"missing compact npz: {npz_path}", None

        base: dict[str, Any] = {}
        with np.load(npz_path, allow_pickle=False) as raw:
            for key in raw.files:
                if key in PER_FRAME_KEYS:
                    continue
                base[key] = raw[key]
            if "static_points_world" in raw:
                points_world = np.asarray(raw["static_points_world"], dtype=np.float32)
            elif "trajs_3d_world" in raw:
                points_world = np.asarray(raw["trajs_3d_world"][0], dtype=np.float32)
            else:
                return seq_name, "compact npz has no static_points_world/trajs_3d_world", None

        if points_world.ndim != 2 or points_world.shape[1] != 3:
            return seq_name, f"bad points_world shape: {points_world.shape}", None

        source_mask_dropped = 0
        if mask_cfg["mask_from_masked_images"] and mask_cfg["filter_source_points"]:
            base, points_world, source_mask_dropped = _filter_source_points_by_mask(
                base=base,
                points_world=points_world,
                adapter=adapter,
                record=record,
                threshold=mask_cfg["mask_threshold"],
                erosion_px=mask_cfg["mask_erosion_px"],
            )
            if points_world.shape[0] == 0:
                return seq_name, "all compact points were rejected by masked-image source filter", None

        T = int(record.num_frames)
        N = int(points_world.shape[0])
        points_world_h = np.concatenate(
            [points_world, np.ones((N, 1), dtype=np.float32)],
            axis=1,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        valid_visible_counts = np.zeros((T,), dtype=np.int32)
        valid_counts = np.zeros((T,), dtype=np.int32)
        with h5py.File(tmp_path, "w") as handle:
            for key in sorted(base.keys()):
                _write_dataset(handle, key, base[key], compression=None)

            trajs_2d_ds = handle.create_dataset(
                "trajs_2d",
                shape=(T, N, 2),
                dtype=np.float32,
                chunks=(1, N, 2),
                compression=compression,
            )
            trajs_3d_ds = handle.create_dataset(
                "trajs_3d_world",
                shape=(T, N, 3),
                dtype=np.float32,
                chunks=(1, N, 3),
                compression=compression,
            )
            valids_ds = handle.create_dataset(
                "valids",
                shape=(T, N),
                dtype=np.bool_,
                chunks=(1, N),
                compression=compression,
            )
            visibs_ds = handle.create_dataset(
                "visibs",
                shape=(T, N),
                dtype=np.bool_,
                chunks=(1, N),
                compression=compression,
            )

            for frame_idx, frame_id in enumerate(record.frame_ids):
                depth = adapter._read_depth(
                    record.depth_path(frame_id),
                    record.scene_id,
                    frame_id,
                )
                cam = _parse_cam_file(record.cam_path(frame_id))
                target_mask = None
                if mask_cfg["mask_from_masked_images"]:
                    target_mask = _read_masked_valid_mask(
                        adapter,
                        record,
                        frame_id,
                        threshold=mask_cfg["mask_threshold"],
                        erosion_px=mask_cfg["mask_erosion_px"],
                    )
                uv, valids, visibs = _project_one_frame(
                    points_world,
                    points_world_h,
                    depth,
                    cam["intrinsic"],
                    cam["extrinsic"],
                    target_valid_mask=target_mask,
                    **policy,
                )
                trajs_2d_ds[frame_idx] = uv.astype(np.float32, copy=False)
                trajs_3d_ds[frame_idx] = points_world
                valids_ds[frame_idx] = valids
                visibs_ds[frame_idx] = visibs
                valid_counts[frame_idx] = int(valids.sum())
                valid_visible_counts[frame_idx] = int((valids & visibs).sum())

            scalars = {
                "num_frames": np.array(T, dtype=np.int32),
                "num_points": np.array(N, dtype=np.int32),
                "track_semantics_version": np.array(TRACK_SEMANTICS_VERSION, dtype=np.int32),
                "compact_static_tracks": np.array(False, dtype=np.bool_),
                "materialized_dense_visibility": np.array(True, dtype=np.bool_),
                "valid_visible_counts": valid_visible_counts,
                "valid_counts": valid_counts,
                "visibility_depth_support_radius": np.array(policy["depth_support_radius"], dtype=np.int32),
                "visibility_visible_abs_thresh": np.array(policy["visible_abs_thresh"], dtype=np.float32),
                "visibility_occlusion_abs_thresh": np.array(policy["occlusion_abs_thresh"], dtype=np.float32),
                "visibility_occlusion_rel_thresh": np.array(policy["occlusion_rel_thresh"], dtype=np.float32),
                "visibility_occlusion_min_support": np.array(policy["occlusion_min_support"], dtype=np.int32),
                "visibility_occlusion_far_depth_start": np.array(policy["occlusion_far_depth_start"] or 0.0, dtype=np.float32),
                "visibility_occlusion_far_depth_end": np.array(policy["occlusion_far_depth_end"] or 0.0, dtype=np.float32),
                "visibility_occlusion_far_abs_thresh": np.array(policy["occlusion_far_abs_thresh"] or 0.0, dtype=np.float32),
                "visibility_occlusion_far_rel_thresh": np.array(policy["occlusion_far_rel_thresh"] or 0.0, dtype=np.float32),
                "visibility_occlusion_far_min_support": np.array(policy["occlusion_far_min_support"] or 0, dtype=np.int32),
                "visibility_far_uncertain_as_visible_depth": np.array(policy["far_uncertain_as_visible_depth"] or 0.0, dtype=np.float32),
                "visibility_depth_max": np.array(policy["depth_max"], dtype=np.float32),
                "visibility_mask_from_masked_images": np.array(mask_cfg["mask_from_masked_images"], dtype=np.bool_),
                "visibility_mask_threshold": np.array(mask_cfg["mask_threshold"], dtype=np.int32),
                "visibility_mask_erosion_px": np.array(mask_cfg["mask_erosion_px"], dtype=np.int32),
                "source_mask_filter_enabled": np.array(
                    mask_cfg["mask_from_masked_images"] and mask_cfg["filter_source_points"],
                    dtype=np.bool_,
                ),
                "source_mask_dropped_points": np.array(source_mask_dropped, dtype=np.int32),
            }
            for key, value in scalars.items():
                if key in handle:
                    del handle[key]
                _write_dataset(handle, key, value, compression=None)

        os.replace(tmp_path, out_path)
        summary = {
            "num_frames": T,
            "num_points": N,
            "visible_min": int(valid_visible_counts.min()) if T else 0,
            "visible_median": float(np.median(valid_visible_counts)) if T else 0.0,
            "visible_max": int(valid_visible_counts.max()) if T else 0,
            "frames_lt_512": int((valid_visible_counts < 512).sum()),
            "frames_zero": int((valid_visible_counts == 0).sum()),
            "source_mask_dropped_points": source_mask_dropped,
        }
        return seq_name, None, summary
    except Exception:
        tmp_path.unlink(missing_ok=True)
        return seq_name, traceback.format_exc(), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--precompute-root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--split", default="all", choices=["train", "val", "all"])
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression", default="lzf", choices=["none", "lzf", "gzip"])
    parser.add_argument("--visibility-depth-support-radius", type=int, default=1)
    parser.add_argument("--visibility-visible-abs-thresh", type=float, default=0.25)
    parser.add_argument("--visibility-occlusion-abs-thresh", type=float, default=0.25)
    parser.add_argument("--visibility-occlusion-rel-thresh", type=float, default=0.03)
    parser.add_argument("--visibility-occlusion-min-support", type=int, default=2)
    parser.add_argument("--visibility-occlusion-far-depth-start", type=float, default=25.0)
    parser.add_argument("--visibility-occlusion-far-depth-end", type=float, default=40.0)
    parser.add_argument("--visibility-occlusion-far-abs-thresh", type=float, default=8.0)
    parser.add_argument("--visibility-occlusion-far-rel-thresh", type=float, default=0.30)
    parser.add_argument("--visibility-occlusion-far-min-support", type=int, default=10)
    parser.add_argument("--visibility-far-uncertain-as-visible-depth", type=float, default=40.0)
    parser.add_argument("--visibility-depth-max", type=float, default=500.0)
    parser.add_argument("--visibility-mask-from-masked-images", action=argparse.BooleanOptionalAction, default=True,
                        help="Use BlendedMVS *_masked.jpg files to invalidate target projections outside the eroded mask")
    parser.add_argument("--visibility-mask-threshold", type=int, default=30,
                        help="A masked RGB pixel is valid when max(channel) is above this value")
    parser.add_argument("--visibility-mask-erosion-px", type=int, default=3,
                        help="Erode the valid masked-image region before visibility projection")
    parser.add_argument("--visibility-filter-source-mask", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop compact points whose source reference pixel is outside the eroded masked-image region")
    args = parser.parse_args()

    root = Path(args.root)
    precompute_root = Path(args.precompute_root)
    output_root = Path(args.output_root) if args.output_root else precompute_root
    splits = ["train", "val"] if args.split == "all" else [args.split]
    requested = _parse_sequences(args.sequences)
    compression = None if args.compression == "none" else args.compression
    policy = {
        "depth_max": float(args.visibility_depth_max),
        "depth_support_radius": int(args.visibility_depth_support_radius),
        "depth_consistency_thresh": 0.05,
        "visible_abs_thresh": float(args.visibility_visible_abs_thresh),
        "occlusion_abs_thresh": float(args.visibility_occlusion_abs_thresh),
        "occlusion_rel_thresh": float(args.visibility_occlusion_rel_thresh),
        "occlusion_min_support": int(args.visibility_occlusion_min_support),
        "visible_min_support": 1,
        "occlusion_far_depth_start": _optional_positive_float(args.visibility_occlusion_far_depth_start),
        "occlusion_far_depth_end": _optional_positive_float(args.visibility_occlusion_far_depth_end),
        "occlusion_far_abs_thresh": _optional_positive_float(args.visibility_occlusion_far_abs_thresh),
        "occlusion_far_rel_thresh": _optional_positive_float(args.visibility_occlusion_far_rel_thresh),
        "occlusion_far_min_support": args.visibility_occlusion_far_min_support,
        "far_uncertain_as_visible_depth": _optional_positive_float(args.visibility_far_uncertain_as_visible_depth),
    }
    mask_cfg = {
        "mask_from_masked_images": bool(args.visibility_mask_from_masked_images),
        "mask_threshold": int(args.visibility_mask_threshold),
        "mask_erosion_px": int(args.visibility_mask_erosion_px),
        "filter_source_points": bool(args.visibility_filter_source_mask),
    }

    jobs: list[tuple] = []
    for split in splits:
        adapter = BlendedMVSAdapter(
            root=str(root),
            split=split,
            load_precomputed=False,
            load_normals=False,
            verbose=False,
        )
        present = set(adapter.list_sequences())
        sequences = adapter.list_sequences()
        if requested is not None:
            sequences = [seq for seq in requested if seq in present]
        for seq in sequences:
            jobs.append(
                (
                    split,
                    seq,
                    str(root),
                    str(precompute_root),
                    str(output_root),
                    bool(args.overwrite),
                    compression,
                    policy,
                    mask_cfg,
                )
            )
    if requested is not None:
        found = {job[1] for job in jobs}
        missing = [seq for seq in requested if seq not in found]
        if missing:
            raise ValueError(f"Unknown BlendedMVS sequence ids: {missing}")

    print(
        f"[blendedmvs-h5] sequences={len(jobs)} output={output_root} "
        f"compression={args.compression} "
        f"mask_from_masked={mask_cfg['mask_from_masked_images']} "
        f"mask_thresh={mask_cfg['mask_threshold']} "
        f"mask_erosion={mask_cfg['mask_erosion_px']}"
    )
    t0 = time.perf_counter()
    failures: list[tuple[str, str]] = []
    summaries: list[dict[str, Any]] = []
    if args.workers <= 1:
        for job in tqdm(jobs, desc="blendedmvs-h5"):
            seq, err, summary = _materialize_one(job)
            if err:
                failures.append((seq, err))
                tqdm.write(f"[ERR] {seq}: {err.splitlines()[-1]}")
            elif summary is not None:
                summaries.append({"sequence": seq, **summary})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_materialize_one, job): job[1] for job in jobs}
            with tqdm(total=len(futures), desc="blendedmvs-h5") as pbar:
                for fut in as_completed(futures):
                    seq, err, summary = fut.result()
                    pbar.update(1)
                    if err:
                        failures.append((seq, err))
                        tqdm.write(f"[ERR] {seq}: {err.splitlines()[-1]}")
                    elif summary is not None:
                        summaries.append({"sequence": seq, **summary})
    elapsed = time.perf_counter() - t0
    print(f"[blendedmvs-h5] done {len(jobs) - len(failures)}/{len(jobs)} in {elapsed:.1f}s")
    if summaries:
        mins = [int(s["visible_min"]) for s in summaries]
        zeros = sum(int(s["frames_zero"]) for s in summaries)
        lt512 = sum(int(s["frames_lt_512"]) for s in summaries)
        source_dropped = sum(int(s.get("source_mask_dropped_points", 0)) for s in summaries)
        print(
            "[blendedmvs-h5] visible_min min/median/max="
            f"{min(mins)}/{float(np.median(mins)):.1f}/{max(mins)} "
            f"frames_zero={zeros} frames_lt512={lt512} "
            f"source_mask_dropped_total={source_dropped}"
        )
    if failures:
        print("[blendedmvs-h5] failures:")
        for seq, err in failures:
            print(f"  {seq}:\n{err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
