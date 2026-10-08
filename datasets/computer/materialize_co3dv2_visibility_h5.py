#!/usr/bin/env python3
"""Materialize Co3Dv2 point tracks into visibility-aware precomputed.h5 files.

The existing Co3Dv2 ``precomputed.npz`` files usually contain
``visibs == valids``.  This script keeps the existing 2D/3D tracks, rebuilds
frame-wise ``valids`` from the Co3D foreground masks, computes a conservative
occlusion visibility signal from Co3D depth maps, and writes chunked HDF5 files
that the Co3Dv2 adapter already prefers over npz.
"""

from __future__ import annotations

import argparse
import json
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

from adapters.co3dv2 import (  # noqa: E402
    Co3Dv2Adapter,
    _load_depth,
    _load_depth_mask,
    _load_soft_mask,
    _ndc_to_pinhole,
    _p3d_to_opencv_extrinsics,
    _precomputed_tracks_look_temporal,
    _summarize_precomputed_track_validity,
)


PER_FRAME_KEYS = {"trajs_2d", "trajs_3d_world", "valids", "visibs", "normals"}
VISIBILITY_POLICY = "co3dv2_foreground_depth_conservative_v1"

_WORKER_ADAPTER: Co3Dv2Adapter | None = None
_WORKER_ROOT: Path | None = None
_WORKER_OUTPUT_ROOT: Path | None = None
_WORKER_ARGS: dict[str, Any] = {}


def _parse_sequences(path: str | None) -> list[str] | None:
    if not path:
        return None
    p = Path(path)
    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text())
        if isinstance(raw, dict):
            for key in ("sequences", "sequence_names", "items"):
                if key in raw:
                    raw = raw[key]
                    break
        if not isinstance(raw, list):
            raise ValueError(f"Unsupported sequence-list JSON format: {path}")
        return [str(x).strip() for x in raw if str(x).strip()]
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    mask = np.asarray(mask, dtype=bool)
    if radius == 0:
        return mask
    try:
        import cv2

        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        out = np.zeros_like(mask, dtype=bool)
        k = 2 * radius + 1
        for yy in range(k):
            for xx in range(k):
                out |= padded[yy : yy + mask.shape[0], xx : xx + mask.shape[1]]
        return out


def _sample_depth_patch_counts(
    depth: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    z: np.ndarray,
    *,
    radius: int,
    occlusion_abs_threshold: float,
    occlusion_rel_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid-depth support and nearer-depth support for projected points."""
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape[:2]
    px = np.asarray(px, dtype=np.int64)
    py = np.asarray(py, dtype=np.int64)
    z = np.asarray(z, dtype=np.float32)

    support = np.zeros(px.shape, dtype=np.int16)
    nearer = np.zeros(px.shape, dtype=np.int16)
    margin = np.maximum(
        float(occlusion_abs_threshold),
        float(occlusion_rel_threshold) * np.maximum(z, 0.0),
    ).astype(np.float32)

    radius = max(int(radius), 0)
    for dy in range(-radius, radius + 1):
        yy = np.clip(py + dy, 0, h - 1)
        for dx in range(-radius, radius + 1):
            xx = np.clip(px + dx, 0, w - 1)
            d = depth[yy, xx].astype(np.float32, copy=False)
            ok = (d > 1e-6) & np.isfinite(d)
            support += ok.astype(np.int16)
            nearer += (ok & (d + margin < z)).astype(np.int16)
    return support, nearer


def _load_frame_geometry(
    adapter: Co3Dv2Adapter,
    category: str,
    sequence_name: str,
    frame_number: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anno = adapter._get_frame_anno(category, sequence_name, frame_number)
    depth = _load_depth(
        adapter.root / anno["depth"]["path"],
        anno["depth"]["scale_adjustment"],
    )
    depth_mask_path = adapter.root / anno["depth"]["mask_path"]
    if depth_mask_path.exists():
        depth[~_load_depth_mask(depth_mask_path)] = 0.0
    fg = _load_soft_mask(adapter.root / anno["mask"]["path"])
    intrinsic = _ndc_to_pinhole(
        anno["viewpoint"]["focal_length"],
        anno["viewpoint"]["principal_point"],
        anno["image"]["size"],
    )
    extrinsic = _p3d_to_opencv_extrinsics(
        anno["viewpoint"]["R"],
        anno["viewpoint"]["T"],
    )
    return depth, fg, intrinsic, extrinsic


def _compute_visibility_arrays(
    adapter: Co3Dv2Adapter,
    sequence: str,
    trajs_2d: np.ndarray,
    trajs_3d_world: np.ndarray,
    source_valids: np.ndarray | None,
    *,
    foreground_threshold: float,
    foreground_dilate_px: int,
    foreground_mask_fallback: bool,
    foreground_mask_fallback_min_source_points: int,
    foreground_mask_fallback_min_kept_points: int,
    foreground_mask_fallback_min_keep_ratio: float,
    depth_max: float,
    depth_support_radius: int,
    occlusion_abs_threshold: float,
    occlusion_rel_threshold: float,
    occlusion_min_support: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    record = adapter._get_record(sequence)
    adapter._ensure_frame_anno_loaded(record.category)
    t_count, n_points = trajs_2d.shape[:2]
    if t_count != len(record.frame_numbers):
        raise ValueError(
            f"{sequence}: cache frames={t_count} but adapter frames={len(record.frame_numbers)}"
        )

    new_valids = np.zeros((t_count, n_points), dtype=bool)
    new_visibs = np.zeros((t_count, n_points), dtype=bool)
    per_frame: list[dict[str, int]] = []
    fallback_frames: list[int] = []
    source_valids_np = None
    if source_valids is not None:
        source_valids_np = np.asarray(source_valids, dtype=bool)
        if source_valids_np.shape != (t_count, n_points):
            source_valids_np = None

    ones = np.ones((n_points, 1), dtype=np.float32)
    for t, frame_number in enumerate(record.frame_numbers):
        depth, fg_soft, intrinsic, extrinsic = _load_frame_geometry(
            adapter,
            record.category,
            record.sequence_name,
            int(frame_number),
        )
        foreground = _dilate_mask(fg_soft >= float(foreground_threshold), foreground_dilate_px)
        h, w = foreground.shape[:2]
        uv = np.asarray(trajs_2d[t], dtype=np.float32)
        points_world = np.asarray(trajs_3d_world[t], dtype=np.float32)
        points_h = np.concatenate([points_world, ones], axis=1)
        points_cam = (np.asarray(extrinsic, dtype=np.float32) @ points_h.T).T[:, :3]
        z = points_cam[:, 2].astype(np.float32)

        finite_uv = np.isfinite(uv).all(axis=1)
        finite_world = np.isfinite(points_world).all(axis=1)
        in_front = (z > 1e-6) & np.isfinite(z) & (z < float(depth_max))
        in_bounds = (
            finite_uv
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        geometry_valid = in_bounds & in_front & finite_world
        source_valid = geometry_valid
        if source_valids_np is not None:
            source_valid = source_valid & source_valids_np[t]

        source_ids = np.flatnonzero(source_valid)
        ids = source_ids
        px = np.empty((0,), dtype=np.int64)
        py = np.empty((0,), dtype=np.int64)
        mask_kept = 0
        mask_fallback = False
        if len(source_ids):
            src_px = np.clip(np.round(uv[source_ids, 0]).astype(np.int64), 0, w - 1)
            src_py = np.clip(np.round(uv[source_ids, 1]).astype(np.int64), 0, h - 1)
            mask_ok = foreground[src_py, src_px]
            mask_kept = int(mask_ok.sum())
            keep_ratio = mask_kept / max(len(source_ids), 1)
            mask_fallback = (
                bool(foreground_mask_fallback)
                and len(source_ids) >= int(foreground_mask_fallback_min_source_points)
                and mask_kept < int(foreground_mask_fallback_min_kept_points)
                and keep_ratio < float(foreground_mask_fallback_min_keep_ratio)
            )
            if mask_fallback:
                fallback_frames.append(int(t))
                px = src_px
                py = src_py
            else:
                ids = source_ids[mask_ok]
                px = src_px[mask_ok]
                py = src_py[mask_ok]

        visible = np.zeros((n_points,), dtype=bool)
        valid = np.zeros((n_points,), dtype=bool)
        occluded_count = 0
        if len(ids):
            valid[ids] = True
            support, nearer = _sample_depth_patch_counts(
                depth,
                px,
                py,
                z[ids],
                radius=depth_support_radius,
                occlusion_abs_threshold=occlusion_abs_threshold,
                occlusion_rel_threshold=occlusion_rel_threshold,
            )
            occluded = nearer >= int(occlusion_min_support)
            visible_ids = ids[~occluded]
            visible[visible_ids] = True
            occluded_count = int(occluded.sum())

        new_valids[t] = valid
        new_visibs[t] = visible
        per_frame.append(
            {
                "frame_index": int(t),
                "frame_number": int(frame_number),
                "source_valid": int(source_valid.sum()),
                "foreground_kept": mask_kept,
                "foreground_mask_fallback": bool(mask_fallback),
                "valid": int(valid.sum()),
                "visible": int(visible.sum()),
                "occluded": occluded_count,
            }
        )

    stats = {
        "frames": int(t_count),
        "points": int(n_points),
        "valid_total": int(new_valids.sum()),
        "visible_total": int(new_visibs.sum()),
        "hidden_total": int((new_valids & ~new_visibs).sum()),
        "visible_outside_valid": int((new_visibs & ~new_valids).sum()),
        "valid_per_frame_min": int(new_valids.sum(axis=1).min()) if t_count else 0,
        "valid_per_frame_median": float(np.median(new_valids.sum(axis=1))) if t_count else 0.0,
        "valid_per_frame_max": int(new_valids.sum(axis=1).max()) if t_count else 0,
        "visible_per_frame_min": int(new_visibs.sum(axis=1).min()) if t_count else 0,
        "visible_per_frame_median": float(np.median(new_visibs.sum(axis=1))) if t_count else 0.0,
        "visible_per_frame_max": int(new_visibs.sum(axis=1).max()) if t_count else 0,
        "foreground_mask_fallback_frames": fallback_frames,
        "foreground_mask_fallback_frame_count": len(fallback_frames),
        "per_frame_preview": per_frame[:5],
    }
    return new_valids, new_visibs, stats


def _normalize_for_h5(value: Any) -> tuple[Any, dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    if arr.dtype.kind == "U":
        kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
        return arr.astype(object), kwargs
    if arr.dtype.kind == "O":
        if arr.ndim == 0:
            item = arr.item()
            if isinstance(item, bytes):
                item = item.decode("utf-8", errors="replace")
            if item is None:
                item = ""
            kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
            return np.asarray(str(item), dtype=object), kwargs
        kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
        flat = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in arr.reshape(-1)]
        return np.asarray(flat, dtype=object).reshape(arr.shape), kwargs
    return arr, kwargs


def _write_dataset(handle: h5py.File, key: str, value: Any, compression: str | None) -> None:
    arr, kwargs = _normalize_for_h5(value)
    if isinstance(arr, np.ndarray) and key in PER_FRAME_KEYS and arr.ndim >= 2 and arr.shape[0] > 1:
        kwargs.setdefault("chunks", (1,) + arr.shape[1:])
        if compression and compression.lower() != "none":
            kwargs.setdefault("compression", compression)
    handle.create_dataset(key, data=arr, **kwargs)


def _materialize_one(sequence: str) -> dict[str, Any]:
    adapter = _WORKER_ADAPTER
    root = _WORKER_ROOT
    output_root = _WORKER_OUTPUT_ROOT
    args = _WORKER_ARGS
    if adapter is None or root is None or output_root is None:
        raise RuntimeError("worker is not initialized")

    src_npz = root / sequence / "precomputed.npz"
    out_h5 = output_root / sequence / "precomputed.h5"
    result = {
        "sequence": sequence,
        "source_npz": str(src_npz),
        "out_h5": str(out_h5),
    }
    if not src_npz.exists():
        result["status"] = "skip_missing_npz"
        return result
    if out_h5.exists() and not args["overwrite"]:
        result["status"] = "skip_h5_exists"
        return result

    with np.load(src_npz, allow_pickle=True) as src:
        payload = {key: src[key] for key in src.files if args["copy_normals"] or key != "normals"}

    required = {"trajs_2d", "trajs_3d_world"}
    missing = required - set(payload)
    if missing:
        result["status"] = "skip_missing_keys"
        result["missing_keys"] = sorted(missing)
        return result

    valids, visibs, stats = _compute_visibility_arrays(
        adapter,
        sequence,
        payload["trajs_2d"],
        payload["trajs_3d_world"],
        payload.get("valids"),
        foreground_threshold=args["foreground_threshold"],
        foreground_dilate_px=args["foreground_dilate_px"],
        foreground_mask_fallback=args["foreground_mask_fallback"],
        foreground_mask_fallback_min_source_points=args["foreground_mask_fallback_min_source_points"],
        foreground_mask_fallback_min_kept_points=args["foreground_mask_fallback_min_kept_points"],
        foreground_mask_fallback_min_keep_ratio=args["foreground_mask_fallback_min_keep_ratio"],
        depth_max=args["depth_max"],
        depth_support_radius=args["depth_support_radius"],
        occlusion_abs_threshold=args["occlusion_abs_threshold"],
        occlusion_rel_threshold=args["occlusion_rel_threshold"],
        occlusion_min_support=args["occlusion_min_support"],
    )
    if not _precomputed_tracks_look_temporal(visibs):
        stats_for_guard = _summarize_precomputed_track_validity(visibs)
        result["status"] = "skip_degenerate_visibility"
        result["track_stats"] = stats_for_guard
        return result

    payload["valids"] = valids
    payload["visibs"] = visibs
    payload["has_real_visibility"] = np.asarray(True)
    payload["co3d_has_real_visibility"] = np.asarray(True)
    payload["visibility_policy"] = np.asarray(VISIBILITY_POLICY)
    payload["visibility_semantics_version"] = np.asarray(1, dtype=np.int32)
    payload["visibility_foreground_threshold"] = np.asarray(args["foreground_threshold"], dtype=np.float32)
    payload["visibility_foreground_dilate_px"] = np.asarray(args["foreground_dilate_px"], dtype=np.int32)
    payload["visibility_foreground_mask_fallback"] = np.asarray(bool(args["foreground_mask_fallback"]))
    payload["visibility_foreground_mask_fallback_min_source_points"] = np.asarray(
        args["foreground_mask_fallback_min_source_points"],
        dtype=np.int32,
    )
    payload["visibility_foreground_mask_fallback_min_kept_points"] = np.asarray(
        args["foreground_mask_fallback_min_kept_points"],
        dtype=np.int32,
    )
    payload["visibility_foreground_mask_fallback_min_keep_ratio"] = np.asarray(
        args["foreground_mask_fallback_min_keep_ratio"],
        dtype=np.float32,
    )
    payload["visibility_depth_support_radius"] = np.asarray(args["depth_support_radius"], dtype=np.int32)
    payload["visibility_occlusion_abs_threshold"] = np.asarray(args["occlusion_abs_threshold"], dtype=np.float32)
    payload["visibility_occlusion_rel_threshold"] = np.asarray(args["occlusion_rel_threshold"], dtype=np.float32)
    payload["visibility_occlusion_min_support"] = np.asarray(args["occlusion_min_support"], dtype=np.int32)
    payload["visibility_uncertain_as_visible"] = np.asarray(True)
    payload["validity_mask_source"] = np.asarray("co3d_foreground_mask")
    payload["validity_mask_fallback"] = np.asarray("source_valid_when_foreground_overlap_is_too_low")
    payload["num_frames"] = np.asarray(valids.shape[0], dtype=np.int32)
    payload["num_points"] = np.asarray(valids.shape[1], dtype=np.int32)

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_h5.with_name(out_h5.name + ".tmp")
    with h5py.File(tmp_path, "w") as handle:
        for key, value in payload.items():
            _write_dataset(handle, key, value, args["compression"])
    os.replace(tmp_path, out_h5)

    result.update(stats)
    result["status"] = "wrote_h5"
    result["copy_normals"] = bool(args["copy_normals"])
    result["size_bytes"] = int(out_h5.stat().st_size)
    return result


def _init_worker(adapter_kwargs: dict[str, Any], root: str, output_root: str, worker_args: dict[str, Any]) -> None:
    global _WORKER_ADAPTER, _WORKER_ROOT, _WORKER_OUTPUT_ROOT, _WORKER_ARGS
    _WORKER_ROOT = Path(root)
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_ARGS = dict(worker_args)
    _WORKER_ADAPTER = Co3Dv2Adapter(**adapter_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write Co3Dv2 foreground/depth visibility precomputed.h5 files."
    )
    parser.add_argument("--root", required=True, help="Co3Dv2 dataset root.")
    parser.add_argument("--output-root", default=None, help="Output root; defaults to --root.")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--subset-name", default="fewview_train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequence-list", default=None)
    parser.add_argument("--sequence-denylist", default=None)
    parser.add_argument("--require-pointcloud", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-normals", action="store_true", help="Copy normals into h5; much larger/slower.")
    parser.add_argument("--compression", default="lzf", choices=["none", "lzf", "gzip"])
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--foreground-dilate-px", type=int, default=2)
    parser.add_argument(
        "--disable-foreground-mask-fallback",
        action="store_true",
        help="Strictly trust Co3D foreground masks, even when a frame has near-zero mask/track overlap.",
    )
    parser.add_argument("--foreground-mask-fallback-min-source-points", type=int, default=512)
    parser.add_argument("--foreground-mask-fallback-min-kept-points", type=int, default=64)
    parser.add_argument("--foreground-mask-fallback-min-keep-ratio", type=float, default=0.01)
    parser.add_argument("--depth-max", type=float, default=50.0)
    parser.add_argument("--depth-support-radius", type=int, default=2)
    parser.add_argument("--occlusion-abs-threshold", type=float, default=0.10)
    parser.add_argument("--occlusion-rel-threshold", type=float, default=0.08)
    parser.add_argument("--occlusion-min-support", type=int, default=5)
    parser.add_argument("--index-workers", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_root) if args.output_root else root
    adapter_kwargs = {
        "root": str(root),
        "categories": args.categories,
        "subset_name": args.subset_name,
        "split": args.split,
        "verbose": True,
        "load_normals": False,
        "require_pointcloud": bool(args.require_pointcloud),
        "sequence_denylist": args.sequence_denylist,
        "index_workers": int(args.index_workers),
        "cache_dir": args.cache_dir,
    }
    adapter = Co3Dv2Adapter(**adapter_kwargs)
    sequences = _parse_sequences(args.sequence_list) or adapter.list_sequences()
    available = set(adapter.list_sequences())
    sequences = [seq for seq in sequences if seq in available]
    if args.limit is not None:
        sequences = sequences[: int(args.limit)]

    worker_args = {
        "overwrite": bool(args.overwrite),
        "copy_normals": bool(args.copy_normals),
        "compression": None if args.compression == "none" else args.compression,
        "foreground_threshold": float(args.foreground_threshold),
        "foreground_dilate_px": int(args.foreground_dilate_px),
        "foreground_mask_fallback": not bool(args.disable_foreground_mask_fallback),
        "foreground_mask_fallback_min_source_points": int(args.foreground_mask_fallback_min_source_points),
        "foreground_mask_fallback_min_kept_points": int(args.foreground_mask_fallback_min_kept_points),
        "foreground_mask_fallback_min_keep_ratio": float(args.foreground_mask_fallback_min_keep_ratio),
        "depth_max": float(args.depth_max),
        "depth_support_radius": int(args.depth_support_radius),
        "occlusion_abs_threshold": float(args.occlusion_abs_threshold),
        "occlusion_rel_threshold": float(args.occlusion_rel_threshold),
        "occlusion_min_support": int(args.occlusion_min_support),
    }
    print(
        json.dumps(
            {
                "root": str(root),
                "output_root": str(output_root),
                "sequences": len(sequences),
                "workers": int(args.workers),
                "policy": VISIBILITY_POLICY,
                **worker_args,
            },
            indent=2,
        )
    )

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        _init_worker(adapter_kwargs, str(root), str(output_root), worker_args)
        for seq in tqdm(sequences, desc="co3d_visibility_h5"):
            try:
                results.append(_materialize_one(seq))
            except Exception as exc:
                results.append(
                    {
                        "sequence": seq,
                        "status": "error",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    else:
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_worker,
            initargs=(adapter_kwargs, str(root), str(output_root), worker_args),
        ) as pool:
            futures = {pool.submit(_materialize_one, seq): seq for seq in sequences}
            with tqdm(total=len(futures), desc="co3d_visibility_h5") as pbar:
                for fut in as_completed(futures):
                    seq = futures[fut]
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        results.append(
                            {
                                "sequence": seq,
                                "status": "error",
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                    pbar.update(1)

    summary: dict[str, Any] = {
        "root": str(root),
        "output_root": str(output_root),
        "policy": VISIBILITY_POLICY,
        "elapsed_sec": time.perf_counter() - t0,
        "total": len(results),
    }
    for status in sorted({r["status"] for r in results}):
        summary[status] = sum(1 for r in results if r["status"] == status)
    if results:
        written = [r for r in results if r.get("status") == "wrote_h5"]
        if written:
            summary["valid_per_frame_min_min"] = min(r["valid_per_frame_min"] for r in written)
            summary["visible_per_frame_min_min"] = min(r["visible_per_frame_min"] for r in written)
            summary["hidden_total"] = int(sum(r["hidden_total"] for r in written))
            summary["visible_outside_valid"] = int(sum(r["visible_outside_valid"] for r in written))
            summary["foreground_mask_fallback_frame_count"] = int(
                sum(r.get("foreground_mask_fallback_frame_count", 0) for r in written)
            )
            summary["written_size_gb"] = float(sum(r["size_bytes"] for r in written) / 1e9)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
