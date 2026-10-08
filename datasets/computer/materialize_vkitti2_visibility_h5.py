#!/usr/bin/env python3
"""Refresh VirtualKitti2 H5 visibility with a distance-aware policy.

This keeps the existing H5 track geometry and metadata, then recomputes
``trajs_2d``, ``valids`` and ``visibs`` from VKITTI2 depth/camera files.
It is intended for visibility-policy experiments where the current tracks
should be preserved, including dynamic VKITTI2 tracks stored in
``trajs_3d_world[t]``.
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

import cv2
import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.VirtualKitti import VKITTI2Adapter


REPLACED_KEYS = {
    "trajs_2d",
    "valids",
    "visibs",
    "valid_visible_counts",
    "valid_counts",
}
PER_FRAME_BIG_KEYS = {
    "trajs_2d",
    "trajs_3d_world",
    "valids",
    "visibs",
    "normals",
}


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
    if arr.ndim >= 2 and arr.shape[0] > 1:
        handle.create_dataset(
            key,
            data=arr,
            chunks=(1,) + arr.shape[1:],
            compression=compression,
        )
    else:
        handle.create_dataset(key, data=arr)


def _copy_metadata(
    src: h5py.File,
    dst: h5py.File,
    *,
    compression: str | None,
    copy_normals: bool,
) -> None:
    for key in sorted(src.keys()):
        if key in REPLACED_KEYS:
            continue
        if key == "trajs_3d_world":
            continue
        if key == "normals" and not copy_normals:
            continue
        value = src[key]
        if key in PER_FRAME_BIG_KEYS and value.ndim >= 2 and value.shape[0] > 1:
            dst.create_dataset(
                key,
                shape=value.shape,
                dtype=value.dtype,
                chunks=(1,) + value.shape[1:],
                compression=compression,
            )
            for t in range(value.shape[0]):
                dst[key][t] = value[t]
        else:
            _write_dataset(dst, key, value[()], compression=None)


def _project_frame(
    points_world: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    policy: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(intrinsic, dtype=np.float32)
    E = np.asarray(extrinsic, dtype=np.float32)
    H, W = depth.shape[:2]
    N = points_world.shape[0]

    uv = np.zeros((N, 2), dtype=np.float32)
    valids = np.zeros((N,), dtype=bool)
    visibs = np.zeros((N,), dtype=bool)
    if N == 0:
        return uv, valids, visibs

    world_finite = np.isfinite(points_world).all(axis=-1)
    valid_world_ids = np.flatnonzero(world_finite)
    if valid_world_ids.size == 0:
        return uv, valids, visibs

    points_h = np.concatenate(
        [
            points_world[valid_world_ids],
            np.ones((valid_world_ids.size, 1), dtype=np.float32),
        ],
        axis=1,
    )
    points_cam = (E @ points_h.T).T[:, :3]
    z = points_cam[:, 2].astype(np.float32)
    safe_z = np.where(z > 0, z, 1.0).astype(np.float32)
    uv_sub = np.empty((valid_world_ids.size, 2), dtype=np.float32)
    uv_sub[:, 0] = points_cam[:, 0] / safe_z * float(K[0, 0]) + float(K[0, 2])
    uv_sub[:, 1] = points_cam[:, 1] / safe_z * float(K[1, 1]) + float(K[1, 2])
    uv[valid_world_ids] = uv_sub

    depth_max = float(policy["depth_max"])
    uv_finite = np.isfinite(uv_sub).all(axis=-1)
    in_front = (z > 0) & np.isfinite(z) & (z < depth_max)
    in_bounds = (
        uv_finite
        & (uv_sub[:, 0] >= 0)
        & (uv_sub[:, 0] < W)
        & (uv_sub[:, 1] >= 0)
        & (uv_sub[:, 1] < H)
    )
    sample_sub = in_bounds & in_front
    if not sample_sub.any():
        return uv, valids, visibs

    sub_ids = np.flatnonzero(sample_sub)
    ids = valid_world_ids[sub_ids]
    uv_sel = uv_sub[sub_ids]
    z_sel = z[sub_ids].astype(np.float32)
    px = np.clip(np.round(uv_sel[:, 0]).astype(np.int32), 0, W - 1)
    py = np.clip(np.round(uv_sel[:, 1]).astype(np.int32), 0, H - 1)

    radius = max(int(policy["depth_support_radius"]), 0)
    min_support = max(int(policy["occlusion_min_support"]), 1)
    vis_min_support = max(int(policy["visible_min_support"]), 1)
    visible_margin = np.maximum(
        float(policy["visible_abs_thresh"]),
        float(policy["depth_consistency_thresh"]) * np.maximum(z_sel, 0.0),
    )
    occlusion_abs = np.full(z_sel.shape, float(policy["occlusion_abs_thresh"]), dtype=np.float32)
    occlusion_rel = np.full(z_sel.shape, float(policy["occlusion_rel_thresh"]), dtype=np.float32)
    support_required = np.full(z_sel.shape, min_support, dtype=np.int16)

    far_start = policy.get("occlusion_far_depth_start")
    if far_start is not None:
        start = float(far_start)
        end = float(policy.get("occlusion_far_depth_end") or start)
        denom = max(end - start, 1e-6)
        alpha = np.clip((z_sel - start) / denom, 0.0, 1.0).astype(np.float32)
        if policy.get("occlusion_far_abs_thresh") is not None:
            occlusion_abs = occlusion_abs + alpha * (
                float(policy["occlusion_far_abs_thresh"]) - occlusion_abs
            )
        if policy.get("occlusion_far_rel_thresh") is not None:
            occlusion_rel = occlusion_rel + alpha * (
                float(policy["occlusion_far_rel_thresh"]) - occlusion_rel
            )
        if policy.get("occlusion_far_min_support") is not None:
            support_required = np.ceil(
                min_support
                + alpha
                * (max(int(policy["occlusion_far_min_support"]), 1) - min_support)
            ).astype(np.int16)

    occlusion_margin = np.maximum(occlusion_abs, occlusion_rel * np.maximum(z_sel, 0.0))
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
                & (sampled < depth_max)
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
    far_uncertain_depth = policy.get("far_uncertain_as_visible_depth")
    if far_uncertain_depth is not None:
        far_uncertain_visible = (
            (z_sel >= float(far_uncertain_depth))
            & (valid_depth_support > 0)
            & (~depth_occluded)
        )
    visible = (visible_support >= vis_min_support) | far_uncertain_visible
    occluded = (~visible) & depth_occluded
    valids[ids] = visible | occluded
    visibs[ids] = visible
    return uv, valids, visibs


def _load_vkitti_frame_payload(
    adapter: VKITTI2Adapter,
    sequence: str,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_dir = adapter.seq_to_dir[sequence]
    frames_dir = seq_dir / "frames"
    depth_dir = adapter._get_actual_dir(frames_dir, "depth")
    depth_path = adapter._frame_path(depth_dir, "depth", frame_idx, ".png")
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Missing VKITTI2 depth frame: {depth_path}")
    sky = depth_raw == np.iinfo(depth_raw.dtype).max
    depth = depth_raw.astype(np.float32) / 100.0
    depth[sky] = 0.0
    pose_dict, K_dict = adapter._load_sequence_metadata(sequence)
    E = pose_dict.get(frame_idx, np.eye(4, dtype=np.float32))
    K = K_dict.get(frame_idx, adapter.default_K.copy())
    return depth, K, E


def _materialize_one(args: tuple) -> tuple[str, str | None, dict[str, Any] | None]:
    (
        sequence,
        root,
        input_root,
        output_root,
        camera,
        overwrite,
        compression,
        copy_normals,
        policy,
    ) = args
    src_path = Path(input_root) / sequence / "precomputed.h5"
    out_dir = Path(output_root) / sequence
    out_path = out_dir / "precomputed.h5"
    tmp_path = out_dir / "precomputed.h5.tmp"
    if out_path.exists() and not overwrite:
        return sequence, None, None
    if not src_path.exists():
        return sequence, f"missing source H5: {src_path}", None
    if overwrite:
        out_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)

    try:
        adapter = VKITTI2Adapter(
            root=root,
            camera=camera,
            load_precomputed=False,
            load_normals=False,
            load_flow=False,
            verbose=False,
        )
        num_frames = adapter.get_num_frames(sequence)
        out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
            if "trajs_3d_world" not in src:
                return sequence, "source H5 missing trajs_3d_world", None
            T, N, C = src["trajs_3d_world"].shape
            if C != 3:
                return sequence, f"bad trajs_3d_world shape: {src['trajs_3d_world'].shape}", None
            if T != num_frames:
                return sequence, f"H5 frames {T} != dataset frames {num_frames}", None

            _copy_metadata(src, dst, compression=compression, copy_normals=copy_normals)
            trajs_3d_ds = dst.create_dataset(
                "trajs_3d_world",
                shape=(T, N, 3),
                dtype=np.float32,
                chunks=(1, N, 3),
                compression=compression,
            )
            trajs_2d_ds = dst.create_dataset(
                "trajs_2d",
                shape=(T, N, 2),
                dtype=np.float32,
                chunks=(1, N, 2),
                compression=compression,
            )
            valids_ds = dst.create_dataset(
                "valids",
                shape=(T, N),
                dtype=np.bool_,
                chunks=(1, N),
                compression=compression,
            )
            visibs_ds = dst.create_dataset(
                "visibs",
                shape=(T, N),
                dtype=np.bool_,
                chunks=(1, N),
                compression=compression,
            )
            valid_counts = np.zeros((T,), dtype=np.int32)
            visible_counts = np.zeros((T,), dtype=np.int32)
            hidden_counts = np.zeros((T,), dtype=np.int32)

            for frame_idx in range(T):
                points_world = np.asarray(src["trajs_3d_world"][frame_idx], dtype=np.float32)
                depth, K, E = _load_vkitti_frame_payload(adapter, sequence, frame_idx)
                uv, valids, visibs = _project_frame(points_world, depth, K, E, policy)
                trajs_3d_ds[frame_idx] = points_world
                trajs_2d_ds[frame_idx] = uv
                valids_ds[frame_idx] = valids
                visibs_ds[frame_idx] = visibs
                valid_counts[frame_idx] = int(valids.sum())
                visible_counts[frame_idx] = int((valids & visibs).sum())
                hidden_counts[frame_idx] = int((valids & ~visibs).sum())

            scalars = {
                "num_frames": np.array(T, dtype=np.int32),
                "num_points": np.array(N, dtype=np.int32),
                "materialized_dense_visibility": np.array(True, dtype=np.bool_),
                "visibility_policy": np.array(1, dtype=np.int32),
                "visibility_policy_name": np.bytes_("vkitti2_far_visible40_from_cached_tracks_v1"),
                "valid_counts": valid_counts,
                "valid_visible_counts": visible_counts,
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
            }
            for key, value in scalars.items():
                if key in dst:
                    del dst[key]
                _write_dataset(dst, key, value, compression=None)
        os.replace(tmp_path, out_path)
        summary = {
            "num_frames": int(T),
            "num_points": int(N),
            "valid_min": int(valid_counts.min()) if T else 0,
            "valid_median": float(np.median(valid_counts)) if T else 0.0,
            "visible_min": int(visible_counts.min()) if T else 0,
            "visible_median": float(np.median(visible_counts)) if T else 0.0,
            "visible_max": int(visible_counts.max()) if T else 0,
            "hidden_total": int(hidden_counts.sum()),
            "frames_lt_512": int((visible_counts < 512).sum()),
            "frames_zero": int((visible_counts == 0).sum()),
            "copy_normals": bool(copy_normals),
            "output_h5": str(out_path),
        }
        return sequence, None, summary
    except Exception:
        tmp_path.unlink(missing_ok=True)
        return sequence, traceback.format_exc(), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--camera", default="Camera_0", choices=["Camera_0", "Camera_1"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression", default="lzf", choices=["none", "lzf", "gzip"])
    parser.add_argument("--copy-normals", action="store_true")
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
    parser.add_argument("--visibility-depth-max", type=float, default=400.0)
    args = parser.parse_args()

    root = Path(args.root)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    compression = None if args.compression == "none" else args.compression
    adapter = VKITTI2Adapter(
        root=str(root),
        camera=args.camera,
        load_precomputed=False,
        load_normals=False,
        load_flow=False,
        verbose=False,
    )
    requested = _parse_sequences(args.sequences)
    sequences = requested or adapter.list_sequences()
    present = set(adapter.list_sequences())
    missing = [seq for seq in sequences if seq not in present]
    if missing:
        raise ValueError(f"Unknown VKITTI2 sequences: {missing}")

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
        "occlusion_far_min_support": int(args.visibility_occlusion_far_min_support),
        "far_uncertain_as_visible_depth": _optional_positive_float(args.visibility_far_uncertain_as_visible_depth),
    }
    jobs = [
        (
            seq,
            str(root),
            str(input_root),
            str(output_root),
            args.camera,
            bool(args.overwrite),
            compression,
            bool(args.copy_normals),
            policy,
        )
        for seq in sequences
    ]
    print(f"[vkitti2-h5] sequences={len(jobs)} output={output_root}")
    print(f"[vkitti2-h5] policy={json.dumps(policy, ensure_ascii=False, sort_keys=True)}")
    t0 = time.perf_counter()
    failures: list[tuple[str, str]] = []
    summaries: dict[str, Any] = {}
    if args.workers <= 1:
        for job in tqdm(jobs, desc="vkitti2-h5"):
            seq, err, summary = _materialize_one(job)
            if err:
                failures.append((seq, err))
                tqdm.write(f"[ERR] {seq}: {err.splitlines()[-1]}")
            elif summary is not None:
                summaries[seq] = summary
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_materialize_one, job): job[0] for job in jobs}
            with tqdm(total=len(futures), desc="vkitti2-h5") as pbar:
                for fut in as_completed(futures):
                    seq, err, summary = fut.result()
                    pbar.update(1)
                    if err:
                        failures.append((seq, err))
                        tqdm.write(f"[ERR] {seq}: {err.splitlines()[-1]}")
                    elif summary is not None:
                        summaries[seq] = summary
    elapsed = time.perf_counter() - t0
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "vkitti2_visibility_materialize_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "root": str(root),
                "input_root": str(input_root),
                "output_root": str(output_root),
                "camera": args.camera,
                "copy_normals": bool(args.copy_normals),
                "policy": policy,
                "num_jobs": len(jobs),
                "num_failures": len(failures),
                "elapsed_s": elapsed,
                "summaries": summaries,
                "failures": [{"sequence": seq, "error": err} for seq, err in failures],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[vkitti2-h5] done {len(jobs) - len(failures)}/{len(jobs)} in {elapsed:.1f}s")
    print(f"[vkitti2-h5] summary={summary_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
