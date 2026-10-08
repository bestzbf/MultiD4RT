#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datasets.adapters.Waymo import _build_depth_dense, _make_v2c_std  # noqa: E402

_WAYMO_DYNAMIC_LABEL_TYPES = (1, 2, 4)  # vehicle, pedestrian, cyclist


def _h5_write_dataset(f: h5py.File, key: str, value: np.ndarray) -> None:
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O", "S"}:
        f.create_dataset(key, data=arr.astype(object), dtype=h5py.string_dtype("utf-8"))
    elif arr.ndim >= 2 and arr.shape[0] > 1:
        f.create_dataset(key, data=arr, chunks=(1,) + arr.shape[1:], compression="lzf")
    else:
        f.create_dataset(key, data=arr)


def _load_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            payload[key] = f[key][()]
    return payload


def _save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as f:
        for key, value in payload.items():
            _h5_write_dataset(f, key, np.asarray(value))
    tmp_path.replace(path)


def _camera_calibration(seq_dir: Path, camera_name: int) -> tuple[np.ndarray, np.ndarray]:
    with open(seq_dir / "calib.json", "r", encoding="utf-8") as f:
        calib_list = json.load(f)
    calib_map = {int(cam_name): info for cam_name, info in calib_list}
    cam_calib = calib_map[int(camera_name)]
    intr = cam_calib["intrinsics"]
    intrinsic = np.array(
        [
            [intr[0], 0.0, intr[2]],
            [0.0, intr[1], intr[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    c2v = np.array(cam_calib["extrinsics"], dtype=np.float64).reshape(4, 4)
    return intrinsic, _make_v2c_std(c2v)


def _frame_shape(seq_dir: Path, frame_idx: int, camera_name: int) -> tuple[int, int]:
    with Image.open(seq_dir / f"{frame_idx:05d}_{camera_name}.jpg") as img:
        width, height = img.size
    return height, width


def _project_points(
    points_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    cam = (np.asarray(extrinsic, dtype=np.float32) @ points_h.T).T[:, :3]
    z = cam[:, 2].astype(np.float32)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    ok = z > 1.0e-6
    if ok.any():
        k = np.asarray(intrinsic, dtype=np.float32)
        uv[ok, 0] = cam[ok, 0] / z[ok] * k[0, 0] + k[0, 2]
        uv[ok, 1] = cam[ok, 1] / z[ok] * k[1, 1] + k[1, 2]
    return uv, z


def _zbuffer_occlusion_mask(
    *,
    defined: np.ndarray,
    uv: np.ndarray,
    z: np.ndarray,
    image_h: int,
    image_w: int,
    margin: np.ndarray,
    support_radius: int = 0,
    min_support: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return in-frame projected tracks and z-buffer occlusion mask.

    With ``min_support > 1`` a point is considered occluded only when multiple
    nearby z-buffer pixels contain a confidently nearer surface. This avoids
    marking occlusion from a single quantized projected point.
    """
    px = np.zeros(len(z), dtype=np.int32)
    py = np.zeros(len(z), dtype=np.int32)
    px[defined] = np.round(uv[defined, 0]).astype(np.int32)
    py[defined] = np.round(uv[defined, 1]).astype(np.int32)
    valid_px = defined & (px >= 0) & (px < image_w) & (py >= 0) & (py < image_h)
    is_occluded = np.zeros(len(z), dtype=bool)
    if not valid_px.any():
        return valid_px, is_occluded

    flat = py[valid_px] * image_w + px[valid_px]
    min_depth_flat = np.full(image_h * image_w, np.inf, dtype=np.float32)
    np.minimum.at(min_depth_flat, flat, z[valid_px])
    valid_ids = np.flatnonzero(valid_px)
    if int(min_support) <= 1 and int(support_radius) <= 0:
        min_depth = min_depth_flat[flat]
        is_occluded[valid_px] = z[valid_px] > (min_depth + margin[valid_px])
        return valid_px, is_occluded

    min_depth_image = min_depth_flat.reshape(image_h, image_w)
    r = max(int(support_radius), 0)
    yy_off, xx_off = np.mgrid[-r : r + 1, -r : r + 1]
    offsets_x = xx_off.reshape(1, -1).astype(np.int32)
    offsets_y = yy_off.reshape(1, -1).astype(np.int32)
    xx = px[valid_ids, None] + offsets_x
    yy = py[valid_ids, None] + offsets_y
    inside = (xx >= 0) & (xx < image_w) & (yy >= 0) & (yy < image_h)
    xx_clip = np.clip(xx, 0, image_w - 1)
    yy_clip = np.clip(yy, 0, image_h - 1)
    local_depth = min_depth_image[yy_clip, xx_clip]
    nearer = inside & np.isfinite(local_depth) & (local_depth + margin[valid_ids, None] < z[valid_ids, None])
    is_occluded[valid_ids] = nearer.sum(axis=1) >= max(int(min_support), 1)
    return valid_px, is_occluded


def _local_depth_support_masks(
    *,
    raw_depth: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    z: np.ndarray,
    margin: np.ndarray,
    radius: int,
    surface_min_support: int,
    occlusion_min_support: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return depth-supported visible/occluded masks for projected points.

    A point is visible only if nearby raw LiDAR supports the same surface depth.
    A point is occluded only if multiple nearby raw LiDAR samples are confidently
    closer.  Everything else is ambiguous and can be made invalid by the caller.
    """
    raw = np.asarray(raw_depth, dtype=np.float32)
    px = np.asarray(px, dtype=np.int32).reshape(-1)
    py = np.asarray(py, dtype=np.int32).reshape(-1)
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    margin = np.asarray(margin, dtype=np.float32).reshape(-1)
    n = int(px.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)

    h, w = raw.shape[:2]
    r = max(int(radius), 0)
    yy_off, xx_off = np.mgrid[-r : r + 1, -r : r + 1]
    offsets_x = xx_off.reshape(1, -1).astype(np.int32)
    offsets_y = yy_off.reshape(1, -1).astype(np.int32)

    xx = px[:, None] + offsets_x
    yy = py[:, None] + offsets_y
    inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    xx_clip = np.clip(xx, 0, w - 1)
    yy_clip = np.clip(yy, 0, h - 1)
    local_depth = raw[yy_clip, xx_clip]
    has_depth = inside & np.isfinite(local_depth) & (local_depth > 0.0)

    z_col = z[:, None]
    margin_col = margin[:, None]
    surface_count = (has_depth & (np.abs(local_depth - z_col) <= margin_col)).sum(axis=1)
    occlusion_count = (has_depth & ((local_depth + margin_col) < z_col)).sum(axis=1)
    surface_supported = surface_count >= max(int(surface_min_support), 1)
    occluded_supported = occlusion_count >= max(int(occlusion_min_support), 1)
    return surface_supported, occluded_supported


def _rot_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _points_in_waymo_boxes(points_vehicle: np.ndarray, boxes_vehicle: np.ndarray, margin: float) -> np.ndarray:
    points_vehicle = np.asarray(points_vehicle, dtype=np.float32).reshape(-1, 3)
    boxes_vehicle = np.asarray(boxes_vehicle, dtype=np.float32).reshape(-1, 7)
    inside_any = np.zeros((len(points_vehicle),), dtype=bool)
    if len(points_vehicle) == 0 or len(boxes_vehicle) == 0:
        return inside_any
    for box in boxes_vehicle:
        local = (points_vehicle - box[:3][None, :]) @ _rot_z(float(box[6]))
        half = np.asarray(box[3:6], dtype=np.float32) * 0.5 + float(margin)
        inside_any |= (
            (np.abs(local[:, 0]) <= half[0])
            & (np.abs(local[:, 1]) <= half[1])
            & (np.abs(local[:, 2]) <= half[2])
        )
    return inside_any


def _load_waymo_dynamic_label_rows(seq_dir: Path) -> tuple[dict[int, list[tuple[str, np.ndarray]]], dict[str, list[tuple[int, np.ndarray]]]]:
    labels_path = seq_dir / "labels.npz"
    if not labels_path.exists():
        return {}, {}
    data = np.load(labels_path)
    rows_by_frame: dict[int, list[tuple[str, np.ndarray]]] = {}
    rows_by_object: dict[str, list[tuple[int, np.ndarray]]] = {}
    for frame_idx, object_id, label_type, box in zip(
        data["frame_indices"],
        data["object_ids"],
        data["label_types"],
        data["boxes_vehicle"],
    ):
        if int(label_type) not in _WAYMO_DYNAMIC_LABEL_TYPES:
            continue
        frame_id = int(frame_idx)
        oid = str(object_id)
        box_arr = np.asarray(box, dtype=np.float32)
        rows_by_frame.setdefault(frame_id, []).append((oid, box_arr))
        rows_by_object.setdefault(oid, []).append((frame_id, box_arr))
    return rows_by_frame, rows_by_object


def _moving_waymo_object_ids(
    *,
    seq_dir: Path,
    rows_by_object: dict[str, list[tuple[int, np.ndarray]]],
    camera_name: int,
    min_extent_m: float,
    min_step_m: float,
    min_frames: int,
) -> set[str]:
    moving: set[str] = set()
    pose_cache: dict[int, np.ndarray] = {}
    for object_id, rows in rows_by_object.items():
        if len(rows) < int(min_frames):
            continue
        centers_world = []
        for frame_id, box in sorted(rows):
            pose = pose_cache.get(frame_id)
            if pose is None:
                npz_path = seq_dir / f"{frame_id:05d}_{camera_name}.npz"
                if not npz_path.exists():
                    continue
                pose = np.load(npz_path)["pose"].reshape(4, 4).astype(np.float32)
                pose_cache[frame_id] = pose
            center_h = np.asarray([box[0], box[1], box[2], 1.0], dtype=np.float32)
            centers_world.append((pose @ center_h)[:3])
        if len(centers_world) < int(min_frames):
            continue
        centers = np.stack(centers_world, axis=0).astype(np.float32)
        extent = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
        steps = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
        max_step = float(steps.max()) if len(steps) else 0.0
        if extent >= float(min_extent_m) and max_step >= float(min_step_m):
            moving.add(object_id)
    return moving


def _compute_static_dynamic_box_reject_mask(
    *,
    seq_dir: Path,
    trajs_3d_world: np.ndarray,
    track_is_dynamic: np.ndarray,
    camera_name: int,
    filter_mode: str,
    box_margin: float,
    moving_min_extent_m: float,
    moving_min_step_m: float,
    moving_min_frames: int,
    min_inside_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_tracks = int(trajs_3d_world.shape[1])
    reject = np.zeros((n_tracks,), dtype=bool)
    if filter_mode == "off":
        return reject, {"static_dynamic_box_filter_mode": "off"}

    rows_by_frame, rows_by_object = _load_waymo_dynamic_label_rows(seq_dir)
    if not rows_by_frame:
        return reject, {
            "static_dynamic_box_filter_mode": filter_mode,
            "static_dynamic_box_labels_found": False,
        }

    moving_ids: set[str] = set()
    if filter_mode == "moving":
        moving_ids = _moving_waymo_object_ids(
            seq_dir=seq_dir,
            rows_by_object=rows_by_object,
            camera_name=camera_name,
            min_extent_m=moving_min_extent_m,
            min_step_m=moving_min_step_m,
            min_frames=moving_min_frames,
        )
        if not moving_ids:
            return reject, {
                "static_dynamic_box_filter_mode": filter_mode,
                "static_dynamic_box_labels_found": True,
                "static_dynamic_box_moving_objects": 0,
            }

    static_mask = ~np.asarray(track_is_dynamic, dtype=bool)
    inside_counts = np.zeros((n_tracks,), dtype=np.int32)
    pose_cache: dict[int, np.ndarray] = {}
    t_total = int(trajs_3d_world.shape[0])
    for frame_id in range(t_total):
        rows = rows_by_frame.get(frame_id, [])
        if filter_mode == "moving":
            rows = [(oid, box) for oid, box in rows if oid in moving_ids]
        if not rows:
            continue
        boxes = np.stack([box for _, box in rows], axis=0).astype(np.float32)
        pose = pose_cache.get(frame_id)
        if pose is None:
            npz_path = seq_dir / f"{frame_id:05d}_{camera_name}.npz"
            if not npz_path.exists():
                continue
            pose = np.load(npz_path)["pose"].reshape(4, 4).astype(np.float32)
            pose_cache[frame_id] = pose
        w2v = np.linalg.inv(pose)
        pts_world = np.asarray(trajs_3d_world[frame_id], dtype=np.float32)
        candidate = static_mask & np.isfinite(pts_world).all(axis=1)
        if not candidate.any():
            continue
        idx = np.flatnonzero(candidate)
        pts_h = np.concatenate([pts_world[idx], np.ones((len(idx), 1), dtype=np.float32)], axis=1)
        pts_vehicle = (w2v @ pts_h.T).T[:, :3]
        inside_counts[idx] += _points_in_waymo_boxes(pts_vehicle, boxes, margin=box_margin).astype(np.int32)

    reject = static_mask & (inside_counts >= int(min_inside_frames))
    return reject, {
        "static_dynamic_box_filter_mode": filter_mode,
        "static_dynamic_box_labels_found": True,
        "static_dynamic_box_moving_objects": int(len(moving_ids)) if filter_mode == "moving" else -1,
        "static_dynamic_box_margin": float(box_margin),
        "static_dynamic_box_min_inside_frames": int(min_inside_frames),
        "static_dynamic_box_rejected_tracks": int(reject.sum()),
        "static_dynamic_box_inside_frames_p50_p90_max": [
            float(np.percentile(inside_counts[static_mask], 50)) if static_mask.any() else 0.0,
            float(np.percentile(inside_counts[static_mask], 90)) if static_mask.any() else 0.0,
            int(inside_counts[static_mask].max()) if static_mask.any() else 0,
        ],
    }


def _stabilize_low_motion_static_tracks(
    *,
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: np.ndarray,
    track_is_dynamic: np.ndarray,
    max_extent_px: float,
    max_step_px: float,
    min_visible_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.asarray(trajs_2d, dtype=np.float32).copy()
    if max_extent_px < 0 or max_step_px < 0:
        return out, {"static_2d_stabilize_applied": False}

    vis = np.asarray(valids, dtype=bool) & np.asarray(visibs, dtype=bool)
    static_mask = ~np.asarray(track_is_dynamic, dtype=bool)
    stabilize = np.zeros((out.shape[1],), dtype=bool)
    extents = []
    max_steps = []
    for q in np.flatnonzero(static_mask):
        visible = vis[:, q] & np.isfinite(out[:, q]).all(axis=1)
        if int(visible.sum()) < int(min_visible_frames):
            continue
        pts = out[visible, q]
        extent = float(np.linalg.norm(np.ptp(pts, axis=0)))
        step = 0.0
        pair_visible = visible[1:] & visible[:-1]
        if pair_visible.any():
            diffs = np.linalg.norm(out[1:, q] - out[:-1, q], axis=1)
            step = float(np.nanmax(diffs[pair_visible]))
        extents.append(extent)
        max_steps.append(step)
        if extent <= float(max_extent_px) and step <= float(max_step_px):
            center = np.median(pts, axis=0).astype(np.float32)
            out[visible, q] = center[None, :]
            stabilize[q] = True

    return out, {
        "static_2d_stabilize_applied": True,
        "static_2d_stabilize_max_extent_px": float(max_extent_px),
        "static_2d_stabilize_max_step_px": float(max_step_px),
        "static_2d_stabilize_min_visible_frames": int(min_visible_frames),
        "static_2d_stabilized_tracks": int(stabilize.sum()),
        "static_2d_stabilize_extent_p50_p90_max": [
            float(np.percentile(extents, 50)) if extents else 0.0,
            float(np.percentile(extents, 90)) if extents else 0.0,
            float(max(extents)) if extents else 0.0,
        ],
        "static_2d_stabilize_step_p50_p90_max": [
            float(np.percentile(max_steps, 50)) if max_steps else 0.0,
            float(np.percentile(max_steps, 90)) if max_steps else 0.0,
            float(max(max_steps)) if max_steps else 0.0,
        ],
    }


def _stabilize_low_motion_static_runs(
    *,
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: np.ndarray,
    track_is_dynamic: np.ndarray,
    max_extent_px: float,
    max_step_px: float,
    min_visible_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stabilize local low-motion static runs without freezing moving-camera spans."""
    out = np.asarray(trajs_2d, dtype=np.float32).copy()
    if max_extent_px < 0 or max_step_px < 0 or min_visible_frames <= 1:
        return out, {"static_2d_run_stabilize_applied": False}

    visible_all = (
        np.asarray(valids, dtype=bool)
        & np.asarray(visibs, dtype=bool)
        & np.isfinite(out).all(axis=-1)
    )
    static_mask = ~np.asarray(track_is_dynamic, dtype=bool)
    stabilized_runs = 0
    stabilized_entries = 0
    run_extents: list[float] = []

    def finish_run(q: int, run: list[int]) -> None:
        nonlocal stabilized_runs, stabilized_entries
        if len(run) < int(min_visible_frames):
            return
        pts = out[np.asarray(run, dtype=np.int64), q]
        extent = float(np.linalg.norm(np.ptp(pts, axis=0)))
        if extent > float(max_extent_px):
            return
        center = np.median(pts, axis=0).astype(np.float32)
        out[np.asarray(run, dtype=np.int64), q] = center[None, :]
        stabilized_runs += 1
        stabilized_entries += len(run)
        run_extents.append(extent)

    for q in np.flatnonzero(static_mask):
        frames = np.flatnonzero(visible_all[:, q])
        if len(frames) < int(min_visible_frames):
            continue
        run = [int(frames[0])]
        for frame in frames[1:]:
            frame_i = int(frame)
            prev = run[-1]
            consecutive = frame_i == prev + 1
            step = float(np.linalg.norm(out[frame_i, q] - out[prev, q])) if consecutive else np.inf
            if consecutive and step <= float(max_step_px):
                run.append(frame_i)
            else:
                finish_run(q, run)
                run = [frame_i]
        finish_run(q, run)

    return out, {
        "static_2d_run_stabilize_applied": True,
        "static_2d_run_stabilize_max_extent_px": float(max_extent_px),
        "static_2d_run_stabilize_max_step_px": float(max_step_px),
        "static_2d_run_stabilize_min_visible_frames": int(min_visible_frames),
        "static_2d_run_stabilized_runs": int(stabilized_runs),
        "static_2d_run_stabilized_entries": int(stabilized_entries),
        "static_2d_run_stabilize_extent_p50_p90_max": [
            float(np.percentile(run_extents, 50)) if run_extents else 0.0,
            float(np.percentile(run_extents, 90)) if run_extents else 0.0,
            float(max(run_extents)) if run_extents else 0.0,
        ],
    }


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_theta = (float(np.trace(rotation)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def _find_low_camera_motion_segments(
    *,
    poses: np.ndarray,
    max_trans_step_m: float,
    max_rot_step_deg: float,
    min_segment_frames: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    t_total = int(poses.shape[0])
    if t_total == 0 or max_trans_step_m < 0 or max_rot_step_deg < 0 or min_segment_frames <= 1:
        return [], {"static_2d_still_camera_segments_applied": False}

    centers = poses[:, :3, 3]
    trans_steps = np.linalg.norm(centers[1:] - centers[:-1], axis=1) if t_total > 1 else np.zeros(0)
    rot_steps = np.asarray(
        [_rotation_angle_deg(poses[t, :3, :3].T @ poses[t + 1, :3, :3]) for t in range(t_total - 1)],
        dtype=np.float32,
    )
    still_edges = (trans_steps <= float(max_trans_step_m)) & (rot_steps <= float(max_rot_step_deg))

    segments: list[tuple[int, int]] = []
    start = 0
    for edge_idx, still in enumerate(still_edges):
        if bool(still):
            continue
        end = int(edge_idx)
        if (end - start + 1) >= int(min_segment_frames):
            segments.append((start, end))
        start = int(edge_idx + 1)
    end = t_total - 1
    if (end - start + 1) >= int(min_segment_frames):
        segments.append((start, end))

    lengths = [end - start + 1 for start, end in segments]
    return segments, {
        "static_2d_still_camera_segments_applied": True,
        "static_2d_still_camera_max_trans_step_m": float(max_trans_step_m),
        "static_2d_still_camera_max_rot_step_deg": float(max_rot_step_deg),
        "static_2d_still_camera_min_segment_frames": int(min_segment_frames),
        "static_2d_still_camera_num_segments": int(len(segments)),
        "static_2d_still_camera_segment_lengths_p50_p90_max": [
            float(np.percentile(lengths, 50)) if lengths else 0.0,
            float(np.percentile(lengths, 90)) if lengths else 0.0,
            int(max(lengths)) if lengths else 0,
        ],
        "static_2d_still_camera_trans_step_p50_p95_max": [
            float(np.percentile(trans_steps, 50)) if len(trans_steps) else 0.0,
            float(np.percentile(trans_steps, 95)) if len(trans_steps) else 0.0,
            float(np.max(trans_steps)) if len(trans_steps) else 0.0,
        ],
        "static_2d_still_camera_rot_step_deg_p50_p95_max": [
            float(np.percentile(rot_steps, 50)) if len(rot_steps) else 0.0,
            float(np.percentile(rot_steps, 95)) if len(rot_steps) else 0.0,
            float(np.max(rot_steps)) if len(rot_steps) else 0.0,
        ],
    }


def _stabilize_static_tracks_in_camera_still_segments(
    *,
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: np.ndarray,
    track_is_dynamic: np.ndarray,
    segments: list[tuple[int, int]],
    max_extent_px: float,
    max_step_px: float,
    min_visible_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.asarray(trajs_2d, dtype=np.float32).copy()
    if max_extent_px < 0 or max_step_px < 0 or min_visible_frames <= 1 or not segments:
        return out, {
            "static_2d_still_camera_stabilize_applied": False,
            "static_2d_still_camera_stabilized_runs": 0,
            "static_2d_still_camera_stabilized_entries": 0,
        }

    visible_all = (
        np.asarray(valids, dtype=bool)
        & np.asarray(visibs, dtype=bool)
        & np.isfinite(out).all(axis=-1)
    )
    static_mask = ~np.asarray(track_is_dynamic, dtype=bool)
    stabilized_runs = 0
    stabilized_entries = 0
    run_extents: list[float] = []

    def finish_run(q: int, run: list[int]) -> None:
        nonlocal stabilized_runs, stabilized_entries
        if len(run) < int(min_visible_frames):
            return
        run_idx = np.asarray(run, dtype=np.int64)
        pts = out[run_idx, q]
        extent = float(np.linalg.norm(np.ptp(pts, axis=0)))
        if extent > float(max_extent_px):
            return
        center = np.median(pts, axis=0).astype(np.float32)
        out[run_idx, q] = center[None, :]
        stabilized_runs += 1
        stabilized_entries += int(len(run_idx))
        run_extents.append(extent)

    for start, end in segments:
        seg_slice = slice(int(start), int(end) + 1)
        for q in np.flatnonzero(static_mask):
            frames = np.flatnonzero(visible_all[seg_slice, q]) + int(start)
            if len(frames) < int(min_visible_frames):
                continue
            run = [int(frames[0])]
            for frame in frames[1:]:
                frame_i = int(frame)
                prev = run[-1]
                consecutive = frame_i == prev + 1
                step = float(np.linalg.norm(out[frame_i, q] - out[prev, q])) if consecutive else np.inf
                if consecutive and step <= float(max_step_px):
                    run.append(frame_i)
                else:
                    finish_run(q, run)
                    run = [frame_i]
            finish_run(q, run)

    return out, {
        "static_2d_still_camera_stabilize_applied": True,
        "static_2d_still_camera_stabilize_max_extent_px": float(max_extent_px),
        "static_2d_still_camera_stabilize_max_step_px": float(max_step_px),
        "static_2d_still_camera_stabilize_min_visible_frames": int(min_visible_frames),
        "static_2d_still_camera_stabilized_runs": int(stabilized_runs),
        "static_2d_still_camera_stabilized_entries": int(stabilized_entries),
        "static_2d_still_camera_stabilize_extent_p50_p90_max": [
            float(np.percentile(run_extents, 50)) if run_extents else 0.0,
            float(np.percentile(run_extents, 90)) if run_extents else 0.0,
            float(max(run_extents)) if run_extents else 0.0,
        ],
    }


def _box_center_world_and_uv(
    *,
    seq_dir: Path,
    camera_name: int,
    frame_id: int,
    box: np.ndarray,
    intrinsic: np.ndarray,
    v2c_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.load(seq_dir / f"{frame_id:05d}_{camera_name}.npz")["pose"].reshape(4, 4).astype(np.float64)
    center_h = np.asarray([box[0], box[1], box[2], 1.0], dtype=np.float64)
    center_world = (pose @ center_h)[:3].astype(np.float32)
    extrinsic = (v2c_std @ np.linalg.inv(pose)).astype(np.float32)
    uv, _ = _project_points(center_world[None], extrinsic, intrinsic)
    return center_world, uv[0].astype(np.float32)


def _find_low_motion_dynamic_object_segments(
    *,
    seq_dir: Path,
    camera_name: int,
    track_is_dynamic: np.ndarray,
    track_object_ids: np.ndarray,
    camera_segments: list[tuple[int, int]],
    t_total: int,
    intrinsic: np.ndarray,
    v2c_std: np.ndarray,
    max_world_extent_m: float,
    max_center_extent_px: float,
    max_center_step_px: float,
    min_segment_frames: int,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, Any]]:
    if (
        max_world_extent_m < 0
        or max_center_extent_px < 0
        or max_center_step_px < 0
        or min_segment_frames <= 1
        or not camera_segments
        or not np.asarray(track_is_dynamic, dtype=bool).any()
    ):
        return {}, {"dynamic_still_object_segments_applied": False}

    rows_by_frame, _ = _load_waymo_dynamic_label_rows(seq_dir)
    if not rows_by_frame:
        return {}, {
            "dynamic_still_object_segments_applied": True,
            "dynamic_still_object_labels_found": False,
        }

    label_by_frame_object: dict[tuple[int, str], np.ndarray] = {}
    for frame_id, rows in rows_by_frame.items():
        for object_id, box in rows:
            label_by_frame_object[(int(frame_id), str(object_id))] = np.asarray(box, dtype=np.float32)

    object_ids = _decode_track_object_ids(track_object_ids, int(len(track_is_dynamic)))
    dynamic_ids = sorted({str(x) for x in object_ids[np.asarray(track_is_dynamic, dtype=bool)] if str(x)})
    segments_by_object: dict[str, list[tuple[int, int]]] = {}
    rows: list[dict[str, Any]] = []

    for object_id in dynamic_ids:
        for start, end in camera_segments:
            frame_ids = [f for f in range(int(start), int(end) + 1) if (f, object_id) in label_by_frame_object]
            if len(frame_ids) < int(min_segment_frames):
                continue
            centers_world = []
            centers_uv = []
            for frame_id in frame_ids:
                center_world, center_uv = _box_center_world_and_uv(
                    seq_dir=seq_dir,
                    camera_name=camera_name,
                    frame_id=frame_id,
                    box=label_by_frame_object[(frame_id, object_id)],
                    intrinsic=intrinsic,
                    v2c_std=v2c_std,
                )
                centers_world.append(center_world)
                centers_uv.append(center_uv)
            centers_world_arr = np.stack(centers_world, axis=0).astype(np.float32)
            centers_uv_arr = np.stack(centers_uv, axis=0).astype(np.float32)
            finite_uv = np.isfinite(centers_uv_arr).all(axis=1)
            if int(finite_uv.sum()) < int(min_segment_frames):
                continue
            uv_good = centers_uv_arr[finite_uv]
            world_good = centers_world_arr[finite_uv]
            world_extent = float(np.linalg.norm(np.ptp(world_good, axis=0)))
            center_extent = float(np.linalg.norm(np.ptp(uv_good, axis=0)))
            center_steps = np.linalg.norm(np.diff(uv_good, axis=0), axis=1)
            max_center_step = float(np.max(center_steps)) if len(center_steps) else 0.0
            keep = (
                world_extent <= float(max_world_extent_m)
                and center_extent <= float(max_center_extent_px)
                and max_center_step <= float(max_center_step_px)
            )
            rows.append(
                {
                    "object_id": object_id,
                    "segment": [int(start), int(end)],
                    "label_frames": int(len(frame_ids)),
                    "world_extent_m": world_extent,
                    "center_extent_px": center_extent,
                    "max_center_step_px": max_center_step,
                    "low_motion": bool(keep),
                }
            )
            if keep:
                segments_by_object.setdefault(object_id, []).append((int(start), int(end)))

    return segments_by_object, {
        "dynamic_still_object_segments_applied": True,
        "dynamic_still_object_labels_found": True,
        "dynamic_still_object_max_world_extent_m": float(max_world_extent_m),
        "dynamic_still_object_max_center_extent_px": float(max_center_extent_px),
        "dynamic_still_object_max_center_step_px": float(max_center_step_px),
        "dynamic_still_object_min_segment_frames": int(min_segment_frames),
        "dynamic_still_object_num_segments": int(sum(len(v) for v in segments_by_object.values())),
        "dynamic_still_object_num_objects": int(len(segments_by_object)),
        "dynamic_still_object_rows": rows[:200],
    }


def _stabilize_dynamic_tracks_in_object_segments(
    *,
    trajs_2d: np.ndarray,
    trajs_3d_world: np.ndarray,
    valids: np.ndarray,
    visibs: np.ndarray,
    base_valids: np.ndarray,
    in_bounds_all: np.ndarray,
    occluded: np.ndarray,
    track_is_dynamic: np.ndarray,
    track_object_ids: np.ndarray,
    object_segments: dict[str, list[tuple[int, int]]],
    max_track_extent_px: float,
    max_track_step_px: float,
    max_track_extent_3d_m: float,
    min_visible_frames: int,
    fill_visibility: bool,
    fill_min_visible_frames: int,
    fill_min_visible_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    out_2d = np.asarray(trajs_2d, dtype=np.float32).copy()
    out_3d = np.asarray(trajs_3d_world, dtype=np.float32).copy()
    out_valids = np.asarray(valids, dtype=bool).copy()
    out_visibs = np.asarray(visibs, dtype=bool).copy()
    if not object_segments or not np.asarray(track_is_dynamic, dtype=bool).any():
        return out_2d, out_3d, out_valids, out_visibs, {
            "dynamic_still_track_postprocess_applied": False,
        }

    object_ids = _decode_track_object_ids(track_object_ids, int(len(track_is_dynamic)))
    dynamic = np.asarray(track_is_dynamic, dtype=bool)
    stabilized_runs = 0
    stabilized_entries = 0
    filled_visible_entries = 0
    filled_occluded_entries = 0
    extents_2d: list[float] = []
    extents_3d: list[float] = []

    for object_id, segments in object_segments.items():
        idx = np.flatnonzero(dynamic & (object_ids == str(object_id)))
        if len(idx) == 0:
            continue
        for start, end in segments:
            seg = np.arange(int(start), int(end) + 1, dtype=np.int64)
            for q in idx:
                defined = (
                    np.asarray(base_valids[seg, q], dtype=bool)
                    & np.asarray(in_bounds_all[seg, q], dtype=bool)
                    & np.isfinite(out_2d[seg, q]).all(axis=-1)
                    & np.isfinite(out_3d[seg, q]).all(axis=-1)
                )
                if int(defined.sum()) < int(min_visible_frames):
                    continue

                visible_before = out_valids[seg, q] & out_visibs[seg, q] & defined
                visible_count = int(visible_before.sum())
                visible_fraction = visible_count / max(float(defined.sum()), 1.0)

                defined_frames = seg[defined]
                pts2 = out_2d[defined_frames, q]
                pts3 = out_3d[defined_frames, q]
                extent2 = float(np.linalg.norm(np.ptp(pts2, axis=0)))
                steps2 = np.linalg.norm(np.diff(pts2, axis=0), axis=1)
                max_step2 = float(np.max(steps2)) if len(steps2) else 0.0
                extent3 = float(np.linalg.norm(np.ptp(pts3, axis=0)))
                extents_2d.append(extent2)
                extents_3d.append(extent3)

                can_stabilize_2d = (
                    max_track_extent_px >= 0
                    and max_track_step_px >= 0
                    and extent2 <= float(max_track_extent_px)
                    and max_step2 <= float(max_track_step_px)
                )
                can_stabilize_3d = max_track_extent_3d_m < 0 or extent3 <= float(max_track_extent_3d_m)

                if can_stabilize_2d and can_stabilize_3d:
                    out_2d[defined_frames, q] = np.median(pts2, axis=0).astype(np.float32)[None, :]
                    out_3d[defined_frames, q] = np.median(pts3, axis=0).astype(np.float32)[None, :]
                    stabilized_runs += 1
                    stabilized_entries += int(len(defined_frames))

                if (
                    fill_visibility
                    and visible_count >= int(fill_min_visible_frames)
                    and visible_fraction >= float(fill_min_visible_fraction)
                ):
                    confident_occluded = np.asarray(occluded[defined_frames, q], dtype=bool)
                    before_valid = out_valids[defined_frames, q].copy()
                    before_vis = out_visibs[defined_frames, q].copy()
                    out_valids[defined_frames, q] = True
                    out_visibs[defined_frames, q] = ~confident_occluded
                    filled_visible_entries += int((out_visibs[defined_frames, q] & ~before_vis).sum())
                    filled_occluded_entries += int((confident_occluded & ~before_valid).sum())

    out_visibs &= out_valids
    return out_2d, out_3d, out_valids, out_visibs, {
        "dynamic_still_track_postprocess_applied": True,
        "dynamic_still_track_stabilize_max_extent_px": float(max_track_extent_px),
        "dynamic_still_track_stabilize_max_step_px": float(max_track_step_px),
        "dynamic_still_track_stabilize_max_extent_3d_m": float(max_track_extent_3d_m),
        "dynamic_still_track_stabilize_min_visible_frames": int(min_visible_frames),
        "dynamic_still_track_stabilized_runs": int(stabilized_runs),
        "dynamic_still_track_stabilized_entries": int(stabilized_entries),
        "dynamic_still_track_filled_visible_entries": int(filled_visible_entries),
        "dynamic_still_track_filled_occluded_entries": int(filled_occluded_entries),
        "dynamic_still_track_extent2d_p50_p90_max": [
            float(np.percentile(extents_2d, 50)) if extents_2d else 0.0,
            float(np.percentile(extents_2d, 90)) if extents_2d else 0.0,
            float(max(extents_2d)) if extents_2d else 0.0,
        ],
        "dynamic_still_track_extent3d_p50_p90_max": [
            float(np.percentile(extents_3d, 50)) if extents_3d else 0.0,
            float(np.percentile(extents_3d, 90)) if extents_3d else 0.0,
            float(max(extents_3d)) if extents_3d else 0.0,
        ],
    }


def _filter_payload_tracks(payload: dict[str, Any], keep_mask: np.ndarray) -> dict[str, Any]:
    keep_mask = np.asarray(keep_mask, dtype=bool)
    out = dict(payload)
    for key, value in list(out.items()):
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[1] == keep_mask.shape[0]:
            out[key] = arr[:, keep_mask].copy()
        elif arr.ndim == 2 and arr.shape[1] == keep_mask.shape[0]:
            out[key] = arr[:, keep_mask].copy()
        elif arr.ndim == 1 and arr.shape[0] == keep_mask.shape[0]:
            out[key] = arr[keep_mask].copy()
    out["num_points"] = np.array(int(keep_mask.sum()), dtype=np.int32)
    return out


def _smooth_static_visibility_sequence(
    vis_q: np.ndarray,
    in_bounds_q: np.ndarray,
    *,
    gap_fill_max_len: int,
    visible_spike_max_len: int,
) -> np.ndarray:
    """Denoise one static-track visibility sequence without changing OOB frames.

    Two flicker patterns are common after geometric visibility recompute:
    1. short invisible gaps inside long visible runs
    2. short visible spikes inside long invisible runs
    Both are usually quantization noise, not meaningful supervision.
    """
    seq = np.asarray(vis_q, dtype=bool).copy()
    valid_q = np.asarray(in_bounds_q, dtype=bool)
    t_total = int(len(seq))

    if gap_fill_max_len > 0:
        i = 0
        while i < t_total:
            if seq[i] or not valid_q[i]:
                i += 1
                continue
            j = i
            while j < t_total and (not seq[j]) and valid_q[j]:
                j += 1
            left_ok = i > 0 and seq[i - 1]
            right_ok = j < t_total and seq[j]
            if left_ok and right_ok and (j - i) <= gap_fill_max_len:
                seq[i:j] = True
            i = j

    if visible_spike_max_len > 0:
        i = 0
        while i < t_total:
            if (not seq[i]) or not valid_q[i]:
                i += 1
                continue
            j = i
            while j < t_total and seq[j] and valid_q[j]:
                j += 1
            left_ok = i > 0 and valid_q[i - 1] and (not seq[i - 1])
            right_ok = j < t_total and valid_q[j] and (not seq[j])
            if left_ok and right_ok and (j - i) <= visible_spike_max_len:
                seq[i:j] = False
            i = j

    return seq


def _compute_static_reject_mask(
    *,
    track_is_dynamic: np.ndarray,
    in_bounds_all: np.ndarray,
    visibs: np.ndarray,
    max_vis_changes: int,
    min_inbounds_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return reject mask plus per-track diagnostics for static-track filtering."""
    in_bounds_counts = in_bounds_all.sum(axis=0).astype(np.int32)
    visible_counts = visibs.sum(axis=0).astype(np.int32)
    vis_changes = np.abs(visibs[1:].astype(np.int8) - visibs[:-1].astype(np.int8)).sum(axis=0)
    static_track_mask = ~np.asarray(track_is_dynamic, dtype=bool)
    static_inbounds_fraction = in_bounds_counts / max(float(visibs.shape[0]), 1.0)
    reject_static = np.zeros((visibs.shape[1],), dtype=bool)
    if max_vis_changes >= 0:
        reject_static = (
            static_track_mask
            & (static_inbounds_fraction >= float(min_inbounds_fraction))
            & (vis_changes > int(max_vis_changes))
        )
    return reject_static, in_bounds_counts, visible_counts, vis_changes


def _decode_track_object_ids(track_object_ids: np.ndarray, n_tracks: int) -> np.ndarray:
    raw = np.asarray(track_object_ids)
    if raw.shape != (n_tracks,):
        return np.asarray([""] * n_tracks, dtype=object)
    decoded = np.empty((n_tracks,), dtype=object)
    for i, value in enumerate(raw):
        if isinstance(value, bytes):
            decoded[i] = value.decode("utf-8", "ignore")
        else:
            decoded[i] = str(value)
    return decoded


def _compute_low_motion_dynamic_keep_mask(
    *,
    trajs_2d: np.ndarray,
    valids: np.ndarray,
    visibs: np.ndarray,
    track_is_dynamic: np.ndarray,
    track_object_ids: np.ndarray,
    max_center_extent_px: float,
    max_center_step_px: float,
    min_visible_frames: int,
    min_visible_points_per_frame: int,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    """Drop dynamic-category objects that remain effectively static for the full sequence.

    Waymo box-track metadata marks category-dynamic objects (vehicle / pedestrian /
    cyclist), but some objects remain parked or otherwise static through the clip.
    Those tracks can leak tiny box-pose jitter into training/visualization.  We
    detect them from object-center motion in image space and remove the entire
    object track group.
    """
    n_tracks = int(trajs_2d.shape[1])
    keep_mask = np.ones((n_tracks,), dtype=bool)
    if (
        max_center_extent_px < 0
        or max_center_step_px < 0
        or min_visible_frames <= 0
        or min_visible_points_per_frame <= 0
        or n_tracks == 0
        or not np.asarray(track_is_dynamic, dtype=bool).any()
    ):
        return keep_mask, [], np.asarray([], dtype=np.str_)

    object_ids = _decode_track_object_ids(track_object_ids, n_tracks)
    drop_object_ids: list[str] = []
    rows: list[dict[str, Any]] = []

    for object_id in sorted(set(object_ids[np.asarray(track_is_dynamic, dtype=bool)])):
        if not object_id:
            continue
        idx = np.flatnonzero(np.asarray(track_is_dynamic, dtype=bool) & (object_ids == object_id))
        if len(idx) == 0:
            continue

        centers: list[np.ndarray] = []
        visible_frames = 0
        for t in range(trajs_2d.shape[0]):
            visible = (
                np.asarray(valids[t, idx], dtype=bool)
                & np.asarray(visibs[t, idx], dtype=bool)
                & np.isfinite(trajs_2d[t, idx]).all(axis=-1)
            )
            pts = np.asarray(trajs_2d[t, idx][visible], dtype=np.float32)
            if len(pts) < min_visible_points_per_frame:
                continue
            centers.append(np.median(pts, axis=0))
            visible_frames += 1

        if len(centers) < max(2, min_visible_frames):
            rows.append(
                {
                    "object_id": str(object_id),
                    "num_tracks": int(len(idx)),
                    "visible_frames": int(visible_frames),
                    "center_extent_px": None,
                    "max_center_step_px": None,
                    "drop_low_motion": False,
                    "reason": "insufficient_visible_frames",
                }
            )
            continue

        centers_arr = np.stack(centers, axis=0).astype(np.float32)
        center_extent = float(np.linalg.norm(np.ptp(centers_arr, axis=0)))
        if len(centers_arr) > 1:
            center_steps = np.linalg.norm(np.diff(centers_arr, axis=0), axis=1)
            max_center_step = float(np.max(center_steps))
        else:
            max_center_step = 0.0

        drop_low_motion = (
            center_extent <= float(max_center_extent_px)
            and max_center_step <= float(max_center_step_px)
        )
        if drop_low_motion:
            keep_mask[idx] = False
            drop_object_ids.append(str(object_id))

        rows.append(
            {
                "object_id": str(object_id),
                "num_tracks": int(len(idx)),
                "visible_frames": int(visible_frames),
                "center_extent_px": center_extent,
                "max_center_step_px": max_center_step,
                "drop_low_motion": bool(drop_low_motion),
                "reason": "low_motion" if drop_low_motion else "keep_dynamic",
            }
        )

    return keep_mask, rows, np.asarray(drop_object_ids, dtype=np.str_)


def _guard_dynamic_visibility_with_input(
    *,
    new_vis: np.ndarray,
    old_vis: np.ndarray,
    track_is_dynamic: np.ndarray,
    preserve_input_vis: bool,
) -> np.ndarray:
    if not preserve_input_vis:
        return np.asarray(new_vis, dtype=bool)
    return np.asarray(new_vis, dtype=bool) & (~np.asarray(track_is_dynamic, dtype=bool) | np.asarray(old_vis, dtype=bool))


def _compute_dynamic_support_reject_mask(
    *,
    old_valids: np.ndarray,
    old_visibs: np.ndarray,
    track_is_dynamic: np.ndarray,
    min_visible_frames: int,
    min_visible_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject dynamic tracks that rarely matched LiDAR surface evidence.

    The original dynamic-box generation used depth consistency to mark whether a
    box-rigid point actually sat on an observed surface.  That signal is too
    sparse/noisy to use as final per-frame visibility, but it is useful as a
    track-level quality gate: points sampled from road/background inside a 3D
    box usually have little or no depth support over the clip.
    """
    valid = np.asarray(old_valids, dtype=bool)
    visible = np.asarray(old_visibs, dtype=bool) & valid
    dynamic = np.asarray(track_is_dynamic, dtype=bool)
    n_tracks = int(valid.shape[1]) if valid.ndim == 2 else int(dynamic.shape[0])
    reject = np.zeros((n_tracks,), dtype=bool)
    if n_tracks == 0 or not dynamic.any() or (min_visible_frames <= 0 and min_visible_fraction <= 0):
        return reject, {
            "dynamic_support_filter_applied": False,
            "dynamic_support_rejected_tracks": 0,
        }

    valid_counts = valid.sum(axis=0).astype(np.int32)
    visible_counts = visible.sum(axis=0).astype(np.int32)
    visible_fraction = visible_counts / np.maximum(valid_counts.astype(np.float32), 1.0)
    reject = dynamic & (
        (visible_counts < int(min_visible_frames))
        | (visible_fraction < float(min_visible_fraction))
    )
    dyn_frac = visible_fraction[dynamic]
    dyn_vis = visible_counts[dynamic]
    return reject, {
        "dynamic_support_filter_applied": True,
        "dynamic_support_min_visible_frames": int(min_visible_frames),
        "dynamic_support_min_visible_fraction": float(min_visible_fraction),
        "dynamic_support_rejected_tracks": int(reject.sum()),
        "dynamic_support_visible_fraction_p05_p25_p50": [
            float(np.percentile(dyn_frac, q)) if len(dyn_frac) else 0.0
            for q in (5, 25, 50)
        ],
        "dynamic_support_visible_frames_p05_p25_p50": [
            float(np.percentile(dyn_vis, q)) if len(dyn_vis) else 0.0
            for q in (5, 25, 50)
        ],
    }


def _compute_dynamic_box_ref_keep_mask(
    *,
    seq_dir: Path,
    trajs_3d_world: np.ndarray,
    old_valids: np.ndarray,
    old_visibs: np.ndarray,
    track_is_dynamic: np.ndarray,
    track_object_ids: np.ndarray,
    track_ref_frames: np.ndarray,
    camera_name: int,
    enabled: bool,
    max_outside_margin_m: float,
    min_bottom_clearance_m: float,
    require_ref_visible: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject dynamic box tracks whose reference point is likely not on the object.

    Dynamic Waymo tracks are generated by taking LiDAR points inside a 3D box and
    then moving those points rigidly with the box track.  Points sampled from the
    box margin or road surface can therefore look like they are stuck to a car.
    This gate verifies the reference-frame point against the original box.
    """
    n_tracks = int(trajs_3d_world.shape[1])
    keep = np.ones((n_tracks,), dtype=bool)
    dynamic = np.asarray(track_is_dynamic, dtype=bool)
    if not enabled or n_tracks == 0 or not dynamic.any():
        return keep, {
            "dynamic_box_ref_filter_applied": bool(enabled),
            "dynamic_box_ref_rejected_tracks": 0,
        }

    rows_by_frame, _ = _load_waymo_dynamic_label_rows(seq_dir)
    if not rows_by_frame:
        return keep, {
            "dynamic_box_ref_filter_applied": True,
            "dynamic_box_ref_labels_found": False,
            "dynamic_box_ref_rejected_tracks": 0,
        }

    label_by_frame_object: dict[tuple[int, str], np.ndarray] = {}
    for frame_id, rows in rows_by_frame.items():
        for object_id, box in rows:
            label_by_frame_object[(int(frame_id), str(object_id))] = np.asarray(box, dtype=np.float32)

    object_ids = _decode_track_object_ids(track_object_ids, n_tracks)
    ref_frames = np.asarray(track_ref_frames)
    if ref_frames.shape != (n_tracks,):
        ref_frames = np.full((n_tracks,), -1, dtype=np.int32)
    ref_frames = ref_frames.astype(np.int32, copy=False)
    old_valids_arr = np.asarray(old_valids, dtype=bool)
    old_visibs_arr = np.asarray(old_visibs, dtype=bool)
    t_total = int(old_valids_arr.shape[0])

    pose_cache: dict[int, np.ndarray] = {}
    reason_counts = {
        "invalid_ref": 0,
        "missing_label": 0,
        "missing_point": 0,
        "ref_invisible": 0,
        "outside_box": 0,
        "near_box_bottom": 0,
    }
    object_stats: dict[str, dict[str, Any]] = {}

    def add_object_stat(object_id: str, rejected: bool, max_outside: float | None, bottom_clearance: float | None) -> None:
        row = object_stats.setdefault(
            object_id,
            {
                "object_id": object_id,
                "tracks": 0,
                "rejected": 0,
                "max_outside_values": [],
                "bottom_clearance_values": [],
            },
        )
        row["tracks"] += 1
        if rejected:
            row["rejected"] += 1
        if max_outside is not None:
            row["max_outside_values"].append(float(max_outside))
        if bottom_clearance is not None:
            row["bottom_clearance_values"].append(float(bottom_clearance))

    for q in np.flatnonzero(dynamic):
        object_id = str(object_ids[q])
        ref_frame = int(ref_frames[q])
        rejected = False
        max_outside_value: float | None = None
        bottom_clearance_value: float | None = None

        if not object_id or ref_frame < 0 or ref_frame >= t_total:
            reason_counts["invalid_ref"] += 1
            rejected = True
        else:
            box = label_by_frame_object.get((ref_frame, object_id))
            if box is None:
                reason_counts["missing_label"] += 1
                rejected = True
            elif not old_valids_arr[ref_frame, q] or not np.isfinite(trajs_3d_world[ref_frame, q]).all():
                reason_counts["missing_point"] += 1
                rejected = True
            elif require_ref_visible and not old_visibs_arr[ref_frame, q]:
                reason_counts["ref_invisible"] += 1
                rejected = True
            else:
                pose = pose_cache.get(ref_frame)
                if pose is None:
                    npz_path = seq_dir / f"{ref_frame:05d}_{camera_name}.npz"
                    if npz_path.exists():
                        pose = np.load(npz_path)["pose"].reshape(4, 4).astype(np.float32)
                        pose_cache[ref_frame] = pose
                if pose is None:
                    reason_counts["invalid_ref"] += 1
                    rejected = True
                else:
                    w2v = np.linalg.inv(pose)
                    point_h = np.concatenate(
                        [np.asarray(trajs_3d_world[ref_frame, q], dtype=np.float32), np.ones((1,), dtype=np.float32)]
                    )
                    point_vehicle = (w2v @ point_h)[:3].astype(np.float32)
                    local = (point_vehicle - box[:3]) @ _rot_z(float(box[6]))
                    half = np.asarray(box[3:6], dtype=np.float32) * 0.5
                    excess = np.abs(local) - half
                    max_outside_value = float(np.max(excess))
                    bottom_clearance_value = float(local[2] + half[2])
                    if max_outside_value > float(max_outside_margin_m):
                        reason_counts["outside_box"] += 1
                        rejected = True
                    elif (
                        min_bottom_clearance_m >= 0
                        and bottom_clearance_value < float(min_bottom_clearance_m)
                    ):
                        reason_counts["near_box_bottom"] += 1
                        rejected = True

        if rejected:
            keep[q] = False
        add_object_stat(object_id, rejected, max_outside_value, bottom_clearance_value)

    object_rows: list[dict[str, Any]] = []
    for row in object_stats.values():
        outside_values = row.pop("max_outside_values")
        bottom_values = row.pop("bottom_clearance_values")
        row["rejected_fraction"] = float(row["rejected"]) / max(float(row["tracks"]), 1.0)
        row["max_outside_m_p50_p95_max"] = [
            float(np.percentile(outside_values, q)) if outside_values else 0.0
            for q in (50, 95, 100)
        ]
        row["bottom_clearance_m_p05_p50"] = [
            float(np.percentile(bottom_values, q)) if bottom_values else 0.0
            for q in (5, 50)
        ]
        object_rows.append(row)
    object_rows.sort(key=lambda item: (int(item["rejected"]), int(item["tracks"])), reverse=True)

    reject = ~keep
    return keep, {
        "dynamic_box_ref_filter_applied": True,
        "dynamic_box_ref_labels_found": True,
        "dynamic_box_ref_max_outside_margin_m": float(max_outside_margin_m),
        "dynamic_box_ref_min_bottom_clearance_m": float(min_bottom_clearance_m),
        "dynamic_box_ref_require_ref_visible": bool(require_ref_visible),
        "dynamic_box_ref_rejected_tracks": int(reject.sum()),
        "dynamic_box_ref_reason_counts": reason_counts,
        "dynamic_box_ref_object_rows": object_rows[:200],
    }


def recompute_visibility_for_payload(
    *,
    seq_dir: Path,
    payload: dict[str, Any],
    camera_name: int,
    visibility_mode: str,
    occlusion_margin_m: float,
    occlusion_margin_rel: float,
    local_radius: int,
    zbuffer_support_radius: int,
    surface_min_support: int,
    occlusion_min_support: int,
    uncertain_visibility_as_invalid: bool,
    static_gap_fill_max_len: int,
    static_visible_spike_max_len: int,
    static_filter_max_vis_changes: int,
    static_filter_min_inbounds_fraction: float,
    static_preserve_input_visibility: bool = False,
    dynamic_low_motion_max_center_extent_px: float,
    dynamic_low_motion_max_center_step_px: float,
    dynamic_low_motion_min_visible_frames: int,
    dynamic_low_motion_min_visible_points_per_frame: int,
    dynamic_preserve_input_vis: bool,
    dynamic_support_min_visible_frames: int,
    dynamic_support_min_visible_fraction: float,
    dynamic_box_ref_filter: bool = False,
    dynamic_box_ref_max_outside_margin_m: float = 0.05,
    dynamic_box_ref_min_bottom_clearance_m: float = 0.15,
    dynamic_box_ref_require_ref_visible: bool = True,
    static_dynamic_box_filter_mode: str = "off",
    static_dynamic_box_margin: float = 0.2,
    static_dynamic_box_moving_min_extent_m: float = 1.0,
    static_dynamic_box_moving_min_step_m: float = 0.05,
    static_dynamic_box_moving_min_frames: int = 5,
    static_dynamic_box_min_inside_frames: int = 1,
    static_2d_stabilize_max_extent_px: float = -1.0,
    static_2d_stabilize_max_step_px: float = -1.0,
    static_2d_stabilize_min_visible_frames: int = 30,
    static_2d_run_stabilize_max_extent_px: float = -1.0,
    static_2d_run_stabilize_max_step_px: float = -1.0,
    static_2d_run_stabilize_min_visible_frames: int = 12,
    static_2d_still_camera_max_trans_step_m: float = -1.0,
    static_2d_still_camera_max_rot_step_deg: float = -1.0,
    static_2d_still_camera_min_segment_frames: int = 12,
    static_2d_still_camera_stabilize_max_extent_px: float = -1.0,
    static_2d_still_camera_stabilize_max_step_px: float = -1.0,
    static_2d_still_camera_stabilize_min_visible_frames: int = 12,
    dynamic_still_object_max_world_extent_m: float = -1.0,
    dynamic_still_object_max_center_extent_px: float = -1.0,
    dynamic_still_object_max_center_step_px: float = -1.0,
    dynamic_still_object_min_segment_frames: int = 12,
    dynamic_still_track_stabilize_max_extent_px: float = -1.0,
    dynamic_still_track_stabilize_max_step_px: float = -1.0,
    dynamic_still_track_stabilize_max_extent_3d_m: float = -1.0,
    dynamic_still_track_stabilize_min_visible_frames: int = 12,
    dynamic_still_track_fill_visibility: bool = False,
    dynamic_still_track_fill_min_visible_frames: int = 6,
    dynamic_still_track_fill_min_visible_fraction: float = 0.1,
    defined_validity_mode: str = "visibility_supported",
) -> tuple[dict[str, Any], dict[str, Any]]:
    trajs_3d = np.asarray(payload["trajs_3d_world"], dtype=np.float32)
    old_valids = np.asarray(payload["valids"], dtype=bool)
    old_visibs = np.asarray(payload["visibs"], dtype=bool)
    t_total, n_tracks = old_valids.shape
    track_is_dynamic = np.asarray(payload.get("track_is_dynamic", np.zeros(n_tracks, dtype=bool)), dtype=bool)
    if track_is_dynamic.shape != (n_tracks,):
        track_is_dynamic = np.zeros(n_tracks, dtype=bool)
    track_object_ids = np.asarray(payload.get("track_object_ids", np.asarray([""] * n_tracks, dtype=np.str_)))
    track_ref_frames = np.asarray(payload.get("track_ref_frames", np.full(n_tracks, -1, dtype=np.int32)))

    dynamic_keep_mask, dynamic_motion_rows, dropped_dynamic_object_ids = _compute_low_motion_dynamic_keep_mask(
        trajs_2d=np.asarray(payload.get("trajs_2d", np.zeros((t_total, n_tracks, 2), dtype=np.float32)), dtype=np.float32),
        valids=old_valids,
        visibs=old_visibs,
        track_is_dynamic=track_is_dynamic,
        track_object_ids=track_object_ids,
        max_center_extent_px=dynamic_low_motion_max_center_extent_px,
        max_center_step_px=dynamic_low_motion_max_center_step_px,
        min_visible_frames=dynamic_low_motion_min_visible_frames,
        min_visible_points_per_frame=dynamic_low_motion_min_visible_points_per_frame,
    )
    reject_low_motion_dynamic = ~dynamic_keep_mask
    reject_low_support_dynamic, dynamic_support_stats = _compute_dynamic_support_reject_mask(
        old_valids=old_valids,
        old_visibs=old_visibs,
        track_is_dynamic=track_is_dynamic,
        min_visible_frames=dynamic_support_min_visible_frames,
        min_visible_fraction=dynamic_support_min_visible_fraction,
    )
    dynamic_box_ref_keep_mask, dynamic_box_ref_stats = _compute_dynamic_box_ref_keep_mask(
        seq_dir=seq_dir,
        trajs_3d_world=trajs_3d,
        old_valids=old_valids,
        old_visibs=old_visibs,
        track_is_dynamic=track_is_dynamic,
        track_object_ids=track_object_ids,
        track_ref_frames=track_ref_frames,
        camera_name=camera_name,
        enabled=dynamic_box_ref_filter,
        max_outside_margin_m=dynamic_box_ref_max_outside_margin_m,
        min_bottom_clearance_m=dynamic_box_ref_min_bottom_clearance_m,
        require_ref_visible=dynamic_box_ref_require_ref_visible,
    )
    reject_dynamic_box_ref = ~dynamic_box_ref_keep_mask

    new_valids = np.zeros_like(old_valids, dtype=bool)
    new_visibs = np.zeros_like(old_visibs, dtype=bool)
    occluded = np.zeros_like(old_visibs, dtype=bool)
    uncertain = np.zeros_like(old_visibs, dtype=bool)
    in_bounds_all = np.zeros_like(old_visibs, dtype=bool)
    poses = np.zeros((t_total, 4, 4), dtype=np.float64)

    intrinsic, v2c_std = _camera_calibration(seq_dir, camera_name)

    for t in range(t_total):
        npz = np.load(seq_dir / f"{t:05d}_{camera_name}.npz")
        v2w = npz["pose"].reshape(4, 4).astype(np.float64)
        poses[t] = v2w
        extrinsic = (v2c_std @ np.linalg.inv(v2w)).astype(np.float32)
        h, w = _frame_shape(seq_dir, t, camera_name)

        uv, z = _project_points(trajs_3d[t], extrinsic, intrinsic)
        finite = np.isfinite(uv).all(axis=1) & np.isfinite(trajs_3d[t]).all(axis=1) & np.isfinite(z)
        in_front = z > 1.0e-3
        defined = finite & in_front
        if track_is_dynamic.any():
            defined = defined & (~track_is_dynamic | old_valids[t])
        in_bounds = (
            defined
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        in_bounds_all[t] = in_bounds

        if visibility_mode == "raw_lidar_local":
            raw_depth = _build_depth_dense((h, w, 3), npz, v2c_std, dilation_radius=0)
            px = np.zeros(n_tracks, dtype=np.int32)
            py = np.zeros(n_tracks, dtype=np.int32)
            px[defined] = np.clip(np.round(uv[defined, 0]).astype(np.int32), 0, w - 1)
            py[defined] = np.clip(np.round(uv[defined, 1]).astype(np.int32), 0, h - 1)
            margin = np.maximum(float(occlusion_margin_m), float(occlusion_margin_rel) * z)
            is_surface_supported = np.zeros(n_tracks, dtype=bool)
            is_occluded = np.zeros(n_tracks, dtype=bool)
            candidate_idx = np.flatnonzero(defined & in_bounds)
            if len(candidate_idx):
                supported, occluded_supported = _local_depth_support_masks(
                    raw_depth=raw_depth,
                    px=px[candidate_idx],
                    py=py[candidate_idx],
                    z=z[candidate_idx],
                    margin=margin[candidate_idx],
                    radius=local_radius,
                    surface_min_support=surface_min_support,
                    occlusion_min_support=occlusion_min_support,
                )
                is_surface_supported[candidate_idx] = supported
                is_occluded[candidate_idx] = occluded_supported & ~supported
            if uncertain_visibility_as_invalid:
                new_valid = is_surface_supported | is_occluded
                new_vis = is_surface_supported
                is_uncertain = defined & in_bounds & ~(is_surface_supported | is_occluded)
            else:
                new_valid = defined
                new_vis = defined & in_bounds & ~is_occluded
                is_uncertain = np.zeros(n_tracks, dtype=bool)
        elif visibility_mode in {"zbuffer", "zbuffer_lidar_hybrid"}:
            margin = np.maximum(float(occlusion_margin_m), float(occlusion_margin_rel) * z)
            valid_px, zbuffer_occluded = _zbuffer_occlusion_mask(
                defined=defined,
                uv=uv,
                z=z,
                image_h=h,
                image_w=w,
                margin=margin,
                support_radius=zbuffer_support_radius,
                min_support=occlusion_min_support,
            )
            is_occluded = zbuffer_occluded.copy()
            if visibility_mode == "zbuffer_lidar_hybrid":
                raw_depth = _build_depth_dense((h, w, 3), npz, v2c_std, dilation_radius=0)
                px = np.zeros(n_tracks, dtype=np.int32)
                py = np.zeros(n_tracks, dtype=np.int32)
                px[valid_px] = np.clip(np.round(uv[valid_px, 0]).astype(np.int32), 0, w - 1)
                py[valid_px] = np.clip(np.round(uv[valid_px, 1]).astype(np.int32), 0, h - 1)
                candidate_idx = np.flatnonzero(valid_px)
                if len(candidate_idx):
                    supported, occluded_supported = _local_depth_support_masks(
                        raw_depth=raw_depth,
                        px=px[candidate_idx],
                        py=py[candidate_idx],
                        z=z[candidate_idx],
                        margin=margin[candidate_idx],
                        radius=local_radius,
                        surface_min_support=surface_min_support,
                        occlusion_min_support=occlusion_min_support,
                    )
                    raw_lidar_occluded = occluded_supported & ~supported
                    is_occluded[candidate_idx] |= raw_lidar_occluded
            new_vis = valid_px & ~is_occluded
            new_valid = valid_px
            is_uncertain = np.zeros(n_tracks, dtype=bool)
        else:
            raise ValueError(f"Unknown visibility_mode={visibility_mode!r}")

        if defined_validity_mode == "visibility_supported":
            valid_for_supervision = new_valid
        elif defined_validity_mode == "projected":
            valid_for_supervision = valid_px if visibility_mode in {"zbuffer", "zbuffer_lidar_hybrid"} else (defined & in_bounds)
        elif defined_validity_mode == "defined":
            valid_for_supervision = defined
        else:
            raise ValueError(f"Unknown defined_validity_mode={defined_validity_mode!r}")
        new_valids[t] = valid_for_supervision
        new_visibs[t] = _guard_dynamic_visibility_with_input(
            new_vis=new_vis,
            old_vis=old_visibs[t],
            track_is_dynamic=track_is_dynamic,
            preserve_input_vis=dynamic_preserve_input_vis,
        )
        occluded[t] = is_occluded
        uncertain[t] = is_uncertain

    if static_preserve_input_visibility:
        static_track_mask = ~track_is_dynamic
        new_valids[:, static_track_mask] = old_valids[:, static_track_mask]
        new_visibs[:, static_track_mask] = old_visibs[:, static_track_mask] & old_valids[:, static_track_mask]

    if static_gap_fill_max_len > 0 or static_visible_spike_max_len > 0:
        static_track_mask = ~track_is_dynamic
        for q in np.flatnonzero(static_track_mask):
            new_visibs[:, q] = _smooth_static_visibility_sequence(
                new_visibs[:, q],
                new_valids[:, q],
                gap_fill_max_len=static_gap_fill_max_len,
                visible_spike_max_len=static_visible_spike_max_len,
            )
        new_visibs &= new_valids

    reject_static, inbounds_counts, visible_counts, vis_changes = _compute_static_reject_mask(
        track_is_dynamic=track_is_dynamic,
        in_bounds_all=in_bounds_all,
        visibs=new_visibs,
        max_vis_changes=static_filter_max_vis_changes,
        min_inbounds_fraction=static_filter_min_inbounds_fraction,
    )
    reject_static_dynamic_box, static_dynamic_box_stats = _compute_static_dynamic_box_reject_mask(
        seq_dir=seq_dir,
        trajs_3d_world=trajs_3d,
        track_is_dynamic=track_is_dynamic,
        camera_name=camera_name,
        filter_mode=static_dynamic_box_filter_mode,
        box_margin=static_dynamic_box_margin,
        moving_min_extent_m=static_dynamic_box_moving_min_extent_m,
        moving_min_step_m=static_dynamic_box_moving_min_step_m,
        moving_min_frames=static_dynamic_box_moving_min_frames,
        min_inside_frames=static_dynamic_box_min_inside_frames,
    )
    static_track_mask = ~track_is_dynamic
    static_inbounds_fraction = inbounds_counts / max(float(t_total), 1.0)

    payload = dict(payload)
    payload["valids"] = new_valids
    payload["visibs"] = new_visibs
    stabilized_trajs_2d, static_2d_stabilize_stats = _stabilize_low_motion_static_tracks(
        trajs_2d=np.asarray(payload.get("trajs_2d", np.zeros((t_total, n_tracks, 2), dtype=np.float32)), dtype=np.float32),
        valids=new_valids,
        visibs=new_visibs,
        track_is_dynamic=track_is_dynamic,
        max_extent_px=static_2d_stabilize_max_extent_px,
        max_step_px=static_2d_stabilize_max_step_px,
        min_visible_frames=static_2d_stabilize_min_visible_frames,
    )
    stabilized_trajs_2d, static_2d_run_stabilize_stats = _stabilize_low_motion_static_runs(
        trajs_2d=stabilized_trajs_2d,
        valids=new_valids,
        visibs=new_visibs,
        track_is_dynamic=track_is_dynamic,
        max_extent_px=static_2d_run_stabilize_max_extent_px,
        max_step_px=static_2d_run_stabilize_max_step_px,
        min_visible_frames=static_2d_run_stabilize_min_visible_frames,
    )
    still_segments, static_2d_still_camera_segment_stats = _find_low_camera_motion_segments(
        poses=poses,
        max_trans_step_m=static_2d_still_camera_max_trans_step_m,
        max_rot_step_deg=static_2d_still_camera_max_rot_step_deg,
        min_segment_frames=static_2d_still_camera_min_segment_frames,
    )
    (
        stabilized_trajs_2d,
        static_2d_still_camera_stabilize_stats,
    ) = _stabilize_static_tracks_in_camera_still_segments(
        trajs_2d=stabilized_trajs_2d,
        valids=new_valids,
        visibs=new_visibs,
        track_is_dynamic=track_is_dynamic,
        segments=still_segments,
        max_extent_px=static_2d_still_camera_stabilize_max_extent_px,
        max_step_px=static_2d_still_camera_stabilize_max_step_px,
        min_visible_frames=static_2d_still_camera_stabilize_min_visible_frames,
    )
    dynamic_object_segments, dynamic_object_segment_stats = _find_low_motion_dynamic_object_segments(
        seq_dir=seq_dir,
        camera_name=camera_name,
        track_is_dynamic=track_is_dynamic,
        track_object_ids=track_object_ids,
        camera_segments=still_segments,
        t_total=t_total,
        intrinsic=intrinsic,
        v2c_std=v2c_std,
        max_world_extent_m=dynamic_still_object_max_world_extent_m,
        max_center_extent_px=dynamic_still_object_max_center_extent_px,
        max_center_step_px=dynamic_still_object_max_center_step_px,
        min_segment_frames=dynamic_still_object_min_segment_frames,
    )
    (
        stabilized_trajs_2d,
        stabilized_trajs_3d,
        new_valids,
        new_visibs,
        dynamic_still_track_stats,
    ) = _stabilize_dynamic_tracks_in_object_segments(
        trajs_2d=stabilized_trajs_2d,
        trajs_3d_world=trajs_3d,
        valids=new_valids,
        visibs=new_visibs,
        base_valids=old_valids,
        in_bounds_all=in_bounds_all,
        occluded=occluded,
        track_is_dynamic=track_is_dynamic,
        track_object_ids=track_object_ids,
        object_segments=dynamic_object_segments,
        max_track_extent_px=dynamic_still_track_stabilize_max_extent_px,
        max_track_step_px=dynamic_still_track_stabilize_max_step_px,
        max_track_extent_3d_m=dynamic_still_track_stabilize_max_extent_3d_m,
        min_visible_frames=dynamic_still_track_stabilize_min_visible_frames,
        fill_visibility=dynamic_still_track_fill_visibility,
        fill_min_visible_frames=dynamic_still_track_fill_min_visible_frames,
        fill_min_visible_fraction=dynamic_still_track_fill_min_visible_fraction,
    )
    payload["valids"] = new_valids
    payload["visibs"] = new_visibs
    payload["trajs_2d"] = stabilized_trajs_2d
    payload["trajs_3d_world"] = stabilized_trajs_3d
    payload["visibility_rule"] = np.asarray(
        f"mode={visibility_mode},radius={local_radius},margin_m={occlusion_margin_m},"
        f"margin_rel={occlusion_margin_rel},zbuffer_support_radius={zbuffer_support_radius},"
        f"surface_min_support={surface_min_support},"
        f"occlusion_min_support={occlusion_min_support},"
        f"uncertain_invalid={int(uncertain_visibility_as_invalid)},"
        f"static_preserve_input_visibility={int(static_preserve_input_visibility)},"
        f"gap_fill={static_gap_fill_max_len},"
        f"visible_spike={static_visible_spike_max_len},"
        f"static_max_vis_changes={static_filter_max_vis_changes},"
        f"static_min_inbounds_fraction={static_filter_min_inbounds_fraction},"
        f"dynamic_low_motion_extent={dynamic_low_motion_max_center_extent_px},"
        f"dynamic_low_motion_step={dynamic_low_motion_max_center_step_px},"
        f"dynamic_low_motion_min_frames={dynamic_low_motion_min_visible_frames},"
        f"dynamic_low_motion_min_pts={dynamic_low_motion_min_visible_points_per_frame},"
        f"dynamic_preserve_input_vis={int(dynamic_preserve_input_vis)},"
        f"dynamic_support_min_visible_frames={dynamic_support_min_visible_frames},"
        f"dynamic_support_min_visible_fraction={dynamic_support_min_visible_fraction},"
        f"dynamic_box_ref_filter={int(dynamic_box_ref_filter)},"
        f"dynamic_box_ref_outside_margin={dynamic_box_ref_max_outside_margin_m},"
        f"dynamic_box_ref_bottom_clearance={dynamic_box_ref_min_bottom_clearance_m},"
        f"static_dynamic_box_filter={static_dynamic_box_filter_mode},"
        f"static_dynamic_box_margin={static_dynamic_box_margin},"
        f"static_2d_stabilize_extent={static_2d_stabilize_max_extent_px},"
        f"static_2d_stabilize_step={static_2d_stabilize_max_step_px},"
        f"static_2d_run_stabilize_extent={static_2d_run_stabilize_max_extent_px},"
        f"static_2d_run_stabilize_step={static_2d_run_stabilize_max_step_px},"
        f"static_2d_still_camera_trans={static_2d_still_camera_max_trans_step_m},"
        f"static_2d_still_camera_rot={static_2d_still_camera_max_rot_step_deg},"
        f"static_2d_still_camera_stabilize_extent={static_2d_still_camera_stabilize_max_extent_px},"
        f"static_2d_still_camera_stabilize_step={static_2d_still_camera_stabilize_max_step_px},"
        f"dynamic_still_object_world_extent={dynamic_still_object_max_world_extent_m},"
        f"dynamic_still_object_center_extent={dynamic_still_object_max_center_extent_px},"
        f"dynamic_still_track_extent={dynamic_still_track_stabilize_max_extent_px},"
        f"dynamic_still_track_fill_visibility={int(dynamic_still_track_fill_visibility)},"
        f"defined_validity_mode={defined_validity_mode}",
        dtype=np.str_,
    )

    before_tracks = int(n_tracks)
    rejected_tracks = int(reject_static.sum())
    rejected_dynamic_tracks = int(reject_low_motion_dynamic.sum())
    rejected_low_support_dynamic_tracks = int(reject_low_support_dynamic.sum())
    rejected_dynamic_box_ref_tracks = int(reject_dynamic_box_ref.sum())
    rejected_static_dynamic_box_tracks = int(reject_static_dynamic_box.sum())
    combined_reject = (
        reject_static
        | reject_low_motion_dynamic
        | reject_low_support_dynamic
        | reject_dynamic_box_ref
        | reject_static_dynamic_box
    )
    if combined_reject.any():
        keep_mask = ~combined_reject
        payload = _filter_payload_tracks(payload, keep_mask)
        payload["track_quality_filter_applied"] = np.array(True)
        payload["track_quality_rejected"] = np.array(rejected_tracks, dtype=np.int32)
        payload["static_dynamic_box_filter_rejected"] = np.array(
            rejected_static_dynamic_box_tracks,
            dtype=np.int32,
        )
        payload["dynamic_low_motion_filter_applied"] = np.array(True)
        payload["dynamic_low_motion_rejected_tracks"] = np.array(rejected_dynamic_tracks, dtype=np.int32)
        payload["dynamic_low_motion_rejected_object_ids"] = dropped_dynamic_object_ids
        payload["dynamic_support_filter_applied"] = np.array(dynamic_support_stats["dynamic_support_filter_applied"])
        payload["dynamic_support_rejected_tracks"] = np.array(rejected_low_support_dynamic_tracks, dtype=np.int32)
        payload["dynamic_box_ref_filter_applied"] = np.array(dynamic_box_ref_stats["dynamic_box_ref_filter_applied"])
        payload["dynamic_box_ref_rejected_tracks"] = np.array(rejected_dynamic_box_ref_tracks, dtype=np.int32)
        payload["track_quality_before_points"] = np.array(before_tracks, dtype=np.int32)
        payload["track_quality_after_points"] = np.array(int(keep_mask.sum()), dtype=np.int32)
    else:
        payload["track_quality_filter_applied"] = np.array(static_filter_max_vis_changes >= 0)
        payload["track_quality_rejected"] = np.array(0, dtype=np.int32)
        payload["static_dynamic_box_filter_rejected"] = np.array(0, dtype=np.int32)
        payload["dynamic_low_motion_filter_applied"] = np.array(dynamic_low_motion_max_center_extent_px >= 0)
        payload["dynamic_low_motion_rejected_tracks"] = np.array(0, dtype=np.int32)
        payload["dynamic_low_motion_rejected_object_ids"] = np.asarray([], dtype=np.str_)
        payload["dynamic_support_filter_applied"] = np.array(dynamic_support_stats["dynamic_support_filter_applied"])
        payload["dynamic_support_rejected_tracks"] = np.array(0, dtype=np.int32)
        payload["dynamic_box_ref_filter_applied"] = np.array(dynamic_box_ref_stats["dynamic_box_ref_filter_applied"])
        payload["dynamic_box_ref_rejected_tracks"] = np.array(0, dtype=np.int32)
        payload["track_quality_before_points"] = np.array(before_tracks, dtype=np.int32)
        payload["track_quality_after_points"] = np.array(before_tracks, dtype=np.int32)

    old_changes = np.abs(old_visibs[1:].astype(np.int8) - old_visibs[:-1].astype(np.int8)).sum(axis=0)
    static_changes = vis_changes[static_track_mask]
    kept_dynamic_after = int(np.asarray(payload.get("track_is_dynamic", np.zeros(0, dtype=bool)), dtype=bool).sum())
    stats = {
        "frames": int(t_total),
        "tracks_before": before_tracks,
        "tracks_after": int(payload["num_points"]),
        "dynamic_tracks_before": int(track_is_dynamic.sum()),
        "dynamic_tracks_after": kept_dynamic_after,
        "old_valid_entries": int(old_valids.sum()),
        "old_visible_entries": int(old_visibs.sum()),
        "new_valid_entries": int(new_valids.sum()),
        "new_visible_entries": int(new_visibs.sum()),
        "new_occluded_entries": int(occluded.sum()),
        "new_uncertain_invalid_entries": int(uncertain.sum()),
        "uncertain_visibility_as_invalid": bool(uncertain_visibility_as_invalid),
        "defined_validity_mode": str(defined_validity_mode),
        "static_preserve_input_visibility": bool(static_preserve_input_visibility),
        "surface_min_support": int(surface_min_support),
        "occlusion_min_support": int(occlusion_min_support),
        "static_gap_fill_max_len": int(static_gap_fill_max_len),
        "static_visible_spike_max_len": int(static_visible_spike_max_len),
        "static_filter_max_vis_changes": int(static_filter_max_vis_changes),
        "static_filter_min_inbounds_fraction": float(static_filter_min_inbounds_fraction),
        "filtered_static_tracks": rejected_tracks,
        "filtered_static_dynamic_box_tracks": rejected_static_dynamic_box_tracks,
        "filtered_low_motion_dynamic_tracks": rejected_dynamic_tracks,
        "filtered_low_support_dynamic_tracks": rejected_low_support_dynamic_tracks,
        "filtered_dynamic_box_ref_tracks": rejected_dynamic_box_ref_tracks,
        "filtered_low_motion_dynamic_objects": int(len(dropped_dynamic_object_ids)),
        "dynamic_low_motion_max_center_extent_px": float(dynamic_low_motion_max_center_extent_px),
        "dynamic_low_motion_max_center_step_px": float(dynamic_low_motion_max_center_step_px),
        "dynamic_low_motion_min_visible_frames": int(dynamic_low_motion_min_visible_frames),
        "dynamic_low_motion_min_visible_points_per_frame": int(dynamic_low_motion_min_visible_points_per_frame),
        "dynamic_low_motion_rows": dynamic_motion_rows,
        "dynamic_preserve_input_vis": bool(dynamic_preserve_input_vis),
        **dynamic_support_stats,
        **dynamic_box_ref_stats,
        **static_dynamic_box_stats,
        **static_2d_stabilize_stats,
        **static_2d_run_stabilize_stats,
        **static_2d_still_camera_segment_stats,
        **static_2d_still_camera_stabilize_stats,
        **dynamic_object_segment_stats,
        **dynamic_still_track_stats,
        "static_inbounds_fraction_p50_p90": [
            float(np.percentile(static_inbounds_fraction[static_track_mask], 50)) if static_track_mask.any() else 0.0,
            float(np.percentile(static_inbounds_fraction[static_track_mask], 90)) if static_track_mask.any() else 0.0,
        ],
        "old_vis_changes_min_p50_p90_max": [
            int(old_changes.min()) if len(old_changes) else 0,
            float(np.percentile(old_changes, 50)) if len(old_changes) else 0.0,
            float(np.percentile(old_changes, 90)) if len(old_changes) else 0.0,
            int(old_changes.max()) if len(old_changes) else 0,
        ],
        "new_vis_changes_static_min_p50_p90_max": [
            int(static_changes.min()) if len(static_changes) else 0,
            float(np.percentile(static_changes, 50)) if len(static_changes) else 0.0,
            float(np.percentile(static_changes, 90)) if len(static_changes) else 0.0,
            int(static_changes.max()) if len(static_changes) else 0,
        ],
        "static_visible_fraction_p50_p90": [
            float(np.percentile((visible_counts / max(float(t_total), 1.0))[static_track_mask], 50)) if static_track_mask.any() else 0.0,
            float(np.percentile((visible_counts / max(float(t_total), 1.0))[static_track_mask], 90)) if static_track_mask.any() else 0.0,
        ],
    }
    return payload, stats


def process_sequence(
    *,
    root: Path,
    input_precompute_root: Path,
    output_precompute_root: Path,
    sequence: str,
    camera_name: int,
    visibility_mode: str,
    occlusion_radius: int,
    zbuffer_support_radius: int,
    occlusion_margin_m: float,
    occlusion_margin_rel: float,
    surface_min_support: int,
    occlusion_min_support: int,
    uncertain_visibility_as_invalid: bool,
    static_gap_fill_max_len: int,
    static_visible_spike_max_len: int,
    static_filter_max_vis_changes: int,
    static_filter_min_inbounds_fraction: float,
    static_preserve_input_visibility: bool,
    dynamic_low_motion_max_center_extent_px: float,
    dynamic_low_motion_max_center_step_px: float,
    dynamic_low_motion_min_visible_frames: int,
    dynamic_low_motion_min_visible_points_per_frame: int,
    dynamic_preserve_input_vis: bool,
    dynamic_support_min_visible_frames: int,
    dynamic_support_min_visible_fraction: float,
    dynamic_box_ref_filter: bool,
    dynamic_box_ref_max_outside_margin_m: float,
    dynamic_box_ref_min_bottom_clearance_m: float,
    dynamic_box_ref_require_ref_visible: bool,
    static_dynamic_box_filter_mode: str,
    static_dynamic_box_margin: float,
    static_dynamic_box_moving_min_extent_m: float,
    static_dynamic_box_moving_min_step_m: float,
    static_dynamic_box_moving_min_frames: int,
    static_dynamic_box_min_inside_frames: int,
    static_2d_stabilize_max_extent_px: float,
    static_2d_stabilize_max_step_px: float,
    static_2d_stabilize_min_visible_frames: int,
    static_2d_run_stabilize_max_extent_px: float,
    static_2d_run_stabilize_max_step_px: float,
    static_2d_run_stabilize_min_visible_frames: int,
    static_2d_still_camera_max_trans_step_m: float,
    static_2d_still_camera_max_rot_step_deg: float,
    static_2d_still_camera_min_segment_frames: int,
    static_2d_still_camera_stabilize_max_extent_px: float,
    static_2d_still_camera_stabilize_max_step_px: float,
    static_2d_still_camera_stabilize_min_visible_frames: int,
    dynamic_still_object_max_world_extent_m: float,
    dynamic_still_object_max_center_extent_px: float,
    dynamic_still_object_max_center_step_px: float,
    dynamic_still_object_min_segment_frames: int,
    dynamic_still_track_stabilize_max_extent_px: float,
    dynamic_still_track_stabilize_max_step_px: float,
    dynamic_still_track_stabilize_max_extent_3d_m: float,
    dynamic_still_track_stabilize_min_visible_frames: int,
    dynamic_still_track_fill_visibility: bool,
    dynamic_still_track_fill_min_visible_frames: int,
    dynamic_still_track_fill_min_visible_fraction: float,
    defined_validity_mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    src_h5 = input_precompute_root / sequence / "precomputed.h5"
    if not src_h5.exists():
        raise FileNotFoundError(src_h5)
    dst_h5 = output_precompute_root / sequence / "precomputed.h5"
    if dst_h5.exists() and not overwrite and dst_h5 != src_h5:
        return {"sequence": sequence, "output": str(dst_h5), "skipped": True}

    payload = _load_payload(src_h5)
    fixed_payload, stats = recompute_visibility_for_payload(
        seq_dir=root / sequence,
        payload=payload,
        camera_name=camera_name,
        visibility_mode=visibility_mode,
        local_radius=occlusion_radius,
        zbuffer_support_radius=zbuffer_support_radius,
        occlusion_margin_m=occlusion_margin_m,
        occlusion_margin_rel=occlusion_margin_rel,
        surface_min_support=surface_min_support,
        occlusion_min_support=occlusion_min_support,
        uncertain_visibility_as_invalid=uncertain_visibility_as_invalid,
        static_gap_fill_max_len=static_gap_fill_max_len,
        static_visible_spike_max_len=static_visible_spike_max_len,
        static_filter_max_vis_changes=static_filter_max_vis_changes,
        static_filter_min_inbounds_fraction=static_filter_min_inbounds_fraction,
        static_preserve_input_visibility=static_preserve_input_visibility,
        dynamic_low_motion_max_center_extent_px=dynamic_low_motion_max_center_extent_px,
        dynamic_low_motion_max_center_step_px=dynamic_low_motion_max_center_step_px,
        dynamic_low_motion_min_visible_frames=dynamic_low_motion_min_visible_frames,
        dynamic_low_motion_min_visible_points_per_frame=dynamic_low_motion_min_visible_points_per_frame,
        dynamic_preserve_input_vis=dynamic_preserve_input_vis,
        dynamic_support_min_visible_frames=dynamic_support_min_visible_frames,
        dynamic_support_min_visible_fraction=dynamic_support_min_visible_fraction,
        dynamic_box_ref_filter=dynamic_box_ref_filter,
        dynamic_box_ref_max_outside_margin_m=dynamic_box_ref_max_outside_margin_m,
        dynamic_box_ref_min_bottom_clearance_m=dynamic_box_ref_min_bottom_clearance_m,
        dynamic_box_ref_require_ref_visible=dynamic_box_ref_require_ref_visible,
        static_dynamic_box_filter_mode=static_dynamic_box_filter_mode,
        static_dynamic_box_margin=static_dynamic_box_margin,
        static_dynamic_box_moving_min_extent_m=static_dynamic_box_moving_min_extent_m,
        static_dynamic_box_moving_min_step_m=static_dynamic_box_moving_min_step_m,
        static_dynamic_box_moving_min_frames=static_dynamic_box_moving_min_frames,
        static_dynamic_box_min_inside_frames=static_dynamic_box_min_inside_frames,
        static_2d_stabilize_max_extent_px=static_2d_stabilize_max_extent_px,
        static_2d_stabilize_max_step_px=static_2d_stabilize_max_step_px,
        static_2d_stabilize_min_visible_frames=static_2d_stabilize_min_visible_frames,
        static_2d_run_stabilize_max_extent_px=static_2d_run_stabilize_max_extent_px,
        static_2d_run_stabilize_max_step_px=static_2d_run_stabilize_max_step_px,
        static_2d_run_stabilize_min_visible_frames=static_2d_run_stabilize_min_visible_frames,
        static_2d_still_camera_max_trans_step_m=static_2d_still_camera_max_trans_step_m,
        static_2d_still_camera_max_rot_step_deg=static_2d_still_camera_max_rot_step_deg,
        static_2d_still_camera_min_segment_frames=static_2d_still_camera_min_segment_frames,
        static_2d_still_camera_stabilize_max_extent_px=static_2d_still_camera_stabilize_max_extent_px,
        static_2d_still_camera_stabilize_max_step_px=static_2d_still_camera_stabilize_max_step_px,
        static_2d_still_camera_stabilize_min_visible_frames=static_2d_still_camera_stabilize_min_visible_frames,
        dynamic_still_object_max_world_extent_m=dynamic_still_object_max_world_extent_m,
        dynamic_still_object_max_center_extent_px=dynamic_still_object_max_center_extent_px,
        dynamic_still_object_max_center_step_px=dynamic_still_object_max_center_step_px,
        dynamic_still_object_min_segment_frames=dynamic_still_object_min_segment_frames,
        dynamic_still_track_stabilize_max_extent_px=dynamic_still_track_stabilize_max_extent_px,
        dynamic_still_track_stabilize_max_step_px=dynamic_still_track_stabilize_max_step_px,
        dynamic_still_track_stabilize_max_extent_3d_m=dynamic_still_track_stabilize_max_extent_3d_m,
        dynamic_still_track_stabilize_min_visible_frames=dynamic_still_track_stabilize_min_visible_frames,
        dynamic_still_track_fill_visibility=dynamic_still_track_fill_visibility,
        dynamic_still_track_fill_min_visible_frames=dynamic_still_track_fill_min_visible_frames,
        dynamic_still_track_fill_min_visible_fraction=dynamic_still_track_fill_min_visible_fraction,
        defined_validity_mode=defined_validity_mode,
    )
    _save_payload(dst_h5, fixed_payload)
    report = {"sequence": sequence, "input": str(src_h5), "output": str(dst_h5), **stats}
    (dst_h5.parent / "visibility_recompute_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _process_sequence_worker(kwargs: dict[str, Any]) -> dict[str, Any]:
    return process_sequence(**kwargs)


def parse_sequences(args: argparse.Namespace) -> list[str]:
    sequences = list(args.sequence)
    if args.summary:
        data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        sequences.extend(str(item["sequence"]) for item in data)
    return sorted(set(sequences))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data2/d4rt/datasets/waymo_pre")
    parser.add_argument("--input-precompute-root", default="/data2/d4rt/datasets/waymo_pre")
    parser.add_argument("--output-precompute-root", required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--summary", default=None)
    parser.add_argument("--camera-name", type=int, default=1)
    parser.add_argument(
        "--visibility-mode",
        choices=("zbuffer", "raw_lidar_local", "zbuffer_lidar_hybrid"),
        default="raw_lidar_local",
        help=(
            "Use raw Waymo LiDAR depth by default. The track-only z-buffer mode is useful for debugging "
            "but misses occluders that are not represented by precomputed tracks. The hybrid mode keeps "
            "z-buffer visibility and only uses raw LiDAR to add confident nearer-surface occlusions."
        ),
    )
    parser.add_argument("--occlusion-radius", type=int, default=3)
    parser.add_argument(
        "--zbuffer-support-radius",
        type=int,
        default=1,
        help="Neighborhood radius for track z-buffer occlusion support. Kept tighter than raw LiDAR support.",
    )
    parser.add_argument("--occlusion-margin-m", type=float, default=0.5)
    parser.add_argument("--occlusion-margin-rel", type=float, default=0.02)
    parser.add_argument(
        "--surface-min-support",
        type=int,
        default=1,
        help="Minimum raw LiDAR samples in the local window matching the point depth to mark it visible.",
    )
    parser.add_argument(
        "--occlusion-min-support",
        type=int,
        default=3,
        help="Minimum closer raw LiDAR samples in the local window required to mark an ambiguous point occluded.",
    )
    parser.add_argument(
        "--uncertain-visibility-as-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For raw LiDAR visibility, mark a projected point valid only when LiDAR locally supports "
            "the same surface depth or a closer occluder. Ambiguous in-bounds points become invalid."
        ),
    )
    parser.add_argument("--static-gap-fill-max-len", type=int, default=2)
    parser.add_argument(
        "--static-visible-spike-max-len",
        type=int,
        default=1,
        help="Suppress short visible spikes for static tracks that are bracketed by invisible in-bounds frames.",
    )
    parser.add_argument(
        "--static-filter-max-vis-changes",
        type=int,
        default=3,
        help="Reject static tracks seen in-view for long spans but still toggling visibility too often. Use -1 to disable.",
    )
    parser.add_argument(
        "--static-filter-min-inbounds-fraction",
        type=float,
        default=0.8,
        help="Only reject static tracks when they stay in-bounds for at least this fraction of the clip.",
    )
    parser.add_argument(
        "--static-preserve-input-visibility",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep original static valids/visibs after raw-LiDAR recompute. Useful for sparse-LiDAR Waymo "
            "where raw visibility can erase most static tracks."
        ),
    )
    parser.add_argument(
        "--dynamic-low-motion-max-center-extent-px",
        type=float,
        default=5.0,
        help="Drop dynamic-category objects whose 2D center stays within this total extent over the whole sequence. Use -1 to disable.",
    )
    parser.add_argument(
        "--dynamic-low-motion-max-center-step-px",
        type=float,
        default=1.0,
        help="Drop dynamic-category objects only if their max frame-to-frame center step also stays below this threshold. Use -1 to disable.",
    )
    parser.add_argument(
        "--dynamic-low-motion-min-visible-frames",
        type=int,
        default=30,
        help="Require at least this many visible frames before dropping a low-motion dynamic object.",
    )
    parser.add_argument(
        "--dynamic-low-motion-min-visible-points-per-frame",
        type=int,
        default=10,
        help="Minimum visible points in a frame to count that frame toward dynamic object motion estimation.",
    )
    parser.add_argument(
        "--dynamic-preserve-input-vis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Never make a dynamic track visible if it was invisible in the input precomputed cache.",
    )
    parser.add_argument(
        "--dynamic-support-min-visible-frames",
        type=int,
        default=5,
        help=(
            "Reject dynamic tracks with fewer than this many input depth-supported frames. "
            "The input support is used only as a track-quality gate, not final per-frame visibility. "
            "Use 0 to disable this part of the filter."
        ),
    )
    parser.add_argument(
        "--dynamic-support-min-visible-fraction",
        type=float,
        default=0.05,
        help=(
            "Reject dynamic tracks whose input depth-supported fraction is below this threshold. "
            "Use 0 to disable this part of the filter."
        ),
    )
    parser.add_argument(
        "--dynamic-box-ref-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reject dynamic box tracks whose reference-frame point is outside the true box core, "
            "near the bottom/road surface, or not visible in the original dynamic precompute."
        ),
    )
    parser.add_argument("--dynamic-box-ref-max-outside-margin-m", type=float, default=0.05)
    parser.add_argument("--dynamic-box-ref-min-bottom-clearance-m", type=float, default=0.15)
    parser.add_argument(
        "--dynamic-box-ref-require-ref-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--static-dynamic-box-filter-mode",
        choices=("off", "moving", "all"),
        default="moving",
        help=(
            "Reject static tracks that fall inside Waymo dynamic-category 3D boxes. "
            "'all' removes vehicle/ped/cyclist box interiors; 'moving' only removes boxes "
            "whose label track moves in world coordinates; 'off' disables this filter."
        ),
    )
    parser.add_argument("--static-dynamic-box-margin", type=float, default=0.2)
    parser.add_argument("--static-dynamic-box-moving-min-extent-m", type=float, default=1.0)
    parser.add_argument("--static-dynamic-box-moving-min-step-m", type=float, default=0.05)
    parser.add_argument("--static-dynamic-box-moving-min-frames", type=int, default=5)
    parser.add_argument("--static-dynamic-box-min-inside-frames", type=int, default=1)
    parser.add_argument("--static-2d-stabilize-max-extent-px", type=float, default=-1.0)
    parser.add_argument("--static-2d-stabilize-max-step-px", type=float, default=-1.0)
    parser.add_argument("--static-2d-stabilize-min-visible-frames", type=int, default=30)
    parser.add_argument("--static-2d-run-stabilize-max-extent-px", type=float, default=0.25)
    parser.add_argument("--static-2d-run-stabilize-max-step-px", type=float, default=0.15)
    parser.add_argument("--static-2d-run-stabilize-min-visible-frames", type=int, default=12)
    parser.add_argument("--static-2d-still-camera-max-trans-step-m", type=float, default=-1.0)
    parser.add_argument("--static-2d-still-camera-max-rot-step-deg", type=float, default=-1.0)
    parser.add_argument("--static-2d-still-camera-min-segment-frames", type=int, default=12)
    parser.add_argument("--static-2d-still-camera-stabilize-max-extent-px", type=float, default=-1.0)
    parser.add_argument("--static-2d-still-camera-stabilize-max-step-px", type=float, default=-1.0)
    parser.add_argument("--static-2d-still-camera-stabilize-min-visible-frames", type=int, default=12)
    parser.add_argument("--dynamic-still-object-max-world-extent-m", type=float, default=-1.0)
    parser.add_argument("--dynamic-still-object-max-center-extent-px", type=float, default=-1.0)
    parser.add_argument("--dynamic-still-object-max-center-step-px", type=float, default=-1.0)
    parser.add_argument("--dynamic-still-object-min-segment-frames", type=int, default=12)
    parser.add_argument("--dynamic-still-track-stabilize-max-extent-px", type=float, default=-1.0)
    parser.add_argument("--dynamic-still-track-stabilize-max-step-px", type=float, default=-1.0)
    parser.add_argument("--dynamic-still-track-stabilize-max-extent-3d-m", type=float, default=-1.0)
    parser.add_argument("--dynamic-still-track-stabilize-min-visible-frames", type=int, default=12)
    parser.add_argument(
        "--dynamic-still-track-fill-visibility",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For low-motion dynamic objects in still-camera segments, fill per-frame valid/visible "
            "from object support unless there is confident occlusion."
        ),
    )
    parser.add_argument("--dynamic-still-track-fill-min-visible-frames", type=int, default=6)
    parser.add_argument("--dynamic-still-track-fill-min-visible-fraction", type=float, default=0.1)
    parser.add_argument(
        "--defined-validity-mode",
        choices=("visibility_supported", "projected", "defined"),
        default="visibility_supported",
        help=(
            "Controls saved valids. 'visibility_supported' keeps previous behavior. "
            "'projected' marks every projected/in-bounds defined track entry valid, so non-visible "
            "entries supervise visibility as negatives via visibs=False. 'defined' additionally keeps "
            "out-of-view but camera-front entries valid, so they supervise visibility as invisible too."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sequences = parse_sequences(args)
    if not sequences:
        raise SystemExit("No sequences provided.")

    root = Path(args.root)
    input_root = Path(args.input_precompute_root)
    output_root = Path(args.output_precompute_root)
    jobs = [
        dict(
            root=root,
            input_precompute_root=input_root,
            output_precompute_root=output_root,
            sequence=seq,
            camera_name=args.camera_name,
            visibility_mode=args.visibility_mode,
            occlusion_radius=args.occlusion_radius,
            zbuffer_support_radius=args.zbuffer_support_radius,
            occlusion_margin_m=args.occlusion_margin_m,
            occlusion_margin_rel=args.occlusion_margin_rel,
            surface_min_support=args.surface_min_support,
            occlusion_min_support=args.occlusion_min_support,
            uncertain_visibility_as_invalid=args.uncertain_visibility_as_invalid,
            static_gap_fill_max_len=args.static_gap_fill_max_len,
            static_visible_spike_max_len=args.static_visible_spike_max_len,
            static_filter_max_vis_changes=args.static_filter_max_vis_changes,
            static_filter_min_inbounds_fraction=args.static_filter_min_inbounds_fraction,
            static_preserve_input_visibility=args.static_preserve_input_visibility,
            dynamic_low_motion_max_center_extent_px=args.dynamic_low_motion_max_center_extent_px,
            dynamic_low_motion_max_center_step_px=args.dynamic_low_motion_max_center_step_px,
            dynamic_low_motion_min_visible_frames=args.dynamic_low_motion_min_visible_frames,
            dynamic_low_motion_min_visible_points_per_frame=args.dynamic_low_motion_min_visible_points_per_frame,
            dynamic_preserve_input_vis=args.dynamic_preserve_input_vis,
            dynamic_support_min_visible_frames=args.dynamic_support_min_visible_frames,
            dynamic_support_min_visible_fraction=args.dynamic_support_min_visible_fraction,
            dynamic_box_ref_filter=args.dynamic_box_ref_filter,
            dynamic_box_ref_max_outside_margin_m=args.dynamic_box_ref_max_outside_margin_m,
            dynamic_box_ref_min_bottom_clearance_m=args.dynamic_box_ref_min_bottom_clearance_m,
            dynamic_box_ref_require_ref_visible=args.dynamic_box_ref_require_ref_visible,
            static_dynamic_box_filter_mode=args.static_dynamic_box_filter_mode,
            static_dynamic_box_margin=args.static_dynamic_box_margin,
            static_dynamic_box_moving_min_extent_m=args.static_dynamic_box_moving_min_extent_m,
            static_dynamic_box_moving_min_step_m=args.static_dynamic_box_moving_min_step_m,
            static_dynamic_box_moving_min_frames=args.static_dynamic_box_moving_min_frames,
            static_dynamic_box_min_inside_frames=args.static_dynamic_box_min_inside_frames,
            static_2d_stabilize_max_extent_px=args.static_2d_stabilize_max_extent_px,
            static_2d_stabilize_max_step_px=args.static_2d_stabilize_max_step_px,
            static_2d_stabilize_min_visible_frames=args.static_2d_stabilize_min_visible_frames,
            static_2d_run_stabilize_max_extent_px=args.static_2d_run_stabilize_max_extent_px,
            static_2d_run_stabilize_max_step_px=args.static_2d_run_stabilize_max_step_px,
            static_2d_run_stabilize_min_visible_frames=args.static_2d_run_stabilize_min_visible_frames,
            static_2d_still_camera_max_trans_step_m=args.static_2d_still_camera_max_trans_step_m,
            static_2d_still_camera_max_rot_step_deg=args.static_2d_still_camera_max_rot_step_deg,
            static_2d_still_camera_min_segment_frames=args.static_2d_still_camera_min_segment_frames,
            static_2d_still_camera_stabilize_max_extent_px=args.static_2d_still_camera_stabilize_max_extent_px,
            static_2d_still_camera_stabilize_max_step_px=args.static_2d_still_camera_stabilize_max_step_px,
            static_2d_still_camera_stabilize_min_visible_frames=args.static_2d_still_camera_stabilize_min_visible_frames,
            dynamic_still_object_max_world_extent_m=args.dynamic_still_object_max_world_extent_m,
            dynamic_still_object_max_center_extent_px=args.dynamic_still_object_max_center_extent_px,
            dynamic_still_object_max_center_step_px=args.dynamic_still_object_max_center_step_px,
            dynamic_still_object_min_segment_frames=args.dynamic_still_object_min_segment_frames,
            dynamic_still_track_stabilize_max_extent_px=args.dynamic_still_track_stabilize_max_extent_px,
            dynamic_still_track_stabilize_max_step_px=args.dynamic_still_track_stabilize_max_step_px,
            dynamic_still_track_stabilize_max_extent_3d_m=args.dynamic_still_track_stabilize_max_extent_3d_m,
            dynamic_still_track_stabilize_min_visible_frames=args.dynamic_still_track_stabilize_min_visible_frames,
            dynamic_still_track_fill_visibility=args.dynamic_still_track_fill_visibility,
            dynamic_still_track_fill_min_visible_frames=args.dynamic_still_track_fill_min_visible_frames,
            dynamic_still_track_fill_min_visible_fraction=args.dynamic_still_track_fill_min_visible_fraction,
            defined_validity_mode=args.defined_validity_mode,
            overwrite=args.overwrite,
        )
        for seq in sequences
    ]

    reports = []
    if args.workers <= 1:
        for job in jobs:
            reports.append(process_sequence(**job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process_sequence_worker, job) for job in jobs]
            for fut in as_completed(futures):
                reports.append(fut.result())
        reports.sort(key=lambda item: str(item.get("sequence", "")))

    summary_path = output_root / "waymo_visibility_recompute_summary.json"
    summary_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "reports": reports}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
