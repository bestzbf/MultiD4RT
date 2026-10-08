"""
Compute 2D/3D point tracks from depth maps and camera poses (strategy A).

Strategy A (static scene):
  - Sample N source points from the reference frame (frame 0) depth map.
  - Unproject them to 3D world coordinates.
  - Project into every other frame using that frame's pose.
  - Mark a point valid if it projects inside the image AND the depth
    at that pixel is consistent with the projected depth.

This is exact for static scenes; dynamic objects will accumulate invalid
marks, which is acceptable because the query builder already handles
sparse valid masks.

Usage:
    from computer.depth_to_tracks import compute_tracks

    result = compute_tracks(depths, intrinsics, extrinsics)
    # result['trajs_2d']       [T,N,2]
    # result['trajs_3d_world'] [T,N,3]
    # result['valids']         [T,N]
    # result['visibs']         [T,N]
"""

from __future__ import annotations

import cv2
import numpy as np
from collections.abc import Sequence


TRACK_SEMANTICS_VERSION = 3
SOURCE_DEPTH_SAMPLING_MODE_IDS = {
    "uniform": 0,
    "log_balanced": 1,
    "linear_balanced": 2,
}


# --------------------------------------------------------------------------- #
# Low-level geometry helpers
# --------------------------------------------------------------------------- #

def _unproject(
    uv: np.ndarray,      # [N,2]  (x=col, y=row) float32
    depth: np.ndarray,   # [N]    float32
    K: np.ndarray,       # [3,3]
) -> np.ndarray:         # [N,3]  camera-space XYZ
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = (uv[:, 0] - cx) * depth / fx
    Y = (uv[:, 1] - cy) * depth / fy
    Z = depth
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def _project(
    pts_cam: np.ndarray,   # [N,3]  camera-space XYZ
    K: np.ndarray,         # [3,3]
) -> tuple[np.ndarray, np.ndarray]:   # uv [N,2], z [N]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = pts_cam[:, 2]
    # avoid division by zero for points behind camera
    safe_z = np.where(z > 0, z, 1.0)
    u = pts_cam[:, 0] / safe_z * fx + cx
    v = pts_cam[:, 1] / safe_z * fy + cy
    return np.stack([u, v], axis=-1).astype(np.float32), z.astype(np.float32)


def _c2w_from_w2c(E: np.ndarray) -> np.ndarray:
    """Invert a w2c [4,4] matrix analytically (numerically stable)."""
    R = E[:3, :3]   # [3,3]
    t = E[:3, 3]    # [3]
    R_inv = R.T
    t_inv = -R.T @ t
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_inv
    c2w[:3, 3]  = t_inv
    return c2w


def _select_ref_frames(
    valid_counts: np.ndarray,
    num_segments: int,
    strategy: str,
) -> list[int]:
    """Split the clip into segments and choose one reference frame per segment."""
    T = len(valid_counts)
    refs: list[int] = []
    num_segments = max(int(num_segments), 1)
    for seg in range(num_segments):
        start = seg * T // num_segments
        end = (seg + 1) * T // num_segments
        if start >= end:
            continue
        if strategy == "first":
            seg_valid = np.where(valid_counts[start:end] > 0)[0]
            ref = start + int(seg_valid[0]) if len(seg_valid) > 0 else start
        elif strategy == "max_valid":
            ref = start + int(np.argmax(valid_counts[start:end]))
        elif strategy == "middle":
            ref = (start + end) // 2
        else:
            raise ValueError(f"Unknown ref_frame_strategy: {strategy!r}")
        refs.append(int(ref))
    return refs


def _depth_boundary_mask(depth: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    """Bool mask [H,W]: pixels at depth discontinuities."""
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    valid = (d > 0) & np.isfinite(d)
    if not valid.any():
        return np.zeros_like(d, dtype=bool)
    vals = mag[valid]
    if vals.size == 0 or float(vals.max()) <= 0:
        return np.zeros_like(d, dtype=bool)
    thr = max(float(np.percentile(vals, percentile)), 1e-6)
    return valid & (mag >= thr)


def _normalise_source_depth_sampling(mode: str) -> str:
    mode = str(mode or "uniform").strip().lower()
    aliases = {
        "balanced": "log_balanced",
        "log": "log_balanced",
        "linear": "linear_balanced",
    }
    mode = aliases.get(mode, mode)
    if mode not in SOURCE_DEPTH_SAMPLING_MODE_IDS:
        raise ValueError(
            f"Unknown source_depth_sampling={mode!r}; expected one of "
            f"{sorted(SOURCE_DEPTH_SAMPLING_MODE_IDS)}"
        )
    return mode


def _normalise_depth_bin_edges(edges: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if edges is None:
        return None
    arr = np.asarray(list(edges), dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    arr = np.unique(arr)
    if arr.size < 2:
        return None
    if np.any(np.diff(arr) <= 0):
        raise ValueError(f"source_depth_bin_edges must be strictly increasing, got {arr.tolist()}")
    return arr


def _normalise_depth_bin_weights(weights: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if weights is None:
        return None
    arr = np.asarray(list(weights), dtype=np.float32).reshape(-1)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    if arr.size == 0 or float(arr.sum()) <= 0.0:
        return None
    return arr


def _dynamic_depth_bin_edges(
    depth_values: np.ndarray,
    depth_max: float,
    source_depth_sampling: str,
    source_depth_num_bins: int,
) -> np.ndarray | None:
    vals = np.asarray(depth_values, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 1e-6)]
    if vals.size == 0:
        return None
    num_bins = max(1, int(source_depth_num_bins))
    if num_bins <= 1:
        return None
    lo = float(vals.min())
    hi = min(float(depth_max), float(vals.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    if source_depth_sampling == "linear_balanced":
        return np.linspace(lo, hi, num_bins + 1, dtype=np.float32)
    lo = max(lo, 1e-3)
    if hi <= lo * 1.001:
        return None
    return np.geomspace(lo, hi, num_bins + 1).astype(np.float32)


def _sample_candidate_pixels(
    ys: np.ndarray,
    xs: np.ndarray,
    n: int,
    rng: np.random.Generator,
    replace_short: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0 or len(ys) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    if len(ys) >= n:
        idx = rng.choice(len(ys), n, replace=False)
        return ys[idx].astype(np.int32), xs[idx].astype(np.int32)
    if not replace_short:
        return ys.astype(np.int32), xs.astype(np.int32)

    short = n - len(ys)
    extra_idx = rng.choice(len(ys), short, replace=True) if short > 0 else np.empty(0, dtype=np.int64)
    out_y = np.concatenate([ys, ys[extra_idx]]) if short > 0 else ys
    out_x = np.concatenate([xs, xs[extra_idx]]) if short > 0 else xs
    return out_y.astype(np.int32), out_x.astype(np.int32)


def _sample_candidate_pixels_depth_balanced(
    depth: np.ndarray,
    mask: np.ndarray,
    n: int,
    rng: np.random.Generator,
    depth_max: float,
    source_depth_sampling: str,
    source_depth_bin_edges: np.ndarray | None,
    source_depth_bin_weights: np.ndarray | None,
    source_depth_num_bins: int,
    replace_short: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if n <= 0 or len(ys) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    depth_values = depth[ys, xs].astype(np.float32, copy=False)
    edges = source_depth_bin_edges
    if edges is None:
        edges = _dynamic_depth_bin_edges(
            depth_values,
            depth_max=depth_max,
            source_depth_sampling=source_depth_sampling,
            source_depth_num_bins=source_depth_num_bins,
        )
    if edges is None or len(edges) < 2:
        return _sample_candidate_pixels(ys, xs, n, rng, replace_short=replace_short)

    # Values outside the configured edge range are folded into the nearest bin.
    bin_ids = np.digitize(depth_values, edges[1:-1], right=False)
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
    nonempty_bins = [int(b) for b in np.unique(bin_ids)]
    if not nonempty_bins:
        return _sample_candidate_pixels(ys, xs, n, rng, replace_short=replace_short)

    if source_depth_bin_weights is not None:
        if source_depth_bin_weights.size != len(edges) - 1:
            raise ValueError(
                "source_depth_bin_weights length must match number of depth bins "
                f"({len(edges) - 1}), got {source_depth_bin_weights.size}"
            )
        weights = source_depth_bin_weights[np.asarray(nonempty_bins, dtype=np.int64)].astype(np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = np.ones(len(nonempty_bins), dtype=np.float64)
        raw_quota = weights / float(weights.sum()) * float(n)
        quotas = np.floor(raw_quota).astype(np.int64)
        remainder = int(n - quotas.sum())
        if remainder > 0:
            order = np.argsort(-(raw_quota - quotas))
            quotas[order[:remainder]] += 1
    else:
        quota_base = n // len(nonempty_bins)
        remainder = n - quota_base * len(nonempty_bins)
        quotas = np.array(
            [quota_base + (1 if out_i < remainder else 0) for out_i in range(len(nonempty_bins))],
            dtype=np.int64,
        )

    selected: list[np.ndarray] = []
    selected_mask = np.zeros(len(ys), dtype=bool)
    remaining = 0

    candidate_ids = np.arange(len(ys), dtype=np.int64)
    for out_i, bin_id in enumerate(nonempty_bins):
        quota = int(quotas[out_i])
        if quota <= 0:
            continue
        bin_candidates = candidate_ids[bin_ids == bin_id]
        take = min(quota, len(bin_candidates))
        if take > 0:
            chosen = rng.choice(bin_candidates, take, replace=False)
            selected.append(chosen)
            selected_mask[chosen] = True
        remaining += quota - take

    if remaining > 0:
        leftovers = candidate_ids[~selected_mask]
        take = min(remaining, len(leftovers))
        if take > 0:
            chosen = rng.choice(leftovers, take, replace=False)
            selected.append(chosen)
            selected_mask[chosen] = True
            remaining -= take

    if remaining > 0 and replace_short:
        chosen = rng.choice(candidate_ids, remaining, replace=True)
        selected.append(chosen)

    if not selected:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    idx = np.concatenate(selected)
    return ys[idx].astype(np.int32), xs[idx].astype(np.int32)


def _project_world_points_to_frames(
    points_world: np.ndarray,
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_consistency_thresh: float = 0.05,
    depth_max: float = 100.0,
    depth_support_radius: int = 2,
    visible_abs_thresh: float = 0.25,
    occlusion_abs_thresh: float = 1.0,
    occlusion_rel_thresh: float = 0.08,
    occlusion_min_support: int = 9,
    visible_min_support: int = 1,
    occlusion_overrides_visible: bool = False,
    occlusion_far_depth_start: float | None = None,
    occlusion_far_depth_end: float | None = None,
    occlusion_far_abs_thresh: float | None = None,
    occlusion_far_rel_thresh: float | None = None,
    occlusion_far_min_support: int | None = None,
    far_uncertain_as_visible_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project static world points into all frames and separate defined/visible masks.

    Semantics:
      - ``valids``: the target 3D / visibility label is *defined* in this frame.
        This requires the point to project in-bounds, lie in front of the camera,
        and land on a pixel with a valid depth sample.
      - ``visibs``: the point is additionally *visible* under a depth-consistency
        check against the target depth map. Occluded points remain valid but are
        marked invisible.
    """
    points_world = np.asarray(points_world, dtype=np.float32)
    depths = [np.asarray(d, dtype=np.float32) for d in depths]
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)

    T = len(depths)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(
            f"Expected points_world to have shape [N,3], got {tuple(points_world.shape)}"
        )
    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics[None], (T, 3, 3)).copy()

    H, W = depths[0].shape
    N = points_world.shape[0]
    trajs_2d = np.zeros((T, N, 2), dtype=np.float32)
    valids = np.zeros((T, N), dtype=bool)
    visibs = np.zeros((T, N), dtype=bool)

    ones = np.ones((N, 1), dtype=np.float32)
    points_world_h = np.concatenate([points_world, ones], axis=-1)

    for t in range(T):
        E_t = extrinsics[t]
        K_t = intrinsics[t]
        depth_t = depths[t]

        points_cam_t = (E_t @ points_world_h.T).T[:, :3]
        uv_t, z_t = _project(points_cam_t, K_t)
        trajs_2d[t] = uv_t

        uv_finite = np.isfinite(uv_t).all(axis=-1)
        in_front = (z_t > 0) & np.isfinite(z_t) & (z_t < depth_max)
        in_bounds = (
            uv_finite
            & (uv_t[:, 0] >= 0)
            & (uv_t[:, 0] < W)
            & (uv_t[:, 1] >= 0)
            & (uv_t[:, 1] < H)
        )

        sample_mask = in_bounds & in_front
        visible_t = np.zeros((N,), dtype=bool)
        occluded_t = np.zeros((N,), dtype=bool)
        if sample_mask.any():
            ids = np.flatnonzero(sample_mask)
            px = np.clip(np.round(uv_t[ids, 0]).astype(np.int32), 0, W - 1)
            py = np.clip(np.round(uv_t[ids, 1]).astype(np.int32), 0, H - 1)
            z_sel = z_t[ids].astype(np.float32)
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
            support_required = np.full(
                z_sel.shape, min_support, dtype=np.int16
            )
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
                        + alpha * (max(int(occlusion_far_min_support), 1) - min_support)
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
                    sampled = depth_t[yy, xx].astype(np.float32)
                    depth_ok = ok & (sampled > 0) & np.isfinite(sampled) & (sampled < depth_max)
                    valid_depth_support += depth_ok.astype(np.int16)
                    visible_support += (depth_ok & (np.abs(sampled - z_sel) <= visible_margin)).astype(np.int16)
                    nearer_support += (depth_ok & (sampled + occlusion_margin < z_sel)).astype(np.int16)
            depth_occluded = nearer_support >= support_required
            far_uncertain_visible = np.zeros((len(ids),), dtype=bool)
            if far_uncertain_as_visible_depth is not None:
                far_uncertain_visible = (
                    (z_sel >= float(far_uncertain_as_visible_depth))
                    & (valid_depth_support > 0)
                    & (~depth_occluded)
                )
            visible_t[ids] = (visible_support >= vis_min_support) | far_uncertain_visible
            if occlusion_overrides_visible:
                visible_t[ids] &= ~depth_occluded
                occluded_t[ids] = depth_occluded
            else:
                occluded_t[ids] = (~visible_t[ids]) & depth_occluded

        valids[t] = visible_t | occluded_t
        visibs[t] = visible_t

    return trajs_2d, valids, visibs


def _voxel_deduplicate_world_points(
    points_world: np.ndarray,
    voxel_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return stable indices for one random representative per 3D voxel."""
    points_world = np.asarray(points_world, dtype=np.float32)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(
            f"Expected points_world to have shape [N,3], got {tuple(points_world.shape)}"
        )
    if points_world.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    finite = np.isfinite(points_world).all(axis=-1)
    finite_idx = np.flatnonzero(finite)
    if finite_idx.size == 0:
        return np.empty(0, dtype=np.int64)

    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        return finite_idx.astype(np.int64, copy=False)

    pts = points_world[finite_idx]
    order = rng.permutation(pts.shape[0])
    voxels = np.floor(pts[order] / voxel_size).astype(np.int64)
    _, first_pos = np.unique(voxels, axis=0, return_index=True)
    keep = finite_idx[order[first_pos]]
    return np.sort(keep).astype(np.int64, copy=False)


def recompute_track_projection_masks(
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    trajs_3d_world: np.ndarray,
    depth_consistency_thresh: float = 0.05,
    depth_max: float = 100.0,
    depth_support_radius: int = 2,
    visible_abs_thresh: float = 0.25,
    occlusion_abs_thresh: float = 1.0,
    occlusion_rel_thresh: float = 0.08,
    occlusion_min_support: int = 9,
    visible_min_support: int = 1,
    occlusion_overrides_visible: bool = False,
    occlusion_far_depth_start: float | None = None,
    occlusion_far_depth_end: float | None = None,
    occlusion_far_abs_thresh: float | None = None,
    occlusion_far_rel_thresh: float | None = None,
    occlusion_far_min_support: int | None = None,
    far_uncertain_as_visible_depth: float | None = None,
) -> dict[str, np.ndarray]:
    """Refresh projected 2D tracks and defined/visible masks from cached world points."""
    trajs_3d_world = np.asarray(trajs_3d_world, dtype=np.float32)
    if trajs_3d_world.ndim != 3 or trajs_3d_world.shape[-1] != 3:
        raise ValueError(
            f"Expected trajs_3d_world to have shape [T,N,3], got {tuple(trajs_3d_world.shape)}"
        )
    if trajs_3d_world.shape[0] == 0:
        return {
            "trajs_2d": np.zeros((0, 0, 2), dtype=np.float32),
            "valids": np.zeros((0, 0), dtype=bool),
            "visibs": np.zeros((0, 0), dtype=bool),
        }

    points_world = trajs_3d_world[0]
    trajs_2d, valids, visibs = _project_world_points_to_frames(
        points_world=points_world,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth_consistency_thresh=depth_consistency_thresh,
        depth_max=depth_max,
        depth_support_radius=depth_support_radius,
        visible_abs_thresh=visible_abs_thresh,
        occlusion_abs_thresh=occlusion_abs_thresh,
        occlusion_rel_thresh=occlusion_rel_thresh,
        occlusion_min_support=occlusion_min_support,
        visible_min_support=visible_min_support,
        occlusion_overrides_visible=occlusion_overrides_visible,
        occlusion_far_depth_start=occlusion_far_depth_start,
        occlusion_far_depth_end=occlusion_far_depth_end,
        occlusion_far_abs_thresh=occlusion_far_abs_thresh,
        occlusion_far_rel_thresh=occlusion_far_rel_thresh,
        occlusion_far_min_support=occlusion_far_min_support,
        far_uncertain_as_visible_depth=far_uncertain_as_visible_depth,
    )
    return {
        "trajs_2d": trajs_2d,
        "valids": valids,
        "visibs": visibs,
    }


# --------------------------------------------------------------------------- #
# Main function
# --------------------------------------------------------------------------- #

def compute_tracks(
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    num_points: int = 8000,
    boundary_ratio: float = 0.3,
    depth_consistency_thresh: float = 0.05,
    depth_max: float = 100.0,
    rng_seed: int = 42,
    num_ref_segments: int = 1,
    ref_frame_strategy: str = "max_valid",
    source_depth_sampling: str = "uniform",
    source_depth_bin_edges: Sequence[float] | np.ndarray | None = None,
    source_depth_bin_weights: Sequence[float] | np.ndarray | None = None,
    source_depth_num_bins: int = 7,
    points_per_ref_frame: int | None = None,
    ref_frame_stride: int = 1,
    world_dedup_voxel_size: float | None = None,
    compact_static_tracks: bool = False,
    occlusion_far_depth_start: float | None = None,
    occlusion_far_depth_end: float | None = None,
    occlusion_far_abs_thresh: float | None = None,
    occlusion_far_rel_thresh: float | None = None,
    occlusion_far_min_support: int | None = None,
    far_uncertain_as_visible_depth: float | None = None,
    source_valid_masks: Sequence[np.ndarray] | None = None,
) -> dict:
    """
    Compute 2D/3D point tracks for a static scene clip (strategy A).

    Args:
        depths:      List of T depth maps [H,W] float32. Invalid = 0/nan/inf.
        intrinsics:  [T,3,3] float32 (or [3,3] broadcast).
        extrinsics:  [T,4,4] float32 world-to-camera.
        num_points:  Number of source points N to sample.
        boundary_ratio:
            Fraction of N sampled near depth discontinuities (30%).
        depth_consistency_thresh:
            Relative depth error threshold for validity check (5%).
        rng_seed:    RNG seed for reproducibility.
        source_depth_sampling:
            ``uniform`` preserves the legacy pixel-uniform sampler.
            ``log_balanced`` / ``linear_balanced`` split candidate source
            pixels into metric-depth bins and sample roughly equal counts from
            each nonempty bin, separately for boundary and non-boundary quotas.
        source_depth_bin_edges:
            Optional explicit bin edges in metres.  When provided, values
            outside the range are folded into the nearest edge bin.
        source_depth_bin_weights:
            Optional per-bin weights.  The length must equal the number of
            bins, i.e. ``len(source_depth_bin_edges) - 1`` for explicit edges.
            Larger weights oversample that depth range.
        source_depth_num_bins:
            Number of dynamic bins when explicit edges are not provided.
        points_per_ref_frame:
            If positive, sample this many source points from every reference
            frame instead of splitting ``num_points`` over temporal segments.
            This is useful for long static-scene videos where any random
            training window should contain local source points.
        ref_frame_stride:
            Frame stride used with ``points_per_ref_frame``.  A value of 1
            samples every frame.
        world_dedup_voxel_size:
            Optional metric voxel size for deduplicating sampled world points.
            One random representative is kept per voxel.
        compact_static_tracks:
            If True, return only ``static_points_world`` and per-point metadata
            instead of dense full-sequence ``trajs_2d``/``valids``/``visibs``.
            Adapters can project the compact points for a requested clip.

    Returns:
        dict with keys:
            trajs_2d       [T,N,2]  float32  pixel coords (x=col, y=row)
            trajs_3d_world [T,N,3]  float32  world coords (same every frame)
            valids         [T,N]    bool     target defined (in-bounds, in-front,
                                             depth sample exists)
            visibs         [T,N]    bool     target additionally visible under
                                             depth-consistency check
            ref_frame      int
            num_points     int      actual N (may be < requested if depth sparse)
            track_semantics_version int     cache semantics version
    """
    rng = np.random.default_rng(rng_seed)

    depths = [np.asarray(d, dtype=np.float32) for d in depths]
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)

    T = len(depths)
    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics[None], (T, 3, 3)).copy()

    if source_valid_masks is not None:
        if len(source_valid_masks) != T:
            raise ValueError(
                f"source_valid_masks length must match depths ({T}), "
                f"got {len(source_valid_masks)}"
            )
        source_masks = [np.asarray(m, dtype=bool) for m in source_valid_masks]
        for idx, (mask, depth) in enumerate(zip(source_masks, depths)):
            if mask.shape != depth.shape:
                raise ValueError(
                    f"source_valid_masks[{idx}] shape {mask.shape} does not "
                    f"match depth shape {depth.shape}"
                )
    else:
        source_masks = None

    # ------------------------------------------------------------------ #
    # 1. Select reference frame(s)                                         #
    # ------------------------------------------------------------------ #
    valid_counts_list = []
    for idx, d in enumerate(depths):
        valid = (d > 0) & np.isfinite(d) & (d < depth_max)
        if source_masks is not None:
            valid = valid & source_masks[idx]
        valid_counts_list.append(int(valid.sum()))
    valid_counts = np.array(valid_counts_list)
    per_frame_mode = points_per_ref_frame is not None and int(points_per_ref_frame) > 0
    if per_frame_mode:
        stride = max(1, int(ref_frame_stride))
        ref_frames = [
            int(i)
            for i in range(0, T, stride)
            if valid_counts[i] > 0
        ]
    else:
        stride = 0
        ref_frames = _select_ref_frames(valid_counts, num_ref_segments, ref_frame_strategy)

    if not ref_frames:
        ref_frames = [int(np.argmax(valid_counts))]

    source_depth_sampling = _normalise_source_depth_sampling(source_depth_sampling)
    source_depth_bin_edges_arr = _normalise_depth_bin_edges(source_depth_bin_edges)
    source_depth_bin_weights_arr = _normalise_depth_bin_weights(source_depth_bin_weights)
    if source_depth_bin_weights_arr is not None:
        expected_bins = (
            len(source_depth_bin_edges_arr) - 1
            if source_depth_bin_edges_arr is not None
            else int(source_depth_num_bins)
        )
        if source_depth_bin_weights_arr.size != expected_bins:
            raise ValueError(
                "source_depth_bin_weights length must match depth bin count "
                f"({expected_bins}), got {source_depth_bin_weights_arr.size}"
            )

    # ------------------------------------------------------------------ #
    # 2. Sample source points from each reference frame                    #
    # ------------------------------------------------------------------ #
    if per_frame_mode:
        points_per_ref = max(1, int(points_per_ref_frame))
        points_per_seg = 0
        remainder = 0
    else:
        points_per_ref = 0
        points_per_seg = num_points // len(ref_frames)
        remainder = num_points - points_per_seg * len(ref_frames)

    all_P_world: list[np.ndarray] = []
    all_ref_frames: list[np.ndarray] = []
    all_ref_uv: list[np.ndarray] = []
    all_ref_depth: list[np.ndarray] = []

    for seg_i, ref in enumerate(ref_frames):
        n_pts = (
            points_per_ref
            if per_frame_mode
            else points_per_seg + (1 if seg_i < remainder else 0)
        )
        if n_pts <= 0:
            continue

        depth_ref = depths[ref]
        K_ref = intrinsics[ref]
        E_ref = extrinsics[ref]

        valid_ref = (depth_ref > 0) & np.isfinite(depth_ref) & (depth_ref < depth_max)
        if source_masks is not None:
            valid_ref = valid_ref & source_masks[ref]

        n_boundary = int(n_pts * boundary_ratio)

        bnd_mask = _depth_boundary_mask(depth_ref) & valid_ref
        if source_depth_sampling == "uniform":
            bnd_ys, bnd_xs = np.where(bnd_mask)
            uni_ys, uni_xs = np.where(valid_ref)
            sel_bnd_y, sel_bnd_x = _sample_candidate_pixels(
                bnd_ys,
                bnd_xs,
                n_boundary,
                rng,
                replace_short=False,
            )
            n_uni_needed = n_pts - len(sel_bnd_y)
            sel_uni_y, sel_uni_x = _sample_candidate_pixels(
                uni_ys,
                uni_xs,
                n_uni_needed,
                rng,
            )
        else:
            sel_bnd_y, sel_bnd_x = _sample_candidate_pixels_depth_balanced(
                depth_ref,
                bnd_mask,
                n_boundary,
                rng,
                depth_max=depth_max,
                source_depth_sampling=source_depth_sampling,
                source_depth_bin_edges=source_depth_bin_edges_arr,
                source_depth_bin_weights=source_depth_bin_weights_arr,
                source_depth_num_bins=source_depth_num_bins,
                replace_short=False,
            )
            n_uni_needed = n_pts - len(sel_bnd_y)
            sel_uni_y, sel_uni_x = _sample_candidate_pixels_depth_balanced(
                depth_ref,
                valid_ref,
                n_uni_needed,
                rng,
                depth_max=depth_max,
                source_depth_sampling=source_depth_sampling,
                source_depth_bin_edges=source_depth_bin_edges_arr,
                source_depth_bin_weights=source_depth_bin_weights_arr,
                source_depth_num_bins=source_depth_num_bins,
            )

        src_y = np.concatenate([sel_bnd_y, sel_uni_y]).astype(np.int32)
        src_x = np.concatenate([sel_bnd_x, sel_uni_x]).astype(np.int32)
        if len(src_y) == 0:
            continue
        src_uv = np.stack([src_x, src_y], axis=-1).astype(np.float32)
        src_d = depth_ref[src_y, src_x].astype(np.float32)

        P_cam_ref = _unproject(src_uv, src_d, K_ref)
        c2w_ref = _c2w_from_w2c(E_ref)
        ones = np.ones((len(src_y), 1), dtype=np.float32)
        P_hom = np.concatenate([P_cam_ref, ones], axis=-1)
        P_world_seg = (c2w_ref @ P_hom.T).T[:, :3]
        all_P_world.append(P_world_seg)
        all_ref_frames.append(np.full((P_world_seg.shape[0],), int(ref), dtype=np.int32))
        all_ref_uv.append(src_uv)
        all_ref_depth.append(src_d.astype(np.float32, copy=False))

    if not all_P_world:
        raise RuntimeError("No valid source points could be sampled for precomputation")

    P_world = np.concatenate(all_P_world, axis=0)
    point_ref_frames = np.concatenate(all_ref_frames, axis=0)
    point_ref_uv = np.concatenate(all_ref_uv, axis=0).astype(np.float32, copy=False)
    point_ref_depth = np.concatenate(all_ref_depth, axis=0).astype(np.float32, copy=False)
    pre_dedup_N = int(P_world.shape[0])

    dedup_voxel = 0.0 if world_dedup_voxel_size is None else float(world_dedup_voxel_size)
    keep_idx = _voxel_deduplicate_world_points(P_world, dedup_voxel, rng)
    if keep_idx.size == 0:
        raise RuntimeError("No finite source points remained after voxel deduplication")
    if keep_idx.size != P_world.shape[0]:
        P_world = P_world[keep_idx]
        point_ref_frames = point_ref_frames[keep_idx]
        point_ref_uv = point_ref_uv[keep_idx]
        point_ref_depth = point_ref_depth[keep_idx]

    N = P_world.shape[0]
    ref_counts = np.bincount(point_ref_frames, minlength=T).astype(np.int32)

    base_result = {
        "ref_frame":      int(ref_frames[0]),
        "ref_frames":     np.array(ref_frames, dtype=np.int32),
        "num_points":     N,
        "pre_dedup_num_points": np.array(pre_dedup_N, dtype=np.int32),
        "track_semantics_version": TRACK_SEMANTICS_VERSION,
        "num_ref_segments": int(num_ref_segments),
        "points_per_ref_frame": int(points_per_ref) if per_frame_mode else np.array(0, dtype=np.int32),
        "ref_frame_stride": int(stride),
        "world_dedup_voxel_size": np.array(dedup_voxel, dtype=np.float32),
        "compact_static_tracks": np.array(bool(compact_static_tracks), dtype=np.bool_),
        "track_depth_max": float(depth_max),
        "track_depth_consistency_thresh": np.array(depth_consistency_thresh, dtype=np.float32),
        "source_depth_sampling_mode": SOURCE_DEPTH_SAMPLING_MODE_IDS[source_depth_sampling],
        "source_depth_bin_edges": (
            source_depth_bin_edges_arr.astype(np.float32)
            if source_depth_bin_edges_arr is not None
            else np.empty(0, dtype=np.float32)
        ),
        "source_depth_bin_weights": (
            source_depth_bin_weights_arr.astype(np.float32)
            if source_depth_bin_weights_arr is not None
            else np.empty(0, dtype=np.float32)
        ),
        "source_depth_num_bins": int(source_depth_num_bins),
        "static_points_world": P_world.astype(np.float32, copy=False),
        "static_point_ref_frames": point_ref_frames.astype(np.int32, copy=False),
        "static_point_ref_uv": point_ref_uv.astype(np.float32, copy=False),
        "static_point_ref_depth": point_ref_depth.astype(np.float32, copy=False),
        "static_point_ref_counts": ref_counts,
        # 1 means points are already in the adapter's output world convention.
        "static_points_world_convention": np.array(1, dtype=np.int32),
    }

    if compact_static_tracks:
        return base_result

    # ------------------------------------------------------------------ #
    # 4. Project into every frame                                          #
    # ------------------------------------------------------------------ #
    trajs_2d, valids, visibs = _project_world_points_to_frames(
        points_world=P_world,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth_consistency_thresh=depth_consistency_thresh,
        depth_max=depth_max,
        occlusion_far_depth_start=occlusion_far_depth_start,
        occlusion_far_depth_end=occlusion_far_depth_end,
        occlusion_far_abs_thresh=occlusion_far_abs_thresh,
        occlusion_far_rel_thresh=occlusion_far_rel_thresh,
        occlusion_far_min_support=occlusion_far_min_support,
        far_uncertain_as_visible_depth=far_uncertain_as_visible_depth,
    )
    trajs_3d_world = np.broadcast_to(P_world[None], (T, N, 3)).copy()
    valid_visible_counts = (valids & visibs).sum(axis=1).astype(np.int32)

    base_result.update({
        "trajs_2d":       trajs_2d,
        "trajs_3d_world": trajs_3d_world,
        "valids":         valids,
        "visibs":         visibs,
        "trajs_3d_world_convention": np.array(1, dtype=np.int32),
        "valid_visible_counts": valid_visible_counts,
    })
    return base_result
