"""
D4RT Query Builder  —  第 3 层（D4RT Query Builder）

职责：
  - 接受 TransformResult（来自 GeometryTransformPipeline）
  - 采样 t_src / t_tgt / t_cam（含 40% t_tgt=t_cam）
  - 采样 num_queries 个 source queries（30% boundary oversampling）
  - 构造所有 targets 和 mask_* 字段
  - 提取 local RGB patches

不做的事：
  - 不读文件
  - 不做 crop / resize
  - 不决定混合采样权重
  - 不对 dataset_name 做 if/else 分叉

has_tracks=False 的 clip（ScanNet、Co3D 等）：
  - mask_2d / mask_disp / mask_vis 全部为 False
  - 仍可产出 mask_3d（通过 depth 反投影），由调用方决定是否启用
  - 该路径由 metadata['has_tracks'] 控制，query builder 本身不感知 dataset 名
"""

from __future__ import annotations

import random as _random_module
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from datasets.transforms import TransformResult
from utils.patches import extract_local_patches


# ---------------------------------------------------------------------------
# Boundary mask helpers
# ---------------------------------------------------------------------------

def _depth_boundary_mask(depth: np.ndarray) -> np.ndarray:
    """
    Bool mask [H,W]：深度边缘（Sobel 梯度 >= 85th percentile）。
    depth: [H,W] float32，无效值为 nan/inf/0。
    """
    d0 = depth.astype(np.float32, copy=False)
    if np.isfinite(d0).all():
        d = d0
    else:
        d = np.nan_to_num(d0, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    sx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    valid = d > 0
    if not valid.any():
        return np.zeros_like(d, dtype=bool)
    vals = mag[valid]
    if vals.size == 0 or float(vals.max()) <= 0.0:
        return np.zeros_like(d, dtype=bool)
    thr = max(float(np.percentile(vals, 85.0)), 1e-6)
    return valid & (mag >= thr)


def _motion_boundary_mask(
    trajs_2d: np.ndarray,   # [T,N,2]
    valids: np.ndarray,     # [T,N] bool
    frame_idx: int,
    crop_h: int,
    crop_w: int,
    temporal_step: int = 1,
) -> np.ndarray:
    """
    Bool mask [H,W]：运动边缘，由相邻帧轨迹速度场的 Sobel 导出。
    仅当 trajs_2d 存在时调用。
    """
    motion_map = np.zeros((crop_h, crop_w), dtype=np.float32)
    support_map = np.zeros((crop_h, crop_w), dtype=np.float32)

    pts = trajs_2d[frame_idx]                                   # [N,2]
    cur_valid = valids[frame_idx] & np.isfinite(pts).all(-1)

    mag = np.zeros(pts.shape[0], dtype=np.float32)
    sup = np.zeros(pts.shape[0], dtype=np.float32)

    for nbr_idx in [frame_idx + temporal_step, frame_idx - temporal_step]:
        if not (0 <= nbr_idx < trajs_2d.shape[0]):
            continue
        nbr = trajs_2d[nbr_idx]
        nbr_valid = valids[nbr_idx] & np.isfinite(nbr).all(-1)
        pair = cur_valid & nbr_valid
        if pair.any():
            mag[pair] += np.linalg.norm(nbr[pair] - pts[pair], axis=1)
            sup[pair] += 1.0

    defined = sup > 0.0
    if not defined.any():
        return np.zeros((crop_h, crop_w), dtype=bool)

    mag[defined] /= sup[defined]
    xy = np.round(pts[defined]).astype(np.int32)
    if xy.size == 0:
        return np.zeros((crop_h, crop_w), dtype=bool)
    xx = np.clip(xy[:, 0], 0, crop_w - 1)
    yy = np.clip(xy[:, 1], 0, crop_h - 1)
    flat_idx    = yy * crop_w + xx
    motion_map  = np.bincount(flat_idx, weights=mag[defined],
                              minlength=crop_h * crop_w).reshape(crop_h, crop_w).astype(np.float32)
    support_map = np.bincount(flat_idx,
                              minlength=crop_h * crop_w).reshape(crop_h, crop_w).astype(np.float32)

    filled = support_map > 0.0
    motion_map[filled] /= support_map[filled]
    motion_map = cv2.GaussianBlur(motion_map, (5, 5), sigmaX=0.0)
    support_mask = (
        cv2.dilate(filled.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    )
    if not support_mask.any():
        return np.zeros((crop_h, crop_w), dtype=bool)

    sx2 = cv2.Sobel(motion_map, cv2.CV_32F, 1, 0, ksize=3)
    sy2 = cv2.Sobel(motion_map, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = np.sqrt(sx2 ** 2 + sy2 ** 2)
    sup_vals = mag2[support_mask]
    if sup_vals.size == 0 or float(sup_vals.max()) <= 0.0:
        return np.zeros((crop_h, crop_w), dtype=bool)

    thr = max(float(np.percentile(sup_vals, 85.0)), 1e-6)
    boundary = support_mask & (mag2 >= thr)
    boundary = (
        cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    )
    return boundary


def _motion_boundary_masks_sequence(
    trajs_2d: np.ndarray,   # [T,N,2]
    valids: np.ndarray,     # [T,N] bool
    crop_h: int,
    crop_w: int,
    temporal_step: int = 1,
) -> list[np.ndarray]:
    """
    Compute all per-frame motion boundary masks while sharing trajectory
    finite checks and neighbor displacement work across frames.
    """
    T = trajs_2d.shape[0]
    if T == 0:
        return []

    pts = trajs_2d.astype(np.float32, copy=False)
    pts_finite = np.isfinite(pts).all(-1)
    valid_pts = valids & pts_finite

    mag_sum = np.zeros(valid_pts.shape, dtype=np.float32)
    support = np.zeros(valid_pts.shape, dtype=np.float32)
    if temporal_step > 0 and temporal_step < T:
        cur = pts[:-temporal_step]
        nbr = pts[temporal_step:]
        pair = valid_pts[:-temporal_step] & valid_pts[temporal_step:]
        if pair.any():
            ti, ni = np.nonzero(pair)
            delta = np.linalg.norm(
                nbr[ti, ni] - cur[ti, ni],
                axis=1,
            ).astype(np.float32, copy=False)
            mag_sum[ti, ni] += delta
            mag_sum[ti + temporal_step, ni] += delta
            support[ti, ni] += 1.0
            support[ti + temporal_step, ni] += 1.0

    defined = support > 0.0
    mag = np.zeros_like(mag_sum, dtype=np.float32)
    mag[defined] = mag_sum[defined] / support[defined]

    kernel = np.ones((5, 5), np.uint8)
    masks: list[np.ndarray] = []
    for fi in range(T):
        frame_defined = defined[fi]
        if not frame_defined.any():
            masks.append(np.zeros((crop_h, crop_w), dtype=bool))
            continue

        xy = np.round(pts[fi, frame_defined]).astype(np.int32)
        if xy.size == 0:
            masks.append(np.zeros((crop_h, crop_w), dtype=bool))
            continue

        xx = np.clip(xy[:, 0], 0, crop_w - 1)
        yy = np.clip(xy[:, 1], 0, crop_h - 1)
        flat_idx = yy * crop_w + xx
        motion_map = np.bincount(
            flat_idx,
            weights=mag[fi, frame_defined],
            minlength=crop_h * crop_w,
        ).reshape(crop_h, crop_w).astype(np.float32)
        support_map = np.bincount(
            flat_idx,
            minlength=crop_h * crop_w,
        ).reshape(crop_h, crop_w).astype(np.float32)

        filled = support_map > 0.0
        motion_map[filled] /= support_map[filled]
        motion_map = cv2.GaussianBlur(motion_map, (5, 5), sigmaX=0.0)
        support_mask = cv2.dilate(filled.astype(np.uint8), kernel, iterations=1) > 0
        if not support_mask.any():
            masks.append(np.zeros((crop_h, crop_w), dtype=bool))
            continue

        sx2 = cv2.Sobel(motion_map, cv2.CV_32F, 1, 0, ksize=3)
        sy2 = cv2.Sobel(motion_map, cv2.CV_32F, 0, 1, ksize=3)
        mag2 = np.sqrt(sx2 ** 2 + sy2 ** 2)
        sup_vals = mag2[support_mask]
        if sup_vals.size == 0 or float(sup_vals.max()) <= 0.0:
            masks.append(np.zeros((crop_h, crop_w), dtype=bool))
            continue

        thr = max(float(np.percentile(sup_vals, 85.0)), 1e-6)
        boundary = support_mask & (mag2 >= thr)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1) > 0
        masks.append(boundary)

    return masks


# ---------------------------------------------------------------------------
# Query builder output
# ---------------------------------------------------------------------------

@dataclass
class QuerySample:
    """最终训练样本，由 D4RTQueryBuilder 产出。"""

    # ---- video ----
    video: torch.Tensor             # [S,3,H,W]  float32 [0,1] or uint8 compact storage
    # Cropped frames at original crop resolution (before resize): [S,3,crop_h,crop_w]
    # Set to None when all frames already match img_size.
    highres_video: Optional[torch.Tensor]
    depths: Optional[torch.Tensor]  # [S,1,H,W]  float32  | None
    normals: Optional[torch.Tensor] # [S,3,H,W]  float32  | None

    # ---- query indices ----
    coords: torch.Tensor            # [Q,2]  归一化到 [0,1]（crop 坐标系）
    t_src: torch.Tensor             # [Q]    long
    t_tgt: torch.Tensor             # [Q]    long
    t_cam: torch.Tensor             # [Q]    long

    # ---- camera ----
    intrinsics: torch.Tensor        # [S,3,3]
    extrinsics: torch.Tensor        # [S,4,4]

    # ---- supervision targets ----
    targets: dict                   # see _build_empty_targets for keys

    # ---- optional patches ----
    local_patches: Optional[torch.Tensor]   # [Q,3,P,P]  | None

    # ---- transform metadata (for sampled_highres patch provider) ----
    transform_metadata: dict

    # ---- aspect ratio ----
    aspect_ratio: torch.Tensor              # [1] actual input view width / height before square resize

    # ---- metadata ----
    dataset_name: str
    sequence_name: str
    metadata: dict


def _build_empty_targets(num_queries: int) -> dict:
    Q = num_queries
    return {
        "pos_2d":                   torch.zeros(Q, 2),
        "pos_3d":                   torch.zeros(Q, 3),
        "visibility":               torch.zeros(Q),
        "displacement":             torch.zeros(Q, 3),
        "normal":                   torch.zeros(Q, 3),
        "mask_3d":                  torch.zeros(Q, dtype=torch.bool),
        "mask_2d":                  torch.zeros(Q, dtype=torch.bool),
        "mask_vis":                 torch.zeros(Q, dtype=torch.bool),
        "mask_disp":                torch.zeros(Q, dtype=torch.bool),
        "mask_normal":              torch.zeros(Q, dtype=torch.bool),
        "source_is_boundary":       torch.zeros(Q, dtype=torch.bool),
        "source_is_depth_boundary": torch.zeros(Q, dtype=torch.bool),
        "source_is_motion_boundary":torch.zeros(Q, dtype=torch.bool),
        "point_indices":            torch.zeros(Q, dtype=torch.long),
        # Semantic marker for static-reprojection supervision on has_tracks=False data.
        "is_static_reprojection":   torch.zeros(Q, dtype=torch.bool),
    }


def _normalize_crop_coords(xy: np.ndarray, crop_w: int, crop_h: int) -> np.ndarray:
    """Normalize crop-plane pixel coords to [0,1] and clamp subpixel edge spill."""
    norm = np.array([max(crop_w - 1, 1), max(crop_h - 1, 1)], dtype=np.float32)
    return np.clip(xy.astype(np.float32) / norm, 0.0, 1.0)


def _hw_pair(value: object) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    try:
        seq = list(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(seq) != 2:
        return None
    try:
        h = float(seq[0])
        w = float(seq[1])
    except (TypeError, ValueError):
        return None
    if h <= 0.0 or w <= 0.0:
        return None
    return h, w


def _native_equivalent_aspect_ratio(result: TransformResult) -> float:
    """Return width/height of the crop in the original RGB coordinate system."""
    crop = result.crop
    ratio = float(crop.crop_w) / max(float(crop.crop_h), 1.0)

    # Some adapters, notably ScanNet++, resize RGB/depth/tracks to a target
    # plane before the generic crop -> square-resize transform.  The model's
    # aspect token is meant to describe the pre-square view, so map target-plane
    # crop ratio back to native RGB when the adapter exposes both planes.
    metadata = result.metadata or {}
    rgb_hw = _hw_pair(metadata.get("rgb_hw"))
    target_hw = _hw_pair(metadata.get("target_hw"))
    if rgb_hw is None or target_hw is None:
        return ratio

    native_h, native_w = rgb_hw
    target_h, target_w = target_hw
    target_ratio = target_w / target_h
    native_ratio = native_w / native_h
    return ratio * native_ratio / max(target_ratio, 1e-6)


# ---------------------------------------------------------------------------
# Per-dataset camera-space Z thresholds for mask_3d gating
# ---------------------------------------------------------------------------

# Camera-space depth upper bound per dataset name.
# Points with Z > threshold get mask_3d=False and are excluded from 3D loss.
# Thresholds derived from empirical Z-distribution survey (~30 sequences each,
# 5 frames per sequence, 2000 points per frame, camera-space Z values).
_DATASET_Z_MAX: dict[str, float] = {
    # 实测相机坐标系 Z 分布（各 ~30 sequences，每帧 2000 点抽样）：
    #   pointodyssey:   p50=2.99  p99=8.48  p99.9=31.76  max=517  → 12 覆盖 >99.7%
    #   kubric:         p50=8.66  p99=49.11 p99.9=49.90  max=50.8 → 55 覆盖全部
    #   dynamic_replica:p50=2.92  p99=7.33  p99.9=8.35   max=9.08 → 12 覆盖全部
    #   scannet:        metric metres, 室内场景，实测 max ≈ 2.18 m → 5 安全
    #   co3dv2:         任意坐标单位，14% 有效点 Z>20 → 50 保守上限
    #   blendedmvs:     rendered depth, 场景尺度差异极大（室内 ~1m 到室外 ~468m）
    #                   35% 场景 max>20m, p99 ≈ 448m, max=468m → 500 覆盖全部
    #   mvssynth:       GTA-V outdoor depth, sampled raw depth p99.5 ≈ 93m,
    #                   long tail can reach 650m → 100m keeps main geometry
    "pointodyssey":    12.0,
    "kubric":          55.0,
    "dynamic_replica": 12.0,
    "scannet":          5.0,
    "co3dv2":          50.0,
    "blendedmvs":     500.0,
    "mvssynth":       100.0,
    # vkitti2: sampled p99 ~= 300m, p99.5 ~= 355m; keep the far-road geometry
    # without training on the 650m depth tail / sky sentinel range.
    "vkitti2":        400.0,
    # tartanair: adapter maps depth >200m to invalid; this keeps valid outdoor
    # tail points while matching that loader-side cutoff.
    "tartanair":      200.0,
    # waymo: vehicle-mounted LiDAR + FRONT camera. Effective LiDAR range ~75m,
    # but projected points on buildings/roads commonly extend to 100m+. The
    # default 20m fallback drops the bulk of valid source points and causes
    # frequent "No valid source points after transform" failures. Use 120m
    # to keep mid-range road geometry while still cutting the sky/long-tail.
    "waymo":          120.0,
}
_DEFAULT_Z_MAX = 20.0   # fallback for unknown datasets


_DATASET_SOURCE_DEPTH_ADAPTIVE_BIN_WEIGHTS: dict[str, np.ndarray] = {
    "vkitti2": np.array([12.0, 12.0, 6.0, 3.0, 1.0, 1.0], dtype=np.float32),
}


def _depth_balanced_choice(
    candidate_indices: np.ndarray,
    candidate_depths: np.ndarray,
    size: int,
    np_rng: np.random.Generator,
    bin_weights: np.ndarray,
) -> np.ndarray:
    """Sample by adaptive log-depth bins computed from current candidates."""
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if len(candidate_indices) == 0:
        return np.empty(size, dtype=np.int64)

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_depths = np.asarray(candidate_depths, dtype=np.float32)
    ok = np.isfinite(candidate_depths) & (candidate_depths > 1e-3)
    if not ok.any():
        return np_rng.choice(candidate_indices, size=size)

    valid_indices = candidate_indices[ok]
    valid_depths = candidate_depths[ok]
    num_bins = min(len(bin_weights), len(valid_indices))
    if num_bins <= 0:
        return np_rng.choice(candidate_indices, size=size)

    lo, hi = np.percentile(valid_depths, [1.0, 99.0])
    lo = max(float(lo), 1e-3)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np_rng.choice(candidate_indices, size=size)

    edges = np.geomspace(lo, hi, num=num_bins + 1).astype(np.float32)
    bin_ids = np.searchsorted(edges, valid_depths, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)
    nonempty_bins = np.array([i for i in range(num_bins) if np.any(bin_ids == i)], dtype=np.int64)
    if nonempty_bins.size == 0:
        return np_rng.choice(candidate_indices, size=size)

    weights = bin_weights[nonempty_bins].astype(np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        return np_rng.choice(candidate_indices, size=size)
    probs = weights / weights.sum()

    chosen_bins = np_rng.choice(nonempty_bins, size=size, p=probs)
    out = np.empty(size, dtype=np.int64)
    for bin_id in np.unique(chosen_bins):
        mask = chosen_bins == bin_id
        pool = valid_indices[bin_ids == bin_id]
        out[mask] = np_rng.choice(pool, size=int(mask.sum()))
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class D4RTQueryBuilder:
    """
    将 TransformResult 转换成最终训练 sample。

    Args:
        num_queries:          每个 clip 采样的 query 数，论文设定 2048。
        boundary_ratio:       boundary oversampling 概率，论文设定 0.3。
        t_tgt_eq_t_cam_ratio: t_tgt = t_cam 的概率，论文设定 0.4。
        patch_size:           local patch 边长（奇数），默认 9。
        use_motion_boundaries:是否计算运动边界 mask（需要 has_tracks=True）。
        precompute_patches:   是否预计算 local patches。
        precompute_from_highres: 是否从 highres_video 提取 patches（需要 precompute_patches=True）。
        return_highres_video:  Whether to include cropped high-res frames in QuerySample.
                               Defaults to old behavior: enabled when precompute_patches=False.
        store_video_uint8:    Store resized video compactly in QuerySample; train transfer restores float.
        store_auxiliary_tensors: Store raw resized depth/normal tensors in QuerySample.
        motion_boundary_on_resized: Compute motion-boundary masks directly on
                              the resized img_size plane. This preserves the
                              final mask resolution while avoiding crop-size
                              Sobel/blur work on large source images.
        dataset_z_max:        per-dataset camera-space Z upper bound for mask_3d.
                              Keys are dataset names; missing keys fall back to
                              _DATASET_Z_MAX then _DEFAULT_Z_MAX.
    """

    def __init__(
        self,
        num_queries: int = 2048,
        boundary_ratio: float = 0.3,
        t_tgt_eq_t_cam_ratio: float = 0.4,
        patch_size: int = 9,
        use_motion_boundaries: bool = True,
        precompute_patches: bool = True,
        precompute_from_highres: bool = False,
        return_highres_video: Optional[bool] = None,
        allow_track_fallback: bool = False,
        store_video_uint8: bool = False,
        store_auxiliary_tensors: bool = True,
        motion_boundary_on_resized: bool = True,
        dataset_z_max: Optional[dict] = None,
    ) -> None:
        self.num_queries = num_queries
        self.boundary_ratio = boundary_ratio
        self.t_tgt_eq_t_cam_ratio = t_tgt_eq_t_cam_ratio
        self.patch_size = patch_size
        self.use_motion_boundaries = use_motion_boundaries
        self.precompute_patches = precompute_patches
        self.precompute_from_highres = precompute_from_highres
        self.return_highres_video = (not precompute_patches) if return_highres_video is None else bool(return_highres_video)
        self.allow_track_fallback = allow_track_fallback
        self.store_video_uint8 = store_video_uint8
        self.store_auxiliary_tensors = store_auxiliary_tensors
        self.motion_boundary_on_resized = bool(motion_boundary_on_resized)
        self.dataset_z_max: dict[str, float] = dataset_z_max or {}

        if precompute_from_highres and not precompute_patches:
            raise ValueError("precompute_from_highres requires precompute_patches=True")

    def _get_z_max(self, dataset_name: str) -> float:
        """Return the camera-space Z upper bound for mask_3d gating."""
        if dataset_name in self.dataset_z_max:
            return float(self.dataset_z_max[dataset_name])
        return _DATASET_Z_MAX.get(dataset_name, _DEFAULT_Z_MAX)

    def _get_source_depth_sampling(self, dataset_name: str) -> Optional[np.ndarray]:
        """Return adaptive source-query depth-bin sampling weights."""
        return _DATASET_SOURCE_DEPTH_ADAPTIVE_BIN_WEIGHTS.get(dataset_name)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        result: TransformResult,
        py_rng: Optional[_random_module.Random] = None,
        np_rng: Optional[np.random.Generator] = None,
        frames_time_ordered: bool = True,
    ) -> QuerySample:
        if py_rng is None:
            py_rng = _random_module.Random()
        if np_rng is None:
            # Keep numpy-side sampling deterministic with the caller-provided
            # python RNG so the same dataset index maps to the same sample.
            np_rng = np.random.default_rng(py_rng.randint(0, 2**32 - 1))

        has_tracks = bool(result.metadata.get("has_tracks", False))
        T = len(result.images)
        cw, ch = result.crop.crop_w, result.crop.crop_h
        depths_np: Optional[np.ndarray] = None
        if result.depths is not None:
            depths_np = np.stack(result.depths, axis=0).astype(np.float32, copy=False)

        sample_metadata = dict(result.metadata)
        sample_metadata["has_temporal_supervision"] = has_tracks
        sample_metadata["query_semantics"] = (
            "full_temporal" if has_tracks else "static_reconstruction"
        )
        aspect_ratio_value = _native_equivalent_aspect_ratio(result)

        # ---- 转 tensor（video / depth / normal） ----
        video = torch.stack(
            [torch.from_numpy(img).permute(2, 0, 1) for img in result.images], dim=0
        )  # [T,3,H,W]

        depths_t: Optional[torch.Tensor] = None
        if self.store_auxiliary_tensors and depths_np is not None:
            depths_t = torch.from_numpy(depths_np[:, None])  # [T,1,H,W]

        normals_t: Optional[torch.Tensor] = None
        if self.store_auxiliary_tensors and result.normals is not None:
            normals_t = torch.stack(
                [torch.from_numpy(n).permute(2, 0, 1) for n in result.normals], dim=0
            )  # [T,3,H,W]

        # Build highres_video from cropped frames only when it is needed.  In
        # precomputed_resized mode we must not carry high-res crops through the
        # planned spool because they can dominate sample build and disk IO.
        highres_video: Optional[torch.Tensor] = None
        need_highres_video = (
            self.return_highres_video
            or (self.precompute_patches and self.precompute_from_highres)
        )
        if need_highres_video:
            if not getattr(result, "cropped_images", None):
                raise ValueError(
                    "High-res crops are required by the configured patch provider, "
                    "but TransformResult.cropped_images is empty. Set "
                    "keep_cropped_images=True or use a resized patch provider."
                )
            ch_actual = result.cropped_images[0].shape[0]
            cw_actual = result.cropped_images[0].shape[1]
            for i, img in enumerate(result.cropped_images):
                if img.shape[0] != ch_actual or img.shape[1] != cw_actual:
                    raise ValueError(
                        f"Frame {i} has inconsistent crop size {img.shape[:2]} vs expected "
                        f"({ch_actual}, {cw_actual})."
                    )
            if ch_actual != result.img_size or cw_actual != result.img_size:
                # Store high-res crops compactly in the planned spool.  They are
                # converted back to floating point in model._prepare_query_frames.
                # This avoids writing hundreds of MB of float32 video per sample.
                highres_video = torch.stack(
                    [
                        torch.from_numpy(
                            np.rint(np.clip(img * 255.0, 0.0, 255.0)).astype(np.uint8)
                        ).permute(2, 0, 1)
                        for img in result.cropped_images
                    ],
                    dim=0,
                )

        # ---- boundary masks（在 crop 分辨率下计算） ----
        depth_boundary: list[np.ndarray] = []
        if depths_np is not None:
            # depths 已是 resize 后的图；在 img_size 分辨率下算边界
            depth_boundary = [_depth_boundary_mask(d) for d in depths_np]
        else:
            S = result.img_size
            depth_boundary = [np.zeros((S, S), dtype=bool) for _ in range(T)]

        motion_boundary: list[np.ndarray] = []
        # frames_time_ordered=False means the clip's frame_indices were permuted
        # (shuffle_prob), so array-adjacent frames are no longer time-adjacent and
        # neighbour-frame motion boundaries would compute spurious huge motion.
        # Skip them for this clip only — ordered clips in the same dataset still
        # get the supervision.
        if has_tracks and self.use_motion_boundaries and frames_time_ordered and result.trajs_2d is not None:
            # trajs_2d 在 crop 坐标系，motion boundary 也在 crop 坐标系
            # → resize 到 img_size 以和 depth_boundary 对齐
            S = result.img_size
            valid_mb = (
                result.valids.astype(bool)
                if result.valids is not None
                else np.ones(result.trajs_2d.shape[:2], dtype=bool)
            )
            if self.motion_boundary_on_resized:
                scale = np.array(
                    [S / max(float(cw), 1.0), S / max(float(ch), 1.0)],
                    dtype=np.float32,
                )
                trajs_for_mb = result.trajs_2d.astype(np.float32, copy=False) * scale
                motion_boundary = _motion_boundary_masks_sequence(
                    trajs_for_mb,
                    valid_mb,
                    crop_h=S,
                    crop_w=S,
                )
            else:
                motion_boundary_crop = _motion_boundary_masks_sequence(
                    result.trajs_2d,
                    valid_mb,
                    crop_h=ch,
                    crop_w=cw,
                )
                motion_boundary = [
                    cv2.resize(
                        mb_crop.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    for mb_crop in motion_boundary_crop
                ]
        else:
            S = result.img_size
            motion_boundary = [np.zeros((S, S), dtype=bool) for _ in range(T)]

        depth_boundary_stack = np.stack(depth_boundary, axis=0)
        motion_boundary_stack = np.stack(motion_boundary, axis=0)
        boundary_stack = np.logical_or(depth_boundary_stack, motion_boundary_stack)

        # ---- 采样 queries ----
        if has_tracks:
            try:
                coords, t_src, t_tgt, t_cam, targets = self._sample_with_tracks(
                    result,
                    depth_boundary_stack,
                    motion_boundary_stack,
                    boundary_stack,
                    depths_np,
                    py_rng,
                    np_rng,
                )
            except RuntimeError:
                if not self.allow_track_fallback:
                    raise
                coords, t_src, t_tgt, t_cam, targets = self._sample_no_tracks(
                    result, depth_boundary_stack, boundary_stack, depths_np, py_rng, np_rng
                )
                sample_metadata["has_temporal_supervision"] = False
                sample_metadata["query_semantics"] = "tracks_missing_fallback"
        else:
            coords, t_src, t_tgt, t_cam, targets = self._sample_no_tracks(
                result, depth_boundary_stack, boundary_stack, depths_np, py_rng, np_rng
            )

        # ---- local patches ----
        local_patches: Optional[torch.Tensor] = None
        if self.precompute_patches:
            patch_queries = torch.cat(
                [coords, t_src.unsqueeze(-1).float()], dim=-1
            )  # [Q,3]
            if self.precompute_from_highres and highres_video is not None:
                highres_hwc = highres_video.permute(0, 2, 3, 1)  # [T,H,W,3]
                local_patches = self._extract_patches(highres_hwc, patch_queries)
                highres_video = None
            else:
                video_hwc = video.permute(0, 2, 3, 1)   # [T,H,W,3]
                local_patches = self._extract_patches(video_hwc, patch_queries)
            if self.precompute_from_highres:
                highres_video = None
            if self.store_video_uint8 and local_patches is not None:
                local_patches = (
                    torch.clamp(local_patches * 255.0, 0.0, 255.0)
                    .round()
                    .to(torch.uint8)
                )

        crop = result.crop
        transform_metadata = {
            "canonical_space": torch.tensor(0, dtype=torch.long),
            "original_hw": torch.tensor(
                [float(result.original_h), float(result.original_w)], dtype=torch.float32
            ),
            "crop_offset_xy": torch.tensor(
                [float(crop.x0), float(crop.y0)], dtype=torch.float32
            ),
            "crop_size_hw": torch.tensor(
                [float(crop.crop_h), float(crop.crop_w)], dtype=torch.float32
            ),
            "resized_hw": torch.tensor(
                [float(result.img_size), float(result.img_size)], dtype=torch.float32
            ),
        }

        if self.store_video_uint8:
            video_out = torch.clamp(video * 255.0, 0.0, 255.0).round().to(torch.uint8)
        else:
            video_out = video

        return QuerySample(
            video=video_out,
            highres_video=highres_video,
            depths=depths_t,
            normals=normals_t,
            coords=coords,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
            intrinsics=torch.from_numpy(result.intrinsics),
            extrinsics=torch.from_numpy(result.extrinsics),
            targets=targets,
            local_patches=local_patches,
            transform_metadata=transform_metadata,
            aspect_ratio=torch.tensor(
                [aspect_ratio_value],
                dtype=torch.float32,
            ),
            dataset_name=result.dataset_name,
            sequence_name=result.sequence_name,
            metadata=sample_metadata,
        )

    # ------------------------------------------------------------------
    # has_tracks=True 路径（PointOdyssey / Kubric / DynamicReplica）
    # ------------------------------------------------------------------

    def _sample_with_tracks(
        self,
        result: TransformResult,
        depth_boundary_stack: np.ndarray,
        motion_boundary_stack: np.ndarray,
        boundary_stack: np.ndarray,
        depths_np: Optional[np.ndarray],
        py_rng: _random_module.Random,
        np_rng: np.random.Generator,
    ) -> tuple:
        trajs_2d    = result.trajs_2d       # [T,N,2]  crop coords
        trajs_3d    = result.trajs_3d_world # [T,N,3]
        trajs_2d_f  = trajs_2d.astype(np.float32, copy=False)
        trajs_3d_f  = trajs_3d.astype(np.float32, copy=False)
        valids      = np.asarray(result.valids) > 0.5  # [T,N]
        visibs      = np.asarray(result.visibs) > 0.5  # [T,N]
        extrinsics  = np.asarray(result.extrinsics)     # [T,4,4]
        intrinsics  = np.asarray(result.intrinsics)     # [T,3,3]
        T, N        = trajs_2d.shape[:2]
        cw, ch      = result.crop.crop_w, result.crop.crop_h
        S           = result.img_size
        z_max       = self._get_z_max(result.dataset_name)

        targets = _build_empty_targets(self.num_queries)

        # ---- 批量预计算每帧有效 source 点（向量化替代 T 帧循环） ----
        valid_by_frame: list[np.ndarray] = []
        boundary_by_frame: list[np.ndarray] = []
        source_depth_by_frame: list[np.ndarray] = []
        source_depth_sampling = self._get_source_depth_sampling(result.dataset_name)

        scale_xy = np.array([S / cw, S / ch], dtype=np.float32)
        pts_finite = np.isfinite(trajs_2d_f).all(-1)
        in_bounds_all = (
            (trajs_2d_f[..., 0] >= 0) & (trajs_2d_f[..., 0] < cw) &
            (trajs_2d_f[..., 1] >= 0) & (trajs_2d_f[..., 1] < ch)
        )
        world_finite_all = np.isfinite(trajs_3d_f).all(-1)

        cam_coords_src = (
            np.einsum("tij,tnj->tni", extrinsics[:, :3, :3], trajs_3d_f)
            + extrinsics[:, None, :3, 3]
        )
        depth_z_all = cam_coords_src[..., 2]
        depth_valid_all = (
            (depth_z_all > 1e-3) &
            (depth_z_all < z_max) &
            np.isfinite(depth_z_all)
        )

        depth_map_valid_all = np.ones((T, N), dtype=bool)
        if depths_np is not None:
            pts_depth_ok = pts_finite & in_bounds_all
            d_px = np.zeros((T, N), dtype=np.float32)
            ok_flat = pts_depth_ok.ravel()
            if ok_flat.any():
                flat_ids = np.flatnonzero(ok_flat)
                pts_img_ok = (trajs_2d_f.reshape(-1, 2)[flat_ids] * scale_xy).astype(np.float32, copy=False)
                xi = np.clip(np.round(pts_img_ok[:, 0]), 0, S - 1).astype(np.int32)
                yi = np.clip(np.round(pts_img_ok[:, 1]), 0, S - 1).astype(np.int32)
                frame_ids = flat_ids // N
                d_px.ravel()[flat_ids] = depths_np[frame_ids, yi, xi]
            depth_map_valid_all = pts_depth_ok & (d_px > 1e-3) & np.isfinite(d_px)

        K_fx = intrinsics[:, 0, 0][:, None]
        K_fy = intrinsics[:, 1, 1][:, None]
        K_cx = intrinsics[:, 0, 2][:, None]
        K_cy = intrinsics[:, 1, 2][:, None]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            inv_z = 1.0 / (depth_z_all + 1e-8)
            proj_x = cam_coords_src[..., 0] * inv_z * K_fx + K_cx
            proj_y = cam_coords_src[..., 1] * inv_z * K_fy + K_cy
            reproj_err = np.sqrt(
                (proj_x - trajs_2d_f[..., 0] * scale_xy[0]) ** 2 +
                (proj_y - trajs_2d_f[..., 1] * scale_xy[1]) ** 2
            )
        reproj_valid_all = (reproj_err < 3.0) | ~depth_valid_all
        valid_src_all = (
            valids &
            visibs &
            in_bounds_all &
            world_finite_all &
            depth_valid_all &
            depth_map_valid_all &
            reproj_valid_all
        )

        # 批量 camera transform: [T,4,4] @ [T,N,4] → [T,N,3]
        ones = np.ones((T, N, 1), dtype=np.float32)
        world_h = np.concatenate([trajs_3d.astype(np.float32), ones], axis=-1)  # [T,N,4]
        all_cam = np.einsum('tij,tnj->tni', extrinsics, world_h)[:, :, :3]     # [T,N,3]
        depth_z = all_cam[:, :, 2]                                               # [T,N]
        depth_valid = (depth_z > 1e-3) & (depth_z < z_max) & np.isfinite(depth_z)

        # 批量 in-bounds / world-finite
        in_bounds = (
            (trajs_2d[:, :, 0] >= 0) & (trajs_2d[:, :, 0] < cw) &
            (trajs_2d[:, :, 1] >= 0) & (trajs_2d[:, :, 1] < ch)
        )  # [T,N]
        world_finite = np.isfinite(trajs_3d).all(-1)  # [T,N]

        # 批量 depth-map validity
        sx_d, sy_d = S / cw, S / ch
        depth_map_valid = np.ones((T, N), dtype=bool)
        if result.depths is not None:
            pts_finite = np.isfinite(trajs_2d).all(-1)  # [T,N]
            pts_dok = pts_finite & in_bounds
            d_px = np.zeros((T, N), dtype=np.float32)
            if pts_dok.any():
                pts_img_all = trajs_2d[pts_dok] * np.array([sx_d, sy_d], dtype=np.float32)
                xi_all = np.clip(np.round(pts_img_all[:, 0]), 0, S - 1).astype(np.int32)
                yi_all = np.clip(np.round(pts_img_all[:, 1]), 0, S - 1).astype(np.int32)
                depths_arr = np.stack(result.depths, axis=0)  # [T,S,S]
                fi_idx, ni_idx = np.where(pts_dok)
                d_px[pts_dok] = depths_arr[fi_idx, yi_all, xi_all]
            depth_map_valid = pts_dok & (d_px > 1e-3) & np.isfinite(d_px)

        # 批量 reprojection error
        K_fx = result.intrinsics[:, 0, 0]  # [T]
        K_cx = result.intrinsics[:, 0, 2]  # [T]
        K_fy = result.intrinsics[:, 1, 1]  # [T]
        K_cy = result.intrinsics[:, 1, 2]  # [T]
        with np.errstate(over='ignore'):
            proj_x = all_cam[:, :, 0] / (depth_z + 1e-8) * K_fx[:, None] + K_cx[:, None]
            proj_y = all_cam[:, :, 1] / (depth_z + 1e-8) * K_fy[:, None] + K_cy[:, None]
            reproj_err = np.sqrt(
                (proj_x - trajs_2d[:, :, 0] * sx_d) ** 2 +
                (proj_y - trajs_2d[:, :, 1] * sy_d) ** 2
            )  # [T,N]
        reproj_valid = (reproj_err < 3.0) | ~depth_valid

        # 合并 validity mask
        all_valid = (
            (valids > 0.5) & (visibs > 0.5) & in_bounds
            & world_finite & depth_valid & depth_map_valid & reproj_valid
        )  # [T,N]

        for fi in range(T):
            pts = trajs_2d_f[fi]                                  # [N,2]
            source_depth_by_frame.append(depth_z_all[fi].astype(np.float32, copy=False))
            valid_src = np.flatnonzero(valid_src_all[fi])
            valid_by_frame.append(valid_src)

            if len(valid_src) == 0 or self.boundary_ratio <= 0:
                boundary_by_frame.append(np.empty(0, dtype=np.int64))
                continue

            # boundary mask は img_size 解像度、trajs は crop 解像度
            # → trajs を img_size へスケール
            xy_img = np.round(pts[valid_src] * scale_xy).astype(np.int32)
            xy_img[:, 0] = np.clip(xy_img[:, 0], 0, S - 1)
            xy_img[:, 1] = np.clip(xy_img[:, 1], 0, S - 1)
            on_boundary = boundary_stack[fi][xy_img[:, 1], xy_img[:, 0]]
            boundary_by_frame.append(valid_src[on_boundary])

        valid_frames = [fi for fi, v in enumerate(valid_by_frame) if len(v) > 0]
        if not valid_frames:
            raise RuntimeError(
                f"[{result.dataset_name}/{result.sequence_name}] "
                "No valid source points after transform."
            )

        # ---- 向量化采样（替换 num_queries 次 Python 循环） ----
        N = self.num_queries
        valid_frames_arr = np.array(valid_frames)

        # 1. 批量随机帧采样
        # PAPER-ALIGNED: t_src/t_tgt/t_cam sampled independently at random.
        # Enforce t_tgt=t_cam with prob t_tgt_eq_t_cam_ratio (paper: 0.4).
        # Previously forced t_tgt=t_src=t_cam for 40% of queries, which
        # collapsed tracking queries into depth-only self-reprojection.
        src_frames = np_rng.choice(valid_frames_arr, size=N)
        tgt_frames = np_rng.integers(0, T, size=N)
        cam_frames = np_rng.integers(0, T, size=N)
        force_eq   = np_rng.random(size=N) < self.t_tgt_eq_t_cam_ratio
        cam_frames = np.where(force_eq, tgt_frames, cam_frames)
        use_bnd    = np_rng.random(size=N) < self.boundary_ratio

        # 2. 按 source frame 分组选点（~S 次迭代而非 N 次）
        pt_indices = np.zeros(N, dtype=np.int64)
        for sf in np.unique(src_frames):
            mask   = src_frames == sf
            q_idxs = np.where(mask)[0]
            bnd_c  = boundary_by_frame[sf]
            val_c  = valid_by_frame[sf]
            wants  = use_bnd[q_idxs] & (len(bnd_c) > 0)
            if wants.any():
                if source_depth_sampling is not None:
                    pt_indices[q_idxs[wants]] = _depth_balanced_choice(
                        bnd_c,
                        source_depth_by_frame[sf][bnd_c],
                        int(wants.sum()),
                        np_rng,
                        source_depth_sampling,
                    )
                else:
                    pt_indices[q_idxs[wants]] = np_rng.choice(bnd_c, size=int(wants.sum()))
            if (~wants).any():
                if source_depth_sampling is not None:
                    pt_indices[q_idxs[~wants]] = _depth_balanced_choice(
                        val_c,
                        source_depth_by_frame[sf][val_c],
                        int((~wants).sum()),
                        np_rng,
                        source_depth_sampling,
                    )
                else:
                    pt_indices[q_idxs[~wants]] = np_rng.choice(val_c, size=int((~wants).sum()))

        # 3. 批量 fancy indexing
        src_xy = trajs_2d[src_frames, pt_indices]   # [N,2]
        tgt_xy = trajs_2d[tgt_frames, pt_indices]   # [N,2]
        src_w3 = trajs_3d[src_frames, pt_indices]   # [N,3]
        tgt_w3 = trajs_3d[tgt_frames, pt_indices]   # [N,3]

        # 4. einsum 批量相机变换
        E_batch  = extrinsics[cam_frames]                                        # [N,4,4]
        R_batch  = E_batch[:, :3, :3]
        t_batch  = E_batch[:, :3, 3]
        src_cam  = np.einsum('nij,nj->ni', R_batch, src_w3.astype(np.float32, copy=False)) + t_batch
        tgt_cam  = np.einsum('nij,nj->ni', R_batch, tgt_w3.astype(np.float32, copy=False)) + t_batch

        # 5. 向量化 validity / boundary 查表
        tgt_defined  = valids[tgt_frames, pt_indices]
        tgt_inbounds = (
            (tgt_xy[:, 0] >= 0) & (tgt_xy[:, 0] < cw) &
            (tgt_xy[:, 1] >= 0) & (tgt_xy[:, 1] < ch)
        )
        vis_flag     = tgt_defined & visibs[tgt_frames, pt_indices] & tgt_inbounds

        # 仅确保点在相机前方且深度量级合理，上限按数据集查表
        # （见 _DATASET_Z_MAX，可由 dataset_z_max 覆盖）。
        src_depth_valid = (src_cam[:, 2] > 1e-3) & (src_cam[:, 2] < z_max) & np.isfinite(src_cam[:, 2])
        tgt_depth_valid = (tgt_cam[:, 2] > 1e-3) & (tgt_cam[:, 2] < z_max) & np.isfinite(tgt_cam[:, 2])
        has_valid_3d = (tgt_defined & tgt_inbounds
                       & np.isfinite(src_cam).all(-1) & np.isfinite(tgt_cam).all(-1)
                       & src_depth_valid & tgt_depth_valid)

        sx, sy  = S / cw, S / ch
        src_ix  = np.clip(np.round(src_xy[:, 0] * sx).astype(np.int32), 0, S - 1)
        src_iy  = np.clip(np.round(src_xy[:, 1] * sy).astype(np.int32), 0, S - 1)

        src_is_bnd  = boundary_stack[src_frames, src_iy, src_ix]
        dbnd_stack  = depth_boundary_stack
        mbnd_stack  = motion_boundary_stack
        src_is_dbnd = dbnd_stack[src_frames, src_iy, src_ix]
        src_is_mbnd = mbnd_stack[src_frames, src_iy, src_ix]

        # 6. 一次性写入输出 tensor
        src_xy_norm = _normalize_crop_coords(src_xy, crop_w=cw, crop_h=ch)
        tgt_xy_norm = _normalize_crop_coords(tgt_xy, crop_w=cw, crop_h=ch)
        coords  = torch.from_numpy(src_xy_norm.astype(np.float32))
        t_src_t = torch.from_numpy(src_frames.astype(np.int64))
        t_tgt_t = torch.from_numpy(tgt_frames.astype(np.int64))
        t_cam_t = torch.from_numpy(cam_frames.astype(np.int64))

        targets["pos_2d"][:]                     = torch.from_numpy(tgt_xy_norm.astype(np.float32))
        targets["pos_3d"][:]                     = torch.from_numpy(tgt_cam.astype(np.float32))
        targets["visibility"][:]                 = torch.from_numpy(vis_flag.astype(np.float32))
        targets["displacement"][:]               = torch.from_numpy((tgt_cam - src_cam).astype(np.float32))
        targets["mask_3d"][:]                    = torch.from_numpy(has_valid_3d)
        targets["mask_2d"][:]                    = torch.from_numpy(vis_flag)
        # mask_vis gates the visibility BCE loss. Co3Dv2 keeps 2D/3D tracks but
        # always sets has_visibility=False because its sparse point-cloud tracks
        # do not provide reliable occlusion supervision.
        has_visibility_supervision = bool(result.metadata.get("has_visibility", True))
        mask_vis_arr = tgt_defined if has_visibility_supervision else np.zeros_like(tgt_defined)
        targets["mask_vis"][:]                   = torch.from_numpy(mask_vis_arr)
        targets["mask_disp"][:]                  = torch.from_numpy(has_valid_3d)
        targets["source_is_boundary"][:]         = torch.from_numpy(src_is_bnd)
        targets["source_is_depth_boundary"][:]   = torch.from_numpy(src_is_dbnd)
        targets["source_is_motion_boundary"][:] = torch.from_numpy(src_is_mbnd)
        targets["point_indices"][:]              = torch.from_numpy(pt_indices)
        # is_static_reprojection: marks queries with no temporal component
        # (t_tgt == t_src). In the paper-aligned sampling this arises naturally
        # when t_tgt happens to equal t_src, not from a forced 40% split.
        targets["is_static_reprojection"][:]     = torch.from_numpy(tgt_frames == src_frames)

        # ---- normal（向量化批量查表） ----
        # 只有 t_cam == t_tgt 时，normal 才与目标 3D 监督处于同一相机坐标系。
        normal_supervision_compatible = bool(
            result.metadata.get("normal_supervision_compatible", result.normals is not None)
        )
        if normal_supervision_compatible and result.normals is not None and result.normal_valids is not None:
            normals_arr = np.stack(result.normals, axis=0)             # [T,S,S,3]
            valid_nf    = np.array(result.normal_valids, dtype=bool)   # [T]
            cam_eq_tgt  = cam_frames == tgt_frames
            active      = vis_flag & valid_nf[tgt_frames] & cam_eq_tgt & has_valid_3d
            if active.any():
                qi  = np.where(active)[0]
                uv  = tgt_xy_norm[qi]                                  # [M,2]
                nx  = np.clip(np.round(uv[:, 0] * (S - 1)).astype(np.int32), 0, S - 1)
                ny  = np.clip(np.round(uv[:, 1] * (S - 1)).astype(np.int32), 0, S - 1)
                nv  = normals_arr[tgt_frames[qi], ny, nx]              # [M,3]
                nv_t = torch.from_numpy(nv.astype(np.float32))
                valid_normal = torch.isfinite(nv_t).all(-1) & (torch.linalg.norm(nv_t, dim=-1) > 1e-6)
                targets["normal"][qi] = torch.where(valid_normal.unsqueeze(-1), nv_t, torch.zeros_like(nv_t))
                targets["mask_normal"][qi] = valid_normal

        return coords, t_src_t, t_tgt_t, t_cam_t, targets

    # ------------------------------------------------------------------
    # has_tracks=False 路径（ScanNet / Co3D / BlendedMVS 等）
    # 只有 depth + pose → 单帧静态假设下产出 pos_3d
    # mask_2d / mask_disp / mask_vis 全部 False
    # ------------------------------------------------------------------

    def _sample_no_tracks(
        self,
        result: TransformResult,
        depth_boundary_stack: np.ndarray,
        boundary_stack: np.ndarray,
        depths_np: Optional[np.ndarray],
        py_rng: _random_module.Random,
        np_rng: np.random.Generator,
    ) -> tuple:
        T  = len(result.images)
        S  = result.img_size
        # Keep no-tracks coordinates in the same crop-normalized convention as
        # the tracks path instead of resized-image coordinates.
        cw = result.crop.crop_w if result.crop is not None else S
        ch = result.crop.crop_h if result.crop is not None else S
        K  = result.intrinsics   # [T,3,3]
        E  = result.extrinsics   # [T,4,4]
        z_max = self._get_z_max(result.dataset_name)

        targets = _build_empty_targets(self.num_queries)

        has_depth = depths_np is not None

        # 预计算每帧有效（有深度）的像素池
        valid_pixels_by_frame: list[Optional[np.ndarray]] = []
        boundary_pixels_by_frame: list[np.ndarray] = []

        # depth / boundary live on the resized SxS plane; coords live on the
        # crop plane. We convert between them explicitly to keep supervision
        # semantics aligned with the tracks path.
        scale_x = cw / max(S, 1)
        scale_y = ch / max(S, 1)

        # Store flat resized indices; materialize crop coords only for sampled points.
        def _flat_to_crop_xy(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            ys_r, xs_r = np.divmod(flat, S)
            xs_c = np.clip(np.rint(xs_r.astype(np.float32) * scale_x), 0, cw - 1).astype(np.int32)
            ys_c = np.clip(np.rint(ys_r.astype(np.float32) * scale_y), 0, ch - 1).astype(np.int32)
            return xs_c, ys_c

        for fi in range(T):
            if not has_depth:
                # 没有 depth：在 crop 平面直接随机采像素，mask_3d 全 False
                valid_pixels_by_frame.append(None)
                boundary_pixels_by_frame.append(np.empty(0, dtype=np.int32))
                continue

            depth = depths_np[fi]           # [S,S] resized plane
            valid_mask = np.isfinite(depth) & (depth > 1e-3) & (depth < z_max)
            flat = np.flatnonzero(valid_mask.ravel()).astype(np.int32, copy=False)
            if len(flat) == 0:
                valid_pixels_by_frame.append(None)
                boundary_pixels_by_frame.append(np.empty(0, dtype=np.int32))
                continue

            valid_pixels_by_frame.append(flat)

            if self.boundary_ratio > 0:
                bpx_mask = depth_boundary_stack[fi].ravel()[flat]
                boundary_pixels_by_frame.append(flat[bpx_mask])
            else:
                boundary_pixels_by_frame.append(np.empty(0, dtype=np.int32))

        # ---- 向量化采样（替换 num_queries 次 Python 循环） ----
        N = self.num_queries

        # PAPER-ALIGNED: t_src/t_tgt/t_cam sampled independently at random.
        # Enforce t_tgt=t_cam with prob t_tgt_eq_t_cam_ratio (paper: 0.4).
        src_frames = np_rng.integers(0, T, size=N, dtype=np.int64)
        tgt_frames = np_rng.integers(0, T, size=N, dtype=np.int64)
        cam_frames = np_rng.integers(0, T, size=N, dtype=np.int64)
        force_eq = np_rng.random(size=N) < self.t_tgt_eq_t_cam_ratio
        cam_frames = np.where(force_eq, tgt_frames, cam_frames).astype(np.int64, copy=False)
        use_bnd = np_rng.random(size=N) < self.boundary_ratio

        chosen_px = np.empty((N, 2), dtype=np.int32)
        for sf in np.unique(src_frames):
            q_idxs = np.where(src_frames == sf)[0]
            bpx = boundary_pixels_by_frame[int(sf)]
            wants_bnd = use_bnd[q_idxs] & (len(bpx) > 0)

            if wants_bnd.any():
                out_idx = q_idxs[wants_bnd]
                flat = bpx[np_rng.integers(0, len(bpx), size=len(out_idx))]
                chosen_px[out_idx, 0], chosen_px[out_idx, 1] = _flat_to_crop_xy(flat)

            if (~wants_bnd).any():
                out_idx = q_idxs[~wants_bnd]
                vpx = valid_pixels_by_frame[int(sf)]
                if vpx is None:
                    chosen_px[out_idx, 0] = np_rng.integers(0, cw, size=len(out_idx))
                    chosen_px[out_idx, 1] = np_rng.integers(0, ch, size=len(out_idx))
                else:
                    flat = vpx[np_rng.integers(0, len(vpx), size=len(out_idx))]
                    chosen_px[out_idx, 0], chosen_px[out_idx, 1] = _flat_to_crop_xy(flat)

        px_x = chosen_px[:, 0]
        px_y = chosen_px[:, 1]
        coords_np = _normalize_crop_coords(chosen_px.astype(np.float32), crop_w=cw, crop_h=ch)

        coords = torch.from_numpy(coords_np.astype(np.float32, copy=False))
        t_src_t = torch.from_numpy(src_frames.astype(np.int64, copy=False))
        t_tgt_t = torch.from_numpy(tgt_frames.astype(np.int64, copy=False))
        t_cam_t = torch.from_numpy(cam_frames.astype(np.int64, copy=False))

        # boundary/depth_boundary live on resized plane; remap crop coord.
        px_x_r = np.clip(np.rint(px_x.astype(np.float32) / scale_x), 0, S - 1).astype(np.int32)
        px_y_r = np.clip(np.rint(px_y.astype(np.float32) / scale_y), 0, S - 1).astype(np.int32)
        targets["source_is_boundary"][:] = torch.from_numpy(boundary_stack[src_frames, px_y_r, px_x_r])
        targets["source_is_depth_boundary"][:] = torch.from_numpy(depth_boundary_stack[src_frames, px_y_r, px_x_r])
        targets["point_indices"][:] = torch.from_numpy(
            px_y.astype(np.int64) * int(cw) + px_x.astype(np.int64)
        )
        # is_static_reprojection: marks queries with no temporal component
        # (t_tgt == t_src), used for loss weighting analysis.
        targets["is_static_reprojection"][:] = torch.from_numpy(tgt_frames == src_frames)

        if has_depth and depths_np is not None:
            K_np = np.asarray(K, dtype=np.float32)
            E_np = np.asarray(E, dtype=np.float32)
            E_inv = np.linalg.inv(E_np).astype(np.float32, copy=False)
            src_to_cam = np.einsum("tij,sjk->tsik", E_np, E_inv).astype(np.float32, copy=False)

            depth_vals = depths_np[src_frames, px_y_r, px_x_r].astype(np.float32, copy=False)
            K_src = K_np[src_frames]
            fx = K_src[:, 0, 0]
            fy = K_src[:, 1, 1]
            cx = K_src[:, 0, 2]
            cy = K_src[:, 1, 2]
            finite_intr = (
                np.isfinite(fx) & np.isfinite(fy) &
                (np.abs(fx) > 1e-8) & (np.abs(fy) > 1e-8)
            )
            depth_valid = (
                np.isfinite(depth_vals) &
                (depth_vals > 1e-3) &
                (depth_vals < z_max) &
                finite_intr
            )

            p_src_cam = np.ones((N, 4), dtype=np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                p_src_cam[:, 0] = (px_x_r.astype(np.float32) - cx) * depth_vals / fx
                p_src_cam[:, 1] = (px_y_r.astype(np.float32) - cy) * depth_vals / fy
            p_src_cam[:, 2] = depth_vals

            p_cam_h = np.einsum("nij,nj->ni", src_to_cam[cam_frames, src_frames], p_src_cam)
            p_cam = p_cam_h[:, :3].astype(np.float32, copy=False)

            # Project point to t_tgt image plane to check in-frame visibility.
            p_tgt_h = np.einsum("nij,nj->ni", src_to_cam[tgt_frames, src_frames], p_src_cam)
            p_tgt = p_tgt_h[:, :3]
            K_tgt_all = K_np[tgt_frames]
            with np.errstate(divide="ignore", invalid="ignore"):
                u_tgt = K_tgt_all[:, 0, 0] * p_tgt[:, 0] / p_tgt[:, 2] + K_tgt_all[:, 0, 2]
                v_tgt = K_tgt_all[:, 1, 1] * p_tgt[:, 1] / p_tgt[:, 2] + K_tgt_all[:, 1, 2]
            tgt_inbounds = (
                np.isfinite(u_tgt) & np.isfinite(v_tgt) &
                (u_tgt * scale_x >= 0.0) & (u_tgt * scale_x < cw) &
                (v_tgt * scale_y >= 0.0) & (v_tgt * scale_y < ch) &
                (p_tgt[:, 2] > 1e-3)
            )
            has_valid_3d = (
                depth_valid &
                np.isfinite(p_cam).all(axis=1) &
                (p_cam[:, 2] > 1e-3) &
                (p_cam[:, 2] < z_max) &
                tgt_inbounds
            )
            valid_q = np.where(has_valid_3d)[0]
            if valid_q.size:
                targets["pos_3d"][valid_q] = torch.from_numpy(p_cam[valid_q])
            targets["mask_3d"][:] = torch.from_numpy(has_valid_3d)

            # pos_2d: only when t_tgt=t_cam (paper design).
            active_2d = has_valid_3d & (tgt_frames == cam_frames)
            if active_2d.any():
                qi = np.where(active_2d)[0]
                p2 = p_cam[qi]
                K_tgt = K_np[tgt_frames[qi]]
                with np.errstate(divide="ignore", invalid="ignore"):
                    u = K_tgt[:, 0, 0] * p2[:, 0] / p2[:, 2] + K_tgt[:, 0, 2]
                    v = K_tgt[:, 1, 1] * p2[:, 1] / p2[:, 2] + K_tgt[:, 1, 2]
                u_c = u * scale_x
                v_c = v * scale_y
                in_bounds = (
                    np.isfinite(u_c) & np.isfinite(v_c) &
                    (u_c >= 0.0) & (u_c < cw) &
                    (v_c >= 0.0) & (v_c < ch)
                )
                if in_bounds.any():
                    out_idx = qi[in_bounds]
                    uv_norm = _normalize_crop_coords(
                        np.stack([u_c[in_bounds], v_c[in_bounds]], axis=1).astype(np.float32),
                        crop_w=cw,
                        crop_h=ch,
                    )
                    targets["pos_2d"][out_idx] = torch.from_numpy(uv_norm)
                    targets["mask_2d"][out_idx] = True

        return coords, t_src_t, t_tgt_t, t_cam_t, targets

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _extract_patches(
        self,
        video_hwc: torch.Tensor,    # [T,H,W,3]
        patch_queries: torch.Tensor, # [Q,3] (u,v,t_src)  u/v in [0,1]
    ) -> torch.Tensor:
        frames_btchw = video_hwc.permute(0, 3, 1, 2).unsqueeze(0)  # [1,T,3,H,W]
        if frames_btchw.dtype == torch.uint8:
            frames_btchw = frames_btchw.float().div_(255.0)
        else:
            frames_btchw = frames_btchw.float()
        coords_uv    = patch_queries[:, :2].unsqueeze(0)             # [1,Q,2]
        t_src_idx    = patch_queries[:, 2].round().long().unsqueeze(0)  # [1,Q]
        patches = extract_local_patches(
            frames_btchw, coords_uv, t_src_idx, patch_size=self.patch_size
        )
        return patches.squeeze(0)   # [Q,3,P,P]
