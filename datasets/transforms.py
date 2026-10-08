"""
Geometry-consistent clip transform pipeline for D4RT training.

Applies random crop, resize, intrinsics update, trajs_2d update,
visibility recompute, and color augmentation to a UnifiedClip.
All spatial operations are kept in sync so that pos_2d / pos_3d
supervision remains geometrically consistent after transforms.

Usage:
    pipeline = GeometryTransformPipeline(img_size=256, use_augs=True)
    clip = pipeline(clip, rng=random_state)
    # clip.images are now list of [256,256,3] float32 in [0,1]
    # clip.intrinsics are updated to match the resized view
"""

from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from datasets.adapters.base import UnifiedClip


# ---------------------------------------------------------------------------
# Low-level geometry helpers (pure functions, no side effects)
# ---------------------------------------------------------------------------

def _crop_intrinsics(K: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """Shift principal point after cropping. K: [...,3,3]."""
    out = K.copy().astype(np.float32)
    out[..., 0, 2] -= x0
    out[..., 1, 2] -= y0
    return out


def _resize_intrinsics(K: np.ndarray, src_h: int, src_w: int, dst_h: int, dst_w: int) -> np.ndarray:
    """Scale focal length and principal point after resizing. K: [...,3,3]."""
    out = K.copy().astype(np.float32)
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    out[..., 0, 0] *= sx
    out[..., 1, 1] *= sy
    out[..., 0, 2] *= sx
    out[..., 1, 2] *= sy
    return out


def _crop_trajs_2d(trajs: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """Shift 2D trajectories after cropping. trajs: [T,N,2] (x,y)."""
    out = trajs.copy().astype(np.float32)
    out[..., 0] -= x0
    out[..., 1] -= y0
    return out


def _compute_inbounds_mask(trajs: np.ndarray, crop_w: int, crop_h: int) -> np.ndarray:
    """Boolean mask [T,N]: True where point falls inside [0,W) x [0,H)."""
    return (
        (trajs[..., 0] >= 0.0)
        & (trajs[..., 0] < float(crop_w))
        & (trajs[..., 1] >= 0.0)
        & (trajs[..., 1] < float(crop_h))
    )


def _resize_image(img: np.ndarray, h: int, w: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=interp)


def _resize_depth(depth: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


def _resize_normal(normal: np.ndarray, h: int, w: int) -> np.ndarray:
    resized = cv2.resize(normal, (w, h), interpolation=cv2.INTER_LINEAR)
    valid = np.linalg.norm(normal, axis=-1) > 1e-6
    valid_resized = cv2.resize(
        valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ) > 0
    # re-normalise after bilinear interpolation to keep unit length
    norm = np.linalg.norm(resized, axis=-1, keepdims=True)
    safe = valid_resized[..., None] & (norm > 1e-6) & np.isfinite(resized).all(axis=-1, keepdims=True)
    resized = np.where(safe, resized / np.where(safe, norm, 1.0), 0.0)
    return resized.astype(np.float32, copy=False)


def _to_float01_image(img: np.ndarray) -> np.ndarray:
    """Convert RGB images to float32 in [0, 1] before color augmentation."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0

    out = img.astype(np.float32, copy=False)
    if out.size == 0:
        return out
    if out.max() > 1.0 or out.min() < 0.0:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

@dataclass
class CropParams:
    x0: int
    y0: int
    crop_w: int
    crop_h: int


def _sample_random_crop(height: int, width: int, rng: _random_module.Random) -> CropParams:
    """
    Random crop matching the original dataset.py logic:
    - area ratio uniformly in [0.3, 1.0]
    - aspect ratio log-uniformly in [3/4, 4/3]
    - 5% chance of additional zoom-in crop
    """
    area = height * width
    min_ratio = 3.0 / 4.0
    max_ratio = 4.0 / 3.0

    x0, y0, crop_w, crop_h = 0, 0, width, height
    for _ in range(10):
        target_area = rng.uniform(0.3, 1.0) * area
        aspect = math.exp(rng.uniform(math.log(min_ratio), math.log(max_ratio)))
        cw = int(round(math.sqrt(target_area * aspect)))
        ch = int(round(math.sqrt(target_area / aspect)))
        if 0 < cw <= width and 0 < ch <= height:
            x0 = rng.randint(0, width - cw)
            y0 = rng.randint(0, height - ch)
            crop_w, crop_h = cw, ch
            break

    if rng.random() < 0.05:
        zoom = rng.uniform(0.7, 0.95)
        zw = max(1, int(round(crop_w * zoom)))
        zh = max(1, int(round(crop_h * zoom)))
        x0 = x0 + max(0, (crop_w - zw) // 2)
        y0 = y0 + max(0, (crop_h - zh) // 2)
        crop_w, crop_h = zw, zh

    return CropParams(x0=x0, y0=y0, crop_w=crop_w, crop_h=crop_h)


def _use_full_frame(height: int, width: int) -> CropParams:
    """Keep the full input view before square resize (paper-consistent eval path)."""
    return CropParams(x0=0, y0=0, crop_w=width, crop_h=height)


# ---------------------------------------------------------------------------
# Color augmentation
# ---------------------------------------------------------------------------

def _apply_color_aug(
    rgb_frames: list[np.ndarray],
    rng: _random_module.Random,
) -> list[np.ndarray]:
    """
    Per-clip colour augmentation (same parameters for every frame in the clip):
    brightness, contrast, saturation, hue jitter, optional grayscale, optional blur.
    Input/output: list of [H,W,3] float32 in [0,1].
    Uses cv2 per-frame (SIMD) — do NOT batch with numpy, it is 8x slower.
    """
    brightness  = rng.uniform(0.6, 1.4)
    contrast    = rng.uniform(0.6, 1.4)
    saturation  = rng.uniform(0.6, 1.4)
    hue_delta   = int(rng.uniform(-0.1, 0.1) * 180.0)
    apply_gray  = rng.random() < 0.2
    apply_blur  = rng.random() < 0.4
    blur_sigma  = rng.uniform(0.1, 2.0)

    out = []
    for frame in rgb_frames:
        f = np.clip(frame * brightness, 0.0, 1.0)
        mean = f.mean()
        f = np.clip((f - mean) * contrast + mean, 0.0, 1.0)

        hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * saturation, 0, 255).astype(np.uint8)
        hsv[..., 0] = (hsv[..., 0].astype(np.int32) + hue_delta) % 180
        f = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        if apply_gray:
            gray = (f * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(-1, keepdims=True)
            f = np.repeat(gray, 3, axis=-1)
        if apply_blur:
            f = cv2.GaussianBlur(f, (0, 0), blur_sigma)
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Transform result
# ---------------------------------------------------------------------------

@dataclass
class TransformResult:
    """
    Output of GeometryTransformPipeline.

    All image-space quantities are in the *resized* (img_size x img_size) frame.
    trajs_2d (if present) remains in the *crop* coordinate frame (pixel units,
    before the final resize) because the Query Builder needs crop-space coords
    to compute in-bounds masks and boundary samples.  The Query Builder is
    responsible for normalising coords to [0,1] using crop_w / crop_h.
    """
    # Resized frames: list[np.ndarray] [H,W,3] float32 [0,1]
    images: list[np.ndarray]
    # Cropped frames before resize: list[np.ndarray] [crop_h, crop_w, 3] float32 [0,1]
    # Used by QueryBuilder for high-resolution patch extraction.
    cropped_images: list[np.ndarray]
    # Resized depth maps: list[np.ndarray] [H,W] float32 | None
    depths: Optional[list[np.ndarray]]
    # Resized normal maps: list[np.ndarray] [H,W,3] float32 | None
    normals: Optional[list[np.ndarray]]
    # Per-frame normal validity: list[bool] | None
    normal_valids: Optional[list[bool]]

    # 2D trajectories in *crop* pixel coordinates: [T,N,2] | None
    trajs_2d: Optional[np.ndarray]
    # 3D world trajectories (unchanged): [T,N,3] | None
    trajs_3d_world: Optional[np.ndarray]
    # Valid mask (& in-bounds): [T,N] bool | None
    valids: Optional[np.ndarray]
    # Visibility mask (& in-bounds): [T,N] bool | None
    visibs: Optional[np.ndarray]

    # Camera parameters in *resized* image space
    intrinsics: np.ndarray          # [T,3,3]
    extrinsics: np.ndarray          # [T,4,4]

    # Crop and resize parameters (needed by Query Builder for coord normalisation)
    crop: CropParams
    img_size: int

    # Original image size before crop
    original_h: int
    original_w: int

    # Passthrough from adapter
    dataset_name: str
    sequence_name: str
    metadata: dict


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class GeometryTransformPipeline:
    """
    Stateless transform pipeline.  Accepts a UnifiedClip and returns a
    TransformResult with geometry-consistent crop + resize applied.

    Args:
        img_size:   Target spatial resolution (square).
        use_augs:   If True, apply random crop and colour augmentation.
                    If False, keep the full frame and only resize to square.
    """

    def __init__(
        self,
        img_size: int = 256,
        use_augs: bool = True,
        *,
        keep_cropped_images: bool = True,
        color_aug_after_resize: bool = False,
        max_track_points: Optional[int] = None,
    ) -> None:
        self.img_size = img_size
        self.use_augs = use_augs
        self.keep_cropped_images = bool(keep_cropped_images)
        self.color_aug_after_resize = bool(color_aug_after_resize)
        if max_track_points is None:
            self.max_track_points = None
        else:
            value = int(max_track_points)
            self.max_track_points = value if value > 0 else None

    def __call__(
        self,
        clip: UnifiedClip,
        rng: Optional[_random_module.Random] = None,
    ) -> TransformResult:
        if rng is None:
            rng = _random_module.Random()

        h, w = clip.image_size
        clip_trajs_2d = clip.trajs_2d
        clip_trajs_3d_world = clip.trajs_3d_world
        clip_valids = clip.valids
        clip_visibs = clip.visibs
        if (
            self.max_track_points is not None
            and clip_trajs_2d is not None
            and clip_trajs_2d.shape[1] > self.max_track_points
        ):
            seed = rng.randint(0, 2**32 - 1)
            np_rng = np.random.default_rng(seed)
            keep_idx = np.sort(
                np_rng.choice(
                    clip_trajs_2d.shape[1],
                    size=self.max_track_points,
                    replace=False,
                ).astype(np.int64)
            )
            clip_trajs_2d = clip_trajs_2d[:, keep_idx]
            if clip_trajs_3d_world is not None:
                clip_trajs_3d_world = clip_trajs_3d_world[:, keep_idx]
            if clip_valids is not None:
                clip_valids = clip_valids[:, keep_idx]
            if clip_visibs is not None:
                clip_visibs = clip_visibs[:, keep_idx]

        # ------------------------------------------------------------------ #
        # 1. Spatial crop                                                      #
        # ------------------------------------------------------------------ #
        if self.use_augs:
            crop = _sample_random_crop(h, w, rng)
        else:
            crop = _use_full_frame(h, w)

        x0, y0 = crop.x0, crop.y0
        cw, ch = crop.crop_w, crop.crop_h

        # ------------------------------------------------------------------ #
        # 2. Crop images / depth / normals                                     #
        # ------------------------------------------------------------------ #
        # High-res RGB crops are only needed for high-res patch providers, or
        # when colour augmentation must run before resize.  In resized-patch
        # mode with post-resize colour aug, resize the uint8 crop first and
        # avoid materialising 48 float32 high-res frames.
        need_float_crops = self.keep_cropped_images or (
            self.use_augs and not self.color_aug_after_resize
        )
        if need_float_crops:
            cropped_images = [
                _to_float01_image(img[y0:y0 + ch, x0:x0 + cw])
                for img in clip.images
            ]
        else:
            cropped_images = []

        cropped_depths: Optional[list[np.ndarray]] = None
        if clip.depths is not None:
            cropped_depths = [d[y0:y0 + ch, x0:x0 + cw] for d in clip.depths]

        cropped_normals: Optional[list[np.ndarray]] = None
        if clip.normals is not None:
            cropped_normals = [n[y0:y0 + ch, x0:x0 + cw] for n in clip.normals]

        # ------------------------------------------------------------------ #
        # 3. Colour augmentation (RGB only, no depth/normal aug)              #
        # ------------------------------------------------------------------ #
        if self.use_augs and not self.color_aug_after_resize:
            cropped_images = _apply_color_aug(cropped_images, rng)

        # ------------------------------------------------------------------ #
        # 4. Resize to img_size × img_size                                    #
        # ------------------------------------------------------------------ #
        S = self.img_size
        if cropped_images:
            resized_images = [_resize_image(img, S, S) for img in cropped_images]
        else:
            resized_images = [
                _to_float01_image(_resize_image(img[y0:y0 + ch, x0:x0 + cw], S, S))
                for img in clip.images
            ]
        if self.use_augs and self.color_aug_after_resize:
            resized_images = _apply_color_aug(resized_images, rng)

        resized_depths: Optional[list[np.ndarray]] = None
        if cropped_depths is not None:
            resized_depths = [_resize_depth(d, S, S) for d in cropped_depths]

        resized_normals: Optional[list[np.ndarray]] = None
        normal_valids: Optional[list[bool]] = None
        if cropped_normals is not None:
            resized_normals = [_resize_normal(n, S, S) for n in cropped_normals]
            # A normal frame is valid if it has at least one finite non-zero vector.
            normal_valids = [
                bool(np.isfinite(n).all() and np.any(np.linalg.norm(n, axis=-1) > 1e-6))
                for n in resized_normals
            ]

        # ------------------------------------------------------------------ #
        # 5. Update intrinsics: crop → resize                                 #
        # ------------------------------------------------------------------ #
        K_crop   = _crop_intrinsics(clip.intrinsics, x0=float(x0), y0=float(y0))
        K_resize = _resize_intrinsics(K_crop, src_h=ch, src_w=cw, dst_h=S, dst_w=S)

        # ------------------------------------------------------------------ #
        # 6. Update 2D trajectories + recompute in-bounds visibility          #
        # ------------------------------------------------------------------ #
        trajs_2d_crop: Optional[np.ndarray] = None
        valids_new:    Optional[np.ndarray] = None
        visibs_new:    Optional[np.ndarray] = None

        if clip_trajs_2d is not None:
            trajs_2d_crop = _crop_trajs_2d(clip_trajs_2d, x0=float(x0), y0=float(y0))
            inbounds = _compute_inbounds_mask(trajs_2d_crop, crop_w=cw, crop_h=ch)
            keep_out_of_view_valids = bool(
                clip.metadata.get("keep_out_of_view_valids", False)
                or clip.metadata.get("visibility_validity_mode") == "defined"
            )

            if clip_valids is not None:
                valids_new = clip_valids if keep_out_of_view_valids else (clip_valids & inbounds)
            if clip_visibs is not None:
                visibs_new = clip_visibs & inbounds

        return TransformResult(
            images=resized_images,
            cropped_images=cropped_images if self.keep_cropped_images else [],
            depths=resized_depths,
            normals=resized_normals,
            normal_valids=normal_valids,
            trajs_2d=trajs_2d_crop,
            trajs_3d_world=clip_trajs_3d_world,   # world coords are view-independent
            valids=valids_new,
            visibs=visibs_new,
            intrinsics=K_resize,
            extrinsics=clip.extrinsics,
            crop=crop,
            img_size=S,
            original_h=h,
            original_w=w,
            dataset_name=clip.dataset_name,
            sequence_name=clip.sequence_name,
            metadata=dict(clip.metadata),
        )
