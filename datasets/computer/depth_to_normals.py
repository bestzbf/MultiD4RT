"""
Compute surface normal maps from depth maps and camera intrinsics.

All outputs are in camera coordinate space (OpenCV convention: +x right, +y down, +z forward).
Normals point toward the camera (negative Z component in camera space).

Usage:
    from computer.depth_to_normals import compute_normals

    normal = compute_normals(depth, K)   # [H,W,3] float32
"""

from __future__ import annotations

import numpy as np


def compute_normals(
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Compute a surface normal map from a depth map and camera intrinsics.

    Args:
        depth: [H,W] float32. Invalid pixels are 0, nan, or inf.
        K:     [3,3] float32. Standard pinhole intrinsics.

    Returns:
        normal: [H,W,3] float32 in camera space, unit vectors.
                Invalid pixels (no valid depth or neighbours) are [0,0,0].
                Normals point toward the camera (Z < 0 in camera space).
    """
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)

    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # ------------------------------------------------------------------ #
    # 1. Valid mask and safe depth (set invalid to 0 for array ops)        #
    # ------------------------------------------------------------------ #
    valid = (depth > 0) & np.isfinite(depth)
    d = np.where(valid, depth, 0.0)

    # ------------------------------------------------------------------ #
    # 2. Unproject every pixel to 3D camera space                          #
    #    pts[y, x] = (X, Y, Z) in camera coordinates                      #
    # ------------------------------------------------------------------ #
    us = np.arange(W, dtype=np.float32)          # [W]
    vs = np.arange(H, dtype=np.float32)          # [H]
    uu, vv = np.meshgrid(us, vs)                  # [H,W] each

    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy
    Z = d
    pts = np.stack([X, Y, Z], axis=-1)            # [H,W,3]

    # zero out invalid pixels so they don't pollute finite-diff neighbours
    pts[~valid] = 0.0

    # ------------------------------------------------------------------ #
    # 3. Compute finite differences with boundary handling                 #
    #    du = d(pts)/d(col),  dv = d(pts)/d(row)                          #
    # ------------------------------------------------------------------ #
    # valid_left[y,x]  = valid[y, x-1]
    # valid_right[y,x] = valid[y, x+1]
    # valid_up[y,x]    = valid[y-1, x]
    # valid_down[y,x]  = valid[y+1, x]

    valid_l = np.zeros((H, W), dtype=bool)
    valid_r = np.zeros((H, W), dtype=bool)
    valid_u = np.zeros((H, W), dtype=bool)
    valid_d = np.zeros((H, W), dtype=bool)

    valid_l[:, 1:]  = valid[:, :-1]
    valid_r[:, :-1] = valid[:, 1:]
    valid_u[1:, :]  = valid[:-1, :]
    valid_d[:-1, :] = valid[1:, :]

    # Horizontal difference (column direction)
    # Use central diff where both neighbours valid, else one-sided
    du = np.zeros((H, W, 3), dtype=np.float32)
    have_central_h = valid_l & valid_r
    have_right_only = (~valid_l) & valid_r & valid
    have_left_only  = valid_l & (~valid_r) & valid

    pts_r = np.zeros_like(pts); pts_r[:, :-1] = pts[:, 1:]
    pts_l = np.zeros_like(pts); pts_l[:, 1:]  = pts[:, :-1]

    du[have_central_h] = (pts_r - pts_l)[have_central_h]
    du[have_right_only] = (pts_r - pts)[have_right_only]
    du[have_left_only]  = (pts - pts_l)[have_left_only]

    # Vertical difference (row direction)
    dv = np.zeros((H, W, 3), dtype=np.float32)
    have_central_v = valid_u & valid_d
    have_down_only = (~valid_u) & valid_d & valid
    have_up_only   = valid_u & (~valid_d) & valid

    pts_d = np.zeros_like(pts); pts_d[:-1, :] = pts[1:, :]
    pts_u = np.zeros_like(pts); pts_u[1:, :]  = pts[:-1, :]

    dv[have_central_v] = (pts_d - pts_u)[have_central_v]
    dv[have_down_only]  = (pts_d - pts)[have_down_only]
    dv[have_up_only]    = (pts - pts_u)[have_up_only]

    # ------------------------------------------------------------------ #
    # 4. Cross product: normal = du × dv                                   #
    # ------------------------------------------------------------------ #
    normal = np.cross(du, dv)   # [H,W,3]

    # ------------------------------------------------------------------ #
    # 5. Normalise to unit length                                          #
    # ------------------------------------------------------------------ #
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)   # [H,W,1]
    valid_normal = (norm[..., 0] > 1e-6) & valid
    # Use masked division so invalid / zero-norm pixels stay zero without
    # emitting RuntimeWarning from the inactive branch.
    normal = np.divide(
        normal,
        norm,
        out=np.zeros_like(normal, dtype=np.float32),
        where=norm > 1e-6,
    ).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # 6. Ensure normals point toward camera (Z < 0 in camera space)        #
    # ------------------------------------------------------------------ #
    flip = valid_normal & (normal[..., 2] > 0)
    normal[flip] *= -1.0

    # Zero out pixels where we could not compute a valid normal
    normal[~valid_normal] = 0.0

    return normal


def compute_normals_sequence(
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
) -> np.ndarray:
    """
    Compute normals for a sequence of depth maps.

    Args:
        depths:     List of T depth maps, each [H,W] float32.
        intrinsics: [T,3,3] or [3,3] float32.

    Returns:
        normals: [T,H,W,3] float32.
    """
    T = len(depths)
    K_seq = np.asarray(intrinsics, dtype=np.float32)
    if K_seq.ndim == 2:
        K_seq = np.broadcast_to(K_seq[None], (T, 3, 3))

    result = []
    for t in range(T):
        result.append(compute_normals(depths[t], K_seq[t]))
    return np.stack(result, axis=0)   # [T,H,W,3]
