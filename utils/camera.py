"""Camera utilities for D4RT: pose estimation, projection, and intrinsics recovery."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import pytorch3d for optimized Umeyama
try:
    from pytorch3d.ops import corresponding_points_alignment
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False


def umeyama_alignment(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    with_scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Umeyama algorithm for finding optimal rigid transformation.

    Finds rotation R, translation t, and scale s that minimize:
        sum_i ||target_i - (s * R @ source_i + t)||^2

    Reference: Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns", TPAMI 1991.

    Uses pytorch3d's optimized implementation when available.

    Args:
        source: (N, 3) source points
        target: (N, 3) target points
        weights: (N,) optional point weights
        with_scale: Whether to estimate scale

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar scale factor
    """
    if PYTORCH3D_AVAILABLE:
        return _umeyama_pytorch3d(source, target, weights, with_scale)
    else:
        return _umeyama_native(source, target, weights, with_scale)


def _umeyama_pytorch3d(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    with_scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Umeyama using pytorch3d's corresponding_points_alignment.

    Convention: target = s * R @ source + t

    PyTorch3D's corresponding_points_alignment uses the same convention as our
    native implementation, so results are directly compatible.
    """
    # pytorch3d expects (B, N, 3) format
    source_batch = source.unsqueeze(0)  # (1, N, 3)
    target_batch = target.unsqueeze(0)  # (1, N, 3)

    if weights is not None:
        weights_batch = weights.unsqueeze(0)  # (1, N)
    else:
        weights_batch = None

    # corresponding_points_alignment returns SimilarityTransform namedtuple
    # with R (rotation), T (translation), s (scale)
    # Convention: X_target = s * R @ X_source + T
    result = corresponding_points_alignment(
        source_batch,
        target_batch,
        weights_batch,
        estimate_scale=with_scale,
        allow_reflection=False
    )

    # Extract results (remove batch dimension)
    R = result.R.squeeze(0)  # (3, 3)
    t = result.T.squeeze(0)  # (3,)
    s = result.s.squeeze(0) if with_scale else torch.ones(1, device=source.device)

    return R, t, s


def _umeyama_native(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    with_scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native PyTorch Umeyama implementation (fallback).

    Convention: target = s * R @ source + t

    This implements the standard Umeyama algorithm (TPAMI 1991) using column vectors.
    """
    assert source.shape == target.shape
    N = source.shape[0]

    if weights is None:
        weights = torch.ones(N, device=source.device, dtype=source.dtype)

    # Normalize weights
    weights = weights / weights.sum()

    # Compute weighted centroids
    mu_source = (weights.unsqueeze(-1) * source).sum(dim=0)
    mu_target = (weights.unsqueeze(-1) * target).sum(dim=0)

    # Center the points
    source_centered = source - mu_source
    target_centered = target - mu_target

    # Compute weighted covariance matrix
    # H = sum_i w_i * target_centered_i @ source_centered_i^T
    H = (weights.unsqueeze(-1).unsqueeze(-1) *
         target_centered.unsqueeze(-1) @ source_centered.unsqueeze(-2)).sum(dim=0)

    # SVD decomposition
    U, S, Vh = torch.linalg.svd(H)

    # Rotation matrix
    R = U @ Vh

    # Handle reflection (ensure proper rotation)
    if torch.det(R.cpu()) < 0:
        Vh[-1] *= -1
        S[-1] *= -1
        R = U @ Vh

    # Compute scale
    if with_scale:
        # Variance of source points
        var_source = (weights * (source_centered ** 2).sum(dim=-1)).sum()
        s = S.sum() / var_source
    else:
        s = torch.ones(1, device=source.device, dtype=source.dtype)

    # Compute translation: t = mu_target - s * R @ mu_source
    t = mu_target - s * R @ mu_source

    return R, t, s


def umeyama_alignment_batched(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    with_scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Umeyama alignment for multiple point sets.

    Args:
        source: (B, N, 3) source points
        target: (B, N, 3) target points
        weights: (B, N) optional point weights
        with_scale: Whether to estimate scale

    Returns:
        R: (B, 3, 3) rotation matrices
        t: (B, 3) translation vectors
        s: (B,) scale factors
    """
    if PYTORCH3D_AVAILABLE:
        result = corresponding_points_alignment(
            source,
            target,
            weights,
            estimate_scale=with_scale,
            allow_reflection=False
        )
        return result.R, result.T, result.s
    else:
        # Fallback: process batch sequentially
        B = source.shape[0]
        Rs, ts, ss = [], [], []
        for i in range(B):
            w = weights[i] if weights is not None else None
            R, t, s = _umeyama_native(source[i], target[i], w, with_scale)
            Rs.append(R)
            ts.append(t)
            ss.append(s)
        return torch.stack(Rs), torch.stack(ts), torch.stack(ss)


def estimate_camera_pose(
    model,
    encoder_features: torch.Tensor,
    frames: torch.Tensor,
    frame_i: int,
    frame_j: int,
    grid_size: Tuple[int, int] = (8, 8),
    sample_margin: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate relative camera pose between two frames.

    Uses D4RT decoder to predict 3D points in both coordinate systems,
    then applies Umeyama algorithm to find the rigid transformation.

    Args:
        model: D4RT model (or decoder)
        encoder_features: (1, N, C) encoded video features
        frames: (1, T, C, H, W) video frames
        frame_i: First frame index
        frame_j: Second frame index
        grid_size: (h, w) grid for sampling points
        sample_margin: Margin from edges (0.0=full image, 0.1=avoid 10% edges)

    Returns:
        R: (3, 3) rotation from frame_i to frame_j
        t: (3,) translation from frame_i to frame_j
    """
    device = encoder_features.device
    h, w = grid_size

    # Create grid of query points (configurable edge margin)
    u = torch.linspace(sample_margin, 1.0 - sample_margin, w, device=device)
    v = torch.linspace(sample_margin, 1.0 - sample_margin, h, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
    coords = torch.stack([grid_u, grid_v], dim=-1).reshape(1, -1, 2)  # (1, h*w, 2)

    n_points = h * w

    # Query points in frame_i coordinate system
    # q_i = (u, v, t_src=i, t_tgt=i, t_cam=i)
    t_src_i = torch.full((1, n_points), frame_i, device=device, dtype=torch.long)
    outputs_i = model.decode(
        encoder_features, frames, coords,
        t_src_i, t_src_i, t_src_i
    )
    points_i = outputs_i['pos_3d'].squeeze(0)  # (h*w, 3)
    conf_i = outputs_i['confidence_weight'].squeeze()  # (h*w,) - exp(-s), true confidence

    # Query same points in frame_j coordinate system
    # q_j = (u, v, t_src=i, t_tgt=i, t_cam=j)
    t_cam_j = torch.full((1, n_points), frame_j, device=device, dtype=torch.long)
    outputs_j = model.decode(
        encoder_features, frames, coords,
        t_src_i, t_src_i, t_cam_j
    )
    points_j = outputs_j['pos_3d'].squeeze(0)  # (h*w, 3)
    conf_j = outputs_j['confidence_weight'].squeeze()  # (h*w,) - exp(-s), true confidence

    # Combine confidences as weights
    weights = conf_i * conf_j

    # Filter low-confidence points
    valid_mask = weights > 0.5
    if valid_mask.sum() < 4:
        # Not enough valid points, return identity
        return torch.eye(3, device=device), torch.zeros(3, device=device)

    points_i_valid = points_i[valid_mask]
    points_j_valid = points_j[valid_mask]
    weights_valid = weights[valid_mask]

    # Rigid alignment (paper main text uses rigid body transform, not Sim(3)).
    # points_i = R @ points_j + t
    R, t, _ = umeyama_alignment(points_j_valid, points_i_valid, weights_valid, with_scale=False)

    # Invert to get transformation from frame_i to frame_j
    R_inv = R.T
    t_inv = -R_inv @ t

    return R_inv, t_inv


def estimate_intrinsics(
    model,
    encoder_features: torch.Tensor,
    frames: torch.Tensor,
    frame_idx: int,
    grid_size: Tuple[int, int] = (8, 8),
    principal_point: Tuple[float, float] = (0.5, 0.5),
    sample_margin: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate camera intrinsics from predicted 3D points.

    Assumes pinhole camera model with given principal point.

    Args:
        model: D4RT model
        encoder_features: (1, N, C) encoded features
        frames: (1, T, C, H, W) video frames
        frame_idx: Frame index to estimate intrinsics for
        grid_size: (h, w) grid for sampling points
        principal_point: (cx, cy) normalized principal point
        sample_margin: Margin from edges (0.0=full image, 0.1=avoid 10% edges)

    Returns:
        fx: Focal length in x direction (normalized)
        fy: Focal length in y direction (normalized)
    """
    device = encoder_features.device
    h, w = grid_size
    cx, cy = principal_point

    # Create grid of query points (configurable edge margin)
    u = torch.linspace(sample_margin, 1.0 - sample_margin, w, device=device)
    v = torch.linspace(sample_margin, 1.0 - sample_margin, h, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
    coords = torch.stack([grid_u, grid_v], dim=-1).reshape(1, -1, 2)

    n_points = h * w
    t_idx = torch.full((1, n_points), frame_idx, device=device, dtype=torch.long)

    # Decode depth-style queries (t_src = t_tgt = t_cam)
    outputs = model.decode(
        encoder_features, frames, coords,
        t_idx, t_idx, t_idx
    )
    points_3d = outputs['pos_3d'].squeeze(0)  # (h*w, 3)
    confidence = outputs['confidence_weight'].squeeze()  # (h*w,) - exp(-s), true confidence

    # Extract coordinates
    px, py, pz = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    u_coords = coords.squeeze(0)[:, 0]
    v_coords = coords.squeeze(0)[:, 1]

    # Compute focal lengths
    # u = fx * px / pz + cx  =>  fx = pz * (u - cx) / px
    # v = fy * py / pz + cy  =>  fy = pz * (v - cy) / py

    # Filter valid points (non-zero depth, sufficient parallax)
    valid = (pz.abs() > 1e-6) & (px.abs() > 1e-6) & (py.abs() > 1e-6)
    valid = valid & (confidence > 0.5)

    if valid.sum() < 4:
        # Not enough valid points, return default
        return torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)

    fx_estimates = pz[valid] * (u_coords[valid] - cx) / px[valid]
    fy_estimates = pz[valid] * (v_coords[valid] - cy) / py[valid]

    # Robust estimation using median
    fx = fx_estimates.median()
    fy = fy_estimates.median()

    return fx, fy


def project_points(
    points_3d: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: (..., 3) 3D points in world coordinates
        intrinsics: (3, 3) camera intrinsics matrix
        extrinsics: (4, 4) optional camera extrinsics (world to camera)

    Returns:
        points_2d: (..., 2) projected 2D coordinates
    """
    # Apply extrinsics if provided
    if extrinsics is not None:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points_3d = points_3d @ R.T + t

    # Project
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2].clamp(min=1e-6)

    u = fx * x / z + cx
    v = fy * y / z + cy

    return torch.stack([u, v], dim=-1)


def unproject_points(
    points_2d: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Unproject 2D points to 3D using depth.

    Args:
        points_2d: (..., 2) 2D image coordinates
        depth: (...,) depth values
        intrinsics: (3, 3) camera intrinsics matrix
        extrinsics: (4, 4) optional camera extrinsics (camera to world)

    Returns:
        points_3d: (..., 3) 3D points in world coordinates
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = points_2d[..., 0]
    v = points_2d[..., 1]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    points_3d = torch.stack([x, y, z], dim=-1)

    # Apply inverse extrinsics if provided
    if extrinsics is not None:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points_3d = (points_3d - t) @ R

    return points_3d


def compute_relative_pose_error(
    pred_R: torch.Tensor,
    pred_t: torch.Tensor,
    gt_R: torch.Tensor,
    gt_t: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Compute relative pose errors.

    Args:
        pred_R: (3, 3) predicted rotation
        pred_t: (3,) predicted translation
        gt_R: (3, 3) ground truth rotation
        gt_t: (3,) ground truth translation

    Returns:
        Dictionary with rotation and translation errors
    """
    # Rotation error (angle in degrees)
    R_diff = pred_R @ gt_R.T
    trace = R_diff.trace().clamp(-1, 3)
    angle_rad = torch.acos((trace - 1) / 2)
    rot_error = torch.rad2deg(angle_rad)

    # Translation error (normalized)
    # Normalize translations to unit length for direction comparison
    pred_t_norm = F.normalize(pred_t, dim=0)
    gt_t_norm = F.normalize(gt_t, dim=0)
    trans_error_angle = torch.acos((pred_t_norm * gt_t_norm).sum().clamp(-1, 1))
    trans_error = torch.rad2deg(trans_error_angle)

    # Scale error
    pred_scale = pred_t.norm()
    gt_scale = gt_t.norm()
    scale_error = torch.abs(pred_scale - gt_scale) / gt_scale.clamp(min=1e-6)

    return {
        'rotation_error': rot_error,
        'translation_error': trans_error,
        'scale_error': scale_error
    }


def sim3_alignment(
    pred_poses: torch.Tensor,
    gt_poses: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sim(3) alignment between predicted and ground truth trajectories.

    NOTE: This is an *eval-time* alignment utility (used for trajectory
    evaluation, e.g. ATE).  It is NOT the paper's pairwise camera pose solver.
    The paper main text uses a rigid body transform (SE(3)); see
    estimate_camera_pose() which calls umeyama_alignment with_scale=False.
    Sim(3) here is appropriate for long-video evaluation as noted in the
    appendix.

    Args:
        pred_poses: (N, 4, 4) predicted camera poses
        gt_poses: (N, 4, 4) ground truth camera poses

    Returns:
        R: (3, 3) rotation
        t: (3,) translation
        s: scale factor
    """
    # Extract camera centers
    pred_centers = pred_poses[:, :3, 3]  # (N, 3)
    gt_centers = gt_poses[:, :3, 3]  # (N, 3)

    # Umeyama alignment with scale (Sim(3)) — eval-time only
    R, t, s = umeyama_alignment(pred_centers, gt_centers, with_scale=True)

    return R, t, s
