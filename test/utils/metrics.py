"""Evaluation metrics and protocol helpers used by D4RT.

This module intentionally keeps the formulas explicit and lightweight so that
standalone evaluators can reuse the same protocol logic without depending on
training-only code paths.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


DEPTH_METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
TRACKING_DEFAULT_THRESHOLDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def _nan_metric_dict(metric_keys: Iterable[str], reference: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
    """Create a metric dictionary filled with NaNs on a sensible device/dtype."""
    kwargs: dict[str, object] = {}
    if reference is not None:
        kwargs["device"] = reference.device
        kwargs["dtype"] = reference.dtype
    return {key: torch.tensor(float("nan"), **kwargs) for key in metric_keys}


def _safe_acos_from_rotation_delta(rotation_delta: torch.Tensor) -> torch.Tensor:
    """Convert a relative rotation matrix into an angle in radians.

    The input may contain minor floating-point drift, so the cosine term is
    clamped into the valid ``[-1, 1]`` range before taking ``acos``.
    """

    trace = rotation_delta.trace().clamp(-1.0, 3.0)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cos_theta)


def _flatten_valid_depth_values(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten valid depth values into paired 1-D tensors."""

    if mask is None:
        mask = (target > 0) & torch.isfinite(target)
    return pred[mask], target[mask]


def _apply_depth_scale_only_alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Apply global scale-only depth alignment.

    This follows the robust single-scale alignment used by CUT3R's video-depth
    evaluator: initialize by mean depth and refine the scalar with a few
    iteratively reweighted least-squares steps for L1 residuals.
    """

    scale = (target.mean() / pred.mean().clamp(min=1e-6)).clamp(min=1e-3)
    for _ in range(10):
        residuals = (scale * pred - target).abs().clamp(min=1e-8)
        weights = 1.0 / residuals
        numerator = (weights * pred * target).sum()
        denominator = (weights * pred.square()).sum().clamp(min=1e-6)
        scale = (numerator / denominator).clamp(min=1e-3)
    return pred * scale


def _apply_depth_scale_and_shift_alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Apply global scale-and-shift depth alignment.

    MegaSaM's Sintel depth evaluator uses a robust median-centered scale and a
    median residual shift. That keeps one global ``s`` and ``t`` for the whole
    sequence without allowing per-frame fitting.
    """

    pred_median = pred.median()
    target_median = target.median()
    pred_centered = pred - pred_median + 1e-6
    target_centered = target - target_median + 1e-6

    finite = torch.isfinite(pred_centered) & torch.isfinite(target_centered) & (pred_centered.abs() > 1e-6)
    if not finite.any():
        return _apply_depth_scale_only_alignment(pred, target)
    scale = (target_centered[finite] / pred_centered[finite]).median()
    if not torch.isfinite(scale) or scale <= 0:
        return _apply_depth_scale_only_alignment(pred, target)
    shift = (target - scale * pred).median()
    return pred * scale + shift


def mean_shift_align_points(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-shift predicted points to the GT centroid.

    This mirrors the point-cloud protocol described in the paper: only a global
    translation is removed, with no rotation or scale alignment.
    """

    pred_mean = pred_points.mean(dim=0, keepdim=True)
    gt_mean = gt_points.mean(dim=0, keepdim=True)
    pred_aligned = pred_points - pred_mean + gt_mean
    return pred_aligned, gt_points


def paired_coordinate_l1(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """Compute paired coordinate-wise L1 averaged over all points and axes."""

    return torch.abs(pred_points - gt_points).mean()


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale_invariant: bool = True,
    shift_invariant: bool = False,
    max_depth: Optional[float] = None,
) -> dict[str, torch.Tensor]:
    """Compute paper-style depth metrics on valid depth pixels.

    Args:
        pred: Predicted depth maps.
        target: Ground-truth depth maps.
        mask: Optional validity mask. If omitted, valid GT pixels are used.
        scale_invariant: Apply global scale-only alignment.
        shift_invariant: Apply global scale-and-shift alignment. When enabled it
            takes precedence over ``scale_invariant``.
        max_depth: Optional upper bound for clamping prediction values before
            alignment so that outliers cannot skew the estimated scale/shift.

    Returns:
        A dictionary with ``abs_rel``, ``sq_rel``, ``rmse``, ``rmse_log``,
        ``a1``, ``a2``, and ``a3``.
    """

    pred_flat, target_flat = _flatten_valid_depth_values(pred, target, mask)
    if pred_flat.numel() == 0:
        return _nan_metric_dict(DEPTH_METRIC_KEYS, reference=pred)

    if max_depth is not None:
        pred_flat = pred_flat.clamp(min=1e-6, max=float(max_depth))

    if shift_invariant:
        pred_flat = _apply_depth_scale_and_shift_alignment(pred_flat, target_flat)
    elif scale_invariant:
        pred_flat = _apply_depth_scale_only_alignment(pred_flat, target_flat)

    pred_flat = pred_flat.clamp(min=1e-6)
    if max_depth is not None:
        pred_flat = pred_flat.clamp(max=float(max_depth))

    abs_diff = torch.abs(pred_flat - target_flat)
    abs_rel = (abs_diff / target_flat).mean()
    sq_rel = (abs_diff.square() / target_flat).mean()
    rmse = torch.sqrt(abs_diff.square().mean())

    log_diff = torch.abs(torch.log(pred_flat) - torch.log(target_flat))
    rmse_log = torch.sqrt(log_diff.square().mean())

    ratio = torch.max(pred_flat / target_flat, target_flat / pred_flat)
    a1 = (ratio < 1.25).float().mean()
    a2 = (ratio < 1.25**2).float().mean()
    a3 = (ratio < 1.25**3).float().mean()

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
    }


def _apply_sim3_alignment_to_poses(pred_poses: torch.Tensor, gt_poses: torch.Tensor) -> torch.Tensor:
    """Apply eval-time Sim(3) alignment to a pose trajectory."""

    from .camera import sim3_alignment

    aligned_poses = pred_poses.clone()
    rotation, translation, scale = sim3_alignment(pred_poses, gt_poses)

    aligned_poses[:, :3, :3] = rotation @ pred_poses[:, :3, :3]
    pred_centers = pred_poses[:, :3, 3]
    aligned_poses[:, :3, 3] = scale * (pred_centers @ rotation.T) + translation
    return aligned_poses


def compute_pose_metrics(
    pred_poses: torch.Tensor,
    gt_poses: torch.Tensor,
    align: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute trajectory-level pose metrics.

    ``align=True`` applies the same Sim(3) trajectory alignment used by the
    paper's reported ATE/RPE metrics. This alignment is purely for evaluation;
    it is not the pairwise pose solver used to estimate relative poses.
    """

    num_poses = pred_poses.shape[0]
    aligned_poses = _apply_sim3_alignment_to_poses(pred_poses, gt_poses) if align and num_poses > 1 else pred_poses

    pred_centers = aligned_poses[:, :3, 3]
    gt_centers = gt_poses[:, :3, 3]
    ate = torch.sqrt((pred_centers - gt_centers).square().sum(dim=-1)).mean()

    if num_poses <= 1:
        zero = pred_poses.new_tensor(0.0)
        return {"ate": ate, "rpe_trans": zero, "rpe_rot": zero}

    rpe_trans_values: list[torch.Tensor] = []
    rpe_rot_values: list[torch.Tensor] = []
    for index in range(num_poses - 1):
        gt_rel = torch.linalg.inv(gt_poses[index]) @ gt_poses[index + 1]
        pred_rel = torch.linalg.inv(aligned_poses[index]) @ aligned_poses[index + 1]

        rpe_trans_values.append(torch.norm(pred_rel[:3, 3] - gt_rel[:3, 3]))
        rotation_delta = pred_rel[:3, :3] @ gt_rel[:3, :3].T
        rpe_rot_values.append(torch.rad2deg(_safe_acos_from_rotation_delta(rotation_delta)))

    return {
        "ate": ate,
        "rpe_trans": torch.stack(rpe_trans_values).mean(),
        "rpe_rot": torch.stack(rpe_rot_values).mean(),
    }


def compute_tracking_metrics(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    visibility: torch.Tensor,
    pred_visibility: Optional[torch.Tensor] = None,
    thresholds: Optional[Sequence[float]] = None,
) -> dict[str, torch.Tensor]:
    """Compute approximate 3D tracking metrics.

    Warning:
        This is **not** the official TAPVid-3D implementation. It is only an
        internal approximation used for regression testing and quick ablations.
        Use the official TAPVid-3D code for paper-quality numbers.
    """

    thresholds = list(thresholds) if thresholds is not None else list(TRACKING_DEFAULT_THRESHOLDS)

    abs_errors = torch.abs(pred_tracks - gt_tracks)
    euclidean_errors = torch.norm(pred_tracks - gt_tracks, dim=-1)
    visible = visibility > 0.5

    l1 = abs_errors[visible].mean() if visible.any() else pred_tracks.new_tensor(0.0)

    apd_scores: list[torch.Tensor] = []
    visible_count = visible.sum().float()
    for threshold in thresholds:
        within_threshold = (euclidean_errors < threshold) & visible
        if visible.any():
            apd_scores.append(within_threshold.sum().float() / visible_count)
        else:
            apd_scores.append(pred_tracks.new_tensor(0.0))
    apd = torch.stack(apd_scores).mean()

    pos_accuracy = (euclidean_errors < 0.1) & visible
    aj_approx = pos_accuracy.sum().float() / visible_count.clamp(min=1.0)

    if pred_visibility is not None:
        pred_vis = pred_visibility > 0.5
        oa = (pred_vis == visible).float().mean()
    else:
        oa = pred_tracks.new_tensor(float("nan"))

    return {
        "l1": l1,
        "apd": apd,
        "aj_approx": aj_approx,
        "oa": oa,
    }


def compute_point_cloud_metrics(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    align: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute point-cloud reconstruction metrics.

    The paper reports mean-shift alignment followed by paired coordinate-wise L1.
    We also return a Chamfer distance for quick qualitative debugging.
    """

    if align:
        pred_aligned, gt_aligned = mean_shift_align_points(pred_points, gt_points)
    else:
        pred_aligned = pred_points
        gt_aligned = gt_points

    if pred_aligned.shape[0] == gt_aligned.shape[0]:
        l1 = paired_coordinate_l1(pred_aligned, gt_aligned)
    else:
        l1 = pred_points.new_tensor(float("nan"))

    dist_pred_to_gt = torch.cdist(pred_aligned, gt_aligned).min(dim=1)[0]
    dist_gt_to_pred = torch.cdist(gt_aligned, pred_aligned).min(dim=1)[0]
    chamfer = (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2

    return {"l1": l1, "chamfer": chamfer}


def compute_pose_auc(
    rotation_errors: torch.Tensor,
    translation_errors: torch.Tensor,
    threshold: float = 30.0,
) -> torch.Tensor:
    """Compute Pose AUC using the max of rotation and translation error."""

    combined_errors = torch.max(rotation_errors, translation_errors)
    if combined_errors.numel() == 0:
        return combined_errors.new_tensor(float("nan"))

    sorted_errors, _ = torch.sort(combined_errors)
    thresholds = torch.linspace(0, threshold, 100, device=sorted_errors.device, dtype=sorted_errors.dtype)

    accuracies = []
    for current_threshold in thresholds:
        accuracies.append((sorted_errors < current_threshold).float().mean())

    return torch.trapezoid(torch.stack(accuracies), thresholds) / threshold
