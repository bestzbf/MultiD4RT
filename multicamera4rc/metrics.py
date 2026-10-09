"""Confidence-independent measurements of geometry and native trajectories."""

import torch
from torch.nn import functional as F


def _resize_scalar(value, size, *, mask=False):
    if value.shape[-2:] == size:
        return value.bool() if mask else value
    leading = value.shape[:-2]
    image = value.reshape(-1, 1, *value.shape[-2:]).float()
    resized = (F.interpolate(image, size=size, mode="nearest") if mask
               else F.interpolate(image, size=size, mode="bilinear", align_corners=False))
    result = resized.reshape(*leading, *size)
    return result.bool() if mask else result.to(value.dtype)


def _valid_mask(target, supplied, *, vector=False):
    finite = torch.isfinite(target).all(-1) if vector else torch.isfinite(target)
    if supplied is None:
        return finite
    mask = supplied.to(device=target.device, dtype=torch.bool)
    if vector and mask.shape == target.shape:
        mask = mask.all(-1)
    try:
        return finite & torch.broadcast_to(mask, finite.shape)
    except RuntimeError as exc:
        raise ValueError(f"valid mask {tuple(mask.shape)} cannot cover target {tuple(finite.shape)}") from exc


METRICS_VERSION = 2


@torch.no_grad()
def rotation_error_degrees(predicted, target):
    """SO(3) distance, including small angles, independent of outer AMP scope."""
    with torch.autocast(device_type=predicted.device.type, enabled=False):
        relative = predicted.float().transpose(-2, -1) @ target.float()
        skew = torch.stack((relative[..., 2, 1]-relative[..., 1, 2],
                            relative[..., 0, 2]-relative[..., 2, 0],
                            relative[..., 1, 0]-relative[..., 0, 1]), -1)
        sine = torch.linalg.vector_norm(skew, dim=-1)/2
        cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2
        return torch.rad2deg(torch.atan2(sine, cosine))


@torch.no_grad()
def native_track_metrics(sampled, target):
    """Measure native particle tracks using existing bilinear samples.

    No second grid sampling pass, confidence weighting or zero-motion labels
    are introduced. Temporal metrics exclude the source-time identity query.
    """
    query_valid = sampled["query_valid"].bool()
    device = query_valid.device
    if "sparse_query_valid" in target:
        query_valid = query_valid & target["sparse_query_valid"].to(device).bool()
    times = sampled["query_time_idx"]
    temporal = sampled["source_time_idx"][..., None] != times[:, None]
    metrics = {}
    supervised = torch.zeros_like(query_valid)
    for name in ("trajectory", "displacement"):
        key = f"sparse_{name}"
        if key not in target:
            continue
        truth = target[key].to(device).float()
        mask = target.get(f"{key}_valid", target.get("sparse_trajectory_valid"))
        if mask is not None:
            mask = mask.to(device)
        valid = _valid_mask(truth, mask, vector=True) & query_valid[..., None]
        supervised |= valid.any(-1)
        if name == "trajectory":
            metrics["native_track_point_times"] = valid.sum().float()
            metrics["native_temporal_point_times"] = (valid & temporal).sum().float()
        if valid.any():
            residual = sampled[name].float()-truth
            metrics[f"native_{name}_epe_normalized"] = torch.linalg.vector_norm(residual[valid], dim=-1).mean()
            if (valid & temporal).any():
                metrics[f"native_{name}_epe_temporal_normalized"] = torch.linalg.vector_norm(residual[valid & temporal], dim=-1).mean()
        if name == "displacement" and times.shape[1] > 1:
            order = times.argsort(1)
            dt = times.gather(1, order).diff(dim=1).float()
            if (dt <= 0).any():
                raise ValueError("native velocity metrics require unique physical frame indices")
            index = order[:, None, :, None].expand_as(truth)
            residual = (sampled[name].float()-truth).gather(2, index)
            ordered_valid = valid.gather(2, index[..., 0])
            pairs = ordered_valid[..., 1:] & ordered_valid[..., :-1]
            if pairs.any():
                error = residual.diff(dim=2)/dt[:, None, :, None]
                metrics["native_velocity_epe_normalized"] = torch.linalg.vector_norm(error[pairs], dim=-1).mean()
    if "sparse_visibility" in target:
        truth = target["sparse_visibility"].to(device)
        mask = target.get("sparse_visibility_valid")
        valid = _valid_mask(truth, None if mask is None else mask.to(device))
        supervised |= valid.flatten(2).any(-1) & query_valid
    metrics["native_track_queries"] = supervised.sum().float()
    return metrics


@torch.no_grad()
def dense_metrics(prediction, target):
    metrics = {}

    def mean(value, mask):
        return value[mask].mean() if mask.any() else value.new_zeros(())

    valid = target["trajectory_valid"].bool()
    foreground = valid & target.get("trajectory_foreground", valid).bool()
    times = prediction["query_time_idx"]
    temporal = (prediction["source_time_idx"][:, :, None] != times[:, None])[..., None, None]
    metrics["supervised_source_pixels"] = target.get("source_pixel_valid", valid.any(2)).sum().float()
    metrics["supervised_trajectory_pixels"] = valid.sum().float()
    metrics["supervised_dynamic_pixels"] = foreground.sum().float()
    for name in ("trajectory", "displacement"):
        error = torch.linalg.vector_norm(prediction[name].float()-target[name].float(), dim=-1)
        task_valid = target.get(f"{name}_valid", valid).bool()
        if task_valid.any():
            metrics[f"{name}_epe_normalized"] = mean(error, task_valid)
        if (task_valid & temporal).any():
            metrics[f"{name}_epe_temporal_normalized"] = mean(error, task_valid & temporal)
        if (foreground & task_valid).any():
            metrics[f"{name}_epe_dynamic_normalized"] = mean(error, foreground & task_valid)

    depth_valid = target["depth_valid"].bool()
    if "excluded_background" in target:
        excluded = target["excluded_background"].sum().float()
        metrics["excluded_background_pixels"] = excluded
        metrics["excluded_background_fraction"] = excluded/(excluded+depth_valid.sum()).clamp_min(1)
    depth = _resize_scalar(prediction["depth"], tuple(target["depth"].shape[-2:])).float()
    depth_error = (depth-target["depth"].float()).abs()
    metrics["depth_mae_normalized"] = mean(depth_error, depth_valid)
    metrics["depth_abs_rel"] = mean(depth_error/target["depth"].float().clamp_min(1e-8), depth_valid)
    if "base_points" in prediction:
        base = target["ray"][..., 3:]+target["depth"][..., None]*target["ray"][..., :3]
        error = torch.linalg.vector_norm(prediction["base_points"].float()-base, dim=-1)
        metrics["base_epe_normalized"] = mean(error, depth_valid)

    if times.shape[1] > 1:
        order = times.argsort(1)
        dt = times.gather(1, order).diff(dim=1).float()
        if (dt <= 0).any():
            raise ValueError("velocity metrics require unique physical frame indices")
        index = order[:, None, :, None, None, None].expand_as(target["displacement"])
        residual = (prediction["displacement"].float()-target["displacement"].float()).gather(2, index)
        error = torch.linalg.vector_norm(residual.diff(dim=2)/dt[:, None, :, None, None, None], dim=-1)
        mask = target.get("displacement_valid", valid).bool().gather(2, index[..., 0])
        fg = foreground.gather(2, index[..., 0])
        pair = mask[:, :, 1:] & mask[:, :, :-1]
        if pair.any():
            metrics["velocity_epe_normalized"] = mean(error, pair)
        # At the source time displacement is identically zero. A moving
        # endpoint still defines a dynamic interval with that identity point.
        dynamic_pair = pair & (fg[:, :, 1:] | fg[:, :, :-1])
        if dynamic_pair.any():
            metrics["velocity_epe_dynamic_normalized"] = mean(error, dynamic_pair)

    for name, key, mask in (("depth", "depth_conf", depth_valid),
                            ("motion", "motion_confidence", valid)):
        if key not in prediction:
            continue
        confidence = _resize_scalar(prediction[key], tuple(mask.shape[-2:])).float()
        if name == "motion":
            eligible = prediction["source_time_idx"][:, :, None] != times[:, None]
            mask = mask & eligible[..., None, None]
            mask &= target.get("displacement_valid", valid).bool()
        if not mask.any():
            continue
        metrics[f"{name}_confidence_mean"] = mean(confidence, mask)
        metrics[f"{name}_confidence_log_mean"] = mean(confidence.clamp_min(1).log(), mask)
        metrics[f"{name}_confidence_saturated_fraction"] = mean((confidence >= 1e4).float(), mask)

    if "c2w" in target:
        mask = target["camera_valid"].bool() & prediction["observation_valid"].bool()
        pred, gt = prediction["c2w"].float(), target["c2w"].float()
        metrics["camera_translation_epe_normalized"] = mean(torch.linalg.vector_norm(pred[..., :3, 3]-gt[..., :3, 3], dim=-1), mask)
        metrics["camera_rotation_degrees"] = mean(rotation_error_degrees(pred[..., :3, :3], gt[..., :3, :3]), mask)

    if "visibility" in target and "visibility" in prediction:
        mask = torch.broadcast_to(target["visibility_valid"].bool(), target["visibility"].shape)
        label = target["visibility"].bool()[mask]
        estimate = prediction["visibility"].float()[mask] >= .5
        tp = (estimate & label).sum().float()
        fp = (estimate & ~label).sum().float()
        fn = (~estimate & label).sum().float()
        metrics["visibility_precision"] = tp/(tp+fp).clamp_min(1)
        metrics["visibility_recall"] = tp/(tp+fn).clamp_min(1)
        metrics["visibility_f1"] = 2*tp/(2*tp+fp+fn).clamp_min(1)
    return metrics
