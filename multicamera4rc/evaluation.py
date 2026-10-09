"""Prediction and physical-error measurements with no optimization dependency."""

import numpy as np
import torch

from .metadata import build_metadata, gather_observations
from .metrics import dense_metrics, native_track_metrics
from .tasks import sample_tracks


def calibrated(K, E, *, allow_skew=False, allow_left_world=False):
    K, E = np.asarray(K, np.float64), np.asarray(E, np.float64)
    if K.ndim != 3 or K.shape[1:] != (3, 3) or E.shape != (len(K), 4, 4):
        raise ValueError("calibration must have native frame axis and 3x3/4x4 matrices")
    valid = np.isfinite(K).all((1, 2)) & np.isfinite(E).all((1, 2))
    valid &= (K[:, 0, 0] > 0) & (K[:, 1, 1] > 0)
    valid &= np.max(abs(K[:, 2]-[0, 0, 1]), axis=1) < 1e-6
    valid &= np.max(abs(E[:, 3]-[0, 0, 0, 1]), axis=1) < 1e-6
    R = np.nan_to_num(E[:, :3, :3])
    valid &= np.max(abs(R @ R.transpose(0, 2, 1)-np.eye(3)), axis=(1, 2)) < 5e-3
    determinant = np.linalg.det(R)
    valid &= (abs(abs(determinant)-1) < 5e-3 if allow_left_world else abs(determinant-1) < 5e-3)
    if not allow_skew:
        valid &= (abs(K[:, 0, 1]) < 1e-6) & (abs(K[:, 1, 0]) < 1e-6)
    return valid


@torch.inference_mode()
def evaluate_batch(model, batch, source, target, need_visibility=False):
    """Keep dense/particle masks and source identity, without computing a loss."""
    images = batch["images"]
    empty = torch.zeros((), device=images.device, dtype=torch.bool)
    geometry_only = ("displacement_valid" in target and not target["displacement_valid"].any()
                     and not target.get("sparse_displacement_valid", empty).any()
                     and not target.get("sparse_trajectory_valid", empty).any()
                     and not need_visibility and not target.get("static_mask", empty).any())
    if geometry_only:
        scene = model.encode(images, batch["camera_id"], batch["time_idx"], batch["observation_valid"])
        meta = build_metadata(scene.camera_id, scene.time_idx, batch["query_time_idx"], source, scene.observation_valid)
        base = gather_observations(scene.geometry["base_points"], meta.source_index)
        identity = (meta.source_times[..., None] == meta.query_times[:, None]) & meta.source_valid[..., None]
        xyz = torch.where(identity[..., None, None, None], base[:, :, None], 0)
        prediction = dict(scene.geometry)
        prediction.update(trajectory=xyz, displacement=torch.zeros_like(xyz),
            source_obs_idx=meta.source_index, source_valid=meta.source_valid,
            source_time_idx=meta.source_times, query_time_idx=meta.query_times,
            camera_id=scene.camera_id, time_idx=scene.time_idx,
            observation_valid=scene.observation_valid)
    else:
        prediction = model(images, batch["camera_id"], batch["time_idx"], batch["query_time_idx"], source,
                           observation_valid=batch["observation_valid"], return_visibility=need_visibility)
    metrics = dense_metrics(prediction, target) if "trajectory_valid" in target and "trajectory" in target else {}
    if "sparse_queries" in target:
        sampled = sample_tracks(prediction, target["sparse_queries"], strict=False)
        metrics.update(native_track_metrics(sampled, target))
    if not metrics or not all(torch.isfinite(value).all() for value in metrics.values()):
        raise FloatingPointError("evaluation produced missing or non-finite physical metrics")
    return {key: value.detach() for key, value in metrics.items()}
