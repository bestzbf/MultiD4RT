"""Sparse official 3D queries over released RGB models' dense predictions.

The native model consumes RGB. Camera calibration registers its arbitrary
world gauge; future tracking labels and benchmark depth maps are never inputs.
This input condition must be reported separately from MVTracker's RGB-D runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class TrackingInputs:
    video: torch.Tensor                 # [C,T,3,H,W], RGB 0..255
    intrinsics: torch.Tensor            # [C,T,3,3]
    w2c: torch.Tensor                   # [C,T,3,4], official normalized world
    queries: torch.Tensor               # [N,4], supplied (t,x,y,z)
    view_ids: tuple[int, ...]

    @classmethod
    def from_official(cls, datapoint, protocol):
        # Deliberately omit trajectory, visibility, valid and videodepth.
        return cls(datapoint.video, datapoint.intrs, datapoint.extrs,
                   datapoint.query_points_3d, protocol.views)


def fit_camera_similarity(predicted, calibrated):
    """Umeyama Sim(3) from camera centers ONLY; never fit ground-truth tracks."""
    x, y = predicted.detach().double().cpu(), calibrated.detach().double().cpu()
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("Camera registration requires at least three corresponding centers")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Non-finite camera centers")
    x_mean, y_mean = x.mean(0), y.mean(0)
    xc, yc = x - x_mean, y - y_mean
    if torch.linalg.matrix_rank(xc).item() < 2 or torch.linalg.matrix_rank(yc).item() < 2:
        raise ValueError("Degenerate camera centers: camera-only Sim(3) is underdetermined")
    u, singular, vh = torch.linalg.svd(xc.T @ yc / len(x))
    signs = torch.ones(3, dtype=torch.float64)
    signs[-1] = torch.linalg.det(vh.T @ u.T).sign()
    rotation = vh.T @ torch.diag(signs) @ u.T
    scale = (singular * signs).sum() / xc.square().sum(-1).mean()
    translation = y_mean - scale * (rotation @ x_mean)
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid camera registration scale")
    residual = (scale * (x @ rotation.T) + translation - y).square().sum(-1).mean().sqrt()
    return float(scale), rotation.float(), translation.float(), float(residual)


def sample_dense(field, xy):
    """[U,D,H,W] at [N,2] pixel centers -> [U,N,D]."""
    h, w = field.shape[-2:]
    xy = xy.to(device=field.device, dtype=torch.float32)
    grid = torch.stack((2 * xy[:, 0] / (w - 1) - 1, 2 * xy[:, 1] / (h - 1) - 1), -1)
    grid = grid[None, :, None].expand(field.shape[0], -1, -1, -1)
    sampled = F.grid_sample(field.float(), grid, align_corners=True, mode="bilinear")
    return sampled[..., 0].transpose(1, 2)


def resize_inputs(inputs, long_side, patch_size, device):
    c, t, _, old_h, old_w = inputs.video.shape
    scale = long_side / max(old_h, old_w)
    h, w = max(2, round(old_h * scale)), max(2, round(old_w * scale))
    images = F.interpolate(inputs.video.reshape(c * t, 3, old_h, old_w).float(),
                           size=(h, w), mode="bilinear", align_corners=True) / 255
    hp, wp = ((v + patch_size - 1) // patch_size * patch_size for v in (h, w))
    images = F.pad(images, (0, wp - w, 0, hp - h)).contiguous()
    checksum = hashlib.sha256(memoryview(images.numpy()).cast('B')).hexdigest()
    intrinsics = inputs.intrinsics.to(device).float().clone()
    intrinsics[..., 0, :] *= (w - 1) / (old_w - 1)
    intrinsics[..., 1, :] *= (h - 1) / (old_h - 1)
    return images[None].to(device), intrinsics, dict(raw_hw=[old_h, old_w], content_hw=[h, w],
                                                   padded_hw=[hp, wp], rgb_sha256=checksum)


def select_sources(scene, queries, intrinsics, w2c, similarity, content_hw):
    """Select the closest predicted source anchor among in-frame projections.

    Uses only the supplied query at its query time, calibration, and predicted
    source geometry. No annotation visibility or future coordinates are read.
    """
    c, t = intrinsics.shape[:2]
    n = len(queries)
    h, w = content_hw
    times = queries[:, 0].long()
    xyz = queries[:, 1:].float()
    homo = torch.cat((xyz, torch.ones_like(xyz[:, :1])), -1)
    camera_xyz = torch.einsum("cnij,nj->cni", w2c[:, times], homo)
    pixel = torch.einsum("cnij,cnj->cni", intrinsics[:, times], camera_xyz)
    xy = pixel[..., :2] / pixel[..., 2:].clamp_min(1e-8)
    eligible = (camera_xyz[..., 2] > 1e-6) & torch.isfinite(xy).all(-1)
    # Match the native visibility head's camera round-trip tolerance at borders.
    eligible &= (xy[..., 0] >= -1e-4) & (xy[..., 0] <= w - 1 + 1e-4)
    eligible &= (xy[..., 1] >= -1e-4) & (xy[..., 1] <= h - 1 + 1e-4)
    xy[..., 0].clamp_(0, w - 1)
    xy[..., 1].clamp_(0, h - 1)
    if not eligible.any(0).all():
        raise ValueError("A supplied query projects outside every input camera; no query may be dropped")
    scale, rotation, translation = similarity
    error = torch.full((c, n), float("inf"), device=queries.device)
    anchors = torch.zeros((c, n, 3), device=queries.device)
    for camera in range(c):
        for frame in times.unique().tolist():
            mask = (times == frame) & eligible[camera]
            if not mask.any():
                continue
            source = camera * t + frame
            field = scene.geometry["base_points"][0, source].permute(2, 0, 1)[None]
            anchor = sample_dense(field, xy[camera, mask])[0]
            anchor = scale * (anchor @ rotation.T) + translation
            anchors[camera, mask] = anchor
            error[camera, mask] = (anchor - xyz[mask]).norm(dim=-1)
    camera = error.argmin(0)
    index = torch.arange(n, device=queries.device)
    return camera * t + times, xy[camera, index], xyz - anchors[camera, index]


@torch.inference_mode()
def predict_official_queries(model, inputs, device="cuda", long_side=266,
                             query_time_chunk=32, amp_dtype=None, source_chunk=4):
    """Full synchronized sequence, with source/target decoding split for memory.

    Chunking never removes encoder observations or changes original time labels.
    The supplied query XYZ anchors the sampled trajectory by a constant offset.
    """
    if min(query_time_chunk, source_chunk, long_side) <= 0:
        raise ValueError("Chunk sizes and long_side must be positive")
    p = model.config.patch_size
    c, t = inputs.video.shape[:2]
    if c != len(inputs.view_ids):
        raise ValueError("Input views do not match official camera order")
    device = torch.device(device)
    images, intrinsics, preprocessing = resize_inputs(inputs, long_side, p, device)
    camera_ids = torch.tensor(inputs.view_ids, device=device).repeat_interleave(t)[None]
    times = torch.arange(t, device=device).repeat(c)[None]
    queries = inputs.queries.to(device).float()
    w2c = inputs.w2c.to(device).float()
    matrices = torch.eye(4, device=device).expand(c, t, 4, 4).clone()
    matrices[..., :3, :] = w2c
    centers = torch.linalg.inv(matrices)[..., :3, 3].reshape(c * t, 3)
    started = time.monotonic()
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        scene = model.encode(images, camera_ids, times)
        scale, rotation, translation, residual = fit_camera_similarity(scene.geometry["c2w"][0, :, :3, 3], centers)
        rotation, translation = rotation.to(device), translation.to(device)
        with torch.autocast(device.type, enabled=False):
            source_indices, xy, offset = select_sources(scene, queries, intrinsics, w2c,
                (scale, rotation, translation), preprocessing['content_hw'])
        n = len(queries)
        tracks = np.empty((t, n, 3), dtype=np.float32)
        visibility = np.empty((t, n), dtype=bool)
        sources = source_indices.unique(sorted=True)
        supports_visibility = getattr(model, 'supports_visibility', True)
        for source_start in range(0, len(sources), source_chunk):
            selected = sources[source_start:source_start + source_chunk]
            for start in range(0, t, query_time_chunk):
                stop = min(start + query_time_chunk, t)
                result = model.decode(scene, query_time_idx=torch.arange(start, stop, device=device)[None],
                                      source_obs_idx=selected[None], return_visibility=supports_visibility)
                for slot, source in enumerate(selected):
                    mask = source_indices == source
                    point_ids = mask.nonzero().flatten().cpu().numpy()
                    field = result['trajectory'][0, slot].permute(0, 3, 1, 2)
                    with torch.autocast(device.type, enabled=False):
                        predicted = sample_dense(field, xy[mask])
                        predicted = scale * (predicted @ rotation.T) + translation + offset[mask][None]
                        if supports_visibility:
                            probabilities = sample_dense(result['visibility'][0, slot], xy[mask])
                            visible = (probabilities > .5).any(-1).cpu().numpy()
                        else:
                            visible = np.ones((stop-start, len(point_ids)), dtype=bool)
                    tracks[start:stop, point_ids] = predicted.cpu().numpy()
                    visibility[start:stop, point_ids] = visible
                    del field, predicted
                del result
            print(f'  decoded {min(source_start+source_chunk,len(sources))}/{len(sources)} source observations', flush=True)
    diagnostics = {"camera_sim3_scale": scale, "camera_sim3_rotation": rotation.cpu().tolist(),
                   "camera_sim3_translation": translation.cpu().tolist(),
                   "camera_registration_rmse_normalized_world": residual,
                   "source_observation_count": len(source_indices.unique()),
                   "input_observations": c * t, "frames": t, "queries": n,
                   "max_query_offset_normalized_world": float(offset.norm(dim=-1).max()),
                   "preprocessing": preprocessing, "seconds": time.monotonic()-started,
                   "visibility_predictor": 'native' if supports_visibility else 'unavailable; all-visible storage placeholder'}
    if not np.isfinite(tracks).all():
        raise ValueError("Model produced non-finite trajectories")
    return tracks, visibility, diagnostics
