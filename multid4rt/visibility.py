"""Per-target-camera visibility prediction with detached geometric cues."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VisibilityHead(nn.Module):
    """Predict visibility without using its score to hide trajectory errors.

    Geometry, depth, and confidence sampling coordinates are detached. Source
    motion features and sampled target image features remain differentiable.
    ``projection_valid`` is an eligibility mask, not an occlusion label.
    """

    def __init__(self, motion_feature_dim: int = 128, patch_feature_dim: int = 1536,
                 hidden_dim: int = 128, query_chunk_size: int | None = 4096):
        super().__init__()
        if min(motion_feature_dim, patch_feature_dim, hidden_dim) <= 0 or (query_chunk_size is not None and query_chunk_size <= 0):
            raise ValueError("feature dimensions and query_chunk_size must be positive")
        self.motion_feature_dim = motion_feature_dim
        self.patch_feature_dim = patch_feature_dim
        self.query_chunk_size = query_chunk_size
        self.source_projection = (nn.Identity() if motion_feature_dim == 128
                                  else nn.Linear(motion_feature_dim, 128))
        self.target_projection = nn.Linear(patch_feature_dim, 128)
        self.classifier = nn.Sequential(nn.Linear(260, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, 64), nn.ReLU(),
                                        nn.Linear(64, 1))

    def forward(self, trajectory: Tensor, motion_features: Tensor,
                target_features: Tensor, target_depth: Tensor,
                target_depth_conf: Tensor, target_K: Tensor, target_c2w: Tensor,
                target_valid: Tensor, patch_grid: tuple[int, int]) -> dict[str, Tensor]:
        if trajectory.ndim != 5 or trajectory.shape[-1] != 3:
            raise ValueError("trajectory must be [B, U, H, W, 3]")
        batch, queries, height, width, _ = trajectory.shape
        if target_features.ndim != 5 or target_features.shape[:2] != (batch, queries):
            raise ValueError("target_features must be [B, U, C, L, D]")
        cameras = target_features.shape[2]
        hp, wp = patch_grid
        if hp <= 0 or wp <= 0 or target_features.shape[-2:] != (hp * wp, self.patch_feature_dim):
            raise ValueError("target feature tokens must match patch_grid and patch_feature_dim")
        if motion_features.shape != (batch, queries, self.motion_feature_dim, height, width):
            raise ValueError("motion_features shape does not match trajectory or configured channels")
        image_shape = (batch, queries, cameras, height, width)
        if target_depth.shape != image_shape or target_depth_conf.shape != image_shape:
            raise ValueError("target depth/confidence must be [B, U, C, H, W]")
        if target_K.shape != (batch, queries, cameras, 3, 3) or target_c2w.shape != (batch, queries, cameras, 4, 4):
            raise ValueError("target camera matrices have incompatible shapes")
        if target_valid.shape != (batch, queries, cameras) or target_valid.dtype != torch.bool:
            raise ValueError("target_valid must be a boolean [B, U, C] tensor")
        if queries == 0 or cameras == 0:
            raise ValueError("visibility requires at least one query and target camera")

        pixels = height * width
        chunk_size = pixels if self.query_chunk_size is None else self.query_chunk_size
        all_logits, all_valid = [], []
        # Work one target camera at a time; no [B,U,C,H,W,260] tensor is built.
        for b in range(batch):
            query_logits, query_valid = [], []
            for u in range(queries):
                points = trajectory[b, u].reshape(pixels, 3)
                source = motion_features[b, u].reshape(self.motion_feature_dim, pixels)
                camera_logits, camera_valid = [], []
                for c in range(cameras):
                    target = self.target_projection(target_features[b, u, c])
                    target = target.transpose(0, 1).reshape(1, 128, hp, wp)
                    image_depth = target_depth[b, u, c].detach().float()[None, None]
                    image_conf = target_depth_conf[b, u, c].detach().float()[None, None]
                    chunks_logits, chunks_valid = [], []
                    for start in range(0, pixels, chunk_size):
                        stop = min(start + chunk_size, pixels)
                        with torch.no_grad():
                            camera = target_c2w[b, u, c].float()
                            intrinsics = target_K[b, u, c].float()
                            camera_points = ((points[start:stop].detach().float() - camera[:3, 3])
                                             @ camera[:3, :3])
                            z = camera_points[:, 2]
                            projected = camera_points @ intrinsics.transpose(0, 1)
                            uv = projected[:, :2] / z.clamp_min(1e-6)[:, None]
                            valid = (target_valid[b, u, c] & torch.isfinite(camera_points).all(-1)
                                     & torch.isfinite(uv).all(-1) & (z > 1e-6)
                                     # Round-trip camera math can move a border
                                     # pixel a few ulps outside the image.
                                     & (uv[:, 0] >= -1e-4) & (uv[:, 0] <= width - 1 + 1e-4)
                                     & (uv[:, 1] >= -1e-4) & (uv[:, 1] <= height - 1 + 1e-4))
                            # align_corners=False: integer pixel center u maps via u+0.5.
                            grid = torch.stack((2 * (uv[:, 0] + 0.5) / width - 1,
                                                2 * (uv[:, 1] + 0.5) / height - 1), -1)
                            grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
                            grid = grid.clamp(-2.0, 2.0)[None, None]
                            sampled_depth = F.grid_sample(image_depth, grid, align_corners=False,
                                                          mode="bilinear", padding_mode="zeros").reshape(-1)
                            sampled_conf = F.grid_sample(image_conf, grid, align_corners=False,
                                                         mode="bilinear", padding_mode="zeros").reshape(-1)
                            depth_known = torch.isfinite(sampled_depth) & (sampled_depth > 1e-6)
                            residual = torch.where(depth_known,
                                                   (z - sampled_depth) / sampled_depth.clamp_min(1e-6),
                                                   torch.zeros_like(z))
                            residual = torch.nan_to_num(residual).clamp(-10, 10)
                            log_conf = torch.nan_to_num(sampled_conf, nan=0.0, posinf=1e6,
                                                       neginf=0.0).clamp(1e-6, 1e6).log()
                            cues = torch.cat((grid.reshape(-1, 2), residual[:, None],
                                              log_conf[:, None]), dim=-1)
                        sampled = F.grid_sample(target, grid.to(target.dtype), align_corners=False,
                                                mode="bilinear", padding_mode="zeros")
                        sampled = sampled.reshape(128, -1).transpose(0, 1)
                        source_feature = self.source_projection(source[:, start:stop].transpose(0, 1))
                        combined = torch.cat((source_feature, sampled, cues.to(source_feature.dtype)), -1)
                        chunks_logits.append(self.classifier(combined).squeeze(-1))
                        chunks_valid.append(valid)
                    camera_logits.append(chunks_logits[0] if len(chunks_logits) == 1 else torch.cat(chunks_logits))
                    camera_valid.append(chunks_valid[0] if len(chunks_valid) == 1 else torch.cat(chunks_valid))
                query_logits.append(torch.stack(camera_logits))
                query_valid.append(torch.stack(camera_valid))
            all_logits.append(torch.stack(query_logits))
            all_valid.append(torch.stack(query_valid))
        logits = torch.stack(all_logits).reshape(image_shape)
        projection_valid = torch.stack(all_valid).reshape(image_shape)
        return {"visibility_logits": logits,
                "visibility": logits.sigmoid() * projection_valid.to(logits.dtype),
                "projection_valid": projection_valid}
