"""
Collate function for D4RT training batches.

Handles:
- Stacking video tensors
- Stacking query coordinates and time indices
- Stacking targets and masks
- Collecting metadata

Usage:
    from torch.utils.data import DataLoader
    from datasets.collate import d4rt_collate_fn

    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=d4rt_collate_fn,
    )
"""

from __future__ import annotations

import os
from typing import Any

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def d4rt_collate_fn(batch: list[Any]) -> dict[str, Any]:
    """
    Collate function for D4RT QuerySample batches.

    Args:
        batch: List of QuerySample dataclass instances from MixtureDataset

    Returns:
        Batched dict with stacked tensors
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")

    defer_large_stack = _env_flag("D4RT_DEFER_LARGE_COLLATE_STACK", False)

    # Stack video tensors [B, S, 3, H, W].  In planned mode this copy dominates
    # CPU collate for bs=40, so allow training to defer large tensor stacking to
    # move_batch_to_device where the tensors are already going to CUDA.
    if defer_large_stack:
        videos = [s.video for s in batch]
    else:
        videos = torch.stack([s.video for s in batch], dim=0)

    # Stack query data [B, Q, ...]
    coords = torch.stack([s.coords for s in batch], dim=0)
    t_src = torch.stack([s.t_src for s in batch], dim=0)
    t_tgt = torch.stack([s.t_tgt for s in batch], dim=0)
    t_cam = torch.stack([s.t_cam for s in batch], dim=0)

    # Stack camera parameters [B, S, ...]
    intrinsics = torch.stack([s.intrinsics for s in batch], dim=0)
    extrinsics = torch.stack([s.extrinsics for s in batch], dim=0)

    # Preserve all target fields emitted by query_builder; this keeps analysis
    # markers such as is_static_reprojection and boundary flags from being
    # silently dropped by collation.
    targets = {
        key: torch.stack([s.targets[key] for s in batch], dim=0)
        for key in batch[0].targets
    }

    # Stack local patches [B, Q, 3, P, P] (may be None if precompute_patches=False)
    if batch[0].local_patches is not None:
        if defer_large_stack:
            local_patches = [s.local_patches for s in batch]
        else:
            local_patches = torch.stack([s.local_patches for s in batch], dim=0)
    else:
        local_patches = None

    if any(getattr(s, "highres_video", None) is not None for s in batch):
        highres_videos = [s.highres_video for s in batch]
    else:
        highres_videos = None

    if getattr(batch[0], "transform_metadata", None) is not None:
        transform_metadata = {
            key: torch.stack([s.transform_metadata[key] for s in batch], dim=0)
            for key in batch[0].transform_metadata
        }
    else:
        transform_metadata = None

    if getattr(batch[0], "aspect_ratio", None) is not None:
        aspect_ratio = torch.stack([s.aspect_ratio for s in batch], dim=0)
    else:
        aspect_ratio = None

    # Collect metadata (list of dicts)
    dataset_names = [s.dataset_name for s in batch]
    sequence_names = [s.sequence_name for s in batch]
    metadata = [s.metadata for s in batch]

    # Build dataset_id tensor (B, N) for per-dataset depth normalization.
    # Each batch element gets a unique integer id based on its dataset_name,
    # so that compute_depth_normalizers() can normalize each dataset's depth
    # scale independently within the same batch.
    unique_names = sorted(set(dataset_names))
    name_to_id = {name: idx for idx, name in enumerate(unique_names)}
    num_queries = coords.shape[1]  # N
    dataset_id = torch.stack(
        [torch.full((num_queries,), name_to_id[name], dtype=torch.long) for name in dataset_names],
        dim=0,
    )  # (B, N)

    return {
        "video": videos,
        "coords": coords,
        "t_src": t_src,
        "t_tgt": t_tgt,
        "t_cam": t_cam,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "targets": targets,
        "local_patches": local_patches,
        "highres_video": highres_videos,
        "transform_metadata": transform_metadata,
        "aspect_ratio": aspect_ratio,
        "dataset_names": dataset_names,
        "dataset_id": dataset_id,
        "sequence_names": sequence_names,
        "metadata": metadata,
    }
