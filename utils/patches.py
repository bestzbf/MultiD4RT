"""Shared local patch extraction helpers."""

import torch
import torch.nn.functional as F


def _build_patch_offsets(patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = patch_size // 2
    offsets_1d = torch.arange(-half, half + 1, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(offsets_1d, offsets_1d, indexing="xy"), dim=-1)


def extract_local_patches(
    frames_btchw: torch.Tensor,
    coords: torch.Tensor,
    t_src: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Extract query-centered RGB patches using the same sampler as decoder inference.

    Args:
        frames_btchw: (B, T, C, H, W) video frames in [0, 1]
        coords: (B, N, 2) normalized coordinates in [0, 1]
        t_src: (B, N) source frame indices
        patch_size: square patch size

    Returns:
        patches: (B, N, C, patch_size, patch_size)
    """
    if frames_btchw.dim() != 5:
        raise ValueError(f"Expected frames_btchw to be 5D, got shape {tuple(frames_btchw.shape)}")
    if coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords to have shape (B, N, 2), got {tuple(coords.shape)}")
    if t_src.dim() != 2:
        raise ValueError(f"Expected t_src to have shape (B, N), got {tuple(t_src.shape)}")

    batch_size, num_frames, channels, height, width = frames_btchw.shape
    _, num_queries, _ = coords.shape
    if t_src.shape != (batch_size, num_queries):
        raise ValueError(
            "Expected t_src shape to match coords batch/query dimensions, "
            f"got coords={tuple(coords.shape)} and t_src={tuple(t_src.shape)}"
        )

    if frames_btchw.dtype == torch.uint8:
        target_dtype = coords.dtype if coords.is_floating_point() else torch.float32
        frames_btchw = frames_btchw.to(dtype=target_dtype).div_(255.0)
    elif not frames_btchw.is_floating_point():
        frames_btchw = frames_btchw.float()

    # Use 5D grid_sample to avoid duplicating high-res frames in memory
    # Input: (B, C, T, H, W)
    input_5d = frames_btchw.permute(0, 2, 1, 3, 4)
    grid_dtype = frames_btchw.dtype

    offsets = _build_patch_offsets(patch_size, device=frames_btchw.device, dtype=grid_dtype)

    coords_pixel = coords.to(dtype=grid_dtype).clone()
    coords_pixel[..., 0] = coords_pixel[..., 0] * (width - 1)
    coords_pixel[..., 1] = coords_pixel[..., 1] * (height - 1)

    # Spatial grid: (B, N, patch_size, patch_size, 2)
    grid_xy = coords_pixel.view(batch_size, num_queries, 1, 1, 2) + offsets.view(1, 1, patch_size, patch_size, 2)
    grid_xy[..., 0] = 2.0 * grid_xy[..., 0] / max(width - 1, 1) - 1.0
    grid_xy[..., 1] = 2.0 * grid_xy[..., 1] / max(height - 1, 1) - 1.0

    # Temporal grid: (B, N, patch_size, patch_size, 1)
    t_src_clamped = t_src.clamp(0, num_frames - 1).to(dtype=grid_dtype)
    grid_z = 2.0 * t_src_clamped / max(num_frames - 1, 1) - 1.0
    grid_z = grid_z.view(batch_size, num_queries, 1, 1, 1).expand(batch_size, num_queries, patch_size, patch_size, 1)

    # Combined 3D grid: (B, N, patch_size, patch_size, 3)
    # Note: 5D grid_sample expects (x, y, z) corresponding to (W, H, D)
    grid_xyz = torch.cat([grid_xy, grid_z], dim=-1)

    patches = F.grid_sample(
        input_5d,
        grid_xyz,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    
    # Output is (B, C, N, patch_size, patch_size), permute to (B, N, C, patch_size, patch_size)
    return patches.permute(0, 2, 1, 3, 4)


def extract_local_patches_with_valid_hw(
    frames_btchw: torch.Tensor,
    coords: torch.Tensor,
    t_src: torch.Tensor,
    patch_size: int,
    valid_hw: torch.Tensor,
) -> torch.Tensor:
    """Extract patches from per-sample valid crop regions inside a padded batch tensor.

    Args:
        frames_btchw: (B, T, C, H_pad, W_pad) padded video frames in [0, 1]
        coords: (B, N, 2) normalized coordinates in canonical crop space [0, 1]
        t_src: (B, N) source frame indices
        patch_size: square patch size
        valid_hw: (B, 2) actual (crop_h, crop_w) before padding

    Returns:
        patches: (B, N, C, patch_size, patch_size)
    """
    if valid_hw.dim() != 2 or valid_hw.shape[-1] != 2:
        raise ValueError(f"Expected valid_hw to have shape (B, 2), got {tuple(valid_hw.shape)}")
    if valid_hw.shape[0] != frames_btchw.shape[0]:
        raise ValueError(
            "Expected valid_hw batch dimension to match frames batch dimension, "
            f"got frames={tuple(frames_btchw.shape)} and valid_hw={tuple(valid_hw.shape)}"
        )
    if frames_btchw.dim() != 5:
        raise ValueError(f"Expected frames_btchw to be 5D, got shape {tuple(frames_btchw.shape)}")
    if coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords to have shape (B, N, 2), got {tuple(coords.shape)}")
    if t_src.dim() != 2:
        raise ValueError(f"Expected t_src to have shape (B, N), got {tuple(t_src.shape)}")

    batch_size, num_frames, _, padded_h, padded_w = frames_btchw.shape
    _, num_queries, _ = coords.shape
    if t_src.shape != (batch_size, num_queries):
        raise ValueError(
            "Expected t_src shape to match coords batch/query dimensions, "
            f"got coords={tuple(coords.shape)} and t_src={tuple(t_src.shape)}"
        )

    # Unlike the old per-sample implementation, build one batched grid over the padded tensor.
    # We clamp spatial samples to each example's valid crop bounds so the semantics match
    # slicing to [:crop_h, :crop_w] and using padding_mode='border'.
    input_5d = frames_btchw.permute(0, 2, 1, 3, 4)
    grid_dtype = frames_btchw.dtype if frames_btchw.is_floating_point() else coords.dtype
    offsets = _build_patch_offsets(patch_size, device=frames_btchw.device, dtype=grid_dtype)

    crop_h = valid_hw[:, 0].to(device=coords.device, dtype=grid_dtype)
    crop_w = valid_hw[:, 1].to(device=coords.device, dtype=grid_dtype)
    crop_h = crop_h.clamp(1.0, float(padded_h))
    crop_w = crop_w.clamp(1.0, float(padded_w))
    crop_h_max = (crop_h - 1.0).clamp_min(0.0).view(batch_size, 1, 1, 1)
    crop_w_max = (crop_w - 1.0).clamp_min(0.0).view(batch_size, 1, 1, 1)

    coords_xy = coords.to(dtype=grid_dtype)
    coords_x = coords_xy[..., 0] * crop_w_max.view(batch_size, 1)
    coords_y = coords_xy[..., 1] * crop_h_max.view(batch_size, 1)

    grid_x = coords_x.view(batch_size, num_queries, 1, 1) + offsets[..., 0].view(1, 1, patch_size, patch_size)
    grid_y = coords_y.view(batch_size, num_queries, 1, 1) + offsets[..., 1].view(1, 1, patch_size, patch_size)

    grid_x = torch.clamp(grid_x, min=0.0)
    grid_y = torch.clamp(grid_y, min=0.0)
    grid_x = torch.minimum(grid_x, crop_w_max)
    grid_y = torch.minimum(grid_y, crop_h_max)

    grid_xy = torch.stack(
        [
            2.0 * grid_x / max(padded_w - 1, 1) - 1.0,
            2.0 * grid_y / max(padded_h - 1, 1) - 1.0,
        ],
        dim=-1,
    )

    t_src_clamped = t_src.clamp(0, num_frames - 1).to(dtype=grid_dtype)
    grid_z = 2.0 * t_src_clamped / max(num_frames - 1, 1) - 1.0
    grid_z = grid_z.view(batch_size, num_queries, 1, 1, 1).expand(batch_size, num_queries, patch_size, patch_size, 1)

    grid_xyz = torch.cat([grid_xy, grid_z], dim=-1)
    patches = F.grid_sample(
        input_5d,
        grid_xyz,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return patches.permute(0, 2, 1, 3, 4)


def extract_local_patches_from_frame_list(
    frames_btchw: list[torch.Tensor],
    coords: torch.Tensor,
    t_src: torch.Tensor,
    patch_size: int,
    valid_hw: torch.Tensor,
) -> torch.Tensor:
    """Extract patches from a list of per-sample frame tensors without padding.

    Each list entry is expected to be (T, C, H, W). This is intentionally
    per-sample to avoid padding an entire batch to one large high-res crop.
    """
    if coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords to have shape (B, N, 2), got {tuple(coords.shape)}")
    if t_src.dim() != 2:
        raise ValueError(f"Expected t_src to have shape (B, N), got {tuple(t_src.shape)}")
    if valid_hw.dim() != 2 or valid_hw.shape[-1] != 2:
        raise ValueError(f"Expected valid_hw to have shape (B, 2), got {tuple(valid_hw.shape)}")

    batch_size, num_queries, _ = coords.shape
    if len(frames_btchw) != batch_size:
        raise ValueError(
            "Expected frame list length to match coords batch dimension, "
            f"got {len(frames_btchw)} and {batch_size}"
        )
    if t_src.shape != (batch_size, num_queries):
        raise ValueError(
            "Expected t_src shape to match coords batch/query dimensions, "
            f"got coords={tuple(coords.shape)} and t_src={tuple(t_src.shape)}"
        )
    if valid_hw.shape[0] != batch_size:
        raise ValueError(
            "Expected valid_hw batch dimension to match coords batch dimension, "
            f"got coords={tuple(coords.shape)} and valid_hw={tuple(valid_hw.shape)}"
        )

    patches = []
    for b, frames_i in enumerate(frames_btchw):
        if frames_i.dim() != 4:
            raise ValueError(
                f"Expected frame list entry {b} to have shape (T, C, H, W), "
                f"got {tuple(frames_i.shape)}"
            )
        if frames_i.shape[1] != 3:
            raise ValueError(
                f"Expected frame list entry {b} channel dimension to be 3, "
                f"got {tuple(frames_i.shape)}"
            )
        target_dtype = coords.dtype if coords.is_floating_point() else torch.float32
        if frames_i.dtype == torch.uint8:
            frames_i = frames_i.to(device=coords.device, dtype=target_dtype).div_(255.0)
        else:
            frames_i = frames_i.to(device=coords.device, dtype=target_dtype)
        patches_i = extract_local_patches_with_valid_hw(
            frames_btchw=frames_i.unsqueeze(0),
            coords=coords[b:b + 1],
            t_src=t_src[b:b + 1],
            patch_size=patch_size,
            valid_hw=valid_hw[b:b + 1],
        )
        patches.append(patches_i.squeeze(0))

    return torch.stack(patches, dim=0)
