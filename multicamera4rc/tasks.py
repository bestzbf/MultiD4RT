"""Task helpers on dense world trajectories; source pixels preserve identity."""

import torch
from torch.nn import functional as F

from .metadata import gather_observations


def scene_flow(prediction, start_time, end_time):
    """World displacement [B,Q,H,W,3] between two frame labels, not output slots."""
    query = prediction["query_time_idx"]
    slots = []
    for time in (start_time, end_time):
        match = query == time
        if not match.any(1).all():
            raise ValueError(f"frame label {time} was not queried in every clip")
        slots.append(match.long().argmax(1))
    frames_first = prediction["trajectory"].transpose(1, 2)
    return gather_observations(frames_first, slots[1])-gather_observations(frames_first, slots[0])


def sample_tracks(prediction, queries, strict=True):
    """Bilinearly sample [source_observation_index,u,v] queries in pixel units.

    Returns [B,N,Tq,3] world tracks and [B,N,Tq,C] visibility. No N-fold copy
    of the dense maps is made. ``strict=False`` masks queries for unselected
    sources, allowing a training step to decode only a subset of source images.
    """
    trajectory = prediction["trajectory"]
    b, q, t, h, w, _ = trajectory.shape
    queries = torch.as_tensor(queries, device=trajectory.device)
    if queries.ndim != 3 or queries.shape[0] != b or queries.shape[-1] != 3:
        raise ValueError("queries must have shape [B,N,3] = [source_obs,u,v]")
    if not torch.isfinite(queries).all():
        raise ValueError("queries must be finite")
    source = queries[..., 0].long()
    if (queries[..., 0] != source).any():
        raise ValueError("query source observation indices must be integers")
    n = queries.shape[1]
    output = dict(trajectory=trajectory.new_zeros(b, n, t, 3),
                  displacement=trajectory.new_zeros(b, n, t, 3),
                  motion_confidence=trajectory.new_zeros(b, n, t),
                  query_valid=torch.zeros(b, n, dtype=torch.bool, device=trajectory.device),
                  source_time_idx=torch.zeros(b, n, dtype=torch.long, device=trajectory.device))
    if "visibility" in prediction:
        c = prediction["visibility"].shape[3]
        for key in ("visibility", "visibility_logits"):
            output[key] = trajectory.new_zeros(b, n, t, c)
        output["projection_valid"] = torch.zeros(b, n, t, c, dtype=torch.bool,
                                                 device=trajectory.device)
    for batch in range(b):
        for slot in range(q):
            index = prediction["source_obs_idx"][batch, slot]
            selected = torch.nonzero(source[batch] == index, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            uv = queries[batch, selected, 1:].to(trajectory.dtype)
            valid = ((uv[:, 0] >= 0) & (uv[:, 0] <= w-1)
                     & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
                     & prediction["source_valid"][batch, slot])
            selected, uv = selected[valid], uv[valid]
            if selected.numel() == 0:
                continue
            grid = torch.stack((2*uv[:, 0]/(w-1)-1, 2*uv[:, 1]/(h-1)-1), -1)
            grid = grid[None, :, None].expand(t, -1, 1, -1)
            for key in ("trajectory", "displacement"):
                maps = prediction[key][batch, slot].permute(0, 3, 1, 2)
                values = F.grid_sample(maps, grid, align_corners=True).squeeze(-1).permute(2, 0, 1)
                output[key][batch, selected] = values
            maps = prediction["motion_confidence"][batch, slot, :, None]
            output["motion_confidence"][batch, selected] = F.grid_sample(
                maps, grid, align_corners=True).squeeze(1).squeeze(-1).transpose(0, 1)
            if "visibility" in prediction:
                for key in ("visibility", "visibility_logits", "projection_valid"):
                    maps = prediction[key][batch, slot].float()
                    values = F.grid_sample(maps, grid, mode="bilinear", align_corners=True)
                    values = values.squeeze(-1).permute(2, 0, 1)
                    # Logits use bilinear support, so every contributing
                    # projection must be valid. A nearest mask can admit
                    # logits blended with invalid neighbors at boundaries.
                    output[key][batch, selected] = (
                        values >= 1.0-1e-6 if key == "projection_valid" else values)
            output["query_valid"][batch, selected] = True
            output["source_time_idx"][batch, selected] = prediction["source_time_idx"][batch, slot]
    if strict and not output["query_valid"].all():
        raise ValueError("some queries are outside the source image, invalid, or use unselected sources")
    if "visibility" in output:
        output["visibility"] = output["visibility"]*output["projection_valid"]
    output["query_time_idx"] = prediction["query_time_idx"]
    return output
