"""Explicit observation, camera and target-time indexing; no inferred timestamps."""

from dataclasses import dataclass
import torch


def integer_tensor(value, name, device=None):
    result = torch.as_tensor(value, device=device)
    if result.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must contain integer indices")
    return result.long()


def gather_observations(values, indices):
    """[B,S,...] indexed by [B,...] -> [B,...,...], without copying every source."""
    batch = torch.arange(values.shape[0], device=values.device)
    batch = batch.reshape(values.shape[0], *([1] * (indices.ndim - 1)))
    return values[batch, indices]


@dataclass
class ObservationMetadata:
    camera_id: torch.Tensor
    time_idx: torch.Tensor
    observation_valid: torch.Tensor
    query_times: torch.Tensor
    source_index: torch.Tensor
    source_valid: torch.Tensor
    source_times: torch.Tensor
    camera_values: torch.Tensor
    camera_valid: torch.Tensor
    internal_times: torch.Tensor
    internal_valid: torch.Tensor
    target_slot: torch.Tensor
    source_slot: torch.Tensor
    memory_index: torch.Tensor
    memory_valid: torch.Tensor


def validate_observations(camera_id, time_idx, observation_valid=None):
    camera_id = integer_tensor(camera_id, "camera_id")
    time_idx = integer_tensor(time_idx, "time_idx", camera_id.device)
    if camera_id.ndim != 2 or time_idx.shape != camera_id.shape:
        raise ValueError("camera_id and time_idx must both have shape [B,S]")
    if min(camera_id.shape) == 0:
        raise ValueError("a clip must have at least one batch item and observation")
    if observation_valid is None:
        observation_valid = torch.ones_like(camera_id, dtype=torch.bool)
    else:
        observation_valid = torch.as_tensor(observation_valid, device=camera_id.device)
        if observation_valid.dtype != torch.bool or observation_valid.shape != camera_id.shape:
            raise ValueError("observation_valid must be boolean [B,S]")
    if not observation_valid[:, 0].all():
        raise ValueError("observation 0 is the fixed reference and must be valid in every clip")
    if (camera_id[observation_valid] < 0).any():
        raise ValueError("valid camera IDs must be nonnegative; -1 is reserved for padding")
    for cameras, times, valid in zip(camera_id.tolist(), time_idx.tolist(),
                                    observation_valid.tolist()):
        seen = set()
        for c, t, v in zip(cameras, times, valid):
            if v and (c, t) in seen:
                raise ValueError(f"duplicate observation for camera={c}, time={t}")
            if v:
                seen.add((c, t))
    return camera_id, time_idx, observation_valid


def build_metadata(camera_id, time_idx, query_time_idx=None, source_obs_idx=None,
                   observation_valid=None):
    camera_id, time_idx, valid = validate_observations(camera_id, time_idx, observation_valid)
    batch_size, observations = camera_id.shape
    device = camera_id.device
    if query_time_idx is None:
        rows = [sorted(set(times[v] for v in range(observations) if mask[v]))
                for times, mask in zip(time_idx.tolist(), valid.tolist())]
        if len({len(row) for row in rows}) != 1:
            raise ValueError("variable time counts need an explicit rectangular query_time_idx")
        query_time_idx = rows
    query = integer_tensor(query_time_idx, "query_time_idx", device)
    if query.ndim == 1:
        query = query[None].expand(batch_size, -1)
    if query.ndim != 2 or query.shape[0] != batch_size or query.shape[1] == 0:
        raise ValueError("query_time_idx must be [B,Tq] or [Tq], with Tq > 0")
    if source_obs_idx is None:
        source = torch.arange(observations, device=device)[None].expand(batch_size, -1)
    else:
        source = integer_tensor(source_obs_idx, "source_obs_idx", device)
        if source.ndim == 1:
            source = source[None].expand(batch_size, -1)
        if source.ndim != 2 or source.shape[0] != batch_size or source.shape[1] == 0:
            raise ValueError("source_obs_idx must be nonempty [Q] or [B,Q]")
        if ((source < 0) | (source >= observations)).any():
            raise ValueError("source observation index is out of range")
    source_valid = gather_observations(valid, source)
    source_times = gather_observations(time_idx, source)
    camera_rows, internal_rows, lookups, anchor_rows = [], [], [], []
    for b in range(batch_size):
        cameras = camera_id[b].tolist()
        times = time_idx[b].tolist()
        masks = valid[b].tolist()
        lookup = {(c, t): s for s, (c, t, v) in enumerate(zip(cameras, times, masks)) if v}
        observed_times = {t for _, t in lookup}
        queries = query[b].tolist()
        if len(set(queries)) != len(queries):
            raise ValueError("query_time_idx must contain unique frame labels per clip")
        if not set(queries).issubset(observed_times):
            raise ValueError("each query time requires at least one valid observed target image")
        src = source[b].tolist()
        if len(set(src)) != len(src):
            raise ValueError("source_obs_idx must not contain duplicates")
        internal = list(queries)
        for s in src:
            if masks[s] and times[s] not in internal:
                internal.append(times[s])
        anchors = [internal.index(times[s]) if masks[s] else 0 for s in src]
        camera_rows.append(sorted({c for c, _ in lookup}))
        internal_rows.append(internal)
        anchor_rows.append(anchors)
        lookups.append(lookup)
    cameras_max = max(map(len, camera_rows))
    times_max = max(map(len, internal_rows))
    camera_values = torch.full((batch_size, cameras_max), -1, device=device, dtype=torch.long)
    camera_valid = torch.zeros_like(camera_values, dtype=torch.bool)
    internal_times = torch.zeros((batch_size, times_max), device=device, dtype=torch.long)
    internal_valid = torch.zeros_like(internal_times, dtype=torch.bool)
    memory_index = torch.zeros((batch_size, times_max, cameras_max), device=device, dtype=torch.long)
    memory_valid = torch.zeros_like(memory_index, dtype=torch.bool)
    for b, (cameras, times, lookup) in enumerate(zip(camera_rows, internal_rows, lookups)):
        camera_values[b, :len(cameras)] = torch.tensor(cameras, device=device)
        camera_valid[b, :len(cameras)] = True
        internal_times[b, :len(times)] = torch.tensor(times, device=device)
        internal_valid[b, :len(times)] = True
        for t_slot, time in enumerate(times):
            for c_slot, camera in enumerate(cameras):
                index = lookup.get((camera, time))
                if index is not None:
                    memory_index[b, t_slot, c_slot] = index
                    memory_valid[b, t_slot, c_slot] = True
    target_slot = torch.arange(query.shape[1], device=device)[None].expand(batch_size, -1)
    source_slot = torch.tensor(anchor_rows, device=device)
    return ObservationMetadata(camera_id, time_idx, valid, query, source, source_valid,
                               source_times, camera_values, camera_valid, internal_times,
                               internal_valid, target_slot, source_slot, memory_index, memory_valid)


def pack_synchronized_images(images, frame_indices=None, camera_ids=None):
    """Pack [B,C,T,3,H,W] in camera-major order; preserve real frame gaps."""
    if images.ndim != 6 or images.shape[3] != 3:
        raise ValueError("images must have shape [B,C,T,3,H,W]")
    b, c, t, _, h, w = images.shape
    device = images.device
    times = torch.arange(t, device=device) if frame_indices is None else integer_tensor(
        frame_indices, "frame_indices", device)
    cameras = torch.arange(c, device=device) if camera_ids is None else integer_tensor(
        camera_ids, "camera_ids", device)
    if times.ndim == 1:
        times = times[None].expand(b, -1)
    if cameras.ndim == 1:
        cameras = cameras[None].expand(b, -1)
    if times.shape != (b, t) or cameras.shape != (b, c):
        raise ValueError("frame_indices must be [T]/[B,T]; camera_ids must be [C]/[B,C]")
    return dict(images=images.reshape(b, c*t, 3, h, w),
                camera_id=cameras[:, :, None].expand(b, c, t).reshape(b, c*t),
                time_idx=times[:, None].expand(b, c, t).reshape(b, c*t),
                query_time_idx=times)
