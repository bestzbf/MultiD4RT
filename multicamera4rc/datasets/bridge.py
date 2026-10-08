"""Convert source adapters' world tracks to the multi-camera 4RC contract.

Calibration is supervision only. Images, camera identity and integer frame
labels are the model inputs; every 3D label shares observation zero's gauge.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def _integers(values, length, name):
    array = np.asarray(values)
    if array.shape != (length,) or array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValueError(f"{name} must contain one integer per observation")
    if not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all() or (array < 0).any():
        raise ValueError(f"{name} must be nonnegative integer frame/camera labels; asynchronous time cannot be quantized")
    if (array > np.iinfo(np.int64).max).any():
        raise ValueError(f"{name} labels exceed int64")
    return array.astype(np.int64, copy=True)


def _mask(values, shape, name):
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind not in "bifu":
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain boolean values")
    return array.astype(bool, copy=True)


def _frame_labels(metadata, count, cameras):
    """Preserve real frame gaps; clip-local BCT ids are never physical time."""
    if "time_idx" in metadata:
        return _integers(metadata["time_idx"], count, "time_idx")
    if "time_indices" in metadata:
        timeline = np.asarray(metadata["time_indices"])
        timeline = _integers(timeline, len(timeline), "time_indices")
        local = metadata.get("slot_time_id")
        # The EgoExo adapter publishes slot_time as its local timeline index.
        if local is None and str(metadata.get("dataset_name", "")) == "egoexo4d":
            local = metadata.get("slot_time")
        if local is not None:
            local = _integers(local, count, "slot_time_id")
            if (local >= len(timeline)).any():
                raise ValueError("slot_time_id exceeds the published source timeline")
            return timeline[local]
        if len(timeline) == count and len(np.unique(cameras)) == 1:
            return timeline
    # Interleaved MUGEN raw indices address storage slots (frame*C+stream),
    # rather than the shared physical frame axis. Retain its logical frame.
    interleaved = metadata.get("frame_interleave", 1) > 1 and "frame_indices" in metadata
    keys = (("frame_numbers", "frame_indices", "raw_frame_indices") if interleaved
            else ("frame_numbers", "raw_frame_indices", "frame_indices"))
    for key in keys:
        if key in metadata and len(np.unique(cameras)) == 1:
            value = np.asarray(metadata[key])
            if value.shape == (count,):
                return _integers(value, count, key)
    if "slot_time" in metadata:
        values = np.asarray(metadata["slot_time"], dtype=np.float64)
        fps = metadata.get("fps")
        unit = metadata.get("physical_time_unit")
        if unit == "seconds" or (fps is not None and "slot_physical_camera_id" in metadata):
            if fps is None or not np.isfinite(float(fps)) or float(fps) <= 0:
                raise ValueError("seconds require a positive fps to recover frame labels")
            values = values * float(fps)
            # Only floating point roundoff in an explicit fps conversion is
            # tolerated. Fractional asynchronous observations remain rejected.
            if not np.isfinite(values).all() or not np.allclose(values, np.rint(values), rtol=0, atol=1e-5):
                raise ValueError("physical times do not correspond to integer synchronized frames")
            values = np.rint(values)
        return _integers(values, count, "slot_time")
    if len(np.unique(cameras)) > 1:
        raise ValueError("multi-camera clips require explicit shared frame labels")
    return np.arange(count, dtype=np.int64)


def _content_size(height, width, image_size, image_mode):
    if image_mode in ("native", "pad"):
        return height, width
    if image_mode == "fit":
        scale = min(image_size[0] / height, image_size[1] / width)
        return min(image_size[0], max(1, round(height * scale))), min(image_size[1], max(1, round(width * scale)))
    return image_size


def _resize_geometry(clip, slots, image_size, K, metadata, image_mode):
    images, depths, depth_valid, transforms = [], [], [], []
    H, W = image_size
    all_depths = getattr(clip, "depths", None)
    if all_depths is not None and len(all_depths) != len(clip.images):
        raise ValueError("depths must cover every source observation")
    pixel_mode = metadata.get("pixel_coordinates", "array_index_v1")
    shift = -0.5 if pixel_mode in ("legacy_h5", "legacy_native") else 0.0
    if pixel_mode not in ("array_index_v1", "legacy_h5", "legacy_native", "dataset_native"):
        raise ValueError(f"unsupported pixel_coordinates: {pixel_mode}")
    support = metadata.get("pixel_support")
    if support is not None and len(support) != len(clip.images):
        raise ValueError("pixel_support must cover every source observation")
    for index, original in enumerate(slots):
        pixels = np.asarray(clip.images[original])
        if pixels.ndim != 3 or pixels.shape[-1] != 3 or min(pixels.shape[:2]) < 1:
            raise ValueError("adapter images must be HWC RGB")
        height, width = pixels.shape[:2]
        image = torch.from_numpy(np.array(pixels, dtype=np.float32, copy=True)).permute(2, 0, 1)
        if not torch.isfinite(image).all() or (image < 0).any() or (image > 255).any():
            raise ValueError("RGB values must be finite and in [0,255]")
        if pixels.dtype == np.uint8 or float(image.max()) > 1:
            image = image / 255
        content_h, content_w = _content_size(height, width, image_size, image_mode)
        if image_mode in ("native", "pad"):
            if height > H or width > W:
                raise ValueError(f"pad canvas {(H, W)} is smaller than native image {(height, width)}")
            # Keep native pixels at their original coordinates. Padding only
            # the bottom/right leaves the array-index calibration unchanged.
            images.append(F.pad(image, (0, W-width, 0, H-height)))
            sx = sy = 1.0
        else:
            resized = F.interpolate(image[None], size=(content_h, content_w), mode="bilinear", align_corners=False)[0]
            images.append(F.pad(resized, (0, W-content_w, 0, H-content_h)))
            sx, sy = content_w / width, content_h / height
        # Map integer array centers through align_corners=False resize.
        transform = np.array([[sx, 0, sx * (shift + 0.5) - 0.5],
                              [0, sy, sy * (shift + 0.5) - 0.5], [0, 0, 1]], dtype=np.float64)
        transforms.append(transform)
        K[index] = transform @ K[index]
        depth = np.zeros((height, width), np.float32)
        if all_depths is not None and all_depths[original] is not None:
            depth = np.asarray(all_depths[original], dtype=np.float32)
            if depth.shape != (height, width):
                raise ValueError("camera-Z depths must match each source image")
        valid = np.isfinite(depth) & (depth > 0)
        if support is not None:
            valid &= _mask(support[original], (height, width), "pixel_support")
        bounds = metadata.get("valid_image_y_bounds")
        if bounds is not None:
            valid &= (np.arange(height)[:, None] >= bounds[0]) & (np.arange(height)[:, None] < bounds[1])
        depth = torch.from_numpy(np.where(valid, depth, 0).copy())[None, None]
        mask = torch.from_numpy(valid.copy())[None, None].float()
        # Nearest sampling preserves sparse support and avoids blending holes
        # or camera-Z values on opposite sides of a depth discontinuity.
        if image_mode in ("native", "pad"):
            padding = (0, W-width, 0, H-height)
            depths.append(F.pad(depth, padding)[0, 0])
            depth_valid.append(F.pad(mask, padding)[0, 0].bool())
        else:
            padding = (0, W-content_w, 0, H-content_h)
            depths.append(F.pad(F.interpolate(depth, size=(content_h, content_w), mode="nearest-exact"), padding)[0, 0])
            depth_valid.append(F.pad(F.interpolate(mask, size=(content_h, content_w), mode="nearest-exact"), padding)[0, 0].bool())
    return torch.stack(images), torch.stack(depths), torch.stack(depth_valid), np.stack(transforms)


def _empty_targets(S, H, W, N, T, C):
    return {
        "depth": torch.zeros(S, H, W), "depth_valid": torch.zeros(S, H, W, dtype=torch.bool),
        "K": torch.eye(3).repeat(S, 1, 1), "c2w": torch.eye(4).repeat(S, 1, 1),
        "camera_valid": torch.zeros(S, dtype=torch.bool),
        "sparse_queries": torch.zeros(N, 3),
        "sparse_query_valid": torch.zeros(N, dtype=torch.bool),
        "sparse_track_id": torch.full((N,), -1, dtype=torch.long),
        "sparse_trajectory": torch.zeros(N, T, 3),
        "sparse_trajectory_valid": torch.zeros(N, T, dtype=torch.bool),
        "sparse_displacement": torch.zeros(N, T, 3),
        "sparse_displacement_valid": torch.zeros(N, T, dtype=torch.bool),
        "sparse_visibility": torch.zeros(N, T, C),
        "sparse_visibility_valid": torch.zeros(N, T, C, dtype=torch.bool),
    }


def prepare_clip(clip, image_size=(280, 504), num_queries=256, seed=0,
                 query_time_idx=None, include_visibility=True, image_mode="fit",
                 all_track_queries=False):
    """Build one rectangular sample from an adapter's ``UnifiedClip``.

    Sparse queries are unique source-observation/track pairs, sampled without
    replacement. A short pool is padded with finite zeros and false masks.
    Occluded points retain valid world-trajectory supervision. Same-time
    per-camera copies must describe identical track identities and positions.
    ``image_mode='pad'`` preserves native pixels and pads the bottom/right to
    image_size; padded depth pixels are invalid and cannot become source queries.
    """
    if image_mode != "native" and (image_size is None or len(image_size) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in image_size)):
        raise ValueError("image_size must be positive integer (height,width)")
    if isinstance(num_queries, bool) or int(num_queries) != num_queries or num_queries < 0:
        raise ValueError("num_queries must be a nonnegative integer")
    if image_mode not in ("resize", "fit", "pad", "native"):
        raise ValueError("image_mode must be 'native', 'fit', 'resize' or 'pad'")
    N = int(num_queries)
    old_count = len(clip.images)
    if old_count < 1:
        raise ValueError("clip has no images")
    metadata = getattr(clip, "metadata", None) or {}
    valid_slot = _mask(metadata.get("valid_slot", np.ones(old_count, bool)), (old_count,), "valid_slot")
    slots = np.flatnonzero(valid_slot)
    if not len(slots):
        raise ValueError("clip has no actual observations")
    if image_mode == "native":
        sizes = [np.asarray(clip.images[i]).shape[:2] for i in slots]
        H, W = (max(28, (max(size[axis] for size in sizes) + 13) // 14 * 14) for axis in (0, 1))
    else:
        H, W = map(int, image_size)
    cameras = _integers(metadata.get("slot_physical_camera_id", metadata.get("slot_camera", np.zeros(old_count, int))), old_count, "camera_id")
    times = _frame_labels(metadata, old_count, cameras)
    cameras, times = cameras[slots], times[slots]
    S = len(slots)
    if len(set(zip(cameras.tolist(), times.tolist()))) != S:
        raise ValueError("duplicate valid (camera, time) observation")
    unique_times = np.unique(times)
    target_times = (unique_times.copy() if query_time_idx is None
                    else _integers(query_time_idx, len(query_time_idx), "query_time_idx"))
    if not len(target_times) or len(np.unique(target_times)) != len(target_times) or not np.isin(target_times, unique_times).all():
        raise ValueError("query_time_idx must be unique observed frame labels")
    camera_values = np.unique(cameras)
    T, C = len(target_times), len(camera_values)
    targets = _empty_targets(S, H, W, N, T, C)

    K = np.asarray(clip.intrinsics, dtype=np.float64)
    E = np.asarray(clip.extrinsics, dtype=np.float64)
    if K.shape != (old_count, 3, 3) or E.shape != (old_count, 4, 4):
        raise ValueError("adapter calibration must be K[S,3,3] and W2C[S,4,4]")
    K, E = K[slots].copy(), E[slots].copy()
    convention = metadata.get("extrinsics_convention", metadata.get("pose_convention", "w2c"))
    convention = {"world_to_camera": "w2c", "camera_to_world": "c2w"}.get(convention, convention)
    if convention not in ("w2c", "c2w"):
        raise ValueError("extrinsics must declare OpenCV w2c or c2w")
    if not np.isfinite(K).all() or not np.isfinite(E).all():
        raise ValueError("calibration contains nonfinite values")
    if not np.allclose(E[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("extrinsics must be homogeneous rigid transforms")
    R = E[:, :3, :3]
    if not np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=5e-3) or not np.allclose(np.linalg.det(R), 1, atol=5e-3):
        raise ValueError("camera rotations are not rigid")
    if (K[:, [0, 1], [0, 1]] <= 0).any() or not np.allclose(K[:, 2], [0, 0, 1], atol=1e-6):
        raise ValueError("invalid pinhole intrinsic matrices")
    if not np.allclose(K[:, 0, 1], 0, atol=1e-6) or not np.allclose(K[:, 1, 0], 0, atol=1e-6):
        raise ValueError("the model requires zero-skew undistorted pinhole images")
    w2c = E if convention == "w2c" else np.linalg.inv(E)
    reference = w2c[0].copy()
    c2w = reference[None] @ np.linalg.inv(w2c)
    c2w[0] = np.eye(4)
    images, depth, depth_valid, transforms = _resize_geometry(clip, slots, (H, W), K, metadata, image_mode)
    camera_valid = _mask(metadata.get("camera_valid", np.ones(old_count, bool)), (old_count,), "camera_valid")[slots]
    if not camera_valid[0]:
        raise ValueError("reference observation needs an authoritative pose")
    targets.update(depth=depth, depth_valid=depth_valid, K=torch.from_numpy(K.astype(np.float32)),
                   c2w=torch.from_numpy(c2w.astype(np.float32)), camera_valid=torch.from_numpy(camera_valid))

    arrays = [getattr(clip, key, None) for key in ("trajs_3d_world", "trajs_2d", "valids", "visibs")]
    if any(value is not None for value in arrays) and not all(value is not None for value in arrays):
        raise ValueError("tracks require world XYZ, pixel UV, geometry validity and visibility together")
    if all(value is not None for value in arrays) and (N or all_track_queries):
        if C > 1 and (getattr(clip, "dataset_name", "") == "mugen_h5" or "h5_group" in metadata):
            # MUGEN adapters have one camera stream per persistent track pool.
            # Concatenating independently sampled pools does not establish
            # cross-camera identity, even if their column counts coincide.
            raise ValueError("MUGEN single-camera track pools cannot be merged into a multi-camera clip")
        world, uv = np.asarray(arrays[0], dtype=np.float64), np.asarray(arrays[1], dtype=np.float64)
        if world.ndim != 3 or world.shape[0] != old_count or world.shape[-1] != 3:
            raise ValueError("world trajectories must be [S,num_tracks,3]")
        tracks = world.shape[1]
        if uv.shape != (old_count, tracks, 2):
            raise ValueError("pixel trajectories must be [S,num_tracks,2]")
        valid = _mask(arrays[2], (old_count, tracks), "valids")[slots]
        visible = _mask(arrays[3], (old_count, tracks), "visibs")[slots]
        visibility_known = getattr(clip, "visibility_valids", None)
        known = (valid.copy() if visibility_known is None else
                 _mask(visibility_known, (old_count, tracks), "visibility_valids")[slots])
        if metadata.get("has_visibility") is False:
            known[:] = False
        world, uv = world[slots], uv[slots].copy()
        valid &= np.isfinite(world).all(-1)
        # Every observation is expressed in one fixed reference coordinate frame.
        world = world @ reference[:3, :3].T + reference[:3, 3]
        for index, affine in enumerate(transforms):
            uv[index] = uv[index] @ affine[:2, :2].T + affine[:2, 2]
        pooled = np.zeros((len(unique_times), tracks, 3), dtype=np.float64)
        support = np.zeros((len(unique_times), tracks), dtype=bool)
        for time_id, time in enumerate(unique_times):
            for observation in np.flatnonzero(times == time):
                both = support[time_id] & valid[observation]
                if both.any() and not np.allclose(pooled[time_id, both], world[observation, both], rtol=1e-4, atol=1e-5):
                    raise ValueError("same-time world trajectories disagree across cameras; track identity pools may differ")
                fresh = valid[observation] & ~support[time_id]
                pooled[time_id, fresh] = world[observation, fresh]
                support[time_id] |= valid[observation]
        eligible = _mask(metadata.get("track_source_eligible", np.ones(tracks, bool)), (tracks,), "track_source_eligible")
        source_ok = valid & visible & known & np.isfinite(uv).all(-1) & eligible[None]
        if "track_source_valid" in metadata:
            source_ok &= _mask(metadata["track_source_valid"], (old_count, tracks), "track_source_valid")[slots]
        relative_w2c = np.linalg.inv(c2w)
        source_depth = np.einsum("sni,si->sn", world, relative_w2c[:, 2, :3]) + relative_w2c[:, 2, 3, None]
        positive_depth = np.isfinite(source_depth) & (source_depth > 0)
        source_ok &= (uv[..., 0] >= 0) & (uv[..., 0] <= W-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H-1)
        # Sources can be cropped/resized, but a point outside the original
        # image cannot become a valid query merely by stretching its boundary.
        for index, original in enumerate(slots):
            height, width = np.asarray(clip.images[original]).shape[:2]
            original_uv = np.asarray(arrays[1], dtype=np.float64)[original]
            shift = -.5 if metadata.get("pixel_coordinates") in ("legacy_h5", "legacy_native") else 0
            original_uv = original_uv + shift
            source_ok[index] &= (original_uv[:, 0] >= 0) & (original_uv[:, 0] <= width-1) & (original_uv[:, 1] >= 0) & (original_uv[:, 1] <= height-1)
            if metadata.get("valid_image_y_bounds") is not None:
                bounds = metadata["valid_image_y_bounds"]
                source_ok[index] &= (original_uv[:, 1] >= bounds[0]) & (original_uv[:, 1] < bounds[1])
            if metadata.get("pixel_support") is not None:
                nearest = np.nan_to_num(original_uv, nan=0, posinf=0, neginf=0)
                nearest[:, 0] = np.clip(nearest[:, 0], 0, width-1)
                nearest[:, 1] = np.clip(nearest[:, 1], 0, height-1)
                nearest = np.rint(nearest).astype(np.int64)
                source_ok[index] &= np.asarray(metadata["pixel_support"][original], bool)[nearest[:, 1], nearest[:, 0]]
        if source_ok.any() and not (source_ok & positive_depth).any():
            raise ValueError("all visible valid source tracks are behind their cameras; check OpenGL/OpenCV extrinsics conventions")
        source_ok &= positive_depth
        candidates = np.argwhere(source_ok)
        if all_track_queries:
            N = len(candidates)
            targets.update({k: v for k, v in _empty_targets(S, H, W, N, T, C).items()
                            if k.startswith("sparse_")})
        elif len(candidates):
            candidates = candidates[np.random.default_rng(seed).permutation(len(candidates))[:N]]
        count = len(candidates)
        if count:
            source, track = candidates.T
            query = np.concatenate((source[:, None], uv[source, track]), axis=1)
            target_id = np.searchsorted(unique_times, target_times)
            truth = pooled[target_id[:, None], track[None]].transpose(1, 0, 2).copy()
            truth_valid = support[target_id[:, None], track[None]].T.copy()
            source_position = world[source, track]
            temporal_allowed = np.ones((count, T), dtype=bool)
            if metadata.get("has_temporal_supervision") is False:
                dynamic = getattr(clip, "track_is_dynamic", None)
                dynamic_known = getattr(clip, "track_dynamic_known", None)
                static = np.zeros(tracks, dtype=bool)
                if dynamic is not None:
                    d = _mask(dynamic, (tracks,), "track_is_dynamic")
                    k = np.ones(tracks, bool) if dynamic_known is None else _mask(dynamic_known, (tracks,), "track_dynamic_known")
                    static = ~d & k
                # Only explicit known-static tracks may extend sparse map
                # geometry across time. Unknown motion is supervised at source.
                truth[static[track]] = source_position[static[track], None]
                truth_valid[static[track]] = True
                temporal_allowed = static[track, None] | (times[source, None] == target_times[None])
                truth_valid &= temporal_allowed
            truth = np.where(truth_valid[..., None], truth, 0)
            displacement = np.where(truth_valid[..., None], truth - source_position[:, None], 0)
            targets["sparse_queries"][:count] = torch.from_numpy(query.astype(np.float32))
            targets["sparse_query_valid"][:count] = True
            targets["sparse_track_id"][:count] = torch.from_numpy(track.copy())
            for key, value in (("sparse_trajectory", truth), ("sparse_displacement", displacement)):
                targets[key][:count] = torch.from_numpy(value.astype(np.float32))
                targets[f"{key}_valid"][:count] = torch.from_numpy(truth_valid.copy())
            if include_visibility:
                for target, time in enumerate(target_times):
                    for camera, label in enumerate(camera_values):
                        observation = np.flatnonzero((times == time) & (cameras == label))
                        if len(observation):
                            slot = observation[0]
                            # A reliable occlusion label can exist without a
                            # trusted target XYZ (e.g. conservative depth
                            # tracks). Visibility is an independent contract.
                            vis_valid = known[slot, track] & temporal_allowed[:, target]
                            targets["sparse_visibility"][:count, target, camera] = torch.from_numpy(visible[slot, track].astype(np.float32))
                            targets["sparse_visibility_valid"][:count, target, camera] = torch.from_numpy(vis_valid)
    return {
        "images": images, "camera_id": torch.from_numpy(cameras), "time_idx": torch.from_numpy(times),
        "query_time_idx": torch.from_numpy(target_times), "observation_valid": torch.ones(S, dtype=torch.bool),
        "clip_id": f"{getattr(clip, 'dataset_name', 'dataset')}/{getattr(clip, 'sequence_name', 'clip')}",
        "targets": targets,
    }


__all__ = ["prepare_clip"]
