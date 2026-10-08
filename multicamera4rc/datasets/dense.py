"""Exact dense Kubric labels from rendered depth, instance IDs and rigid poses.

The H5 point pool is never rasterized, interpolated, or treated as dense GT.
Only compact renderer metadata is loaded on CPU. Full source-pixel trajectories
are constructed after source selection, on the training device, without query
chunks. Calibration and object poses remain supervision, never model inputs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ..geometry import quaternion_xyzw_to_matrix, unproject_depth
from ..metadata import gather_observations


DENSE_TARGET_VERSION = 2
KUBRIC_BACKGROUND_POLICIES = ("auto", "objects_ground", "all")


def attach_kubric_dense_recipe(clip, sample, *, image_mode, include_visibility=True,
                               background_policy="auto"):
    """Load authoritative per-pixel material identity and XYZW object poses."""
    import h5py
    from .bridge import _content_size

    if background_policy not in KUBRIC_BACKGROUND_POLICIES:
        raise ValueError(f"unknown Kubric background policy: {background_policy}")
    metadata = clip.metadata
    valid_slots = np.flatnonzero(np.asarray(metadata.get("valid_slot", [True] * len(clip.images))))
    times = sample["time_idx"].numpy()
    cameras = sample["camera_id"].numpy()
    timeline = sample["query_time_idx"].numpy()
    height, width = sample["images"].shape[-2:]
    segmentations = []
    with h5py.File(metadata["h5_path"], "r") as f:
        # Only this audited generator guarantees background 0 = ground + HDRI
        # dome and a Blender-world ground plane at z=0. Other schemas retain
        # their original masks; explicit filtering fails on unknown schemas.
        known_background = (f.attrs.get("schema_version") == "movi_tracking_v3"
                            and f.attrs.get("coordinate_system") == "opencv_camera__blender_world")
        if background_policy == "objects_ground" and not known_background:
            raise ValueError("objects_ground requires the audited movi_tracking_v3 Blender-world schema")
        filter_background = known_background and background_policy != "all"
        required = {"segmentation", "object_trajectories", "object_is_dynamic", "depth"}
        missing = required - set(f)
        if missing:
            raise ValueError(f"dense Kubric supervision lacks renderer labels: {sorted(missing)}")
        poses = np.asarray(f["object_trajectories"][:, timeline], dtype=np.float32)
        dynamic = np.asarray(f["object_is_dynamic"], dtype=bool)
        if poses.shape != (len(dynamic), len(timeline), 7) or not np.isfinite(poses).all():
            raise ValueError("object trajectories must be finite [objects,times,position+quaternion_xyzw]")
        if (np.linalg.norm(poses[..., 3:], axis=-1) < 1e-8).any():
            raise ValueError("renderer supplied a zero object quaternion")
        for slot, time, camera in zip(valid_slots, times, cameras):
            ids = np.asarray(f["segmentation"][int(time), int(camera)], dtype=np.int64)
            if ids.shape != np.asarray(clip.images[slot]).shape[:2]:
                raise ValueError("instance segmentation must match the source RGB dimensions")
            if ids.min() < 0 or ids.max() > len(dynamic):
                raise ValueError("instance IDs must be background 0 or object indices 1..O")
            content_hw = _content_size(*ids.shape, (height, width), image_mode)
            ids = torch.from_numpy(ids.copy())[None, None]
            if tuple(ids.shape[-2:]) != content_hw:
                ids = F.interpolate(ids.double(), size=content_hw, mode="nearest-exact").long()
            segmentations.append(F.pad(ids, (0, width-content_hw[1], 0, height-content_hw[0]), value=-1)[0, 0])
        tolerance = torch.tensor([float(f.attrs.get("depth_absolute_tolerance_m", 0.03)),
                                  float(f.attrs.get("depth_relative_tolerance", 0.01))])
        radius = int(f.attrs.get("visibility_search_radius_px", 1))
        # The native generator rounds renderer UV (array UV + .5) for its
        # visibility neighborhood. Preserve that convention, independently
        # of the corrected array-index K used by model geometry.
        visibility_pixel_offset = float(f.attrs.get("sampling_pixel_center_offset", .5))
        if (tolerance < 0).any() or not torch.isfinite(tolerance).all() or not 0 <= radius <= 4:
            raise ValueError("invalid renderer visibility tolerance/radius")

    reference = torch.as_tensor(np.asarray(clip.extrinsics[valid_slots[0]]).copy(), dtype=torch.float32)
    if metadata.get("extrinsics_convention", "w2c") in ("c2w", "camera_to_world"):
        reference = torch.linalg.inv(reference)
    # Plane equation in the FIRST CAMERA frame used by prepare_clip. Applying
    # a world-Z threshold directly to reference-camera coordinates is wrong.
    normal = reference[:3, 2]
    ground_plane = torch.cat((normal, -(normal @ reference[:3, 3]).reshape(1)))
    poses = torch.from_numpy(poses).transpose(0, 1)
    rotation = reference[:3, :3] @ quaternion_xyzw_to_matrix(poses[..., 3:])
    translation = poses[..., :3] @ reference[:3, :3].T + reference[:3, 3]
    # Instance 0 is the stationary ground/background in this renderer schema.
    rotation = torch.cat((torch.eye(3).expand(len(timeline), 1, 3, 3), rotation), 1)
    translation = torch.cat((torch.zeros(len(timeline), 1, 3), translation), 1)
    target = {k: v for k, v in sample["targets"].items() if not k.startswith("sparse_")}
    target.update(dense_segmentation=torch.stack(segmentations), dense_object_R=rotation,
                  dense_object_t=translation,
                  dense_object_dynamic=torch.cat((torch.zeros(1, dtype=torch.bool), torch.from_numpy(dynamic))),
                  dense_object_valid=torch.ones(len(dynamic)+1, dtype=torch.bool),
                  dense_visibility_tolerance=tolerance,
                  dense_visibility_radius=torch.tensor(radius),
                  dense_visibility_pixel_offset=torch.tensor(visibility_pixel_offset),
                  dense_include_visibility=torch.tensor(bool(include_visibility)),
                  dense_filter_background=torch.tensor(filter_background),
                  dense_ground_plane=ground_plane,
                  dense_ground_tolerance=torch.tensor(.05))
    sample["targets"] = target


@torch.no_grad()
def materialize_dense_targets(batch, source_indices, *, normalize_scene=True):
    """Build [B,Q,T,H,W,3] GT for every valid pixel of each decoded source.

    A single scale per clip makes the mean valid base-point distance 1 (4RC
    section 3.4). Depth, camera/object translations, positions and displacement
    share this scale; directions, intrinsics and frame gaps are unchanged.
    """
    recipe = batch["targets"]
    if "dense_syn4d_vertices" in recipe:
        from .syn4d_dense import materialize_syn4d_targets
        return materialize_syn4d_targets(batch, source_indices, normalize_scene=normalize_scene)
    if "dense_partial" in recipe:
        from .partial_dense import materialize_partial_dense_targets
        return materialize_partial_dense_targets(batch, source_indices, normalize_scene=normalize_scene)
    if "dense_segmentation" not in recipe:
        raise ValueError("dense supervision needs authoritative per-pixel renderer metadata")
    target = {k: v for k, v in recipe.items() if not k.startswith("dense_")}
    depth, K, camera = target["depth"].float(), target["K"].float(), target["c2w"].float()
    base_all = unproject_depth(depth, K, camera)
    observation_valid = batch["observation_valid"] & target["camera_valid"].bool()
    pixel_valid = target["depth_valid"].bool() & observation_valid[..., None, None]
    excluded_background = torch.zeros_like(pixel_valid)
    if "dense_filter_background" in recipe:
        plane = recipe["dense_ground_plane"].float()
        height = (base_all * plane[:, None, None, None, :3]).sum(-1) + plane[:, None, None, None, 3]
        excluded_background = (pixel_valid & recipe["dense_filter_background"][:, None, None, None]
                               & (recipe["dense_segmentation"] == 0)
                               & (height.abs() > recipe["dense_ground_tolerance"][:, None, None, None]))
        pixel_valid = pixel_valid & ~excluded_background
    # This SAME mask controls the scale, all geometric targets, temporal/static
    # labels and visibility. RGB is never masked. Object IDs >0 are retained
    # regardless of their distance or height, including flying objects.
    distance = torch.linalg.vector_norm(base_all, dim=-1)
    count = pixel_valid.sum(dim=(1, 2, 3))
    if (count == 0).any():
        raise ValueError("dense scene has no valid calibrated depth pixels")
    scale = (torch.where(pixel_valid, distance, 0).sum(dim=(1, 2, 3)) / count).clamp_min(1e-8)
    if not normalize_scene:
        scale = torch.ones_like(scale)
    depth = depth / scale[:, None, None, None]
    camera = camera.clone()
    camera[..., :3, 3] /= scale[:, None, None]
    base_all = base_all / scale[:, None, None, None, None]
    object_R = recipe["dense_object_R"].float()
    object_t = recipe["dense_object_t"].float() / scale[:, None, None, None]
    source_base = gather_observations(base_all, source_indices)
    source_valid = gather_observations(pixel_valid, source_indices)
    source_ids = gather_observations(recipe["dense_segmentation"], source_indices)
    source_times = gather_observations(batch["time_idx"], source_indices)
    timeline = batch["query_time_idx"]
    matches = source_times[..., None] == timeline[:, None]
    source_observation_valid = gather_observations(observation_valid, source_indices)
    if ((~matches.any(-1)) & source_observation_valid).any():
        raise ValueError("dense source times must occur in the renderer pose timeline")
    source_slot = matches.long().argmax(-1)
    b, q, h, w = source_ids.shape
    times = timeline.shape[1]
    if object_R.shape[:2] != (b, times) or object_t.shape[:2] != (b, times):
        raise ValueError("object pose timeline does not match query_time_idx")
    ids = source_ids.clamp_min(0)
    if (ids >= object_R.shape[2]).any():
        raise ValueError("source instance ID exceeds the pose table")
    bi = torch.arange(b, device=depth.device)[:, None, None, None]
    source_valid &= (source_ids >= 0) & recipe["dense_object_valid"][bi, ids]
    R_source = object_R[bi, source_slot[:, :, None, None], ids]
    t_source = object_t[bi, source_slot[:, :, None, None], ids]
    local = torch.einsum("bqhwij,bqhwi->bqhwj", R_source, source_base-t_source)
    ti = torch.arange(times, device=depth.device)[None, None, :, None, None]
    R_target = object_R[bi[:, :, None], ti, ids[:, :, None]]
    t_target = object_t[bi[:, :, None], ti, ids[:, :, None]]
    xyz = torch.einsum("bqthwij,bqhwj->bqthwi", R_target, local) + t_target
    identity = matches[..., None, None, None]
    xyz = torch.where(identity, source_base[:, :, None], xyz)
    motion_valid = source_valid[:, :, None].expand(b, q, times, h, w)
    xyz = torch.where(motion_valid[..., None], xyz, 0)
    displacement = torch.where(motion_valid[..., None], xyz-source_base[:, :, None], 0)
    dynamic = recipe["dense_object_dynamic"][bi, ids] & source_valid
    static = source_valid & ~dynamic & (displacement.abs().amax(dim=(2, 5)) < 1e-5)
    origin = camera[..., :3, 3][:, :, None, None].expand(*depth.shape, 3)
    direction = unproject_depth(torch.ones_like(depth), K, camera)-origin
    # Released 4RC/DA3 and OmniX use [world direction, world camera origin].
    target.update(depth=depth, depth_valid=pixel_valid, c2w=camera,
                  excluded_background=excluded_background,
                  ray=torch.cat((direction, origin), -1), ray_valid=pixel_valid,
                  trajectory=xyz, trajectory_valid=motion_valid,
                  displacement=displacement, displacement_valid=motion_valid,
                  trajectory_foreground=dynamic[:, :, None].expand_as(motion_valid),
                  static_mask=static, scene_scale=scale)
    target["source_pixel_valid"] = source_valid
    if recipe["dense_include_visibility"].any():
        if not recipe["dense_include_visibility"].all():
            raise ValueError("dense batches must share include_visibility")
        visible, known = _dense_visibility(batch, recipe, target, source_ids, motion_valid, scale)
        target.update(visibility=visible, visibility_valid=known)
    return target


def _dense_visibility(batch, recipe, target, source_ids, motion_valid, scale):
    """Use renderer instance AND radial-depth tests, including occluded points."""
    xyz = target["trajectory"]
    b, q, times, height, width, _ = xyz.shape
    camera_values = [batch["camera_id"][i, batch["observation_valid"][i]].unique(sorted=True) for i in range(b)]
    cameras = max(len(v) for v in camera_values)
    shape = (b, q, times, cameras, height, width)
    visible = torch.zeros(shape, dtype=torch.bool, device=xyz.device)
    known = torch.zeros_like(visible)
    if not (recipe["dense_visibility_radius"] == recipe["dense_visibility_radius"][0]).all():
        raise ValueError("dense batches must share renderer visibility radius")
    radius = int(recipe["dense_visibility_radius"][0])
    v, u = torch.meshgrid(torch.arange(height, device=xyz.device),
                          torch.arange(width, device=xyz.device), indexing="ij")
    for i in range(b):
        for t in range(times):
            points = xyz[i, :, t].reshape(-1, 3)
            instance = source_ids[i].reshape(-1)
            for c, physical_camera in enumerate(camera_values[i]):
                observations = ((batch["camera_id"][i] == physical_camera)
                                & (batch["time_idx"][i] == batch["query_time_idx"][i, t])
                                & batch["observation_valid"][i]
                                & target["camera_valid"][i]).nonzero().flatten()
                if not len(observations):
                    continue
                if len(observations) != 1:
                    raise ValueError("dense visibility has duplicate camera/time observations")
                s = int(observations[0])
                camera, K = target["c2w"][i, s], target["K"][i, s]
                camera_points = (points-camera[:3, 3]) @ camera[:3, :3]
                z = camera_points[:, 2]
                projected = camera_points @ K.T
                uv = projected[:, :2] / z.clamp_min(1e-8)[:, None]
                point_range = torch.linalg.vector_norm(camera_points, dim=-1)
                range_image = target["depth"][i, s] * torch.sqrt(
                    ((u-K[0, 2])/K[0, 0]).square() + ((v-K[1, 2])/K[1, 1]).square() + 1)
                defined = torch.isfinite(uv).all(-1) & (z > 1e-4/scale[i])
                pixel = torch.nan_to_num(uv+recipe["dense_visibility_pixel_offset"][i], nan=-1e6, posinf=1e6, neginf=-1e6).round().long()
                seen = torch.zeros_like(defined)
                absolute, relative = recipe["dense_visibility_tolerance"][i]
                absolute = absolute / scale[i]
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        x, y = pixel[:, 0]+dx, pixel[:, 1]+dy
                        in_bounds = defined & (x >= 0) & (x < width) & (y >= 0) & (y < height)
                        x, y = x.clamp(0, width-1), y.clamp(0, height-1)
                        radial_depth = range_image[y, x]
                        same_instance = recipe["dense_segmentation"][i, s, y, x] == instance
                        tolerance = torch.maximum(absolute, relative*radial_depth)
                        seen |= in_bounds & same_instance & target["depth_valid"][i, s, y, x] & ((point_range-radial_depth).abs() <= tolerance)
                mask = motion_valid[i, :, t]
                visible[i, :, t, c] = seen.reshape(q, height, width) & mask
                known[i, :, t, c] = mask
    return visible, known
