"""Dense Syn4D mesh trajectories, with missing motion explicitly unknown.

Compact face IDs are surface identities, not a sparse particle pool. Every
released valid pixel participates. Full Q x T tracks are built only after
source selection; no S x T image-sized track bank is loaded on the CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..geometry import unproject_depth
from ..metadata import gather_observations


def _read_camera_masks(scene, camera, stems, shapes, face_offsets):
    import cv2
    from safetensors import safe_open
    track = scene / 'tracking_safetensors' / f'{camera}.safetensors'
    index_path = track.with_name(f'{camera}_index.json')
    index = json.loads(index_path.read_text())
    if index.get('format') != 'syn4d_compact_faceid_safetensors_v1':
        raise ValueError('unsupported Syn4D tracking format')
    records = index['record_key_to_id']
    env_path = scene / 'env_mask_safetensors' / f'{camera}.safetensors'
    env_index_path = env_path.with_name(f'{camera}_index.json')
    env_index = json.loads(env_index_path.read_text()) if env_index_path.is_file() else None
    if env_index is not None and env_index.get('format') != 'syn4d_env_mask_png_bytes_safetensors_v1':
        raise ValueError('unsupported Syn4D environment mask format')
    # Handles are scoped to this clip; no inherited mmap/video handles in workers.
    with safe_open(str(track), framework='np') as handle:
        offsets = handle.get_tensor('offsets')
        sizes = handle.get_tensor('record_hw')
        if len(offsets) != len(sizes) + 1 or (np.diff(offsets) < 0).any():
            raise ValueError('malformed Syn4D record offsets')
        flat_slice = handle.get_slice('valid_flat_idx_values')
        face_slice = handle.get_slice('face_ids_values')
        if int(offsets[-1]) != flat_slice.get_shape()[0] or flat_slice.get_shape() != face_slice.get_shape():
            raise ValueError('Syn4D record sizes do not match shard tensors')
        result = []
        for stem, shape in zip(stems, shapes):
            h, w = shape
            env = np.zeros(shape, bool)
            env_png = scene / 'exr_layers/masks' / camera / f'{stem}_env.png'
            if env_png.is_file():
                raw = cv2.imread(str(env_png), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise ValueError(f'cannot decode {env_png}')
                env = raw > 0
            elif env_index is not None and stem in env_index['frame_to_id']:
                with safe_open(str(env_path), framework='np') as masks:
                    bounds = masks.get_tensor('offsets')
                    i = int(env_index['frame_to_id'][stem])
                    if not 0 <= i < len(bounds) - 1:
                        raise ValueError('invalid Syn4D env-mask record id')
                    encoded = masks.get_slice('png_bytes_values')[int(bounds[i]):int(bounds[i + 1])]
                    raw = cv2.imdecode(np.asarray(encoded, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if raw is None:
                        raise ValueError(f'cannot decode environment mask for {stem}')
                    env = raw > 0
            if env.shape != shape:
                raise ValueError('Syn4D environment mask dimensions differ from RGB/depth')
            face_map = np.full(h * w, -1, np.int64)
            occupied = np.zeros(h * w, bool)
            ambiguous = np.zeros(h * w, bool)
            for key, record_id in records.items():
                frame, mask = key.split('/', 1)
                if frame != stem:
                    continue
                if not 0 <= record_id < len(sizes) or tuple(sizes[record_id]) != shape:
                    raise ValueError(f'invalid Syn4D tracking record dimensions: {key}')
                start, end = int(offsets[record_id]), int(offsets[record_id + 1])
                flat = np.asarray(flat_slice[start:end], np.int64)
                ids = np.asarray(face_slice[start:end], np.int64)
                if flat.ndim != 1 or ids.shape != flat.shape or ((flat < 0) | (flat >= h * w)).any():
                    raise ValueError(f'invalid Syn4D pixel indices: {key}')
                if len(np.unique(flat)) != len(flat):
                    raise ValueError(f'duplicate Syn4D source pixels in {key}')
                ambiguous[flat] |= occupied[flat]
                occupied[flat] = True
                if mask in face_offsets:
                    offset, count = face_offsets[mask]
                    valid = (ids >= 0) & (ids < count)
                    face_map[flat[valid]] = ids[valid] + offset
            # Missing body/mesh pixels must never become static environment.
            env.reshape(-1)[occupied] = False
            face_map[ambiguous] = -1
            result.append((face_map.reshape(shape), env))
    return result


def attach_syn4d_dense_recipe(clip, sample, *, image_mode, include_visibility=True):
    from .bridge import _content_size
    m = clip.metadata
    H, W = sample['images'].shape[-2:]
    T = len(sample['query_time_idx'])
    vertices, faces, offsets = [], [], {}
    nv = nf = 0
    for name, (v, f) in sorted(m['syn4d_meshes'].items()):
        if v.shape[0] != T:
            raise ValueError('Syn4D mesh timeline differs from model query timeline')
        vertices.append(v); faces.append(f + nv); offsets[name] = (nf, len(f))
        nv += v.shape[1]; nf += len(f)
    v = np.concatenate(vertices, axis=1) if vertices else np.zeros((T, 1, 3), np.float32)
    f = np.concatenate(faces) if faces else np.zeros((1, 3), np.int64)
    E = np.linalg.inv(np.asarray(clip.extrinsics[0], np.float64))
    v = (v @ E[:3, :3].T + E[:3, 3]).astype(np.float32)
    mappings = []
    if m['syn4d_load_tracking']:
        stems = m['syn4d_frame_stems']
        for start in range(0, len(stems), T):
            camera = stems[start].rsplit('_', 1)[0]
            mappings.extend(_read_camera_masks(Path(m['syn4d_scene']), camera, stems[start:start + T],
                            [x.shape[:2] for x in clip.images[start:start + T]], offsets))
    else:
        mappings = [(np.full(x.shape[:2], -1, np.int64), np.zeros(x.shape[:2], bool)) for x in clip.images]
    face_maps, environments, content_sizes = [], [], []
    for face_map, env in mappings:
        content = _content_size(*face_map.shape, (H, W), image_mode)
        content_sizes.append(content)
        fields = []
        for array, fill in ((face_map, -1), (env, 0)):
            tensor = torch.from_numpy(array.copy()).double()[None, None]
            if tuple(array.shape) != content:
                tensor = F.interpolate(tensor, size=content, mode='nearest-exact')
            fields.append(F.pad(tensor, (0, W-content[1], 0, H-content[0]), value=fill)[0, 0])
        face_maps.append(fields[0].long()); environments.append(fields[1].bool())
    face_maps = torch.stack(face_maps)
    target = {k: value for k, value in sample['targets'].items() if not k.startswith('sparse_')}
    target.update(dense_syn4d_vertices=torch.from_numpy(v), dense_syn4d_faces=torch.from_numpy(f),
                  dense_syn4d_face_valid=torch.arange(len(f)) < nf,
                  dense_syn4d_face_map=face_maps, dense_syn4d_environment=torch.stack(environments),
                  dense_syn4d_content_hw=torch.tensor(content_sizes, dtype=torch.long),
                  dense_motion_source_valid=(face_maps >= 0).flatten(1).any(1),
                  dense_syn4d_include_visibility=torch.tensor(bool(include_visibility)))
    sample['targets'] = target
    sample['syn4d'] = dict(frame_policy=m['syn4d_frame_policy'],
                           raw_time_indices=m['time_indices'], mesh_masks=sorted(offsets),
                           missing_meshes=m['syn4d_missing_meshes'],
                           visibility_source='depth_consistency' if include_visibility else 'disabled',
                           pixels_with_face=int((face_maps >= 0).sum()))


def _normal(triangle):
    normal = torch.linalg.cross(triangle[..., 1, :] - triangle[..., 0, :],
                               triangle[..., 2, :] - triangle[..., 0, :], dim=-1)
    length = normal.norm(dim=-1)
    return normal / length.clamp_min(1e-6)[..., None], length


@torch.no_grad()
def materialize_syn4d_targets(batch, source_indices, *, normalize_scene=True):
    recipe = batch['targets']
    target = {k: v for k, v in recipe.items() if not k.startswith('dense_')}
    depth, K, camera = (target[k].float() for k in ('depth', 'K', 'c2w'))
    base = unproject_depth(depth, K, camera)
    valid = target['depth_valid'].bool() & (batch['observation_valid'] & target['camera_valid'])[..., None, None]
    count = valid.sum((1, 2, 3))
    if (count == 0).any():
        raise ValueError('Syn4D clip has no valid calibrated depth')
    scale = (torch.where(valid, base.norm(dim=-1), 0).sum((1, 2, 3)) / count).clamp_min(1e-8)
    if not normalize_scene:
        scale = torch.ones_like(scale)
    source_base = gather_observations(base, source_indices)
    source_valid = gather_observations(valid, source_indices)
    face_map = gather_observations(recipe['dense_syn4d_face_map'], source_indices)
    environment = gather_observations(recipe['dense_syn4d_environment'], source_indices) & source_valid
    times = batch['query_time_idx']
    identity = gather_observations(batch['time_idx'], source_indices)[..., None] == times[:, None]
    if not identity.any(-1).all():
        raise ValueError('Syn4D source times must be in the target timeline')
    source_t = identity.long().argmax(-1)
    B, Q, H, W = source_valid.shape
    T = times.shape[1]
    vertices, faces = recipe['dense_syn4d_vertices'].float(), recipe['dense_syn4d_faces'].long()
    bi = torch.arange(B, device=depth.device)[:, None, None, None]
    if (face_map >= faces.shape[1]).any():
        raise ValueError('Syn4D face ID exceeds the mesh table')
    known = (face_map >= 0) & source_valid & recipe['dense_syn4d_face_valid'][bi, face_map.clamp_min(0)]
    vi = faces[bi, face_map.clamp_min(0)]
    tri = vertices[bi[..., None], source_t[:, :, None, None, None], vi]
    a, edge0, edge1 = tri[..., 0, :], tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :]
    offset = source_base - a
    d00, d01, d11 = (edge0 * edge0).sum(-1), (edge0 * edge1).sum(-1), (edge1 * edge1).sum(-1)
    d20, d21 = (offset * edge0).sum(-1), (offset * edge1).sum(-1)
    denom = d00 * d11 - d01 * d01
    b1 = (d11 * d20 - d01 * d21) / denom.clamp_min(1e-12)
    b2 = (d00 * d21 - d01 * d20) / denom.clamp_min(1e-12)
    bary = torch.stack((1 - b1 - b2, b1, b2), -1)
    normal, length = _normal(tri)
    projected = (tri * bary[..., None]).sum(-2)
    signed = ((source_base - projected) * normal).sum(-1)
    # Official compact-face recovery threshold in meters, BEFORE normalization.
    known &= ((denom > 1e-12) & (length > 1e-6) & torch.isfinite(bary).all(-1)
              & torch.isfinite(signed) & (signed.abs() <= .1))
    bary = torch.where(known[..., None], bary, 0)
    signed = torch.where(known, signed, 0)
    trajectory = torch.zeros(B, Q, T, H, W, 3, device=depth.device)
    motion_valid = torch.zeros(B, Q, T, H, W, dtype=torch.bool, device=depth.device)
    trajectory_valid = torch.zeros_like(motion_valid)
    # Each iteration handles every source pixel; no query chunks or sparse pool.
    for t in range(T):
        triangle = vertices[bi[..., None], t, vi]
        n, area = _normal(triangle)
        xyz = (triangle * bary[..., None]).sum(-2) + signed[..., None] * n
        mesh_valid = known & (area > 1e-6) & torch.isfinite(xyz).all(-1)
        mv = mesh_valid | environment
        xyz = torch.where(environment[..., None], source_base, xyz)
        same = identity[:, :, t, None, None] & source_valid
        tv = mv | same
        xyz = torch.where(same[..., None], source_base, xyz)
        trajectory[:, :, t] = torch.where(tv[..., None], xyz, 0)
        trajectory_valid[:, :, t], motion_valid[:, :, t] = tv, mv
    displacement = torch.where(motion_valid[..., None], trajectory - source_base[:, :, None], 0)
    trajectory /= scale[:, None, None, None, None, None]
    displacement /= scale[:, None, None, None, None, None]
    camera = camera.clone(); camera[..., :3, 3] /= scale[:, None, None]
    depth = depth / scale[:, None, None, None]
    origin = camera[..., :3, 3][:, :, None, None].expand(*depth.shape, 3)
    direction = unproject_depth(torch.ones_like(depth), K, camera) - origin
    target.update(depth=depth, depth_valid=valid, c2w=camera, ray=torch.cat((direction, origin), -1),
                  ray_valid=valid, trajectory=trajectory, trajectory_valid=trajectory_valid,
                  displacement=displacement, displacement_valid=motion_valid,
                  trajectory_foreground=known[:, :, None].expand_as(motion_valid) & motion_valid,
                  source_pixel_valid=source_valid, static_mask=environment, scene_scale=scale)
    if recipe['dense_syn4d_include_visibility'].any():
        if not recipe['dense_syn4d_include_visibility'].all():
            raise ValueError('Syn4D batches must share include_visibility')
        visible, labeled = _depth_visibility(batch, target)
        target.update(visibility=visible, visibility_valid=labeled)
    return target


def _depth_visibility(batch, target):
    """Conservative depth-derived visibility, explicitly three-state.

    Known hidden/out-of-frustum points retain 3D GT. Missing target depth and
    points in front of the rendered surface are unknown, not negative labels.
    """
    xyz = target['trajectory']
    B, Q, T, H, W, _ = xyz.shape
    camera_ids = [batch['camera_id'][i, batch['observation_valid'][i]].unique(sorted=True) for i in range(B)]
    visible = torch.zeros(B, Q, T, max(map(len, camera_ids)), H, W, dtype=torch.bool, device=xyz.device)
    labeled = torch.zeros_like(visible)
    for b in range(B):
        for t in range(T):
            points = xyz[b, :, t].reshape(-1, 3)
            valid = target['trajectory_valid'][b, :, t].flatten()
            for c, camera_id in enumerate(camera_ids[b]):
                matches = ((batch['camera_id'][b] == camera_id) & batch['observation_valid'][b]
                           & (batch['time_idx'][b] == batch['query_time_idx'][b, t]) & target['camera_valid'][b]).nonzero().flatten()
                if len(matches) != 1:
                    if len(matches) > 1:
                        raise ValueError('duplicate Syn4D camera/time slot')
                    continue
                slot = int(matches[0])
                C, K = target['c2w'][b, slot], target['K'][b, slot]
                pc = (points - C[:3, 3]) @ C[:3, :3]
                z = pc[:, 2]
                uv = (pc @ K.T)[:, :2] / z.clamp_min(1e-8)[:, None]
                # The padded tensor is not the physical camera image. Padding
                # is outside the frustum; missing depth inside the image stays
                # unknown. Do not infer this boundary from positive depth.
                content_h, content_w = batch['targets']['dense_syn4d_content_hw'][b, slot]
                # Array centres are integers, so each boundary pixel extends
                # half a pixel beyond its centre (also for subpixel tracks).
                in_frame = (torch.isfinite(uv).all(-1) & (z > 0) & (uv[:, 0] >= -.5)
                            & (uv[:, 0] < content_w - .5) & (uv[:, 1] >= -.5) & (uv[:, 1] < content_h - .5))
                pixel = torch.nan_to_num(uv, nan=-1e6, posinf=1e6, neginf=-1e6).round().long()
                x, y = pixel[:, 0].clamp(0, W - 1), pixel[:, 1].clamp(0, H - 1)
                d = target['depth'][b, slot, y, x]
                has_depth = target['depth_valid'][b, slot, y, x]
                tolerance = .03 / target['scene_scale'][b] + .01 * d
                seen = in_frame & has_depth & ((z - d).abs() <= tolerance)
                known = ~in_frame | (has_depth & (z >= d - tolerance))
                visible[b, :, t, c] = (seen & valid).reshape(Q, H, W)
                labeled[b, :, t, c] = (known & valid).reshape(Q, H, W)
    return visible, labeled
