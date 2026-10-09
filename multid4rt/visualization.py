"""Display/export actual model predictions while preserving source identity."""

import base64
from importlib import resources
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .tasks import sample_tracks


def _numpy(value):
    return value.detach().float().cpu().numpy()


def _pick(indices, limit):
    if not limit or len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices)-1, limit, dtype=np.int64)]


def _allocate_points(capacities, limit):
    """Split an exact point budget proportionally without exceeding capacity."""
    capacities = np.asarray(capacities, dtype=np.int64)
    total = int(capacities.sum())
    if not total or not limit or total <= limit:
        return capacities.copy()
    quotas = capacities.astype(np.float64)*(limit/total)
    allocated = np.floor(quotas).astype(np.int64)
    remainder = limit-int(allocated.sum())
    order = np.argsort(-(quotas-allocated), kind='stable')
    allocated[order[:remainder]] += 1
    return allocated


def _sample_pixels(valid, observations, limit):
    """Spread display samples over both image axes, preserving source slots.

    A flat stride aliases a row-major image into diagonal stripes. Use
    uniform sampling without replacement and a fixed seed instead; neither
    prediction values nor full-resolution inference are changed.
    """
    capacities = [int(valid[source].sum()) for _, source in observations]
    budgets = _allocate_points(capacities, limit)
    selected = []
    for (slot, source), capacity, budget in zip(observations, capacities, budgets):
        if not budget:
            continue
        mask = valid[source]
        if budget == capacity:
            y, x = np.where(mask)
        else:
            rng = np.random.default_rng(9157+int(source))
            chosen = rng.choice(np.flatnonzero(mask), int(budget), replace=False)
            chosen.sort()
            y, x = np.divmod(chosen, mask.shape[1])
        selected.append(np.column_stack((np.full(len(x), slot), np.full(len(x), source), y, x)))
    return np.concatenate(selected) if selected else np.empty((0, 4), np.int64)


def _image_url(pixels, *, depth=False):
    image = Image.fromarray(pixels.astype(np.uint8))
    image.thumbnail((360, 240), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG' if depth else 'JPEG', **({} if depth else {'quality': 85}))
    return 'data:image/'+('png' if depth else 'jpeg')+';base64,'+base64.b64encode(buffer.getvalue()).decode()


def write_ply(path, points, colors):
    keep = np.isfinite(points).all(-1)
    points, colors = points[keep], colors[keep]
    data = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for index, key in enumerate(('x', 'y', 'z')):
        data[key] = points[:, index]
    for index, key in enumerate(('red', 'green', 'blue')):
        data[key] = colors[:, index]
    header = ('ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(data)}\n'+
              'property float x\nproperty float y\nproperty float z\n'+
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as file:
        file.write(header.encode('ascii'))
        file.write(data.tobytes())


def _queries(visual, prediction, surfaces, queries, num_tracks):
    if queries is None:
        selected = _pick(np.arange(len(surfaces['source'])), num_tracks)
        return np.column_stack((surfaces['source'][selected], surfaces['pixel'][selected])).astype(np.float32)
    query = np.array(queries, dtype=np.float32, copy=True)
    if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
        raise ValueError('queries JSON must be an array of [source_observation,x,y]')
    for row in query:
        source = int(row[0])
        if row[0] != source or not 0 <= source < len(visual.observations):
            raise ValueError('track source must be an integer observation index')
        info = visual.observations[source]
        h, w = info['raw_hw']
        if not (0 <= row[1] <= w-1 and 0 <= row[2] <= h-1):
            raise ValueError('track query is outside its original source image')
        sx, sy = info['scale_xy']
        row[1:] = [(row[1]+.5)*sx-.5, (row[2]+.5)*sy-.5]
        ch, cw = info['content_hw']
        row[1] = np.clip(row[1], 0, cw-1)
        row[2] = np.clip(row[2], 0, ch-1)
    return query


def build_visualization(prediction, visual, *, max_points=20000, num_tracks=24, queries=None, max_depth=None):
    if max_points < 0 or num_tracks < 1 or max_depth is not None and max_depth <= 0:
        raise ValueError('point limits, track count and depth limit must be valid')
    images = visual.images[0].permute(0, 2, 3, 1).numpy()
    rgb = (images*255).round().astype(np.uint8)
    base, depth = _numpy(prediction['base_points'][0]), _numpy(prediction['depth'][0])
    confidence = _numpy(prediction['depth_conf'][0])
    masks = visual.pixel_valid.numpy()
    if base.shape[:3] != masks.shape:
        raise ValueError('prediction resolution does not match prepared inputs')
    for key, value in (('camera_id', visual.camera_id), ('time_idx', visual.time_idx)):
        if not torch.equal(prediction[key].cpu(), value):
            raise ValueError('prediction camera/time layout does not match inputs')
    valid = masks & np.isfinite(base).all(-1) & np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth <= max_depth
    sources = prediction['source_obs_idx'][0].cpu().numpy()
    source_valid = prediction['source_valid'][0].cpu().numpy()
    active_sources = [(slot, source) for slot, source in enumerate(sources) if source_valid[slot]]
    selected = _sample_pixels(valid, active_sources, max_points)
    if not len(selected):
        raise ValueError('no finite source points remain after display filtering')
    slots, source, y, x = selected.T
    points = _numpy(prediction['trajectory'][0, slots, :, y, x]).transpose(1, 0, 2)
    movement = np.linalg.norm(_numpy(prediction['displacement'][0, slots, :, y, x]), axis=-1).T
    finite = np.isfinite(points).all(-1)
    surface = dict(points=np.where(finite[..., None], points, 0), valid=finite,
                   rgb=rgb[source, y, x], source=source, pixel=np.column_stack((x, y)),
                   camera_id=visual.camera_id[0].numpy()[source], motion=np.where(finite, movement, 0))
    if 'visibility' in prediction:
        surface['visibility'] = _numpy(prediction['visibility'][0, slots, :, :, y, x]).max(-1).T
    query = _queries(visual, prediction, surface, queries, num_tracks)
    sampled = sample_tracks(prediction, torch.from_numpy(query)[None], strict=True)
    track_xyz = _numpy(sampled['trajectory'][0]).transpose(1, 0, 2)
    track_valid = np.isfinite(track_xyz).all(-1)
    track_xyz = np.where(track_valid[..., None], track_xyz, 0)
    times = prediction['query_time_idx'][0].cpu().tolist()
    cameras = sorted(set(visual.camera_id[0].tolist()))
    K, c2w = _numpy(prediction['K'][0]), _numpy(prediction['c2w'][0])
    depth_pool = depth[valid]
    low, high = np.percentile(np.log(np.maximum(depth_pool, 1e-8)), [2, 98])
    scale = max(float(np.median(depth_pool))*.12, 1e-6)
    palette = np.array([[68,1,84], [59,82,139], [33,145,140], [94,201,98], [253,231,37]])
    geometry, observations, frusta = [], [], []
    for frame in times:
        ids = np.flatnonzero(visual.time_idx[0].numpy() == frame)
        display = _sample_pixels(valid, list(enumerate(ids)), max_points)
        _, si, yy, xx = display.T
        geometry.append(dict(points=base[si, yy, xx], rgb=rgb[si, yy, xx],
                             camera_id=visual.camera_id[0].numpy()[si], confidence=confidence[si, yy, xx]))
        views, outlines = [], []
        for index in ids:
            ch, cw = visual.observations[index]['content_hw']
            normalized = np.clip((np.log(np.maximum(depth[index, :ch, :cw], 1e-8))-low)/max(high-low, 1e-8), 0, 1)
            normalized = np.nan_to_num(normalized)
            heat = np.stack([np.interp(normalized, np.linspace(0, 1, len(palette)), palette[:, channel])
                             for channel in range(3)], -1)
            views.append(dict(camera_id=int(visual.camera_id[0, index]), observation=int(index),
                rgb=_image_url(rgb[index, :ch, :cw]), depth=_image_url(heat, depth=True),
                content_hw=[ch, cw], K=K[index], c2w=c2w[index]))
            corners = np.array([[0,0,1], [cw-1,0,1], [cw-1,ch-1,1], [0,ch-1,1]], np.float32)
            if np.isfinite(K[index]).all() and np.isfinite(c2w[index]).all():
                try:
                    rays = corners @ np.linalg.inv(K[index]).T
                except np.linalg.LinAlgError:
                    continue
                rim = (rays*scale) @ c2w[index,:3,:3].T+c2w[index,:3,3]
                center = c2w[index,:3,3]
                line = [rim[0],rim[1],rim[2],rim[3],rim[0],None]
                for point in rim:
                    line += [center,point,None]
                outlines.append(dict(camera_id=int(visual.camera_id[0,index]), points=line, center=center))
        observations.append(views)
        frusta.append(outlines)
    payload = dict(clip_id=visual.clip_id, mode=visual.mode, times=times, cameras=cameras,
                   geometry=geometry, surface=surface,
                   tracks=dict(points=track_xyz, valid=track_valid, queries=query),
                   views=observations, frusta=frusta,
                   settings=dict(max_points=max_points, max_depth=max_depth, source_observations=sources.tolist(),
                                 displayed_points=len(selected), sampling='uniform image pixels without replacement; seed 9157 + observation',
                                 units='checkpoint world units'))
    return payload


def _json_arrays(value):
    if isinstance(value, np.ndarray):
        # Non-finite auxiliary confidence/camera fields must not break the viewer.
        return _json_arrays(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        return round(value, 6) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_arrays(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_arrays(x) for x in value]
    return value


def export_visualization(payload, directory):
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise ImportError("visualization requires: pip install 'multid4rt[viz]'") from exc
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ply = directory/'ply'
    ply.mkdir(exist_ok=True)
    for index, frame in enumerate(payload['times']):
        geometry, surface = payload['geometry'][index], payload['surface']
        write_ply(ply/f'frame_{index:04d}_t{frame}_geometry.ply', geometry['points'], geometry['rgb'])
        mask = surface['valid'][index]
        write_ply(ply/f'frame_{index:04d}_t{frame}_tracked.ply', surface['points'][index, mask], surface['rgb'][mask])
    tracks = payload['tracks']
    np.savez_compressed(directory/'tracks.npz', trajectory=tracks['points'].transpose(1,0,2),
                        valid=tracks['valid'].T, queries=tracks['queries'], time_idx=payload['times'])
    surface = payload['surface']
    flow_valid = surface['valid'][1:] & surface['valid'][:-1]
    flow = np.where(flow_valid[..., None], np.diff(surface['points'], axis=0), 0)
    np.savez_compressed(directory/'scene_flow.npz', displacement=flow,
        valid=flow_valid, source_obs_idx=surface['source'],
        source_uv=surface['pixel'], start_time=payload['times'][:-1], end_time=payload['times'][1:])
    body = json.dumps(_json_arrays(payload), separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    template = resources.files('multid4rt').joinpath('assets/viewer.html').read_text()
    html = template.replace('__PLOTLY_JS__', get_plotlyjs()).replace('__VISUAL_DATA__', body)
    path = directory/'viewer.html'
    path.write_text(html, encoding='utf-8')
    (directory/'visualization.json').write_text(json.dumps(dict(clip_id=payload['clip_id'], mode=payload['mode'],
        times=payload['times'], cameras=payload['cameras'], **payload['settings']), indent=2)+'\n')
    return path
