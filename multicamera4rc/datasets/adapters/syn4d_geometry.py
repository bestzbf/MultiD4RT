"""Syn4D release geometry conventions (see docs/syn4d.md for provenance).

The official visualizer works in Unreal's left-handed world. Reflecting world
Y, for BOTH cameras and vertices, gives proper OpenCV C2W rotations without
changing any image projection. Depth and translations are released in cm.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tarfile

import numpy as np

WORLD_REFLECTION = np.diag([1., -1., 1.])
CAMERA_AXES = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
BODY_AXES = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
SMPL_AXES = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def rotation(yaw, pitch, roll):
    y, p, r = np.deg2rad([yaw, pitch, roll])
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]]))


def calibration(row, height, width):
    focal, sw, sh = (float(row[k]) for k in ('focal_length', 'sensor_width', 'sensor_height'))
    if not np.isfinite([focal, sw, sh]).all() or min(focal, sw, sh) <= 0:
        raise ValueError('invalid Syn4D focal length / sensor dimensions')
    # Exactly the official array-index convention: do not subtract half a pixel.
    K = np.array([[focal / sw * width, 0, width / 2],
                  [0, focal / sh * height, height / 2], [0, 0, 1]], np.float32)
    C = np.eye(4)
    C[:3, :3] = WORLD_REFLECTION @ rotation(*(float(row[k]) for k in ('yaw', 'pitch', 'roll'))) @ CAMERA_AXES
    C[:3, 3] = WORLD_REFLECTION @ np.array([float(row[k]) for k in ('x', 'y', 'z')]) / 100
    if not np.isfinite(C).all():
        raise ValueError('nonfinite Syn4D camera pose')
    return K, C.astype(np.float32)


def read_depth(path, max_depth=300.):
    import OpenEXR
    with OpenEXR.File(str(path), separate_channels=True) as f:
        channels = f.channels()
        if 'Depth' not in channels:
            raise ValueError(f'Syn4D depth EXR has no Depth channel: {path}')
        depth = np.asarray(channels['Depth'].pixels, np.float32).copy() / 100.
    if depth.ndim != 2:
        raise ValueError(f'Syn4D depth must be HxW: {path}')
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    return np.where(valid, depth, 0).astype(np.float32)


def read_scene(scene):
    scene = Path(scene)
    csv_path = next((scene / n for n in ('be_seq.csv', 'be_seq_global_motion.csv') if (scene / n).is_file()), None)
    transform_path = next((scene / n for n in ('be_seq_transform_channels.json',
                                             'be_seq_global_motion_transform_channels.json') if (scene / n).is_file()), None)
    if csv_path is None or transform_path is None:
        raise FileNotFoundError(f'Syn4D scene lacks actor/transform metadata: {scene}')
    groups = {}
    current = None
    with csv_path.open(newline='') as f:
        for row in csv.DictReader(f):
            config = dict(v.split('=', 1) for v in row.get('Comment', '').split(';') if '=' in v)
            if row['Type'] == 'Group':
                current = dict(config=config, frames=int(config['frames']), actors=[])
                groups[config['sequence_name']] = current
            elif row['Type'] in ('Body', 'Object') and current is not None:
                # CSV row Index differs between cameras and is not actor identity.
                row = {k: v for k, v in row.items() if k != 'Index'}
                row['config'] = config
                current['actors'].append(row)
    transforms = {}
    for item in json.loads(transform_path.read_text()):
        name = item['sequence_id']
        record = {k: v for k, v in item.items() if k != 'sequence_id'}
        transforms.setdefault(name, []).append(record)
    for values in transforms.values():
        values.sort(key=lambda v: int(v['sequence_body_index']))
    return groups, transforms


def validate_mesh(vertices, faces, path):
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError(f'invalid mesh vertices [frames,V,3]: {path}')
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError(f'invalid mesh faces [F,3]: {path}')
    if not len(faces) or (faces < 0).any() or (faces >= vertices.shape[1]).any():
        raise ValueError(f'mesh face index out of range: {path}')
    return vertices, faces.astype(np.int64)


def object_vertices(actor, transform, vertices, frames, sequence_frames):
    """Reconstruct only requested native frames, matching the official reader."""
    frames = np.asarray(frames, np.int64)
    # The release starts animation at mesh frame 1 and repeats the asset cycle.
    local = vertices[(frames + 1) % len(vertices)].copy()
    local[..., 1] *= -1
    channels = transform['transform_channels']
    scales = np.array([float(channels.get('scale_' + k, 1.)) for k in 'xyz'])
    if not np.isfinite(scales).all() or (scales <= 0).any() or not np.allclose(scales, scales[0]):
        raise ValueError('unsupported nonuniform/invalid Syn4D actor scale')
    scale = scales[0]
    origin = np.array([float(transform.get('map_camera_target_actor_location_' + k, 0.)) for k in 'xyz']) / 100
    defaults = {k: float(actor[k.capitalize()]) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll')}
    defaults['scale'] = scale
    raw_motion = actor.get('PerFrameMotion', '')
    motion = json.loads(raw_motion) if raw_motion else []
    output = np.empty_like(local)
    if motion:
        by_frame = {int(v['frame']): v for v in motion}
        keys = sorted(by_frame)
        if len(keys) != len(motion):
            raise ValueError('duplicate frames in Syn4D PerFrameMotion')
        values = {key: np.interp(frames, keys, [float(by_frame[t].get(key, default)) for t in keys])
                  for key, default in defaults.items()}
        for i in range(len(frames)):
            R = rotation(*(values[k][i] for k in ('yaw', 'pitch', 'roll')))
            t = np.array([values[k][i] for k in 'xyz']) / 100 + origin
            output[i] = (local[i] * values['scale'][i]) @ R.T + t
    else:
        R = rotation(*(defaults[k] for k in ('yaw', 'pitch', 'roll')))
        shift = np.zeros((len(frames), 1, 3))
        shift[:, 0, 1] = frames / max(1, sequence_frames - 1) * float(actor.get('Shift') or 0.) / 100
        t = np.array([defaults[k] for k in 'xyz']) / 100 + origin
        output[:] = (local * scale + shift) @ R.T + t
    return (output @ WORLD_REFLECTION.T).astype(np.float32)


def body_asset_path(actor, metadata_root):
    body = actor['Body']
    subject, animation = (body.split('_', 1)[0], body) if body.startswith('moyo') else body.rsplit('_', 1)
    return subject, animation, Path(metadata_root) / 'bedlam2_smpl_npz' / subject / f'{subject}_{animation}.npz'


def clothing_mesh(actor, metadata_root):
    subject, animation, _ = body_asset_path(actor, metadata_root)
    root = Path(metadata_root) / 'b2_assetdata_download/clothing/npz'
    relative = f'{subject}/{animation}/{animation}.npz'
    direct = root / relative
    if direct.is_file():
        with np.load(direct, allow_pickle=False) as data:
            return validate_mesh(data['vertices_seq'], data['faces'], direct)
    index = root / 'archive_map.json'
    if not index.is_file():
        return None
    for archive, members in json.loads(index.read_text()).items():
        if relative not in members:
            continue
        path = root / f'{archive}.tar'
        if not path.is_file():
            return None
        with tarfile.open(path) as tar:
            matches = [m for m in tar.getmembers() if m.isfile() and (m.name == relative or m.name.endswith('/' + relative))]
            if len(matches) != 1:
                raise ValueError(f'clothing archive must contain exactly one {relative}')
            with tar.extractfile(matches[0]) as f, np.load(io.BytesIO(f.read()), allow_pickle=False) as data:
                return validate_mesh(data['vertices_seq'], data['faces'], path)
    return None


def body_vertices(actor, transform, vertices, frames, *, clothing=False):
    start = int(actor['config'].get('start_frame', 0))
    indices = np.asarray(frames, np.int64) + start
    if indices.min() < 0 or indices.max() >= len(vertices):
        raise ValueError('BEDLAM mesh animation does not cover selected Syn4D frames')
    R = rotation(*(float(actor[k]) for k in ('Yaw', 'Pitch', 'Roll'))) @ BODY_AXES
    local = vertices[indices]
    if not clothing:
        local = local @ SMPL_AXES.T
    origin = np.array([float(transform.get('map_camera_target_actor_location_' + k, 0.)) for k in 'xyz']) / 100
    t = np.array([float(actor[k.upper()]) for k in 'xyz']) / 100 + origin
    # Body meshes already have the scale used by the official body conversion.
    return ((local @ R.T + t) @ WORLD_REFLECTION.T).astype(np.float32)
