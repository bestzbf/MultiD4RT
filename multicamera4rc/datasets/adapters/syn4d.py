"""Native Syn4D safetensors/EXR/MP4 adapter; no particle subsampling.

Raw frames and compact mesh correspondences are loaded on CPU. Dense tracks
are reconstructed for selected source images later, on the training device.
"""
from __future__ import annotations

from collections import OrderedDict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .base import BaseAdapter, UnifiedClip
from .syn4d_geometry import (body_asset_path, body_vertices, calibration, clothing_mesh,
                             object_vertices, read_depth, read_scene, validate_mesh)


def sequence_split(scene, sequence):
    """Stable 80/10/10 sequence split shared by cameras and stride releases."""
    bucket = int(hashlib.sha256(f'{scene}/{sequence}'.encode()).hexdigest()[:8], 16) % 10
    return 'val' if bucket == 8 else 'test' if bucket == 9 else 'train'


def _json(path):
    return json.loads(Path(path).read_text())


class Syn4DAdapter(BaseAdapter):
    dataset_name = 'syn4d'

    def __init__(self, root, split='train', *, subsets=None, scenes=None, metadata_root=None,
                 frame_policy='reference', missing_mesh='mask', max_depth=300.,
                 mesh_cache_mb=256, load_tracking=True, require_human_meshes=False):
        if split not in ('train', 'val', 'test', 'all'):
            raise ValueError('Syn4D split must be train, val, test or all')
        if frame_policy not in ('reference', 'all') or missing_mesh not in ('mask', 'error'):
            raise ValueError('frame_policy must be reference/all and missing_mesh mask/error')
        if not np.isfinite(max_depth) or max_depth <= 0 or mesh_cache_mb < 0:
            raise ValueError('invalid Syn4D depth/cache limit')
        self.root = Path(root).resolve()
        self.split, self.frame_policy, self.missing_mesh = split, frame_policy, missing_mesh
        self.require_human_meshes = bool(require_human_meshes)
        self.max_depth, self.load_tracking = float(max_depth), bool(load_tracking)
        self.mesh_cache_bytes = int(mesh_cache_mb * 1024 * 1024)
        self._meshes, self._scenes, self._cameras = OrderedDict(), OrderedDict(), OrderedDict()
        if (self.root / 'data').is_dir():
            base = self.root / 'data'
        elif self.root.name.startswith('syn4d_'):
            base = self.root.parent
        else:
            base = self.root
        self.metadata_root = Path(metadata_root).resolve() if metadata_root else base / 'metadata'
        if subsets is None:
            subsets = [self.root.name] if self.root.name.startswith('syn4d_') else ['syn4d_v1_stride_5', 'syn4d_sim']
        if isinstance(subsets, str) or not subsets or isinstance(scenes, str):
            raise ValueError('subsets/scenes must be lists of directory names')
        self.index, self.rejected = {}, []
        for subset in subsets:
            if Path(subset).name != subset or subset not in ('syn4d_v1_stride_1', 'syn4d_v1_stride_5', 'syn4d_sim'):
                raise ValueError(f'unsupported Syn4D subset {subset!r}')
            directory = base / subset
            if not directory.is_dir():
                self.rejected.append(dict(path=str(directory), reason='subset not downloaded'))
                continue
            for scene in sorted(directory.iterdir()):
                if not scene.is_dir() or (scenes is not None and scene.name not in scenes):
                    continue
                camera_root = scene / 'ground_truth/meta_exr_csv'
                if not camera_root.is_dir():
                    continue  # .cache, MP4 previews and unpublished extraction directories.
                if (not any((scene / n).is_file() for n in ('be_seq.csv', 'be_seq_global_motion.csv'))
                        or not any((scene / n).is_file() for n in ('be_seq_transform_channels.json', 'be_seq_global_motion_transform_channels.json'))):
                    self.rejected.append(dict(path=str(scene), reason='incomplete actor/transform metadata'))
                    continue
                by_sequence = {}
                for path in sorted(camera_root.glob('*_camera.csv')):
                    match = re.fullmatch(r'(seq_\d+)_(\d+)_camera\.csv', path.name)
                    if match is None:
                        continue
                    sequence, cid = match.group(1), int(match.group(2))
                    if split != 'all' and sequence_split(scene.name, sequence) != split:
                        continue
                    camera = f'{sequence}_{cid}'
                    try:
                        info = self._index_camera(scene, directory, camera, path)
                    except (ValueError, FileNotFoundError, OSError) as error:
                        self.rejected.append(dict(path=str(path), reason=str(error)))
                        continue
                    by_sequence.setdefault(sequence, {})[cid] = info
                for sequence, cameras in by_sequence.items():
                    shared = set.intersection(*(set(v['frames']) for v in cameras.values()))
                    if self.load_tracking and frame_policy == 'reference':
                        shared &= set.intersection(*(set(v['references']) for v in cameras.values()))
                    if not shared:
                        self.rejected.append(dict(path=str(scene / sequence), reason='no synchronized frames'))
                        continue
                    key = f'{subset}/{scene.name}/{sequence}'
                    self.index[key] = dict(scene=scene, cameras=cameras, frames=sorted(shared), sequence=sequence)
        self.sequences = sorted(self.index)

    def _index_camera(self, scene, subset, camera, path):
        import cv2
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        labels = []
        for row in rows:
            name = Path(row['name']).name
            match = re.fullmatch(re.escape(camera) + r'_(\d+)\.png', name)
            if match is None:
                raise ValueError(f'invalid frame name {name}')
            labels.append(int(match.group(1)))
        if not labels or labels != sorted(set(labels)):
            raise ValueError('camera CSV frame numbers must be sorted and unique')
        png = scene / 'png' / camera
        video = next((p for p in (scene / 'mp4' / f'{camera}.mp4', subset / 'mp4' / scene.name / f'{camera}.mp4')
                      if p.is_file()), None)
        if png.is_dir():
            rgb = {int(p.stem.rsplit('_', 1)[-1]) for p in png.glob(f'{camera}_*.png')}
        elif video:
            cap = cv2.VideoCapture(str(video))
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if not cap.isOpened() or count != len(rows):
                    raise ValueError(f'MP4/CSV frame count mismatch: video={count}, CSV={len(rows)}')
                if labels != list(range(count)):
                    raise ValueError('MP4 frame numbers must match zero-based decode indices')
            finally:
                cap.release()
            rgb = set(labels)
        else:
            raise FileNotFoundError('RGB video/PNG frames not downloaded')
        depths = scene / 'exr_layers/depth' / camera
        depth_labels = {int(p.stem.rsplit('_', 2)[-2]) for p in depths.glob(f'{camera}_*_depth.exr')}
        # Admit only complete camera streams; a running download must not alter
        # the meaning or numbering of an indexed training sequence.
        if not set(labels) <= rgb & depth_labels:
            raise FileNotFoundError('incomplete RGB/depth stream')
        track_index = scene / 'tracking_safetensors' / f'{camera}_index.json'
        references = []
        if self.load_tracking and track_index.is_file():
            records = _json(track_index)
            if records.get('format') != 'syn4d_compact_faceid_safetensors_v1':
                raise ValueError('unsupported Syn4D tracking shard schema')
            if not track_index.with_name(f'{camera}.safetensors').is_file():
                raise FileNotFoundError('tracking shard is not downloaded')
            references = sorted({int(k.split('/')[0].rsplit('_', 1)[-1]) for k in records['record_key_to_id']})
        elif self.load_tracking:
            raise FileNotFoundError('tracking index is not downloaded')
        return dict(csv=path, png=png, video=video, camera=camera, frames=labels, references=references)

    def __len__(self):
        return len(self.sequences)

    def list_sequences(self):
        return list(self.sequences)

    def get_sequence_name(self, index):
        return self.sequences[index]

    def get_num_frames(self, sequence_name):
        return len(self.index[sequence_name]['frames'])

    def get_sequence_info(self, sequence_name):
        record = self.index[sequence_name]
        return dict(num_frames=len(record['frames']), num_cameras=len(record['cameras']),
                    cameras=sorted(record['cameras']), frame_indices=list(record['frames']),
                    frame_policy=self.frame_policy, split=sequence_split(record['scene'].name, record['sequence']))

    def _scene(self, scene):
        if scene not in self._scenes:
            self._scenes[scene] = read_scene(scene)
            while len(self._scenes) > 2:
                self._scenes.popitem(last=False)
        self._scenes.move_to_end(scene)
        return self._scenes[scene]

    def _mesh(self, path):
        path = Path(path)
        if not path.is_file():
            return None
        if path not in self._meshes:
            with np.load(path, allow_pickle=False) as f:
                data = validate_mesh(f['vertices_seq'], f['faces'], path)
            size = sum(x.nbytes for x in data)
            while self._meshes and sum(sum(x.nbytes for x in v) for v in self._meshes.values()) + size > self.mesh_cache_bytes:
                self._meshes.popitem(last=False)
            if size <= self.mesh_cache_bytes:
                self._meshes[path] = data
            return data
        self._meshes.move_to_end(path)
        return self._meshes[path]

    def _geometry(self, record, camera_ids, frames):
        groups, transforms = self._scene(record['scene'])
        names = [record['cameras'][c]['camera'] for c in camera_ids]
        group, transform = groups[names[0]], transforms[names[0]]
        for name in names[1:]:
            if groups[name]['actors'] != group['actors'] or transforms[name] != transform or groups[name]['frames'] != group['frames']:
                raise ValueError(f'camera streams do not share synchronized actor geometry: {names[0]}, {name}')
        meshes, missing = {}, []
        objects = [a for a in group['actors'] if a['Type'] == 'Object']
        bodies = [a for a in group['actors'] if a['Type'] == 'Body']
        if len(bodies) > 1:
            raise ValueError('released Syn4D body mask convention is verified for one body per sequence')
        for index, actor in enumerate(objects):
            c = actor['config']
            path = self.metadata_root / 'new_weight_bone' / c['object_group'] / c['object_id'] / 'vertices_sequence.npz'
            label = f'obj_{index + bool(bodies):02d}'
            data = self._mesh(path)
            if data is None:
                missing.append(dict(mask=label, path=str(path)))
                continue
            if index >= len(transform):
                raise ValueError('missing Syn4D object transform channels')
            vertices, faces = data
            meshes[label] = (object_vertices(actor, transform[index], vertices, frames, group['frames']), faces)
        for actor in bodies:
            _, _, path = body_asset_path(actor, self.metadata_root)
            for label, data, clothing in (('_00_body', self._mesh(path), False),
                                           ('_00_clothing', clothing_mesh(actor, self.metadata_root), True)):
                if data is None:
                    missing.append(dict(mask=label, path=str(path) if not clothing else str(self.metadata_root / 'b2_assetdata_download/clothing/npz')))
                else:
                    if not transform:
                        raise ValueError('missing body world-origin transform')
                    meshes[label] = (body_vertices(actor, transform[0], data[0], frames, clothing=clothing), data[1])
        missing_human = [item for item in missing if item['mask'] in ('_00_body', '_00_clothing')]
        if missing_human and self.require_human_meshes:
            raise FileNotFoundError(f'required Syn4D human meshes missing: {missing_human}')
        if missing and self.missing_mesh == 'error':
            raise FileNotFoundError(f'Syn4D mesh resources missing: {missing}')
        return meshes, missing

    def _read_rgb(self, info, frames):
        import cv2
        if info['csv'] not in self._cameras:
            with info['csv'].open(newline='') as f:
                self._cameras[info['csv']] = list(csv.DictReader(f))
            while len(self._cameras) > 16:
                self._cameras.popitem(last=False)
        self._cameras.move_to_end(info['csv'])
        rows = self._cameras[info['csv']]
        positions = {frame: i for i, frame in enumerate(info['frames'])}
        selected_rows = [rows[positions[t]] for t in frames]
        images = {}
        if info['png'].is_dir():
            for frame, row in zip(frames, selected_rows):
                image = cv2.imread(str(info['png'] / Path(row['name']).name))
                if image is None:
                    raise ValueError(f'cannot decode RGB frame {row["name"]}')
                images[frame] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            cap = cv2.VideoCapture(str(info['video']))
            try:
                # One seek, then sequential decoding avoids per-frame GOP seeks.
                first, last = min(positions[t] for t in frames), max(positions[t] for t in frames)
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
                    raise ValueError(f'cannot seek Syn4D video {info["video"]}')
                wanted = {positions[t]: t for t in frames}
                for position in range(first, last + 1):
                    if not cap.grab():
                        raise ValueError(f'truncated Syn4D RGB video {info["video"]} at {position}')
                    if position in wanted:
                        ok, image = cap.retrieve()
                        if not ok:
                            raise ValueError(f'cannot decode Syn4D video {info["video"]}')
                        images[wanted[position]] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()
        return [images[t] for t in frames], selected_rows

    def load_clip(self, sequence_name, frame_indices, *, layout_num_cameras=1,
                  layout_num_times=None, sampling_seed=0, canonical_anchor_required=False):
        if canonical_anchor_required:
            raise NotImplementedError('Syn4D adapter provides the MultiCamera4RC clip contract only')
        C = int(layout_num_cameras)
        T = len(frame_indices) // C if layout_num_times is None else int(layout_num_times)
        if C < 1 or T < 1 or len(frame_indices) != C * T or frame_indices != frame_indices[:T] * C:
            raise ValueError('Syn4D input must be C camera-major copies of T unique frame positions')
        positions = frame_indices[:T]
        record = self.index[sequence_name]
        if len(set(positions)) != T or any(isinstance(t, bool) or int(t) != t or not 0 <= t < len(record['frames']) for t in positions):
            raise ValueError('invalid Syn4D frame positions')
        frames = [record['frames'][t] for t in positions]
        if frames != sorted(frames):
            raise ValueError('Syn4D times must be increasing')
        available = sorted(record['cameras'])
        if C > len(available):
            raise ValueError(f'Syn4D sequence has only {len(available)} complete cameras')
        cameras = sorted(np.random.default_rng(sampling_seed).choice(available, C, replace=False).tolist())
        images, depths, intrinsics, poses, stems = [], [], [], [], []
        for camera in cameras:
            info = record['cameras'][camera]
            rgb, rows = self._read_rgb(info, frames)
            for image, row in zip(rgb, rows):
                stem = Path(row['name']).stem
                depth = read_depth(record['scene'] / 'exr_layers/depth' / info['camera'] / f'{stem}_depth.exr', self.max_depth)
                if depth.shape != image.shape[:2]:
                    raise ValueError('Syn4D RGB/depth dimensions disagree')
                K, pose = calibration(row, *depth.shape)
                images.append(image); depths.append(depth); intrinsics.append(K); poses.append(pose); stems.append(stem)
        meshes, missing = self._geometry(record, cameras, frames) if self.load_tracking else ({}, [])
        metadata = dict(extrinsics_convention='c2w', pixel_coordinates='array_index_v1',
                        time_indices=frames, slot_time_id=list(range(T)) * C,
                        slot_physical_camera_id=[c for c in cameras for _ in frames],
                        valid_slot=[True] * (C * T), syn4d_scene=str(record['scene']),
                        syn4d_frame_stems=stems, syn4d_meshes=meshes,
                        syn4d_missing_meshes=missing, syn4d_load_tracking=self.load_tracking,
                        syn4d_frame_policy=self.frame_policy)
        return UnifiedClip(dataset_name='syn4d', sequence_name=sequence_name, frame_paths=None,
                           images=images, depths=depths, normals=None, trajs_2d=None,
                           trajs_3d_world=None, valids=None, visibs=None,
                           intrinsics=np.stack(intrinsics), extrinsics=np.stack(poses), metadata=metadata)

    def sanity_check(self, sequence_name):
        info = self.get_sequence_info(sequence_name)
        clip = self.load_clip(sequence_name, [0])
        return {**info, 'image_size': list(clip.images[0].shape[:2]),
                'valid_depth_pixels': int((clip.depths[0] > 0).sum()),
                'mesh_masks': sorted(clip.metadata['syn4d_meshes']),
                'missing_meshes': clip.metadata['syn4d_missing_meshes']}
