"""Audit real Syn4D clips against the released visualizer, without its UI.

Only definitions are imported from the user-specified official Python file;
its dataset initialization, global sampling and UI startup are not executed.
The official _get_views method performs mesh animation, face recovery and
camera geometry independently. Its crop is disabled for native-pixel checks.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
import types

import numpy as np
import torch

from multicamera4rc.datasets.adapters.syn4d import Syn4DAdapter
from multicamera4rc.datasets.adapters.syn4d_geometry import WORLD_REFLECTION
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import collate_adapter_samples
from multicamera4rc.datasets.syn4d_dense import attach_syn4d_dense_recipe


def official_module(path):
    import cv2
    import OpenEXR
    import Imath
    import pandas
    module = types.ModuleType('_syn4d_official_geometry_audit')
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(np=np, torch=torch, os=os, sys=sys, time=time, Path=Path,
                     pd=pandas, csv=csv, json=json, tarfile=tarfile, math=math,
                     cv2=cv2, exr=OpenEXR, Imath=Imath, dataclass=dataclass,
                     OrderedDict=OrderedDict, Any=object,
                     BaseStereoDynamicViewDataset=type('BaseStereoDynamicViewDataset', (), {}))
    def lift(depthmap, camera_intrinsics, camera_pose):
        y, x = np.indices(depthmap.shape)
        K = camera_intrinsics
        pc = np.stack(((x-K[0, 2])*depthmap/K[0, 0],
                       (y-K[1, 2])*depthmap/K[1, 1], depthmap), -1).astype(np.float32)
        return pc @ camera_pose[:3, :3].T + camera_pose[:3, 3], depthmap > 0
    namespace['depthmap_to_absolute_camera_coordinates'] = lift
    namespace['imread_cv2'] = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    parsed = ast.parse(Path(path).read_text())
    definitions = [v for v in parsed.body if isinstance(v, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return module


def official_views(module, adapter, clip, camera_position, query_position):
    m, scene = clip.metadata, Path(clip.metadata['syn4d_scene'])
    frames = m['time_indices']; T = len(frames)
    stems = m['syn4d_frame_stems'][camera_position*T:(camera_position+1)*T]
    camera = stems[0].rsplit('_', 1)[0]
    obj = module.Syn4D_Track.__new__(module.Syn4D_Track)
    csv_path = next(p for p in (scene/'be_seq.csv', scene/'be_seq_global_motion.csv') if p.is_file())
    groups = module.read_seq_csv(csv_path)
    obj.be_seq_info_dict_map = {scene.name: {g['sequence_name']: g for g in groups}}
    transform = next(p for p in (scene/'be_seq_transform_channels.json', scene/'be_seq_global_motion_transform_channels.json') if p.is_file())
    scales = {}
    for item in json.loads(transform.read_text()):
        xyz = np.array([float(item.get('map_camera_target_actor_location_'+k, 0.)) for k in 'xyz'], np.float32) / 100
        scales.setdefault(item['sequence_id'], []).append((item['sequence_body_index'],
                     float(item['transform_channels'].get('scale_x', 1.)), xyz))
    obj.transformation_scale_dict = {scene.name: {k: [(s, xyz) for _, s, xyz in sorted(v, key=lambda x:x[0])] for k, v in scales.items()}}
    obj.annotation_paths = [str(scene/'ground_truth/meta_exr_csv'/f'{camera}_camera.csv')]
    obj.rgb_paths = [[[str(scene/'png'/camera/f'{s}.png') for s in stems]]]
    obj.depth_paths = [[[str(scene/'exr_layers/depth'/camera/f'{s}_depth.exr') for s in stems]]]
    obj.track_npz_paths = [[[str(scene/'npz'/camera/s) for s in stems]]]
    obj.seg_paths = [[{'_env': [str(scene/'exr_layers/masks'/camera/f'{s}_env.png') for s in stems]}]]
    obj.full_idxs = [np.array(frames)]; obj.sample_stride = [1]
    obj.root_obj_npz_path = module.MultiRootPath([adapter.metadata_root])/'new_weight_bone'
    obj.root_smpl_npz_path = module.MultiRootPath([adapter.metadata_root])/'bedlam2_smpl_npz'
    obj.cloth_tar_root = adapter.metadata_root/'b2_assetdata_download/clothing/npz'
    cloth_index = obj.cloth_tar_root/'archive_map.json'
    obj.cloth_npz_to_tar = {entry: name for name, entries in json.loads(cloth_index.read_text()).items()
                            for entry in entries} if cloth_index.is_file() else {}
    obj._tracking_safetensor_cache = {}; obj._env_mask_safetensor_cache = {}
    obj._mp4_capture_cache = OrderedDict(); obj._mp4_capture_cache_size = 4
    obj.tracking_format = 'safetensor'; obj.debug_faceid_timing = False
    obj.track_query_idx_override = query_position; obj.allow_repeat = False
    obj.min_interval = obj.max_interval = 1
    obj.is_filter_dynamic = False; obj.caption = ''; obj.is_metric = True
    obj.get_seq_from_start_id = lambda *args, **kwargs: (list(range(T)), True, None)
    obj._crop_resize_if_necessary = lambda image, depth, K, resolution, **kw: (image, depth, K, kw['track'], kw['track_valid_mask'])
    try:
        return obj._get_views(0, ((clip.images[0].shape[1], clip.images[0].shape[0]), T), np.random.default_rng(0))
    finally:
        for cap in obj._mp4_capture_cache.values():
            cap.release()


def compare_camera_pose(expected, actual):
    """Bound cm-to-m float32 rounding separately from rotation/geometry error."""
    expected, actual = np.asarray(expected, np.float32), np.asarray(actual, np.float32)
    np.testing.assert_allclose(expected[:3, :3], actual[:3, :3], atol=3e-6, rtol=1e-7)
    np.testing.assert_array_equal(expected[3], actual[3])
    # Official code rounds cm to float32 before division; the adapter rounds
    # once after float64 division. At 556 m one float32 ULP is 0.061 mm.
    scale = np.maximum(np.abs(expected[:3, 3]), np.abs(actual[:3, 3]))
    tolerance = np.maximum(3e-6, 2*np.spacing(scale))
    error = np.abs(expected[:3, 3].astype(np.float64)-actual[:3, 3])
    if not np.isfinite(error).all() or (error > tolerance).any():
        raise AssertionError(f'camera translation differs beyond float32 rounding: {error}, {tolerance}')
    return float(error.max())


def audit_case(adapter, sequence, C, T, start, oracle, stride=1):
    started = time.monotonic()
    positions = [start+t*stride for t in range(T)]
    clip = adapter.load_clip(sequence, positions*C, layout_num_cameras=C, layout_num_times=T, sampling_seed=19)
    sample = prepare_clip(clip, num_queries=0, image_mode='native')
    attach_syn4d_dense_recipe(clip, sample, image_mode='native')
    batch = collate_adapter_samples([sample])
    sources = [0] if C == 1 else [0, (C-1)*T + T-1]
    source = torch.tensor([sources])
    raw = materialize_dense_targets(batch, source, normalize_scene=False)
    scaled = materialize_dense_targets(batch, source)
    scale = scaled['scene_scale'][0]
    for name in ('depth', 'trajectory', 'displacement'):
        assert torch.isfinite(raw[name]).all(), name
        torch.testing.assert_close(scaled[name]*scale, raw[name], atol=1e-5, rtol=1e-5)
    checks = []
    if oracle is not None:
        reference = np.linalg.inv(clip.extrinsics[0].astype(np.float64))
        for q, source_slot in enumerate(sources):
            c, query = divmod(source_slot, T)
            views = official_views(oracle, adapter, clip, c, query)
            errors, counts, extra, pose_error = [], [], 0, 0.
            for t, view in enumerate(views):
                h, w = view['depthmap'].shape
                np.testing.assert_array_equal(view['img'], clip.images[c*T+t])
                np.testing.assert_allclose(view['depthmap'], clip.depths[c*T+t], atol=1e-6)
                np.testing.assert_allclose(view['camera_intrinsics'], clip.intrinsics[c*T+t], atol=1e-5)
                pose = view['camera_pose']
                expected_pose = pose.copy(); expected_pose[:3] = WORLD_REFLECTION @ pose[:3]
                pose_error = max(pose_error, compare_camera_pose(expected_pose, clip.extrinsics[c*T+t]))
                world = (view['track'] @ pose[:3, :3].T + pose[:3, 3]) @ WORLD_REFLECTION.T
                expected = world @ reference[:3, :3].T + reference[:3, 3]
                known = raw['displacement_valid'][0, q, t, :h, :w].numpy()
                valid = view['track_valid_mask']
                extra += int((known & ~valid).sum())
                common = known & valid
                difference = raw['trajectory'][0, q, t, :h, :w].numpy()[common]-expected[common]
                errors.append(np.linalg.norm(difference, axis=-1)); counts.append(int(common.sum()))
            errors = np.concatenate(errors)
            result = dict(source_slot=source_slot, compared_pixel_times=sum(counts),
                          mean_error_m=float(errors.mean()), p99_error_m=float(np.quantile(errors,.99)),
                          max_error_m=float(errors.max()), additional_valid_pixel_times=extra,
                          max_camera_translation_roundoff_m=pose_error)
            # A small boundary/roundoff difference is recorded, never hidden.
            assert errors.mean() < 2e-4 and np.quantile(errors,.99) < 2e-3, result
            assert extra <= max(8, .0001 * sum(counts)), result
            checks.append(result)
    human_pixels, offset = {}, 0
    face_map = batch['targets']['dense_syn4d_face_map'][:, sources]
    for name, (_, faces) in sorted(clip.metadata['syn4d_meshes'].items()):
        if name in ('_00_body', '_00_clothing'):
            region = (face_map >= offset) & (face_map < offset+len(faces))
            human_pixels[name] = int((region[:, :, None] & raw['displacement_valid']).sum())
        offset += len(faces)
    return dict(sequence=sequence, cameras=C, times=T, raw_frames=clip.metadata['time_indices'],
                native_hw=list(clip.images[0].shape[:2]), padded_hw=list(sample['images'].shape[-2:]),
                physical_camera_ids=sample['camera_id'].unique().tolist(),
                depth_pixels=int(raw['depth_valid'].sum()), trajectory_pixel_times=int(raw['trajectory_valid'].sum()),
                dynamic_pixel_times=int(raw['trajectory_foreground'].sum()),
                max_displacement_m=float(raw['displacement'].norm(dim=-1).max()),
                human_motion_pixel_times=human_pixels,
                missing_meshes=clip.metadata['syn4d_missing_meshes'], official_checks=checks,
                seconds=round(time.monotonic()-started, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--official-loader')
    parser.add_argument('--sequences-per-subset', type=int, default=2)
    parser.add_argument('--subsets', nargs='+',
                        choices=['syn4d_v1_stride_1', 'syn4d_v1_stride_5', 'syn4d_sim'])
    parser.add_argument('--scenes', nargs='+', help='Audit selected extracted scenes')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    adapter = Syn4DAdapter(args.root, split='all', subsets=args.subsets, scenes=args.scenes)
    official_path = Path(args.official_loader) if args.official_loader else Path(args.root)/'code/visualizer/syn4d_track.py'
    oracle = official_module(official_path)
    cases = []
    for subset in sorted({s.split('/')[0] for s in adapter.sequences}):
        sequences = [s for s in adapter.sequences if s.startswith(subset+'/')][:args.sequences_per_subset]
        for si, sequence in enumerate(sequences):
            available = adapter.get_sequence_info(sequence)['num_cameras']
            layouts = [(1, 3), (min(available, 4), 3)]
            if available >= 8:
                layouts.append((8, 2))
            for C, T in layouts:
                result = audit_case(adapter, sequence, C, T, si*2, oracle)
                cases.append(result); print(json.dumps(result), flush=True)
            if si == 0 and adapter.get_num_frames(sequence) >= 16:
                start = min(30, adapter.get_num_frames(sequence)-16)
                result = audit_case(adapter, sequence, 1, 16, start, oracle)
                cases.append(result); print(json.dumps(result), flush=True)
    if not cases:
        raise ValueError('no extracted, complete Syn4D sequences found')
    report = dict(passed=True, official_loader=str(official_path),
                  official_loader_sha256=hashlib.sha256(official_path.read_bytes()).hexdigest(),
                  indexed_sequences=len(adapter), rejected=adapter.rejected, cases=cases)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
