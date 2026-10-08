"""Independent raw-renderer oracle for a fixed Kubric camera/time case.

Checks every supervised source pixel at every selected time, and separately
checks the original sparse world tracks against the object's rigid poses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import attach_kubric_dense_recipe, materialize_dense_targets
from multicamera4rc.datasets.mixture import collate_adapter_samples, read_data_config, create_dataset
from multicamera4rc.evaluate import fixed_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-config', required=True)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--case-index', type=int, required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    config = read_data_config(args.data_config)
    case = json.loads(args.plan.read_text())['cases'][args.case_index]
    source = config['sources'][case['layout']]
    C, T = source['cameras'], source['num_times']
    source['split'] = 'val'
    config['sources'] = [source]
    adapter = create_dataset(config).sources[0]['adapter']
    clip = adapter.load_clip(case['sequence'], case['times']*C, layout_num_cameras=C,
        layout_num_times=T, sampling_seed=case['sampling_seed'])
    sample = prepare_clip(clip, image_mode='native', num_queries=0)
    attach_kubric_dense_recipe(clip, sample, image_mode='native', background_policy='objects_ground')
    batch = collate_adapter_samples([sample])
    sources = fixed_sources(batch, 4)
    target = materialize_dense_targets(batch, sources, normalize_scene=False)
    assert sample['query_time_idx'].tolist() == case['times']
    poses = np.asarray(clip.extrinsics, np.float64)
    first = poses[0]
    h, w = clip.depths[0].shape
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack((xx+.5, yy+.5, np.ones_like(xx)), -1)
    points, masks, errors, particle_errors = [], [], [], []
    with h5py.File(clip.metadata['h5_path'], 'r') as file:
        raw_poses = np.asarray(file['object_trajectories'][:, case['times']], np.float64)
        rotation = Rotation.from_quat(raw_poses[..., 3:].reshape(-1, 4)).as_matrix().reshape(*raw_poses.shape[:2], 3, 3)
        translation = raw_poses[..., :3]
        for slot, (camera, time) in enumerate(zip(sample['camera_id'].tolist(), sample['time_idx'].tolist())):
            depth, ids = file['depth'][time, camera], file['segmentation'][time, camera]
            K, pose = file['intrinsics'][camera], file['extrinsics'][camera]
            rays = pixels @ np.linalg.inv(K.astype(np.float64)).T
            ray_norm = np.linalg.norm(rays, axis=-1, keepdims=True)
            xyz = (rays/ray_norm*depth[..., None]) @ pose[:3, :3].astype(np.float64).T + pose[:3, 3]
            expected_valid = (np.isfinite(depth) & (depth > 0)
                & ((ids > 0) | ((ids == 0) & (abs(xyz[..., 2]) <= .05))))
            np.testing.assert_array_equal(target['depth_valid'][0, slot, :h, :w], expected_valid)
            np.testing.assert_allclose(clip.depths[slot], depth/ray_norm[..., 0], atol=3e-6, rtol=3e-6)
            assert not target['depth_valid'][0, slot, h:, :].any()
            assert not target['depth_valid'][0, slot, :, w:].any()
            points.append(xyz); masks.append(expected_valid)
        for q, slot in enumerate(sources[0].tolist()):
            frame = int(sample['time_idx'][slot]); t0 = case['times'].index(frame)
            cam = int(sample['camera_id'][slot]); ids = file['segmentation'][frame, cam]
            valid = masks[slot]
            world = points[slot][valid]
            labels = ids[valid]
            local = world.copy()
            dynamic = labels > 0
            obj = labels[dynamic]-1
            local[dynamic] = np.einsum('ni,nij->nj', world[dynamic]-translation[obj, t0], rotation[obj, t0])
            for t, raw_frame in enumerate(case['times']):
                moved = world.copy()
                moved[dynamic] = np.einsum('nij,nj->ni', rotation[obj, t], local[dynamic])+translation[obj, t]
                expected = moved @ first[:3, :3].T + first[:3, 3]
                predicted = target['trajectory'][0, q, t, :h, :w].numpy()[valid]
                error = np.linalg.norm(expected-predicted, axis=-1)
                assert target['trajectory_valid'][0, q, t, :h, :w][valid].all()
                assert target['displacement_valid'][0, q, t, :h, :w][valid].all()
                assert error.max() < 5e-5, (slot, raw_frame, float(error.max()))
                errors.append(dict(source_slot=slot, raw_frame=raw_frame, points=len(error),
                    p99_m=float(np.quantile(error, .99)), max_m=float(error.max())))
            assert not target['displacement'][0, q, t0].any()
        ids = np.asarray(file['track_object_ids'], int)-1
        tracks = np.asarray(file['tracks_3d'][:, case['times']], np.float64)
        assert (ids >= 0).all() and (ids < len(raw_poses)).all()
        local = np.einsum('ni,nij->nj', tracks[:, 0]-translation[ids, 0], rotation[ids, 0])
        for t in range(T):
            expected = np.einsum('nij,nj->ni', rotation[ids, t], local)+translation[ids, t]
            error = np.linalg.norm(expected-tracks[:, t], axis=-1)
            assert error.max() < 5e-5, (t, float(error.max()))
            particle_errors.append(dict(raw_frame=case['times'][t], points=len(error),
                p99_m=float(np.quantile(error, .99)), max_m=float(error.max())))
    report = dict(passed=True, case=case, layout=[C, T], h5=clip.metadata['h5_path'],
        source_indices=sources.tolist(), input_shape=list(sample['images'].shape),
        source_depth_ray_conversion_correct=True, background_mask_matches_raw_world=True,
        self_time_displacement_zero=True, dense_point_times=sum(row['points'] for row in errors),
        dense_oracle=errors, independent_particle_oracle=particle_errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('oracle')}), flush=True)


if __name__ == '__main__':
    main()
