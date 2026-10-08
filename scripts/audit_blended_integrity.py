"""Check held-out Blended H5 pixels, calibration and independent view geometry.

This does not certify the unavailable pre-H5 source reconstruction. It checks
the exact held-out clips against the stored H5 and reports physical diagnostics
without confusing points behind a target surface with incorrect GT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import collate_adapter_samples
from scripts.evaluate_static_multiview import validation_dataset


def cross_view(depths, Ks, poses, shift=0., ray_distance=False):
    """Float64 raw-H5 oracle; smooth regions use bilinear target depth."""
    records = []
    for source, depth in enumerate(depths):
        h, w = depth.shape
        yy, xx = np.mgrid[1:h-1:4, 1:w-1:4]
        yy, xx = yy.ravel(), xx.ravel()
        valid = np.isfinite(depth[yy, xx]) & (depth[yy, xx] > 0)
        yy, xx = yy[valid], xx[valid]
        rays = np.c_[xx+shift, yy+shift, np.ones(len(xx))] @ np.linalg.inv(Ks[source]).T
        if ray_distance:
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        world = (rays*depth[yy, xx, None]) @ poses[source, :3, :3].T + poses[source, :3, 3]
        for other, observed in enumerate(depths):
            if other == source:
                continue
            camera = (world-poses[other, :3, 3]) @ poses[other, :3, :3]
            uvz = camera @ Ks[other].T
            uv = uvz[:, :2] / np.maximum(uvz[:, 2:3], 1e-12) - shift
            inside = (np.isfinite(uv).all(-1) & (camera[:, 2] > 0)
                      & (uv[:, 0] >= 1) & (uv[:, 0] < w-2)
                      & (uv[:, 1] >= 1) & (uv[:, 1] < h-2))
            uv, camera = uv[inside], camera[inside]
            if not len(uv):
                records.append(dict(source=source, target=other, interior_points=0,
                    matched_1pct=0, matched_2pct=0, in_front_2pct=0, behind_2pct=0,
                    median_absolute_relative_error=None))
                continue
            x, y = np.rint(uv).astype(int).T
            lo = cv2.erode(observed, np.ones((3, 3), np.uint8))[y, x]
            hi = cv2.dilate(observed, np.ones((3, 3), np.uint8))[y, x]
            smooth = (lo > 0) & (hi-lo < .01*lo)
            measured = cv2.remap(observed, uv[:, 0].astype(np.float32)[:, None],
                uv[:, 1].astype(np.float32)[:, None], cv2.INTER_LINEAR).ravel()
            valid = np.isfinite(measured) & (measured > 0) & smooth
            predicted = np.linalg.norm(camera, axis=-1) if ray_distance else camera[:, 2]
            residual = (predicted[valid]-measured[valid])/measured[valid]
            records.append(dict(source=source, target=other, interior_points=len(residual),
                matched_1pct=int((abs(residual) < .01).sum()),
                matched_2pct=int((abs(residual) < .02).sum()),
                in_front_2pct=int((residual < -.02).sum()),
                behind_2pct=int((residual > .02).sum()),
                median_absolute_relative_error=float(np.median(abs(residual))) if len(residual) else None))
    totals = {key: sum(row[key] for row in records) for key in
              ('interior_points', 'matched_1pct', 'matched_2pct', 'in_front_2pct', 'behind_2pct')}
    return dict(totals=totals, pairs=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-config', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset, splits = validation_dataset(args.train_config, args.output.parent/'static_manifests')
    plan = json.loads(args.plan.read_text())
    assert plan['sources'] == [source['display_name'] for source in dataset.sources]
    report = dict(stage='running', splits=splits, cases=[])
    for case in plan['cases']:
        source = dataset.sources[case['source']]
        if source['display_name'] != 'blended':
            continue
        source['camera_choices'] = (case['requested_cameras'],)
        sample = dataset[case['index']]
        views = sample['static_multiview']['view_indices']
        assert views == case['view_indices']
        clip = source['adapter'].load_clip(case['sequence'], views)
        raw_depths, raw_K, raw_E = [], [], []
        with h5py.File(source['adapter'].root, 'r') as file:
            group = file['train/'+case['sequence']]
            crops = []
            for slot, frame in enumerate(views):
                image, depth = group['images'][frame], group['depths'][frame]
                K, E = group['intrinsics'][frame], group['extrinsics'][frame]
                # Recover and independently check the exact crop from cy.
                crop_top = int(round(float(K[1, 2]-clip.intrinsics[slot, 1, 2])))
                h, w = clip.images[slot].shape[:2]
                crop_bottom = image.shape[0]-crop_top-h
                assert crop_top >= 0 and crop_bottom >= 0
                np.testing.assert_array_equal(clip.images[slot], image[crop_top:crop_top+h])
                expected_depth = depth[crop_top:crop_top+h].copy()
                expected_depth[~np.isfinite(expected_depth) | (expected_depth <= 0)] = 0
                np.testing.assert_array_equal(clip.depths[slot], expected_depth)
                expected_K = K.copy(); expected_K[1, 2] -= crop_top
                np.testing.assert_array_equal(clip.intrinsics[slot], expected_K)
                np.testing.assert_array_equal(clip.extrinsics[slot], E)
                assert not sample['targets']['depth_valid'][slot, h:, :].any()
                assert not sample['targets']['depth_valid'][slot, :, w:].any()
                torch.testing.assert_close(sample['images'][slot, :, :h, :w],
                    torch.from_numpy(clip.images[slot]).permute(2, 0, 1).float()/255, atol=0, rtol=0)
                raw_depths.append(depth); raw_K.append(K); raw_E.append(E)
                crops.append([crop_top, crop_bottom])
        target = materialize_dense_targets(collate_adapter_samples([sample]),
                                           torch.arange(min(4, len(views)))[None])
        assert not target['displacement_valid'].any() and not target['static_mask'].any()
        assert target['trajectory_valid'].sum() == target['source_pixel_valid'].sum()
        for key in ('depth', 'ray', 'trajectory', 'displacement', 'c2w'):
            assert torch.isfinite(target[key]).all(), key
        poses = np.linalg.inv(np.asarray(raw_E, np.float64))
        ref = np.asarray(raw_E[0], np.float64)
        expected = ref[None] @ poses
        expected[:, :3, 3] /= float(target['scene_scale'][0])
        np.testing.assert_allclose(target['c2w'][0], expected, atol=3e-6, rtol=3e-6)
        for slot, d in enumerate(clip.depths):
            h, w = d.shape
            np.testing.assert_allclose(target['depth'][0, slot, :h, :w],
                d/float(target['scene_scale'][0]), atol=1e-7, rtol=2e-6)
        row = dict(case=case, native_shape=list(sample['images'].shape), crops=crops,
            raw_h5_rgb_depth_calibration_equal=True, normalized_loss_targets_equal=True,
            no_fabricated_motion=True, padded_pixels_excluded=True,
            diagnostics={name: cross_view(raw_depths, np.asarray(raw_K, np.float64), poses, **kw)
                for name, kw in [('z_depth_integer_pixels', {}),
                    ('z_depth_plus_half_pixel', dict(shift=.5)),
                    ('z_depth_minus_half_pixel', dict(shift=-.5)),
                    ('ray_distance_hypothesis', dict(ray_distance=True))]})
        report['cases'].append(row)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(sequence=case['sequence'], cameras=len(views),
            diagnostics={k: v['totals'] for k, v in row['diagnostics'].items()})), flush=True)
    assert report['cases']
    report['stage'] = 'completed'
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
