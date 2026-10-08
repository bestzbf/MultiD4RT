"""Independent float64 and raw-depth diagnostics for native Syn4D supervision.

All source pixels are materialized by the production dense loader. Projection
diagnostics sample surface pixels per mesh; they never change training masks.
Occluded points are reported separately from points in front of raw depth.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from multicamera4rc.datasets.adapters.syn4d import Syn4DAdapter
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import collate_adapter_samples
from multicamera4rc.datasets.syn4d_dense import attach_syn4d_dense_recipe


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(n=len(values), **{name: float(v) for name, v in zip(
        ('p50', 'p90', 'p99', 'max'), np.quantile(values, [.5, .9, .99, 1]))}) if len(values) else dict(n=0)


def lift(xy, depth, K, C):
    rays = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(K.astype(np.float64)).T
    return rays * depth[:, None] @ C[:3, :3].astype(np.float64).T + C[:3, 3]


def project(points, depth, K, C, source_xy=None):
    """Classify against raw target depth, with no target visibility GT reuse."""
    h, w = depth.shape
    pc = (points - C[:3, 3]) @ C[:3, :3].astype(np.float64)
    uvz = pc @ K.astype(np.float64).T
    uv = uvz[:, :2] / np.maximum(uvz[:, 2:3], 1e-12)
    inside = (np.isfinite(uv).all(-1) & (pc[:, 2] > 0) & (uv[:, 0] >= 1)
              & (uv[:, 0] < w-2) & (uv[:, 1] >= 1) & (uv[:, 1] < h-2))
    ids = np.flatnonzero(inside)
    result = dict(points=len(points), inside=len(ids))
    if source_xy is not None:
        result['self_uv_error_px'] = quantiles(np.linalg.norm(uv - source_xy, axis=-1))
    if not len(ids):
        return result
    pixel = np.rint(uv[ids]).astype(np.int64)
    x, y = pixel.T
    measured, predicted = depth[y, x], pc[ids, 2]
    valid = np.isfinite(measured) & (measured > 0)
    result['has_depth'] = int(valid.sum())
    if not valid.any():
        return result
    measured, predicted, x, y = measured[valid], predicted[valid], x[valid], y[valid]
    residual = predicted - measured
    tolerance = .03 + .01 * measured
    lo = cv2.erode(depth, np.ones((3, 3), np.uint8))[y, x]
    hi = cv2.dilate(depth, np.ones((3, 3), np.uint8))[y, x]
    interior = (lo > 0) & ((hi-lo) <= .01 + .005*measured)
    result.update(consistent=int((abs(residual) <= tolerance).sum()),
        behind_surface=int((residual > tolerance).sum()),
        in_front_of_surface=int((residual < -tolerance).sum()),
        interior_points=int(interior.sum()),
        interior_in_front=int(((residual < -tolerance) & interior).sum()),
        absolute_depth_residual_m=quantiles(abs(residual)),
        negative_depth_residual_m=quantiles(-residual[residual < -tolerance]))
    return result


def audit_case(root, case, *, sample_pixels=2000):
    started = time.monotonic()
    subset, scene, _ = case['sequence'].split('/')
    adapter = Syn4DAdapter(root, split='all', subsets=[subset], scenes=[scene],
        frame_policy='all' if 'v1' in subset else 'reference', require_human_meshes='v1' in subset)
    times, C = case['times'], case.get('cameras', 2)
    T = len(times)
    clip = adapter.load_clip(case['sequence'], times*C, layout_num_cameras=C,
        layout_num_times=T, sampling_seed=case['sampling_seed'])
    sample = prepare_clip(clip, image_mode='native', num_queries=0)
    attach_syn4d_dense_recipe(clip, sample, image_mode='native', include_visibility=False)
    batch = collate_adapter_samples([sample])
    sources = [0, (C-1)*T+T-1]
    target = materialize_dense_targets(batch, torch.tensor([sources]), normalize_scene=False)
    assert all(torch.isfinite(target[k]).all() for k in ('depth', 'trajectory', 'displacement', 'c2w'))
    raw_poses = np.asarray(clip.extrinsics, np.float64)
    reference = np.linalg.inv(raw_poses[0])
    rows = []
    offsets, offset = {}, 0
    for name, (_, faces) in sorted(clip.metadata['syn4d_meshes'].items()):
        offsets[name] = offset
        offset += len(faces)
    for q, slot in enumerate(sources):
        source_time = slot % T
        h, w = clip.depths[slot].shape
        face_map = sample['targets']['dense_syn4d_face_map'][slot, :h, :w].numpy()
        for name, (vertices, faces) in sorted(clip.metadata['syn4d_meshes'].items()):
            local_ids = face_map - offsets[name]
            region = (local_ids >= 0) & (local_ids < len(faces)) & (clip.depths[slot] > 0)
            flat = np.flatnonzero(region)
            total = len(flat)
            if not total:
                continue
            rng = np.random.default_rng(20261008 + slot)
            flat = rng.choice(flat, min(sample_pixels, total), replace=False)
            y, x = np.unravel_index(flat, (h, w))
            xy = np.stack((x, y), -1)
            base_world = lift(xy, clip.depths[slot][y, x], clip.intrinsics[slot], raw_poses[slot])
            ids = local_ids[y, x]
            triangles = vertices[source_time, faces[ids]].astype(np.float64)
            origin = triangles[:, 0]
            e0, e1 = triangles[:, 1]-origin, triangles[:, 2]-origin
            delta = base_world-origin
            design = np.stack((e0, e1), -1)
            # Float64 least-squares, independent of production barycentric arithmetic.
            bary12 = (np.linalg.pinv(design) @ delta[..., None])[..., 0]
            bary = np.c_[1-bary12.sum(-1), bary12]
            normal = np.cross(e0, e1)
            length = np.linalg.norm(normal, axis=-1)
            normal /= np.maximum(length[:, None], 1e-30)
            signed = ((base_world-(triangles*bary[..., None]).sum(1))*normal).sum(-1)
            admitted = target['displacement_valid'][0, q, source_time, y, x].numpy()
            one = dict(mesh=name, source_slot=slot, raw_source_frame=clip.metadata['time_indices'][source_time],
                released_pixels=total, sampled_pixels=len(flat), admitted_pixels=int(admitted.sum()),
                signed_distance_abs_m=quantiles(abs(signed)),
                bary_outside_2pct=int(((bary.min(-1) < -.02) | (bary.max(-1) > 1.02)).sum()),
                bary_abs_max=quantiles(abs(bary).max(-1)), projections=[], production_error_m=[])
            for t in range(T):
                triangle = vertices[t, faces[ids]].astype(np.float64)
                n = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
                n /= np.maximum(np.linalg.norm(n, axis=-1)[:, None], 1e-30)
                expected_world = (triangle*bary[..., None]).sum(1) + signed[:, None]*n
                if t == source_time:
                    expected_world = base_world
                expected = expected_world @ reference[:3, :3].T + reference[:3, 3]
                actual = target['trajectory'][0, q, t, y, x].numpy()
                valid = target['displacement_valid'][0, q, t, y, x].numpy()
                diff = np.linalg.norm(actual-expected, axis=-1)
                one['production_error_m'].append(dict(time=t, **quantiles(diff[valid])))
                for c in range(C):
                    other = c*T+t
                    projection = project(expected_world[valid], clip.depths[other], clip.intrinsics[other],
                        raw_poses[other], source_xy=xy[valid] if other == slot else None)
                    one['projections'].append(dict(target_slot=other,
                        raw_target_frame=clip.metadata['time_indices'][t], **projection))
            rows.append(one)
    # Background geometry dominates the dense depth loss. Check it separately
    # from compact mesh tracks, including distant surfaces and static motion.
    environment_rows = []
    for q, slot in enumerate(sources):
        h, w = clip.depths[slot].shape
        raw_depth = clip.depths[slot]
        env = sample['targets']['dense_syn4d_environment'][slot, :h, :w].numpy()
        region = env & (raw_depth > 0)
        for label, lower, upper in [('near', 0, 10), ('middle', 10, 50), ('far', 50, np.inf)]:
            flat = np.flatnonzero(region & (raw_depth >= lower) & (raw_depth < upper))
            total = len(flat)
            if not total:
                continue
            rng = np.random.default_rng(751 + slot)
            flat = rng.choice(flat, min(5000, total), replace=False)
            y, x = np.unravel_index(flat, (h, w))
            xy = np.stack((x, y), -1)
            world = lift(xy, raw_depth[y, x], clip.intrinsics[slot], raw_poses[slot])
            expected = world @ reference[:3, :3].T + reference[:3, 3]
            errors, projections = [], []
            for t in range(T):
                valid = target['displacement_valid'][0, q, t, y, x].numpy()
                assert valid.all(), 'valid environment pixels lost their static targets'
                assert not target['displacement'][0, q, t, y, x].any()
                actual = target['trajectory'][0, q, t, y, x].numpy()
                errors.append(dict(time=t, **quantiles(np.linalg.norm(actual-expected, axis=-1))))
                for c in range(C):
                    other = c*T+t
                    projection = project(world, clip.depths[other], clip.intrinsics[other],
                        raw_poses[other], source_xy=xy if other == slot else None)
                    projections.append(dict(target_slot=other,
                        raw_target_frame=clip.metadata['time_indices'][t], **projection))
            environment_rows.append(dict(source_slot=slot, depth_range=label, released_pixels=total,
                sampled_pixels=len(flat), depth_m=quantiles(raw_depth[y, x]),
                production_error_m=errors, projections=projections))
    result = dict(case=case, raw_times=clip.metadata['time_indices'], source_slots=sources,
        input_shape=list(sample['images'].shape),
        physical_camera_ids=sample['camera_id'].unique().tolist(), missing_meshes=clip.metadata['syn4d_missing_meshes'],
        all_materialized_targets_finite=True, meshes=rows, environment=environment_rows,
        positive_depth_m=quantiles(np.concatenate([d[d > 0] for d in clip.depths])),
        seconds=time.monotonic()-started)
    del target, batch, clip, sample, adapter
    gc.collect()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/data6/Syn4D')
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--pixels-per-mesh', type=int, default=2000)
    p.add_argument('--cpu-threads', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.cpu_threads)
    plan = json.loads(args.plan.read_text())
    report = dict(stage='running', diagnostic_only=True,
        pixels_per_mesh=args.pixels_per_mesh, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in plan['cases']:
        result = audit_case(args.root, case, sample_pixels=args.pixels_per_mesh)
        report['cases'].append(result)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(sequence=case['sequence'], times=len(case['times']),
            mesh_diagnostics=len(result['meshes']), seconds=result['seconds'])), flush=True)
    report['stage'] = 'completed'
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
