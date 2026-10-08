"""Read-only checks against independent 2D, 3D, depth and calibration labels."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


def distribution(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    return dict(count=len(a), mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.quantile(a, .95)), max=float(a.max())) if len(a) else dict(count=0)


def audit_clip(clip):
    from multicamera4rc.datasets.bridge import prepare_clip

    K = np.asarray(clip.intrinsics, dtype=np.float64)
    E = np.asarray(clip.extrinsics, dtype=np.float64)
    if clip.metadata.get('extrinsics_convention') in ('c2w', 'camera_to_world'):
        E = np.linalg.inv(E)
    R = E[:, :3, :3]
    record = dict(sequence=clip.sequence_name,
                  frames=clip.metadata.get('frame_numbers', clip.metadata.get('frame_indices')),
                  native_hw=[list(np.asarray(im).shape[:2]) for im in clip.images],
                  pose_finite=bool(np.isfinite(E).all()),
                  rotation_orthogonality_max=float(np.abs(R @ R.transpose(0, 2, 1)-np.eye(3)).max()),
                  rotation_determinants=distribution(np.linalg.det(R)),
                  depth_encoding=clip.metadata.get('depth_encoding'),
                  depth_unit=clip.metadata.get('depth_unit'))
    X = np.asarray(clip.trajs_3d_world, dtype=np.float64)
    U = np.asarray(clip.trajs_2d, dtype=np.float64)
    valid = np.asarray(clip.valids, dtype=bool)
    visible = np.asarray(clip.visibs, dtype=bool) & valid
    record.update(track_count=X.shape[1], valid_3d_fraction=float(valid.mean()),
                  visible_fraction=float(visible.mean()),
                  finite_3d_given_valid=bool(np.isfinite(X[valid]).all()))
    camera = np.einsum('tij,tnj->tni', R, X) + E[:, None, :3, 3]
    homogeneous = np.einsum('tij,tnj->tni', K, camera)
    uv = homogeneous[..., :2] / np.where(np.abs(homogeneous[..., 2:]) > 1e-12, homogeneous[..., 2:], np.nan)
    error = np.linalg.norm(uv-U, axis=-1)
    record['reprojection_px_given_valid'] = distribution(error[valid])
    record['reprojection_px_given_visible'] = distribution(error[visible])
    record['positive_z_given_visible'] = float((camera[..., 2][visible] > 0).mean()) if visible.any() else None
    record['world_motion_from_first_frame'] = distribution(np.linalg.norm(X-X[:1], axis=-1)[valid])
    abs_errors, rel_errors, patch_errors = [], [], []
    support = []
    for t, depth in enumerate(clip.depths):
        D = np.asarray(depth, dtype=np.float64)
        h, w = D.shape
        m = visible[t] & np.isfinite(U[t]).all(-1) & (camera[t, :, 2] > 0)
        ids = np.flatnonzero(m)
        px = np.rint(U[t, ids]).astype(np.int64)
        inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        ids, px = ids[inside], px[inside]
        z = camera[t, ids, 2]
        d = D[px[:, 1], px[:, 0]]
        ok = np.isfinite(d) & (d > 0)
        abs_errors.extend(np.abs(d[ok]-z[ok]))
        rel_errors.extend(np.abs(d[ok]-z[ok])/np.maximum(z[ok], 1e-8))
        best = np.full(len(ids), np.inf)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x, y = px[:, 0]+dx, px[:, 1]+dy
                bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                sampled = D[y.clip(0, h-1), x.clip(0, w-1)]
                current = np.abs(sampled-z)/np.maximum(z, 1e-8)
                current[~bounds | ~np.isfinite(sampled) | (sampled <= 0)] = np.inf
                best = np.minimum(best, current)
        patch_errors.extend(best[np.isfinite(best)])
        support.extend(best <= .01)
    record['visible_depth_nearest_absolute_error'] = distribution(abs_errors)
    record['visible_depth_nearest_relative_error'] = distribution(rel_errors)
    record['visible_depth_3x3_relative_error'] = distribution(patch_errors)
    record['visible_depth_3x3_within_one_percent'] = float(np.mean(support)) if support else None
    sample = prepare_clip(clip, image_mode='native', num_queries=0)
    record['bridge'] = dict(input_shape=list(sample['images'].shape),
                            camera_ids=sample['camera_id'].tolist(),
                            time_indices=sample['time_idx'].tolist(),
                            query_time_indices=sample['query_time_idx'].tolist(),
                            valid_depth_pixels=int(sample['targets']['depth_valid'].sum()),
                            valid_camera_observations=int(sample['targets']['camera_valid'].sum()),
                            sparse_query_count=int(sample['targets']['sparse_queries'].shape[0]))
    record['checks'] = dict(
        pose_is_rotation=record['rotation_orthogonality_max'] < 1e-4,
        finite_valid_3d=record['finite_3d_given_valid'],
        track_projection_matches_2d=record['reprojection_px_given_visible'].get('p95', float('inf')) < .05,
        visible_vertices_have_positive_z=record['positive_z_given_visible'] is not None and record['positive_z_given_visible'] > .999,
        visible_vertex_depth_agrees=record['visible_depth_3x3_relative_error'].get('p95', float('inf')) < .05,
        native_depth_available=record['bridge']['valid_depth_pixels'] > 0,
        same_camera_across_frames=len(set(record['bridge']['camera_ids'])) == 1,
    )
    record['sample_passed'] = all(record['checks'].values())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('pointodyssey', 'dynamic_replica'), required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(2)
    began = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = str(args.output.parent/'index_cache')
    if args.dataset == 'pointodyssey':
        from multicamera4rc.datasets.adapters.pointodyssey import PointOdysseyAdapter
        adapter = PointOdysseyAdapter(args.root, split='train', require_tracks=True,
                    index_workers=2, io_workers=1, load_normals=False, verbose=False, cache_dir=cache)
        pool = adapter.list_sequences()
    else:
        from multicamera4rc.datasets.adapters.dynamic_replica import DynamicReplicaAdapter
        adapter = DynamicReplicaAdapter(args.root, split='train', load_trajectories=True,
                    rgb_from_trajectory=False, skip_depth_when_tracks=False,
                    index_workers=2, io_workers=1, verbose=False, cache_dir=cache,
                    pixel_coordinates='array_index_v1')
        pool = [s for s in adapter.list_sequences() if adapter.get_sequence_info(s)['has_tracks']]
    picked = random.Random(9157).sample(sorted(pool), min(args.samples, len(pool)))
    if args.dataset == 'pointodyssey' and 'ani' in pool and 'ani' not in picked:
        picked[0] = 'ani'
    report = dict(stage='running', dataset=args.dataset, root=args.root, indexed_sequences=len(adapter.list_sequences()),
                  sequences_with_tracks=len(pool), selected_sequences=picked, samples=[], errors=[],
                  gpu_used=False, native_tracks_densified=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','errors')}), flush=True)
    for sequence in picked:
        n = adapter.get_num_frames(sequence)
        frames = sorted(set([0, (n-1)//4, (n-1)//2, 3*(n-1)//4, n-1]))
        try:
            clip = adapter.load_clip(sequence, frames)
            result = audit_clip(clip)
            report['samples'].append(result)
            print(json.dumps({k:result[k] for k in ('sequence','sample_passed','track_count','native_hw','reprojection_px_given_visible','visible_depth_3x3_relative_error','checks')}), flush=True)
            del clip
        except Exception as error:
            import traceback
            record = dict(sequence=sequence, error=repr(error), traceback=traceback.format_exc())
            report['errors'].append(record)
            print(json.dumps(record), flush=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    consistent = bool(report['samples']) and not report['errors'] and all(s['sample_passed'] for s in report['samples'])
    report['stage'] = ('sampled_consistent' if consistent else
                       'needs_source_precision_review' if args.dataset == 'pointodyssey' and not report['errors'] else 'needs_review')
    report['strict_diagnostic_thresholds'] = {'rotation_orthogonality':1e-4,'visible_reprojection_p95_px':.05,
        'visible_depth_3x3_relative_p95':.05}
    if args.dataset == 'pointodyssey':
        report['source_precision_note'] = 'The original export uses float16 cameras/UV and can snap UV at segmentation boundaries. Run audit_point_precision.py before interpreting strict diagnostic failures as corrupted GT.'
    report['elapsed_seconds'] = time.monotonic()-began
    report['dense_geometry_available'] = bool(report['samples']) and all(s['checks']['native_depth_available'] for s in report['samples'])
    report['dense_long_term_3d_trajectory_available'] = False
    report['limitation'] = 'Selected train sequences and five frames per sequence only; index completeness does not prove all labels accurate. Persistent mesh or particle tracks are not full image-pixel trajectories.'
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'completed':str(args.output),'stage':report['stage'],'elapsed_seconds':report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
