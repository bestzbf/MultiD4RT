"""Paired released-weight evaluation on an explicit common 16-frame Sintel plan.

Uses the original 4RC/CUT3R depth scorer and evo trajectory metrics. The exact
OmniX paper split/runner is not published; this shared rerun is identified as
such instead of silently claiming replication of its reported table.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from .common import (digest, fingerprint, load_model, load_module, prepare_rgb,
                     predict_geometry, read_sintel_camera, read_sintel_depth, write_json)


def build_plan(root, long_side):
    scenes = sorted(p.name for p in (root / 'training/final').iterdir() if p.is_dir())
    cases, coverage = [], []
    for scene in scenes:
        paths = sorted((root / 'training/final' / scene).glob('frame_*.png'))
        coverage.append(dict(scene=scene, available_frames=len(paths),
                             evaluated_frames=len(paths) // 16 * 16,
                             short_tail_frames=len(paths) % 16))
        for start in range(0, len(paths) - 15, 16):
            ids = list(range(start, start + 16))
            case = dict(id=f'{scene}_{start:04d}', scene=scene, frame_indices=ids, files=[])
            for index in ids:
                prefix = f'frame_{index + 1:04d}'
                files = [root / 'training' / sub / scene / (prefix + extension)
                         for sub, extension in [('final', '.png'), ('depth', '.dpt'), ('camdata_left', '.cam')]]
                for path in files:
                    case['files'].append(dict(path=str(path), sha256=digest(path)))
                depth = read_sintel_depth(files[1])
                k, c2w = read_sintel_camera(files[2])
                if not np.isfinite(k).all() or not np.isfinite(c2w).all() or not ((depth > 0) & (depth < 70)).any():
                    raise ValueError('Invalid GT ' + str(files[1]))
            cases.append(case)
    if not cases:
        raise ValueError('Empty Sintel benchmark')
    return dict(schema='shared-sintel-16-v1', root=str(root), frames=16, long_side=long_side,
                scene_count=len(scenes), cases=cases, coverage=coverage,
                input='RGB only; entire field of view, aspect resize then right/bottom patch padding',
                depth='4RC/CUT3R original depth_evaluation; one S/SS alignment per clip, max_depth=70m',
                pose='evo Sim3 APE RMSE; adjacent-frame RPE translation RMSE and rotation-degree RMSE',
                comparison='Shared rerun of published checkpoints; not verified as the unreleased OmniX paper split')


def score_pose(predicted, ground_truth):
    from evo.core import metrics, trajectory
    gt = trajectory.PosePath3D(poses_se3=ground_truth.astype(np.float64))
    pred = trajectory.PosePath3D(poses_se3=predicted.astype(np.float64))
    try:
        # Identifiability is decided exclusively from fixed GT, independently
        # of either model's output. A collapsed prediction must not be dropped.
        deepcopy(gt).align(gt, correct_scale=True)
    except Exception as error:
        from evo.core.geometry import GeometryException
        if not isinstance(error, GeometryException):
            raise
        # The fixed GT has no identifiable Sim3 for this clip. Record it for
        # BOTH methods, rather than silently reporting scale-zero ATE as good.
        return dict(pose_alignment='undefined', pose_reason=str(error))
    rotation, translation, scale = pred.align(gt, correct_scale=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, pred))
    rt = metrics.RPE(metrics.PoseRelation.translation_part, delta=1,
                     delta_unit=metrics.Unit.frames, all_pairs=True)
    rr = metrics.RPE(metrics.PoseRelation.rotation_angle_deg, delta=1,
                     delta_unit=metrics.Unit.frames, all_pairs=True)
    rt.process_data((gt, pred))
    rr.process_data((gt, pred))
    return dict(pose_alignment='sim3', pose_scale=float(scale),
                ate_rmse_m=float(ape.get_statistic(metrics.StatisticsType.rmse)),
                rpe_translation_rmse_m=float(rt.get_statistic(metrics.StatisticsType.rmse)),
                rpe_rotation_rmse_deg=float(rr.get_statistic(metrics.StatisticsType.rmse)))


def score_case(prediction, case, depth_tools, device):
    gt_depth, gt_pose = [], []
    for i in range(0, len(case['files']), 3):
        gt_depth.append(read_sintel_depth(case['files'][i + 1]['path']))
        gt_pose.append(read_sintel_camera(case['files'][i + 2]['path'])[1])
    gt_depth = np.stack(gt_depth)
    height, width = gt_depth.shape[-2:]
    h, w = prediction['content_hw']
    depth = prediction['depth'][:, :h, :w]
    # Match the original video-depth evaluation's interpolation back to native GT.
    depth = np.stack([cv2.resize(x, (width, height), interpolation=cv2.INTER_CUBIC) for x in depth])
    # Bicubic interpolation can overshoot a positive depth discontinuity below
    # zero. Preserve the original scorer's handling of those values; never
    # filter them. Native model depth was already checked before interpolation.
    if not np.isfinite(depth).all():
        raise ValueError('Non-finite resized depth; prediction pixels cannot be filtered')
    result = {}
    for mode, kwargs in [('S', dict(align_with_scale=True)), ('SS', dict(align_with_lad2=True))]:
        values, *_ = depth_tools.depth_evaluation(depth, gt_depth, max_depth=70,
            use_gpu=device.type == 'cuda', post_clip_max=70, **kwargs)
        for key, value in values.items():
            if np.isscalar(value) or isinstance(value, torch.Tensor) and value.numel() == 1:
                number = float(value)
                if np.isfinite(number): result['depth_' + mode + '_' + key] = number
    result.update(score_pose(prediction['c2w'], np.stack(gt_pose)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['multid4rt', 'omnix'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--omnix-runner')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--depth-tools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--long-side', type=int, default=504)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.88, device)
    if args.plan.exists():
        plan = json.loads(args.plan.read_text())
        if plan['root'] != str(args.root.resolve()) or plan['long_side'] != args.long_side:
            raise ValueError('Requested inputs differ from the frozen plan')
    else:
        plan = build_plan(args.root.resolve(), args.long_side)
        write_json(args.plan, plan)
    args.output.mkdir(parents=True, exist_ok=True)
    depth_tools = load_module('_comparison_depth_tools', args.depth_tools)
    model, loading = load_model(args.method, args.checkpoint, device, args.omnix_runner)
    report_path = args.output / 'results.json'
    contract = dict(plan_sha256=fingerprint(plan), loading=loading,
                    depth_scorer_sha256=digest(args.depth_tools),
                    implementation={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                    input='RGB only; calibration/depth labels are accessed only by the scorer')
    contract = json.loads(json.dumps(contract))
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if not args.resume or old['contract'] != contract:
            raise ValueError('Cannot mix different evaluation contracts')
        records = old['records']
    else:
        records = []
    report = dict(stage='running', contract=contract, requested_cases=len(plan['cases']),
                  requested_scenes=plan['scene_count'], records=records, failures=[])
    write_json(report_path, report)
    done = {r['id'] for r in records}
    for case in plan['cases']:
        if case['id'] in done:
            continue
        began = time.monotonic()
        try:
            for file in case['files']:
                if digest(file['path']) != file['sha256']:
                    raise ValueError('GT/RGB changed after the plan was frozen')
            rgb = [np.asarray(Image.open(case['files'][i]['path']).convert('RGB'))
                   for i in range(0, len(case['files']), 3)]
            images, metadata = prepare_rgb(rgb, plan['long_side'])
            torch.cuda.reset_peak_memory_stats(device)
            prediction = predict_geometry(model, args.method, images, case['frame_indices'], device)
            prediction.update(metadata)
            np.savez_compressed(args.output / (case['id'] + '.npz'), **prediction)
            values = score_case(prediction, case, depth_tools, device)
            row = dict(id=case['id'], scene=case['scene'], frame_indices=case['frame_indices'],
                       input=metadata, metrics=values, seconds=time.monotonic() - began,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(device))
            records.append(row)
            print(json.dumps(dict(completed=len(records), total=len(plan['cases']), **row)), flush=True)
        except Exception:
            report['failures'].append(dict(id=case['id'], traceback=traceback.format_exc()))
            write_json(report_path, report)
            raise
        write_json(report_path, report)
        del images, prediction
        gc.collect()
        torch.cuda.empty_cache()
    numeric = sorted({k for row in records for k,v in row['metrics'].items()
                      if isinstance(v, (int, float)) and k != 'pose_scale'})
    report['mean_metrics'] = {
        key:float(np.mean([row['metrics'][key] for row in records if key in row['metrics']]))
        for key in numeric}
    report['metric_case_counts'] = {key:sum(key in row['metrics'] for row in records) for key in numeric}
    report['undefined_pose_cases'] = [r['id'] for r in records if r['metrics']['pose_alignment'] == 'undefined']
    report['stage'] = 'passed' if len(records) == len(plan['cases']) else 'incomplete'
    write_json(report_path, report)
    print(json.dumps({'stage':report['stage'], 'mean_metrics':report['mean_metrics'],
                      'undefined_pose_cases':report['undefined_pose_cases']}), flush=True)


if __name__ == '__main__':
    main()
