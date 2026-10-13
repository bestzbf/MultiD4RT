"""OmniX-style 16-frame first-visible tracking with the official TAPVid-3D scorer.

APD3D uses the untouched official depth-adaptive thresholds and median global
scale. EPE is visible-point Euclidean error in meters after that same scale.
This protocol is separate from full-video, original-query D4RT evaluation.
"""
from __future__ import annotations

import argparse
import ast
import gc
import io
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .common import digest, fingerprint, load_model, load_module, prepare_rgb, write_json


def build_plan(manifest, root, subset, split, long_side):
    cases, tails, empty = [], [], []
    for name in manifest['datasets'][subset][split]:
        path = root / subset / name
        checksum = digest(path)
        with np.load(path, allow_pickle=True) as data:
            visibility = data['visibility'].astype(bool)
            tracks = data['tracks_XYZ']
            if tracks.shape != (*visibility.shape, 3):
                raise ValueError('Invalid official GT shape')
            frames = len(data['images_jpeg_bytes'])
            tails.append(dict(sequence=name, full_frames=frames, tail_frames=frames % 16))
            for start in range(0, frames - 15, 16):
                # These labels define supplied starting queries only. Future
                # visibility is never made available to the model or sampler.
                ids = np.flatnonzero(visibility[start]).tolist()
                if not ids:
                    empty.append(dict(sequence=name, start=start, reason='no visible starting query'))
                    continue
                cases.append(dict(id=name.removesuffix('.npz') + f'_{start:04d}',
                                  path=str(path), sequence=name, start=start, frames=16,
                                  point_ids=ids, sha256=checksum))
    return dict(schema='omnix16-first-visible-v2', subset=subset, split=split,
                long_side=long_side, frames=16, cases=cases, short_tails=tails,
                empty_query_clips=empty, source_commit=manifest['commit'],
                query='every GT-visible point at the first frame of each 16-frame clip; supplied XY only',
                camera='local coordinates using predicted target c2w; no GT pose registration',
                metric_eval_short_side=256,
                scoring='official depth-adaptive APD3D; scoring intrinsics at short side 256; median scale on GT-visible camera-coordinate norms',
                epe='same clip-wide median scale, then mean Euclidean error at GT-visible point-times, meters',
                visibility='APD ignores predicted occlusion; no AJ/OA reported for OmniX without an official occlusion predictor')


def sample(field, xy):
    h, w = field.shape[-2:]
    grid = torch.stack((2 * xy[:, 0] / (w - 1) - 1, 2 * xy[:, 1] / (h - 1) - 1), -1)
    grid = grid[None, :, None].expand(len(field), -1, -1, -1)
    return F.grid_sample(field.float(), grid, align_corners=True, mode='bilinear')[..., 0].transpose(1, 2)


@torch.inference_mode()
def predict(model, method, images, raw_xy, metadata, start, device):
    images = images.to(device)
    t = images.shape[1]
    raw_h, raw_w = metadata['raw_hw']
    h, w = metadata['content_hw']
    xy = torch.as_tensor(raw_xy, device=device, dtype=torch.float32).clone()
    xy[:, 0] = (xy[:, 0] + .5) * w / raw_w - .5
    xy[:, 1] = (xy[:, 1] + .5) * h / raw_h - .5
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        if method == 'multid4rt':
            ids = torch.arange(start, start + t, device=device)[None]
            encoded = model.encode(images, torch.zeros_like(ids), ids)
            output = model.decode(encoded, query_time_idx=ids,
                                  source_obs_idx=torch.tensor([[0]], device=device), return_visibility=False)
            field = output['trajectory'][0, 0].permute(0, 3, 1, 2)
            pose = encoded.geometry['c2w'][0]
        else:
            ids = torch.arange(t, device=device)
            info = torch.stack((ids, torch.zeros_like(ids), ids, ids), -1)[None]
            output = model(dict(image=images, image_info=info))
            field = output['trajectory'][0, 0].permute(0, 3, 1, 2)
            pose = output['camera_pose'][0]
    with torch.autocast(device.type, enabled=False):
        world = sample(field, xy)
        # c2w is supplied by each method's own camera head, including at t=0.
        local = torch.einsum('tji,tnj->tni', pose[:, :3, :3].float(),
                             world - pose[:, None, :3, 3].float())
    if not torch.isfinite(local).all():
        raise ValueError('Non-finite track prediction; no query may be discarded')
    return local.cpu().numpy()


def metric_intrinsics(intrinsics, raw_hw):
    """Match evaluate_model.py: evaluation resolution means short side 256.

    This camera is only used for distance thresholds. The camera-coordinate
    XYZ in meters and the model's RGB/input camera remain unchanged.
    """
    if len(raw_hw)!=2 or min(raw_hw)<=0:
        raise ValueError('Original image dimensions are required for scoring')
    return np.asarray(intrinsics,dtype=np.float64)*(256/min(raw_hw))


def score(official, predicted, truth, visible, intrinsics, raw_hw):
    # APD does not require a predicted occlusion classifier. Treat all predicted
    # points as visible here for BOTH methods; never feed GT as model visibility.
    scoring_intrinsics=metric_intrinsics(intrinsics,raw_hw)
    result = official.compute_tapvid3d_metrics(~visible, truth,
        np.zeros_like(visible, dtype=bool), predicted, scoring_intrinsics,
        order='t n', scaling='median')
    gt_norm = np.sqrt(np.maximum(1e-12, np.square(truth).sum(-1)))
    pred_norm = np.sqrt(np.maximum(1e-12, np.square(predicted).sum(-1)))
    # The original scorer's float64 nan_mat promotes the already computed
    # float32 norms BEFORE taking medians. Preserve this exact order, including
    # the float64 product; boundary thresholds can change at float32 precision.
    scale = np.median(gt_norm[visible].astype(np.float64)) / np.median(pred_norm[visible].astype(np.float64))
    scaled = predicted.astype(np.float64) * scale
    # Independently verify the explicitly computed scale against the official
    # result, without depending on the scorer's internal temporary variables.
    check = official.compute_tapvid3d_metrics(~visible, truth,
        np.zeros_like(visible, dtype=bool), scaled, scoring_intrinsics,
        order='t n', scaling='none')
    np.testing.assert_allclose(result['average_pts_within_thresh'],
                               check['average_pts_within_thresh'], rtol=0, atol=1e-12)
    return dict(apd3d=float(result['average_pts_within_thresh']),
                epe_m=float(np.linalg.norm(scaled - truth, axis=-1)[visible].mean()),
                global_scale=float(scale), queries=truth.shape[1],
                visible_point_times=int(visible.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['multid4rt', 'omnix'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--omnix-runner')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--official-metrics', type=Path, required=True)
    parser.add_argument('--subset', choices=['pstudio', 'adt', 'drivetrack'], required=True)
    parser.add_argument('--split', choices=['full_eval', 'minival'], default='full_eval')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--long-side', type=int, default=504)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--migrate-scoring', action='store_true')
    parser.add_argument('--partitions', type=int, default=1)
    parser.add_argument('--partition-index', type=int, default=0)
    parser.add_argument('--memory-fraction', type=float, default=.88)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if not 0 <= args.partition_index < args.partitions or not 0 < args.memory_fraction <= 1:
        raise ValueError('Invalid resource partition')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    manifest = json.loads(args.manifest.read_text())
    if args.plan.exists():
        plan = json.loads(args.plan.read_text())
        if any(plan[k] != getattr(args,k) for k in ['subset','split','long_side']):
            raise ValueError('Requested evaluation differs from frozen plan')
        if plan.get('metric_eval_short_side')!=256:
            raise ValueError('Legacy scoring plan: migrate with evaluation.rescore_tapvid_resolution')
    else:
        plan = build_plan(manifest, args.root, args.subset, args.split, args.long_side)
        write_json(args.plan, plan)
    scorer_relative = 'tapnet/tapvid3d/evaluation/metrics.py'
    if digest(args.official_metrics) != manifest['sha256'][scorer_relative]:
        raise ValueError('Official scorer source changed')
    official = load_module('_official_tapvid3d_metrics', args.official_metrics)
    model, loading = load_model(args.method, args.checkpoint, device, args.omnix_runner)
    selected_cases = [case for index,case in enumerate(plan['cases'])
                      if index % args.partitions == args.partition_index]
    contract = dict(plan_sha256=fingerprint(plan), loading=loading,
        metric_eval_short_side=256,
        official_caller_sha256=digest(Path(__file__).parent/'references/tapvid3d_evaluate_model.py'),
        partitions=args.partitions, partition_index=args.partition_index,
        memory_fraction=args.memory_fraction,
        scorer_sha256=digest(args.official_metrics),
        implementation={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('common.py'),Path(__file__).with_name('native_omnix.py')]})
    contract = json.loads(json.dumps(contract))
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / 'results.json'
    records = []
    reuse_predictions = False
    migration = None
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if not args.resume:
            raise ValueError('Existing results require explicit resume')
        if old['contract'] == contract:
            records = old['records']
        elif args.migrate_scoring:
            previous_source = Path(__file__).parent/'references/run_tapvid_v1.py'
            old_contract = old['contract']
            expected = dict(old_contract)
            expected['implementation'] = contract['implementation']
            if expected != contract:
                raise ValueError('Scoring migration cannot change weights, input plan, partition or scorer')
            for name in ['common.py','native_omnix.py']:
                if old_contract['implementation'][name] != contract['implementation'][name]:
                    raise ValueError('Model execution changed; cached inference cannot be reused')
            if digest(previous_source) != old_contract['implementation']['run_tapvid.py']:
                raise ValueError('Previous executed source snapshot is unavailable')
            def functions(path):
                tree = ast.parse(path.read_text())
                return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
                        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            before,after = functions(previous_source),functions(Path(__file__))
            if any(before[name] != after[name] for name in ['predict','sample','build_plan']):
                raise ValueError('Prediction/input functions changed; do not reuse inference')
            migration = dict(reason='EPE scale precision matched to original float64 scorer',
                             previous_contract=old_contract, old_failures=old.get('failures',[]),
                             unchanged_prediction_functions=True, new_model_forwards_for_cached_predictions=0)
            backup = args.output/'results_before_precision_migration.json'
            if not backup.exists(): write_json(backup,old)
            reuse_predictions = True
        else:
            raise ValueError('Cannot mix different checkpoint/protocol predictions')
    report = dict(stage='running', requested_cases=len(plan['cases']),
                  partition_cases=len(selected_cases),
                  requested_sequences=len(manifest['datasets'][args.subset][args.split]),
                  contract=contract, records=records, failures=[])
    if migration is not None:
        report['scoring_migration'] = migration
    write_json(result_path, report)
    done = {r['id'] for r in records}
    verified_files = set()
    for case in selected_cases:
        if case['id'] in done:
            continue
        began = time.monotonic()
        try:
            if case['path'] not in verified_files:
                if digest(case['path']) != case['sha256']:
                    raise ValueError('Official RGB/GT archive changed after freezing the plan')
                verified_files.add(case['path'])
            with np.load(case['path'], allow_pickle=True) as data:
                start, stop = case['start'], case['start'] + 16
                points = case['point_ids']
                rgb = [np.asarray(Image.open(io.BytesIO(bytes(x))).convert('RGB'))
                       for x in data['images_jpeg_bytes'][start:stop]]
                truth = data['tracks_XYZ'][start:stop, points].astype(np.float32)
                visible = data['visibility'][start:stop, points].astype(bool)
                intrinsics = data['fx_fy_cx_cy'].astype(np.float64)
            source = truth[0]
            raw_xy = source[:, :2] / source[:, 2:] * intrinsics[:2] + intrinsics[2:]
            images, metadata = prepare_rgb(rgb, plan['long_side'])
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            inference_began = time.monotonic()
            cached_path = args.output/(case['id']+'.npz')
            if reuse_predictions and cached_path.exists():
                with np.load(cached_path,allow_pickle=False) as cached:
                    if int(cached['start']) != start or not np.array_equal(cached['point_ids'],points):
                        raise ValueError('Cached predictions belong to other starting queries')
                    prediction = cached['tracks_XYZ'].copy()
                if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                    raise ValueError('Invalid cached predictions')
                reused = True
            else:
                prediction = predict(model, args.method, images, raw_xy, metadata, start, device)
                reused = False
            torch.cuda.synchronize(device)
            inference_seconds = time.monotonic() - inference_began
            np.savez_compressed(args.output / (case['id'] + '.npz'),
                                tracks_XYZ=prediction, point_ids=points, start=start)
            values = score(official, prediction, truth, visible, intrinsics, metadata['raw_hw'])
            row = dict(id=case['id'], sequence=case['sequence'], metrics=values, input=metadata,
                       seconds=time.monotonic() - began,
                       inference_seconds=inference_seconds,
                       reused_inference=reused,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(device))
            records.append(row)
            print(json.dumps(dict(completed=len(records), total=len(plan['cases']), **row)), flush=True)
        except Exception:
            report['failures'].append(dict(id=case['id'], traceback=traceback.format_exc()))
            write_json(result_path, report)
            raise
        write_json(result_path, report)
        del images, prediction
        gc.collect()
        torch.cuda.empty_cache()
    # Paper values are video averages. Keep equal sequence weight after clips
    # have been averaged within each video; also expose clip-macro separately.
    by_sequence = {}
    for sequence in {r['sequence'] for r in records}:
        selected = [r for r in records if r['sequence'] == sequence]
        by_sequence[sequence] = {k:float(np.mean([r['metrics'][k] for r in selected]))
                                 for k in ['apd3d', 'epe_m']}
    report['by_sequence'] = by_sequence
    report['mean_metrics'] = {k:float(np.mean([v[k] for v in by_sequence.values()])) for k in ['apd3d','epe_m']}
    report['clip_mean_metrics'] = {k:float(np.mean([r['metrics'][k] for r in records])) for k in ['apd3d','epe_m']}
    complete = len(records) == len(selected_cases)
    report['stage'] = ('passed' if args.partitions == 1 else 'partition_complete') if complete else 'incomplete'
    write_json(result_path, report)
    print(json.dumps({'stage':report['stage'], 'mean_metrics':report['mean_metrics']}), flush=True)


if __name__ == '__main__':
    main()
