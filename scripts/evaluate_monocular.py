"""Evaluate MultiCamera4RC on legacy D4RT Sintel / ScanNet geometry benchmarks."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from scripts.monocular_protocol import (SINTEL_14_SEQUENCES, frame_indices, load_clip,
                                        geometry_metrics, geometry_on_eval_grid)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def build_plan(args):
    import h5py
    roots = [Path(p).resolve() for p in args.roots]
    cases = []
    if args.dataset == 'sintel':
        names = args.scene_names or list(SINTEL_14_SEQUENCES)
        for name in names:
            matches = [r for r in roots if (r/'training/final'/name).is_dir()]
            if len(matches) != 1:
                raise ValueError('Sintel scene must resolve to exactly one root: '+name)
            n = len(list((matches[0]/'training/final'/name).glob('frame_*.png')))
            cases.append(dict(dataset='sintel', root=str(matches[0]), scene=name,
                frame_indices=frame_indices(n, args.num_frames, args.frame_stride, args.start_frame, dataset='sintel')))
    else:
        found = {}
        for r in roots:
            for p in sorted(r.glob('scene*')):
                if not p.is_dir(): continue
                if p.name in found: raise ValueError('duplicate ScanNet scene: '+p.name)
                found[p.name] = r
        names = args.scene_names or sorted(found)
        for name in names:
            r = found[name]
            with h5py.File(r/name/'processed/precomputed.h5', 'r') as f:
                n = int(f['num_frames'][()])
            cases.append(dict(dataset='scannet', root=str(r), scene=name,
                frame_indices=frame_indices(n, args.num_frames, args.frame_stride, args.start_frame, dataset='scannet')))
    if args.max_scenes:
        cases = cases[:args.max_scenes]
    if not cases: raise ValueError('empty benchmark')
    return dict(format_version=1, dataset=args.dataset, eval_size=args.eval_size,
                long_side=args.long_side, cases=cases)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=('sintel', 'scannet'))
    p.add_argument('--roots', nargs='+', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--scene-names', nargs='+')
    p.add_argument('--num-frames', type=int, default=48)
    p.add_argument('--frame-stride', type=int, default=1)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--eval-size', type=int, default=256)
    p.add_argument('--long-side', type=int, default=518, help='Aspect-preserving input maximum; 0 keeps native RGB')
    p.add_argument('--max-scenes', type=int)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--memory-fraction', type=float, default=.8)
    p.add_argument('--resume-results', action='store_true')
    args = p.parse_args(argv)
    if args.cpu_threads <= 0 or args.long_side < 0 or args.eval_size < 2 or not 0 < args.memory_fraction <= 1:
        p.error('invalid resource / resolution settings')
    if not args.audit_only and not args.checkpoint:
        p.error('--checkpoint is required for inference')
    torch.set_num_threads(args.cpu_threads)
    if args.plan.exists():
        plan = json.loads(args.plan.read_text())
        if any(plan[k] != getattr(args, k) for k in ('dataset', 'eval_size', 'long_side')):
            raise ValueError('requested settings differ from frozen plan')
    else:
        plan = build_plan(args)
        write_json(args.plan, plan)
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        audit = dict(passed=False, plan_sha256=plan_sha, requested_cases=len(plan['cases']), cases=[], failed_cases=[])
        for case in plan['cases']:
            try:
                clip = load_clip(case, eval_size=args.eval_size, long_side=args.long_side)
                limit = 70 if args.dataset == 'sintel' else 10
                mask = torch.isfinite(clip.depth) & (clip.depth > 0) & (clip.depth < limit)
                row = dict(scene=case['scene'], frame_indices=case['frame_indices'], raw_hw=clip.raw_hw,
                    content_hw=clip.content_hw, input_shape=list(clip.images.shape), calibration_sha256=clip.calibration_sha256,
                    valid_depth_fraction=float(mask.float().mean()), valid_depth_pixels=int(mask.sum()))
                audit['cases'].append(row)
                del clip
                print(json.dumps(dict(audited=row['scene'], progress=len(audit['cases']))), flush=True)
            except Exception as e:
                audit['failed_cases'].append(dict(scene=case['scene'], error=repr(e)))
            write_json(args.output/'gt_audit.json', audit)
        audit['passed'] = not audit['failed_cases'] and len(audit['cases']) == len(plan['cases'])
        write_json(args.output/'gt_audit.json', audit)
        if not audit['passed']: raise RuntimeError('benchmark GT audit failed; see gt_audit.json')
        return audit
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index if device.index is not None else 0)
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    cfg = replace(config_from_checkpoint(payload), gradient_checkpointing=False,
                  attention_chunk_size=None, visibility_chunk_size=None)
    model = MultiCamera4RC(cfg); model.load_4rc_weights(payload)
    step = int(payload['step']); del payload
    model.to(device).eval()
    report = dict(stage='running', checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_step=step,
        plan_sha256=plan_sha, model_config=json.loads(json.dumps(cfg.to_dict())), amp_dtype='bfloat16' if device.type == 'cuda' else 'float32',
        protocol=dict(dataset=args.dataset, single_camera=True, long_side=args.long_side,
            eval_grid=[args.eval_size]*2, input_resize='aspect_preserving_then_bottom_right_patch_padding',
            depth='D4RT per-frame median scale / median-centred scale+shift; all valid pixels',
            gt_depth_max=70 if args.dataset == 'sintel' else 10, pose='camera_head; legacy D4RT Sim3 ATE/RPE and adjacent-pair AUC@30',
            pointcloud='reference camera; one clip depth scale + mean shift; separate Sim3 result',
            confidence_filter=False, intrinsics_input=False, sparse_queries=False,
            legacy_input_resolution_differs=True, legacy_pose_estimator_differs=True,
            legacy_mean_shift_only_uses_a_different_scale_gauge=True), records=[], failed_cases=[])
    audit_path = args.output/'gt_audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit['passed'] or audit['plan_sha256'] != plan_sha:
        raise ValueError('requires passed GT audit for the exact evaluation plan')
    audit_rows = {r['scene']: r for r in audit['cases']}
    path = args.output/'evaluation.json'
    if args.resume_results and path.exists():
        old = json.loads(path.read_text())
        if any(old[k] != report[k] for k in ('checkpoint', 'plan_sha256', 'model_config', 'protocol')):
            raise ValueError('cannot resume a different checkpoint or benchmark protocol')
        report['records'] = old['records']
    done = {r['scene'] for r in report['records']}
    write_json(path, report)
    for case in plan['cases']:
        if case['scene'] in done: continue
        began = time.monotonic()
        try:
            clip = load_clip(case, eval_size=args.eval_size, long_side=args.long_side, patch_size=cfg.patch_size)
            if clip.calibration_sha256 != audit_rows[case['scene']]['calibration_sha256']:
                raise ValueError('GT calibration/depth changed after audit')
            images = clip.images.to(device)
            cameras = torch.zeros(images.shape[:2], device=device, dtype=torch.long)
            times = torch.tensor(clip.frame_indices, device=device, dtype=torch.long)[None]
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                encoded = model.encode(images, cameras, times)
            prediction = geometry_on_eval_grid(encoded.geometry, clip)
            peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
            del encoded, images, cameras, times
            if device.type == 'cuda': torch.cuda.empty_cache()
            metrics = geometry_metrics(*prediction, clip, dataset=args.dataset)
            row = dict(scene=case['scene'], root=case['root'], frame_indices=clip.frame_indices,
                actual_times=len(clip.frame_indices), input_shape=list(clip.images.shape), raw_hw=list(clip.raw_hw),
                content_hw=list(clip.content_hw), metrics=metrics, seconds=time.monotonic()-began, peak_cuda_bytes=peak)
            report['records'].append(row)
            print(json.dumps(dict(completed=len(report['records']), **row)), flush=True)
            del clip, prediction
        except Exception:
            report['failed_cases'].append(dict(scene=case['scene'], error=traceback.format_exc()))
            print(json.dumps(report['failed_cases'][-1]), flush=True)
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        write_json(path, report)
    rows = report['records']
    report['mean_metrics'] = {k:sum(r['metrics'][k] for r in rows)/len(rows) for k in rows[0]['metrics']} if rows else {}
    report['num_requested_scenes'], report['num_scenes'] = len(plan['cases']), len(rows)
    report['stage'] = 'passed' if not report['failed_cases'] and len(rows) == len(plan['cases']) else 'failed'
    write_json(path, report)
    if report['stage'] != 'passed': raise RuntimeError('evaluation has failed/missing scenes; see evaluation.json')
    return report


if __name__ == '__main__': main()
