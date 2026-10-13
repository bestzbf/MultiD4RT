"""Released checkpoints on immutable official MVTracker test queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import digest, load_model, write_json
from .multiview_adapter import TrackingInputs, predict_official_queries
from .mvtracker.official import OfficialMVTracker
from .mvtracker.plan import (cache_for_scene, prediction_path, read_plan, read_prediction,
                             save_prediction, score_predictions, write_report)
from .mvtracker.protocols import PROTOCOLS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['multid4rt', 'omnix'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/data6/benchmarks'))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--omnix-runner', type=Path)
    p.add_argument('--parity-report', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--long-side', type=int, default=266)
    p.add_argument('--target-chunk', type=int, default=32)
    p.add_argument('--source-chunk', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-scenes', type=int)
    args = p.parse_args()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    official = OfficialMVTracker(args.root / 'code/mvtracker')
    plan = read_plan(args.plan, args.root)
    model, report = load_model(args.method, args.checkpoint, device, args.omnix_runner)
    if args.method == 'omnix':
        from .omnix_queries import OmniXQueries
        if not args.parity_report:
            raise ValueError('Actual native release-output parity report is required')
        parity=json.loads(args.parity_report.read_text())
        if parity.get('stage')!='passed' or parity['checkpoint_sha256']!=report['checkpoint_sha256'] or parity['adapter_sha256']!=digest(Path(__file__).with_name('omnix_queries.py')):
            raise ValueError('Parity report belongs to another checkpoint or adapter')
        model = OmniXQueries(model)
        report['query_memory_adapter'] = model.provenance
        if any(case['adapter_provenance']!=model.provenance for case in parity['cases']):
            raise ValueError('OmniX source changed since the native-output parity check')
        report['parity_report_sha256']=digest(args.parity_report)
    code_paths = [Path(__file__), Path(__file__).with_name('multiview_adapter.py')]
    if args.method == 'omnix':
        code_paths.append(Path(__file__).with_name('omnix_queries.py'))
    run = dict(method=args.method, checkpoint_sha256=report['checkpoint_sha256'],
               input_condition='rgb_calibrated_gauge_and_supplied_3d_queries',
               network_inputs='RGB plus camera/time IDs; no GT depth, poses, visibility or future tracks',
               alignment='one camera-center Sim(3) per complete sequence, using provided calibration',
               query_adapter='closest predicted source anchor among calibrated in-frame projections',
               query_anchor='constant offset from supplied query XYZ; no future track fitting',
               visibility='native head >0.5 OR across input views' if args.method=='multid4rt'
                          else 'no occlusion predictor; all-visible storage placeholder; AJ/OA unsupported',
               long_side=args.long_side, resize='full-FOV aspect preserving, bilinear align_corners=True, bottom/right zero padding to patch14',
               target_chunk=args.target_chunk, source_chunk=args.source_chunk,
               amp='bfloat16; regression float32', load_report=report,
               code_sha256={str(f.relative_to(Path(__file__).parent)):digest(f) for f in code_paths})
    run = json.loads(json.dumps(run))
    run_path = args.output/'run.json'
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError('Existing run belongs to different weights, inputs or code')
    write_json(run_path, run)
    processed = 0
    for name in ('dexycb','mv-kubric','panoptic'):
        if name not in plan['profiles']:
            continue
        protocol = PROTOCOLS[name]
        dataset = official.dataset(protocol, args.root/'datasets/mvtracker')
        scenes = [s for s in plan['scenes'] if s['profile']==name]
        if dataset.seq_names != [s['sequence'] for s in scenes]:
            raise ValueError('Official loader scene order differs from frozen plan')
        for index, scene in enumerate(scenes):
            destination = prediction_path(args.output, scene)
            if destination.exists():
                if not args.resume:
                    raise FileExistsError(destination)
                cache_for_scene(plan, scene)
                read_prediction(destination, plan, scene, run)
                continue
            if args.max_scenes is not None and processed>=args.max_scenes:
                break
            started = time.monotonic()
            cache_for_scene(plan, scene)
            datapoint, valid = dataset[index]
            if not valid:
                raise ValueError(f'Official loader rejected {scene["id"]}')
            labels = official.cached_labels(cache_for_scene(plan,scene),protocol)
            if not np.array_equal(datapoint.query_points_3d.numpy(),labels['query_points']):
                raise ValueError('Reader queries do not match immutable official cache')
            inputs = TrackingInputs.from_official(datapoint,protocol)
            del datapoint, labels
            print(f'Inferring {scene["id"]}: C{len(protocol.views)} T{scene["frames"]} Q{scene["queries"]}',flush=True)
            tracks, visible, diagnostics = predict_official_queries(model,inputs,device,
                args.long_side,args.target_chunk,torch.bfloat16,args.source_chunk)
            save_prediction(destination,plan,scene,tracks,visible,run)
            diagnostics['total_seconds']=time.monotonic()-started
            write_json(destination.with_suffix('.diagnostics.json'),diagnostics)
            report = score_predictions(plan,official,args.output,allow_partial=True)
            write_report(args.output/'score',report)
            write_json(args.output/'progress.json',dict(stage='running',done=len(report['per_sequence']),
                total=len(plan['scenes']),last_scene=scene['id'],last_seconds=diagnostics['total_seconds']))
            processed+=1
            del inputs
            if device.type=='cuda': torch.cuda.empty_cache()
    report = score_predictions(plan,official,args.output,allow_partial=args.max_scenes is not None)
    write_report(args.output/'score',report)
    write_json(args.output/'progress.json',dict(stage='passed' if report['complete'] else 'partial',
        done=len(report['per_sequence']),total=len(plan['scenes'])))
    print(json.dumps(dict(complete=report['complete'],summaries=report['summaries'])),flush=True)


if __name__=='__main__':
    main()
