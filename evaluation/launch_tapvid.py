"""Prepare one full official subset, run both releases, collect complete results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .common import fingerprint, write_json
from .run_tapvid import build_plan


def collect(plan, directory, method, partitions):
    records, contracts = [], []
    for index in range(partitions):
        report = json.loads((directory / method / f'worker{index}' / 'results.json').read_text())
        if report['stage'] != 'partition_complete' or report['failures']:
            raise ValueError(f'Incomplete partition: {method}/{index}')
        if report['contract']['plan_sha256'] != fingerprint(plan):
            raise ValueError('Partition used a different benchmark plan')
        if report['contract'].get('metric_eval_short_side')!=256:
            raise ValueError('Partition omitted the official scoring resolution')
        records.extend(report['records'])
        contracts.append(report['contract'])
    expected = {case['id'] for case in plan['cases']}
    if len(records) != len(expected) or {row['id'] for row in records} != expected:
        raise ValueError('Missing, extra, or duplicate benchmark clips')
    if len({c['loading']['checkpoint_sha256'] for c in contracts}) != 1:
        raise ValueError('Partitions used different weights')
    by_sequence = {}
    for sequence in sorted({row['sequence'] for row in records}):
        subset = [row for row in records if row['sequence'] == sequence]
        by_sequence[sequence] = {k:float(np.mean([r['metrics'][k] for r in subset])) for k in ['apd3d','epe_m']}
    report = dict(stage='passed', method=method, plan_sha256=fingerprint(plan),
                  clips=len(records), sequences=len(by_sequence), records=records,
                  contracts=contracts, by_sequence=by_sequence,
                  mean_metrics={k:float(np.mean([v[k] for v in by_sequence.values()])) for k in ['apd3d','epe_m']},
                  clip_mean_metrics={k:float(np.mean([r['metrics'][k] for r in records])) for k in ['apd3d','epe_m']})
    write_json(directory / method / 'results.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--subset', default='pstudio', choices=['pstudio','adt','drivetrack'])
    p.add_argument('--split', default='full_eval', choices=['full_eval','minival'])
    p.add_argument('--partitions', type=int, default=4)
    p.add_argument('--multid4rt-checkpoint', required=True)
    p.add_argument('--omnix-checkpoint', required=True)
    args = p.parse_args()
    root = args.campaign.resolve()
    code = Path(__file__).resolve().parents[1]
    manifest_path = code / 'evaluation/references/tapvid3d_sources.json'
    data_root = root / 'tapvid3d_data'
    ready_path = data_root / ('download_' + args.split + '.json')
    directory = root / ('tapvid16_' + args.subset + '_' + args.split)
    directory.mkdir(exist_ok=True)
    progress_path = directory / 'progress.json'
    status = dict(stage='waiting_for_data', subset=args.subset, split=args.split, jobs=[])
    write_json(progress_path, status)
    while True:
        if ready_path.exists():
            readiness = json.loads(ready_path.read_text())
            if readiness['stage'] == 'failed':
                raise RuntimeError('Official dataset preparation failed')
            if readiness['stage'] == 'passed':
                break
        time.sleep(5)
    manifest = json.loads(manifest_path.read_text())
    plan_path = directory / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = build_plan(manifest, data_root, args.subset, args.split, 504)
        write_json(plan_path, plan)
    # Score a synthetic identity/scaled prediction with the original scorer
    # before any GPU workers are admitted. This is explicitly a scorer check,
    # never a model accuracy result.
    from .common import load_module
    from .run_tapvid import score
    metrics_path = code / 'evaluation/references/tapvid3d_metrics.py'
    scorer = load_module('_tapvid3d_preflight', metrics_path)
    rng = np.random.default_rng(13)
    truth = rng.normal(size=(16, 8, 3)).astype(np.float32)
    truth[..., 2] = np.abs(truth[..., 2]) + 3
    visible = np.ones((16,8), dtype=bool)
    for factor in [1., .25]:
        result = score(scorer, truth * factor, truth, visible, np.array([256.,256.,128.,128.]), [256,256])
        if result['apd3d'] != 1. or result['epe_m'] > 1e-6:
            raise ValueError('Official scorer preflight failed')
    # An identity check cannot detect a missing resolution normalization.
    # This 8 mm lateral error is outside the raw 360p one-pixel threshold,
    # and inside the official short-side-256 one-pixel threshold.
    constant=np.zeros((16,8,3),dtype=np.float32);constant[...,2]=3
    offset=constant.copy();offset[...,0]=.008
    check=score(scorer,offset,constant,visible,np.array([500.,500.,320.,180.]),[360,640])
    if check['apd3d']!=1.:
        raise ValueError('Official evaluation-resolution regression check failed')
    status.update(stage='running', requested_cases=len(plan['cases']),
                  plan_sha256=fingerprint(plan), scorer_check='passed; not model inference')
    processes = []
    env = os.environ.copy()
    env.update(PYTHONPATH=str(code)+':'+str(root/'deps'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
               OMNIX_SOURCE=str(root/'official/OmniX'),
               DA3_SOURCE='/data9/zbf/omega_trace_dense_v10_multicam_omnix/third_party/depth-anything-3',
               OMNIX_DEPS='/data9/zbf/omega_trace_dense_v10_multicam_omnix/third_party/omnix_python_deps')
    fraction = .84 / args.partitions
    for method, gpu, checkpoint in [('multid4rt',0,args.multid4rt_checkpoint),('omnix',1,args.omnix_checkpoint)]:
        for index in range(args.partitions):
            output = directory / method / f'worker{index}'
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable,'-u','-m','evaluation.run_tapvid',
                '--method',method,'--checkpoint',checkpoint,'--root',str(data_root),
                '--manifest',str(manifest_path),'--official-metrics',str(metrics_path),
                '--subset',args.subset,'--split',args.split,'--plan',str(plan_path),
                '--output',str(output),'--device',f'cuda:{gpu}',
                '--partitions',str(args.partitions),'--partition-index',str(index),
                '--memory-fraction',str(fraction),'--resume']
            command.append('--migrate-scoring')
            if method == 'omnix':
                command.extend(['--omnix-runner',str(code/'evaluation/native_omnix.py')])
            with (output/'run.log').open('ab',buffering=0) as stream:
                process = subprocess.Popen(command,cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(process)
            status['jobs'].append(dict(method=method,partition=index,gpu=gpu,pid=process.pid,command=command,output=str(output)))
            write_json(progress_path,status)
    while True:
        status['worker_exit_codes'] = [process.poll() for process in processes]
        status['worker_completed'] = []
        for job in status['jobs']:
            report_path = Path(job['output'])/'results.json'
            if report_path.exists():
                report = json.loads(report_path.read_text())
                status['worker_completed'].append(dict(method=job['method'],partition=job['partition'],
                                                      done=len(report['records']),stage=report['stage']))
        write_json(progress_path,status)
        if all(value is not None for value in status['worker_exit_codes']):
            break
        time.sleep(5)
    if any(value != 0 for value in status['worker_exit_codes']):
        status['stage']='failed'
        write_json(progress_path,status)
        raise RuntimeError('A model partition failed; no full benchmark average was produced')
    outputs = {method:collect(plan,directory,method,args.partitions) for method in ['multid4rt','omnix']}
    left, right = outputs['multid4rt']['records'], outputs['omnix']['records']
    a,b = {r['id']:r['input'] for r in left},{r['id']:r['input'] for r in right}
    if a != b:
        raise ValueError('The two methods received different RGB inputs')
    comparison = dict(stage='passed', protocol=plan['schema'], subset=args.subset, split=args.split,
        metric_eval_short_side=256,
        clips=len(plan['cases']), sequences=len(outputs['multid4rt']['by_sequence']),
        plan_sha256=fingerprint(plan), identical_rgb=True,
        methods={method:report['mean_metrics'] for method,report in outputs.items()})
    write_json(directory/'comparison.json',comparison)
    status['stage']='passed'
    write_json(progress_path,status)
    print(json.dumps(comparison),flush=True)


if __name__ == '__main__':
    main()
