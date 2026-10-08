"""Collect completed benchmark results without touching training state."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path('/data8/zbf/multicamera4rc_runs/monocular_benchmark_20261008')
RUNTIME = Path('/data9/zbf/my_dfrt_monocular_eval_20261008')
PRODUCTION = Path('/data9/zbf/my_dfrt_multicamera4rc')


def read(path):
    return json.loads(path.read_text())


def command(args, cwd=None):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return p.stdout.strip()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


selection = read(ROOT / 'checkpoint_selection.json')
split = read(ROOT / 'scannet_split_proof.json')
tests = read(ROOT / 'unit_tests_cuda_fix.json')
assert tests['passed'] and split['passed'] and not split['physical_scene_overlap']
code_commit = command(['git', 'rev-parse', 'HEAD'], RUNTIME)
assert code_commit == selection['code_commit'] == tests['commit']
assert not command(['git', 'status', '--porcelain'], RUNTIME)
datasets = {}
all_rows = []
for dataset, expected in [('scannet', 100), ('sintel', 14)]:
    directory = ROOT / dataset
    report = read(directory / 'evaluation.json')
    audit = read(directory / 'gt_audit.json')
    plan = read(directory / 'fixed_plan.json')
    assert report['stage'] == 'passed', (dataset, report['stage'])
    assert report['num_scenes'] == report['num_requested_scenes'] == expected
    assert len(report['records']) == len(plan['cases']) == len(audit['cases']) == expected
    assert not report['failed_cases'] and not audit['failed_cases'] and audit['passed']
    assert report['checkpoint_step'] == selection['step'] == 55500
    assert report['checkpoint'] == selection['checkpoint']
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    assert plan_hash == audit['plan_sha256'] == report['plan_sha256']
    assert report['protocol']['single_camera'] and not report['protocol']['intrinsics_input']
    assert not report['protocol']['confidence_filter'] and not report['protocol']['sparse_queries']
    cases = {c['scene']: c for c in plan['cases']}
    audits = {c['scene']: c for c in audit['cases']}
    scenes = [c['scene'] for c in report['records']]
    assert len(scenes) == len(set(scenes)) and set(scenes) == set(cases) == set(audits)
    if dataset == 'scannet':
        assert set(scenes) == set(split['names'])
    for row in report['records']:
        name = row['scene']
        assert row['frame_indices'] == cases[name]['frame_indices'] == audits[name]['frame_indices']
        assert row['frame_indices'] == sorted(set(row['frame_indices']))
        assert row['actual_times'] == len(row['frame_indices']) >= 2
        assert row['input_shape'][:3] == [1, row['actual_times'], 3]
        assert all(n % report['model_config']['patch_size'] == 0 for n in row['input_shape'][-2:])
        assert row['metrics']['depth_num_valid_pixels'] == row['metrics']['pointcloud_num_points'] > 0
        assert all(math.isfinite(v) for v in row['metrics'].values())
        assert row['metrics']['pose_num_pairs'] == row['actual_times'] - 1
        all_rows.append(dict(dataset=dataset, scene=name, frames=row['actual_times'],
                             input_height=row['input_shape'][-2], input_width=row['input_shape'][-1],
                             seconds=row['seconds'], peak_cuda_bytes=row['peak_cuda_bytes'], **row['metrics']))
    for key, value in report['mean_metrics'].items():
        mean = statistics.mean(row['metrics'][key] for row in report['records'])
        assert math.isclose(value, mean, rel_tol=1e-12, abs_tol=1e-12), (dataset, key)
    datasets[dataset] = dict(
        passed=True, scenes=expected, failed_scenes=0,
        actual_frames=sum(row['actual_times'] for row in report['records']),
        min_times=min(row['actual_times'] for row in report['records']),
        max_times=max(row['actual_times'] for row in report['records']),
        total_valid_depth_pixels=sum(row['metrics']['depth_num_valid_pixels'] for row in report['records']),
        total_adjacent_pose_pairs=sum(row['metrics']['pose_num_pairs'] for row in report['records']),
        input_shapes=sorted({tuple(row['input_shape'][-2:]) for row in report['records']}),
        total_case_seconds=sum(row['seconds'] for row in report['records']),
        peak_cuda_gib=max(row['peak_cuda_bytes'] for row in report['records']) / 2**30,
        mean_metrics=report['mean_metrics'],
        median_metrics={k: statistics.median(row['metrics'][k] for row in report['records'])
                        for k in report['mean_metrics']},
        plan_sha256=plan_hash,
        report_finished_at=datetime.datetime.fromtimestamp((directory / 'evaluation.json').stat().st_mtime,
                                                           datetime.timezone(datetime.timedelta(hours=8))).isoformat())

with (ROOT / 'per_scene.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(all_rows)

# Keep the original checkpoint source fixed; do not follow subsequent best-checkpoint updates.
checkpoint = Path(selection['checkpoint'])
checkpoint_stat = checkpoint.stat()
gpu_state = command(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                     '--format=csv,noheader,nounits'])
gpu_processes = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                         '--format=csv,noheader,nounits'])
source_files = ['scripts/evaluate_monocular.py', 'scripts/monocular_protocol.py',
                'test/eval_scannet_checkpoint.py', 'test/eval_sintel_checkpoint.py',
                'utils/metrics.py', 'utils/camera.py']
summary = dict(
    passed=True, collected_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    checkpoint_step=selection['step'], checkpoint=str(checkpoint),
    checkpoint_size_bytes=checkpoint_stat.st_size, checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns,
    evaluation_commit=code_commit, production_commit=command(['git', 'rev-parse', 'HEAD'], PRODUCTION),
    runtime_worktree=str(RUNTIME), output_root=str(ROOT), physical_evaluation_gpu=0,
    source_sha256={name: hashlib.sha256((RUNTIME / name).read_bytes()).hexdigest() for name in source_files},
    total_scenes=len(all_rows), failed_scenes=0, tests_passed=tests['passed'],
    scannet_train_test_physical_overlap=split['physical_scene_overlap'],
    gpu_state_at_collection=gpu_state, gpu_processes_at_collection=gpu_processes,
    aggregation='equal scene weights; full valid evaluation-grid pixels within each clip',
    limitations=['aspect-preserving 518-long-side model input differs from legacy stretched 256 input',
                 'current camera head differs from D4RT query-based camera estimation',
                 'clip-level Sim3 pointcloud metric is scale aligned, not absolute metric reconstruction',
                 'geometry-only evaluation: motion decoder, scene flow and tracking are not scored'],
    datasets=datasets)
write(ROOT / 'summary.json', summary)
write(ROOT / 'scannet_status.json', dict(stage='passed', passed=True, physical_gpu=0,
                                       checkpoint_step=selection['step'], completed_at=summary['collected_at'],
                                       report=str(ROOT / 'scannet/evaluation.json')))
print(json.dumps(dict(passed=True, scenes=len(all_rows), checkpoint_step=selection['step'],
                      datasets={k: v['mean_metrics'] for k, v in datasets.items()}), indent=2))
