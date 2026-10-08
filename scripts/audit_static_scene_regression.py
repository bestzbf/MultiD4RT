"""Compare matched static-validation cases without changing the validation set.

Sequence means give each scene identifier equal weight (one identifier per
physical scene in BlendedMVS). Leave-one-scene-out results are sensitivity
diagnostics, never an automatic exclusion list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_evaluation(path, label, source):
    raw = Path(path).read_bytes()
    report = json.loads(raw)
    if report.get('stage') != 'passed':
        raise ValueError(f'incomplete evaluation: {path}')
    checkpoints = report['checkpoints']
    if label in checkpoints:
        checkpoint = checkpoints[label]
    elif len(checkpoints) == 1:
        checkpoint = next(iter(checkpoints.values()))
    else:
        raise ValueError(f'checkpoint label {label!r} missing in {path}')
    records = {}
    for row in checkpoint['records']:
        if row['source'] != source:
            continue
        key = (row['sequence'], row['actual_cameras'], tuple(row['view_indices']))
        if key in records:
            raise ValueError(f'duplicate validation case: {key}')
        records[key] = row['metrics']
    if not records:
        raise ValueError(f'no {source!r} cases in {path}')
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                checkpoint=checkpoint['path'], step=checkpoint['step'],
                metrics_version=report['metrics_version'],
                plan_sha256=report['plan_sha256'], records=records,
                model_config=checkpoint.get('model_config'),
                amp_dtype=checkpoint.get('amp_dtype'))


def compare(before, after, metric):
    """Use matched camera sets; report both clip and scene-identifier aggregates."""
    if set(before) != set(after):
        raise ValueError('cannot compare different validation cases')
    if not before:
        raise ValueError('cannot compare empty evaluations')
    cases = []
    for (scene, cameras, views), values in before.items():
        key = (scene, cameras, views)
        a, b = float(values[metric]), float(after[key][metric])
        if not all(math.isfinite(v) and v >= 0 for v in (a, b)):
            raise ValueError(f'invalid {metric}: {key}')
        cases.append(dict(scene=scene, cameras=cameras, views=list(views),
                          before=a, after=b, delta=b-a,
                          ratio=b/a if a else None))
    scenes = []
    for scene in sorted({r['scene'] for r in cases}):
        rows = [r for r in cases if r['scene'] == scene]
        a = statistics.mean(r['before'] for r in rows)
        b = statistics.mean(r['after'] for r in rows)
        scenes.append(dict(scene=scene, clips=len(rows), before=a, after=b,
                           delta=b-a, ratio=b/a if a else None))
    total_delta = sum(r['delta'] for r in scenes)
    for row in scenes:
        row['signed_share_of_scene_mean_delta'] = row['delta']/total_delta if total_delta else None
    omitted = []
    if len(scenes) > 1:
        for scene in scenes:
            remaining = [r for r in scenes if r['scene'] != scene['scene']]
            a = statistics.mean(r['before'] for r in remaining)
            b = statistics.mean(r['after'] for r in remaining)
            omitted.append(dict(omitted_scene=scene['scene'], remaining_scenes=len(remaining),
                                before=a, after=b, ratio=b/a if a else None))
    a = statistics.mean(r['before'] for r in scenes)
    b = statistics.mean(r['after'] for r in scenes)
    ratios = [r['ratio'] for r in scenes]
    positive_ratios = all(v is not None and v > 0 for v in ratios)
    return dict(metric=metric, scene_count=len(scenes), clip_count=len(cases),
                scene_mean_before=a, scene_mean_after=b,
                scene_mean_ratio=b/a if a else None,
                clip_mean_before=statistics.mean(r['before'] for r in cases),
                clip_mean_after=statistics.mean(r['after'] for r in cases),
                scene_median_before=statistics.median(r['before'] for r in scenes),
                scene_median_after=statistics.median(r['after'] for r in scenes),
                scene_ratio_median=statistics.median(ratios) if positive_ratios else None,
                scene_ratio_geomean=math.exp(statistics.mean(math.log(v) for v in ratios)) if positive_ratios else None,
                improved_scenes=sum(r['delta'] < 0 for r in scenes),
                worsened_scenes=sum(r['delta'] > 0 for r in scenes),
                scenes=sorted(scenes, key=lambda r: r['delta'], reverse=True),
                cases=cases, leave_one_scene_out=omitted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', nargs='+', required=True, metavar='LABEL=REPORT')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--source', default='blended')
    parser.add_argument('--metrics', nargs='+', default=['depth_abs_rel', 'base_epe_normalized'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluations = {}
    for entry in args.evaluations:
        label, path = entry.split('=', 1)
        if label in evaluations:
            raise ValueError(f'duplicate label: {label}')
        evaluations[label] = read_evaluation(path, label, args.source)
    reference = evaluations[args.reference]
    comparisons = {}
    for label, candidate in evaluations.items():
        if label == args.reference:
            continue
        for field in ('metrics_version', 'plan_sha256', 'amp_dtype', 'model_config'):
            if reference[field] != candidate[field]:
                raise ValueError(f'unmatched {field}: {label}')
        comparisons[label] = {metric: compare(reference['records'], candidate['records'], metric)
                              for metric in args.metrics}
    if not comparisons:
        raise ValueError('at least one candidate evaluation is required')
    report = dict(stage='completed', source=args.source, reference=args.reference,
                  exclusion_policy='diagnostic_only_no_cases_removed',
                  scene_unit='sequence identifier from physically held-out static manifests',
                  evaluations={k: {f: v for f, v in row.items() if f != 'records'}
                               for k, row in evaluations.items()},
                  comparisons=comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), reference=args.reference,
                         candidates=list(comparisons))))


if __name__ == '__main__':
    main()
