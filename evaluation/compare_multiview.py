"""Admit only complete paired test runs with identical RGB/query/scorer hashes."""
import argparse
import json
from pathlib import Path

from .common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args()
    reports={m:json.loads((args.directory/m/'score/results.json').read_text()) for m in ['multid4rt','omnix']}
    left,right=reports.values()
    if any(not r['complete'] or r['result_kind']!='full_benchmark' or r['missing'] for r in reports.values()):
        raise ValueError('Partial runs cannot enter the full comparison')
    if left['plan_sha256']!=right['plan_sha256'] or left['official']!=right['official']:
        raise ValueError('Different official queries or scorer')
    if {r['id'] for r in left['per_sequence']}!={r['id'] for r in right['per_sequence']}:
        raise ValueError('Different test sequences')
    rows=[]
    for case in left['per_sequence']:
        relative=Path(case['profile'])/(case['sequence']+'.diagnostics.json')
        a,b=[json.loads((args.directory/m/relative).read_text()) for m in reports]
        if a['preprocessing']!=b['preprocessing'] or a['queries']!=b['queries'] or a['frames']!=b['frames']:
            raise ValueError(f'Input mismatch: {case["id"]}')
    for name,summary in left['summaries'].items():
        group='any' if name=='panoptic' else 'dynamic-static-mean'
        for method,report in reports.items():
            metric=report['summaries'][name]['metrics']
            def get(key): return metric[f'3dpt/model__{key}__{group}']
            rows.append(dict(dataset=name,method=method,sequences=summary['completed'],point_group=group,
                delta_avg_percent=get('average_pts_within_thresh'),mte=get('mte_visible'),ate=get('ate_visible'),
                fde=get('fde_visible'),survival_percent=get('survival'),
                aj3d_percent=get('average_jaccard') if method=='multid4rt' else None,
                oa_percent=get('occlusion_accuracy') if method=='multid4rt' else None,
                units='centimeters' if name!='mv-kubric' else 'original scene units *100'))
    migration=args.directory/'omnix/cache_migration.json'
    write_json(args.directory/'comparison.json',dict(stage='passed',plan_sha256=left['plan_sha256'],
        official=left['official'],identical_rgb=True,sequences=len(left['per_sequence']),rows=rows,
        input_condition=left['run']['input_condition'],
        note='Provided calibration registers a single camera-center Sim(3); supplied 3D query XYZ anchors the trajectory. RGB networks receive no camera matrices or future tracks.',
        omnix_visibility='no official occlusion head; AJ/OA are unsupported, not copied from always-visible diagnostic scoring',
        equivalent_cache_reuse=json.loads(migration.read_text())['reused_scenes'] if migration.exists() else 0))


if __name__=='__main__': main()
