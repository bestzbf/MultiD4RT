"""Paired Sintel reports and alignment-free rotation RPE on all saved clips."""
import argparse
import json
from pathlib import Path

import numpy as np
from evo.core import metrics,trajectory

from .common import digest, fingerprint, read_sintel_camera,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args()
    plan=json.loads((args.directory/'plan.json').read_text())
    reports={m:json.loads((args.directory/m/'results.json').read_text()) for m in ['multid4rt','omnix']}
    if any(r['stage']!='passed' or r['failures'] or len(r['records'])!=len(plan['cases']) for r in reports.values()):
        raise ValueError('Both checkpoints must have a complete, failure-free run')
    left,right=reports.values()
    if {r['id']:r['input'] for r in left['records']}!={r['id']:r['input'] for r in right['records']}:
        raise ValueError('Models used different RGB')
    if left['contract']['depth_scorer_sha256']!=right['contract']['depth_scorer_sha256']:
        raise ValueError('Models used different scorers')
    if left['undefined_pose_cases']!=right['undefined_pose_cases']:
        raise ValueError('Different fixed-GT pose exclusions')
    supplement={}
    for method,report in reports.items():
        by_id={r['id']:r for r in report['records']}
        rows=[]
        for case in plan['cases']:
            truth=np.stack([read_sintel_camera(case['files'][i]['path'])[1] for i in range(2,len(case['files']),3)])
            saved=args.directory/method/(case['id']+'.npz')
            with np.load(saved,allow_pickle=False) as d: pose=d['c2w'].astype(np.float64)
            # SDK camera rotations are rounded (~2.5e-6 orthogonality error).
            # Convert both rotation inputs to proper SO(3), as a quaternion
            # pose-file reader would, before evo's strict Lie group check.
            corrections=[]
            for matrices in [truth,pose]:
                u,_,vh=np.linalg.svd(matrices[:,:3,:3])
                proper=u@vh
                correction=np.linalg.norm(proper-matrices[:,:3,:3],axis=(1,2)).max()
                if correction>2e-5 or (np.linalg.det(proper)<=0).any():
                    raise ValueError('Rotations require more than numerical SO(3) cleanup')
                corrections.append(float(correction))
                matrices[:,:3,:3]=proper
            gt=trajectory.PosePath3D(poses_se3=truth.astype(np.float64))
            pred=trajectory.PosePath3D(poses_se3=pose)
            rr=metrics.RPE(metrics.PoseRelation.rotation_angle_deg,delta=1,
                           delta_unit=metrics.Unit.frames,all_pairs=True)
            rr.process_data((gt,pred))
            value=float(rr.get_statistic(metrics.StatisticsType.rmse))
            old=by_id[case['id']]['metrics'].get('rpe_rotation_rmse_deg')
            if old is not None and not np.isclose(value,old,rtol=1e-5,atol=1e-4):
                raise ValueError('Rotation RPE should be invariant to world Sim(3) alignment')
            rows.append(dict(id=case['id'],scene=case['scene'],rpe_rotation_rmse_deg=value,
                             numerical_so3_correction_frobenius_gt_pred=corrections,
                             saved_prediction_sha256=digest(saved)))
        numeric=[k for k in report['mean_metrics'] if k.startswith('depth_')]
        scenes=sorted({r['scene'] for r in report['records']})
        per_scene={scene:{key:float(np.mean([r['metrics'][key] for r in report['records'] if r['scene']==scene]))
                         for key in numeric} for scene in scenes}
        supplement[method]=dict(all37_rotation_rpe_rmse_deg=float(np.mean([r['rpe_rotation_rmse_deg'] for r in rows])),
            rotation_cases=len(rows),rotation_per_clip=rows,
            equal_scene_depth_mean={key:float(np.mean([v[key] for v in per_scene.values()])) for key in numeric},
            depth_per_scene=per_scene)
    write_json(args.directory/'comparison.json',dict(stage='passed',plan_sha256=fingerprint(plan),
        identical_rgb=True,clips=len(plan['cases']),scenes=plan['scene_count'],
        methods={m:r['mean_metrics'] for m,r in reports.items()},supplement=supplement,
        primary_aggregation='equal 16-frame clip means; equal-scene depth means also disclosed',
        translation_pose_cases=len(plan['cases'])-len(left['undefined_pose_cases']),
        undefined_pose_cases=left['undefined_pose_cases'],
        pose_exclusion='fixed GT camera-center Sim(3) rank deficiency; identical exclusions for both methods; all clips retained for rotation RPE',
        supplemental_rotation_input='bounded numerical SO(3) projection of both rounded SDK GT and float32 model rotations; maximum correction disclosed per clip'))


if __name__=='__main__': main()
