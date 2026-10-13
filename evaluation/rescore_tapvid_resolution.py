"""Fix official short-side-256 thresholds, preserving all model predictions.

Archive every old report before changing any score. Verify unchanged inference
functions, original RGB/GT archives, supplied query identities, and official
caller resolution math. Scores are recomputed on all clips of both methods.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .common import digest,fingerprint,load_module,write_json
from .launch_tapvid import collect
from .run_tapvid import metric_intrinsics,score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args();base=Path(__file__).parent
    root=args.campaign/'tapvid16_pstudio_full_eval'
    legacy=root/'legacy_raw_intrinsics'
    if legacy.exists():raise ValueError('Migration already started; inspect the archived reports before retrying')
    old_plan=json.loads((root/'plan.json').read_text())
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old=functions(base/'references/run_tapvid_raw_intrinsics.py');new=functions(base/'run_tapvid.py')
    if any(old[n]!=new[n] for n in ['predict','sample']):raise AssertionError('Model inference changed')
    # Run the unchanged official caller's resolution function, extracted without
    # its optional CLI dependencies. No scoring implementation is rewritten.
    caller=base/'references/tapvid3d_evaluate_model.py'
    caller_tree=ast.parse(caller.read_text())
    fn=next(n for n in caller_tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_new_hw_with_given_smallest_side_length')
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(caller),'exec'),env)
    verified=[]
    for h,w in [(360,640),(1080,1920),(1024,512),(256,256)]:
        _,factor=env[fn.name](orig_height=h,orig_width=w,smallest_side_length=256)
        np.testing.assert_array_equal(metric_intrinsics(np.array([510.,515.,320.,180.]),[h,w]),np.array([510.,515.,320.,180.])*factor)
        verified.append(dict(raw_hw=[h,w],scale=float(factor)))
    legacy.mkdir()
    snapshots=[*root.glob('*.json')]
    for method in ['multid4rt','omnix']:
        snapshots.extend((root/method).glob('results.json'))
        snapshots.extend((root/method).glob('worker*/results.json'))
    for src in snapshots:
        dst=legacy/src.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    plan=dict(old_plan,schema='omnix16-first-visible-v2',metric_eval_short_side=256,
        scoring='official depth-adaptive APD3D; scoring intrinsics at short side 256; median scale on GT-visible camera-coordinate norms')
    write_json(root/'plan.json',plan)
    official=load_module('_tapvid_rescore_resolution',base/'references/tapvid3d_metrics.py')
    constant=np.zeros((16,8,3),dtype=np.float32);constant[...,2]=3
    offset=constant.copy();offset[...,0]=.008;vis=np.ones((16,8),dtype=bool)
    regression=score(official,offset,constant,vis,np.array([500.,500.,320.,180.]),[360,640])
    if regression['apd3d']!=1.:raise AssertionError('Nonzero-error resolution regression failed')
    cases={c['id']:c for c in plan['cases']};ground={};prediction_evidence=[]
    for method in ['multid4rt','omnix']:
        for worker in sorted((root/method).glob('worker*')):
            path=worker/'results.json'
            if not path.exists():continue
            report=json.loads(path.read_text())
            if report['stage']!='partition_complete' or report['failures']:raise AssertionError('Incomplete inference')
            if report['contract']['implementation']['run_tapvid.py']!=digest(base/'references/run_tapvid_raw_intrinsics.py'):
                raise AssertionError('Unexpected legacy execution source')
            for row in report['records']:
                case=cases[row['id']];gt_path=case['path']
                if gt_path not in ground:
                    if digest(gt_path)!=case['sha256']:raise AssertionError('GT/RGB archive changed')
                    with np.load(gt_path,allow_pickle=True) as data:
                        first=Image.open(io.BytesIO(bytes(data['images_jpeg_bytes'][0])))
                        hw=[first.height,first.width]
                        ground[gt_path]=(data['tracks_XYZ'].astype(np.float32),data['visibility'].astype(bool),data['fx_fy_cx_cy'].astype(np.float64),hw)
                tracks,visibility,k,hw=ground[gt_path]
                if hw!=row['input']['raw_hw']:raise AssertionError('Recorded RGB resolution disagrees with official archive')
                file=worker/(row['id']+'.npz');checksum=digest(file)
                with np.load(file,allow_pickle=False) as data:
                    pred=data['tracks_XYZ'].copy()
                    np.testing.assert_array_equal(data['point_ids'],case['point_ids'])
                    if int(data['start'])!=case['start']:raise AssertionError('Wrong query time')
                s=case['start'];ids=case['point_ids'];truth=tracks[s:s+16,ids];visible=visibility[s:s+16,ids]
                old_metrics=row['metrics'];row['metrics']=score(official,pred,truth,visible,k,hw)
                for key in ['epe_m','global_scale']:
                    np.testing.assert_allclose(row['metrics'][key],old_metrics[key],rtol=0,atol=1e-12)
                if row['metrics']['apd3d']+1e-12<old_metrics['apd3d']:raise AssertionError('Looser official thresholds unexpectedly decrease APD')
                prediction_evidence.append(dict(method=method,id=row['id'],sha256=checksum,legacy_apd3d=old_metrics['apd3d'],corrected_apd3d=row['metrics']['apd3d']))
            contract=report['contract'];contract.update(plan_sha256=fingerprint(plan),metric_eval_short_side=256,official_caller_sha256=digest(caller))
            contract['implementation']['run_tapvid.py']=digest(base/'run_tapvid.py')
            report['resolution_scoring_migration']=dict(legacy_report=str(legacy/path.relative_to(root)),new_model_forwards=0,
                unchanged_prediction_functions=True,reason='Missing official short-side-256 camera scaling')
            write_json(path,report)
        print(method,'rescored',flush=True)
    outputs={m:collect(plan,root,m,4) for m in ['multid4rt','omnix']}
    inputs=[{r['id']:r['input'] for r in outputs[m]['records']} for m in outputs]
    if inputs[0]!=inputs[1]:raise AssertionError('Different RGB between methods')
    comparison=dict(stage='passed',protocol=plan['schema'],subset='pstudio',split='full_eval',clips=len(cases),sequences=len(outputs['multid4rt']['by_sequence']),
        plan_sha256=fingerprint(plan),identical_rgb=True,metric_eval_short_side=256,methods={m:r['mean_metrics'] for m,r in outputs.items()})
    write_json(root/'comparison.json',comparison)
    progress_path=root/'progress.json'
    progress=json.loads(progress_path.read_text())
    progress.update(stage='passed',plan_sha256=fingerprint(plan),metric_eval_short_side=256,
                    scoring_correction='Missing official short-side-256 intrinsic scaling fixed; original prediction arrays reused')
    write_json(progress_path,progress)
    proof=dict(stage='passed',reason='Official evaluate_model.py rescales intrinsics by 256 / min(original height,width); previous adapter omitted this',
        official_caller_sha256=digest(caller),official_scaling_checks=verified,unchanged_inference_ast=True,new_model_forwards=0,
        nonzero_error_resolution_regression=regression,
        clips_per_method=len(cases),sequences=106,legacy_reports=str(legacy),predictions=prediction_evidence)
    write_json(root/'resolution_correction.json',proof)
    # Re-score captured diagnostic forwards with the same correction, no reruns.
    diag=args.campaign/'pstudio_diagnosis_20261013'
    for method in ['60k','67k','omnix']:
        path=diag/method/'capture_results.json';shutil.copy2(path,path.with_name('capture_results_raw_intrinsics.json'))
        report=json.loads(path.read_text())
        for row in report['records']:
            case=cases[row['id']];tracks,visibility,k,hw=ground[case['path']];s=case['start'];ids=case['point_ids']
            with np.load(diag/method/f'{row["id"]}_{row["variant"]}.npz',allow_pickle=False) as data:pred=data['tracks_XYZ']
            row['metrics']=score(official,pred,tracks[s:s+16,ids],visibility[s:s+16,ids],k,hw)
        report['metric_eval_short_side']=256;write_json(path,report)
    print(json.dumps(comparison),flush=True)


if __name__=='__main__':main()
