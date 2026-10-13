"""Verify captured heads, causal geometry controls, and immutable-cache parity."""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import digest, load_module, write_json
from .diagnose_pstudio import initial_diagnostics
from .run_tapvid import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args()
    root=args.campaign/'pstudio_diagnosis_20261013'
    cache=args.campaign/'tapvid16_pstudio_full_eval'
    plan=json.loads((cache/'plan.json').read_text())
    cases={c['id']:c for c in plan['cases']}
    official=load_module('_pstudio_finish_scorer',Path(__file__).parent/'references/tapvid3d_metrics.py')
    rows=[]; controls=[]; parity=[]
    for method in ['60k','67k','omnix']:
        report=json.loads((root/method/'capture_results.json').read_text())
        if report['stage']!='passed': raise ValueError('Incomplete capture')
        for row in report['records']:
            case=cases[row['id']];s=case['start'];ids=case['point_ids']
            with np.load(case['path'],allow_pickle=True) as data:
                truth=data['tracks_XYZ'][s:s+16,ids].astype(np.float32)
                visible=data['visibility'][s:s+16,ids].astype(bool)
                k=data['fx_fy_cx_cy'].astype(np.float64)
            # Verify original official GT is an exact identity, without filtering.
            identity=score(official,truth.copy(),truth,visible,k,row['input']['raw_hw'])
            if identity['apd3d']!=1 or identity['epe_m']!=0:
                raise AssertionError('Official scorer identity fails')
            with np.load(root/method/f'{row["id"]}_{row["variant"]}.npz',allow_pickle=False) as data:
                pred=data['tracks_XYZ'];K=data['K'];cam=data['c2w'];xy=data['query_xy']
            meta=row['input'];sc=np.array(meta['query_affine_scale']);off=np.array(meta['query_affine_offset'])
            transformed_k=np.r_[k[:2]*sc,k[2:]*sc+off]
            initial=initial_diagnostics(pred,truth,xy,transformed_k,row['metrics']['global_scale'])
            initial.update(method=method,id=row['id'],variant=row['variant'],metrics=row['metrics'],
                actual_head_focal_ratio=[float(K[0,0,0]/transformed_k[0]),float(K[0,1,1]/transformed_k[1])],
                actual_head_principal_error_px=[float(K[0,0,2]-transformed_k[2]),float(K[0,1,2]-transformed_k[3])])
            if method in ['60k','67k']:
                fitted=initial['fitted_source_intrinsics']
                actual=[K[0,0,0],K[0,1,1]]
                fit_error=np.max(np.abs(np.array([fitted['fx'],fitted['fy']])/actual-1))
                initial['fitted_to_head_focal_max_relative_error']=float(fit_error)
                if row['variant']=='standard' and fit_error>0.002:
                    raise AssertionError('Cached intrinsic regression disagrees with actual camera head')
                # Replace only the initial camera rays, keeping predicted depth,
                # world motion and all predicted target camera transforms.
                source=pred[0].astype(np.float64)
                corrected_source=truth[0].astype(np.float64)/truth[0,:,2:3]*source[:,2:3]
                world_shift=(corrected_source-source)@cam[0,:3,:3].T
                delta_local=np.einsum('tji,nj->tni',cam[:,:3,:3].astype(np.float64),world_shift)
                corrected=pred.astype(np.float64)+delta_local
                control=dict(method=method,id=row['id'],variant=row['variant'],
                    original=row['metrics'],oracle_source_K_same_official_scaling=score(official,corrected,truth,visible,k,row['input']['raw_hw']))
                controls.append(control)
            rows.append(initial)
            if method in ['60k','omnix'] and row['variant']=='standard':
                m='multid4rt' if method=='60k' else method
                files=list((cache/m).glob(f'worker*/{case["id"]}.npz'))
                if len(files)!=1:raise AssertionError('Invalid original cache')
                with np.load(files[0],allow_pickle=False) as data: original=data['tracks_XYZ']
                max_error=float(np.max(np.abs(pred-original)))
                # Capture applies its general affine in NumPy float64 before
                # casting query pixels. The benchmark casts before resizing;
                # high-gradient pixels differ by a few microunits. Verify both
                # the coordinates and the unchanged actual benchmark metrics.
                original_report=json.loads((cache/m/'results.json').read_text())
                original_row=next(r for r in original_report['records'] if r['id']==case['id'])
                apd_delta=abs(row['metrics']['apd3d']-original_row['metrics']['apd3d'])
                epe_delta=abs(row['metrics']['epe_m']-original_row['metrics']['epe_m'])
                if max_error>1e-5 or apd_delta!=0 or epe_delta>1e-6:
                    raise AssertionError('Capture differs materially from original benchmark')
                parity.append(dict(method=method,id=row['id'],max_xyz_error=max_error,
                    apd_delta=apd_delta,epe_delta_m=epe_delta,
                    note='Query affine float64-vs-float32 rounding; original benchmark cache unchanged'))
    groups=defaultdict(list)
    for row in rows:groups[(row['method'],row['variant'])].append(row)
    means={}
    for (method,variant),rs in groups.items():
        means[f'{method}/{variant}']=dict(clips=len(rs),
            apd3d=float(np.mean([r['metrics']['apd3d'] for r in rs])),
            epe_m=float(np.mean([r['metrics']['epe_m'] for r in rs])),
            focal_ratio=np.mean([r['actual_head_focal_ratio'] for r in rs],axis=0).tolist())
    write_json(root/'head_verification.json',dict(stage='passed',records=rows,means=means,
        identity_cases=len(rows),cache_parity=parity,geometry_controls=controls,
        note='Four fixed clips selected by indices 0,318,636,953 before looking at metrics. Not a full 67k benchmark. Oracle K controls are diagnostic GT interventions, not published scores.'))

    # Qualitative reprojection of the first preselected clip, in original pixels.
    case=plan['cases'][0];s=case['start'];ids=case['point_ids']
    with np.load(case['path'],allow_pickle=True) as data:
        rgb=Image.open(io.BytesIO(bytes(data['images_jpeg_bytes'][s]))).convert('RGB')
        truth=data['tracks_XYZ'][s,ids];k=data['fx_fy_cx_cy']
    gt_xy=truth[:,:2]/truth[:,2:]*k[:2]+k[2:]
    width,height=rgb.size
    canvas=Image.new('RGB',(width*3,height+58),'#101620')
    methods=['GT','60k','OmniX']
    for i,title in enumerate(methods):
        panel=rgb.copy();draw=ImageDraw.Draw(panel)
        if i:
            name='60k' if i==1 else 'omnix'
            with np.load(root/name/f'{case["id"]}_standard.npz',allow_pickle=False) as data:pred=data['tracks_XYZ'][0]
            xy=pred[:,:2]/pred[:,2:]*k[:2]+k[2:]
        for n,point in enumerate(gt_xy):
            x,y=map(float,point)
            if i and n%4==0:
                px,py=map(float,xy[n]);draw.line([(x,y),(px,py)],fill='#ff965e',width=2)
                draw.ellipse((px-2,py-2,px+2,py+2),fill='#ff965e')
            draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill='#66ffd4')
        canvas.paste(panel,(i*width,38))
        draw=ImageDraw.Draw(canvas);draw.text((i*width+12,12),title,fill='white')
    ImageDraw.Draw(canvas).text((12,height+43),'Green: GT query. Orange: predicted 3D reprojected with GT K. Fixed diagnostic clip; not a selected showcase.',fill='white')
    canvas.save(root/'source_reprojection.png')
    print(json.dumps(means),flush=True)
    for c in controls:
        if c['variant']=='standard':print(json.dumps(c),flush=True)


if __name__=='__main__':main()
