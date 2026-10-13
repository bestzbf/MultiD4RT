"""Regression against the untouched official evaluate_data_source caller."""
from __future__ import annotations

import argparse
import ast
import io
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import tqdm

from .common import digest,load_module,write_json
from .run_tapvid import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args();base=Path(__file__).parent
    caller=base/'references/tapvid3d_evaluate_model.py'
    official=load_module('_official_resolution_regression_metrics',base/'references/tapvid3d_metrics.py')
    names={'get_new_hw_with_given_smallest_side_length','get_jpeg_byte_hw','get_average_over_metrics','evaluate_data_source'}
    functions=[n for n in ast.parse(caller.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in functions}!=names:raise AssertionError('Missing official caller function')
    env=dict(np=np,io=io,os=os,Image=Image,tqdm=tqdm,logging=logging,_DEBUG=SimpleNamespace(value=False))
    env['metrics']=official
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(caller),'exec'),env)
    # The original function bodies are executed unchanged; only CLI imports
    # are avoided. Inputs and output directory structure match the release.
    root=args.campaign/'pstudio_diagnosis_20261013/official_caller_parity'
    root.mkdir(parents=True,exist_ok=True)
    cache=args.campaign/'tapvid16_pstudio_full_eval'
    plan=json.loads((cache/'plan.json').read_text())
    fixture=[]
    constant=np.zeros((16,8,3),dtype=np.float32);constant[...,2]=3
    pred=constant.copy();pred[...,0]=.008
    visible=np.ones((16,8),dtype=bool)
    buffer=io.BytesIO();Image.new('RGB',(640,360)).save(buffer,format='JPEG')
    images=np.array([buffer.getvalue()]*16,dtype=object)
    fixture.append(dict(id='nonzero_resolution_regression',truth=constant,pred=pred,visible=visible,
                        k=np.array([500.,500.,320.,180.]),hw=[360,640],images=images))
    for index in [0,318,636,953]:
        case=plan['cases'][index];s=case['start'];ids=case['point_ids']
        with np.load(case['path'],allow_pickle=True) as data:
            truth=data['tracks_XYZ'][s:s+16,ids].astype(np.float32)
            visible=data['visibility'][s:s+16,ids].astype(bool)
            k=data['fx_fy_cx_cy'].astype(np.float64)
            images=data['images_jpeg_bytes'][s:s+16].copy()
        img=Image.open(io.BytesIO(bytes(images[0])));hw=[img.height,img.width]
        for method in ['multid4rt','omnix']:
            file=next((cache/method).glob(f'worker*/{case["id"]}.npz'))
            with np.load(file,allow_pickle=False) as data:pred=data['tracks_XYZ'].copy()
            fixture.append(dict(id=method+'_'+case['id'],truth=truth,pred=pred,visible=visible,k=k,hw=hw,images=images))
    rows=[]
    for item in fixture:
        folder=root/item['id'];gt=folder/'gt';out=folder/'pred';gt.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=True)
        # Median scale does not consume query points, but supply the required
        # original schema including zero source timestamps.
        np.savez_compressed(gt/'video.npz',images_jpeg_bytes=item['images'],
            queries_xyt=np.zeros((item['truth'].shape[1],3),dtype=np.float32),
            tracks_XYZ=item['truth'],visibility=item['visible'],fx_fy_cx_cy=item['k'])
        np.savez_compressed(out/'video.npz',tracks_XYZ=item['pred'],visibility=np.ones_like(item['visible']))
        original=env['evaluate_data_source'](['video.npz'],str(gt),str(out),['median'])['median']['average_pts_within_thresh']
        adapter=score(official,item['pred'],item['truth'],item['visible'],item['k'],item['hw'])['apd3d']
        np.testing.assert_allclose(original,adapter,rtol=0,atol=1e-12)
        if item['id']=='nonzero_resolution_regression' and adapter!=1.:raise AssertionError('Resolution bug not detected')
        rows.append(dict(id=item['id'],official_caller_apd3d=float(original),adapter_apd3d=adapter))
    write_json(root/'proof.json',dict(stage='passed',official_caller_sha256=digest(caller),
        executed_unchanged_official_function_bodies=sorted(names),new_model_forwards=0,records=rows))
    correction=cache/'resolution_correction.json'
    d=json.loads(correction.read_text());d['independent_official_caller_regression']=str(root/'proof.json');write_json(correction,d)
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
