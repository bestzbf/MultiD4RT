"""Audit native static multiview GT without changing training supervision.

All decoded source pixels retain dense labels. The strided pixel selection
below is only an independent cross-view depth/calibration diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.datasets.partial_dense import attach_partial_dense_recipe
from multicamera4rc.train import _sources


def cross_view_depth(target, source=0, stride=8):
    depth=target['depth'][0].cpu().numpy()
    valid=target['depth_valid'][0].cpu().numpy().astype(bool)
    K=target['K'][0].cpu().numpy().astype(np.float64)
    poses=target['c2w'][0].cpu().numpy().astype(np.float64)
    h,w=depth.shape[-2:]
    yy,xx=np.mgrid[0:h:stride,0:w:stride]
    xx,yy=xx.ravel(),yy.ravel()
    keep=valid[source,yy,xx];xx,yy=xx[keep],yy[keep]
    pixels=np.stack([xx,yy,np.ones_like(xx)],-1)
    camera=(pixels@np.linalg.inv(K[source]).T)*depth[source,yy,xx,None]
    world=camera@poses[source,:3,:3].T+poses[source,:3,3]
    pairs=[]
    for other in range(len(depth)):
        if other==source: continue
        xyz=(world-poses[other,:3,3])@poses[other,:3,:3]
        uvz=xyz@K[other].T
        uv=uvz[:,:2]/np.maximum(uvz[:,2:3],1e-10)
        finite=np.isfinite(uv).all(-1)
        inside=finite & (xyz[:,2]>0) & (uv[:,0]>=0) & (uv[:,0]<w-.5) & (uv[:,1]>=0) & (uv[:,1]<h-.5)
        uv=uv[inside];xyz=xyz[inside]
        ix,iy=np.rint(uv[:,0]).astype(int),np.rint(uv[:,1]).astype(int)
        measured=depth[other,iy,ix];good=valid[other,iy,ix]
        predicted=xyz[good,2];measured=measured[good]
        relative=(predicted-measured)/np.maximum(measured,1e-10)
        matches=np.abs(relative)<.05
        # Positive residuals may be occluded by a closer target surface.
        pairs.append(dict(target_camera=other,diagnostic_source_pixels=len(world),
            projected_valid_pixels=len(relative),matched_within_5pct=int(matches.sum()),
            within_1pct_fraction=float((np.abs(relative)<.01).mean()) if len(relative) else None,
            within_5pct_fraction=float(matches.mean()) if len(relative) else None,
            median_relative_error=float(np.median(np.abs(relative))) if len(relative) else None,
            possible_occlusion_fraction=float((relative>.05).mean()) if len(relative) else None))
    return pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-config',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sources',nargs='*',default=['blended','taskonomy','hypersim'])
    p.add_argument('--sequences-per-source',type=int,default=3)
    p.add_argument('--anchors-per-sequence',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(2)
    dataset=create_dataset(args.data_config)
    entries={(si,seq):(dataset.coverage_ends[i-1] if i else 0, end)
        for i,((si,seq,_,_),end) in enumerate(zip(dataset.coverage_entries,dataset.coverage_ends))}
    report=dict(data_config=args.data_config,coverage_windows=dataset.coverage_size,
        epoch_size=len(dataset),diagnostic_stride=8,supervision='all valid pixels of decoded sources',samples=[])
    for si,source in enumerate(dataset.sources):
        if source['display_name'] not in args.sources: continue
        assert source['view_mode']=='static_multiview'
        pool=source['pool']; choices=source['camera_choices']
        positions=sorted(set(round(i*(len(pool)-1)/max(1,args.sequences_per_source-1))
            for i in range(args.sequences_per_source)))
        for pos in positions:
            seq=pool[pos][0];begin,end=entries[si,seq]
            anchors=sorted(set(begin+round(i*(end-begin-1)/max(1,args.anchors_per_sequence-1))
                for i in range(args.anchors_per_sequence)))
            for index in anchors:
                for count in choices:
                    began=time.monotonic();source['camera_choices']=(count,)
                    sample=dataset[index];info=sample['static_multiview'];views=info['view_indices']
                    C=len(views)
                    assert sample['clip_id'].endswith('/'+seq) and sample['dataset_name']==source['display_name']
                    assert len(set(views))==C and info['actual_cameras']==C
                    assert sample['camera_id'].tolist()==list(range(C))
                    assert sample['time_idx'].tolist()==[0]*C and sample['query_time_idx'].tolist()==[0]
                    assert set(views)<=set(source['frame_pool'].get(seq,range(pool[pos][1])))
                    raw=source['adapter'].load_clip(seq,views)
                    original=prepare_clip(raw,num_queries=0,image_mode='native')
                    attach_partial_dense_recipe(raw,original,kind='geometry',image_mode='native')
                    for key in ('images',):
                        torch.testing.assert_close(sample[key],original[key],rtol=0,atol=0)
                    for key in ('depth','depth_valid','K','c2w'):
                        torch.testing.assert_close(sample['targets'][key],original['targets'][key],rtol=0,atol=0)
                    batch=collate_adapter_samples([sample]);sources=_sources(batch,4)
                    target=materialize_dense_targets(batch,sources)
                    assert not any(k.startswith('sparse_') for k in target)
                    for key in ('depth','ray','trajectory','displacement','K','c2w'):
                        assert torch.isfinite(target[key]).all(),key
                    assert not target['displacement_valid'].any() and not target['static_mask'].any()
                    assert target['trajectory_valid'].sum()==target['source_pixel_valid'].sum()
                    all_views=materialize_dense_targets(batch,torch.arange(C)[None]) if C>sources.shape[1] else target
                    assert torch.allclose(all_views['trajectory'][all_views['trajectory_valid']].norm(dim=-1).mean(),
                        torch.tensor(1.),atol=2e-5,rtol=2e-5)
                    del all_views
                    pairs=cross_view_depth(target)
                    # This rejects gross pose/depth-frame disagreement, while
                    # retaining ordinary occlusions in the diagnostic report.
                    matches=sum(p['matched_within_5pct'] for p in pairs)
                    if C>1 and matches<128:
                        raise AssertionError(f'no corroborating static overlap: {seq}, views={views}, pairs={pairs}')
                    centers=target['c2w'][0,:,:3,3]
                    item=dict(source=source['display_name'],sequence=seq,index=index,requested_cameras=count,
                        actual_cameras=C,view_indices=views,selection=info['selection'],
                        input_shape=list(batch['images'].shape),time_idx=sample['time_idx'].tolist(),
                        scene_scale=float(target['scene_scale'][0]),decoded_sources=sources.shape[1],
                        dense_source_pixels=int(target['source_pixel_valid'].sum()),
                        depth_pixels=int(target['depth_valid'].sum()),motion_pixels=0,
                        max_baseline_normalized=float(torch.cdist(centers,centers).max()),
                        rgb_depth_calibration_unchanged=True,cross_view_depth=pairs,
                        elapsed_seconds=time.monotonic()-began)
                    report['samples'].append(item);print(json.dumps(item),flush=True)
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
                    del sample,original,raw,batch,target
        source['camera_choices']=choices
    assert report['samples'],'no static sources audited'
    report['passed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,samples=len(report['samples']))),flush=True)


if __name__=='__main__': main()
