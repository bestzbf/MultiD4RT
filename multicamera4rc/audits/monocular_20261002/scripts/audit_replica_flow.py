"""Read-only dense flow audit against independent native vertex trajectories."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch


def stats(a):
    a=np.asarray(a,dtype=np.float64).reshape(-1)
    a=a[np.isfinite(a)]
    return dict(count=len(a),mean=float(a.mean()),p50=float(np.median(a)),
                p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(count=0)


def flow(path):
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype!=np.uint16 or raw.ndim!=2:
        raise ValueError(f'Unexpected official flow encoding: {path}')
    a=np.ascontiguousarray(raw).view(np.float16).astype(np.float32)
    w=a.shape[1]//2
    return np.stack((a[:,:w],a[:,w:]),-1)


def remap(a,uv,nearest=False):
    shape=uv.shape[:-1]
    flat=np.asarray(uv,np.float32).reshape(-1,2)
    if not len(flat):
        return np.empty(shape+a.shape[2:],dtype=a.dtype)
    width=min(1024,len(flat));height=(len(flat)+width-1)//width
    padded=np.zeros((height*width,2),np.float32);padded[:len(flat)]=flat
    xy=padded.reshape(height,width,2)
    out=cv2.remap(a,xy[...,0],xy[...,1],cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return out.reshape((-1,)+a.shape[2:])[:len(flat)].reshape(shape+a.shape[2:])


def unproject(uv,z,K,E):
    pixels=np.concatenate((uv,np.ones((len(uv),1))),-1)
    xyz=(pixels@np.linalg.inv(K).T)*z[:,None]
    inv=np.linalg.inv(E)
    return xyz@inv[:3,:3].T+inv[:3,3]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2);began=time.monotonic()
    from multicamera4rc.datasets.adapters.dynamic_replica import DynamicReplicaAdapter
    adapter=DynamicReplicaAdapter(args.root,split='train',load_trajectories=True,
        rgb_from_trajectory=False,skip_depth_when_tracks=False,index_workers=2,io_workers=1,
        verbose=False,cache_dir=str(args.report_dir/'index_cache'),pixel_coordinates='array_index_v1')
    audited=json.loads((args.report_dir/'dynamic_replica.json').read_text())
    report=dict(dataset='DynamicReplica',samples=[],gpu_used=False,
        official_flow_decoder='https://github.com/facebookresearch/dynamic_stereo/blob/main/notebooks/Dynamic_Replica_demo.ipynb',
        depth_decoder='PNG uint16 bits reinterpreted as float16 camera-Z',
        flow_decoder='PNG uint16 bits reinterpreted as float16, horizontal halves are u and v displacement; masks > 0',
        limitation='Dense adjacent visible pair motion is possible after validity/occlusion masking. Persistent vertex checks do not provide full per-pixel long-term occluded trajectories.')
    for seq in audited['selected_sequences']:
        for position in [10,149]:
            clip=adapter.load_clip(seq,[position,position+1])
            f0,f1=clip.metadata['frame_numbers']
            base=Path(args.root)/'train'/seq
            F=flow(base/'flow_forward'/f'{seq}_{f0:04d}.png')
            B=flow(base/'flow_backward'/f'{seq}_{f1:04d}.png')
            M0=cv2.imread(str(base/'flow_forward_mask'/f'{seq}_{f0:04d}.png'),cv2.IMREAD_UNCHANGED)>0
            M1=cv2.imread(str(base/'flow_backward_mask'/f'{seq}_{f1:04d}.png'),cv2.IMREAD_UNCHANGED)>0
            h,w=M0.shape;y,x=np.indices((h,w));pixels=np.stack((x,y),-1).astype(np.float32)
            end=pixels+F
            inside=(end[...,0]>=0)&(end[...,0]<=w-1)&(end[...,1]>=0)&(end[...,1]<=h-1)
            bm=remap(M1.astype(np.uint8),end,True)>0
            D0,D1=[np.asarray(d,np.float32) for d in clip.depths]
            targetz=remap(D1,end)
            pair=M0&bm&inside&np.isfinite(F).all(-1)&(D0>0)&(targetz>0)
            kernel=np.ones((3,3),np.uint8)
            def interior_depth(d):
                lo=cv2.erode(d,kernel);hi=cv2.dilate(d,kernel)
                return (lo>0)&np.isfinite(d)&((hi-lo)<(.01*d+.02))
            interior0,interior1=interior_depth(D0),interior_depth(D1)
            pair_interior=pair&interior0&(remap(interior1.astype(np.uint8),end,True)>0)
            composition=F+remap(B,end)
            U=np.asarray(clip.trajs_2d,float);X=np.asarray(clip.trajs_3d_world,float)
            K=np.asarray(clip.intrinsics,float);E=np.asarray(clip.extrinsics,float)
            visible=np.asarray(clip.visibs,bool).all(0)&np.asarray(clip.valids,bool).all(0)
            visible &= np.isfinite(U).all((0,2))
            for t in [0,1]:
                visible &= (U[t,:,0]>=0)&(U[t,:,0]<=w-1)&(U[t,:,1]>=0)&(U[t,:,1]<=h-1)
            ids=np.flatnonzero(visible)
            mask=remap(M0.astype(np.uint8),U[0,ids],True)>0
            ids=ids[mask]
            sampled=remap(F,U[0,ids]);epe=np.linalg.norm(sampled-(U[1,ids]-U[0,ids]),axis=-1)
            cameras=np.einsum('tij,tnj->tni',E[:,:3,:3],X)+E[:,None,:3,3]
            best=np.full(len(ids),np.inf)
            rounded=np.rint(U[0,ids]).astype(int)
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    sx=(rounded[:,0]+dx).clip(0,w-1);sy=(rounded[:,1]+dy).clip(0,h-1)
                    z=D0[sy,sx]
                    er=np.abs(z-cameras[0,ids,2])/np.maximum(cameras[0,ids,2],1e-8)
                    er[z<=0]=np.inf;best=np.minimum(best,er)
            surface=best<.01
            targetuv=U[0,ids]+sampled
            z0=remap(D0,U[0,ids]);z1=remap(D1,targetuv)
            depthok=(z0>0)&(z1>0)&surface
            safe=(depthok&(remap(interior0.astype(np.uint8),U[0,ids],True)>0)
                  &(remap(interior1.astype(np.uint8),targetuv,True)>0)
                  &(remap(M1.astype(np.uint8),targetuv,True)>0))
            safe &= (targetuv[:,0]>=0)&(targetuv[:,0]<=w-1)&(targetuv[:,1]>=0)&(targetuv[:,1]<=h-1)
            world0=unproject(U[0,ids],z0,K[0],E[0]);world1=unproject(targetuv,z1,K[1],E[1])
            enderr=np.linalg.norm(world1-X[1,ids],axis=-1)
            deltaerr=np.linalg.norm((world1-world0)-(X[1,ids]-X[0,ids]),axis=-1)
            item=dict(sequence=seq,frames=[f0,f1],native_hw=[h,w],
                dense_pair_valid_pixels=int(pair.sum()),dense_pair_valid_fraction=float(pair.mean()),
                dense_pair_boundary_safe_pixels=int(pair_interior.sum()),
                dense_pair_boundary_safe_fraction=float(pair_interior.mean()),
                forward_backward_epe_px=stats(np.linalg.norm(composition,axis=-1)[pair]),
                independently_checked_visible_vertices=len(ids),
                flow_vs_vertex_projection_epe_px=stats(epe),
                flow_vs_surface_vertex_epe_px=stats(epe[surface]),
                target_world_endpoint_error_m=stats(enderr[depthok]),
                displacement_vs_native_vertex_error_m=stats(deltaerr[depthok]),
                target_world_endpoint_error_relative_to_depth=stats(enderr[depthok]/np.maximum(cameras[1,ids,2][depthok],1e-8)),
                interior_flow_vs_vertex_projection_epe_px=stats(epe[safe]),
                interior_target_world_endpoint_error_m=stats(enderr[safe]),
                interior_target_world_endpoint_relative_error=stats(enderr[safe]/np.maximum(cameras[1,ids,2][safe],1e-8)),
                interior_displacement_vs_vertex_error_m=stats(deltaerr[safe]),
                motion_valid_mask_recipe='Forward-valid, backward-valid at endpoint, endpoint in bounds, finite flow and both positive depths. Boundary-safe diagnostics also require 3x3 depth min>0 and max-min<0.01*depth+0.02 at source and target; thresholds are declared filters, not proof of full dataset accuracy.')
            item['sampled_flow_consistent']=item['flow_vs_surface_vertex_epe_px'].get('p95',float('inf'))<1.
            report['samples'].append(item)
            print(json.dumps({k:v for k,v in item.items() if k in ['sequence','frames','dense_pair_boundary_safe_fraction','interior_flow_vs_vertex_projection_epe_px','interior_target_world_endpoint_relative_error']}),flush=True)
    report['stage']='sampled_flow_consistent' if all(s['sampled_flow_consistent'] for s in report['samples']) else 'needs_flow_review'
    report['elapsed_seconds']=time.monotonic()-began
    (args.report_dir/'dynamic_replica_flow.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'stage':report['stage'],'pairs':len(report['samples'])}),flush=True)


if __name__=='__main__':main()
