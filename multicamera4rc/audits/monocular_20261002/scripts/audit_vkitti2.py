"""Read-only VKITTI2 dense depth, camera, optical flow and scene-flow audit."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image


def stats(a):
    a=np.asarray(a,dtype=np.float64).reshape(-1);a=a[np.isfinite(a)]
    return dict(count=len(a),mean=float(a.mean()),p50=float(np.median(a)),
                p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(count=0)


def project(xyz,K):
    a=xyz@K.T
    return a[...,:2]/np.where(a[...,2:]>1e-8,a[...,2:],np.nan)


def sample(a,uv):
    return cv2.remap(a,uv[...,0].astype(np.float32),uv[...,1].astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def rotation_y(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);began=time.monotonic()
    cv2.setNumThreads(2)
    from multicamera4rc.datasets.adapters.VirtualKitti import VKITTI2Adapter
    from multicamera4rc.datasets.bridge import prepare_clip
    root=Path(args.root)
    report=dict(dataset='VKITTI2_native',root=str(root),samples=[],gpu_used=False,
        source='https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/',
        label_recipe='Depth PNG is camera-Z centimeters. Optical flow uses RGB16 normalized u/v and B-valid. Scene flow uses RGB16 camera-space displacement in meters, clamped [-10,10]. Exact world conversion depends on the native camera-space convention verified below.',
        limitation='Audited clone variations only; all variations of one SceneXX share a physical scene and must stay in the same train/validation split. Adjacent scene flow does not by itself prove long-term occluded material trajectories.')
    for cam in ['Camera_0','Camera_1']:
        adapter=VKITTI2Adapter(str(root),camera=cam,load_precomputed=False,
                              load_normals=False,load_flow=True,verbose=False)
        report.setdefault('indexed_sequences_per_camera',{})[cam]=len(adapter.list_sequences())
        for scene in ['Scene01','Scene02','Scene06','Scene18','Scene20']:
            seq=scene+'_clone';base=root/scene/'clone';length=adapter.get_num_frames(seq)
            poses,Ks=adapter._load_sequence_metadata(seq)
            annotations=np.genfromtxt(base/'pose.txt',names=True,dtype=None,encoding='utf-8')
            annocam=annotations[annotations['cameraID']==int(cam[-1])]
            report.setdefault('object_pose_coverage',{})[seq+'/'+cam]={
                'sequence_frames':length,
                'track_count':len(np.unique(annocam['trackID'])),
                'all_tracks_have_every_frame':all(len(np.unique(annocam['frame'][annocam['trackID']==tid]))==length for tid in np.unique(annocam['trackID'])),
                'max_abs_world_pitch_roll':float(max(np.abs(annocam['rotation_world_space_x']).max(),np.abs(annocam['rotation_world_space_z']).max()))}
            for f0 in [10,length//2,length-2]:
                f1=f0+1
                if f0 not in poses or f1 not in poses or f0 not in Ks or f1 not in Ks:
                    raise ValueError('Incomplete metadata; never substitute identity poses.')
                clip=adapter.load_clip(seq,[f0,f1]);clip.metadata['frame_indices']=[f0,f1]
                E0,E1=np.asarray(clip.extrinsics,float);K0,K1=np.asarray(clip.intrinsics,float)
                D0,D1=[np.asarray(d,float) for d in clip.depths];h,w=D0.shape
                native=base/'frames'
                raw=cv2.imread(str(native/'forwardFlow'/cam/f'flow_{f0:05d}.png'),cv2.IMREAD_UNCHANGED)
                sfraw=cv2.imread(str(native/'forwardSceneFlow'/cam/f'sceneFlow_{f0:05d}.png'),cv2.IMREAD_UNCHANGED)
                with Image.open(native/'instanceSegmentation'/cam/f'instancegt_{f0:05d}.png') as indexed:
                    inst=np.asarray(indexed)
                if inst.shape!=D0.shape:
                    raise ValueError('Expected native indexed instance IDs; do not decode palette colors as IDs.')
                F=(2.*raw[...,2:0:-1].astype(float)/65535.-1.)*np.array([w-1,h-1])
                SF=(2.*sfraw[...,::-1].astype(float)/65535.-1.)*10.
                y,x=np.indices((h,w));uv=np.stack((x,y),-1).astype(float)
                pixels=np.concatenate((uv,np.ones((h,w,1))),-1)
                X0=(pixels@np.linalg.inv(K0).T)*D0[...,None]
                C0=np.linalg.inv(E0);W0=X0@C0[:3,:3].T+C0[:3,3]
                X1static=W0@E1[:3,:3].T+E1[:3,3]
                valid=(raw[...,0]>0)&(D0>0)&(D0<655.35)&np.isfinite(SF).all(-1)
                saturated=(sfraw==0).any(-1)|(sfraw==65535).any(-1)
                valid &= ~saturated
                bg=valid&(inst==0)&(X1static[...,2]>0)
                # Test both plausible conventions against independent static-world geometry.
                camera_diff=SF-(X1static-X0)
                world_disp=SF
                endpoint_A=X0+SF
                endpoint_B=X1static+SF
                uv1=uv+F
                optical_A=np.linalg.norm(project(endpoint_A,K1)-uv1,axis=-1)
                optical_B=np.linalg.norm(project(endpoint_B,K1)-uv1,axis=-1)
                flowstatic=np.linalg.norm(project(X1static,K1)-uv1,axis=-1)
                targetdepth=sample(D1.astype(np.float32),uv1)
                inside=(uv1[...,0]>=0)&(uv1[...,0]<=w-1)&(uv1[...,1]>=0)&(uv1[...,1]<=h-1)
                endpoint_valid=valid&inside&(targetdepth>0)&(targetdepth<655.35)
                obj=annotations[(annotations['frame']==f0)&(annotations['cameraID']==int(cam[-1]))]
                obj1=annotations[(annotations['frame']==f1)&(annotations['cameraID']==int(cam[-1]))]
                centers=np.column_stack([obj[n] for n in ['world_space_X','world_space_Y','world_space_Z']])
                truth=np.column_stack([obj[n] for n in ['camera_space_X','camera_space_Y','camera_space_Z']])
                centererror=np.linalg.norm(centers@E0[:3,:3].T+E0[:3,3]-truth,axis=-1)
                rigid_errors={1:[],-1:[]};moving_yaw_pixels=0
                for row in obj:
                    match=obj1[obj1['trackID']==row['trackID']]
                    if not len(match):continue
                    nxt=match[0]
                    if any(abs(r[n])>1e-7 for r in [row,nxt] for n in ['rotation_world_space_x','rotation_world_space_z']):
                        continue  # Other Euler orders have not been verified.
                    ids=valid&(inst==int(row['trackID'])+1)
                    if not ids.any():continue
                    center0=np.array([row[n] for n in ['world_space_X','world_space_Y','world_space_Z']])
                    center1=np.array([nxt[n] for n in ['world_space_X','world_space_Y','world_space_Z']])
                    angle=float(nxt['rotation_world_space_y']-row['rotation_world_space_y'])
                    if abs(angle)>1e-5:moving_yaw_pixels+=int(ids.sum())
                    for sign in [1,-1]:
                        W1=(W0[ids]-center0)@rotation_y(sign*angle).T+center1
                        expected=W1@E1[:3,:3].T+E1[:3,3]-X0[ids]
                        rigid_errors[sign].extend(np.linalg.norm(expected-SF[ids],axis=-1))
                sample_bridge=prepare_clip(clip,image_mode='native',num_queries=0)
                r=dict(sequence=seq,camera=cam,frames=[f0,f1],native_hw=[h,w],
                    rotation_orthogonality_max=float(np.max(np.abs(E0[:3,:3]@E0[:3,:3].T-np.eye(3)))),
                    object_world_camera_center_consistency_m=stats(centererror),
                    object_pose_rigid_yaw_plus_vs_native_scene_flow_error_m=stats(rigid_errors[1]),
                    object_pose_rigid_yaw_minus_vs_native_scene_flow_error_m=stats(rigid_errors[-1]),
                    object_pixels_with_nonzero_yaw_change=moving_yaw_pixels,
                    valid_dense_scene_flow_pixels=int(valid.sum()),scene_flow_saturated_pixels=int(saturated.sum()),
                    background_static_flow_epe_px=stats(flowstatic[bg]),
                    background_scene_flow_vs_camera_coordinate_difference_m=stats(np.linalg.norm(camera_diff,axis=-1)[bg]),
                    background_scene_flow_vs_stationary_world_m=stats(np.linalg.norm(world_disp,axis=-1)[bg]),
                    optical_flow_from_camera_difference_scene_flow_epe_px=stats(optical_A[valid]),
                    optical_flow_from_world_displacement_scene_flow_epe_px=stats(optical_B[valid]),
                    target_depth_vs_camera_difference_endpoint_relative_error=stats((np.abs(targetdepth-endpoint_A[...,2])/np.maximum(targetdepth,1e-8))[endpoint_valid]),
                    target_depth_vs_world_displacement_endpoint_relative_error=stats((np.abs(targetdepth-endpoint_B[...,2])/np.maximum(targetdepth,1e-8))[endpoint_valid]),
                    bridge_shape=list(sample_bridge['images'].shape),bridge_times=sample_bridge['time_idx'].tolist(),
                    bridge_has_valid_dense_depth=bool(sample_bridge['targets']['depth_valid'].any()))
                report['samples'].append(r)
                print(json.dumps({k:v for k,v in r.items() if k in ['sequence','camera','frames','object_pose_rigid_yaw_plus_vs_native_scene_flow_error_m','object_pose_rigid_yaw_minus_vs_native_scene_flow_error_m','object_pixels_with_nonzero_yaw_change']}),flush=True)
    report['stage']='sampled_scene_flow_convention_audited'
    report['elapsed_seconds']=time.monotonic()-began
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'stage':report['stage'],'pairs':len(report['samples'])}),flush=True)


if __name__=='__main__':main()
