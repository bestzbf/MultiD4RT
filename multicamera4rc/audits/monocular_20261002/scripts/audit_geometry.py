"""Read-only geometry checks for packed, TartanAir and Hypersim data."""
import argparse
import csv
import io
import json
from pathlib import Path
import random
import time
import zipfile

import h5py
import numpy as np
import torch


def stats(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    return dict(count=len(a), mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.quantile(a, .95)), max=float(a.max())) if len(a) else dict(count=0)


def points_from_z(depth, K, E, uv=None):
    if uv is None:
        y, x = np.indices(depth.shape)
        uv = np.stack((x, y), -1).reshape(-1, 2)
        z = depth.reshape(-1)
    else:
        z = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    pixel = np.concatenate((uv, np.ones((len(uv), 1))), -1)
    camera = (pixel @ np.linalg.inv(K).T) * z[:, None]
    inv = np.linalg.inv(E)
    return camera @ inv[:3, :3].T + inv[:3, 3]


def official_camera_factorization(P, height, width):
    """Factor official central projection, rather than fit it to GT points."""
    A=np.stack((width/2*P[0,:3]+(width/2-.5)*P[3,:3],
                -height/2*P[1,:3]+(height/2-.5)*P[3,:3],P[3,:3])) @ np.diag([1.,-1.,-1.])
    # RQ via QR and row/column reversal; no extra SciPy dependency.
    Q,B=np.linalg.qr(np.flipud(A).T)
    K=np.fliplr(np.flipud(B.T));R=np.flipud(Q.T)
    sign=np.diag(np.sign(np.diag(K)));K=K@sign;R=sign@R
    scale=K[2,2];K=K/scale
    error=float(np.abs(A-scale*K@R).max())
    return dict(K=K.tolist(),pose_camera_axis_rotation=R.tolist(),
                intrinsic_skew_px=float(K[0,1]),homogeneous_projection_scale=float(scale),
                rotation_determinant=float(np.linalg.det(R)),
                reconstruction_error=error,
                within_current_pinhole_family=bool(abs(K[0,1])<1e-5 and np.linalg.det(R)>.999999
                                                  and error<1e-8 and np.abs(P[[0,1,3],3]).max()<1e-10))


def hypersim_camera_models(root,camera_metadata):
    if not camera_metadata:raise ValueError('--camera-metadata is required')
    available={p.stem for p in Path(root).glob('*.zip')}
    rows=list(csv.DictReader(Path(camera_metadata).open()));scenes=[]
    for row in rows:
        P=np.array([[float(row[f'M_proj_{i}{j}']) for j in range(4)] for i in range(4)])
        factor=official_camera_factorization(P,float(row['settings_output_img_height']),
                                            float(row['settings_output_img_width']))
        scenes.append(dict(scene=row['scene_name'],available=row['scene_name'] in available,**factor))
    bad=[s['scene'] for s in scenes if s['available'] and not s['within_current_pinhole_family']]
    return dict(dataset='Hypersim_camera_family',stage='official_camera_models_checked',
                official_scene_count=len(rows),available_scene_archives=len(available),
                zero_skew_compatible_available_scenes=sum(s['available'] and s['within_current_pinhole_family'] for s in scenes),
                requires_skew_support_or_rectification=bad,
                max_abs_skew_px=max(abs(s['intrinsic_skew_px']) for s in scenes),scenes=scenes,
                limitation='Official camera matrix algebra only; this does not validate all image/depth labels.')


def static_depth_consistency(D0, D1, K0, K1, E0, E1):
    y, x = np.indices(D0.shape)
    valid = np.isfinite(D0) & (D0 > 0)
    ids = np.flatnonzero(valid.reshape(-1))
    ids = ids[np.linspace(0, len(ids)-1, min(16384, len(ids))).astype(int)] if len(ids) else ids
    uv0 = np.stack((x.reshape(-1)[ids], y.reshape(-1)[ids]), -1)
    world = points_from_z(D0, K0, E0, uv0)
    camera = world @ E1[:3, :3].T + E1[:3, 3]
    homogeneous = camera @ K1.T
    uv = homogeneous[:, :2]/np.where(camera[:, 2:] > 1e-8, camera[:, 2:], np.nan)
    px = np.rint(np.nan_to_num(uv, nan=-1e8)).astype(int)
    h, w = D1.shape
    inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h) & (camera[:, 2] > 0)
    px, z = px[inside], camera[inside, 2]
    sample = D1[px[:, 1], px[:, 0]]
    good = np.isfinite(sample) & (sample > 0)
    residual = np.abs(sample[good]-z[good])/np.maximum(sample[good], 1e-8)
    return dict(source_pixels_checked=len(ids), projected_in_frame=int(inside.sum()),
                target_depth_available=int(good.sum()), relative_residual=stats(residual),
                within_one_percent=float((residual < .01).mean()) if len(residual) else None,
                within_five_percent=float((residual < .05).mean()) if len(residual) else None,
                limitation='Static-world diagnostic only; occlusion, dynamics and sensor noise can raise residuals.')


def packed(path, samples, frame_interleave=1, camera_stream_index=0):
    from multicamera4rc.datasets.adapters.mugen_h5 import MugenH5Adapter
    from multicamera4rc.datasets.bridge import prepare_clip

    adapter = MugenH5Adapter(path, split='train', verbose=False, generate_static_tracks=False,
                  prefer_precomputed_tracks=False, require_generated_tracks=False,
                  assume_static_scene=False, strip_padding=True,
                  frame_interleave=frame_interleave, camera_stream_index=camera_stream_index)
    pool = adapter.list_sequences()
    report = dict(dataset=Path(path).stem, path=str(path), train_sequences=len(pool),
                  frame_interleave=frame_interleave, camera_stream_index=camera_stream_index, samples=[])
    for seq in random.Random(9157).sample(sorted(pool), min(samples, len(pool))):
        clip = adapter.load_clip(seq, list(range(min(3, adapter.get_num_frames(seq)))))
        E, K = np.asarray(clip.extrinsics, float), np.asarray(clip.intrinsics, float)
        R = E[:, :3, :3]
        D = [np.asarray(d, float) for d in clip.depths] if clip.depths is not None else []
        sample = prepare_clip(clip, image_mode='native', num_queries=0)
        r = dict(sequence=seq, native_content_hw=[list(a.shape[:2]) for a in clip.images],
                 pose_finite=bool(np.isfinite(E).all()), rotation_orthogonality_max=float(np.abs(R @ R.transpose(0,2,1)-np.eye(3)).max()),
                 rotation_determinants=stats(np.linalg.det(R)),
                 camera_translation_between_first_two=float(np.linalg.norm(np.linalg.inv(E[0])[:3,3]-np.linalg.inv(E[1])[:3,3])) if len(E)>1 else None,
                 has_authoritative_tracks=clip.trajs_3d_world is not None,
                 depth_positive_fraction=[float((np.isfinite(d) & (d > 0)).mean()) for d in D],
                 depth_positive_values=stats(np.concatenate([d[np.isfinite(d) & (d > 0)] for d in D])) if D else dict(count=0),
                 bridge_input_shape=list(sample['images'].shape),
                 valid_depth_pixels=int(sample['targets']['depth_valid'].sum()),
                 calibration_convention=clip.metadata.get('extrinsics_convention'),
                 depth_unit=clip.metadata.get('depth_unit'),
                 source_padding_removed=clip.metadata.get('padding_stripped'))
        r['raw_frame_indices']=clip.metadata.get('raw_frame_indices')
        r['source_crop_top_bottom']=[clip.metadata.get('source_pad_top'),clip.metadata.get('source_pad_bottom')]
        if len(D)>1:
            r['static_world_diagnostic']=static_depth_consistency(D[0],D[1],K[0],K[1],E[0],E[1])
            r['inverted_pose_diagnostic']=static_depth_consistency(D[0],D[1],K[0],K[1],np.linalg.inv(E[0]),np.linalg.inv(E[1]))
        report['samples'].append(r)
        print(json.dumps({'dataset':report['dataset'],'sequence':seq,'pose_orthogonality':r['rotation_orthogonality_max'],
                          'depth_fraction':r['depth_positive_fraction'],'static_diagnostic':r.get('static_world_diagnostic')}),flush=True)
    report['stage']='sampled_geometry_readable'
    report['limitation']='Packed labels have no persistent trajectory or per-pixel material-motion identity. Static-depth checks cannot establish dynamic GT or the accuracy of SfM/pseudo-depth.'
    return report


def tartan(root, samples):
    from multicamera4rc.datasets.adapters.TartanAir import quat_to_rotation_matrix
    root = Path(root)
    envs = [p for p in sorted(root.iterdir()) if (p/'Easy/depth_left.zip').exists() and (p/'Easy/flow_flow.zip').exists()]
    report=dict(dataset='TartanAir_V1',root=str(root),available_easy_environments=len(envs),samples=[])
    for env in random.Random(9157).sample(envs,min(samples,len(envs))):
        with zipfile.ZipFile(env/'Easy/depth_left.zip') as depths, zipfile.ZipFile(env/'Easy/flow_flow.zip') as flows:
            poses=sorted(n for n in depths.namelist() if n.endswith('/pose_left.txt'))
            name=poses[0];base=name.rsplit('/',1)[0]
            pose=np.loadtxt(io.BytesIO(depths.read(name)))
            f0=min(10,len(pose)-2);f1=f0+1
            arrays=[];mats=[]
            for frame in [f0,f1]:
                data=np.load(io.BytesIO(depths.read(f'{base}/depth_left/{frame:06d}_left_depth.npy'))).astype(float)
                arrays.append(data)
                p=pose[frame];R=np.asarray(quat_to_rotation_matrix(p[3:]),float)
                E=np.eye(4);E[:3,:3]=R.T;E[:3,3]=-R.T @ p[:3]
                cv=np.eye(4);cv[:3,:3]=[[0,1,0],[0,0,1],[1,0,0]]
                mats.append(cv @ E)
            K=np.array([[320.,0,320.],[0,320.,240.],[0,0,1.]])
            D0,D1=arrays;E0,E1=mats
            world=points_from_z(D0,K,E0)
            camera=world @ E1[:3,:3].T+E1[:3,3]
            homogeneous=camera @ K.T
            projection=homogeneous[:,:2]/np.where(camera[:,2:] > 1e-8,camera[:,2:],np.nan)
            y,x=np.indices(D0.shape);pixels=np.stack((x,y),-1).reshape(-1,2)
            gt_flow=np.load(io.BytesIO(flows.read(f'{base}/flow/{f0:06d}_{f1:06d}_flow.npy'))).reshape(-1,2)
            err=np.linalg.norm(projection-pixels-gt_flow,axis=-1)
            valid=np.isfinite(D0.reshape(-1)) & (D0.reshape(-1)>0) & (camera[:,2]>0) & np.isfinite(gt_flow).all(-1)
            r=dict(environment=env.name,trajectory=base.rsplit('/',1)[-1],frames=[f0,f1],native_hw=list(D0.shape),
                   checked_pixels=int(valid.sum()),static_world_projected_flow_epe_px=stats(err[valid]),
                   depth_temporal_consistency=static_depth_consistency(D0,D1,K,K,E0,E1),
                   calibration='Official V1 pinhole fx=fy=320,cx=320,cy=240; pose XYZW camera-to-world NED converted to OpenCV camera axes')
            r['sample_passed']=r['static_world_projected_flow_epe_px'].get('p95',float('inf'))<.05
            report['samples'].append(r);print(json.dumps(r),flush=True)
    report['stage']='sampled_consistent' if report['samples'] and all(s['sample_passed'] for s in report['samples']) else 'needs_review'
    report['capability']='Dense rendered depth, calibration and optical flow. Audited flow agrees with stationary world surfaces; does not add dynamic-object trajectories.'
    return report


def hypersim(root, samples, camera_metadata=None):
    root=Path(root);pool=sorted(root.glob('*.zip'))
    report=dict(dataset='Hypersim',root=str(root),available_scene_archives=len(pool),samples=[])
    official={r['scene_name']:r for r in csv.DictReader(Path(camera_metadata).open())} if camera_metadata else {}
    for path in random.Random(9157).sample(pool,min(samples,len(pool))):
        with zipfile.ZipFile(path) as z:
            names=z.namelist();scene=path.stem
            csv_name=next(n for n in names if n.endswith('/metadata_scene.csv'))
            units={r['parameter_name']:float(r['parameter_value']) for r in csv.DictReader(io.StringIO(z.read(csv_name).decode()))}
            scale=units['meters_per_asset_unit']
            def array(name):
                with h5py.File(io.BytesIO(z.read(name)),'r') as f:return np.asarray(f['dataset'])
            camera_name=next(n for n in names if n.endswith('/camera_keyframe_positions.hdf5'))
            base=camera_name.rsplit('/',1)[0];cam=base.rsplit('/',1)[-1]
            pos=array(camera_name);rot=array(base+'/camera_keyframe_orientations.hdf5');ids=array(base+'/camera_keyframe_frame_indices.hdf5')
            chosen=[0,len(ids)//2,len(ids)-1]
            for index in chosen:
                frame=int(ids[index]);geo=f'{scene}/images/scene_{cam}_geometry_hdf5/frame.{frame:04d}'
                if geo+'.position.hdf5' not in names:continue
                world=array(geo+'.position.hdf5').astype(float);depth=array(geo+'.depth_meters.hdf5').astype(float)
                valid=np.isfinite(world).all(-1) & np.isfinite(depth) & (depth>0)
                expected=np.linalg.norm(world-pos[index],axis=-1)*scale
                err=np.abs(expected[valid]-depth[valid])
                relative=err/np.maximum(depth[valid],1e-8)
                camera=(world-pos[index]) @ rot[index]
                camera*=np.array([1.,-1.,-1.])
                y,x=np.indices(depth.shape);good=valid & (camera[...,2]>0)
                sample_ids=np.flatnonzero(good.reshape(-1))[::97]
                pc=camera.reshape(-1,3)[sample_ids];u=x.reshape(-1)[sample_ids];v=y.reshape(-1)[sample_ids]
                ax=np.stack((pc[:,0]/pc[:,2],np.ones(len(pc))),1);ay=np.stack((pc[:,1]/pc[:,2],np.ones(len(pc))),1)
                kx=np.linalg.lstsq(ax,u,rcond=None)[0];ky=np.linalg.lstsq(ay,v,rcond=None)[0]
                reproj=np.hypot(ax@kx-u,ay@ky-v)
                r=dict(scene=scene,camera=cam,frame=frame,native_hw=list(depth.shape),meters_per_asset_unit=scale,
                       valid_pixel_fraction=float(valid.mean()),range_vs_world_position_absolute_error_m=stats(err),
                       range_vs_world_position_relative_error=stats(relative),
                       rotation_orthogonality_max=float(np.abs(rot[index].T @ rot[index]-np.eye(3)).max()),
                       pinhole_fit_diagnostic={'fx':float(kx[0]),'fy':float(ky[0]),'cx':float(kx[1]),'cy':float(ky[1]),'reprojection_px':stats(reproj)},
                       depth_definition='Euclidean ray distance in meters; must convert to camera-Z for the current geometry bridge',
                       sample_passed=stats(relative).get('p95',float('inf')) < 1e-4 and stats(reproj).get('p95',float('inf')) < .01)
                if scene in official:
                    entry=official[scene]
                    P=np.array([[float(entry[f'M_proj_{i}{j}']) for j in range(4)] for i in range(4)])
                    points_gl=(world-pos[index]) @ rot[index]
                    selected=points_gl.reshape(-1,3)[sample_ids]
                    projected=np.concatenate((selected,np.ones((len(selected),1))),1) @ P.T
                    ndc=projected[:,:2]/projected[:,3:]
                    h,w=depth.shape
                    pixels=np.stack(((ndc[:,0]+1)*w/2-.5,(1-ndc[:,1])*h/2-.5),1)
                    error=np.linalg.norm(pixels-np.stack((u,v),1),axis=1)
                    r['official_projection_reprojection_px']=stats(error)
                    r['official_projection_matrix']=P.tolist()
                    r['official_pinhole_factorization']=official_camera_factorization(P,h,w)
                    r['native_label_precision']='Original depth and world-position HDF5 store float16; millimeter range discrepancies and subpixel projection discrepancies are expected.'
                    r['sample_passed']=stats(relative).get('p95',float('inf')) < .005 and stats(error).get('p95',float('inf')) < 2.
                report['samples'].append(r);print(json.dumps(r),flush=True)
    report['stage']='sampled_consistent' if report['samples'] and all(s['sample_passed'] for s in report['samples']) else 'needs_review'
    report['capability']='Authoritative dense per-pixel world positions and static-scene geometry. ZIP/HDR reader and official camera-parameter integration still needed.'
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['packed','tartan','hypersim','hypersim_camera_models'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--samples',type=int,default=3);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--camera-metadata')
    p.add_argument('--frame-interleave',type=int,default=1)
    p.add_argument('--camera-stream-index',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(2);began=time.monotonic()
    report=(hypersim_camera_models(args.root,args.camera_metadata) if args.kind=='hypersim_camera_models' else
            hypersim(args.root,args.samples,args.camera_metadata) if args.kind=='hypersim' else
            packed(args.root,args.samples,args.frame_interleave,args.camera_stream_index) if args.kind=='packed' else
            tartan(args.root,args.samples))
    report['elapsed_seconds']=time.monotonic()-began;report['gpu_used']=False
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed':str(args.output),'stage':report['stage']}),flush=True)


if __name__=='__main__':main()
