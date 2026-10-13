"""Audit cached PStudio errors and capture heads on fixed, unselected clips.

Oracle replacements are explicitly diagnostic and never benchmark results.
The existing immutable 60k comparison and its predictions are not modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .common import digest, load_model, load_module, prepare_rgb, write_json
from .run_tapvid import metric_intrinsics, sample, score


def initial_diagnostics(pred, truth, xy, k, scale):
    """Source depth/ray decomposition; intrinsic fit is verified by real heads."""
    p = pred[0].astype(np.float64) * scale
    g = truth[0].astype(np.float64)
    fit, residual = [], []
    for axis in range(2):
        ray = p[:, axis] / p[:, 2]
        design = np.stack((ray, np.ones_like(ray)), -1)
        beta = np.linalg.lstsq(design, xy[:, axis], rcond=None)[0]
        fit.append(beta)
        residual.append(np.sqrt(np.mean(np.square(design @ beta - xy[:, axis]))))
    # Keep the official clip-wide scale unchanged for depth-only replacement.
    oracle_k = g / g[:, 2:3] * p[:, 2:3]
    oracle_z = p / p[:, 2:3] * g[:, 2:3]
    # Source-only scale control separates camera ray bias from depth shape.
    z_scale = np.median(g[:,2]) / np.median(p[:,2])
    return dict(source_epe_m=float(np.linalg.norm(p-g, axis=-1).mean()),
                source_xy_epe_m=float(np.linalg.norm(p[:, :2]-g[:, :2], axis=-1).mean()),
                source_z_mae_m=float(np.abs(p[:, 2]-g[:, 2]).mean()),
                source_depth_abs_rel=float(np.mean(np.abs(p[:, 2]/g[:, 2]-1))),
                oracle_source_intrinsics_epe_m=float(np.linalg.norm(oracle_k-g, axis=-1).mean()),
                oracle_source_depth_epe_m=float(np.linalg.norm(oracle_z-g, axis=-1).mean()),
                source_depth_abs_rel_source_scale=float(np.mean(np.abs(p[:,2]*z_scale/g[:,2]-1))),
                oracle_source_intrinsics_source_scale_epe_m=float(np.linalg.norm(oracle_k*z_scale-g,axis=-1).mean()),
                fitted_source_intrinsics=dict(fx=float(fit[0][0]),fy=float(fit[1][0]),
                    cx=float(fit[0][1]),cy=float(fit[1][1])),
                fitted_focal_ratio=[float(fit[0][0]/k[0]),float(fit[1][0]/k[1])],
                fitted_principal_error_px=[float(fit[0][1]-k[2]),float(fit[1][1]-k[3])],
                fitted_reprojection_rmse_px=[float(x) for x in residual])


def cached_audit(campaign, output):
    root = campaign / 'tapvid16_pstudio_full_eval'
    plan = json.loads((root/'plan.json').read_text())
    official = load_module('_pstudio_diagnostic_scorer', Path(__file__).parent/'references/tapvid3d_metrics.py')
    cases = defaultdict(list)
    for case in plan['cases']:
        cases[case['sequence']].append(case)
    methods = {}
    for method in ['multid4rt', 'omnix']:
        report = json.loads((root/method/'results.json').read_text())
        records = {r['id']:r for r in report['records']}
        rows = []
        for sequence, clips in cases.items():
            with np.load(clips[0]['path'],allow_pickle=True) as data:
                tracks = data['tracks_XYZ'].astype(np.float32)
                visibility = data['visibility'].astype(bool)
                raw_k = data['fx_fy_cx_cy'].astype(np.float64)
            for case in clips:
                r = records[case['id']]
                paths = list((root/method).glob(f'worker*/{case["id"]}.npz'))
                if len(paths) != 1:
                    raise ValueError('Cached inference missing or duplicated')
                with np.load(paths[0],allow_pickle=False) as data:
                    pred = data['tracks_XYZ'].astype(np.float64)
                    np.testing.assert_array_equal(data['point_ids'],case['point_ids'])
                start = case['start']; ids = case['point_ids']
                truth = tracks[start:start+16,ids]; visible = visibility[start:start+16,ids]
                raw_xy = truth[0,:,:2]/truth[0,:,2:3]*raw_k[:2]+raw_k[2:]
                rh,rw = r['input']['raw_hw']; h,w = r['input']['content_hw']
                xy = (raw_xy+.5)*[w/rw,h/rh]-.5
                k = np.r_[raw_k[:2]*[w/rw,h/rh],(raw_k[2:]+.5)*[w/rw,h/rh]-.5]
                scale = r['metrics']['global_scale']
                stats = initial_diagnostics(pred,truth,xy,k,scale)
                hp,wp=r['input']['padded_hw']
                stats['raw_query_outside_image_fraction']=float(np.mean((raw_xy[:,0]<0)|(raw_xy[:,0]>rw-1)|(raw_xy[:,1]<0)|(raw_xy[:,1]>rh-1)))
                stats['prepared_query_outside_field_fraction']=float(np.mean((xy[:,0]<0)|(xy[:,0]>wp-1)|(xy[:,1]<0)|(xy[:,1]>hp-1)))
                scaled = pred*scale
                anchored = scaled + truth[:1] - scaled[:1]
                apd = official.compute_tapvid3d_metrics(~visible,truth,np.zeros_like(visible),
                    anchored,metric_intrinsics(raw_k,r['input']['raw_hw']),order='t n',scaling='none')['average_pts_within_thresh']
                stats.update(id=case['id'],sequence=sequence,
                    apd3d=r['metrics']['apd3d'],epe_m=r['metrics']['epe_m'],
                    oracle_initial_point_apd3d=float(apd),
                    oracle_initial_point_epe_m=float(np.linalg.norm(anchored-truth,axis=-1)[visible].mean()))
                rows.append(stats)
        numeric = [k for k,v in rows[0].items() if isinstance(v,float)]
        means = {k:float(np.mean([r[k] for r in rows])) for k in numeric}
        for k in ['fitted_focal_ratio','fitted_principal_error_px','fitted_reprojection_rmse_px']:
            means[k] = np.mean([r[k] for r in rows],axis=0).tolist()
            means[k+'_median'] = np.median([r[k] for r in rows],axis=0).tolist()
        methods[method] = dict(clips=len(rows),sequences=len(cases),mean=means,records=rows)
        print(method,json.dumps(means),flush=True)
    write_json(output/'cached_decomposition.json',dict(stage='passed',methods=methods,
        caveat='Oracle source intrinsics/depth/3D anchoring are diagnostics with supplied GT; not model performance. Intrinsics fit must be checked against captured heads.'))


def prepare_variant(rgb, variant):
    images, meta = prepare_rgb(rgb)
    if variant == 'standard':
        meta['query_affine_scale'] = [meta['content_hw'][1]/meta['raw_hw'][1],meta['content_hw'][0]/meta['raw_hw'][0]]
        meta['query_affine_offset'] = [(v-1)/2 for v in meta['query_affine_scale']]
        return images, meta
    rh,rw = meta['raw_hw']
    array = torch.from_numpy(np.stack(rgb).copy()).permute(0,3,1,2).float()/255
    if variant == 'native_resolution':
        h,w = rh,rw
        hp,wp = ((v+13)//14*14 for v in [h,w])
        result = F.pad(array,(0,wp-w,0,hp-h))
        scale, offset = [1,1],[0,0]
    elif variant == 'patch_resize':
        h,w = 280,504
        hp,wp = h,w
        result = F.interpolate(array,size=(h,w),mode='bilinear',align_corners=False)
        scale = [w/rw,h/rh]; offset = [(v-1)/2 for v in scale]
    else:
        raise ValueError(variant)
    meta.update(content_hw=[h,w],padded_hw=[hp,wp],variant=variant,
                query_affine_scale=scale,query_affine_offset=offset)
    return result[None],meta


@torch.inference_mode()
def capture(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device); torch.set_num_threads(4)
    if args.method == 'multid4rt' and args.allow_latest:
        from multid4rt import MultiD4RT
        from multid4rt.checkpoint import config_from_checkpoint
        data = torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
        config = replace(config_from_checkpoint(data),gradient_checkpointing=False)
        model = MultiD4RT(config)
        model.load_state_dict(data['model'],strict=True)
        loading = dict(step=data['step'],config=config.to_dict(),strict=True,
                       checkpoint=args.checkpoint,checkpoint_sha256=digest(args.checkpoint))
        del data
        model = model.to(device).eval().requires_grad_(False)
    else:
        model,loading = load_model(args.method,args.checkpoint,device,args.omnix_runner)
    plan = json.loads((args.campaign/'tapvid16_pstudio_full_eval/plan.json').read_text())
    chosen = [plan['cases'][i] for i in [0,318,636,953]]
    if args.all_clips:
        chosen = plan['cases']
    rows=[]
    official = load_module('_capture_tapvid_metrics',Path(__file__).parent/'references/tapvid3d_metrics.py')
    write_json(args.output/'capture_contract.json',dict(loading=loading,cases=[c['id'] for c in chosen],variants=args.variants))
    for case in chosen:
        with np.load(case['path'],allow_pickle=True) as data:
            s=case['start']; ids=case['point_ids']
            rgb=[np.asarray(Image.open(io.BytesIO(bytes(x))).convert('RGB')) for x in data['images_jpeg_bytes'][s:s+16]]
            truth=data['tracks_XYZ'][s:s+16,ids].astype(np.float32)
            visible=data['visibility'][s:s+16,ids].astype(bool)
            k=data['fx_fy_cx_cy'].astype(np.float64)
        raw_xy=truth[0,:,:2]/truth[0,:,2:3]*k[:2]+k[2:]
        for variant in args.variants:
            began=time.monotonic()
            images,meta=prepare_variant(rgb,variant); images=images.to(device)
            xy=torch.as_tensor(raw_xy*np.array(meta['query_affine_scale'])+meta['query_affine_offset'],device=device,dtype=torch.float32)
            times=torch.arange(s,s+16,device=device)[None]
            with torch.autocast(device.type,dtype=torch.bfloat16):
                if args.method=='multid4rt':
                    enc=model.encode(images,torch.zeros_like(times),times)
                    out=model.decode(enc,query_time_idx=times,source_obs_idx=times.new_tensor([[0]]),return_visibility=False)
                    cam=enc.geometry['c2w'][0]; intrinsic=enc.geometry['K'][0]
                    depth=enc.geometry['depth'][0]
                else:
                    t=torch.arange(16,device=device)
                    out=model(dict(image=images,image_info=torch.stack((t,torch.zeros_like(t),t,t),-1)[None]))
                    cam=out['camera_pose'][0]; intrinsic=out['intrinsics'][0]
                    depth=out['depth'][0]
                trajectory=out['trajectory'][0,0].permute(0,3,1,2)
            with torch.autocast(device.type,enabled=False):
                world=sample(trajectory,xy)
                local=torch.einsum('tji,tnj->tni',cam[:,:3,:3].float(),world-cam[:,None,:3,3].float())
                if depth.ndim==4 and depth.shape[-1]==1: depth=depth[...,0]
                sampled_depth=sample(depth[:,None],xy)[...,0]
            pred=local.float().cpu().numpy()
            metrics=score(official,pred,truth,visible,k,meta['raw_hw'])
            arrays=dict(tracks_XYZ=pred,point_ids=np.array(ids),K=intrinsic.float().cpu().numpy(),
                        c2w=cam.float().cpu().numpy(),sampled_depth=sampled_depth.cpu().numpy(),
                        source_depth=depth[0].float().cpu().numpy(),query_xy=xy.cpu().numpy())
            np.savez_compressed(args.output/f'{case["id"]}_{variant}.npz',**arrays)
            row=dict(id=case['id'],variant=variant,metrics=metrics,input=meta,seconds=time.monotonic()-began,
                     K0=arrays['K'][0].tolist(),source_z_depth_max_difference=float(np.max(np.abs(pred[0,:,2]-arrays['sampled_depth'][0]))))
            rows.append(row); print(json.dumps(row),flush=True)
            write_json(args.output/'capture_results.json',dict(stage='running',loading=loading,records=rows))
            del out,trajectory,images
            if args.method=='multid4rt': del enc
    write_json(args.output/'capture_results.json',dict(stage='passed',loading=loading,records=rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['cached','multid4rt','omnix'],default='cached')
    p.add_argument('--checkpoint');p.add_argument('--omnix-runner');p.add_argument('--allow-latest',action='store_true')
    p.add_argument('--device',default='cuda:0');p.add_argument('--all-clips',action='store_true')
    p.add_argument('--variants',nargs='+',choices=['standard','native_resolution','patch_resize'],default=['standard'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.method=='cached': cached_audit(args.campaign,args.output)
    else: capture(args)


if __name__=='__main__': main()
