"""Compare raw/p98/range+p98 on original TartanAir labels and replay spikes."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torch
import yaml

from multicamera4rc.datasets.adapters.dense_geometry import TartanAirV1ZipAdapter, depth_tail_support


def cases(config, per_environment):
    for s in config['sources']:
        if s['adapter'] != 'tartanair_v1_zip': continue
        adapter = TartanAirV1ZipAdapter(s['root'], cache_dir=s['options'].get('cache_dir'), camera=s['options']['camera'])
        names = sorted(json.loads(Path(s['manifest']).read_text())['sequences'])
        chosen = list(names)
        if per_environment:
            selected = Counter(); chosen = []
            for name in names:
                env = adapter.records[name]['physical_scene']+'/'+Path(adapter.records[name]['depth_zip']).parent.name
                if selected[env] < per_environment:
                    selected[env] += 1; chosen.append(name)
        for name in chosen:
            length = adapter.get_num_frames(name); start = max(0, length//2-4)
            yield s['name'], adapter, name, list(range(start,min(start+8,length)))


def point_ranges(clip):
    E = np.asarray(clip.extrinsics,dtype=np.float64)
    C = E[0][None] @ np.linalg.inv(E)
    output=[]
    for i, d in enumerate(clip.depths):
        h,w=d.shape; y,x=np.indices((h,w)); uv=np.stack((x,y,np.ones((h,w))),-1)
        ray=(uv @ np.linalg.inv(clip.intrinsics[i]).T) @ C[i,:3,:3].T
        depth=np.where(np.isfinite(d),d,0)
        output.append(np.linalg.norm(ray*depth[...,None]+C[i,:3,3],axis=-1))
    return output


def compare(clip):
    ranges=point_ranges(clip); results={}
    for policy, options in [('raw',{}),('p98_only',dict(percentile=98.)),
            ('1000m_p98',dict(depth_max=1000.,percentile=98.)),
            ('200m_p98',dict(depth_max=200.,percentile=98.))]:
        masks=[]; stats=[]
        for d in clip.depths:
            mask,s=depth_tail_support(d,**options); masks.append(mask);stats.append(s)
        values=np.concatenate([d[m] for d,m in zip(clip.depths,masks)])
        radius=np.concatenate([r[m] for r,m in zip(ranges,masks)])
        if not len(values):
            results[policy]=dict(valid_pixels=0,empty=True,frames=stats);continue
        scale=float(radius.mean())
        results[policy]=dict(valid_pixels=len(values),retained_fraction=len(values)/sum(d.size for d in clip.depths),
            raw_depth_quantiles=np.quantile(values,[0,.01,.5,.98,1]).tolist(),
            scene_scale=scale, normalized_depth_quantiles=np.quantile(values/scale,[0,.01,.5,.98,1]).tolist(),
            normalized_point_radius_max=float(radius.max()/scale),frames=stats)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-config',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-environment',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(2)
    config=yaml.safe_load(Path(args.data_config).read_text())
    report=dict(data_config=args.data_config,samples=[],replayed_spikes=[])
    for source,adapter,name,times in cases(config,args.per_environment):
        clip=adapter.load_clip(name,times)
        result=dict(source=source,sequence=name,times=times,policies=compare(clip))
        report['samples'].append(result)
        print(json.dumps(dict(source=source,sequence=name,done=len(report['samples']),
            old_scale=result['policies']['raw'].get('scene_scale'),new_scale=result['policies']['200m_p98'].get('scene_scale'))),flush=True)
    root=Path(config['sources'][0]['options']['cache_dir']).parent
    spike_path=root/'tartanair_spike_gt_check_20261002.json'
    spikes=json.loads(spike_path.read_text()) if spike_path.exists() else []
    adapter=TartanAirV1ZipAdapter('/data8/mugen/TartanAir_V1',camera='right',cache_dir=str(root/'index_cache'))
    for r in spikes:
        name=r['clip_id'][0].split('/',1)[1]; times=r['time_idx'][0]
        report['replayed_spikes'].append(dict(step=r['step'],sequence=name,times=times,policies=compare(adapter.load_clip(name,times))))
    report['passed']=bool(report['samples']) and all(r['policies']['200m_p98'].get('valid_pixels',0)>0 for r in report['samples'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],samples=len(report['samples']),spikes=len(report['replayed_spikes']))),flush=True)

if __name__=='__main__':main()
