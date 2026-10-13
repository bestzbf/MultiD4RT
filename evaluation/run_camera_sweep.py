"""Fixed-query C4/C6/C8/C10 diagnostic, separate from full MVTracker tables."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import digest,load_model,write_json
from .multiview_adapter import TrackingInputs,predict_official_queries
from .mvtracker.official import OfficialMVTracker
from .mvtracker.plan import cache_for_scene,read_plan
from .mvtracker.protocols import PROTOCOLS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['multid4rt','omnix'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/data6/benchmarks'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--omnix-runner',type=Path)
    p.add_argument('--parity-report',type=Path)
    args=p.parse_args()
    torch.set_num_threads(4)
    device=torch.device(args.device)
    official=OfficialMVTracker(args.root/'code/mvtracker')
    plan=read_plan(args.plan,args.root)
    protocol=PROTOCOLS['mv-kubric']
    dataset=official.dataset(protocol,args.root/'datasets/mvtracker')
    model,report=load_model(args.method,args.checkpoint,device,args.omnix_runner)
    if args.method=='omnix':
        from .omnix_queries import OmniXQueries
        proof=json.loads(args.parity_report.read_text())
        if proof['stage']!='passed' or proof['checkpoint_sha256']!=report['checkpoint_sha256'] or proof['adapter_sha256']!=digest(Path(__file__).with_name('omnix_queries.py')):
            raise ValueError('Invalid native release-output parity')
        model=OmniXQueries(model)
        report['query_memory_adapter']=model.provenance
    rows=[]
    args.output.mkdir(parents=True,exist_ok=True)
    for name in ['1','15','30']:
        scene=next(s for s in plan['scenes'] if s['profile']=='mv-kubric' and s['sequence']==name)
        cache=cache_for_scene(plan,scene)
        labels=official.cached_labels(cache,protocol)
        query=labels['query_points']
        ids=np.flatnonzero((query[:,0]==0)&labels['visible'][1:8].any(axis=0))
        if not len(ids): raise ValueError('Fixed first-frame query pool is empty')
        # This is an unchanged upstream raw reader, including its calibration
        # conversion and projection check. None of its depth/track fields are
        # retained in the thin TrackingInputs contract consumed by inference.
        raw=dataset.getitem_raw_datapoint(args.root/'datasets/mvtracker/kubric-multiview/test'/name)
        views=raw['views']
        video=torch.stack([v['rgba'][:8,...,:3].permute(0,3,1,2).float() for v in views])
        k=torch.stack([v['intrinsics'].float()[None].expand(8,-1,-1) for v in views])
        w2c=torch.stack([v['extrinsics'][:8].float() for v in views])
        del raw,views
        original,valid=dataset[dataset.seq_names.index(name)]
        if not valid or not torch.equal(video[:4],original.video[:,:8]) or not torch.equal(k[:4],original.intrs[:,:8]) or not torch.equal(w2c[:4],original.extrs[:,:8]):
            raise ValueError('Expanded-view RGB/calibration does not match the official four-view reader')
        del original
        selected=dict(query_points=query[ids],tracks=labels['tracks'][:8,ids],
                      visible=labels['visible'][:8,ids],track_indices=labels['track_indices'][ids],
                      track_upscaling_factor=labels['track_upscaling_factor'])
        query_hash=hashlib.sha256(query[ids].tobytes()).hexdigest()
        for c in [4,6,8,10]:
            print(f'Camera diagnostic {name}: C{c}T8 Q{len(ids)}',flush=True)
            inputs=TrackingInputs(video[:c],k[:c],w2c[:c],torch.from_numpy(query[ids]),tuple(range(c)))
            torch.cuda.reset_peak_memory_stats(device)
            tracks,visible,diagnostics=predict_official_queries(model,inputs,device,
                long_side=266,query_time_chunk=32,amp_dtype=torch.bfloat16,source_chunk=4)
            metric=official.score(selected,tracks,visible,protocol)
            row=dict(scene=name,cameras=c,frames=8,queries=len(ids),query_points_sha256=query_hash,
                official_cache_sha256=scene['cache_sha256'],point_ids=ids.tolist(),
                metrics={key:float(value) if np.isfinite(value) else None for key,value in metric.items()},
                diagnostics=diagnostics,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
            rows.append(row)
            np.savez_compressed(args.output/f'{name}_C{c}T8.npz',tracks=tracks,visible=visible,
                query_points=query[ids],point_ids=ids)
            write_json(args.output/'results.json',dict(stage='running',method=args.method,loading=report,
                records=rows,expected=12,input_condition='rgb_calibrated_gauge_and_supplied_3d_queries',
                selection='fixed sequence IDs 1,15,30; all cached first-frame queries with a future visible point in T8',
                visibility='fixed GT union of reference cameras 0,1,2,3 at all camera counts',
                result_kind='camera_count_diagnostic; not the complete official 24-frame benchmark'))
            del inputs,tracks,visible
            torch.cuda.empty_cache()
    result=json.loads((args.output/'results.json').read_text());result['stage']='passed'
    write_json(args.output/'results.json',result)


if __name__=='__main__':main()
