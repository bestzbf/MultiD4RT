"""Evaluate static geometry on fixed, physically held-out multiview clips."""
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples, read_data_config
from multicamera4rc.datasets.registry import create_adapter
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.metrics import METRICS_VERSION
from multicamera4rc.train import _TrainingStep
from scripts.build_full_dense_mixture import calibrated


def validation_dataset(train_config, directory):
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    config=deepcopy(read_data_config(train_config));sources=[];splits={}
    for source in config['sources']:
        if source.get('view_mode')!='static_multiview': continue
        manifest=json.loads(Path(source['manifest']).read_text())
        train_scenes={r['physical_scene'] for r in manifest['sequences'].values()}
        if source['adapter']=='hypersim_zip':
            source['split']='val'
            adapter=create_adapter(source['adapter'],root=source['root'],split='val',**source['options'])
            records={}
            for seq,r in adapter.records.items():
                mask=calibrated(np.repeat(r['K'][None],len(r['E']),axis=0),r['E'],allow_skew=True)
                frames=np.flatnonzero(mask).tolist()
                if not frames: continue
                records[seq]=dict(length=len(frames),physical_scene=r['physical_scene'])
                if len(frames)!=len(mask):records[seq]['valid_frames']=frames
            del adapter
        else:
            # These held-out scenes live in the pack's train group. The
            # admitted training manifest already excludes them physically.
            records=manifest['validation_sequences']
        val_scenes={r['physical_scene'] for r in records.values()}
        if not val_scenes or train_scenes & val_scenes:raise ValueError('static validation must be physically disjoint')
        path=directory/(source['name']+'_validation_manifest.json')
        path.write_text(json.dumps(dict(root=source['root'],sequences=records),indent=2)+'\n')
        source['manifest']=str(path);sources.append(source)
        splits[source['name']]=dict(train_scenes=len(train_scenes),val_scenes=len(val_scenes),scene_overlap=0)
    config.update(sources=sources,skip_invalid=False)
    config.pop('invalid_log',None)
    path=directory/'static_validation_native.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
    return create_dataset(path),splits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-config',required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--memory-fraction',type=float,default=.12)
    p.add_argument('--scenes-per-source',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(2);args.plan.parent.mkdir(parents=True,exist_ok=True)
    dataset,splits=validation_dataset(args.train_config,args.plan.parent)
    entries={(si,seq):(dataset.coverage_ends[i-1] if i else 0,end)
        for i,((si,seq,_,_),end) in enumerate(zip(dataset.coverage_entries,dataset.coverage_ends))}
    if args.plan.exists():
        plan=json.loads(args.plan.read_text())
    else:
        cases=[]
        for si,s in enumerate(dataset.sources):
            choices=s['camera_choices'];pool=s['pool']
            selected=random.Random('static-heldout:'+s['display_name']).sample(pool,min(len(pool),args.scenes_per_source))
            for seq,n in selected:
                begin,end=entries[si,seq];index=begin+(end-begin-1)//2
                for C in choices:
                    s['camera_choices']=(C,);sample=dataset[index]
                    cases.append(dict(source=si,sequence=seq,index=index,requested_cameras=C,
                        view_indices=sample['static_multiview']['view_indices']))
            s['camera_choices']=choices
        plan=dict(format_version=1,sources=[s['display_name'] for s in dataset.sources],cases=cases)
        args.plan.write_text(json.dumps(plan,indent=2)+'\n')
    assert plan['sources']==[s['display_name'] for s in dataset.sources]
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    report=dict(stage='running',metrics_version=METRICS_VERSION,
        plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
        splits=splits,geometry_only=True,checkpoints={})
    args.output.mkdir(parents=True,exist_ok=True)
    for entry in args.checkpoints:
        label,path=entry.split('=',1)
        payload=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        cfg=replace(config_from_checkpoint(payload),gradient_checkpointing=False,
            attention_chunk_size=None,visibility_chunk_size=None)
        model=MultiCamera4RC(cfg);model.load_4rc_weights(payload)
        profile=payload.get('training_state',{}).get('loss_config')
        criterion=MultiCamera4RCLoss(**profile) if profile else MultiCamera4RCLoss.dense()
        checkpoint_step=payload['step'];del payload
        step=_TrainingStep(model.to(device).eval(),criterion);records=[]
        for case in plan['cases']:
            s=dataset.sources[case['source']];s['camera_choices']=(case['requested_cameras'],)
            assert case['sequence'] in dict(s['pool'])
            sample=dataset[case['index']]
            assert sample['static_multiview']['view_indices']==case['view_indices']
            assert sample['clip_id'].endswith('/'+case['sequence'])
            batch=collate_adapter_samples([sample])
            for key,value in list(batch.items()):
                if isinstance(value,torch.Tensor):batch[key]=value.to(device)
            batch['targets']={k:v.to(device) for k,v in batch['targets'].items()}
            source=torch.arange(min(4,len(sample['images'])),device=device)[None]
            with torch.inference_mode():
                target=materialize_dense_targets(batch,source)
                assert not target['displacement_valid'].any() and not target['static_mask'].any()
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    loss,metrics=step(batch['images'],batch['camera_id'],batch['time_idx'],batch['query_time_idx'],
                        source,batch['observation_valid'],target,False)
                assert torch.isfinite(loss)
            record=dict(source=s['display_name'],sequence=case['sequence'],actual_cameras=len(sample['images']),
                input_shape=list(batch['images'].shape),view_indices=case['view_indices'],
                metrics={k:float(v) for k,v in metrics.items()})
            records.append(record);print(json.dumps(dict(checkpoint=label,**record)),flush=True)
            del sample,batch,target,loss,metrics
        grouped=defaultdict(list)
        for record in records:grouped[record['source']].append(record['metrics'])
        means={name:{k:sum(r[k] for r in rows)/len(rows) for k in rows[0]} for name,rows in grouped.items()}
        by_scene={}
        for record in records:
            by_scene.setdefault(record['source'],{}).setdefault(record['sequence'],[]).append(record)
        by_scene={name:{sequence:dict(clips=len(rows),
            camera_counts=[r['actual_cameras'] for r in rows],
            metrics={k:sum(r['metrics'][k] for r in rows)/len(rows) for k in rows[0]['metrics']})
            for sequence,rows in scenes.items()} for name,scenes in by_scene.items()}
        report['checkpoints'][label]=dict(path=path,step=checkpoint_step,clips=len(records),
            model_config=cfg.to_dict(),loss_config=criterion.to_config(),amp_dtype='bfloat16' if device.type=='cuda' else 'float32',
            by_source=means,by_scene=by_scene,records=records)
        (args.output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        del step,model,criterion;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    report['stage']='passed';(args.output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(stage='passed',plan_sha256=report['plan_sha256'])),flush=True)


if __name__=='__main__':main()
