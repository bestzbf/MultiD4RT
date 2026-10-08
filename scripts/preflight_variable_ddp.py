"""Real resumed AdamW updates across different native C/T and GT on GPU4/5."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

if __package__:
    from .audit_variable_curriculum import check_native, write
else:  # Direct script invocation with the project installed or PYTHONPATH=.
    from audit_variable_curriculum import check_native, write
from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.mixture import create_dataset
from multicamera4rc.distributed import DistributedContext, aggregate_metrics
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.schedule import LearningRatePlan
from multicamera4rc.train import _TrainingStep


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curriculum",required=True,type=Path)
    p.add_argument("--native-audit",required=True,type=Path)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--memory-fraction",type=float,default=.5)
    args=p.parse_args();torch.set_num_threads(2)
    context=DistributedContext.initialize("cuda")
    try:
        assert context.world_size == 2
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,context.device)
        plan=json.loads(args.curriculum.read_text());audit=json.loads(args.native_audit.read_text())
        assert audit["passed"]
        rows=audit["samples"]
        def row(name,C=None,T=None,sequence=None):
            return next(r for r in rows if r["source"] == name and
                (C is None or r["cameras"] == C) and (T is None or r["times"] == T) and
                (sequence is None or sequence in r["sequence"]))
        pairs=[(row("kubric_variable",10,4),row("pointodyssey",1,16,"ani")),
               (row("tartanair_v2_front",2,12),row("kubric_variable",4,12)),
               (row("dynamic_replica",1,12,"left"),row("scannetpp",1,32))]
        checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True,mmap=True)
        cfg=replace(config_from_checkpoint(checkpoint),attention_chunk_size=None,
                    visibility_chunk_size=None,gradient_checkpointing=True)
        model=MultiCamera4RC.from_checkpoint(checkpoint,config=cfg).to(context.device).train()
        backbone,other=[],[]
        for name,v in model.named_parameters():
            (backbone if name.startswith("backbone.pretrained.") and not name.endswith("camera_relation_weight") else other).append(v)
        optimizer=torch.optim.AdamW([dict(params=backbone,lr=1e-6),dict(params=other,lr=1e-5)],weight_decay=.05)
        optimizer.load_state_dict(checkpoint["optimizer"])
        criterion=MultiCamera4RCLoss(**checkpoint["training_state"]["loss_config"])
        saved_step=checkpoint["step"];del checkpoint
        stepper=DistributedDataParallel(_TrainingStep(model,criterion),device_ids=[context.local_rank],find_unused_parameters=True)
        schedule=LearningRatePlan("cosine",25000,50000,[1e-6,1e-5],warmup_steps=500,min_ratio=.1,warmup_start_ratio=.3)
        report=dict(passed=False,checkpoint=args.checkpoint,restored_step=saved_step,world_size=2,updates=[])
        if context.primary:args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,report)
        datasets={}
        for update,pair in enumerate(pairs):
            case=pair[context.rank];stage=case["stage"]
            if stage not in datasets:datasets[stage]=create_dataset(plan["stages"][stage]["data_config"])
            optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();began=time.monotonic()
            batch,selected,target,evidence=check_native(datasets[stage],case,context.device)
            for key in ("visibility","sparse_visibility"):
                if key in target and not target[key+"_valid"].any():target.pop(key);target.pop(key+"_valid")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss,metrics=stepper(batch["images"],batch["camera_id"],batch["time_idx"],batch["query_time_idx"],
                    selected,batch["observation_valid"],target,"visibility" in target or "sparse_visibility" in target)
            assert torch.isfinite(loss);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            schedule.apply(optimizer,25000+update);optimizer.step()
            probe=model.track_head.scratch.output_conv2[2].weight.detach().clone()
            reference=probe.clone();dist.broadcast(reference,src=0);torch.testing.assert_close(probe,reference,atol=0,rtol=0)
            actual=dict(rank=context.rank,source=case["source"],cameras=case["cameras"],times=case["times"],
                native_shape=evidence["shape"],loss=float(loss.detach()),grad_norm=norm,optimizer_step=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.monotonic()-began)
            gathered=[None]*2;dist.all_gather_object(gathered,actual)
            aggregate=aggregate_metrics(context,metrics)
            if context.primary:
                report["updates"].append(dict(update=update,ranks=gathered,metrics={k:float(v) for k,v in aggregate.items()},
                    parameters_equal_after_update=True,no_query_chunking=True))
                write(args.output,report);print(json.dumps(report["updates"][-1]),flush=True)
            del batch,selected,target,loss,metrics;optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        if context.primary:report["passed"]=True;write(args.output,report);print(json.dumps(dict(passed=True,updates=3)),flush=True)
    finally:context.close()


if __name__ == "__main__":main()
