"""Audit variable layouts against native GT, optionally run giant BF16 backward."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.train import _TrainingStep, _sources
from multicamera4rc.geometry import unproject_depth


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n"); temp.replace(path)


def find_case(data, name, C, T, sequence=None, camera_suffix=None, gap=None):
    si = next(i for i,s in enumerate(data.sources) if s["display_name"] == name)
    begin = data.quota_ends[si-1] if si else 0
    for index in range(begin, data.quota_ends[si]):
        p = data.sample_plan(index)
        if (p["cameras"], p["num_times"]) != (C,T): continue
        if sequence and p["sequence"] != sequence: continue
        if camera_suffix and not p["sequence"].endswith(camera_suffix): continue
        if gap and p["times"][1]-p["times"][0] != gap: continue
        return dict(index=index, source=name, cameras=C, times=T, sequence=p["sequence"], frames=p["times"])
    raise RuntimeError(f"no sampled case {name} C{C}T{T} {sequence} {camera_suffix} gap{gap}")


def cases(datasets):
    a,b,c = datasets
    rows=[dict(stage=0, **find_case(a,"kubric_variable",1,16))]
    rows += [dict(stage=0, **find_case(a,"kubric_variable",C,4)) for C in range(5,11)]
    rows += [dict(stage=1, **find_case(b,"kubric_variable",4,12)),
             dict(stage=1, **find_case(b,"kubric_variable",8,6))]
    rows += [dict(stage=2, **find_case(c,"kubric_variable",C,T)) for C,T in [(1,32),(2,24),(3,16)]]
    rows += [dict(stage=2, **find_case(c,"pointodyssey",1,24,sequence=s,gap=4)) for s in ["ani","character"]]
    rows.append(dict(stage=0, **find_case(a,"pointodyssey",1,16,sequence="ani")))
    rows += [dict(stage=2, **find_case(c,"dynamic_replica",1,16,camera_suffix=s)) for s in ["left","right"]]
    rows.append(dict(stage=0, **find_case(a,"dynamic_replica",1,12,camera_suffix="left")))
    for name,C,T in [("vkitti2_Camera_0",1,24),("vkitti2_Camera_1",1,24),("vkitti2_stereo",2,12),
                     ("tartanair_left",1,32),("tartanair_v1_stereo",2,12),
                     ("tartanair_v2_front_mono",1,32),
                     ("scannet",1,32),("scannetpp",1,32),("wildrgbd",1,32)]:
        rows.append(dict(stage=2, **find_case(c,name,C,T)))
    for direction in ("front","back","left","right","top","bottom"):
        rows.append(dict(stage=2, **find_case(c,"tartanair_v2_"+direction,2,12)))
    return rows


def check_native(data, row, device):
    plan=data.sample_plan(row["index"]); source=plan["source"]
    sample=data[row["index"]]; batch=collate_adapter_samples([sample])
    C,T=row["cameras"],row["times"]
    assert sample["images"].shape[0] == C*T
    assert sample["camera_id"].unique().numel() == C
    assert sample["query_time_idx"].unique().numel() == T
    pairs=list(zip(sample["camera_id"].tolist(),sample["time_idx"].tolist()))
    assert len(set(pairs)) == C*T
    kw=dict(layout_num_cameras=C,layout_num_times=T,sampling_seed=plan["sample_seed"]) if source["name"] == "kubric_multicam" else {}
    raw=source["adapter"].load_clip(plan["sequence"],plan["times"]*C,**kw)
    for i,image in enumerate(raw.images):
        h,w=image.shape[:2]
        expected=np.asarray(image,np.float32)
        if image.dtype == np.uint8 or expected.max() > 1: expected=expected/255.
        torch.testing.assert_close(sample["images"][i,:,:h,:w],torch.from_numpy(expected).permute(2,0,1),atol=0,rtol=0)
        valid=sample["targets"]["depth_valid"][i,:h,:w]
        torch.testing.assert_close(sample["targets"]["depth"][i,:h,:w][valid],torch.tensor(raw.depths[i])[valid],atol=0,rtol=0)
        torch.testing.assert_close(sample["targets"]["K"][i],torch.tensor(raw.intrinsics[i]).float(),atol=0,rtol=0)
        assert not sample["targets"]["depth_valid"][i,h:,:].any()
        assert not sample["targets"]["depth_valid"][i,:,w:].any()
    particle_truth = None
    if source["dense_kind"] == "pointodyssey_tracks":
        particle_truth = dict(world=raw.trajs_3d_world, first_extrinsic=raw.extrinsics[0],
            visible=raw.visibs, native_world=raw.metadata["has_native_world_trajectories"])
    del raw
    for k,v in batch.items():
        if isinstance(v,torch.Tensor): batch[k]=v.to(device)
    batch["targets"]={k:v.to(device) for k,v in batch["targets"].items()}
    torch.manual_seed(49); selected=_sources(batch,4,max_source_time_pairs=64)
    target=materialize_dense_targets(batch,selected)
    assert torch.allclose(target["c2w"][0,0],torch.eye(4,device=device),atol=2e-6)
    for k,v in target.items():
        if v.is_floating_point(): assert torch.isfinite(v).all(),k
    base=unproject_depth(target["depth"],target["K"],target["c2w"])
    # Independent geometry identity at every decoded source's physical time.
    for q,s in enumerate(selected[0].tolist()):
        slot=int((batch["query_time_idx"][0] == batch["time_idx"][0,s]).nonzero()[0])
        mask=target["trajectory_valid"][0,q,slot]
        torch.testing.assert_close(target["trajectory"][0,q,slot][mask],base[0,s][mask],atol=2e-6,rtol=2e-6)
        assert not target["displacement"][0,q,slot].any()
    if source["dense_kind"] == "geometry":
        assert not target["displacement_valid"].any() and not target["static_mask"].any()
    if source["dense_kind"] in ("replica_flow","vkitti_scene_flow"):
        # No non-adjacent motion target may be inferred from one-step native flow.
        for q,s in enumerate(selected[0].tolist()):
            gap=(batch["query_time_idx"][0]-batch["time_idx"][0,s]).abs()
            assert not target["displacement_valid"][0,q,gap>1].any()
    if particle_truth is not None:
        ids=target["sparse_track_id"][0].cpu().numpy()
        expected=particle_truth["world"][:,ids].transpose(1,0,2).astype(np.float64)
        E=particle_truth["first_extrinsic"]
        expected=expected@E[:3,:3].T+E[:3,3]
        expected/=float(target["scene_scale"][0])
        known=target["sparse_trajectory_valid"][0]
        torch.testing.assert_close(target["sparse_trajectory"][0][known],
            torch.as_tensor(expected,dtype=torch.float32,device=device)[known],atol=3e-5,rtol=2e-5)
        if not particle_truth["native_world"]:
            visible=torch.as_tensor(particle_truth["visible"][:,ids].T,device=device).bool()
            assert not (known & ~visible).any()
    evidence=dict(native_rgb_depth_K_equal=True, native_resolution=True,
        shape=list(batch["images"].shape),camera_id=batch["camera_id"].cpu().tolist(),
        time_idx=batch["time_idx"].cpu().tolist(),query_time_idx=batch["query_time_idx"].cpu().tolist(),
        depth_pixels=int(target["depth_valid"].sum()),motion_point_times=int(target["displacement_valid"].sum()),
        particle_point_times=int(target.get("sparse_trajectory_valid",torch.tensor(0)).sum()),
        particle_provenance=sample.get("native_tracks"),native_source_time_identity=True,
        independent_native_particle_xyz_equal=particle_truth is not None)
    evidence.update(decoded_sources=int(selected.shape[1]),source_time_pairs=int(selected.shape[1])*T)
    return batch,selected,target,evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curriculum",required=True,type=Path)
    p.add_argument("--original-config",required=True)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--mode",choices=("plan","native","giant"),default="native")
    p.add_argument("--checkpoint")
    p.add_argument("--case-sources",nargs="+")
    p.add_argument("--memory-fraction",type=float,default=.5)
    args=p.parse_args();torch.set_num_threads(2)
    plan=json.loads(args.curriculum.read_text());original=create_dataset(args.original_config)
    datasets=[create_dataset(stage["data_config"]) for stage in plan["stages"]]
    report=dict(passed=False,mode=args.mode,stages=[],samples=[])
    args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,report)
    for stage,data in enumerate(datasets):
        for source in data.sources:
            names=[s for s in original.sources if (s["display_name"].startswith("kubric_") if source["display_name"] == "kubric_variable" else s["display_name"] == source["display_name"])]
            assert names and all(set(s["pool"]) <= set(source["pool"]) for s in names)
        hist=Counter();layouts=Counter()
        for i in range(len(data)):
            row=data.sample_plan(i);hist[row["source"]["display_name"]]+=1
            layouts[(row["source"]["display_name"],row["cameras"],row["num_times"])]+=1
            assert len(set(row["times"])) == row["num_times"]
            assert min(row["times"]) >= 0 and max(row["times"]) < row["source"]["adapter"].get_num_frames(row["sequence"])
        report["stages"].append(dict(stage=stage,all_admitted_sequences_preserved=True,source_counts=dict(hist),
            layouts=[dict(source=n,cameras=C,times=T,count=count) for (n,C,T),count in layouts.items()]))
        write(args.output,report);print(json.dumps(dict(stage=stage,source_counts=dict(hist))),flush=True)
    if args.mode != "plan":
        rows=cases(datasets)
        if args.case_sources: rows=[r for r in rows if r["source"] in args.case_sources]
        device="cuda" if args.mode == "giant" else "cpu"
        if device == "cuda":
            if not args.checkpoint: raise ValueError("giant verification requires checkpoint")
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
            checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True,mmap=True)
            cfg=replace(config_from_checkpoint(checkpoint),attention_chunk_size=None,
                        visibility_chunk_size=None,gradient_checkpointing=True)
            model=MultiCamera4RC.from_checkpoint(checkpoint,config=cfg).cuda().train()
            criterion=MultiCamera4RCLoss(**checkpoint["training_state"]["loss_config"])
            del checkpoint;stepper=_TrainingStep(model,criterion)
        for row in rows:
            began=time.monotonic()
            if device == "cuda": model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            batch,selected,target,evidence=check_native(datasets[row["stage"]],row,device)
            result=dict(**row,**evidence)
            if device == "cuda":
                for key in ("visibility","sparse_visibility"):
                    if key in target and not target[key+"_valid"].any():target.pop(key);target.pop(key+"_valid")
                visibility="visibility" in target or "sparse_visibility" in target
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    loss,metrics=stepper(batch["images"],batch["camera_id"],batch["time_idx"],batch["query_time_idx"],
                        selected,batch["observation_valid"],target,visibility)
                assert torch.isfinite(loss);loss.backward()
                grads=[v.grad for v in model.parameters() if v.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                motion=any(v.grad is not None and bool(v.grad.abs().sum()>0) for v in model.motion_decoder.parameters())
                if row["source"] in ("kubric_variable","pointodyssey","vkitti2_stereo"):assert motion
                result.update(loss=float(loss),grad_norm=norm,all_gradients_finite=True,
                    motion_decoder_gradient=motion,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    no_query_chunking=True,metrics={k:float(v) for k,v in metrics.items()})
                del loss,metrics,grads;model.zero_grad(set_to_none=True)
            result["elapsed_seconds"]=time.monotonic()-began
            report["samples"].append(result);write(args.output,report);print(json.dumps(result),flush=True)
            del batch,selected,target;gc.collect()
            if device == "cuda":torch.cuda.empty_cache()
    report["passed"]=True;write(args.output,report)
    print(json.dumps(dict(passed=True,samples=len(report["samples"]))),flush=True)


if __name__ == "__main__":main()
