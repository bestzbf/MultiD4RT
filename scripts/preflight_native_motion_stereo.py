"""Giant native-resolution forwards/backwards on verified motion/stereo data."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.train import _TrainingStep, _sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--memory-fraction", type=float, default=.12)
    args = p.parse_args(); torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    data = create_dataset(args.data_config)
    entries = {(si, seq):(data.coverage_ends[i-1] if i else 0, end)
        for i, ((si, seq, _, _), end) in enumerate(zip(data.coverage_entries, data.coverage_ends))}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = replace(config_from_checkpoint(checkpoint),
                  attention_chunk_size=None, visibility_chunk_size=None, gradient_checkpointing=True)
    model = MultiCamera4RC.from_checkpoint(checkpoint, config=cfg).cuda().train(); del checkpoint
    stepper = _TrainingStep(model, MultiCamera4RCLoss.dense())
    report = dict(passed=False, checkpoint=args.checkpoint, data_config=args.data_config,
                  memory_fraction=args.memory_fraction, physical_gpu=4, samples=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for si, s in enumerate(data.sources):
        name = s["display_name"]
        if name == "pointodyssey":
            wanted = ["ani", "character", "seminar_h52_ego2"]
        elif name == "vkitti2_stereo":
            wanted = ["Scene18_clone", "Scene20_clone"]
        elif name == "tartanair_v1_stereo":
            wanted = [s["pool"][0][0], s["pool"][len(s["pool"])//2][0]]
        elif name == "tartanair_v2_front_mono":
            wanted = [next(seq for seq, _ in s["pool"] if seq.split("/")[0] == environment)
                      for environment in ("DesertGasStation", "Ocean")]
        elif name.startswith("tartanair_v2_"):
            wanted = [s["pool"][0][0]]
        else: continue
        for seq in wanted:
            if (si, seq) not in entries: raise ValueError(f"missing preflight sequence {name}/{seq}")
            first, end = entries[si, seq]
            cases.append((name, seq, (first+end-1)//2))
    for name, seq, index in cases:
        began = time.monotonic(); model.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        sample = data[index]; batch = collate_adapter_samples([sample])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k] = v.cuda()
        batch["targets"] = {k:v.cuda() for k, v in batch["targets"].items()}
        torch.manual_seed(49); source = _sources(batch, 4)
        target = materialize_dense_targets(batch, source)
        for key in ("visibility", "sparse_visibility"):
            if key in target and not target[key+"_valid"].any():
                target.pop(key); target.pop(key+"_valid")
        visibility = "visibility" in target or "sparse_visibility" in target
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = stepper(batch["images"], batch["camera_id"], batch["time_idx"],
                batch["query_time_idx"], source, batch["observation_valid"], target, visibility)
        assert torch.isfinite(loss); loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        motion_gradient = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.motion_decoder.parameters())
        if name in ("pointodyssey", "vkitti2_stereo"): assert motion_gradient
        else: assert not motion_gradient
        if name == "pointodyssey": assert metrics["native_track_queries"] > 0 and metrics["sparse_velocity"] > 0
        row = dict(source=name, sequence=seq, input_shape=list(batch["images"].shape),
            time_idx=batch["time_idx"].cpu().tolist(), camera_id=batch["camera_id"].cpu().tolist(),
            decoded_sources=source.cpu().tolist(), loss=float(loss.detach()), metrics={k:float(v) for k, v in metrics.items()},
            grad_norm=norm, all_gradients_finite=True, motion_decoder_receives_gradient=motion_gradient,
            dense_geometry_pixels=int(target["depth_valid"].sum()), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            no_query_chunking=True, native_resolution=True, elapsed_seconds=time.monotonic()-began)
        report["samples"].append(row); args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(row), flush=True)
        del sample, batch, source, target, loss, metrics, grads
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    report["passed"] = True; args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(passed=True, samples=len(report["samples"]))), flush=True)


if __name__ == "__main__": main()
