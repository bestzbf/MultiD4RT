"""Real-source GT and forward/backward preflight, with no GT camera inputs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.train import _TrainingStep, _sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-config", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu"); p.add_argument("--checkpoint")
    p.add_argument("--network", choices=("none", "tiny", "giant"), default="none")
    p.add_argument("--sources", nargs="*"); p.add_argument("--sequences-per-source", type=int, default=3)
    p.add_argument("--sequences", nargs="*", help="Optional exact sequence names for convention-specific checks")
    p.add_argument("--source-count", type=int, default=4); p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--memory-fraction", type=float, default=.7)
    args = p.parse_args(); torch.set_num_threads(args.cpu_threads)
    dataset = create_dataset(args.data_config); device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda": torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    model = None
    if args.network != "none":
        if args.checkpoint:
            model = MultiCamera4RC.from_checkpoint(args.checkpoint)
            model.config = replace(model.config, attention_chunk_size=None, visibility_chunk_size=None)
            for module in model.modules():
                if hasattr(module, "query_chunk_size"): module.query_chunk_size = None
            model.motion_decoder.attention_chunk_size = None
            model.visibility_head.query_chunk_size = None
        else:
            cfg = MultiCamera4RCConfig.tiny(attention_chunk_size=None, visibility_chunk_size=None)
            if args.network == "giant": cfg = MultiCamera4RCConfig(attention_chunk_size=None, visibility_chunk_size=None)
            model = MultiCamera4RC(cfg)
        model = model.to(device).train()
        step = _TrainingStep(model, MultiCamera4RCLoss.dense())
    report = dict(data_config=args.data_config, device=str(device), network=args.network,
        coverage_windows=dataset.coverage_size, epoch_size=len(dataset), samples=[], skipped=[])
    for source_index, source in enumerate(dataset.sources):
        name = source["display_name"]
        if args.sources and name not in args.sources: continue
        pool = source["pool"]
        selected = sorted(set(round(i*(len(pool)-1)/max(1, args.sequences_per_source-1))
                              for i in range(args.sequences_per_source)))
        if args.sequences:
            selected = [i for i, (sequence, _) in enumerate(pool) if sequence in args.sequences]
        for position in selected:
            sequence = pool[position][0]
            entry = next(i for i, (si, seq, _, _) in enumerate(dataset.coverage_entries) if si == source_index and seq == sequence)
            index = dataset.coverage_ends[entry-1] if entry else 0
            began = time.monotonic()
            sample = dataset[index]
            # Runtime no-depth filtering may replace an invalid frame/window;
            # keep this visible rather than claim this source was tested.
            if sample["dataset_name"] != name:
                report["skipped"].append(dict(source=name, sequence=sequence, reason="empty depth window; runtime log records rejection", replacement=sample["dataset_name"]))
                continue
            batch = collate_adapter_samples([sample])
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor): batch[key] = value.to(device)
            batch["targets"] = {k: v.to(device) for k, v in batch["targets"].items()}
            sources = _sources(batch, args.source_count)
            target = materialize_dense_targets(batch, sources)
            if any(k.startswith("sparse_") for k in target): raise AssertionError("sparse labels leaked into dense preflight")
            for key in ("depth", "ray", "trajectory", "displacement", "K", "c2w"):
                if not torch.isfinite(target[key]).all(): raise AssertionError(f"nonfinite GT {name}/{key}")
            motion = target["displacement_valid"].bool()
            item = dict(source=name, sequence=sequence, actual_clip=sample["clip_id"],
                input_shape=list(batch["images"].shape), camera_count=sample["camera_id"].unique().numel(),
                target_times=batch["query_time_idx"].shape[1], decoded_sources=sources.shape[1],
                depth_pixels=int(target["depth_valid"].sum()), source_pixels=int(target["source_pixel_valid"].sum()),
                motion_pixels=int(motion.sum()), trajectory_pixels=int(target["trajectory_valid"].sum()),
                unknown_motion_is_masked=not motion.any().item() if source["dense_kind"] == "geometry" else None)
            if model is not None:
                model.zero_grad(set_to_none=True)
                if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss, metrics = step(batch["images"], batch["camera_id"], batch["time_idx"],
                        batch["query_time_idx"], sources, batch["observation_valid"], target, "visibility" in target)
                if not torch.isfinite(loss): raise FloatingPointError("preflight loss is not finite")
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError("preflight gradient is not finite")
                item.update(loss=float(loss.detach()), gradients_finite=True,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else None,
                    metrics={k: float(v) for k, v in metrics.items()})
                model.zero_grad(set_to_none=True)
                del loss, metrics
            item["elapsed_seconds"] = time.monotonic()-began
            report["samples"].append(item); print(json.dumps(item), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            del sample, batch, target
            if device.type == "cuda": torch.cuda.empty_cache()
    if not report["samples"]: raise ValueError("no requested sources were checked")
    report["passed"] = True; args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": True, "checked": len(report["samples"]), "skipped": len(report["skipped"])}), flush=True)


if __name__ == "__main__": main()
