"""Bounded real-Kubric giant forward/backward check for a chosen memory scope.

Select a GPU with CUDA_VISIBLE_DEVICES. This writes evidence, never optimizer
updates or training checkpoints. Run in an isolated worktree when training is
already active in another checkout.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint, read_checkpoint
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples, read_data_config
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.train import _TrainingStep, _sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decoder-memory", choices=("all_observations", "target_time"),
                        default="all_observations")
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--times", type=int, default=6)
    parser.add_argument("--source-count", type=int, default=2)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=.16)
    args = parser.parse_args(argv)
    if min(args.cameras, args.times, args.source_count) <= 0 or args.sample_index < 0:
        parser.error("positive C/T/source count and nonnegative index required")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory-fraction must lie in (0,1]")
    torch.set_num_threads(2)
    torch.manual_seed(53)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    data = read_data_config(args.data_config)
    source = deepcopy(next(s for s in data["sources"] if s["adapter"] == "kubric_multicam"))
    for key in ("layouts", "camera_choices"):
        source.pop(key, None)
    source.update(cameras=args.cameras, num_times=args.times, strides=[1], weight=1.)
    data.update(sources=[source], sampling="weighted", image_mode="native",
                supervision="dense", num_queries=0, include_visibility=True,
                epoch_size=max(1, args.sample_index+1), skip_invalid=False)
    dataset = create_dataset(data)
    batch = collate_adapter_samples([dataset[args.sample_index]])
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    batch["targets"] = {key: value.cuda() for key, value in batch["targets"].items()}
    sources = _sources(batch, args.source_count)
    targets = materialize_dense_targets(batch, sources)
    path = Path(args.checkpoint)
    payload = (read_checkpoint(path) if path.suffix == ".safetensors"
               else torch.load(path, map_location="cpu", weights_only=True, mmap=True))
    cfg = replace(config_from_checkpoint(payload), decoder_memory=args.decoder_memory,
                  attention_chunk_size=None, visibility_chunk_size=None,
                  gradient_checkpointing=True)
    if cfg.embed_dim != 1536:
        raise ValueError("This preflight requires a giant checkpoint")
    model = MultiCamera4RC.from_checkpoint(payload, config=cfg).cuda().train()
    checkpoint_step = payload.get("step")
    del payload
    stepper = _TrainingStep(model, MultiCamera4RCLoss.dense())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, metrics = stepper(batch["images"], batch["camera_id"], batch["time_idx"],
                                batch["query_time_idx"], sources, batch["observation_valid"],
                                targets, True)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite native loss")
    loss.backward()
    torch.cuda.synchronize()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    invalid = [name for name, grad in gradients.items() if not torch.isfinite(grad).all()]
    if invalid:
        raise FloatingPointError(f"non-finite parameter gradients: {invalid}")
    if not gradients:
        raise AssertionError("expected differentiable native supervision")
    if args.cameras > 1 and args.times > 1:
        if not model.motion_decoder.camera_bias.grad.abs().sum() > 0:
            raise AssertionError("expected differentiable multi-camera motion supervision")
    root = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root/"multicamera4rc").rglob("*.py"))}
    report = dict(passed=True, kind="one_native_forward_backward_no_optimizer",
                  torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  checkpoint=str(path), checkpoint_step=checkpoint_step,
                  model_config=cfg.to_dict(), source_sha256=hashes,
                  image_shape=list(batch["images"].shape), source_indices=sources.cpu().tolist(),
                  camera_id=batch["camera_id"].cpu().tolist(), time_idx=batch["time_idx"].cpu().tolist(),
                  clip_id=batch["clip_id"], loss=float(loss.detach()),
                  metrics={key: float(value) for key, value in metrics.items()},
                  gradient_tensors=len(gradients), memory_fraction=args.memory_fraction,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("passed", "image_shape", "loss",
                      "peak_allocated_gib", "peak_reserved_gib", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
