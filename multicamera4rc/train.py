"""Supervised training entry point: python -m multicamera4rc.train --help."""

import argparse
from dataclasses import replace
import inspect
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn, distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from . import MultiCamera4RC, MultiCamera4RCConfig
from .checkpoint import config_from_checkpoint, read_checkpoint, save_checkpoint
from .data import MultiCameraManifestDataset, collate_multicamera, select_source_targets
from .losses import MultiCamera4RCLoss
from .metrics import METRICS_VERSION, dense_metrics
from .schedule import LearningRatePlan
from .metadata import build_metadata, gather_observations
from .distributed import (DistributedContext, aggregate_metrics, resolve_amp_dtype,
                          restore_rng)


def _sparse_supervised_queries(targets, index):
    """Prefer sources that carry finite, valid labels rather than padded queries."""
    queries = targets["sparse_queries"][index]
    supervised = torch.zeros(len(queries), dtype=torch.bool, device=queries.device)
    if not len(queries):
        return supervised
    for name in ("trajectory", "displacement", "visibility"):
        key = f"sparse_{name}"
        if key not in targets:
            continue
        truth = targets[key][index]
        vector = name != "visibility"
        finite = torch.isfinite(truth).all(-1) if vector else torch.isfinite(truth)
        mask_key = f"{key}_valid"
        supplied = targets.get(mask_key)
        if supplied is None and name == "displacement":
            supplied = targets.get("sparse_trajectory_valid")
        if supplied is not None:
            mask = supplied[index].bool()
            if vector and mask.shape == truth.shape:
                mask = mask.all(-1)
            finite = finite & torch.broadcast_to(mask, finite.shape)
        supervised |= finite.reshape(len(queries), -1).any(-1)
    if "sparse_query_valid" in targets:
        supervised &= targets["sparse_query_valid"][index].bool()
    return supervised


def _sources(batch, count, max_source_time_pairs=0, prefer_motion=False):
    valid = batch["observation_valid"]
    b, s = valid.shape
    if max_source_time_pairs:
        times = batch["query_time_idx"].shape[1]
        budget_count = max(1, max_source_time_pairs // times)
        count = min(count or s, budget_count)
    if count == 0:
        return torch.arange(s, device=valid.device)[None].expand(b, -1)
    count = min(count, int(valid.sum(1).min()))
    rows = []
    for index in range(b):
        candidates = valid[index].nonzero(as_tuple=False).flatten()
        labels = candidates[:0]
        prioritized = "sparse_queries" in batch["targets"]
        if prioritized:
            queries = batch["targets"]["sparse_queries"][index]
            active = _sparse_supervised_queries(batch["targets"], index)
            height, width = batch["images"].shape[-2:]
            active &= (torch.isfinite(queries).all(-1)
                       & (queries[:, 0] == queries[:, 0].round())
                       & (queries[:, 1] >= 0) & (queries[:, 1] <= width-1)
                       & (queries[:, 2] >= 0) & (queries[:, 2] <= height-1))
            labels = queries[active, 0].long().unique()
            labels = labels[(labels >= 0) & (labels < s)]
            labels = labels[valid[index, labels]]
        if prefer_motion and "dense_pair_valid" in batch["targets"]:
            known = batch["targets"]["dense_pair_valid"][index].bool().flatten(1).any(1)
            labels = torch.cat((labels, (known & valid[index]).nonzero().flatten())).unique()
            prioritized = True
        if prefer_motion and "dense_motion_source_valid" in batch["targets"]:
            known = batch["targets"]["dense_motion_source_valid"][index].bool()
            labels = torch.cat((labels, (known & valid[index]).nonzero().flatten())).unique()
            prioritized = True
        if prioritized:
            labels = labels[torch.randperm(len(labels), device=labels.device)]
            rest = candidates[~torch.isin(candidates, labels)]
            rest = rest[torch.randperm(len(rest), device=rest.device)]
            candidates = torch.cat((labels, rest))
        else:
            candidates = candidates[torch.randperm(len(candidates), device=candidates.device)]
        rows.append(candidates[:count])
    return torch.stack(rows)


class _CursorBatchSampler(BatchSampler):
    """Skip sampler indices before dataset reads, preserving the epoch order."""
    start_batch = 0

    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            if index >= self.start_batch:
                yield batch

    def __len__(self):
        return max(0, super().__len__()-self.start_batch)


def _resume_data_cursor(state, step, dataset_size, world_size, batch_size, *, reset=False):
    """Preserve the consumed permutation prefix when the number of ranks changes.

    DistributedSampler partitions one shared permutation by rank. A saved
    batch cursor therefore represents old_world * old_batch samples, not the
    optimizer step count (which may span earlier data configurations).
    """
    if reset:
        return 0, 0
    old_world = int(state.get("world_size", world_size))
    old_batch = int(state.get("batch_size", batch_size))
    if min(dataset_size, old_world, old_batch, world_size, batch_size) <= 0:
        raise ValueError("dataset and training layout sizes must be positive")
    if state.get("dataset_size", dataset_size) != dataset_size:
        raise ValueError("dataset size changed; use --reset-data-cursor for a new data configuration")
    old_samples = (dataset_size + old_world - 1) // old_world
    old_batches = (old_samples + old_batch - 1) // old_batch
    epoch = int(state.get("data_epoch", step // old_batches))
    cursor = int(state.get("batches_consumed", step % old_batches))
    if epoch < 0 or not 0 <= cursor <= old_batches:
        raise ValueError("checkpoint has an invalid training data cursor")
    if cursor == old_batches:
        return epoch + 1, 0
    if (old_world, old_batch) == (world_size, batch_size):
        return epoch, cursor
    if old_world == 1 or world_size == 1:
        # The single-process RandomSampler uses a different RNG stream.
        raise ValueError("single/distributed sampler migration requires --reset-data-cursor")
    consumed = cursor * old_world * old_batch
    new_global_batch = world_size * batch_size
    if consumed % new_global_batch:
        raise ValueError("saved sample cursor is not aligned with the new global batch; "
                         "resume at an aligned checkpoint or use --reset-data-cursor")
    return epoch, consumed // new_global_batch


def _criterion(args, dense_supervision, training_state):
    profile = args.loss_profile
    if args.loss_config:
        path = Path(args.loss_config)
        if path.suffix.lower() == ".json":
            config = json.loads(path.read_text())
        else:
            import yaml
            config = yaml.safe_load(path.read_text())
        return MultiCamera4RCLoss(**config), "custom"
    if profile == "auto" and "loss_config" in training_state:
        return MultiCamera4RCLoss(**training_state["loss_config"]), training_state.get("loss_profile", "custom")
    if profile in ("dense", "dense_balanced") and not dense_supervision:
        raise ValueError("dense loss profiles require renderer-derived full-pixel supervision")
    if profile == "dense_balanced":
        return MultiCamera4RCLoss.balanced_dense(), profile
    return (MultiCamera4RCLoss.dense(), "dense") if dense_supervision else (MultiCamera4RCLoss(), "default")


class _TrainingStep(nn.Module):
    """Return the actual loss graph so DDP can identify unused task heads.

    Wrapping the prediction-only model would expose ray/visibility outputs to
    DDP even when no corresponding GT participates in backward. Restricting
    this wrapper's differentiable outputs to total loss avoids that mismatch.
    """

    def __init__(self, model, criterion):
        super().__init__()
        self.model, self.criterion = model, criterion

    def forward(self, images, camera_id, time_idx, query_time_idx, source,
                observation_valid, targets, need_visibility):
        geometry_only = ("displacement_valid" in targets and not targets["displacement_valid"].any()
                         and not targets.get("sparse_displacement_valid", torch.zeros((), device=images.device)).any()
                         and not targets.get("sparse_trajectory_valid", torch.zeros((), device=images.device)).any()
                         and not need_visibility and not targets.get("static_mask", torch.zeros((), device=images.device)).any())
        if geometry_only:
            # A geometry-only sample has no temporal labels. Train geometry
            # on every observation and source base point; avoid an expensive
            # motion forward whose outputs would receive no gradient.
            scene = self.model.encode(images, camera_id, time_idx, observation_valid)
            meta = build_metadata(scene.camera_id, scene.time_idx, query_time_idx, source, scene.observation_valid)
            base = gather_observations(scene.geometry["base_points"], meta.source_index)
            identity = (meta.source_times[..., None] == meta.query_times[:, None]) & meta.source_valid[..., None]
            xyz = torch.where(identity[..., None, None, None], base[:, :, None], 0)
            prediction = dict(scene.geometry)
            prediction.update(trajectory=xyz, displacement=torch.zeros_like(xyz),
                source_obs_idx=meta.source_index, source_valid=meta.source_valid,
                source_time_idx=meta.source_times, query_time_idx=meta.query_times,
                camera_id=scene.camera_id, time_idx=scene.time_idx,
                observation_valid=scene.observation_valid)
        else:
            prediction = self.model(images, camera_id, time_idx, query_time_idx, source,
                                    observation_valid=observation_valid,
                                    return_visibility=need_visibility)
        terms = self.criterion(prediction, targets)
        if "scene_scale" in targets and "trajectory_valid" in targets and "trajectory" in targets:
            terms.update(dense_metrics(prediction, targets))
        return terms["loss"], {name: value.detach() for name, value in terms.items()}


def _optimizer_step(loss, model, optimizer, scaler, clip_grad):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # FP16 overflow is handled by GradScaler's standard skipped step. A hard
    # clipping error would abort before it can lower the scale. Unscaled/BF16
    # gradients still fail promptly instead of silently updating with NaNs.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad,
                                         error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
    return norm, skipped


def _reset_absolute_time_projections(model, optimizer):
    """Explicit migration only: retain relative-time weights and all other Adam states."""
    if model.config.time_origin != 'clip_start':
        raise ValueError('time projection reset requires time_origin=clip_start')
    names = ('backbone.frame_projection.weight', 'target_projection.weight')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            parameter = parameters[name]
            parameter.zero_()
            optimizer.state.pop(parameter, None)
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    data_input = parser.add_mutually_exclusive_group(required=True)
    data_input.add_argument("--manifest", help="Prepared JSON/JSONL observations and tensor targets")
    data_input.add_argument("--data-config", help="Native adapter mixture JSON/YAML configuration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", choices=("giant", "tiny"), default="giant")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--pretrained", help="Original 4RC .pth/.safetensors; requires matching giant preset")
    weights.add_argument("--resume", help="Native model+optimizer training checkpoint")
    parser.add_argument("--reset-data-cursor", action="store_true",
                        help="With --resume, restart sampling at epoch/batch zero for a new data configuration")
    parser.add_argument("--partial-pretrained", action="store_true", help="Report, rather than reject, missing original modules")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Local batch size per process; torchrun multiplies it by WORLD_SIZE")
    parser.add_argument("--source-count", type=int, default=2, help="0 decodes all sources")
    parser.add_argument("--max-source-time-pairs", type=int, default=0,
                        help="Limit decoded reference count by Q*T; 0 disables. All pixels/particles of each selected reference remain supervised")
    parser.add_argument("--prefer-motion-sources", action=argparse.BooleanOptionalAction, default=None,
                        help="Prefer references with valid native pair labels; resume restores its saved policy")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--override-lr", action="store_true", help="With resume, apply CLI learning rates while preserving optimizer moments")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default=None,
                        help="Start an explicit LR phase; otherwise restore the checkpoint schedule")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-start-ratio", type=float, default=None,
                        help="Resume warmup from this fraction of the new peak LR")
    parser.add_argument("--lr-end-step", type=int, default=None,
                        help="End of the LR phase, independent of this curriculum stage's --steps")
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    loss_settings = parser.add_mutually_exclusive_group()
    loss_settings.add_argument("--loss-profile", choices=("auto", "dense", "dense_balanced"), default="auto")
    loss_settings.add_argument("--loss-config", help="JSON/YAML loss constructor settings; checkpointed in full")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto",
                        help="Dtype with --amp: auto prefers CUDA BF16 and uses CPU BF16")
    parser.add_argument("--freeze-pretrained-backbone", action="store_true")
    parser.add_argument("--freeze-geometry", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--decoder-memory", choices=("all_observations", "target_time"),
                        help="Explicit memory scope; resume otherwise preserves checkpoint behavior")
    parser.add_argument("--time-origin", choices=("clip_start", "absolute"),
                        help="Explicit time encoding migration; resume otherwise preserves checkpoint behavior")
    parser.add_argument("--reset-time-projections", action="store_true",
                        help="On resume with clip_start, reset only the two absolute-time projections and their Adam states")
    parser.add_argument("--cpu-threads", type=int, default=4)
    attention = parser.add_mutually_exclusive_group()
    attention.add_argument("--attention-chunk-size", type=int,
                        help="Override attention query chunk size, including native checkpoint resume")
    attention.add_argument("--no-query-chunking", action="store_true",
                           help="Compute whole attention queries and visibility pixels without splitting")
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.save_every, args.cpu_threads) <= 0 or args.source_count < 0 or args.workers < 0 or args.max_source_time_pairs < 0:
        parser.error("steps, batch size, save interval and threads must be positive; source count/workers nonnegative")
    if min(args.lr, args.backbone_lr, args.clip_grad) <= 0 or args.weight_decay < 0:
        parser.error("learning rates and clip-grad must be positive; weight-decay nonnegative")
    if args.attention_chunk_size is not None and args.attention_chunk_size <= 0:
        parser.error("attention-chunk-size must be positive")
    if args.reset_data_cursor and not args.resume:
        parser.error("reset-data-cursor requires --resume")
    if args.override_lr and not args.resume:
        parser.error("override-lr requires --resume")
    if args.reset_time_projections and not args.resume:
        parser.error("reset-time-projections requires --resume")
    context = DistributedContext.initialize(args.device)
    try:
        return _train(args, context)
    finally:
        context.close()


def _train(args, context):
    device = context.device
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.data_config:
        from .datasets.mixture import create_dataset, collate_adapter_samples
        dataset = create_dataset(args.data_config)
        collate_fn = collate_adapter_samples
    else:
        dataset = MultiCameraManifestDataset(args.manifest)
        collate_fn = collate_multicamera
    resume = read_checkpoint(args.resume) if args.resume else None
    training_state = resume.get("training_state", {}) if resume is not None else {}
    prefer_motion_sources = (args.prefer_motion_sources if args.prefer_motion_sources is not None
                             else training_state.get("prefer_motion_sources", False))
    training_seed = int(training_state.get("seed", args.seed))
    sampler = (DistributedSampler(dataset, num_replicas=context.world_size,
                                  rank=context.rank, shuffle=True, seed=training_seed)
               if context.distributed else None)
    loader_generator = torch.Generator()
    if sampler is None:
        sampler = RandomSampler(dataset, generator=loader_generator)
    batch_sampler = _CursorBatchSampler(sampler, args.batch_size, drop_last=False)
    loader_options = {}
    if args.workers and device.type == "cuda":
        # CUDA/NCCL has already initialized in the parent; fork would inherit
        # unsafe CUDA state. Native adapters support spawn-safe H5 reopening.
        loader_options["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, batch_sampler=batch_sampler, generator=loader_generator,
                        num_workers=args.workers, collate_fn=collate_fn,
                        pin_memory=device.type == "cuda", **loader_options)
    if not len(loader):
        raise ValueError("the training dataset supplies no batches")
    batches_per_epoch = len(loader)
    cfg = (MultiCamera4RCConfig.tiny(gradient_checkpointing=not args.no_checkpointing)
           if args.preset == "tiny" else MultiCamera4RCConfig(gradient_checkpointing=not args.no_checkpointing))
    cfg = config_from_checkpoint(resume, fallback=cfg)
    if args.no_checkpointing:
        cfg = replace(cfg, gradient_checkpointing=False)
    if args.decoder_memory is not None:
        cfg = replace(cfg, decoder_memory=args.decoder_memory)
    if args.time_origin is not None:
        cfg = replace(cfg, time_origin=args.time_origin)
    if args.attention_chunk_size is not None:
        cfg = replace(cfg, attention_chunk_size=args.attention_chunk_size)
    if args.no_query_chunking:
        cfg = replace(cfg, attention_chunk_size=None, visibility_chunk_size=None)
    if context.primary:
        print(json.dumps({"model_config": cfg.to_dict()}), flush=True)
    if resume is not None:
        model = MultiCamera4RC.from_checkpoint(resume, config=cfg)
    else:
        model = MultiCamera4RC(cfg)
        if args.pretrained:
            report = model.load_4rc_weights(args.pretrained, strict_base=not args.partial_pretrained)
            if context.primary:
                print(json.dumps({"pretrained": report}), flush=True)
    model = model.to(device).train()
    if args.freeze_pretrained_backbone:
        model.backbone.pretrained.requires_grad_(False)
        for name, parameter in model.backbone.pretrained.named_parameters():
            if name.endswith("camera_relation_weight"):
                parameter.requires_grad_(True)
    if args.freeze_geometry:
        model.head.requires_grad_(False)
        model.cam_dec.requires_grad_(False)
    backbone_parameters, other_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.pretrained.") and not name.endswith("camera_relation_weight"):
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": args.backbone_lr},
        {"params": other_parameters, "lr": args.lr}], weight_decay=args.weight_decay)
    step = 0
    if resume is not None:
        if "optimizer" not in resume:
            raise ValueError("resume checkpoint has no optimizer; use --pretrained for weights-only initialization")
        optimizer.load_state_dict(resume["optimizer"])
        step = int(resume.get("step", 0))
        if args.override_lr:
            for group, lr in zip(optimizer.param_groups, (args.backbone_lr, args.lr)):
                group["lr"] = lr
    time_projection_reset_step = training_state.get('time_projection_reset_step')
    if args.reset_time_projections:
        reset_names = _reset_absolute_time_projections(model, optimizer)
        time_projection_reset_step = step
        if context.primary:
            print(json.dumps(dict(reset_time_projections=reset_names, reset_step=step)), flush=True)
    dense_supervision = getattr(dataset, "supervision", None) == "dense"
    if context.primary and hasattr(dataset, "sources"):
        print(json.dumps({"dataset_plan": {"sampling": dataset.sampling, "epoch_size": len(dataset),
            "coverage_windows": dataset.coverage_size,
            "sources": [{"name": s["display_name"], "sequences": len(s["pool"]),
                "frames": sum(n for _, n in s["pool"]), "cameras": s["cameras"],
                "camera_choices": list(s["camera_choices"]), "view_mode": s["view_mode"],
                "max_times": s["num_times"], "dense_kind": s["dense_kind"],
                "layouts": s.get("layouts", []), "weight": dataset.weights[i],
                "sampling_fraction": dataset.weights[i]/sum(dataset.weights)} for i, s in enumerate(dataset.sources)]}}), flush=True)
    criterion, loss_profile = _criterion(args, dense_supervision, training_state)
    explicit_lr_phase = args.override_lr or any(v is not None for v in
        (args.lr_schedule, args.warmup_steps, args.min_lr_ratio, args.warmup_start_ratio, args.lr_end_step))
    if not explicit_lr_phase and "lr_plan" in training_state:
        lr_plan = LearningRatePlan(**training_state["lr_plan"])
    else:
        lr_plan = LearningRatePlan(mode=args.lr_schedule or "constant", start_step=step,
                    end_step=args.lr_end_step if args.lr_end_step is not None else max(step+1, args.steps),
                    base_lrs=[g["lr"] for g in optimizer.param_groups],
                    warmup_steps=args.warmup_steps or 0,
                    warmup_start_ratio=args.warmup_start_ratio or 0.,
                    min_ratio=.1 if args.min_lr_ratio is None else args.min_lr_ratio)
    if args.lr_end_step is not None and lr_plan.end_step < args.steps:
        raise ValueError("LR phase ends before the requested training target; start an explicit continuation phase")
    if context.primary:
        print(json.dumps({"loss_profile": loss_profile, "loss_config": criterion.to_config(),
                          "lr_plan": lr_plan.to_config(), "resume_step": step,
                          "optimizer_moments_restored": resume is not None}), flush=True)
    amp = args.amp and device.type in ("cuda", "cpu")
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype) if amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda" and amp_dtype == torch.float16)
    if resume is not None and scaler.is_enabled() and resume.get("grad_scaler"):
        scaler.load_state_dict(resume["grad_scaler"])
    training_step = _TrainingStep(model, criterion)
    if context.distributed:
        # Newer PyTorch renamed this option. Retain 2.5 compatibility without
        # triggering its replacement's deprecation warning on newer releases.
        ddp_options = {"forward_sync_buffers": False} if "forward_sync_buffers" in inspect.signature(DistributedDataParallel).parameters else {"broadcast_buffers": False}
        training_step = DistributedDataParallel(
            training_step, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True, gradient_as_bucket_view=True, **ddp_options)
    # Parameters have already been broadcast by DDP. Vary per-rank source
    # sampling while keeping model initialization and dataset planning shared.
    torch.manual_seed(training_seed + context.rank)
    random.seed(training_seed + context.rank)
    np.random.seed((training_seed + context.rank) % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed(training_seed + context.rank)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    same_layout = (training_state.get("world_size", context.world_size) == context.world_size
                   and training_state.get("batch_size", args.batch_size) == args.batch_size)
    data_epoch, batches_consumed = _resume_data_cursor(
        training_state, step, len(dataset), context.world_size, args.batch_size,
        reset=args.reset_data_cursor)
    optimizer_updates = int(training_state.get("optimizer_updates", step))
    rank_rng = resume.get("rank_rng", []) if resume is not None else []
    if context.rank < len(rank_rng):
        restore_rng(rank_rng[context.rank], device)
    if resume is not None and not same_layout and context.primary:
        print(json.dumps({"resume_layout": {
            "previous_world_size": training_state.get("world_size"),
            "world_size": context.world_size, "batch_size": args.batch_size,
            "data_epoch": data_epoch, "batches_consumed": batches_consumed,
            "global_samples_consumed_in_epoch": batches_consumed * context.world_size * args.batch_size,
            "preserved_rank_rng": min(len(rank_rng), context.world_size),
            "new_rank_rng": "training_seed + rank", "reset_data_cursor": args.reset_data_cursor,
        }}), flush=True)

    def checkpoint(path):
        from .datasets.dense import DENSE_TARGET_VERSION
        # All ranks contribute RNG metadata; only rank zero writes weights,
        # optimizer/scaler state and the common data cursor. Save the bare model
        # so existing native/4RC checkpoint key names remain unchanged.
        rng = context.gather_rng()
        if context.primary:
            save_checkpoint(path, model, optimizer, step=step,
                            grad_scaler=scaler.state_dict(), rank_rng=rng,
                            training_state={"data_epoch": data_epoch,
                                            "batches_consumed": batches_consumed,
                                            "world_size": context.world_size,
                                            "batch_size": args.batch_size,
                                            "dataset_size": len(dataset),
                                            "source_count": args.source_count,
                                            "max_source_time_pairs": args.max_source_time_pairs,
                                            "prefer_motion_sources": prefer_motion_sources,
                                            "time_projection_reset_step": time_projection_reset_step,
                                            "seed": training_seed,
                                            "optimizer_updates": optimizer_updates,
                                            "supervision": "dense" if dense_supervision else "sparse_or_manifest",
                                            "scene_normalized": dense_supervision,
                                            "dense_target_version": DENSE_TARGET_VERSION,
                                            "kubric_background_policy": getattr(dataset, "kubric_background_policy", None),
                                             "loss_weights": criterion.weights,
                                             "loss_profile": loss_profile,
                                             "loss_config": criterion.to_config(),
                                             "lr_plan": lr_plan.to_config(),
                                            "amp_dtype": str(amp_dtype).removeprefix("torch.")})

    while step < args.steps:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(data_epoch)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(data_epoch)
        loader_generator.manual_seed(training_seed + data_epoch)
        batch_sampler.start_batch = batches_consumed
        for batch_index, batch in enumerate(loader, start=batches_consumed):
            if step >= args.steps:
                break
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
            batch["targets"] = {key: value.to(device, non_blocking=True)
                                for key, value in batch["targets"].items()}
            source = _sources(batch, args.source_count, args.max_source_time_pairs, prefer_motion_sources)
            if dense_supervision:
                from .datasets.dense import materialize_dense_targets
                targets = materialize_dense_targets(batch, source)
            else:
                targets = select_source_targets(batch["targets"], source, batch["images"].shape[1])
            # Native adapters publish fixed fields even when labels are
            # unavailable. Avoid a visibility forward/loss without supervision.
            targets = dict(targets)
            for name in ("visibility", "sparse_visibility"):
                mask = targets.get(f"{name}_valid")
                if name in targets and mask is not None and not mask.any():
                    targets.pop(name)
                    targets.pop(f"{name}_valid")
            need_visibility = any(key in targets for key in ("visibility", "sparse_visibility"))
            optimizer.zero_grad(set_to_none=True)
            lr_plan.apply(optimizer, step)
            forward_error = None
            try:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    loss, metrics = training_step(
                        batch["images"], batch["camera_id"], batch["time_idx"],
                        batch["query_time_idx"], source, batch["observation_valid"],
                        targets, need_visibility)
            except Exception as error:
                forward_error = error
            locally_finite = forward_error is None and torch.isfinite(loss).item()
            if not context.all_true(locally_finite):
                if forward_error is not None:
                    raise forward_error
                if locally_finite:
                    raise RuntimeError("another distributed rank failed during forward or loss evaluation")
                raise FloatingPointError("training loss is not finite")
            norm, skipped = _optimizer_step(loss, model, optimizer, scaler, args.clip_grad)
            optimizer_updates += int(not skipped)
            step += 1
            batches_consumed = batch_index + 1
            metrics["grad_norm"] = norm.detach()
            averaged = aggregate_metrics(context, metrics)
            rank_batch = dict(dataset=batch.get("dataset_name", []), clip_id=batch.get("clip_id", []),
                input_hw=list(batch["images"].shape[-2:]), observations=int(batch["observation_valid"].sum()),
                times=batch["query_time_idx"].shape[1], decoded_sources=source.shape[1])
            # Averaged metrics can mix unrelated datasets. Retain local error
            # measurements to attribute outliers without another forward.
            local_keys = ("loss", "depth_abs_rel", "depth_mae_normalized", "trajectory_epe_normalized",
                          "displacement_epe_dynamic_normalized", "depth_confidence_saturated_fraction",
                          "trajectory_epe_temporal_normalized", "displacement_epe_temporal_normalized",
                          "supervised_source_pixels", "supervised_trajectory_pixels",
                          "sparse_trajectory", "sparse_displacement", "sparse_velocity", "sparse_visibility",
                          "native_track_queries", "native_track_point_times", "native_temporal_point_times",
                          "native_trajectory_epe_normalized", "native_displacement_epe_normalized",
                          "native_trajectory_epe_temporal_normalized", "native_displacement_epe_temporal_normalized",
                          "native_velocity_epe_normalized", "camera_rotation_degrees",
                          "excluded_background_pixels", "excluded_background_fraction")
            rank_batch["metrics"] = {k: float(metrics[k].detach()) for k in local_keys if k in metrics}
            if "scene_scale" in targets:
                rank_batch["scene_scale"] = targets["scene_scale"].detach().cpu().tolist()
            if "depth_filter" in batch:
                rank_batch["depth_filter"] = batch["depth_filter"]
            if "static_multiview" in batch:
                rank_batch["static_multiview"] = batch["static_multiview"]
            if "native_tracks" in batch:
                rank_batch["native_tracks"] = batch["native_tracks"]
            if "syn4d" in batch:
                rank_batch["syn4d"] = batch["syn4d"]
            rank_batches = [rank_batch]
            if context.distributed:
                rank_batches = [None]*context.world_size
                dist.all_gather_object(rank_batches, rank_batch)
            if context.primary:
                print(json.dumps({"step": step, "world_size": context.world_size,
                                  "metrics_version": METRICS_VERSION,
                                  "supervision": "dense" if dense_supervision else "sparse_or_manifest",
                                  "input_hw": list(batch["images"].shape[-2:]),
                                  "rank_batches": rank_batches,
                                  "data_epoch": data_epoch,
                                  "optimizer_step_skipped": bool(skipped),
                                   "amp_dtype": str(amp_dtype).removeprefix("torch."),
                                   "loss_profile": loss_profile,
                                   "backbone_lr": optimizer.param_groups[0]["lr"],
                                   "lr": optimizer.param_groups[1]["lr"],
                                  **averaged}, allow_nan=False), flush=True)
            if step % args.save_every == 0:
                checkpoint(output/f"step_{step:08d}.pt")
        if batches_consumed >= batches_per_epoch:
            data_epoch += 1
            batches_consumed = 0
    checkpoint(output/"latest.pt")
    return output/"latest.pt"


if __name__ == "__main__":
    main()
