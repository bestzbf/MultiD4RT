"""Small torchrun helpers; local CUDA indices respect CUDA_VISIBLE_DEVICES."""

from dataclasses import dataclass
from datetime import timedelta
import math
import os
import random

import numpy as np
import torch
from torch import distributed as dist


def resolve_device(requested, local_rank=0, distributed=False):
    """Under torchrun, visible GPU LOCAL_RANK is this process's only device."""
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if distributed:
            device = torch.device("cuda", local_rank)
        elif device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} exceeds the visible GPU count")
    elif distributed and device.type != "cpu":
        raise ValueError("distributed training supports CUDA/NCCL or CPU/Gloo")
    return device


def resolve_amp_dtype(device, requested="auto"):
    """Prefer CUDA BF16 when supported; CPU mixed precision uses BF16."""
    if requested not in ("auto", "bfloat16", "float16"):
        raise ValueError("AMP dtype must be auto, bfloat16, or float16")
    if device.type == "cpu":
        if requested == "float16":
            raise ValueError("CPU AMP training uses bfloat16; float16 requires CUDA GradScaler")
        return torch.bfloat16
    if device.type == "cuda":
        supports_bf16 = torch.cuda.is_bf16_supported()
        if requested == "bfloat16" and not supports_bf16:
            raise ValueError("this CUDA device does not support bfloat16 AMP")
        if requested == "float16" or (requested == "auto" and not supports_bf16):
            return torch.float16
        return torch.bfloat16
    return torch.float16


@dataclass
class DistributedContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    owns_group: bool = False

    @classmethod
    def initialize(cls, requested):
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = resolve_device(requested, local_rank, world > 1)
            if device.type == "cuda":
                torch.cuda.set_device(device)
            return cls(device, rank, world, local_rank)
        try:
            world = int(os.environ.get("WORLD_SIZE", 1))
            rank = int(os.environ.get("RANK", 0))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        except ValueError as error:
            raise ValueError("torchrun rank variables must be integers") from error
        if world < 1 or not 0 <= rank < world or local_rank < 0:
            raise ValueError("invalid torchrun WORLD_SIZE/RANK/LOCAL_RANK")
        if world > 1 and any(key not in os.environ for key in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")):
            raise ValueError("launch distributed training with torchrun to provide rank and rendezvous variables")
        device = resolve_device(requested, local_rank, world > 1)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        context = cls(device, rank, world, local_rank)
        if world > 1:
            if not dist.is_available():
                raise RuntimeError("this PyTorch build has no torch.distributed support")
            backend = "nccl" if device.type == "cuda" else "gloo"
            # A bounded timeout also lets other ranks leave if a peer fails
            # before torchrun's elastic launcher terminates the remaining jobs.
            dist.init_process_group(backend=backend, init_method="env://",
                                    rank=rank, world_size=world,
                                    timeout=timedelta(minutes=5))
            context.owns_group = True
        return context

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def primary(self):
        return self.rank == 0

    def close(self):
        # Do not enter a final barrier: exception cleanup must not wait for a
        # rank that already failed. Leave externally owned process groups alone.
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
            self.owns_group = False

    def all_true(self, condition):
        flag = torch.tensor(int(bool(condition)), dtype=torch.int32, device=self.device)
        if self.distributed:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def gather_rng(self):
        state = capture_rng(self.device)
        if not self.distributed:
            return [state]
        states = [None] * self.world_size
        dist.all_gather_object(states, state)
        return states


def capture_rng(device):
    # Keep this checkpoint metadata compatible with torch.load(weights_only=True).
    numpy_state = np.random.get_state()
    state = {"torch": torch.get_rng_state(), "python": random.getstate(),
             "numpy": (numpy_state[0], numpy_state[1].tolist(),
                       int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4]))}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


_METRICS = ("loss", "depth", "camera", "ray", "trajectory", "displacement",
            "visibility", "static", "sparse_trajectory", "sparse_displacement",
            "sparse_visibility", "sparse_velocity", "native_track_queries", "native_track_point_times",
            "native_temporal_point_times", "native_trajectory_epe_normalized",
            "native_displacement_epe_normalized", "native_trajectory_epe_temporal_normalized",
            "native_displacement_epe_temporal_normalized", "native_velocity_epe_normalized",
            "confidence", "grad_norm", "depth_gradient", "velocity",
            "supervised_source_pixels", "supervised_trajectory_pixels",
            "trajectory_epe_normalized", "displacement_epe_normalized",
            "trajectory_epe_temporal_normalized", "displacement_epe_temporal_normalized",
            "trajectory_epe_dynamic_normalized", "displacement_epe_dynamic_normalized",
            "supervised_dynamic_pixels", "depth_mae_normalized", "depth_abs_rel", "base_epe_normalized",
            "velocity_epe_normalized", "velocity_epe_dynamic_normalized",
            "depth_confidence_mean", "depth_confidence_log_mean", "depth_confidence_saturated_fraction",
            "motion_confidence_mean", "motion_confidence_log_mean", "motion_confidence_saturated_fraction",
            "camera_translation_epe_normalized", "camera_rotation_degrees",
            "visibility_precision", "visibility_recall", "visibility_f1",
            "camera_translation", "camera_rotation", "camera_fov", "camera_principal",
            "excluded_background_pixels", "excluded_background_fraction")


def aggregate_metrics(context, metrics):
    """Average metrics with a fixed collective layout even when rank GT differs."""
    unknown = set(metrics) - set(_METRICS)
    if unknown:
        raise ValueError(f"unsupported distributed training metrics: {sorted(unknown)}")
    values = torch.zeros(len(_METRICS), dtype=torch.float64, device=context.device)
    present = torch.zeros(len(_METRICS), dtype=torch.float64, device=context.device)
    for index, name in enumerate(_METRICS):
        if name in metrics:
            values[index] = torch.as_tensor(metrics[name], device=context.device).detach()
            present[index] = 1
    if context.distributed:
        dist.all_reduce(values)
        dist.all_reduce(present)
    result = {}
    for name, value, exists in zip(_METRICS, values.cpu().tolist(), present.cpu().tolist()):
        if exists:
            # Losses are averaged per training example, including examples
            # with a task disabled. Diagnostics only average ranks on which
            # the corresponding GT/metric was actually present.
            denominator = exists if ("_epe_" in name or "confidence_" in name or name in ("depth_abs_rel", "depth_mae_normalized", "camera_rotation_degrees", "visibility_precision", "visibility_recall", "visibility_f1", "excluded_background_fraction")) else context.world_size
            average = value / denominator
            result[name] = average if math.isfinite(average) else None
    return result
