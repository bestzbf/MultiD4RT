"""Device and precision helpers for inference and distributed evaluation."""

from dataclasses import dataclass
from datetime import timedelta
import os

import torch
from torch import distributed as dist


def resolve_device(requested, local_rank=0, distributed=False):
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
        raise ValueError("distributed evaluation supports CUDA/NCCL or CPU/Gloo")
    return device


def resolve_amp_dtype(device, requested="auto"):
    if requested not in ("auto", "bfloat16", "float16"):
        raise ValueError("AMP dtype must be auto, bfloat16, or float16")
    if device.type == "cpu":
        if requested == "float16":
            raise ValueError("CPU AMP uses bfloat16; float16 requires CUDA")
        return torch.bfloat16
    if device.type == "cuda":
        supported = torch.cuda.is_bf16_supported()
        if requested == "bfloat16" and not supported:
            raise ValueError("this CUDA device does not support bfloat16 AMP")
        return torch.float16 if requested == "float16" or (requested == "auto" and not supported) else torch.bfloat16
    return torch.float16


@dataclass
class EvaluationContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    owns_group: bool = False

    @classmethod
    def initialize(cls, requested):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            device = resolve_device(requested, local_rank, world > 1)
            if device.type == "cuda":
                torch.cuda.set_device(device)
            return cls(device, rank, world)
        world = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        if world < 1 or not 0 <= rank < world or local_rank < 0:
            raise ValueError("invalid torchrun WORLD_SIZE/RANK/LOCAL_RANK")
        if world > 1 and any(k not in os.environ for k in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")):
            raise ValueError("use torchrun to provide evaluation rank and rendezvous variables")
        device = resolve_device(requested, local_rank, world > 1)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        context = cls(device, rank, world)
        if world > 1:
            if not dist.is_available():
                raise RuntimeError("this PyTorch build has no distributed evaluation support")
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                    init_method="env://", rank=rank, world_size=world,
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
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
            self.owns_group = False
