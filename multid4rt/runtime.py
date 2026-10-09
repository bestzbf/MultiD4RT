"""Precision selection for model inference."""

import torch


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
