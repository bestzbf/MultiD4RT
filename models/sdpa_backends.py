"""Scaled-dot-product attention backend selection helpers."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import ContextManager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


_SDPA_BACKEND_ENV = "D4RT_SDPA_BACKEND"

_BACKEND_ALIASES = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "flash_attention": SDPBackend.FLASH_ATTENTION,
    "flash-attention": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "efficient_attention": SDPBackend.EFFICIENT_ATTENTION,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "memory_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "cudnn_attention": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}


def _is_blackwell(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(tensor.device)[0] >= 10


def _parse_sdpa_backends(value: str, *, blackwell: bool) -> list[SDPBackend] | None:
    mode = value.strip().lower()
    if mode in {"", "auto"}:
        if blackwell:
            return [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        return None

    if mode in {"default", "torch"}:
        return None

    if mode in {"no_cudnn", "no-cudnn", "flash_math", "flash-math"}:
        return [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]

    if "," in mode or "+" in mode:
        separator = "," if "," in mode else "+"
        tokens = [token.strip() for token in mode.split(separator) if token.strip()]
        try:
            return [_BACKEND_ALIASES[token] for token in tokens]
        except KeyError as exc:
            raise ValueError(
                f"Unknown {_SDPA_BACKEND_ENV} backend {exc.args[0]!r}. "
                "Use auto, default, no_cudnn, flash, efficient, cudnn, math, "
                "or a comma-separated list such as flash,efficient,math."
            ) from exc

    if mode in _BACKEND_ALIASES:
        return [_BACKEND_ALIASES[mode]]

    raise ValueError(
        f"Unknown {_SDPA_BACKEND_ENV}={value!r}. "
        "Use auto, default, no_cudnn, flash, efficient, cudnn, math, "
        "or a comma-separated list such as flash,efficient,math."
    )


def sdpa_kernel_context(reference: torch.Tensor) -> ContextManager[None]:
    """Return the SDPA backend context for D4RT attention calls.

    Default behavior is unchanged off Blackwell. On Blackwell, avoid cuDNN
    attention because it can fail to build execution plans for D4RT shapes,
    while keeping FlashAttention and math fallback available.
    """

    backends = _parse_sdpa_backends(
        os.getenv(_SDPA_BACKEND_ENV, "auto"),
        blackwell=_is_blackwell(reference),
    )
    if backends is None:
        return nullcontext()
    if len(backends) == 1:
        return sdpa_kernel(backends[0])
    return sdpa_kernel(backends)
