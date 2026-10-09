"""Frame-index embeddings shared by the encoder and motion decoder."""

import math

import torch
from torch import nn


class SinusoidalFrameEmbedding(nn.Module):
    """Encode a signed frame index without a learned maximum sequence length.

    Frame indices use the same origin and sampling units for every camera.
    The returned embedding is float32, leaving mixed-precision casting to the
    consuming projection or token stream. Frequencies are not checkpoint state.
    """

    def __init__(self, dim: int, base: float = 100.0):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("frame embedding dim must be a positive even integer")
        if not math.isfinite(base) or base <= 1:
            raise ValueError("frame embedding base must be finite and greater than 1")
        self.dim = dim
        self.base = base
        frequencies = base ** (-torch.arange(dim // 2, dtype=torch.float32) / (dim // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time_tensor: torch.Tensor) -> torch.Tensor:
        phase = time_tensor.to(dtype=torch.float32)[..., None] * self.frequencies.float()
        return torch.cat((phase.sin(), phase.cos()), dim=-1)
