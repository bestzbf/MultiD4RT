"""Embedding modules for D4RT queries."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.patches import extract_local_patches


class FourierEmbedding(nn.Module):
    """Fourier feature embedding for continuous 2D coordinates.

    Maps (u, v) coordinates to higher-dimensional space using sinusoidal functions.
    """

    def __init__(self, embed_dim: int, num_frequencies: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies

        # Frequency bands for positional encoding
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freqs', freqs)

        # Project fourier features to embed_dim
        fourier_dim = 2 * 2 * num_frequencies  # 2 coords * (sin + cos) * num_freqs
        self.proj = nn.Linear(fourier_dim, embed_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2) normalized coordinates in [0, 1]

        Returns:
            embeddings: (B, N, embed_dim)
        """
        # Compute Fourier features in fp32 to avoid overflow from high frequency bands
        # under autocast/half precision. The projected values are bounded again after
        # sin/cos, so casting back for the linear layer is safe.
        B, N, _ = coords.shape
        coords_fp32 = coords.to(torch.float32)
        freqs_fp32 = self.freqs.to(torch.float32)

        # Expand for frequency multiplication: (B, N, 2, num_freqs)
        coords_freq = coords_fp32.unsqueeze(-1) * freqs_fp32 * (2 * math.pi)

        # Apply sin and cos: (B, N, 2, num_freqs * 2)
        fourier_features = torch.cat([
            torch.sin(coords_freq),
            torch.cos(coords_freq)
        ], dim=-1)

        # Flatten: (B, N, 2 * num_freqs * 2)
        fourier_features = fourier_features.reshape(B, N, -1)

        # Project to embed_dim
        return self.proj(fourier_features.to(self.proj.weight.dtype))


class TimestepEmbedding(nn.Module):
    """Learnable discrete timestep embeddings.

    Provides separate embeddings for source, target, and camera timesteps.
    """

    def __init__(self, max_timesteps: int, embed_dim: int):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.embed_dim = embed_dim

        # Separate embeddings for each timestep type
        self.src_embedding = nn.Embedding(max_timesteps, embed_dim)
        self.tgt_embedding = nn.Embedding(max_timesteps, embed_dim)
        self.cam_embedding = nn.Embedding(max_timesteps, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.src_embedding.weight, std=0.02)
        nn.init.normal_(self.tgt_embedding.weight, std=0.02)
        nn.init.normal_(self.cam_embedding.weight, std=0.02)

    def forward(
        self,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            t_src: (B, N) source timestep indices
            t_tgt: (B, N) target timestep indices
            t_cam: (B, N) camera reference timestep indices

        Returns:
            Tuple of embeddings, each (B, N, embed_dim)
        """
        src_emb = self.src_embedding(t_src)
        tgt_emb = self.tgt_embedding(t_tgt)
        cam_emb = self.cam_embedding(t_cam)

        return src_emb, tgt_emb, cam_emb


class PatchEmbedding(nn.Module):
    """Local RGB patch embedding.

    Extracts and embeds a local patch around each query point.
    This dramatically improves performance by providing low-level appearance cues.
    """

    def __init__(self, patch_size: int = 9, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # MLP to embed flattened RGB patch
        patch_dim = patch_size * patch_size * 3
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def extract_patches(
        self,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor
    ) -> torch.Tensor:
        """Extract local patches around query coordinates.

        Args:
            frames: (B, T, C, H, W) video frames
            coords: (B, N, 2) normalized coordinates in [0, 1]
            t_src: (B, N) source frame indices

        Returns:
            patches: (B, N, patch_size, patch_size, 3)
        """
        B, T, C, H, W = frames.shape
        N = coords.shape[1]
        device = frames.device

        # Denormalize coordinates to pixel space
        u = coords[..., 0] * (W - 1)  # (B, N)
        v = coords[..., 1] * (H - 1)  # (B, N)

        # Get integer coordinates (center of patch)
        u_int = u.long()
        v_int = v.long()

        half_size = self.patch_size // 2
        patches = []

        for b in range(B):
            batch_patches = []
            for n in range(N):
                t = t_src[b, n].item()
                cx = u_int[b, n].item()
                cy = v_int[b, n].item()

                # Extract patch with padding for boundary cases
                frame = frames[b, t]  # (C, H, W)

                # Compute patch boundaries with clamping
                x_start = max(0, cx - half_size)
                x_end = min(W, cx + half_size + 1)
                y_start = max(0, cy - half_size)
                y_end = min(H, cy + half_size + 1)

                # Extract patch
                patch = frame[:, y_start:y_end, x_start:x_end]  # (C, h, w)

                # Pad if necessary
                pad_left = half_size - (cx - x_start)
                pad_right = half_size - (x_end - cx - 1)
                pad_top = half_size - (cy - y_start)
                pad_bottom = half_size - (y_end - cy - 1)

                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

                batch_patches.append(patch)

            patches.append(torch.stack(batch_patches, dim=0))  # (N, C, ps, ps)

        patches = torch.stack(patches, dim=0)  # (B, N, C, ps, ps)
        patches = patches.permute(0, 1, 3, 4, 2)  # (B, N, ps, ps, C)

        return patches

    def forward(
        self,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames
            coords: (B, N, 2) normalized coordinates
            t_src: (B, N) source frame indices

        Returns:
            embeddings: (B, N, embed_dim)
        """
        patches = self.extract_patches(frames, coords, t_src)
        B, N = patches.shape[:2]

        # Flatten patches
        patches_flat = patches.reshape(B, N, -1)  # (B, N, ps*ps*3)

        # Embed
        return self.mlp(patches_flat)


class PatchEmbeddingFast(nn.Module):
    """Faster vectorized patch embedding using grid_sample."""

    def __init__(self, patch_size: int = 9, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        patch_dim = patch_size * patch_size * 3
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Create relative grid offsets
        half = patch_size // 2
        offsets = torch.stack(torch.meshgrid(
            torch.arange(-half, half + 1),
            torch.arange(-half, half + 1),
            indexing='xy'
        ), dim=-1).float()  # (ps, ps, 2)
        self.register_buffer('offsets', offsets)

    def embed_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed pre-extracted patches.

        Args:
            patches: (B, N, C, ps, ps) or (B, N, ps, ps, C)

        Returns:
            embeddings: (B, N, embed_dim)
        """
        if patches.dim() != 5:
            raise ValueError(f"Expected 5D patches tensor, got shape {tuple(patches.shape)}")

        if patches.shape[2] == 3:
            patches = patches.permute(0, 1, 3, 4, 2)
        elif patches.shape[-1] != 3:
            raise ValueError(
                "Unsupported patch layout. Expected (B, N, C, ps, ps) or (B, N, ps, ps, C). "
                f"Got shape {tuple(patches.shape)}"
            )
        if patches.dtype == torch.uint8:
            patches = patches.float().div_(255.0)
        elif not patches.is_floating_point():
            patches = patches.float()

        B, N = patches.shape[:2]
        patches_flat = patches.reshape(B, N, -1)
        return self.mlp(patches_flat)

    def forward(
        self,
        frames: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor,
        local_patches: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames
            coords: (B, N, 2) normalized coordinates in [0, 1]
            t_src: (B, N) source frame indices
            local_patches: Optional pre-extracted patches

        Returns:
            embeddings: (B, N, embed_dim)
        """
        if local_patches is not None:
            return self.embed_patches(local_patches)
        patches = extract_local_patches(frames, coords, t_src, patch_size=self.patch_size)
        return self.embed_patches(patches)


class AspectRatioEmbedding(nn.Module):
    """Embedding for original video aspect ratio.

    Since videos are resized to square, we need to preserve aspect ratio info.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            aspect_ratio: (B,), (B, 1), or legacy (B, 2)

        Returns:
            embedding: (B, embed_dim)
        """
        if aspect_ratio.dim() == 1:
            aspect_ratio = aspect_ratio.unsqueeze(-1)
        elif aspect_ratio.dim() == 2 and aspect_ratio.shape[-1] == 2:
            first = aspect_ratio[:, :1]
            second = aspect_ratio[:, 1:2].clamp_min(1e-6)
            reciprocal_mask = torch.isclose(first * second, torch.ones_like(first), atol=1e-3, rtol=1e-3)
            aspect_ratio = torch.where(reciprocal_mask, first, first / second)
        return self.proj(aspect_ratio)
