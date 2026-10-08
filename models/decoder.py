"""D4RT Decoder: Lightweight cross-attention transformer for point queries."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from utils.patches import (
    extract_local_patches_from_frame_list,
    extract_local_patches_with_valid_hw,
)
from .embeddings import FourierEmbedding, TimestepEmbedding, PatchEmbeddingFast
from .sdpa_backends import sdpa_kernel_context


CANONICAL_QUERY_SPACE_CROP_NORMALIZED = 0
VALID_PATCH_PROVIDERS = (
    "auto",
    "sampled_resized",
    "precomputed_resized",
    "sampled_highres",
    "precomputed_highres",
)


class CrossAttention(nn.Module):
    """Efficient cross-attention using PyTorch's scaled_dot_product_attention.

    Automatically uses FlashAttention or memory-efficient attention when available.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) query tokens
            key: (B, N_kv, C) key tokens from the fixed encoder memory
            value: Optional (B, N_kv, C) value tokens, defaults to ``key``
            mask: Optional attention mask

        Returns:
            out: (B, N_q, C)
        """
        if value is None:
            value = key

        B, N_q, C = query.shape
        N_kv = key.shape[1]

        # Project queries, keys, values. Queries never attend to each other here.
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's efficient attention. On Blackwell, avoid cuDNN SDPA
        # execution-plan failures while keeping FlashAttention enabled.
        with sdpa_kernel_context(q):
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop if self.training else 0.0
            )

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DecoderBlock(nn.Module):
    """Pre-LN cross-attention + MLP decoder block without query self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        disable_cross_attention: bool = False,
    ):
        super().__init__()
        self.disable_cross_attention = disable_cross_attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        query: torch.Tensor,
        encoder_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) query tokens
            encoder_features: (B, N_kv, C) encoder output (Global Scene Representation)

        Returns:
            out: (B, N_q, C)
        """
        if not self.disable_cross_attention:
            memory = self.norm_kv(encoder_features)
            query = query + self.cross_attn(
                query=self.norm1(query),
                key=memory,
                value=memory,
            )
        # MLP
        query = query + self.mlp(self.norm2(query))

        return query


class D4RTDecoder(nn.Module):
    """D4RT Pointwise Decoder.

    Lightweight cross-attention transformer that decodes queries independently.
    Each query (u, v, t_src, t_tgt, t_cam) is decoded to predict:
    - 3D position (X, Y, Z)
    - 2D position (u, v) reprojection
    - Visibility flag
    - Motion displacement
    - Surface normal
    - Confidence score
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_timesteps: int = 128,
        patch_size: int = 9,
        patch_provider: str = "auto",
        num_fourier_freqs: int = 64,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        disable_query_patch_embedding: bool = False,
        disable_query_timestep_embedding: bool = False,
        disable_cross_attention: bool = False,
        debug_3d_head_mode: str = "linear",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        if patch_provider not in VALID_PATCH_PROVIDERS:
            raise ValueError(
                f"Unknown patch_provider={patch_provider!r}. Choose from {VALID_PATCH_PROVIDERS}"
            )
        self.patch_provider = patch_provider
        self.disable_query_patch_embedding = disable_query_patch_embedding
        self.disable_query_timestep_embedding = disable_query_timestep_embedding
        self.disable_cross_attention = disable_cross_attention
        if debug_3d_head_mode not in {"linear", "mlp256"}:
            raise ValueError(
                f"Unknown debug_3d_head_mode={debug_3d_head_mode!r}. "
                "Choose from {'linear', 'mlp256'}."
            )
        self.debug_3d_head_mode = debug_3d_head_mode

        # Query embeddings
        self.fourier_embed = FourierEmbedding(embed_dim, num_fourier_freqs)
        self.timestep_embed = TimestepEmbedding(max_timesteps, embed_dim)
        self.patch_embed = PatchEmbeddingFast(patch_size, embed_dim)

        # Learnable query token (base)
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Mix the summed query components into the final decoder query token.
        self.query_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                True,
                drop_rate,
                attn_drop_rate,
                disable_cross_attention=disable_cross_attention,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output heads
        self.head_3d = self._build_3d_head(embed_dim, debug_3d_head_mode)
        self.head_2d = nn.Linear(embed_dim, 2)  # 2D position
        self.head_vis = nn.Linear(embed_dim, 1)  # Visibility
        self.head_disp = nn.Linear(embed_dim, 3)  # Displacement/motion
        self.head_normal = nn.Linear(embed_dim, 3)  # Surface normal
        self.head_conf = nn.Linear(embed_dim, 1)  # Positive confidence weight

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _build_3d_head(self, embed_dim: int, mode: str) -> nn.Module:
        if mode == "linear":
            return nn.Linear(embed_dim, 3)
        if mode == "mlp256":
            return nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
            )
        raise ValueError(f"Unhandled debug_3d_head_mode={mode!r}")

    def _validate_transform_metadata(
        self,
        transform_metadata: Optional[dict[str, torch.Tensor]],
        batch_size: int,
    ) -> None:
        if transform_metadata is None:
            return

        required_fields = {
            "canonical_space": 1,
            "original_hw": 2,
            "crop_offset_xy": 2,
            "crop_size_hw": 2,
            "resized_hw": 2,
        }
        for field, trailing_dim in required_fields.items():
            if field not in transform_metadata:
                raise ValueError(f"Missing transform_metadata[{field!r}] for patch provider")
            value = transform_metadata[field]
            if not torch.is_tensor(value):
                raise TypeError(f"transform_metadata[{field!r}] must be a tensor")
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"transform_metadata[{field!r}] batch dim must be {batch_size}, got {tuple(value.shape)}"
                )
            if trailing_dim != 1 and value.shape[-1] != trailing_dim:
                raise ValueError(
                    f"transform_metadata[{field!r}] trailing dim must be {trailing_dim}, got {tuple(value.shape)}"
                )

        canonical_space = transform_metadata["canonical_space"].reshape(batch_size, -1)
        if not torch.all(canonical_space == CANONICAL_QUERY_SPACE_CROP_NORMALIZED):
            raise ValueError("Only crop-normalized canonical query coordinates are currently supported")

    def _resolve_patch_provider(self, local_patches: torch.Tensor | None) -> str:
        if self.patch_provider == "auto":
            return "precomputed_resized" if local_patches is not None else "sampled_resized"
        return self.patch_provider

    def get_patch_provider_info(self, local_patches: torch.Tensor | None,
                                 transform_metadata: Optional[dict[str, torch.Tensor]] = None) -> dict:
        """Get diagnostic info about which patch provider will be used.

        Returns:
            dict with keys: configured, resolved, has_local_patches, has_transform_metadata
        """
        resolved = self._resolve_patch_provider(local_patches)
        return {
            "configured": self.patch_provider,
            "resolved": resolved,
            "has_local_patches": local_patches is not None,
            "has_transform_metadata": transform_metadata is not None,
        }

    def _embed_query_patches(
        self,
        frames: torch.Tensor | list[torch.Tensor],
        coords: torch.Tensor,
        t_src: torch.Tensor,
        local_patches: torch.Tensor | None,
        transform_metadata: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        provider = self._resolve_patch_provider(local_patches)
        self._validate_transform_metadata(transform_metadata, batch_size=coords.shape[0])

        if provider in {"precomputed_resized", "precomputed_highres"}:
            if local_patches is None:
                raise ValueError(f"patch_provider='{provider}' requires local_patches in the batch")
            return self.patch_embed(frames, coords, t_src, local_patches=local_patches)
        if provider == "sampled_resized":
            if isinstance(frames, list):
                raise ValueError("patch_provider='sampled_resized' requires batched tensor frames")
            return self.patch_embed(frames, coords, t_src, local_patches=None)
        if provider == "sampled_highres":
            if transform_metadata is None:
                raise ValueError("patch_provider='sampled_highres' requires transform_metadata in the batch")
            if isinstance(frames, list):
                crop_size_hw = transform_metadata["crop_size_hw"].to(device=coords.device)
                patches = extract_local_patches_from_frame_list(
                    frames_btchw=frames,
                    coords=coords,
                    t_src=t_src,
                    patch_size=self.patch_embed.patch_size,
                    valid_hw=crop_size_hw,
                )
            else:
                crop_size_hw = transform_metadata["crop_size_hw"].to(device=frames.device)
                patches = extract_local_patches_with_valid_hw(
                    frames_btchw=frames,
                    coords=coords,
                    t_src=t_src,
                    patch_size=self.patch_embed.patch_size,
                    valid_hw=crop_size_hw,
                )
            return self.patch_embed.embed_patches(patches)
        raise ValueError(f"Unhandled patch provider: {provider}")

    def build_query(
        self,
        frames: torch.Tensor | list[torch.Tensor],
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        local_patches: torch.Tensor | None = None,
        transform_metadata: Optional[dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Build query embeddings from components.

        Args:
            frames: (B, T, C, H, W) video frames for patch extraction
            coords: (B, N, 2) normalized (u, v) coordinates
            t_src: (B, N) source timestep indices
            t_tgt: (B, N) target timestep indices
            t_cam: (B, N) camera reference timestep indices
            local_patches: Optional pre-extracted local RGB patches
            transform_metadata: Optional geometry metadata for canonical query coordinates

        Returns:
            query: (B, N, embed_dim)
        """
        B, N = coords.shape[:2]

        # Fourier embedding of coordinates
        coord_emb = self.fourier_embed(coords)  # (B, N, embed_dim)

        # Timestep embeddings
        if self.disable_query_timestep_embedding:
            src_emb = torch.zeros_like(coord_emb)
            tgt_emb = torch.zeros_like(coord_emb)
            cam_emb = torch.zeros_like(coord_emb)
        else:
            src_emb, tgt_emb, cam_emb = self.timestep_embed(t_src, t_tgt, t_cam)

        # Local RGB patch embedding
        # Reshape frames for patch extraction: (B, T, C, H, W)
        if isinstance(frames, list):
            if len(frames) != B:
                raise ValueError(
                    f"Expected frame list length to match batch size {B}, got {len(frames)}"
                )
        elif frames.dim() == 5 and frames.shape[-1] == 3:
            frames = frames.permute(0, 1, 4, 2, 3)  # (B, T, H, W, C) -> (B, T, C, H, W)

        if self.disable_query_patch_embedding:
            patch_emb = torch.zeros_like(coord_emb)
        else:
            patch_emb = self._embed_query_patches(
                frames,
                coords,
                t_src,
                local_patches=local_patches,
                transform_metadata=transform_metadata,
            )  # (B, N, embed_dim)

        # Combine all embeddings
        query = coord_emb + src_emb + tgt_emb + cam_emb + patch_emb

        # Add learnable query token, then mix all query components before decoding.
        query = query + self.query_token.expand(B, N, -1)
        query = self.query_mlp(query)

        return query

    def _decode_query_tensor(
        self,
        encoder_features: torch.Tensor,
        query: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run decoder blocks and heads on a prebuilt query tensor."""
        for block in self.blocks:
            query = block(query, encoder_features)

        query = self.norm(query)

        pos_3d = self.head_3d(query)
        delta_2d = self.head_2d(query)
        # Predict 2D residuals around the input query to make identity mappings easy to learn.
        pos_2d = delta_2d if coords is None else coords + delta_2d
        visibility = self.head_vis(query)
        displacement = self.head_disp(query)
        normal = self.head_normal(query)
        normal = F.normalize(normal, dim=-1)
        # Kendall log-parameterization: s = log(σ²), σ² is the predicted variance, s ∈ ℝ.
        # Equivalent to the paper's c·L − log(c) with c = exp(-s).
        # s < 0 means high confidence (σ² < 1), s > 0 means low confidence (σ² > 1).
        # Optimal: s* = log(L / λ_conf), so s* < 0 when loss_3d < lambda_conf (normal at convergence).
        s = self.head_conf(query)

        # Clamp to prevent numerical overflow in exp(-s) and exp(s).
        s = torch.clamp(s, min=-10.0, max=10.0)

        # loss weight = exp(-s) = 1/σ²: down-weights high-variance (uncertain) predictions.
        confidence_weight = torch.exp(-s)

        return {
            'pos_3d': pos_3d,
            'pos_2d': pos_2d,
            'visibility': visibility,
            'displacement': displacement,
            'normal': normal,
            'uncertainty': s,            # s = log(σ²); penalty term in loss is λ_conf * s
            'confidence_weight': confidence_weight  # exp(-s) = 1/σ²; weights the 3D loss
        }

    def forward(
        self,
        encoder_features: torch.Tensor,
        frames: torch.Tensor | list[torch.Tensor],
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        local_patches: torch.Tensor | None = None,
        transform_metadata: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: (B, N_enc, embed_dim) Global Scene Representation
            frames: (B, T, C, H, W) video frames
            coords: (B, N_q, 2) normalized query coordinates
            t_src: (B, N_q) source timesteps
            t_tgt: (B, N_q) target timesteps
            t_cam: (B, N_q) camera reference timesteps
            local_patches: Optional pre-extracted local RGB patches
            transform_metadata: Optional geometry metadata for canonical query coordinates

        Returns:
            Dictionary with predictions:
                - pos_3d: (B, N_q, 3) 3D positions
                - pos_2d: (B, N_q, 2) 2D positions
                - visibility: (B, N_q, 1) visibility logits
                - displacement: (B, N_q, 3) motion displacement
                - normal: (B, N_q, 3) surface normals
                - confidence: (B, N_q, 1) confidence scores
        """
        query = self.build_query(
            frames,
            coords,
            t_src,
            t_tgt,
            t_cam,
            local_patches=local_patches,
            transform_metadata=transform_metadata,
        )
        return self._decode_query_tensor(encoder_features, query, coords=coords)

    def decode_3d_position(
        self,
        encoder_features: torch.Tensor,
        frames: torch.Tensor | list[torch.Tensor],
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        local_patches: torch.Tensor | None = None,
        transform_metadata: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Convenience method to get only 3D positions.

        Returns:
            pos_3d: (B, N_q, 3) 3D positions
        """
        outputs = self.forward(
            encoder_features,
            frames,
            coords,
            t_src,
            t_tgt,
            t_cam,
            local_patches=local_patches,
            transform_metadata=transform_metadata,
        )
        return outputs['pos_3d']
