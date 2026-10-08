"""D4RT Encoder: Video encoder using timm ViT backbone with local/global attention."""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from functools import partial
from pathlib import Path
from torch.utils.checkpoint import checkpoint
from .sdpa_backends import sdpa_kernel_context

try:
    import timm
    from timm.models.vision_transformer import VisionTransformer, Block
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Using custom implementation.")

# VideoMAE support via HuggingFace transformers
try:
    from transformers import VideoMAEModel, VideoMAEConfig
    VIDEOMAE_AVAILABLE = True
except ImportError:
    VIDEOMAE_AVAILABLE = False
    print("Warning: transformers not available. VideoMAE encoder disabled.")

try:
    from safetensors.torch import load_file as safetensors_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    safetensors_load_file = None


def canonicalize_video(video: torch.Tensor) -> torch.Tensor:
    """Convert video to (B, C, T, H, W)."""
    if video.dim() != 5:
        raise ValueError(f"Expected 5D video tensor, got shape {tuple(video.shape)}")

    if video.shape[1] == 3:
        out = video
    elif video.shape[2] == 3:
        out = video.permute(0, 2, 1, 3, 4)
    elif video.shape[-1] == 3:
        out = video.permute(0, 4, 1, 2, 3)
    else:
        raise ValueError(
            "Unsupported video layout. Expected one of "
            "(B, C, T, H, W), (B, T, C, H, W), or (B, T, H, W, C). "
            f"Got shape {tuple(video.shape)}"
        )

    if out.dtype == torch.uint8:
        return out.float().div_(255.0)
    if not out.is_floating_point():
        return out.float()
    return out


def canonicalize_aspect_ratio(aspect_ratio: torch.Tensor) -> torch.Tensor:
    """Convert aspect ratio input to a single width/height scalar per sample."""
    if aspect_ratio.dim() == 1:
        aspect_ratio = aspect_ratio.unsqueeze(-1)

    if aspect_ratio.dim() != 2:
        raise ValueError(
            "Expected aspect_ratio to have shape (B,), (B, 1), or (B, 2), "
            f"got {tuple(aspect_ratio.shape)}"
        )

    if aspect_ratio.shape[-1] == 1:
        return aspect_ratio

    if aspect_ratio.shape[-1] != 2:
        raise ValueError(
            "Expected aspect_ratio to have shape (B, 1) or (B, 2), "
            f"got {tuple(aspect_ratio.shape)}"
        )

    width_like = aspect_ratio[:, :1]
    height_like = aspect_ratio[:, 1:2].clamp_min(1e-6)
    reciprocal_like = width_like * height_like

    reciprocal_mask = torch.isclose(
        reciprocal_like,
        torch.ones_like(reciprocal_like),
        atol=1e-3,
        rtol=1e-3,
    )
    return torch.where(reciprocal_mask, width_like, width_like / height_like)


def build_sinusoid_encoding_table(num_positions: int, hidden_size: int) -> torch.Tensor:
    """Build fixed sin-cos positional embeddings matching VideoMAE's convention."""
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size)
    )

    table = torch.zeros(num_positions, hidden_size, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term[: table[:, 1::2].shape[1]])
    return table.unsqueeze(0)


class FactorizedPositionEncoding3D(nn.Module):
    """Learnable factorized 3D absolute positional encoding for video patches."""

    def __init__(
        self,
        num_frames: int,
        num_rows: int,
        num_cols: int,
        embed_dim: int,
        scale_init: Optional[float] = None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.embed_dim = embed_dim

        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, 1, 1, embed_dim))
        self.row_embed = nn.Parameter(torch.zeros(1, 1, num_rows, 1, embed_dim))
        self.col_embed = nn.Parameter(torch.zeros(1, 1, 1, num_cols, embed_dim))
        self.scale = None
        if scale_init is not None:
            self.scale = nn.Parameter(torch.full((1,), scale_init))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

    def _position_grid(self) -> torch.Tensor:
        pos = self.time_embed + self.row_embed + self.col_embed
        if self.scale is not None:
            pos = pos * self.scale
        return pos

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos = self._position_grid()
        if x is None:
            return pos.reshape(1, self.num_frames * self.num_rows * self.num_cols, self.embed_dim)
        if x.dim() != 5:
            raise ValueError(f"Expected patch tokens shaped (B, T, H, W, D), got {tuple(x.shape)}")
        return x + pos.type_as(x)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_time_key = prefix + "temporal_embed"
        time_key = prefix + "time_embed"
        if legacy_time_key in state_dict and time_key not in state_dict:
            state_dict[time_key] = state_dict.pop(legacy_time_key)
        if self.scale is None:
            state_dict.pop(prefix + "scale", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video.

    Converts video to sequence of patch embeddings with spatio-temporal patches.
    """

    def __init__(
        self,
        img_size: int = 256,
        temporal_size: int = 48,
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.temporal_size = temporal_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches_t = temporal_size // patch_size[0]
        self.num_patches_h = img_size // patch_size[1]
        self.num_patches_w = img_size // patch_size[2]
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) video tensor

        Returns:
            patches: (B, N, embed_dim) where N = num_patches
        """
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class EfficientAttention(nn.Module):
    """Efficient multi-head attention using PyTorch's scaled_dot_product_attention.

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)

        Returns:
            out: (B, N, C)
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Use PyTorch's efficient attention. On Blackwell, avoid cuDNN SDPA
        # execution-plan failures while keeping FlashAttention enabled.
        with sdpa_kernel_context(q):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop if self.training else 0.0
            )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LocalAttention(nn.Module):
    """Frame-wise local self-attention using efficient attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.attention = EfficientAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        patches_per_frame: int,
        num_special_tokens: int = 0,
    ) -> torch.Tensor:
        B, N, C = x.shape
        if num_special_tokens > 0:
            special_tokens = x[:, -num_special_tokens:]
            x = x[:, :-num_special_tokens]
        else:
            special_tokens = None

        x = x.reshape(B * num_frames, patches_per_frame, C)
        x = self.attention(x)
        x = x.reshape(B, N - num_special_tokens, C)

        if special_tokens is not None:
            x = torch.cat([x, special_tokens], dim=1)

        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""

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


class EncoderBlock(nn.Module):
    """Encoder block with either local or global attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        attention_type: str = 'global'
    ):
        super().__init__()
        self.attention_type = attention_type

        self.norm1 = nn.LayerNorm(dim)
        if attention_type == 'local':
            self.attn = LocalAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        else:
            self.attn = EfficientAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        num_frames: Optional[int] = None,
        patches_per_frame: Optional[int] = None,
        num_special_tokens: int = 0,
    ) -> torch.Tensor:
        if self.attention_type == 'local':
            x = x + self.attn(
                self.norm1(x),
                num_frames,
                patches_per_frame,
                num_special_tokens=num_special_tokens,
            )
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LocalBlock(nn.Module):
    """Local self-attention block over each temporal slice of the unified encoder state."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected local encoder state shaped (B, T, P, C), got {tuple(x.shape)}")

        B, T, P, C = x.shape
        tokens = x.reshape(B * T, P, C)
        if tokens.shape != (B * T, P, C):
            raise RuntimeError("LocalBlock failed to flatten to (B*T, P, C)")

        tokens = tokens + self.attn(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))

        x = tokens.reshape(B, T, P, C)
        if x.shape != (B, T, P, C):
            raise RuntimeError("LocalBlock failed to restore (B, T, P, C)")
        return x


class GlobalBlock(nn.Module):
    """Global self-attention block over the full spatio-temporal token sequence."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected global encoder state shaped (B, T, P, C), got {tuple(x.shape)}")

        B, T, P, C = x.shape
        tokens = x.reshape(B, T * P, C)
        if tokens.shape != (B, T * P, C):
            raise RuntimeError("GlobalBlock failed to flatten to (B, T*P, C)")

        tokens = tokens + self.attn(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))

        x = tokens.reshape(B, T, P, C)
        if x.shape != (B, T, P, C):
            raise RuntimeError("GlobalBlock failed to restore (B, T, P, C)")
        return x


class EncoderStage(nn.Module):
    """One encoder stage: local attention followed by global attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.local_block = LocalBlock(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
        )
        self.global_block = GlobalBlock(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        input_dtype = x.dtype
        input_device = x.device

        x = self.local_block(x)
        x = self.global_block(x)

        if x.shape != input_shape:
            raise RuntimeError(
                f"EncoderStage must preserve shape {tuple(input_shape)}, got {tuple(x.shape)}"
            )
        if x.dtype != input_dtype:
            raise RuntimeError(f"EncoderStage changed dtype from {input_dtype} to {x.dtype}")
        if x.device != input_device:
            raise RuntimeError(f"EncoderStage changed device from {input_device} to {x.device}")
        return x


class D4RTEncoder(nn.Module):
    """D4RT video encoder with a unified [B, T, P, C] token state."""

    def __init__(
        self,
        img_size: int = 256,
        temporal_size: int = 48,
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        local_global_pattern: str = 'interleaved',
        use_aspect_ratio_token: bool = True,
        keep_special_tokens_in_output: bool = False,
        use_timm_init: bool = False,
        use_videomae_init: bool = False,
        use_vggt_init: bool = False,
        use_monst3r_init: bool = False,
        videomae_model: str = 'MCG-NJU/videomae-base',
        timm_model: str = 'vit_base_patch16_224',
        vggt_checkpoint: str = 'facebook/VGGT-1B',
        monst3r_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        if depth % 2 != 0:
            raise ValueError(
                f"D4RTEncoder depth must be even so each stage has one local and one global block, got {depth}"
            )
        if local_global_pattern != 'interleaved':
            raise ValueError(
                "The staged encoder keeps a fixed Local->Global stage order; "
                f"got local_global_pattern={local_global_pattern!r}"
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.num_stages = depth // 2
        self.use_timm_init = use_timm_init
        self.use_videomae_init = use_videomae_init
        self.use_vggt_init = use_vggt_init
        self.use_monst3r_init = use_monst3r_init
        self.use_aspect_ratio_token = use_aspect_ratio_token
        self.keep_special_tokens_in_output = keep_special_tokens_in_output
        self.gradient_checkpointing = False

        self.patch_embed = PatchEmbed3D(
            img_size, temporal_size, patch_size, in_channels, embed_dim
        )

        self.num_frames = self.patch_embed.num_patches_t
        self.num_patches_h = self.patch_embed.num_patches_h
        self.num_patches_w = self.patch_embed.num_patches_w
        self.num_spatial_tokens = self.patch_embed.num_patches_h * self.patch_embed.num_patches_w
        self.patches_per_frame = self.num_spatial_tokens

        self.factorized_pos_embed = FactorizedPositionEncoding3D(
            self.num_frames,
            self.num_patches_h,
            self.num_patches_w,
            embed_dim,
        )

        self.aspect_ratio_embed = nn.Linear(1, embed_dim)

        self.stages = nn.ModuleList([
            EncoderStage(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(self.num_stages)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

        if use_videomae_init:
            if videomae_model is not None:
                self._load_videomae_weights(videomae_model)
        elif use_vggt_init:
            self._load_vggt_weights(vggt_checkpoint)
        elif use_monst3r_init:
            if monst3r_checkpoint is None:
                raise ValueError("monst3r_checkpoint must be provided when use_monst3r_init=True")
            self._load_monst3r_weights(monst3r_checkpoint)
        elif use_timm_init and TIMM_AVAILABLE:
            self._load_timm_weights(timm_model)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict.pop(prefix + "pos_embed", None)
        legacy_blocks_prefix = prefix + "blocks."
        for key in list(state_dict.keys()):
            if not key.startswith(legacy_blocks_prefix):
                continue
            remainder = key[len(legacy_blocks_prefix):]
            if "." not in remainder:
                continue
            block_idx_str, rest = remainder.split(".", 1)
            if not block_idx_str.isdigit():
                continue
            block_idx = int(block_idx_str)
            stage_idx = block_idx // 2
            if stage_idx >= self.num_stages:
                continue
            stage_block = "local_block" if block_idx % 2 == 0 else "global_block"
            if rest.startswith("attn.attention."):
                rest = "attn." + rest[len("attn.attention."):]
            new_key = f"{prefix}stages.{stage_idx}.{stage_block}.{rest}"
            if new_key not in state_dict:
                state_dict[new_key] = state_dict[key]
            del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _init_weights(self):
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _iter_subblocks(self):
        for stage in self.stages:
            yield stage.local_block
            yield stage.global_block

    def _load_timm_weights(self, model_name: str):
        """Load pretrained weights from a timm ViT model."""
        if not TIMM_AVAILABLE:
            print("timm not available, skipping pretrained weight loading")
            return

        print(f"Loading pretrained weights from timm model: {model_name}")
        timm_model = timm.create_model(model_name, pretrained=True)

        timm_blocks = list(timm_model.blocks.children())
        for i, (our_block, timm_block) in enumerate(zip(self._iter_subblocks(), timm_blocks)):
            if i >= len(timm_blocks):
                break

            our_block.attn.qkv.weight.data.copy_(timm_block.attn.qkv.weight.data)
            our_block.attn.qkv.bias.data.copy_(timm_block.attn.qkv.bias.data)
            our_block.attn.proj.weight.data.copy_(timm_block.attn.proj.weight.data)
            our_block.attn.proj.bias.data.copy_(timm_block.attn.proj.bias.data)

            our_block.mlp.fc1.weight.data.copy_(timm_block.mlp.fc1.weight.data)
            our_block.mlp.fc1.bias.data.copy_(timm_block.mlp.fc1.bias.data)
            our_block.mlp.fc2.weight.data.copy_(timm_block.mlp.fc2.weight.data)
            our_block.mlp.fc2.bias.data.copy_(timm_block.mlp.fc2.bias.data)

            our_block.norm1.weight.data.copy_(timm_block.norm1.weight.data)
            our_block.norm1.bias.data.copy_(timm_block.norm1.bias.data)
            our_block.norm2.weight.data.copy_(timm_block.norm2.weight.data)
            our_block.norm2.bias.data.copy_(timm_block.norm2.bias.data)

        self.norm.weight.data.copy_(timm_model.norm.weight.data)
        self.norm.bias.data.copy_(timm_model.norm.bias.data)

        print(f"Loaded pretrained weights for {min(self.depth, len(timm_blocks))} blocks")

    def _load_state_dict_file(self, checkpoint_path: Path) -> Optional[dict[str, torch.Tensor]]:
        if checkpoint_path.suffix == ".safetensors":
            if not SAFETENSORS_AVAILABLE:
                print(
                    f"safetensors not available, cannot load VideoMAE checkpoint: {checkpoint_path}"
                )
                return None
            return safetensors_load_file(str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                return checkpoint["model"]
        return checkpoint

    def _maybe_snapshot_hf_checkpoint(self, model_name: str) -> Optional[Path]:
        """Resolve a HuggingFace repo id to a local snapshot with raw weights."""
        path_like = Path(model_name).expanduser()
        if path_like.is_absolute() or path_like.exists() or model_name.count("/") != 1:
            return None

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub not available, cannot download VideoMAE V2 checkpoint")
            return None

        try:
            snapshot_dir = snapshot_download(
                repo_id=model_name,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "model.bin",
                ],
            )
        except Exception as exc:
            print(f"Failed to download VideoMAE V2 checkpoint from {model_name}: {exc}")
            return None
        return Path(snapshot_dir)

    def _load_videomae_v2_weights(self, model_name: str) -> bool:
        model_path = Path(model_name).expanduser()
        if not model_path.exists():
            snapshot_path = self._maybe_snapshot_hf_checkpoint(model_name)
            if snapshot_path is None:
                return False
            model_path = snapshot_path

        config_path = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
        if not config_path.exists():
            return False

        try:
            config_data = json.loads(config_path.read_text())
        except Exception:
            return False

        model_type = str(config_data.get("model_type", "")).lower()
        architectures = [str(item).lower() for item in config_data.get("architectures", [])]
        if "videomaev2" not in model_type and not any("videomaev2" in item for item in architectures):
            return False

        candidate_files = [model_path] if model_path.is_file() else [
            model_path / "model.safetensors",
            model_path / "pytorch_model.bin",
            model_path / "model.bin",
        ]

        state_dict = None
        for candidate in candidate_files:
            if not candidate.exists():
                continue
            loaded = self._load_state_dict_file(candidate)
            if isinstance(loaded, dict):
                state_dict = loaded
                break

        if not isinstance(state_dict, dict):
            print(f"Failed to load VideoMAE V2 checkpoint state from {model_name}")
            return False

        prefix = "model." if any(key.startswith("model.blocks.") for key in state_dict) else ""
        block_prefix = f"{prefix}blocks."
        if not any(key.startswith(block_prefix) for key in state_dict):
            print(f"VideoMAE V2 checkpoint at {model_name} does not contain {block_prefix}* keys")
            return False

        print(f"Loading VideoMAE V2 weights for custom encoder init: {model_name}")

        loaded_patch_embed = False
        patch_weight = state_dict.get(f"{prefix}patch_embed.proj.weight")
        patch_bias = state_dict.get(f"{prefix}patch_embed.proj.bias")
        if patch_weight is not None and patch_weight.shape == self.patch_embed.proj.weight.shape:
            self.patch_embed.proj.weight.data.copy_(
                patch_weight.to(dtype=self.patch_embed.proj.weight.dtype)
            )
            if patch_bias is not None and self.patch_embed.proj.bias is not None:
                self.patch_embed.proj.bias.data.copy_(
                    patch_bias.to(dtype=self.patch_embed.proj.bias.dtype)
                )
            loaded_patch_embed = True
        elif patch_weight is not None:
            print(
                "Skipping VideoMAE V2 patch embedding init due to shape mismatch: "
                f"checkpoint={tuple(patch_weight.shape)} "
                f"model={tuple(self.patch_embed.proj.weight.shape)}"
            )

        block_indices = sorted(
            {
                int(key[len(block_prefix):].split(".", 1)[0])
                for key in state_dict.keys()
                if key.startswith(block_prefix)
                and key[len(block_prefix):].split(".", 1)[0].isdigit()
            }
        )
        loaded_blocks = 0
        fully_loaded_blocks = 0
        for our_block, block_idx in zip(self._iter_subblocks(), block_indices):
            source_prefix = f"{block_prefix}{block_idx}."
            qkv_weight = state_dict.get(f"{source_prefix}attn.qkv.weight")
            if qkv_weight is None or qkv_weight.shape != our_block.attn.qkv.weight.shape:
                continue

            our_block.attn.qkv.weight.data.copy_(
                qkv_weight.to(dtype=our_block.attn.qkv.weight.dtype)
            )
            loaded_qkv_bias = our_block.attn.qkv.bias is None
            if our_block.attn.qkv.bias is not None:
                q_bias = state_dict.get(f"{source_prefix}attn.q_bias")
                v_bias = state_dict.get(f"{source_prefix}attn.v_bias")
                if q_bias is not None and v_bias is not None:
                    k_bias = torch.zeros_like(q_bias)
                    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                    if qkv_bias.shape == our_block.attn.qkv.bias.shape:
                        our_block.attn.qkv.bias.data.copy_(
                            qkv_bias.to(dtype=our_block.attn.qkv.bias.dtype)
                        )
                        loaded_qkv_bias = True
                else:
                    qkv_bias = state_dict.get(f"{source_prefix}attn.qkv.bias")
                    if qkv_bias is not None and qkv_bias.shape == our_block.attn.qkv.bias.shape:
                        our_block.attn.qkv.bias.data.copy_(
                            qkv_bias.to(dtype=our_block.attn.qkv.bias.dtype)
                        )
                        loaded_qkv_bias = True

            pairs = (
                (our_block.attn.proj.weight, f"{source_prefix}attn.proj.weight"),
                (our_block.attn.proj.bias, f"{source_prefix}attn.proj.bias"),
                (our_block.mlp.fc1.weight, f"{source_prefix}mlp.fc1.weight"),
                (our_block.mlp.fc1.bias, f"{source_prefix}mlp.fc1.bias"),
                (our_block.mlp.fc2.weight, f"{source_prefix}mlp.fc2.weight"),
                (our_block.mlp.fc2.bias, f"{source_prefix}mlp.fc2.bias"),
                (our_block.norm1.weight, f"{source_prefix}norm1.weight"),
                (our_block.norm1.bias, f"{source_prefix}norm1.bias"),
                (our_block.norm2.weight, f"{source_prefix}norm2.weight"),
                (our_block.norm2.bias, f"{source_prefix}norm2.bias"),
            )
            loaded_pair_count = 0
            for target, source_key in pairs:
                source_tensor = state_dict.get(source_key)
                if source_tensor is not None and source_tensor.shape == target.shape:
                    target.data.copy_(source_tensor.to(dtype=target.dtype))
                    loaded_pair_count += 1

            loaded_blocks += 1
            if loaded_qkv_bias and loaded_pair_count == len(pairs):
                fully_loaded_blocks += 1

        norm_weight = state_dict.get(f"{prefix}fc_norm.weight")
        norm_bias = state_dict.get(f"{prefix}fc_norm.bias")
        if norm_weight is None or norm_bias is None:
            norm_weight = state_dict.get(f"{prefix}norm.weight")
            norm_bias = state_dict.get(f"{prefix}norm.bias")
        if norm_weight is not None and norm_bias is not None:
            if norm_weight.shape == self.norm.weight.shape and norm_bias.shape == self.norm.bias.shape:
                self.norm.weight.data.copy_(norm_weight.to(dtype=self.norm.weight.dtype))
                self.norm.bias.data.copy_(norm_bias.to(dtype=self.norm.bias.dtype))

        if loaded_blocks == 0:
            print(f"VideoMAE V2 checkpoint at {model_name} did not match any encoder blocks")
            return False
        if loaded_patch_embed and fully_loaded_blocks != loaded_blocks:
            print(
                "VideoMAE V2 checkpoint only partially matched the custom encoder: "
                f"fully_loaded_blocks={fully_loaded_blocks}/{loaded_blocks}. "
                "Check embed_dim, patch_size, and mlp_ratio."
            )
            return False

        print(
            f"Loaded VideoMAE V2 weights for {fully_loaded_blocks} encoder blocks "
            f"(patch_embed={'yes' if loaded_patch_embed else 'no'})"
        )
        return True

    def _load_videomae_weights(self, model_name: str):
        """Initialize the staged encoder from VideoMAE weights."""
        if model_name is None:
            return

        if self._load_videomae_v2_weights(model_name):
            return

        if not VIDEOMAE_AVAILABLE:
            print("transformers not available, skipping VideoMAE weight loading")
            return

        print(f"Loading VideoMAE weights for custom encoder init: {model_name}")
        backbone = VideoMAEModel.from_pretrained(model_name)

        if backbone.config.hidden_size != self.embed_dim:
            print(
                "VideoMAE hidden size does not match custom encoder. "
                f"Expected {self.embed_dim}, got {backbone.config.hidden_size}. "
                "Skipping VideoMAE init."
            )
            return

        patch_proj = backbone.embeddings.patch_embeddings.projection
        if patch_proj.weight.shape == self.patch_embed.proj.weight.shape:
            self.patch_embed.proj.weight.data.copy_(patch_proj.weight.data)
            if patch_proj.bias is not None and self.patch_embed.proj.bias is not None:
                self.patch_embed.proj.bias.data.copy_(patch_proj.bias.data)

        videomae_blocks = list(backbone.encoder.layer)
        for our_block, videomae_block in zip(self._iter_subblocks(), videomae_blocks):
            source_attn = videomae_block.attention.attention

            qkv_weight = torch.cat(
                [
                    source_attn.query.weight.data,
                    source_attn.key.weight.data,
                    source_attn.value.weight.data,
                ],
                dim=0,
            )
            our_block.attn.qkv.weight.data.copy_(qkv_weight)

            if our_block.attn.qkv.bias is not None:
                if source_attn.q_bias is not None and source_attn.v_bias is not None:
                    k_bias = torch.zeros_like(source_attn.q_bias.data)
                    qkv_bias = torch.cat(
                        [
                            source_attn.q_bias.data,
                            k_bias,
                            source_attn.v_bias.data,
                        ],
                        dim=0,
                    )
                else:
                    qkv_bias = torch.zeros_like(our_block.attn.qkv.bias.data)
                our_block.attn.qkv.bias.data.copy_(qkv_bias)

            our_block.attn.proj.weight.data.copy_(videomae_block.attention.output.dense.weight.data)
            our_block.attn.proj.bias.data.copy_(videomae_block.attention.output.dense.bias.data)

            our_block.mlp.fc1.weight.data.copy_(videomae_block.intermediate.dense.weight.data)
            our_block.mlp.fc1.bias.data.copy_(videomae_block.intermediate.dense.bias.data)
            our_block.mlp.fc2.weight.data.copy_(videomae_block.output.dense.weight.data)
            our_block.mlp.fc2.bias.data.copy_(videomae_block.output.dense.bias.data)

            our_block.norm1.weight.data.copy_(videomae_block.layernorm_before.weight.data)
            our_block.norm1.bias.data.copy_(videomae_block.layernorm_before.bias.data)
            our_block.norm2.weight.data.copy_(videomae_block.layernorm_after.weight.data)
            our_block.norm2.bias.data.copy_(videomae_block.layernorm_after.bias.data)

        if backbone.layernorm is not None:
            self.norm.weight.data.copy_(backbone.layernorm.weight.data)
            self.norm.bias.data.copy_(backbone.layernorm.bias.data)

    def _load_vggt_weights(self, checkpoint: str):
        """Initialize attention blocks from VGGT (facebook/VGGT-1B).

        VGGT uses alternating within-frame and cross-frame attention blocks in its
        aggregator — architecturally identical to D4RT's Local/Global stages.
        embed_dim=1024 (matches D4RT-large). patch_size=14 (DINOv2), so patch_embed
        is skipped; only the 24 aggregator blocks are transferred.

        Args:
            checkpoint: HuggingFace model ID or local path, e.g. 'facebook/VGGT-1B'
        """
        try:
            import huggingface_hub
        except ImportError:
            print("huggingface_hub not available, skipping VGGT weight loading")
            return

        print(f"Loading VGGT aggregator weights from: {checkpoint}")
        try:
            from huggingface_hub import hf_hub_download
            import os
            # Try loading as a local path first, then HuggingFace
            if os.path.isfile(checkpoint):
                state_dict = torch.load(checkpoint, map_location='cpu')
            else:
                ckpt_path = hf_hub_download(repo_id=checkpoint, filename="model.pt")
                state_dict = torch.load(ckpt_path, map_location='cpu')
        except Exception as e:
            print(f"Failed to load VGGT checkpoint: {e}")
            return

        if 'model' in state_dict:
            state_dict = state_dict['model']

        # VGGT aggregator blocks: aggregator.layers.{i}.{frame_attn|global_attn}.{...}
        # Map to D4RT stages: stages.{i}.{local_block|global_block}.{...}
        # Collect aggregator block keys grouped by layer index
        agg_prefix = 'aggregator.layers.'
        agg_blocks: dict[int, dict[str, torch.Tensor]] = {}
        for k, v in state_dict.items():
            if not k.startswith(agg_prefix):
                continue
            rest = k[len(agg_prefix):]
            dot = rest.index('.')
            layer_idx = int(rest[:dot])
            sub_key = rest[dot + 1:]
            agg_blocks.setdefault(layer_idx, {})[sub_key] = v

        if not agg_blocks:
            print("VGGT: no aggregator.layers.* keys found — check checkpoint structure")
            return

        loaded = 0
        for stage_idx, stage in enumerate(self.stages):
            if stage_idx not in agg_blocks:
                break
            block_dict = agg_blocks[stage_idx]

            def _copy_attn_block(our_block, prefix):
                # Try standard VGGT AlternatingAttentionBlock sub-key layout
                for qkv_key in (f'{prefix}.attn.qkv.weight', f'{prefix}.qkv.weight'):
                    if qkv_key in block_dict and our_block.attn.qkv.weight.shape == block_dict[qkv_key].shape:
                        our_block.attn.qkv.weight.data.copy_(block_dict[qkv_key])
                        break
                for qkv_key in (f'{prefix}.attn.qkv.bias', f'{prefix}.qkv.bias'):
                    if qkv_key in block_dict and our_block.attn.qkv.bias is not None:
                        if our_block.attn.qkv.bias.shape == block_dict[qkv_key].shape:
                            our_block.attn.qkv.bias.data.copy_(block_dict[qkv_key])
                        break
                for proj_key in (f'{prefix}.attn.proj.weight', f'{prefix}.proj.weight'):
                    if proj_key in block_dict and our_block.attn.proj.weight.shape == block_dict[proj_key].shape:
                        our_block.attn.proj.weight.data.copy_(block_dict[proj_key])
                        break
                for proj_key in (f'{prefix}.attn.proj.bias', f'{prefix}.proj.bias'):
                    if proj_key in block_dict and our_block.attn.proj.bias is not None:
                        if our_block.attn.proj.bias.shape == block_dict[proj_key].shape:
                            our_block.attn.proj.bias.data.copy_(block_dict[proj_key])
                        break
                for fc1_key in (f'{prefix}.mlp.fc1.weight', f'{prefix}.mlp.0.weight'):
                    if fc1_key in block_dict and our_block.mlp.fc1.weight.shape == block_dict[fc1_key].shape:
                        our_block.mlp.fc1.weight.data.copy_(block_dict[fc1_key])
                        break
                for fc1_key in (f'{prefix}.mlp.fc1.bias', f'{prefix}.mlp.0.bias'):
                    if fc1_key in block_dict and our_block.mlp.fc1.bias is not None:
                        if our_block.mlp.fc1.bias.shape == block_dict[fc1_key].shape:
                            our_block.mlp.fc1.bias.data.copy_(block_dict[fc1_key])
                        break
                for fc2_key in (f'{prefix}.mlp.fc2.weight', f'{prefix}.mlp.2.weight'):
                    if fc2_key in block_dict and our_block.mlp.fc2.weight.shape == block_dict[fc2_key].shape:
                        our_block.mlp.fc2.weight.data.copy_(block_dict[fc2_key])
                        break
                for fc2_key in (f'{prefix}.mlp.fc2.bias', f'{prefix}.mlp.2.bias'):
                    if fc2_key in block_dict and our_block.mlp.fc2.bias is not None:
                        if our_block.mlp.fc2.bias.shape == block_dict[fc2_key].shape:
                            our_block.mlp.fc2.bias.data.copy_(block_dict[fc2_key])
                        break
                for n1_key in (f'{prefix}.norm1.weight', f'{prefix}.ln1.weight'):
                    if n1_key in block_dict and our_block.norm1.weight.shape == block_dict[n1_key].shape:
                        our_block.norm1.weight.data.copy_(block_dict[n1_key])
                        our_block.norm1.bias.data.copy_(block_dict[n1_key.replace('weight', 'bias')])
                        break
                for n2_key in (f'{prefix}.norm2.weight', f'{prefix}.ln2.weight'):
                    if n2_key in block_dict and our_block.norm2.weight.shape == block_dict[n2_key].shape:
                        our_block.norm2.weight.data.copy_(block_dict[n2_key])
                        our_block.norm2.bias.data.copy_(block_dict[n2_key.replace('weight', 'bias')])
                        break

            _copy_attn_block(stage.local_block, 'frame_attn')
            _copy_attn_block(stage.global_block, 'global_attn')
            loaded += 1

        print(f"VGGT: loaded {loaded}/{self.num_stages} stages (patch_embed skipped — re-initialized from scratch)")

    def _load_monst3r_weights(self, checkpoint_path: str):
        """Initialize attention blocks from MonST3R / DUSt3R / CroCo v2 checkpoint.

        These models use a CroCo v2 ViT-Large encoder: embed_dim=1024, depth=24,
        heads=16, patch_size=16. The patch_size matches D4RT's spatial patch size
        but the embedding is 2D (Conv2d), so patch_embed is skipped. The 24 standard
        ViT blocks are transferred block-by-block.

        Args:
            checkpoint_path: Path to .pth checkpoint file
        """
        print(f"Loading MonST3R/DUSt3R encoder weights from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f"Failed to load MonST3R checkpoint: {e}")
            return

        state_dict = ckpt.get('model', ckpt)

        # CroCo/DUSt3R encoder key prefix: 'encoder.patch_embed', 'encoder.blocks.{i}.*'
        # Also try 'backbone.encoder.blocks.{i}.*' for some variants
        for enc_prefix in ('encoder.blocks.', 'backbone.encoder.blocks.', 'enc_blocks.'):
            blocks: dict[int, dict[str, torch.Tensor]] = {}
            for k, v in state_dict.items():
                if not k.startswith(enc_prefix):
                    continue
                rest = k[len(enc_prefix):]
                dot = rest.index('.')
                block_idx = int(rest[:dot])
                sub_key = rest[dot + 1:]
                blocks.setdefault(block_idx, {})[sub_key] = v
            if blocks:
                break

        if not blocks:
            print("MonST3R: no encoder block keys found — check checkpoint structure")
            return

        loaded = 0
        for our_block, block_idx in zip(self._iter_subblocks(), sorted(blocks.keys())):
            bd = blocks[block_idx]

            def _try_copy(dst: torch.Tensor, *keys):
                for k in keys:
                    if k in bd and dst.shape == bd[k].shape:
                        dst.data.copy_(bd[k])
                        return True
                return False

            # CroCo/DUSt3R uses standard ViT block layout
            _try_copy(our_block.attn.qkv.weight, 'attn.qkv.weight')
            if our_block.attn.qkv.bias is not None:
                _try_copy(our_block.attn.qkv.bias, 'attn.qkv.bias')
            _try_copy(our_block.attn.proj.weight, 'attn.proj.weight')
            if our_block.attn.proj.bias is not None:
                _try_copy(our_block.attn.proj.bias, 'attn.proj.bias')
            _try_copy(our_block.mlp.fc1.weight, 'mlp.fc1.weight', 'mlp.0.weight')
            if our_block.mlp.fc1.bias is not None:
                _try_copy(our_block.mlp.fc1.bias, 'mlp.fc1.bias', 'mlp.0.bias')
            _try_copy(our_block.mlp.fc2.weight, 'mlp.fc2.weight', 'mlp.2.weight')
            if our_block.mlp.fc2.bias is not None:
                _try_copy(our_block.mlp.fc2.bias, 'mlp.fc2.bias', 'mlp.2.bias')
            _try_copy(our_block.norm1.weight, 'norm1.weight')
            _try_copy(our_block.norm1.bias, 'norm1.bias')
            _try_copy(our_block.norm2.weight, 'norm2.weight')
            _try_copy(our_block.norm2.bias, 'norm2.bias')
            loaded += 1

        # Try to load final norm
        for norm_key in ('encoder.norm.weight', 'backbone.encoder.norm.weight', 'enc_norm.weight'):
            if norm_key in state_dict and self.norm.weight.shape == state_dict[norm_key].shape:
                self.norm.weight.data.copy_(state_dict[norm_key])
                self.norm.bias.data.copy_(state_dict[norm_key.replace('weight', 'bias')])
                break

        print(f"MonST3R: loaded {loaded}/{self.depth} blocks (patch_embed skipped — re-initialized from scratch)")

    def _tokenize_video(self, video: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embed(video)
        patch_tokens = patch_tokens.reshape(
            video.shape[0],
            self.num_frames,
            self.num_spatial_tokens,
            self.embed_dim,
        )
        expected_shape = (video.shape[0], self.num_frames, self.num_spatial_tokens, self.embed_dim)
        if patch_tokens.ndim != 4 or patch_tokens.shape != expected_shape:
            raise RuntimeError(
                f"Tokenizer must produce [B, T, N_spatial, C]={expected_shape}, got {tuple(patch_tokens.shape)}"
            )
        return patch_tokens

    def _add_patch_position_encoding(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        pos = self.factorized_pos_embed().reshape(
            1,
            self.num_frames,
            self.num_spatial_tokens,
            self.embed_dim,
        )
        pos = pos.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        if pos.shape != (1, self.num_frames, self.num_spatial_tokens, self.embed_dim):
            raise RuntimeError("Patch positional encoding has an unexpected shape")
        return patch_tokens + pos

    def _build_local_special_tokens(
        self,
        aspect_ratio: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.use_aspect_ratio_token or aspect_ratio is None:
            return None

        ratio = canonicalize_aspect_ratio(aspect_ratio).to(device=device)
        aspect_token = self.aspect_ratio_embed(ratio).to(dtype=dtype)
        return aspect_token.unsqueeze(1).unsqueeze(2).expand(-1, self.num_frames, -1, -1)

    def _compose_encoder_state(
        self,
        patch_tokens: torch.Tensor,
        local_special_tokens: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        if local_special_tokens is None:
            x = patch_tokens
            num_special_tokens_local = 0
        else:
            if local_special_tokens.shape[:2] != patch_tokens.shape[:2]:
                raise RuntimeError("Local special tokens must align with [B, T] of patch tokens")
            if local_special_tokens.shape[-1] != patch_tokens.shape[-1]:
                raise RuntimeError("Local special tokens must match the patch token hidden size")
            x = torch.cat([patch_tokens, local_special_tokens], dim=2)
            num_special_tokens_local = local_special_tokens.shape[2]

        expected_p = self.num_spatial_tokens + num_special_tokens_local
        expected_shape = (patch_tokens.shape[0], self.num_frames, expected_p, self.embed_dim)
        if x.shape != expected_shape:
            raise RuntimeError(
                f"Unified encoder state must be [B, T, P, C]={expected_shape}, got {tuple(x.shape)}"
            )
        return x, num_special_tokens_local

    def _flatten_output_tokens(self, x: torch.Tensor, num_special_tokens_local: int) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected encoder state shaped (B, T, P, C), got {tuple(x.shape)}")
        if not self.keep_special_tokens_in_output and num_special_tokens_local:
            x = x[:, :, :self.num_spatial_tokens]
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

    def forward(
        self,
        video: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W), (B, T, C, H, W), or (B, T, H, W, C) video tensor
            aspect_ratio: (B, 1) original width/height ratio, or legacy (B, 2)

        Returns:
            features: (B, N, embed_dim) flattened encoder memory
        """
        video = canonicalize_video(video)

        patch_tokens = self._tokenize_video(video)
        patch_tokens = self._add_patch_position_encoding(patch_tokens)
        local_special_tokens = self._build_local_special_tokens(
            aspect_ratio,
            dtype=patch_tokens.dtype,
            device=patch_tokens.device,
        )
        x, num_special_tokens_local = self._compose_encoder_state(patch_tokens, local_special_tokens)

        for stage in self.stages:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda tensor, module=stage: module(tensor),
                    x,
                    use_reentrant=False,
                )
            else:
                x = stage(x)

        x = self.norm(x)
        return self._flatten_output_tokens(x, num_special_tokens_local)


class TimmVideoEncoder(nn.Module):
    """Video encoder that wraps a timm ViT model.

    Processes video frame-by-frame with a timm ViT, then applies
    temporal attention to aggregate features across frames.
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        temporal_size: int = 48,
        temporal_stride: int = 2,
        freeze_backbone: bool = False
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for TimmVideoEncoder")

        # Create timm model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        self.embed_dim = self.backbone.embed_dim

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Temporal aggregation
        self.temporal_stride = temporal_stride
        self.num_frames = temporal_size // temporal_stride

        # Temporal position embedding
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_frames, self.embed_dim)
        )

        # Temporal attention blocks
        self.temporal_blocks = nn.ModuleList([
            EncoderBlock(
                self.embed_dim, num_heads=12, mlp_ratio=4.0,
                attention_type='global'
            )
            for _ in range(4)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def forward(
        self,
        video: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W), (B, T, C, H, W), or (B, T, H, W, C) video tensor

        Returns:
            features: (B, T'*N, embed_dim) where T' = T // temporal_stride
        """
        video = canonicalize_video(video)

        B, C, T, H, W = video.shape

        # Subsample temporally
        frame_indices = list(range(0, T, self.temporal_stride))[:self.num_frames]
        video = video[:, :, frame_indices]
        T_sub = len(frame_indices)

        # Process each frame with backbone
        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        video_flat = video.permute(0, 2, 1, 3, 4).reshape(B * T_sub, C, H, W)

        # Get features from timm backbone
        features = self.backbone.forward_features(video_flat)  # (B*T, N, C)
        N = features.shape[1]

        # Reshape back: (B*T, N, C) -> (B, T, N, C)
        features = features.view(B, T_sub, N, self.embed_dim)

        # Add temporal position embedding
        features = features + self.temporal_pos_embed[:, :T_sub].unsqueeze(2)

        # Reshape for temporal attention: (B, T*N, C)
        features = features.view(B, T_sub * N, self.embed_dim)

        # Apply temporal attention blocks
        for block in self.temporal_blocks:
            features = block(features)

        features = self.norm(features)

        return features


class VideoMAEEncoder(nn.Module):
    """Video encoder using pretrained VideoMAE from HuggingFace.

    VideoMAE is specifically pretrained on video data using masked autoencoding,
    making it well-suited for video understanding tasks.

    Supported pretrained models:
        - MCG-NJU/videomae-base (ViT-B, 86M params)
        - MCG-NJU/videomae-large (ViT-L, 305M params)
        - MCG-NJU/videomae-huge (ViT-H, 633M params)
        - MCG-NJU/videomae-base-finetuned-kinetics (finetuned on K400)
        - MCG-NJU/videomae-large-finetuned-kinetics (finetuned on K400)
    """

    def __init__(
        self,
        model_name: str = 'MCG-NJU/videomae-base',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_frames: int = 16,
        image_size: Optional[int] = None,
        use_mean_pooling: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model name or path
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            num_frames: Number of frames VideoMAE expects
            image_size: Input spatial resolution
            use_mean_pooling: Whether to mean pool over time dimension
        """
        super().__init__()

        if not VIDEOMAE_AVAILABLE:
            raise ImportError(
                "transformers library required for VideoMAE. "
                "Install with: pip install transformers"
            )

        print(f"Loading VideoMAE encoder: {model_name}")

        if pretrained:
            self.backbone = VideoMAEModel.from_pretrained(model_name)
        else:
            config = VideoMAEConfig.from_pretrained(model_name)
            self.backbone = VideoMAEModel(config)

        self.embed_dim = self.backbone.config.hidden_size
        self.use_mean_pooling = use_mean_pooling
        backbone_image_size = self.backbone.config.image_size
        if isinstance(backbone_image_size, (list, tuple)):
            backbone_image_size = backbone_image_size[0]
        self.patch_size = (
            self.backbone.config.tubelet_size,
            self.backbone.config.patch_size,
            self.backbone.config.patch_size
        )
        target_image_size = int(image_size) if image_size is not None else int(backbone_image_size)
        self._set_backbone_geometry(num_frames=num_frames, image_size=target_image_size)
        self.num_patches_t = num_frames // self.patch_size[0]
        self.num_patches_h = target_image_size // self.patch_size[1]
        self.num_patches_w = target_image_size // self.patch_size[2]
        self.factorized_pos_bias = FactorizedPositionEncoding3D(
            self.num_patches_t,
            self.num_patches_h,
            self.num_patches_w,
            self.embed_dim,
            scale_init=1e-3,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("VideoMAE backbone frozen")

        # Aspect ratio embedding
        self.aspect_ratio_embed = nn.Linear(1, self.embed_dim)

        # Additional projection for interleaved local/global attention pattern
        # VideoMAE already has global attention, we add local attention layers
        self.local_attention_layers = nn.ModuleList([
            EncoderBlock(
                self.embed_dim,
                num_heads=self.backbone.config.num_attention_heads,
                mlp_ratio=4.0,
                attention_type='local'
            )
            for _ in range(4)  # Add 4 local attention layers
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

    def _set_backbone_geometry(self, num_frames: int, image_size: int):
        """Retarget VideoMAE's fixed sin-cos positions to the requested clip geometry."""
        if num_frames % self.patch_size[0] != 0:
            raise ValueError(
                f"num_frames={num_frames} must be divisible by tubelet size {self.patch_size[0]}"
            )
        if image_size % self.patch_size[1] != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by spatial patch size {self.patch_size[1]}"
            )

        patch_embeddings = self.backbone.embeddings.patch_embeddings
        image_h = image_size
        image_w = image_size
        patch_h, patch_w = patch_embeddings.patch_size
        num_patches = (
            (image_h // patch_h)
            * (image_w // patch_w)
            * (num_frames // patch_embeddings.tubelet_size)
        )

        self.backbone.config.num_frames = num_frames
        self.backbone.config.image_size = image_size
        patch_embeddings.image_size = (image_size, image_size)
        patch_embeddings.num_patches = num_patches
        self.backbone.embeddings.num_patches = num_patches
        self.backbone.embeddings.position_embeddings = build_sinusoid_encoding_table(
            num_patches,
            self.embed_dim,
        )
        self.num_frames = num_frames
        self.image_size = image_size

    def forward(
        self,
        video: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W), (B, T, C, H, W), or (B, T, H, W, C) video tensor
            aspect_ratio: (B, 1) original width/height ratio, or legacy (B, 2)

        Returns:
            features: (B, N, embed_dim) Global Scene Representation F
        """
        video = canonicalize_video(video)

        B, C, T, H, W = video.shape

        # VideoMAE expects (B, T, C, H, W) format
        video_mae_input = video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # Subsample or pad frames to match VideoMAE's expected num_frames
        if T != self.num_frames:
            if T > self.num_frames:
                # Subsample frames uniformly
                indices = torch.linspace(0, T - 1, self.num_frames).long()
                video_mae_input = video_mae_input[:, indices]
            else:
                # Pad by repeating last frame
                pad_frames = self.num_frames - T
                last_frame = video_mae_input[:, -1:].expand(-1, pad_frames, -1, -1, -1)
                video_mae_input = torch.cat([video_mae_input, last_frame], dim=1)

        if H != self.image_size or W != self.image_size:
            video_mae_input = video_mae_input.reshape(B * self.num_frames, C, H, W)
            video_mae_input = F.interpolate(
                video_mae_input,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
            video_mae_input = video_mae_input.view(
                B, self.num_frames, C, self.image_size, self.image_size
            )

        patch_embeddings = self.backbone.embeddings.patch_embeddings(video_mae_input)
        position_embeddings = self.backbone.embeddings.position_embeddings
        features = patch_embeddings + position_embeddings.detach().type_as(patch_embeddings).to(
            device=patch_embeddings.device,
            copy=True,
        )
        features = features + self.factorized_pos_bias().type_as(features)

        num_special_tokens = 0
        if aspect_ratio is not None:
            ar_embed = self.aspect_ratio_embed(canonicalize_aspect_ratio(aspect_ratio))  # (B, embed_dim)
            ar_token = ar_embed.unsqueeze(1)  # (B, 1, embed_dim)
            features = torch.cat([features, ar_token], dim=1)
            num_special_tokens = 1

        encoder_outputs = self.backbone.encoder(features)
        features = encoder_outputs.last_hidden_state
        if self.backbone.layernorm is not None:
            features = self.backbone.layernorm(features)

        # Calculate spatial dimensions for local attention.
        num_patches_t = self.num_patches_t
        patches_per_frame = self.num_patches_h * self.num_patches_w

        for local_layer in self.local_attention_layers:
            features = local_layer(
                features,
                num_patches_t,
                patches_per_frame,
                num_special_tokens=num_special_tokens,
            )

        features = self.norm(features)

        # Remove aspect ratio token from output
        if num_special_tokens:
            features = features[:, :-num_special_tokens]

        return features


def create_encoder(
    variant: str = 'base',
    use_timm: bool = False,
    use_videomae: bool = True,
    use_videomae_v2: bool = False,
    use_vggt: bool = False,
    use_monst3r: bool = False,
    pretrained: bool = True,
    use_videomae_backbone: bool = False,
    vggt_checkpoint: Optional[str] = None,
    monst3r_checkpoint: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """Create encoder with predefined configurations.

    Args:
        variant: One of 'base', 'large', 'huge', 'giant'
        use_timm: Whether to use timm-based encoder
        use_videomae: Whether to initialize from VideoMAE V1 weights
        use_videomae_v2: Whether to initialize from VideoMAE V2 weights (stronger, drop-in upgrade)
        use_vggt: Whether to initialize attention blocks from VGGT (geometric pretraining)
        use_monst3r: Whether to initialize attention blocks from MonST3R/DUSt3R/CroCo
        pretrained: Whether to load pretrained weights
        vggt_checkpoint: Path or HuggingFace ID for VGGT checkpoint (default: 'facebook/VGGT-1B')
        monst3r_checkpoint: Path to MonST3R/DUSt3R .pth checkpoint file

    Returns:
        Configured encoder
    """
    configs = {
        'base': dict(embed_dim=768, depth=12, num_heads=12),
        'large': dict(embed_dim=1024, depth=24, num_heads=16),
        'huge': dict(embed_dim=1280, depth=32, num_heads=16),
        'giant': dict(embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48 / 11),
    }

    timm_models = {
        'base': 'vit_base_patch16_224',
        'large': 'vit_large_patch16_224',
        'huge': 'vit_huge_patch14_224',
        'giant': 'vit_giant_patch14_224',
    }

    videomae_models = {
        'base': 'MCG-NJU/videomae-base',
        'large': 'MCG-NJU/videomae-large',
        'huge': 'MCG-NJU/videomae-huge',
        'giant': 'MCG-NJU/videomae-huge',  # No giant, use huge
    }

    # VideoMAE V2: stronger dual-masking pretraining on 1M clips (drop-in replacement)
    videomae_v2_models = {
        'base': 'OpenGVLab/VideoMAEv2-Base',
        'large': 'OpenGVLab/VideoMAEv2-Large',
        'huge': 'OpenGVLab/VideoMAEv2-Huge',
        'giant': 'OpenGVLab/VideoMAEv2-giant',
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

    videomae_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ['freeze_backbone', 'num_frames', 'use_mean_pooling', 'temporal_size', 'img_size']
    }
    if 'num_frames' not in videomae_kwargs and 'temporal_size' in videomae_kwargs:
        videomae_kwargs['num_frames'] = videomae_kwargs.pop('temporal_size')
    else:
        videomae_kwargs.pop('temporal_size', None)
    if 'img_size' in videomae_kwargs:
        videomae_kwargs['image_size'] = videomae_kwargs.pop('img_size')

    custom_kwargs = dict(kwargs)
    custom_kwargs.pop('freeze_backbone', None)
    custom_kwargs.pop('num_frames', None)
    custom_kwargs.pop('use_mean_pooling', None)
    custom_kwargs.pop('image_size', None)

    if use_videomae_backbone:
        if not (use_videomae and VIDEOMAE_AVAILABLE and pretrained):
            raise ValueError("VideoMAE backbone requires transformers and pretrained=True")
        if 'num_frames' not in videomae_kwargs and 'temporal_size' in videomae_kwargs:
            videomae_kwargs['num_frames'] = videomae_kwargs.pop('temporal_size')
        else:
            videomae_kwargs.pop('temporal_size', None)
        if 'img_size' in videomae_kwargs:
            videomae_kwargs['image_size'] = videomae_kwargs.pop('img_size')
        print(f"Using VideoMAE encoder: {videomae_models[variant]}")
        return VideoMAEEncoder(
            model_name=videomae_models[variant],
            pretrained=True,
            **videomae_kwargs,
        )

    # Paper-faithful default: use the custom interleaved encoder, initialized from VideoMAE.
    config = dict(configs[variant])
    config.update(custom_kwargs)

    if use_timm and TIMM_AVAILABLE and pretrained:
        config['use_timm_init'] = True
        config['timm_model'] = timm_models.get(variant, 'vit_base_patch16_224')
        return D4RTEncoder(**config)

    if use_vggt and pretrained:
        config['use_vggt_init'] = True
        config['vggt_checkpoint'] = vggt_checkpoint or 'facebook/VGGT-1B'
        return D4RTEncoder(**config)

    if use_monst3r and pretrained:
        if monst3r_checkpoint is None:
            raise ValueError("monst3r_checkpoint path is required when use_monst3r=True")
        config['use_monst3r_init'] = True
        config['monst3r_checkpoint'] = monst3r_checkpoint
        return D4RTEncoder(**config)

    if use_videomae_v2 and VIDEOMAE_AVAILABLE and pretrained:
        config['use_videomae_init'] = True
        config.setdefault('videomae_model', videomae_v2_models[variant])
        return D4RTEncoder(**config)

    if use_videomae and VIDEOMAE_AVAILABLE and pretrained:
        if variant == 'giant':
            print(
                "No public VideoMAE checkpoint matches the custom ViT-g geometry. "
                "Falling back to random init for the giant encoder."
            )
        else:
            config['use_videomae_init'] = True
            config.setdefault('videomae_model', videomae_models[variant])

    return D4RTEncoder(**config)
