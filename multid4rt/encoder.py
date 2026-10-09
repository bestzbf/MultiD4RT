"""4RC-compatible shared encoder for synchronized multi-camera observations."""

from typing import Optional, Sequence

import torch
from torch import nn

from .embeddings import SinusoidalFrameEmbedding
from .vendor.dinov2.vision_transformer import DinoVisionTransformer


class MultiCameraEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1536,
        depth: int = 40,
        num_heads: int = 24,
        out_layers: Sequence[int] = (19, 27, 33, 39),
        alt_start: int = 13,
        img_size: int = 518,
        patch_size: int = 14,
        ffn_layer: str = "swiglufused",
        attention_chunk_size: Optional[int] = 128,
        checkpoint: bool = True,
    ):
        super().__init__()
        if not 0 <= alt_start < depth:
            raise ValueError("alt_start must index an encoder block")
        if embed_dim % num_heads or (embed_dim // num_heads) % 4:
            raise ValueError("RoPE requires head_dim divisible by 4")
        self.out_layers = tuple(out_layers)
        if not self.out_layers or tuple(sorted(set(self.out_layers))) != self.out_layers:
            raise ValueError("out_layers must be nonempty, unique, and increasing")
        if any(index < alt_start or index >= depth for index in self.out_layers):
            raise ValueError("out_layers must be valid blocks at or after alt_start")
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.frame_embedding = SinusoidalFrameEmbedding(embed_dim)
        self.frame_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.frame_projection.weight)
        # Public inputs are RGB [0,1]; the original DA3/4RC backbone expects
        # ImageNet normalization. Nonpersistent buffers keep legacy keys intact.
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), persistent=False)
        # Keep the original backbone.pretrained.* checkpoint key hierarchy.
        self.pretrained = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=alt_start,
            rope_start=alt_start,
            cat_token=True,
            has_time_token=True,
        )
        self.pretrained.checkpoint_blocks = checkpoint
        for index, block in enumerate(self.pretrained.blocks):
            block.attn.query_chunk_size = attention_chunk_size
            block.attn.checkpoint_chunks = checkpoint
            if index >= alt_start and index % 2 == 1:
                block.attn.camera_relation_weight = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        images: torch.Tensor,
        camera_id: torch.Tensor,
        time_idx: torch.Tensor,
        observation_valid: Optional[torch.Tensor] = None,
    ):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,S,3,H,W]")
        batch, observations = images.shape[:2]
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        if camera_id.shape != (batch, observations) or time_idx.shape != (batch, observations):
            raise ValueError("camera_id and time_idx must match images [B,S]")
        camera_id = camera_id.to(device=images.device)
        time_idx = time_idx.to(device=images.device)
        if observation_valid is None:
            observation_valid = torch.ones((batch, observations), dtype=torch.bool, device=images.device)
        elif observation_valid.shape != (batch, observations) or observation_valid.dtype != torch.bool:
            raise ValueError("observation_valid must be bool [B,S]")
        else:
            observation_valid = observation_valid.to(device=images.device)
        frame_encoding = self.frame_projection(
            self.frame_embedding(time_idx).to(self.frame_projection.weight.dtype)
        )
        taps, _ = self.pretrained.get_intermediate_layers(
            (images-self.image_mean)/self.image_std,
            self.out_layers,
            camera_id=camera_id,
            observation_valid=observation_valid,
            frame_encoding=frame_encoding,
        )
        # Invalid observations can query valid observations internally but are
        # excluded as keys; zero their readouts so padding is never an output.
        return [
            (
                patches * observation_valid[:, :, None, None].to(patches.dtype),
                camera * observation_valid[:, :, None].to(camera.dtype),
                time * observation_valid[:, :, None].to(time.dtype),
            )
            for patches, camera, time in taps
        ]
