"""Dense 4RC-style motion decoding with synchronized multi-camera memory."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from .vendor.heads.dpt_head import DPTHead, custom_interpolate
from .vendor.heads.motiondecoder import MotionDecoder


class MultiViewMotionDecoder(MotionDecoder):
    """Decode one source image's patches at each physical target time.

    Targets have independent prefix/AdaLN conditions. Shared all-observation
    memory projects K/V once per batch item, with target queries concatenated
    only during cross-attention. Self-attention remains per target time.
    The historical per-target memory interface remains supported. Parameter
    names and shapes remain compatible with original 4RC weights.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 14,
        depth: int = 4,
        num_heads: int = 16,
        attention_chunk_size: int | None = 128,
        checkpoint: bool = True,
    ) -> None:
        if embed_dim % num_heads or (embed_dim // num_heads) % 4:
            raise ValueError("2D RoPE requires embed_dim/num_heads divisible by four")
        if depth < 1 or patch_size < 1 or (attention_chunk_size is not None and attention_chunk_size < 1):
            raise ValueError("depth, patch_size and attention_chunk_size must be positive")
        super().__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            use_adaln=True,
            has_self_attention=True,
            has_cross_attention=True,
        )
        self.embed_dim = embed_dim
        self.attention_chunk_size = attention_chunk_size
        self.use_checkpointing = checkpoint
        self.camera_bias = nn.Parameter(torch.zeros(depth, num_heads))

    def forward(
        self,
        source_patches: Tensor,
        memory: Tensor,
        cond: Tensor,
        source_camera: Tensor,
        memory_camera: Tensor,
        memory_valid: Tensor,
        patch_grid: tuple[int, int],
    ) -> Tensor:
        """
        Args:
            source_patches: [B,L,D], a single source image's dense patches.
            memory: [B,S*L,D] shared all-observation memory, or historical
                [B,U,C*L,D] memory selected separately for each target time.
            cond: [B,U,D], fixed time/content condition for all rounds.
            source_camera: [B], source observation's camera identity.
            memory_camera: memory.shape[:-1], camera identity per patch.
            memory_valid: memory.shape[:-1], validity of each memory patch.
            patch_grid: (Hp,Wp), with L=Hp*Wp.

        Returns:
            [B,U,L+1,D], including the updated prefix token at position zero.
        """
        if source_patches.ndim != 3 or memory.ndim not in (3, 4) or cond.ndim != 3:
            raise ValueError("expected source [B,L,D], memory [B,M,D]/[B,U,M,D], cond [B,U,D]")
        batch, patches, dim = source_patches.shape
        shared_memory = memory.ndim == 3
        mem_batch, memory_patches, mem_dim = memory.shape[0], memory.shape[-2], memory.shape[-1]
        targets = cond.shape[1]
        hp, wp = patch_grid
        if hp < 1 or wp < 1 or hp * wp != patches:
            raise ValueError("patch_grid does not match the source patch count")
        if batch != mem_batch or dim != self.embed_dim or mem_dim != dim:
            raise ValueError("source and memory batch/embedding dimensions must agree")
        if targets < 1 or memory_patches < 1 or memory_patches % patches:
            raise ValueError("memory must contain a positive whole number of camera grids")
        if cond.shape != (batch, targets, dim):
            raise ValueError("cond must match [B,U,D]")
        if not shared_memory and memory.shape[1] != targets:
            raise ValueError("per-target memory must match cond's U axis")
        if source_camera.shape != (batch,):
            raise ValueError("source_camera must have shape [B]")
        if memory_camera.shape != memory.shape[:-1]:
            raise ValueError("memory_camera must match memory.shape[:-1]")
        if memory_valid.shape != memory_camera.shape or memory_valid.dtype != torch.bool:
            raise ValueError("memory_valid must be a bool tensor matching memory_camera")
        if any(t.device != source_patches.device for t in (
            memory, cond, source_camera, memory_camera, memory_valid
        )):
            raise ValueError("decoder inputs must be on the same device")

        # The same condition initializes the prefix and modulates every
        # AdaLN round; it is not replaced by the updated prefix token.
        cond = cond.to(dtype=source_patches.dtype)
        source = source_patches[:, None].expand(-1, targets, -1, -1)
        query = torch.cat((cond[:, :, None], source), dim=2)
        query = query.reshape(batch * targets, patches + 1, dim)
        kv = memory if shared_memory else memory.reshape(batch * targets, memory_patches, dim)
        condition = cond.reshape(batch * targets, dim)

        source_ids = source_camera[:, None, None].expand(batch, targets, patches)
        query_camera = torch.cat((
            source_camera.new_full((batch, targets, 1), -1), source_ids
        ), dim=-1).reshape(batch * targets, patches + 1)
        cross_batch = batch if shared_memory else batch * targets
        query_camera = query_camera.reshape(cross_batch, -1)
        key_camera = memory_camera.reshape(cross_batch, memory_patches)
        key_valid = memory_valid.reshape(cross_batch, memory_patches)

        pos_patches = self.position_getter(
            batch * targets, hp, wp, device=source_patches.device
        ).to(source_patches.device) + 1
        pos_prefix = pos_patches.new_zeros(batch * targets, 1, 2)
        pos_q = torch.cat((pos_prefix, pos_patches), dim=1)
        grids = memory_patches // patches
        pos_k = (pos_patches.reshape(batch, targets, patches, 2)[:, 0]
                 if shared_memory else pos_patches).repeat(1, grids, 1)
        pos_cross = (pos_q.reshape(cross_batch, -1, 2), pos_k)

        checkpoint_round = self.use_checkpointing and self.training and torch.is_grad_enabled()
        for round_idx, (cross_block, self_block) in enumerate(zip(
            self.cross_blocks, self.self_blocks
        )):
            # Bind the current module explicitly. A loop-index closure would
            # select the final round when checkpoint recomputes in backward.
            def cross_forward(q, k, weight, block=cross_block):
                result = block(
                    q.reshape(cross_batch, -1, dim),
                    k,
                    pos=pos_cross,
                    query_camera=query_camera,
                    key_camera=key_camera,
                    relation_weight=weight,
                    key_valid=key_valid,
                    query_chunk_size=self.attention_chunk_size,
                    checkpoint_chunks=self.use_checkpointing,
                )
                return result.reshape(batch * targets, patches + 1, dim)

            def self_forward(q, condition_, block=self_block):
                return block(q, cond=condition_, pos=pos_q)

            if checkpoint_round:
                query = checkpoint_fn(
                    cross_forward, query, kv, self.camera_bias[round_idx],
                    use_reentrant=False,
                )
                query = checkpoint_fn(
                    self_forward, query, condition, use_reentrant=False,
                )
            else:
                query = cross_forward(query, kv, self.camera_bias[round_idx])
                query = self_forward(query, condition)

        return query.reshape(batch, targets, patches + 1, dim)


class MotionDPT(DPTHead):
    """Fuse four decoded feature taps into dense motion and confidence.

    Output axes come from decoded target slots, independently of how many
    input observations were supplied to the backbone.
    """

    def __init__(
        self,
        embed_dim: int,
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        patch_size: int = 14,
    ) -> None:
        if len(out_channels) != 4 or features < 2 or patch_size < 1:
            raise ValueError("MotionDPT needs four projection widths and positive dimensions")
        super().__init__(
            dim_in=embed_dim,
            patch_size=patch_size,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1",
            features=features,
            out_channels=list(out_channels),
            intermediate_layer_idx=[0, 1, 2, 3],
            pos_embed=True,
            feature_only=False,
        )

    def forward(
        self,
        decoded_taps: Sequence[Tensor],
        output_hw: tuple[int, int],
    ) -> dict[str, Tensor]:
        """Return displacement g, confidence and pre-output-conv2 features.

        decoded_taps consists of four [B,U,L+1,D] tensors. Source-time
        anchoring (g(target)-g(source)) is performed by the enclosing model.
        """
        if len(decoded_taps) != 4 or decoded_taps[0].ndim != 4:
            raise ValueError("decoded_taps must contain four [B,U,L+1,D] tensors")
        height, width = output_hw
        if height < 1 or width < 1:
            raise ValueError("output_hw must be positive")
        hp, wp = height // self.patch_size, width // self.patch_size
        if hp < 1 or wp < 1:
            raise ValueError("output_hw must be at least one patch in each dimension")
        batch, targets, tokens, dim = decoded_taps[0].shape
        if tokens != hp * wp + 1 or dim != self.norm.normalized_shape[0]:
            raise ValueError("decoded tokens do not match output_hw's patch grid/embedding width")
        if any(tap.shape != decoded_taps[0].shape for tap in decoded_taps):
            raise ValueError("all four decoded taps must have the same shape")

        pyramid = []
        for tap, projection, resize in zip(
            decoded_taps, self.projects, self.resize_layers
        ):
            patches = self.norm(tap[:, :, 1:].reshape(batch * targets, hp * wp, dim))
            grid = patches.transpose(1, 2).reshape(batch * targets, dim, hp, wp)
            grid = projection(grid)
            if self.pos_embed:
                grid = self._apply_pos_embed(grid, width, height)
            pyramid.append(resize(grid))

        # scratch_forward includes output_conv1 (features -> features/2)
        # before the final interpolation, preserving 4RC's DPT ordering.
        features = self.scratch_forward(pyramid)
        features = custom_interpolate(
            features, (height, width), mode="bilinear", align_corners=True
        )
        if self.pos_embed:
            features = self._apply_pos_embed(features, width, height)
        raw = self.scratch.output_conv2(features).float()
        xyz = raw[:, :3].permute(0, 2, 3, 1)

        # Compute exponentials in float32 and bound only overflow-scale
        # logits. The where form also keeps the derivative at zero equal to
        # one, unlike sign(x)*expm1(abs(x)).
        displacement = torch.where(
            xyz >= 0,
            torch.expm1(xyz.clamp(max=80.0)),
            -torch.expm1((-xyz).clamp(max=80.0)),
        )
        confidence = 1.0 + torch.exp(raw[:, 3].clamp(max=80.0))
        return {
            "displacement": displacement.reshape(batch, targets, height, width, 3),
            "confidence": confidence.reshape(batch, targets, height, width),
            "features": features.reshape(batch, targets, features.shape[1], height, width),
        }
