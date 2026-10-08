"""D4RT: Dynamic 4D Reconstruction and Tracking Model."""

import torch
import torch.nn as nn
from typing import Optional

from .encoder import D4RTEncoder, create_encoder
from .decoder import D4RTDecoder


class D4RT(nn.Module):
    """D4RT: Unified model for Dynamic 4D Reconstruction and Tracking.

    A feedforward transformer model that encodes video into a Global Scene
    Representation, then decodes arbitrary spatiotemporal queries to predict
    3D positions.

    The query interface (u, v, t_src, t_tgt, t_cam) supports multiple tasks:
    - Point tracking: Fixed (u,v,t_src), varying t_tgt=t_cam
    - Point cloud: Varying (u,v,t_src), fixed t_cam
    - Depth map: Varying (u,v), t_src=t_tgt=t_cam
    - Camera extrinsics: Grid of (u,v), fixed t_src, varying t_cam
    - Camera intrinsics: Grid of (u,v), t_src=t_tgt=t_cam
    """

    def __init__(
        self,
        encoder_variant: str = 'base',
        img_size: int = 256,
        temporal_size: int = 48,
        patch_size: tuple[int, int, int] = (2, 16, 16),
        decoder_depth: int = 8,
        decoder_num_heads: int = 12,
        max_timesteps: int = 128,
        query_patch_size: int = 9,
        videomae_model: Optional[str] = None,
        patch_provider: str = 'auto',
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        disable_query_patch_embedding: bool = False,
        disable_query_timestep_embedding: bool = False,
        disable_decoder_cross_attention: bool = False,
        debug_3d_head_mode: str = 'linear',
        encoder_pretrained: bool = True,
        use_videomae: bool = True,
        use_videomae_v2: bool = False,
        use_vggt: bool = False,
        use_monst3r: bool = False,
        use_timm: bool = False,
        use_videomae_backbone: bool = False,
        use_videomae_init: bool = False,
        use_timm_init: bool = False,
        use_aspect_ratio_token: bool = True,
        vggt_checkpoint: Optional[str] = None,
        monst3r_checkpoint: Optional[str] = None,
    ):
        super().__init__()

        encoder_kwargs = dict(
            variant=encoder_variant,
            img_size=img_size,
            temporal_size=temporal_size,
            patch_size=patch_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            pretrained=encoder_pretrained,
            use_videomae=use_videomae,
            use_videomae_v2=use_videomae_v2,
            use_vggt=use_vggt,
            use_monst3r=use_monst3r,
            use_timm=use_timm,
            use_videomae_backbone=use_videomae_backbone,
            use_videomae_init=use_videomae_init,
            use_timm_init=use_timm_init,
            use_aspect_ratio_token=use_aspect_ratio_token,
        )
        if videomae_model is not None:
            encoder_kwargs["videomae_model"] = videomae_model
        if vggt_checkpoint is not None:
            encoder_kwargs["vggt_checkpoint"] = vggt_checkpoint
        if monst3r_checkpoint is not None:
            encoder_kwargs["monst3r_checkpoint"] = monst3r_checkpoint

        # Create encoder
        self.encoder = create_encoder(**encoder_kwargs)

        embed_dim = self.encoder.embed_dim

        # Create decoder
        self.decoder = D4RTDecoder(
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            max_timesteps=max_timesteps,
            patch_size=query_patch_size,
            patch_provider=patch_provider,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            disable_query_patch_embedding=disable_query_patch_embedding,
            disable_query_timestep_embedding=disable_query_timestep_embedding,
            disable_cross_attention=disable_decoder_cross_attention,
            debug_3d_head_mode=debug_3d_head_mode,
        )

        self.img_size = img_size
        self.temporal_size = temporal_size

    def encode(
        self,
        video: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode video into Global Scene Representation.

        Args:
            video: (B, T, H, W, C) or (B, C, T, H, W) video tensor
            aspect_ratio: (B, 1) original width/height ratio, or legacy (B, 2)

        Returns:
            features: (B, N, embed_dim) Global Scene Representation F
        """
        return self.encoder(video, aspect_ratio)

    def decode(
        self,
        encoder_features: torch.Tensor,
        frames: torch.Tensor | list[torch.Tensor],
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        transform_metadata: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Decode queries using Global Scene Representation.

        Args:
            encoder_features: (B, N, embed_dim) encoded features
            frames: (B, T, H, W, C) or (B, T, C, H, W) video frames
            coords: (B, N_q, 2) normalized query coordinates
            t_src: (B, N_q) source timesteps
            t_tgt: (B, N_q) target timesteps
            t_cam: (B, N_q) camera reference timesteps
            local_patches: Optional pre-extracted local RGB patches
            transform_metadata: Optional geometry metadata for canonical query coordinates

        Returns:
            Dictionary with predictions
        """
        return self.decoder(
            encoder_features,
            frames,
            coords,
            t_src,
            t_tgt,
            t_cam,
            local_patches=local_patches,
            transform_metadata=transform_metadata,
        )

    @staticmethod
    def _prepare_query_frames(
        video: torch.Tensor,
        query_frames: Optional[torch.Tensor | list[Optional[torch.Tensor]]],
    ) -> torch.Tensor | list[torch.Tensor]:
        """Normalize patch-extraction frames to (B, T, C, H, W)."""
        if query_frames is not None and isinstance(query_frames, list):
            # query_frames is a list of Optional[Tensor] with variable crop sizes.
            # Keep it as a list: padding all entries to one outlier high-res crop
            # can allocate tens of GiB before we only need tiny local patches.
            if video.dim() == 5 and video.shape[1] == 3:
                resized_frames = video.permute(0, 2, 1, 3, 4)  # (B,C,T,H,W) -> (B,T,C,H,W)
            elif video.dim() == 5 and video.shape[2] == 3:
                resized_frames = video  # already (B,T,C,H,W)
            elif video.dim() == 5 and video.shape[-1] == 3:
                resized_frames = video.permute(0, 1, 4, 2, 3)  # (B,T,H,W,C) -> (B,T,C,H,W)
            else:
                raise ValueError(f"Unsupported video shape for query frame prep: {tuple(video.shape)}")

            resolved: list[torch.Tensor] = []
            for i, qf in enumerate(query_frames):
                if qf is not None:
                    if qf.dim() == 4 and qf.shape[1] == 3:
                        resolved.append(qf)  # [T, C, H, W]
                    elif qf.dim() == 4 and qf.shape[-1] == 3:
                        resolved.append(qf.permute(0, 3, 1, 2))  # [T, H, W, C] -> [T, C, H, W]
                    else:
                        raise ValueError(
                            "Unsupported query frame shape for list entry "
                            f"{i}: {tuple(qf.shape)}"
                        )
                else:
                    resolved.append(resized_frames[i])  # [T, C, H, W]
            return resolved

        if query_frames is not None:
            return query_frames
        if video.dim() == 5 and video.shape[1] == 3:
            return video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
        if video.dim() == 5 and video.shape[2] == 3:
            return video  # already (B, T, C, H, W)
        if video.dim() == 5 and video.shape[-1] == 3:
            return video.permute(0, 1, 4, 2, 3)  # (B, T, H, W, C) -> (B, T, C, H, W)
        raise ValueError(f"Unsupported video shape for query frame prep: {tuple(video.shape)}")

    def forward(
        self,
        video: torch.Tensor,
        coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None,
        query_frames: Optional[torch.Tensor] = None,
        local_patches: Optional[torch.Tensor] = None,
        transform_metadata: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode video and decode queries.

        Args:
            video: (B, T, H, W, C) or (B, C, T, H, W) video tensor
            coords: (B, N_q, 2) normalized query coordinates in [0, 1]
            t_src: (B, N_q) source timestep indices
            t_tgt: (B, N_q) target timestep indices
            t_cam: (B, N_q) camera reference timestep indices
            aspect_ratio: (B, 1) optional width/height ratio, or legacy (B, 2)
            query_frames: Optional (B, T, C, H, W) or (B, T, H, W, C) frames used only for local patch extraction
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
        # Encode video
        encoder_features = self.encode(video, aspect_ratio)

        frames = self._prepare_query_frames(video, query_frames)

        # Decode queries
        return self.decode(
            encoder_features,
            frames,
            coords,
            t_src,
            t_tgt,
            t_cam,
            local_patches=local_patches,
            transform_metadata=transform_metadata,
        )

    @torch.no_grad()
    def predict_depth(
        self,
        video: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None,
        output_resolution: Optional[tuple[int, int]] = None
    ) -> torch.Tensor:
        """Predict depth maps for all frames.

        Args:
            video: (B, T, H, W, C) or (B, C, T, H, W) video tensor
            aspect_ratio: (B, 1) original width/height ratio, or legacy (B, 2)
            output_resolution: (H, W) output resolution, defaults to video resolution

        Returns:
            depth: (B, T, H, W) depth maps
        """
        # Normalize to (B, T, H, W, C) format
        if video.dim() == 5 and video.shape[1] == 3:
            # (B, C, T, H, W) -> (B, T, H, W, C)
            video = video.permute(0, 2, 3, 4, 1)

        B, T, H, W, C = video.shape

        if output_resolution is None:
            out_H, out_W = H, W
        else:
            out_H, out_W = output_resolution

        # Encode video
        encoder_features = self.encode(video, aspect_ratio)

        # Create query grid for all pixels
        device = video.device
        u = torch.linspace(0, 1, out_W, device=device)
        v = torch.linspace(0, 1, out_H, device=device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
        coords = torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)  # (H*W, 2)

        depths = []

        for t in range(T):
            # For depth: t_src = t_tgt = t_cam = t
            t_indices = torch.full((out_H * out_W,), t, device=device, dtype=torch.long)

            # Batch queries
            coords_batch = coords.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
            t_src = t_indices.unsqueeze(0).expand(B, -1)
            t_tgt = t_indices.unsqueeze(0).expand(B, -1)
            t_cam = t_indices.unsqueeze(0).expand(B, -1)

            # Prepare frames
            frames = video.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

            # Decode
            outputs = self.decode(encoder_features, frames, coords_batch, t_src, t_tgt, t_cam)

            # Extract depth (Z component of 3D position)
            depth = outputs['pos_3d'][:, :, 2]  # (B, H*W)
            depth = depth.reshape(B, out_H, out_W)
            depths.append(depth)

        return torch.stack(depths, dim=1)  # (B, T, H, W)

    @torch.no_grad()
    def predict_point_tracks(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        query_frames: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None,
        patch_frames: Optional[torch.Tensor | list[Optional[torch.Tensor]]] = None,
        transform_metadata: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Predict 3D point tracks through the video.

        Args:
            video: (B, T, H, W, C) or (B, C, T, H, W) video tensor
            query_points: (B, N, 2) query point coordinates in [0, 1]
            query_frames: (B, N) source frame indices for each query point
            aspect_ratio: (B, 1) original width/height ratio, or legacy (B, 2)
            patch_frames: Optional frames used only for query patch extraction.
            transform_metadata: Optional geometry metadata for sampled_highres.

        Returns:
            Dictionary with:
                - tracks_3d: (B, N, T, 3) 3D positions at each timestep
                - tracks_2d: (B, N, T, 2) 2D positions at each timestep
                - visibility: (B, N, T) visibility at each timestep
        """
        # Normalize to (B, T, H, W, C) format
        if video.dim() == 5 and video.shape[1] == 3:
            # (B, C, T, H, W) -> (B, T, H, W, C)
            video = video.permute(0, 2, 3, 4, 1)

        B, T, H, W, C = video.shape
        N = query_points.shape[1]
        device = video.device

        # Encode video
        encoder_features = self.encode(video, aspect_ratio)

        # Prepare frames used for patch extraction.
        frames = self._prepare_query_frames(video, patch_frames)

        tracks_3d = []
        tracks_2d = []
        visibility = []

        for t in range(T):
            # For point tracking: fixed (u,v,t_src), t_tgt = t_cam = t
            t_tgt = torch.full((B, N), t, device=device, dtype=torch.long)
            t_cam = torch.full((B, N), t, device=device, dtype=torch.long)

            # Decode
            outputs = self.decode(
                encoder_features, frames,
                query_points, query_frames, t_tgt, t_cam,
                transform_metadata=transform_metadata,
            )

            tracks_3d.append(outputs['pos_3d'])
            tracks_2d.append(outputs['pos_2d'])
            visibility.append(torch.sigmoid(outputs['visibility']).squeeze(-1))

        return {
            'tracks_3d': torch.stack(tracks_3d, dim=2),  # (B, N, T, 3)
            'tracks_2d': torch.stack(tracks_2d, dim=2),  # (B, N, T, 2)
            'visibility': torch.stack(visibility, dim=2)  # (B, N, T)
        }

    @torch.no_grad()
    def predict_point_cloud(
        self,
        video: torch.Tensor,
        reference_frame: int = 0,
        aspect_ratio: Optional[torch.Tensor] = None,
        stride: int = 1
    ) -> dict[str, torch.Tensor]:
        """Predict point cloud in a unified reference frame.

        Args:
            video: (B, T, H, W, C) video tensor
            reference_frame: Frame index for camera reference
            aspect_ratio: (B, 1) original width/height ratio, or legacy (B, 2)
            stride: Spatial stride for point sampling

        Returns:
            Dictionary with:
                - points: (B, N, 3) 3D points
                - colors: (B, N, 3) RGB colors
                - normals: (B, N, 3) surface normals
        """
        B, T, H, W, C = video.shape
        device = video.device

        # Encode video
        encoder_features = self.encode(video, aspect_ratio)

        # Prepare frames
        frames = video.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

        all_points = []
        all_colors = []
        all_normals = []

        # Create coordinate grid
        x_pixels = torch.arange(0, W, stride, device=device, dtype=video.dtype)
        y_pixels = torch.arange(0, H, stride, device=device, dtype=video.dtype)
        out_H = y_pixels.numel()
        out_W = x_pixels.numel()
        u = x_pixels / max(W - 1, 1)
        v = y_pixels / max(H - 1, 1)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
        coords = torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)
        coords_batch = coords.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)

        for t in range(T):
            # For point cloud: t_src = t_tgt = t, t_cam = reference_frame
            n_points = out_H * out_W
            t_src = torch.full((B, n_points), t, device=device, dtype=torch.long)
            t_tgt = torch.full((B, n_points), t, device=device, dtype=torch.long)
            t_cam = torch.full((B, n_points), reference_frame, device=device, dtype=torch.long)

            # Decode
            outputs = self.decode(encoder_features, frames, coords_batch, t_src, t_tgt, t_cam)

            all_points.append(outputs['pos_3d'])
            all_normals.append(outputs['normal'])

            # Get colors from video frames
            frame_colors = video[:, t][:, y_pixels.long()][:, :, x_pixels.long(), :].reshape(B, -1, 3)
            all_colors.append(frame_colors)

        return {
            'points': torch.cat(all_points, dim=1),  # (B, T*H*W, 3)
            'colors': torch.cat(all_colors, dim=1),  # (B, T*H*W, 3)
            'normals': torch.cat(all_normals, dim=1)  # (B, T*H*W, 3)
        }


def create_d4rt(
    variant: str = 'base',
    pretrained: bool = False,
    encoder_pretrained: bool = True,
    **kwargs
) -> D4RT:
    """Create D4RT model with predefined configurations.

    Args:
        variant: One of 'base', 'large', 'huge', 'giant'
        pretrained: Whether to load full D4RT pretrained weights (not implemented)
        encoder_pretrained: Whether encoder uses pretrained weights (VideoMAE/timm)
        **kwargs: Additional arguments passed to D4RT

    Returns:
        Configured D4RT model
    """
    decoder_configs = {
        'base': dict(decoder_depth=6, decoder_num_heads=12),
        'large': dict(decoder_depth=6, decoder_num_heads=16),
        'huge': dict(decoder_depth=8, decoder_num_heads=16),
        'giant': dict(decoder_depth=8, decoder_num_heads=16),
    }

    config = decoder_configs.get(variant, decoder_configs['base'])
    config['encoder_variant'] = variant

    # Pass encoder_pretrained to D4RT.__init__ (which forwards to encoder creation)
    if 'encoder_pretrained' not in kwargs:
        kwargs['encoder_pretrained'] = encoder_pretrained

    config.update(kwargs)

    model = D4RT(**config)

    if pretrained:
        # TODO: Load pretrained weights
        raise NotImplementedError("Pretrained weights not available yet")

    return model
