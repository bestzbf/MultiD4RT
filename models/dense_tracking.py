"""Algorithm 1: Efficient Dense Tracking of All Pixels.

This module implements the efficient dense tracking algorithm from the D4RT paper,
which tracks all pixels in a video using an occupancy grid to avoid redundant queries.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DenseTrackingConfig:
    """Configuration for dense tracking algorithm."""
    batch_size: int = 256  # Number of tracks to process in parallel
    visibility_threshold: float = 0.5  # Threshold for marking pixels as visible
    min_track_length: int = 2  # Minimum frames a track must be visible
    spatial_stride: int = 1  # Stride for spatial sampling (1 = all pixels)


class DenseTracker:
    """Efficient Dense Tracking of All Pixels (Algorithm 1).

    This algorithm efficiently tracks all pixels in a video by:
    1. Maintaining an occupancy grid to track which pixels have been visited
    2. Only initiating new tracks from unvisited pixels
    3. Marking all visible pixels along each track as visited

    This reduces complexity from O(T²HW) to approximately O(THW/visibility_ratio),
    yielding 5-15× speedup depending on scene motion complexity.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[DenseTrackingConfig] = None
    ):
        """
        Args:
            model: D4RT model with encode() and decode() methods
            config: Configuration for tracking algorithm
        """
        self.model = model
        self.config = config or DenseTrackingConfig()

    @torch.no_grad()
    def track_all_pixels(
        self,
        video: torch.Tensor,
        return_3d: bool = True,
        return_2d: bool = True,
        verbose: bool = False
    ) -> dict[str, torch.Tensor]:
        """Track all pixels in a video efficiently.

        Implements Algorithm 1 from the D4RT paper.

        NOTE: This function returns tracks in each target camera's frame, NOT a unified
        reference frame. For world-coordinate tracking, use track_all_pixels_to_world().

        Args:
            video: (B, T, H, W, C) video tensor (only B=1 supported currently)
            return_3d: Whether to return 3D tracks
            return_2d: Whether to return 2D tracks
            verbose: Print progress information

        Returns:
            Dictionary with:
                - tracks_3d: (N_tracks, T, 3) 3D positions in each target camera frame
                - tracks_2d: (N_tracks, T, 2) 2D positions for each track
                - visibility: (N_tracks, T) visibility for each track
                - source_pixels: (N_tracks, 3) source (t, y, x) for each track
                - colors: (N_tracks, 3) RGB color at source pixel
        """
        assert video.shape[0] == 1, "Batch size must be 1 for dense tracking"

        B, T, H, W, C = video.shape
        device = video.device
        stride = self.config.spatial_stride

        # Effective dimensions after stride (ceil to include edge pixels)
        H_eff = (H + stride - 1) // stride
        W_eff = (W + stride - 1) // stride

        # Step 1: Compute Global Scene Representation
        if verbose:
            print("Encoding video...")
        encoder_features = self.model.encode(video)

        # Prepare frames for patch extraction
        if video.dim() == 5 and video.shape[-1] == 3:
            frames = video.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        else:
            frames = video

        # Step 2: Initialize Occupancy Grid G ← {false}^(T×H×W)
        occupancy_grid = torch.zeros(T, H_eff, W_eff, dtype=torch.bool, device=device)

        # Step 3: Initialize Set of Dense Tracks T ← ∅
        all_tracks_3d = []
        all_tracks_2d = []
        all_visibility = []
        all_source_pixels = []
        all_colors = []

        # Create coordinate grids
        y_coords = torch.arange(0, H, stride, device=device)
        x_coords = torch.arange(0, W, stride, device=device)

        iteration = 0
        total_pixels = T * H_eff * W_eff

        # Step 4: while any(G = false) do
        while not occupancy_grid.all():
            iteration += 1

            # Step 5: Sample a batch B of unvisited source points from G
            unvisited_mask = ~occupancy_grid  # (T, H_eff, W_eff)
            unvisited_indices = unvisited_mask.nonzero()  # (N_unvisited, 3)

            if len(unvisited_indices) == 0:
                break

            # Sample batch of unvisited points
            n_sample = min(self.config.batch_size, len(unvisited_indices))
            perm = torch.randperm(len(unvisited_indices), device=device)[:n_sample]
            batch_indices = unvisited_indices[perm]  # (n_sample, 3) -> (t, y_idx, x_idx)

            # Convert indices to coordinates
            t_src = batch_indices[:, 0]  # (n_sample,)
            y_idx = batch_indices[:, 1]
            x_idx = batch_indices[:, 2]

            # Convert to normalized coordinates (corner-normalized, consistent with
            # predict_depth / predict_point_cloud which use x/(W-1) / linspace(0,1))
            u = (x_idx.float() * stride) / max(W - 1, 1)  # (n_sample,)
            v = (y_idx.float() * stride) / max(H - 1, 1)  # (n_sample,)
            coords = torch.stack([u, v], dim=-1).unsqueeze(0)  # (1, n_sample, 2)

            # Step 6-8: For each source point, get full track queries and run decoder
            tracks_3d_batch = []
            tracks_2d_batch = []
            visibility_batch = []

            # Process all timesteps for this batch of source points
            for t_tgt in range(T):
                # Q ← {u, v, t_src, t_tgt=t_cam=k}
                t_src_expanded = t_src.unsqueeze(0)  # (1, n_sample)
                t_tgt_tensor = torch.full((1, n_sample), t_tgt, device=device, dtype=torch.long)
                t_cam_tensor = torch.full((1, n_sample), t_tgt, device=device, dtype=torch.long)

                # P ← D(q, F) - Run the decoder
                outputs = self.model.decode(
                    encoder_features, frames, coords,
                    t_src_expanded, t_tgt_tensor, t_cam_tensor
                )

                tracks_3d_batch.append(outputs['pos_3d'].squeeze(0))  # (n_sample, 3)
                tracks_2d_batch.append(outputs['pos_2d'].squeeze(0))  # (n_sample, 2)
                vis = torch.sigmoid(outputs['visibility'].squeeze(0).squeeze(-1))  # (n_sample,)
                visibility_batch.append(vis)

            # Stack into tracks: (n_sample, T, dim)
            tracks_3d = torch.stack(tracks_3d_batch, dim=1)  # (n_sample, T, 3)
            tracks_2d = torch.stack(tracks_2d_batch, dim=1)  # (n_sample, T, 2)
            visibility = torch.stack(visibility_batch, dim=1)  # (n_sample, T)

            # Step 9: G ← Visible(P) - Set visible track pixels as visited
            visible_mask = visibility > self.config.visibility_threshold  # (n_sample, T)

            # Vectorized occupancy update (avoid Python loops with .item())
            # Get all visible positions at once
            visible_indices = visible_mask.nonzero()  # (M, 2) where M = num visible
            if len(visible_indices) > 0:
                sample_idx = visible_indices[:, 0]  # (M,)
                time_idx = visible_indices[:, 1]    # (M,)

                # Get 2D positions for all visible points (corner-normalized)
                px = tracks_2d[sample_idx, time_idx, 0] * max(W - 1, 1)  # (M,)
                py = tracks_2d[sample_idx, time_idx, 1] * max(H - 1, 1)  # (M,)

                # Filter valid positions (within bounds)
                valid = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
                if valid.any():
                    px_valid = px[valid]
                    py_valid = py[valid]
                    t_valid = time_idx[valid]

                    # Convert to grid indices
                    x_grid = (px_valid.round().long() // stride).clamp(0, W_eff - 1)
                    y_grid = (py_valid.round().long() // stride).clamp(0, H_eff - 1)

                    # Mark as visited (vectorized)
                    occupancy_grid[t_valid, y_grid, x_grid] = True

            # Also mark source pixels as visited (vectorized)
            occupancy_grid[t_src, y_idx, x_idx] = True

            # Step 10: T ← T ∪ P - Add new tracks to the output
            all_tracks_3d.append(tracks_3d)
            all_tracks_2d.append(tracks_2d)
            all_visibility.append(visibility)

            # Store source pixel locations
            source_pixels = torch.stack([
                t_src.float(),
                y_idx.float() * stride,
                x_idx.float() * stride
            ], dim=-1)  # (n_sample, 3)
            all_source_pixels.append(source_pixels)

            # Get colors from source frames (vectorized)
            y_px = (y_idx * stride).long()  # (n_sample,)
            x_px = (x_idx * stride).long()  # (n_sample,)
            all_colors.append(video[0, t_src, y_px, x_px, :])  # (n_sample, 3)

            if verbose:
                # .item() here is verbose-only, not in the hot path
                visited = int(occupancy_grid.sum())
                progress = visited / total_pixels * 100
                print(f"Iteration {iteration}: {visited}/{total_pixels} pixels visited ({progress:.1f}%)")

        # Concatenate all tracks
        result = {
            'tracks_3d': torch.cat(all_tracks_3d, dim=0) if return_3d else None,
            'tracks_2d': torch.cat(all_tracks_2d, dim=0) if return_2d else None,
            'visibility': torch.cat(all_visibility, dim=0),
            'source_pixels': torch.cat(all_source_pixels, dim=0),
            'colors': torch.cat(all_colors, dim=0),
            'num_iterations': iteration
        }

        if verbose:
            print(f"Dense tracking complete: {len(result['visibility'])} tracks in {iteration} iterations")

        return result

    @torch.no_grad()
    def track_all_pixels_to_world(
        self,
        video: torch.Tensor,
        reference_frame: int = 0,
        verbose: bool = False
    ) -> dict[str, torch.Tensor]:
        """Track all pixels and return in world coordinate frame.

        Similar to track_all_pixels but all 3D positions are expressed
        in a single world coordinate frame (reference_frame).

        Args:
            video: (B, T, H, W, C) video tensor
            reference_frame: Frame to use as world coordinate reference
            verbose: Print progress

        Returns:
            Dictionary with tracks in world coordinates
        """
        assert video.shape[0] == 1, "Batch size must be 1"

        B, T, H, W, C = video.shape
        device = video.device
        stride = self.config.spatial_stride

        H_eff = (H + stride - 1) // stride
        W_eff = (W + stride - 1) // stride

        if verbose:
            print("Encoding video...")
        encoder_features = self.model.encode(video)

        if video.dim() == 5 and video.shape[-1] == 3:
            frames = video.permute(0, 1, 4, 2, 3)
        else:
            frames = video

        # Initialize occupancy grid and tracks
        occupancy_grid = torch.zeros(T, H_eff, W_eff, dtype=torch.bool, device=device)

        all_tracks_3d = []
        all_visibility = []
        all_source_pixels = []
        all_colors = []

        iteration = 0
        total_pixels = T * H_eff * W_eff

        while not occupancy_grid.all():
            iteration += 1

            unvisited_mask = ~occupancy_grid
            unvisited_indices = unvisited_mask.nonzero()

            if len(unvisited_indices) == 0:
                break

            n_sample = min(self.config.batch_size, len(unvisited_indices))
            perm = torch.randperm(len(unvisited_indices), device=device)[:n_sample]
            batch_indices = unvisited_indices[perm]

            t_src = batch_indices[:, 0]
            y_idx = batch_indices[:, 1]
            x_idx = batch_indices[:, 2]

            u = (x_idx.float() * stride) / max(W - 1, 1)
            v = (y_idx.float() * stride) / max(H - 1, 1)
            coords = torch.stack([u, v], dim=-1).unsqueeze(0)

            tracks_3d_batch = []
            visibility_batch = []

            for t_tgt in range(T):
                t_src_expanded = t_src.unsqueeze(0)
                t_tgt_tensor = torch.full((1, n_sample), t_tgt, device=device, dtype=torch.long)
                # Use reference_frame as t_cam for world coordinates
                t_cam_tensor = torch.full((1, n_sample), reference_frame, device=device, dtype=torch.long)

                outputs = self.model.decode(
                    encoder_features, frames, coords,
                    t_src_expanded, t_tgt_tensor, t_cam_tensor
                )

                tracks_3d_batch.append(outputs['pos_3d'].squeeze(0))
                vis = torch.sigmoid(outputs['visibility'].squeeze(0).squeeze(-1))
                visibility_batch.append(vis)

            tracks_3d = torch.stack(tracks_3d_batch, dim=1)
            visibility = torch.stack(visibility_batch, dim=1)

            # Update occupancy grid based on visibility
            visible_mask = visibility > self.config.visibility_threshold

            # Vectorized occupancy update: batch decode all timesteps at once
            # Instead of T separate decode calls, do one batched call
            t_all = torch.arange(T, device=device)  # (T,)
            t_src_expanded = t_src.unsqueeze(0).expand(T, -1).reshape(-1)  # (T*n_sample,)
            coords_expanded = coords.expand(T, -1, -1).reshape(1, -1, 2)  # (1, T*n_sample, 2)
            t_tgt_all = t_all.unsqueeze(1).expand(-1, n_sample).reshape(-1)  # (T*n_sample,)
            t_cam_all = t_tgt_all  # Same as t_tgt

            # Single batched decode for all (t, sample) pairs
            outputs_cam = self.model.decode(
                encoder_features, frames, coords_expanded,
                t_src_expanded.unsqueeze(0), t_tgt_all.unsqueeze(0), t_cam_all.unsqueeze(0)
            )
            pos_2d_all = outputs_cam['pos_2d'].squeeze(0).reshape(T, n_sample, 2)  # (T, n_sample, 2)

            # Vectorized occupancy marking
            visible_indices = visible_mask.nonzero()  # (M, 2) where M = num visible
            if len(visible_indices) > 0:
                time_idx = visible_indices[:, 1]    # (M,)
                sample_idx = visible_indices[:, 0]  # (M,)

                # Get 2D positions for all visible points
                px = pos_2d_all[time_idx, sample_idx, 0] * max(W - 1, 1)  # (M,)
                py = pos_2d_all[time_idx, sample_idx, 1] * max(H - 1, 1)  # (M,)

                # Filter valid positions
                valid = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
                if valid.any():
                    px_valid = px[valid]
                    py_valid = py[valid]
                    t_valid = time_idx[valid]

                    # Convert to grid indices
                    x_grid = (px_valid.round().long() // stride).clamp(0, W_eff - 1)
                    y_grid = (py_valid.round().long() // stride).clamp(0, H_eff - 1)

                    # Mark as visited
                    occupancy_grid[t_valid, y_grid, x_grid] = True

            # Mark source pixels as visited (vectorized)
            occupancy_grid[t_src, y_idx, x_idx] = True

            all_tracks_3d.append(tracks_3d)
            all_visibility.append(visibility)

            source_pixels = torch.stack([
                t_src.float(),
                y_idx.float() * stride,
                x_idx.float() * stride
            ], dim=-1)
            all_source_pixels.append(source_pixels)

            colors = []
            for i in range(n_sample):
                t = t_src[i].item()
                y = int(y_idx[i].item() * stride)
                x = int(x_idx[i].item() * stride)
                colors.append(video[0, t, y, x, :])
            all_colors.append(torch.stack(colors, dim=0))

            if verbose:
                visited = occupancy_grid.sum().item()
                progress = visited / total_pixels * 100
                print(f"Iteration {iteration}: {progress:.1f}% complete")

        result = {
            'tracks_3d': torch.cat(all_tracks_3d, dim=0),
            'visibility': torch.cat(all_visibility, dim=0),
            'source_pixels': torch.cat(all_source_pixels, dim=0),
            'colors': torch.cat(all_colors, dim=0),
            'reference_frame': reference_frame,
            'num_iterations': iteration
        }

        if verbose:
            print(f"World coordinate tracking complete: {len(result['visibility'])} tracks")

        return result


def build_point_cloud_from_tracks(
    tracks: dict[str, torch.Tensor],
    frame_idx: Optional[int] = None
) -> dict[str, torch.Tensor]:
    """Build a point cloud from dense tracking results.

    Args:
        tracks: Output from DenseTracker.track_all_pixels
        frame_idx: Optional frame to extract point cloud for.
                   If None, returns all visible points across all frames.

    Returns:
        Dictionary with:
            - points: (N, 3) 3D points
            - colors: (N, 3) RGB colors
            - visibility: (N,) visibility scores
    """
    tracks_3d = tracks['tracks_3d']  # (N_tracks, T, 3)
    visibility = tracks['visibility']  # (N_tracks, T)
    colors = tracks['colors']  # (N_tracks, 3)

    if frame_idx is not None:
        # Extract points for specific frame
        points = tracks_3d[:, frame_idx, :]  # (N_tracks, 3)
        vis = visibility[:, frame_idx]  # (N_tracks,)

        # Filter by visibility
        visible_mask = vis > 0.5
        points = points[visible_mask]
        colors_out = colors[visible_mask]
        vis_out = vis[visible_mask]
    else:
        # Flatten all visible points
        N_tracks, T, _ = tracks_3d.shape

        points_list = []
        colors_list = []
        vis_list = []

        for t in range(T):
            vis_t = visibility[:, t]
            visible_mask = vis_t > 0.5

            points_list.append(tracks_3d[visible_mask, t, :])
            colors_list.append(colors[visible_mask])
            vis_list.append(vis_t[visible_mask])

        points = torch.cat(points_list, dim=0)
        colors_out = torch.cat(colors_list, dim=0)
        vis_out = torch.cat(vis_list, dim=0)

    return {
        'points': points,
        'colors': colors_out,
        'visibility': vis_out
    }
