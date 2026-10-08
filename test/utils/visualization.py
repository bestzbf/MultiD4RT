"""Visualization utilities for D4RT."""

import torch
import numpy as np
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _tensor_to_numpy(array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensors to numpy safely, normalizing unsupported low-precision dtypes."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu()
        if array.dtype in {torch.bfloat16, torch.float16}:
            array = array.float()
        return array.numpy()
    return array


def visualize_depth(
    depth: torch.Tensor,
    colormap: str = 'magma',
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    return_numpy: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """Visualize depth map with colormap.

    Args:
        depth: (H, W) depth map
        colormap: Matplotlib colormap name
        min_depth: Minimum depth for normalization
        max_depth: Maximum depth for normalization
        return_numpy: Whether to return numpy array

    Returns:
        Colored depth visualization (H, W, 3) in [0, 255] uint8
    """
    depth = _tensor_to_numpy(depth)

    # Handle invalid values
    valid = np.isfinite(depth) & (depth > 0)

    if min_depth is None:
        min_depth = depth[valid].min() if valid.any() else 0

    if max_depth is None:
        max_depth = depth[valid].max() if valid.any() else 1

    # Normalize
    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(depth_norm)[..., :3]  # (H, W, 3)

    # Mark invalid regions
    colored[~valid] = 0

    # Convert to uint8
    colored = (colored * 255).astype(np.uint8)

    return colored


def visualize_point_cloud(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    point_size: float = 1.0,
    elevation: float = 30,
    azimuth: float = 45,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """Visualize 3D point cloud.

    Args:
        points: (N, 3) point positions
        colors: (N, 3) RGB colors in [0, 1]
        point_size: Size of points
        elevation: Camera elevation angle
        azimuth: Camera azimuth angle
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    points = _tensor_to_numpy(points)

    if colors is not None:
        colors = _tensor_to_numpy(colors)
    else:
        # Default: color by depth
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        colors = cm.viridis(z_norm)[:, :3]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, s=point_size, alpha=0.8
    )

    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Equal aspect ratio
    max_range = np.abs(points).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    return fig


def visualize_tracks(
    frames: torch.Tensor,
    tracks_2d: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    num_tracks: int = 50,
    track_length: int = 10,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """Visualize 2D point tracks on video frames.

    Args:
        frames: (T, H, W, 3) video frames in [0, 1]
        tracks_2d: (N, T, 2) 2D track positions (normalized or pixel)
        visibility: (N, T) track visibility
        num_tracks: Number of tracks to visualize
        track_length: Number of frames to show track history
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    frames = _tensor_to_numpy(frames)
    tracks_2d = _tensor_to_numpy(tracks_2d)

    if visibility is not None:
        visibility = _tensor_to_numpy(visibility)

    T, H, W, _ = frames.shape
    N = tracks_2d.shape[0]

    # Denormalize if coordinates are in [0, 1]
    if tracks_2d.max() <= 1.0:
        tracks_2d = tracks_2d * np.array([W, H])

    # Select subset of tracks
    track_indices = np.random.choice(N, min(num_tracks, N), replace=False)

    # Create figure with multiple frames
    num_display_frames = min(5, T)
    frame_indices = np.linspace(0, T - 1, num_display_frames, dtype=int)

    fig, axes = plt.subplots(1, num_display_frames, figsize=figsize)
    if num_display_frames == 1:
        axes = [axes]

    # Generate colors for tracks
    colors = cm.rainbow(np.linspace(0, 1, len(track_indices)))

    for ax_idx, t in enumerate(frame_indices):
        ax = axes[ax_idx]
        ax.imshow(frames[t])
        ax.set_title(f'Frame {t}')
        ax.axis('off')

        # Draw tracks
        for idx, track_idx in enumerate(track_indices):
            # Get track history
            start_t = max(0, t - track_length + 1)
            track = tracks_2d[track_idx, start_t:t + 1]

            # Check visibility
            if visibility is not None:
                vis = visibility[track_idx, start_t:t + 1]
                visible_mask = vis > 0.5
            else:
                visible_mask = np.ones(len(track), dtype=bool)

            if not visible_mask.any():
                continue

            # Draw track line
            if len(track) > 1:
                for i in range(len(track) - 1):
                    if visible_mask[i] and visible_mask[i + 1]:
                        ax.plot(
                            [track[i, 0], track[i + 1, 0]],
                            [track[i, 1], track[i + 1, 1]],
                            color=colors[idx], linewidth=1, alpha=0.5
                        )

            # Draw current point
            if visible_mask[-1]:
                ax.scatter(
                    track[-1, 0], track[-1, 1],
                    color=colors[idx], s=20, zorder=10
                )

    plt.tight_layout()
    return fig


def visualize_3d_tracks(
    tracks_3d: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    num_tracks: int = 50,
    elevation: float = 30,
    azimuth: float = 45,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """Visualize 3D point tracks.

    Args:
        tracks_3d: (N, T, 3) 3D track positions
        colors: (N, 3) track colors
        num_tracks: Number of tracks to visualize
        elevation: Camera elevation angle
        azimuth: Camera azimuth angle
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    tracks_3d = _tensor_to_numpy(tracks_3d)

    N, T, _ = tracks_3d.shape

    # Select subset
    track_indices = np.random.choice(N, min(num_tracks, N), replace=False)

    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, len(track_indices)))
    else:
        colors = _tensor_to_numpy(colors)
        colors = colors[track_indices]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for idx, track_idx in enumerate(track_indices):
        track = tracks_3d[track_idx]

        # Draw track line
        ax.plot(
            track[:, 0], track[:, 1], track[:, 2],
            color=colors[idx], linewidth=1, alpha=0.7
        )

        # Mark start and end
        ax.scatter(
            track[0, 0], track[0, 1], track[0, 2],
            color=colors[idx], s=30, marker='o'
        )
        ax.scatter(
            track[-1, 0], track[-1, 1], track[-1, 2],
            color=colors[idx], s=30, marker='^'
        )

    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio
    all_tracks = tracks_3d[track_indices].reshape(-1, 3)
    max_range = np.abs(all_tracks).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    return fig


def save_point_cloud_ply(
    filename: str,
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None
):
    """Save point cloud to PLY file.

    Args:
        filename: Output file path
        points: (N, 3) point positions
        colors: (N, 3) RGB colors in [0, 1] or [0, 255]
        normals: (N, 3) surface normals
    """
    points = _tensor_to_numpy(points)
    N = points.shape[0]

    # Prepare header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if colors is not None:
        colors = _tensor_to_numpy(colors)
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    if normals is not None:
        normals = _tensor_to_numpy(normals)
        header.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])

    header.append("end_header")

    # Write file
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')

        for i in range(N):
            line = f"{points[i, 0]} {points[i, 1]} {points[i, 2]}"

            if colors is not None:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"

            if normals is not None:
                line += f" {normals[i, 0]} {normals[i, 1]} {normals[i, 2]}"

            f.write(line + '\n')
