from .camera import (
    umeyama_alignment,
    estimate_camera_pose,
    estimate_intrinsics,
    project_points,
    unproject_points
)

from .geometry import project_3d_to_2d, compute_surface_normal, GeomUtils
from .misc import farthest_point_sample_py

from .metrics import (
    compute_depth_metrics,
    compute_point_cloud_metrics,
    compute_pose_auc,
    compute_pose_metrics,
    compute_tracking_metrics
)
from .patches import extract_local_patches


def __getattr__(name):
    # Metric-only evaluation does not need matplotlib. Keep visualization
    # helpers available to existing callers when they are actually requested.
    if name in {"visualize_depth", "visualize_point_cloud", "visualize_tracks"}:
        from importlib import import_module
        return getattr(import_module(".visualization", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'umeyama_alignment',
    'estimate_camera_pose',
    'estimate_intrinsics',
    'project_points',
    'unproject_points',
    'compute_depth_metrics',
    'compute_point_cloud_metrics',
    'compute_pose_auc',
    'compute_pose_metrics',
    'compute_tracking_metrics',
    'visualize_depth',
    'visualize_point_cloud',
    'visualize_tracks',
    'extract_local_patches',
]
