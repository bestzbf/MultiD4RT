"""Group verified static-scene views on the camera axis, at one shared time."""
from __future__ import annotations

from dataclasses import replace
import numpy as np


def camera_choices(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("static cameras must be a positive integer or a nonempty list of positive integers")
    if len(set(values)) != len(values):
        raise ValueError("static camera choices must be unique")
    return tuple(sorted(values))


def select_static_views(adapter, sequence, frames, anchor, choices, rng):
    """Keep the coverage anchor and select distinct, same-scene neighbors.

    Packed MVS/Taskonomy uses its authored covisibility graph. Hypersim uses
    nearby camera centers with compatible directions and measured anchor
    surfaces projecting into the target frustum. A short/isolated neighborhood remains a smaller real clip;
    observations are never repeated to meet a nominal camera count.
    """
    allowed = set(frames)
    if anchor not in allowed:
        raise ValueError("static anchor must belong to the admitted frame pool")
    verified = getattr(adapter, "get_static_covisibility_neighbors", None)
    getter = getattr(adapter, "get_covisibility_neighbors", None)
    neighbors = (verified(sequence, anchor, allowed) if verified is not None else
                 getter(sequence, anchor) if getter is not None else None)
    method = "packed_depth_covisibility" if verified is not None else "packed_covisibility"
    if neighbors is None:
        record = getattr(adapter, "records", {}).get(sequence, {})
        if "E" in record and "camera_centers" in record:
            E = np.asarray(record["E"], np.float64)
            centers = np.asarray(record["camera_centers"], np.float64)
            candidates = np.array([i for i in frames if i != anchor], dtype=np.int64)
            forward = E[:, 2, :3]
            alignment = forward[candidates] @ forward[anchor]
            candidates = candidates[alignment >= .5]  # at most 60 degrees
            # Nearby poses can face different rooms. Use measured anchor
            # depth to reject views outside its visible surface frustum.
            # This is data selection only; calibration/depth never enter
            # the model's input tensors.
            if "K" in record and "native_hw" in record:
                clip = adapter.load_clip(sequence, [anchor])
                depth = np.asarray(clip.depths[0])
                yy, xx = np.mgrid[0:depth.shape[0]:16, 0:depth.shape[1]:16]
                z = depth[yy.ravel(), xx.ravel()]
                valid = np.isfinite(z) & (z > 0)
                pixels = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], -1)[valid]
                camera = (pixels @ np.linalg.inv(clip.intrinsics[0]).T)*z[valid, None]
                pose = clip.extrinsics[0]
                world = (camera-pose[:3, 3]) @ pose[:3, :3]
                visible = []
                h, w = record["native_hw"]
                for i in candidates:
                    xyz = world @ E[i, :3, :3].T+E[i, :3, 3]
                    projected = xyz @ np.asarray(record["K"]).T
                    uv = projected[:, :2]/np.maximum(projected[:, 2:], 1e-10)
                    inside = ((xyz[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                              & (uv[:, 1] >= 0) & (uv[:, 1] < h))
                    if len(inside) and inside.mean() >= .2:
                        visible.append(i)
                candidates = np.asarray(visible, dtype=np.int64)
                method = "renderer_depth_frustum"
            else:
                method = "renderer_pose_neighbors"
            distance = np.linalg.norm(centers[candidates]-centers[anchor], axis=-1)
            neighbors = candidates[np.argsort(distance, kind="stable")].tolist()
        else:
            neighbors = []
            method = "isolated_view"
    neighbors = list(dict.fromkeys(int(i) for i in neighbors if int(i) in allowed and int(i) != anchor))
    fitting = [c for c in choices if c <= len(neighbors)+1]
    count = rng.choice(fitting) if fitting else len(neighbors)+1
    # Select among the strongest neighbors, preserving varied baselines across
    # samples without drawing unrelated rooms/views from an entire building.
    candidates = neighbors[:max(16, 2*count)]
    selected = [anchor]+rng.sample(candidates, count-1)
    return selected, dict(selection=method, camera_choices=list(choices), actual_cameras=count,
                          available_neighbors=len(neighbors), view_indices=selected, time_idx=0)


def as_static_multiview(clip, info):
    count = len(clip.images)
    if count != info["actual_cameras"] or len(set(info["view_indices"])) != count:
        raise ValueError("static multiview requires one distinct image per camera slot")
    if clip.trajs_3d_world is not None or clip.trajs_2d is not None:
        raise ValueError("static view grouping cannot reinterpret temporal track labels")
    metadata = dict(clip.metadata)
    metadata.update(slot_physical_camera_id=np.arange(count, dtype=np.int64),
                    slot_camera=np.arange(count, dtype=np.int64),
                    time_idx=np.zeros(count, dtype=np.int64), num_cameras=count, num_times=1,
                    view_mode="static_multiview", static_multiview=dict(info))
    return replace(clip, metadata=metadata)
