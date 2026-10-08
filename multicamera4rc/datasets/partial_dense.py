"""Dense partial-task labels. Missing motion is unknown, never zero motion.

Replica endpoints use native flow and depth; VKITTI endpoints use its native
camera-coordinate scene flow, including endpoints hidden in the target view.
Neither path rasterizes a particle/mesh track pool.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..geometry import unproject_depth
from ..metadata import gather_observations


def _world(uv, depth, K, E):
    pixels = np.concatenate((uv, np.ones((*uv.shape[:-1], 1))), -1)
    camera = (pixels @ np.linalg.inv(K).T) * depth[..., None]
    C = np.linalg.inv(E)
    return camera @ C[:3, :3].T + C[:3, 3]


def _flow16(path):
    import cv2
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.dtype != np.uint16 or raw.shape[1] % 2:
        raise ValueError(f"invalid Replica float16 flow: {path}")
    a = np.ascontiguousarray(raw).view(np.float16).astype(np.float32)
    w = a.shape[1] // 2
    return np.stack((a[:, :w], a[:, w:]), -1)


def replica_pairs(clip):
    import cv2
    base = Path(clip.metadata["sequence_dir"])
    sequence = base.name
    frames = clip.metadata["frame_numbers"]
    indices = {int(f): i for i, f in enumerate(frames)}
    K, E = np.asarray(clip.intrinsics), np.asarray(clip.extrinsics)
    depths = [np.asarray(d, np.float32) for d in clip.depths]
    kernel = np.ones((3, 3), np.uint8)

    def interior(d):
        lo, hi = cv2.erode(d, kernel), cv2.dilate(d, kernel)
        return np.isfinite(d) & (lo > 0) & ((hi-lo) < .01*d+.02)

    def sample(a, uv, nearest=False):
        return cv2.remap(a, uv[..., 0], uv[..., 1],
                         cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    safe = [interior(d) for d in depths]
    result = []
    for s, frame in enumerate(frames):
        h, w = depths[s].shape
        y, x = np.indices((h, w)); uv = np.stack((x, y), -1).astype(np.float32)
        world0 = _world(uv, depths[s], K[s], E[s])
        row = []
        for direction, step, reverse in (("forward", 1, "backward"), ("backward", -1, "forward")):
            t = indices.get(int(frame)+step)
            if t is None:
                row.append(None); continue
            fp = base/f"flow_{direction}"/f"{sequence}_{frame:04d}.png"
            rp = base/f"flow_{reverse}"/f"{sequence}_{frames[t]:04d}.png"
            mp = base/f"flow_{direction}_mask"/fp.name
            bp = base/f"flow_{reverse}_mask"/rp.name
            # Right-camera releases can lack motion; they remain geometry-only.
            if not all(p.is_file() for p in (fp, rp, mp, bp)):
                if clip.metadata.get("camera_name") == "right":
                    row.append(None); continue
                raise FileNotFoundError(f"missing official Replica pair labels: {fp}")
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            b = cv2.imread(str(bp), cv2.IMREAD_UNCHANGED)
            if m is None or b is None or m.shape != (h, w) or b.shape != (h, w):
                raise ValueError("Replica flow/mask dimensions disagree with RGB/depth")
            # Some native files are 8-bit all-zero placeholders where the
            # official validity mask is empty. The cycle check requires both
            # directions, so an empty mask makes this pair wholly unknown.
            # Keep its geometry, but never decode the placeholder as float16
            # or manufacture zero-motion supervision. Nonempty pairs retain
            # strict validation of both native flow files.
            if not (m > 0).any() or not (b > 0).any():
                row.append(None); continue
            flow, backward = _flow16(fp), _flow16(rp)
            if flow.shape != (h, w, 2) or backward.shape != (h, w, 2):
                raise ValueError("Replica flow/mask dimensions disagree with RGB/depth")
            endpoint = uv + flow
            finite = np.isfinite(flow).all(-1)
            coords = np.nan_to_num(endpoint, nan=-1e6, posinf=1e6, neginf=-1e6)
            inside = finite & (coords[..., 0] >= 0) & (coords[..., 0] <= w-1) & (coords[..., 1] >= 0) & (coords[..., 1] <= h-1)
            z = sample(depths[t], coords)
            composition = flow + sample(backward, coords)
            # Native half precision is coarser for large flow. A relative
            # tolerance is used in addition to one pixel for cycle consistency.
            cycle = np.linalg.norm(composition, axis=-1) <= 1.+.01*np.linalg.norm(flow, axis=-1)
            valid = (m > 0) & (sample((b > 0).astype(np.uint8), coords, True) > 0)
            valid &= inside & safe[s] & (sample(safe[t].astype(np.uint8), coords, True) > 0) & (z > 0) & np.isfinite(z) & cycle
            delta = _world(coords, z, K[t], E[t]) - world0
            valid &= np.isfinite(delta).all(-1)
            row.append((t, np.where(valid[..., None], delta, 0).astype(np.float32), valid))
        result.append(row)
    return result


def vkitti_pairs(clip, adapter):
    import cv2
    frames = clip.metadata["frame_indices"]
    cameras = clip.metadata.get("camera_names", [adapter.camera]*len(frames)) if hasattr(adapter, "camera") else clip.metadata["camera_names"]
    indices = {(c, int(f)): i for i, (c, f) in enumerate(zip(cameras, frames))}
    base = adapter.seq_to_dir[clip.sequence_name]/"frames"
    K, E = np.asarray(clip.intrinsics), np.asarray(clip.extrinsics)
    result = []
    for s, frame in enumerate(frames):
        d = np.asarray(clip.depths[s], np.float32); h, w = d.shape
        y, x = np.indices((h, w)); uv = np.stack((x, y), -1)
        pixels = np.concatenate((uv, np.ones((h, w, 1))), -1)
        cam0 = (pixels @ np.linalg.inv(K[s]).T)*d[..., None]
        world0 = _world(uv, d, K[s], E[s])
        row = []
        for step, sfdir, sfprefix, fdir, fprefix in (
                (1, "forwardSceneFlow", "sceneFlow", "forwardFlow", "flow"),
                (-1, "backwardSceneFlow", "backwardSceneFlow", "backwardFlow", "backwardFlow")):
            t = indices.get((cameras[s], int(frame)+step))
            if t is None:
                row.append(None); continue
            sp = base/sfdir/cameras[s]/f"{sfprefix}_{frame:05d}.png"
            fp = base/fdir/cameras[s]/f"{fprefix}_{frame:05d}.png"
            raw = cv2.imread(str(sp), cv2.IMREAD_UNCHANGED)
            optical = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if raw is None or optical is None:
                raise FileNotFoundError(f"missing native VKITTI scene/optical flow: {sp}")
            if raw.shape != (h, w, 3) or optical.shape != raw.shape or raw.dtype != np.uint16 or optical.dtype != np.uint16:
                raise ValueError("VKITTI scene-flow encoding must be native RGB16")
            sf = (2.*raw[..., ::-1].astype(np.float64)/65535.-1.)*10.
            C1 = np.linalg.inv(E[t])
            end = (cam0+sf) @ C1[:3, :3].T + C1[:3, 3]
            delta = end-world0
            valid = (d > 0) & (d < 655.35) & (optical[..., 0] > 0)
            valid &= ~((raw == 0).any(-1) | (raw == 65535).any(-1)) & np.isfinite(delta).all(-1)
            cam1 = cam0+sf
            projection = cam1 @ K[t].T
            projected_uv = projection[..., :2]/np.maximum(projection[..., 2:], 1e-8)
            flow_uv = uv+(2.*optical[..., 2:0:-1].astype(np.float64)/65535.-1.)*np.array([w-1, h-1])
            valid &= (cam1[..., 2] > 0) & (np.linalg.norm(projected_uv-flow_uv, axis=-1) <= 1.5)
            # No target-depth visibility test: native scene flow also labels
            # valid endpoints which are occluded or outside the target image.
            row.append((t, np.where(valid[..., None], delta, 0).astype(np.float32), valid))
        result.append(row)
    return result


def attach_partial_dense_recipe(clip, sample, *, kind, image_mode, adapter=None):
    from .bridge import _content_size
    if kind not in ("geometry", "replica_flow", "vkitti_scene_flow", "pointodyssey_tracks"):
        raise ValueError(f"unsupported dense partial task: {kind}")
    target = {k: v for k, v in sample["targets"].items()
              if kind == "pointodyssey_tracks" or not k.startswith("sparse_")}
    S, _, H, W = sample["images"].shape
    target["dense_partial"] = torch.tensor(True)
    # Compact adjacent recipes, independent of the requested T and Q. World
    # displacement uses the same canonical gauge as geometry supervision.
    delta = torch.zeros(S, 2, H, W, 3)
    valid = torch.zeros(S, 2, H, W, dtype=torch.bool)
    times = torch.full((S, 2), -1, dtype=torch.long)
    if kind not in ("geometry", "pointodyssey_tracks"):
        pairs = replica_pairs(clip) if kind == "replica_flow" else vkitti_pairs(clip, adapter)
        E = np.asarray(clip.extrinsics[0], np.float64)
        if clip.metadata.get("extrinsics_convention", "w2c") == "c2w":
            E = np.linalg.inv(E)
        for s, row in enumerate(pairs):
            for d, pair in enumerate(row):
                if pair is None:
                    continue
                t, dv, mask = pair
                content = _content_size(*mask.shape, (H, W), image_mode)
                dv = np.where(mask[..., None], dv @ E[:3, :3].T, 0).astype(np.float32)
                xyz = torch.from_numpy(dv).permute(2, 0, 1)[None]
                m = torch.from_numpy(mask.copy())[None, None].float()
                if tuple(mask.shape) != content:
                    xyz = F.interpolate(xyz, size=content, mode="nearest-exact")
                    m = F.interpolate(m, size=content, mode="nearest-exact")
                ch, cw = content
                delta[s, d, :ch, :cw] = xyz[0].permute(1, 2, 0)
                valid[s, d, :ch, :cw] = m[0, 0].bool()
                times[s, d] = sample["time_idx"][t]
    target.update(dense_pair_delta=delta, dense_pair_valid=valid, dense_pair_time=times)
    sample["targets"] = target


@torch.no_grad()
def materialize_partial_dense_targets(batch, source_indices, *, normalize_scene=True):
    recipe = batch["targets"]
    target = {k: v for k, v in recipe.items() if not k.startswith("dense_")}
    depth, K, camera = target["depth"].float(), target["K"].float(), target["c2w"].float()
    base = unproject_depth(depth, K, camera)
    ov = batch["observation_valid"] & target["camera_valid"].bool()
    pv = target["depth_valid"].bool() & ov[..., None, None]
    count = pv.sum((1, 2, 3))
    if (count == 0).any():
        raise ValueError("dense geometry clip has no valid calibrated depth pixels")
    scale = (torch.where(pv, base.norm(dim=-1), 0).sum((1, 2, 3))/count).clamp_min(1e-8)
    if not normalize_scene:
        scale = torch.ones_like(scale)
    depth = depth/scale[:, None, None, None]
    camera = camera.clone(); camera[..., :3, 3] /= scale[:, None, None]
    base = base/scale[:, None, None, None, None]
    source_base = gather_observations(base, source_indices)
    source_valid = gather_observations(pv, source_indices)
    st = gather_observations(batch["time_idx"], source_indices)
    timeline = batch["query_time_idx"]
    identity = st[..., None] == timeline[:, None]
    b, q, h, w = source_valid.shape; T = timeline.shape[1]
    trajectory_valid = identity[..., None, None] & source_valid[:, :, None]
    displacement = torch.zeros(b, q, T, h, w, 3, device=depth.device)
    motion_valid = torch.zeros_like(trajectory_valid)
    pair_delta = gather_observations(recipe["dense_pair_delta"], source_indices)/scale[:, None, None, None, None, None]
    pair_valid = gather_observations(recipe["dense_pair_valid"], source_indices) & source_valid[:, :, None]
    pair_time = gather_observations(recipe["dense_pair_time"], source_indices)
    # Identity is valid base-point geometry, not a negative motion label.
    # Retain identity in displacement_valid only for sources with actual pair
    # labels so the finite-difference velocity loss can use the adjacent pair.
    known_source = pair_valid.any(dim=(2, 3, 4))
    motion_valid |= trajectory_valid & known_source[..., None, None, None]
    for direction in range(2):
        match = pair_time[:, :, direction, None] == timeline[:, None]
        mask = match[..., None, None] & pair_valid[:, :, direction, None]
        displacement = torch.where(mask[..., None], pair_delta[:, :, direction, None], displacement)
        motion_valid |= mask
        trajectory_valid |= mask
    xyz = torch.where(trajectory_valid[..., None], source_base[:, :, None]+displacement, 0)
    origin = camera[..., :3, 3][:, :, None, None].expand(*depth.shape, 3)
    direction = unproject_depth(torch.ones_like(depth), K, camera)-origin
    # Threshold affects diagnostics/foreground balancing only, never validity.
    foreground = (displacement.norm(dim=-1) > .002) & motion_valid
    target.update(depth=depth, c2w=camera, ray=torch.cat((direction, origin), -1), ray_valid=pv,
                  trajectory=xyz, trajectory_valid=trajectory_valid,
                  displacement=displacement, displacement_valid=motion_valid,
                  trajectory_foreground=foreground, source_pixel_valid=source_valid,
                  static_mask=torch.zeros_like(source_valid), scene_scale=scale)
    for name in ("sparse_trajectory", "sparse_displacement"):
        if name in target:
            target[name] = target[name]/scale[:, None, None, None]
    return target
