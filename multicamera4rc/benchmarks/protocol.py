"""Sintel / ScanNet protocol adapters for MultiCamera4RC geometry evaluation.

Reuse the legacy D4RT error functions, without constructing its model or
loading cached tracks/normals. Calibration is evaluation GT only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from multicamera4rc.geometry import unproject_depth
from .camera import compute_relative_pose_error, umeyama_alignment
from .metrics import (compute_depth_metrics, compute_pose_auc,
                           compute_pose_metrics, mean_shift_align_points, paired_coordinate_l1)


SINTEL_14_SEQUENCES = (
    'alley_1', 'alley_2', 'ambush_2', 'ambush_4', 'ambush_5', 'ambush_6', 'ambush_7',
    'bamboo_1', 'bamboo_2', 'bandage_1', 'bandage_2', 'cave_2', 'cave_4', 'market_2',
)
MAX_DEPTH = {'sintel': 70., 'scannet': 10.}


@dataclass
class BenchmarkClip:
    images: torch.Tensor  # [1,T,3,Hpad,Wpad], RGB [0,1]
    depth: torch.Tensor  # [T,Heval,Weval], metric camera-Z GT
    K: torch.Tensor  # [T,3,3], evaluation-grid GT
    c2w: torch.Tensor  # [T,4,4], GT in metres
    frame_indices: list[int]
    content_hw: tuple[int, int]
    raw_hw: tuple[int, int]
    calibration_sha256: str


def resize_intrinsics(K, source_hw, target_hw, *, pixel_centers=False):
    """D4RT GT uses row scaling; model arrays use half-pixel resize centres."""
    sy, sx = target_hw[0]/source_hw[0], target_hw[1]/source_hw[1]
    transform = torch.tensor([[sx, 0, (sx-1)/2 if pixel_centers else 0],
                              [0, sy, (sy-1)/2 if pixel_centers else 0], [0, 0, 1]], dtype=K.dtype)
    return transform @ K


def read_sintel_depth(path):
    with Path(path).open('rb') as f:
        tag = np.fromfile(f, dtype='<f4', count=1)
        shape = np.fromfile(f, dtype='<i4', count=2)
        if len(tag) != 1 or tag[0] != 202021.25 or len(shape) != 2 or min(shape) <= 0:
            raise ValueError('invalid Sintel depth header: '+str(path))
        width, height = map(int, shape)
        data = np.fromfile(f, dtype='<f4')
    if data.size != width*height:
        raise ValueError('truncated Sintel depth: '+str(path))
    return data.reshape(height, width)


def read_sintel_camera(path):
    with Path(path).open('rb') as f:
        tag = np.fromfile(f, dtype='<f4', count=1)
        matrices = np.fromfile(f, dtype='<f8')
    if len(tag) != 1 or tag[0] != 202021.25 or matrices.size != 21:
        raise ValueError('invalid Sintel camera: '+str(path))
    K = matrices[:9].reshape(3, 3).astype(np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3] = matrices[9:].reshape(3, 4)
    return K, w2c


def frame_indices(total, frames, stride=1, start=0, *, dataset):
    """Match the old consecutive-window choices; never duplicate real frames."""
    if min(total, frames, stride) < 1 or start < 0 or start >= total:
        raise ValueError('invalid frame window')
    if dataset == 'sintel':
        begin = min(start, max(0, total-frames))
        return list(range(begin, min(total, begin+frames)))
    if start+frames*stride <= total:
        return list(range(start, start+frames*stride, stride))
    step = max((total-start)//frames, 1)
    return list(range(start, min(total, start+frames*step), step))


def load_clip(case, *, eval_size=256, long_side=518, patch_size=14):
    """Read only RGB, depth and calibration selected by a frozen scene case."""
    dataset, root, scene = case['dataset'], Path(case['root']), case['scene']
    if not re.fullmatch(r'[A-Za-z0-9_]+', scene):
        raise ValueError('unsafe scene name')
    ids = case['frame_indices']
    if not ids or ids != sorted(set(ids)):
        raise ValueError('evaluation requires unique increasing frame indices')
    if dataset == 'scannet':
        import h5py
        processed = root/scene/'processed'
        with h5py.File(processed/'precomputed.h5', 'r') as f:
            intrinsics = np.stack([f['intrinsics'][i] for i in ids]).astype(np.float32)
            cached_poses = np.stack([f['extrinsics'][i] for i in ids]) if 'extrinsics' in f else None
        names = [f'{i:05d}' for i in ids]
        image_paths = [processed/'images'/(n+'.jpg') for n in names]
        depth_paths = [processed/'depth'/(n+'.png') for n in names]
        poses = np.stack([np.loadtxt(processed/'pose'/(n+'.txt'), dtype=np.float32) for n in names])
        if cached_poses is not None and not np.allclose(poses, cached_poses, atol=1e-5, rtol=1e-5):
            raise ValueError('ScanNet pose files disagree with cached w2c matrices')
    elif dataset == 'sintel':
        files = sorted((root/'training/final'/scene).glob('frame_*.png'))
        image_paths = [files[i] for i in ids]
        depth_paths = [root/'training/depth'/scene/(p.stem+'.dpt') for p in image_paths]
        calibration = [read_sintel_camera(root/'training/camdata_left'/scene/(p.stem+'.cam')) for p in image_paths]
        intrinsics, poses = map(np.stack, zip(*calibration))
    else:
        raise ValueError('unsupported monocular benchmark: '+dataset)
    if (not np.isfinite(intrinsics).all() or not np.isfinite(poses).all()
            or (intrinsics[:, [0, 1], [0, 1]] <= 0).any()):
        raise ValueError('invalid GT calibration')
    rotations = poses[:, :3, :3]
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3)
            or not np.allclose(poses[:, 3], [0, 0, 0, 1])):
        raise ValueError('GT poses are not finite rigid transforms')
    images, depths = [], []
    raw_hw = content_hw = None
    for rgb_path, depth_path in zip(image_paths, depth_paths):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if raw_hw is None:
            raw_hw = rgb.shape[:2]
            scale = min(1., long_side/max(raw_hw)) if long_side else 1.
            content_hw = tuple(max(1, round(n*scale)) for n in raw_hw)
        if rgb.shape[:2] != raw_hw:
            raise ValueError('scene contains variable RGB dimensions')
        if dataset == 'sintel':
            depth = read_sintel_depth(depth_path)
            if depth.shape != raw_hw:
                raise ValueError('Sintel depth and RGB dimensions disagree')
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
                raise ValueError('ScanNet depth must be uint16 millimetres: '+str(depth_path))
            depth = depth.astype(np.float32)/1000.
            # Same nearest-neighbour registration to RGB as the legacy adapter.
            depth = cv2.resize(depth, (raw_hw[1], raw_hw[0]), interpolation=cv2.INTER_NEAREST)
        images.append(torch.from_numpy(cv2.resize(rgb, content_hw[::-1], interpolation=cv2.INTER_LINEAR).copy()).permute(2, 0, 1))
        depths.append(cv2.resize(depth, (eval_size, eval_size), interpolation=cv2.INTER_NEAREST))
    depth = torch.from_numpy(np.stack(depths).copy()).float()
    valid = torch.isfinite(depth) & (depth > 0) & (depth < MAX_DEPTH[dataset])
    if not valid.flatten(1).any(1).all():
        raise ValueError('a requested frame has no valid benchmark depth')
    digest = hashlib.sha256()
    for values in (intrinsics, poses, depth.numpy()):
        digest.update(np.ascontiguousarray(values).tobytes())
    K = resize_intrinsics(torch.from_numpy(intrinsics.copy()).float(), raw_hw, (eval_size, eval_size))
    images = torch.stack(images).float()/255
    h, w = content_hw
    hp, wp = max(2*patch_size, ((h+patch_size-1)//patch_size)*patch_size), max(2*patch_size, ((w+patch_size-1)//patch_size)*patch_size)
    images = F.pad(images, (0, wp-w, 0, hp-h))[None]
    return BenchmarkClip(images, depth, K, torch.linalg.inv(torch.from_numpy(poses).float()),
                         ids, content_hw, raw_hw, digest.hexdigest())


def align_depth_per_frame(pred, gt, mask, mode, max_depth):
    """The legacy evaluators' per-frame S / robust median-centred SS fit."""
    out = pred.clamp(min=1e-6, max=max_depth).clone()
    for i in range(len(out)):
        p, g = out[i][mask[i]], gt[i][mask[i]]
        if not len(p):
            raise ValueError('empty frame in depth metric')
        scale = (g/p.clamp_min(1e-6)).median()
        shift = p.new_tensor(0.)
        if mode == 'scale_shift':
            pc, gc = p-p.median()+1e-6, g-g.median()+1e-6
            valid = torch.isfinite(pc) & torch.isfinite(gc) & (pc.abs() > 1e-6)
            candidate = (gc[valid]/pc[valid]).median() if valid.any() else scale
            if torch.isfinite(candidate) and candidate > 0:
                scale, shift = candidate, (g-candidate*p).median()
        elif mode != 'scale':
            raise ValueError('unknown depth alignment')
        out[i] = out[i]*scale+shift
    return out


def geometry_on_eval_grid(geometry, clip):
    h, w = clip.content_hw
    depth = geometry['depth'][0, :, :h, :w].detach().float().cpu()
    depth = F.interpolate(depth[:, None], size=clip.depth.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    K = resize_intrinsics(geometry['K'][0].detach().float().cpu(), (h, w), clip.depth.shape[-2:], pixel_centers=True)
    c2w = geometry['c2w'][0].detach().float().cpu()
    if not all(torch.isfinite(v).all() for v in (depth, K, c2w)):
        raise ValueError('non-finite prediction; refusing to hide failed pixels')
    return depth, K, c2w


def geometry_metrics(pred_depth, pred_K, pred_c2w, clip, *, dataset):
    limit = MAX_DEPTH[dataset]
    mask = torch.isfinite(clip.depth) & (clip.depth > 0) & (clip.depth < limit)
    if (pred_depth <= 0).any() or not torch.isfinite(pred_depth).all():
        raise ValueError('model returned non-positive or non-finite depth')
    result = dict(depth_valid_ratio=float(mask.float().mean()), depth_num_valid_pixels=int(mask.sum()))
    for prefix, mode in [('depth_S', 'scale'), ('depth_SS', 'scale_shift')]:
        aligned = align_depth_per_frame(pred_depth, clip.depth, mask, mode, limit)
        metrics = compute_depth_metrics(aligned, clip.depth, mask=mask, scale_invariant=False)
        result.update({prefix+'_'+k: float(v) for k, v in metrics.items()})
    # A single scale for the entire clip is evaluation gauge alignment only.
    # Per-frame depth fits above are never used to repair the 3D reconstruction.
    depth_scale = (clip.depth[mask]/pred_depth[mask]).median()
    pred_ref = torch.linalg.inv(pred_c2w[0]) @ pred_c2w
    gt_ref = torch.linalg.inv(clip.c2w[0]) @ clip.c2w
    pp = unproject_depth(pred_depth[None], pred_K[None], pred_ref[None])[0][mask].double()
    gp = unproject_depth(clip.depth[None], clip.K[None], gt_ref[None])[0][mask].double()
    shifted, _ = mean_shift_align_points(pp, gp)
    scaled, _ = mean_shift_align_points(pp*depth_scale, gp)
    R, t, s = umeyama_alignment(pp, gp, with_scale=True)
    result.update(pointcloud_num_points=len(pp), pointcloud_l1_raw_gauge=float(paired_coordinate_l1(shifted, gp)),
        pointcloud_l1_scale_shift_m=float(paired_coordinate_l1(scaled, gp)),
        pointcloud_l1_sim3_m=float(paired_coordinate_l1(s*(pp @ R.T)+t, gp)),
        pointcloud_depth_scale=float(depth_scale), pointcloud_sim3_scale=float(s))
    # Same ATE / RPE / AUC error functions as D4RT; use the current camera head.
    poses = compute_pose_metrics(pred_c2w.double(), clip.c2w.double(), align=True)
    result.update(pose_ate_m=float(poses['ate']), pose_rpe_trans_m=float(poses['rpe_trans']),
                  pose_rpe_rot_deg=float(poses['rpe_rot']))
    rotation, translation = [], []
    for i in range(1, len(pred_c2w)):
        pr = torch.linalg.inv(pred_c2w[i].double()) @ pred_c2w[i-1].double()
        gt = torch.linalg.inv(clip.c2w[i].double()) @ clip.c2w[i-1].double()
        errors = compute_relative_pose_error(pr[:3, :3], pr[:3, 3], gt[:3, :3], gt[:3, 3])
        rotation.append(errors['rotation_error']); translation.append(errors['translation_error'])
    if not rotation:
        raise ValueError('pose benchmark requires at least two real frames')
    result.update(pose_num_pairs=len(rotation), pose_auc_30_percent=float(
        compute_pose_auc(torch.stack(rotation), torch.stack(translation), threshold=30.)*100))
    return result
