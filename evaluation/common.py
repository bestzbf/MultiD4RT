from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

RELEASE_SHA256 = '1161d6bfaa0d277b22108c5b9e803b9e1c36e289f98799e35bd59a9b821197e0'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_model(method, checkpoint, device, omnix_runner=None):
    checksum = digest(checkpoint)
    if method == 'multid4rt':
        from multid4rt import MultiD4RT
        from multid4rt.checkpoint import config_from_checkpoint
        if checksum != RELEASE_SHA256:
            raise ValueError('This evaluation requires the published, byte-identical 60k checkpoint')
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        if payload['step'] != 60000:
            raise ValueError('Wrong checkpoint step')
        config = replace(config_from_checkpoint(payload), gradient_checkpointing=False)
        model = MultiD4RT(config)
        model.load_state_dict(payload['model'], strict=True)
        report = dict(strict=True, tensors=len(payload['model']), step=60000,
                      config=config.to_dict())
        del payload
        model = model.to(device).eval()
    elif method == 'omnix':
        if not omnix_runner:
            raise ValueError('Specify the complete native OmniX inference runner')
        runner = load_module('_comparison_native_omnix', omnix_runner)
        kernel_error = runner.audit_deform_reference()
        model, report = runner.load_model(checkpoint, device)
        report['deform_reference_max_error'] = kernel_error
        report['kernel_substitutions'] = ['FA3 to PyTorch SDPA', 'upstream MSDeformAttn grid_sample reference']
        report['runner_sha256'] = digest(omnix_runner)
    else:
        raise ValueError(method)
    model.requires_grad_(False)
    report.update(method=method, checkpoint=str(Path(checkpoint).resolve()),
                  checkpoint_sha256=checksum, parameters=sum(p.numel() for p in model.parameters()),
                  torch=torch.__version__, numpy=np.__version__)
    return model, report


def prepare_rgb(rgb, long_side=504, patch_size=14):
    """Entire RGB field of view, shared aspect-preserving resize and patch padding."""
    array = np.stack(rgb)
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError('Expected RGB uint8 [T,H,W,3]')
    raw_h, raw_w = array.shape[1:3]
    scale = long_side / max(raw_h, raw_w)
    h, w = max(1, round(raw_h * scale)), max(1, round(raw_w * scale))
    images = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float() / 255
    images = F.interpolate(images, size=(h, w), mode='bilinear', align_corners=False)
    hp, wp = ((v + patch_size - 1) // patch_size * patch_size for v in (h, w))
    images = F.pad(images, (0, wp - w, 0, hp - h))
    return images[None], dict(raw_hw=[raw_h, raw_w], content_hw=[h, w], padded_hw=[hp, wp],
                             rgb_sha256=hashlib.sha256(array.tobytes()).hexdigest())


@torch.inference_mode()
def predict_geometry(model, method, images, frame_indices, device):
    images = images.to(device)
    t = images.shape[1]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        if method == 'multid4rt':
            cameras = torch.zeros((1, t), device=device, dtype=torch.long)
            times = torch.tensor(frame_indices, device=device, dtype=torch.long)[None]
            encoded = model.encode(images, cameras, times)
            output = encoded.geometry
            depth, intrinsics, pose = output['depth'], output['K'], output['c2w']
        else:
            ids = torch.arange(t, device=device)
            info = torch.stack((ids, torch.zeros_like(ids), ids, ids), -1)[None]
            output = model(dict(image=images, image_info=info))
            depth, intrinsics, pose = output['depth'], output['intrinsics'], output['camera_pose']
        depth, intrinsics, pose = (x[0].detach().float().cpu() for x in (depth, intrinsics, pose))
    if pose.shape[-2:] == (3, 4):
        row = pose.new_tensor([0, 0, 0, 1]).expand(t, 1, 4)
        pose = torch.cat((pose, row), -2)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3 or pose.shape != (t, 4, 4) or intrinsics.shape != (t, 3, 3):
        raise ValueError(f'Unexpected geometry shapes: {depth.shape}, {pose.shape}, {intrinsics.shape}')
    if not all(torch.isfinite(x).all() for x in (depth, intrinsics, pose)) or not (depth > 0).all():
        raise ValueError('Invalid model prediction; no failed pixels may be removed')
    return dict(depth=depth.numpy(), intrinsics=intrinsics.numpy(), c2w=pose.numpy())


def read_sintel_depth(path):
    with Path(path).open('rb') as stream:
        tag = np.fromfile(stream, '<f4', 1)
        shape = np.fromfile(stream, '<i4', 2)
        data = np.fromfile(stream, '<f4')
    if len(tag) != 1 or tag[0] != 202021.25 or len(shape) != 2:
        raise ValueError('Invalid Sintel depth header')
    w, h = map(int, shape)
    if min(h, w) <= 0 or data.size != h * w:
        raise ValueError('Truncated Sintel depth')
    return data.reshape(h, w)


def read_sintel_camera(path):
    with Path(path).open('rb') as stream:
        tag = np.fromfile(stream, '<f4', 1)
        values = np.fromfile(stream, '<f8')
    if len(tag) != 1 or tag[0] != 202021.25 or len(values) != 21:
        raise ValueError('Invalid Sintel camera file')
    intrinsics = values[:9].reshape(3, 3)
    w2c = np.eye(4)
    w2c[:3] = values[9:].reshape(3, 4)
    return intrinsics, np.linalg.inv(w2c)
