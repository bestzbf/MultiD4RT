"""Complete pristine OmniX release with isolated, equivalent inference kernels.

Does not import another research project's models package or learned adapters.
Source checkouts and dependency directories are explicit environment settings.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

import torch
from torch.nn import functional as F


def source_root():
    return Path(os.environ['OMNIX_SOURCE']).resolve()


def ms_deform_reference(value, spatial_shapes, sampling_locations, weights, chunk=2048):
    n, _, heads, channels = value.shape
    shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
    levels = value.split([h * w for h, w in shapes], dim=1)
    levels = [v.flatten(2).transpose(1, 2).reshape(n * heads, channels, h, w)
              for v, (h, w) in zip(levels, shapes)]
    result = []
    for start in range(0, sampling_locations.shape[1], chunk):
        loc, attn = sampling_locations[:, start:start + chunk], weights[:, start:start + chunk]
        total = None
        for level, values in enumerate(levels):
            grid = (2 * loc[:, :, :, level] - 1).transpose(1, 2).flatten(0, 1)
            sampled = F.grid_sample(values, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            weight = attn[:, :, :, level].transpose(1, 2).reshape(n * heads, 1, loc.shape[1], -1)
            weighted = (sampled * weight).sum(-1)
            total = weighted if total is None else total + weighted
        result.append(total.view(n, heads * channels, loc.shape[1]).transpose(1, 2))
    return torch.cat(result, dim=1).contiguous()


def audit_deform_reference():
    path = source_root() / 'dependencies/Deformable_DETR/models/ops/functions/ms_deform_attn_func.py'
    name = 'MultiScaleDeformableAttention'
    previous = sys.modules.get(name)
    sys.modules[name] = types.ModuleType(name)
    try:
        spec = importlib.util.spec_from_file_location('_official_deform_reference', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        generator = torch.Generator().manual_seed(7)
        shapes = torch.tensor([[3, 4], [2, 2]])
        values = torch.randn(2, 16, 2, 5, generator=generator)
        locations = torch.rand(2, 11, 2, 2, 3, 2, generator=generator) * 1.4 - .2
        weights = torch.rand(2, 11, 2, 2, 3, generator=generator)
        expected = module.ms_deform_attn_core_pytorch(values, shapes, locations, weights)
        actual = ms_deform_reference(values, shapes, locations, weights, chunk=4)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        return float((actual - expected).abs().max())
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def install_kernels():
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    flash = types.ModuleType('flash_attn_interface')

    def flash_attn_func(q, k, v, **kwargs):
        if kwargs:
            raise ValueError(f'Unsupported FA3 options: {kwargs}')
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                              v.transpose(1, 2), dropout_p=0.).transpose(1, 2)

    flash.flash_attn_func = flash_attn_func
    sys.modules[flash.__name__] = flash
    for name in ['models', 'models.ops', 'models.ops.functions']:
        if name in sys.modules:
            raise RuntimeError('Another models package was already imported')
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    deform = types.ModuleType('models.ops.functions.ms_deform_attn_func')

    class MSDeformAttnFunction:
        @staticmethod
        def apply(value, shapes, starts, locations, weights, step):
            if torch.is_grad_enabled():
                raise RuntimeError('This reference kernel is inference-only')
            expected = torch.cat((starts.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]))
            if not torch.equal(starts, expected):
                raise ValueError('Unexpected attention level offsets')
            return ms_deform_reference(value, shapes, locations, weights)

    deform.MSDeformAttnFunction = MSDeformAttnFunction
    sys.modules[deform.__name__] = deform


def load_model(checkpoint, device):
    deps, da3 = Path(os.environ['OMNIX_DEPS']), Path(os.environ['DA3_SOURCE'])
    sys.path[:0] = [str(source_root()), str(da3 / 'src'), str(deps)]
    install_kernels()
    from omegaconf import OmegaConf
    from src.models.model import CustomModel
    base = OmegaConf.load(source_root() / 'configs/model/default.yaml').net
    release = OmegaConf.load(source_root() / 'configs/experiment/release_train.yaml').model.net
    config = OmegaConf.to_container(OmegaConf.merge(base, release), resolve=True)
    config.pop('_target_')
    config['backbone_args']['checkpointing_start_idx'] = None
    for name in ['st_mod_args', 'offset_head_args', 'trajectory_mod_args']:
        config[name]['use_checkpointing'] = False
    model = CustomModel(**config)
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    state = {key[4:]: value for key, value in payload['state_dict'].items() if key.startswith('net.')}
    model.load_state_dict(state, strict=True, assign=True)
    model._resnet_mean = torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
    model._resnet_std = torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
    model = model.to(device).eval()
    report = dict(strict=True, loaded_tensors=len(state), missing_keys=[], unexpected_keys=[],
                  network_config=config, source=str(source_root()), da3_source=str(da3))
    return model, report
