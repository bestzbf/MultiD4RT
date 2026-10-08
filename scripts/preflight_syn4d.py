"""Run native Syn4D layouts through the saved training model and loss.

Does not update weights. The memory limit also permits a small smoke check
beside an existing run; production layouts must pass separately with enough
space for the restored optimizer and DDP buffers.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.adapters.syn4d import Syn4DAdapter
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import collate_adapter_samples, read_data_config
from multicamera4rc.datasets.syn4d_dense import attach_syn4d_dense_recipe
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.train import _sources, _TrainingStep
from scripts.cutover_native_motion_stereo import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--memory-fraction', type=float, default=.85)
    parser.add_argument('--layouts', nargs='+', help='Optional smoke layouts, e.g. 2x2; not a production proof')
    parser.add_argument('--scenes', nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    data = read_data_config(args.data_config)
    if data.get('image_mode') != 'native' or data.get('supervision') != 'dense':
        raise ValueError('preflight requires native dense data')
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    cfg = replace(config_from_checkpoint(payload), attention_chunk_size=None,
                  visibility_chunk_size=None, gradient_checkpointing=True)
    loss_config = payload['training_state']['loss_config']
    model = MultiCamera4RC.from_checkpoint(payload, config=cfg).cuda().train()
    checkpoint_step = payload['step']
    del payload
    stepper = _TrainingStep(model, MultiCamera4RCLoss(**loss_config))
    report = dict(passed=False, checkpoint=args.checkpoint, checkpoint_step=checkpoint_step,
                  data_config=args.data_config, production_layouts=args.layouts is None,
                  memory_fraction=args.memory_fraction, loss_config=loss_config,
                  native_resolution=True, no_query_chunking=True, optimizer_updated=False, samples=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for source in data['sources']:
        if source['adapter'] != 'syn4d':
            continue
        options = dict(source.get('options', {}))
        if args.scenes:
            options['scenes'] = args.scenes
        adapter = Syn4DAdapter(source['root'], split=source.get('split', 'train'), **options)
        layouts = ([(int(c), int(t)) for c, t in (s.split('x') for s in args.layouts)]
                   if args.layouts else [(v['cameras'], v['num_times']) for v in source['layouts']])
        for C, T in layouts:
            began = time.monotonic()
            eligible = [s for s in adapter.sequences if adapter.get_num_frames(s) >= T
                        and adapter.get_sequence_info(s)['num_cameras'] >= C]
            if not eligible:
                raise ValueError(f'no eligible sequence for {source["name"]} C{C}T{T}')
            sequence = eligible[0]
            # Test the largest admitted temporal stride as well as the tensor size.
            stride = max(v for v in source.get('strides', [1])
                         if 1+(T-1)*v <= adapter.get_num_frames(sequence))
            positions = [t*stride for t in range(T)]
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            clip = adapter.load_clip(sequence, positions*C, layout_num_cameras=C,
                                     layout_num_times=T, sampling_seed=19)
            sample = prepare_clip(clip, num_queries=0, image_mode='native')
            attach_syn4d_dense_recipe(clip, sample, image_mode='native')
            batch = collate_adapter_samples([sample])
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.cuda()
            batch['targets'] = {k: v.cuda() for k, v in batch['targets'].items()}
            torch.manual_seed(49)
            sources = _sources(batch, 4, 64, prefer_motion=True)
            target = materialize_dense_targets(batch, sources)
            for name in ('visibility', 'sparse_visibility'):
                if name in target and not target[name+'_valid'].any():
                    target.pop(name); target.pop(name+'_valid')
            visibility = any(k in target for k in ('visibility', 'sparse_visibility'))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = stepper(batch['images'], batch['camera_id'], batch['time_idx'],
                    batch['query_time_idx'], sources, batch['observation_valid'], target, visibility)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite preflight loss')
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            motion_gradient = any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                  for p in model.motion_decoder.parameters())
            if not motion_gradient or not target['trajectory_foreground'].any():
                raise ValueError('preflight clip did not exercise dynamic supervision')
            row = dict(source=source['name'], sequence=sequence, cameras=C, times=T, stride=stride,
                       input_shape=list(batch['images'].shape),
                       raw_times=batch['query_time_idx'].cpu().tolist(),
                       decoded_sources=sources.cpu().tolist(), loss=float(loss.detach()),
                       metrics={k: float(v) for k, v in metrics.items()}, grad_norm=norm,
                       all_gradients_finite=True, motion_decoder_receives_gradient=motion_gradient,
                       dense_geometry_pixels=int(target['depth_valid'].sum()),
                       trajectory_pixel_times=int(target['trajectory_valid'].sum()),
                       dynamic_pixel_times=int(target['trajectory_foreground'].sum()),
                       missing_meshes=clip.metadata['syn4d_missing_meshes'],
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       elapsed_seconds=time.monotonic()-began)
            report['samples'].append(row)
            write_json(args.output, report)
            print(json.dumps(row), flush=True)
            del clip, sample, batch, target, sources, loss, metrics
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if not report['samples']:
        raise ValueError('no Syn4D samples tested')
    report['passed'] = True
    write_json(args.output, report)
    print(json.dumps(dict(passed=True, samples=len(report['samples']))), flush=True)


if __name__ == '__main__':
    main()
