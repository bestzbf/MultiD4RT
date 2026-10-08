"""Evaluate a prepared native dynamic-GT plan without subsampling its particles.

The data config must already select held-out sequences. The plan records each
source, sequence and deterministic coverage index; it is never rewritten here.
"""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path

import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.evaluate import fixed_sources
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.metrics import METRICS_VERSION
from multicamera4rc.train import _TrainingStep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True, help='LABEL=PATH')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--memory-fraction', type=float, default=.10)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error('memory-fraction must lie in (0,1]')
    torch.set_num_threads(2)
    plan = json.loads(args.plan.read_text())
    if not plan['cases']:
        raise ValueError('native evaluation requires a nonempty fixed plan')
    dataset = create_dataset(args.data_config)
    if dataset.sampling != 'coverage':
        raise ValueError('native fixed-plan indices require sampling: coverage')
    device = torch.device(args.device)
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, index)
    report = dict(stage='running', metrics_version=METRICS_VERSION,
                  plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                  data_config_sha256=hashlib.sha256(Path(args.data_config).read_bytes()).hexdigest(),
                  splits=plan.get('splits', {}), checkpoints={})
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in args.checkpoints:
        label, path = entry.split('=', 1)
        payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        cfg = replace(config_from_checkpoint(payload), gradient_checkpointing=False,
                      attention_chunk_size=None, visibility_chunk_size=None)
        model = MultiCamera4RC(cfg)
        loading = model.load_4rc_weights(payload)
        profile = payload.get('training_state', {}).get('loss_config')
        criterion = MultiCamera4RCLoss(**profile) if profile else MultiCamera4RCLoss.dense()
        checkpoint_step = payload['step']
        del payload
        step = _TrainingStep(model.to(device).eval(), criterion)
        records = []
        for case in plan['cases']:
            source = dataset.sources[case['source']]
            if source['display_name'] != case['source_name'] or case['sequence'] not in dict(source['pool']):
                raise ValueError('native plan does not match its source/sequence pool')
            sample = dataset[case['index']]
            if not sample['clip_id'].endswith('/'+case['sequence']):
                raise ValueError('native plan coverage index now resolves to another sequence')
            batch = collate_adapter_samples([sample])
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch['targets'] = {k: v.to(device) for k, v in batch['targets'].items()}
            selected = fixed_sources(batch, 4)
            with torch.inference_mode():
                target = materialize_dense_targets(batch, selected)
                for key in ('visibility', 'sparse_visibility'):
                    if key in target and not target[key+'_valid'].any():
                        target.pop(key)
                        target.pop(key+'_valid')
                visibility = any(key in target for key in ('visibility', 'sparse_visibility'))
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    loss, metrics = step(batch['images'], batch['camera_id'], batch['time_idx'],
                                         batch['query_time_idx'], selected, batch['observation_valid'], target, visibility)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite native validation loss')
            record = dict(case, input_shape=list(batch['images'].shape),
                          camera_ids=batch['camera_id'].tolist(), time_indices=batch['time_idx'].tolist(),
                          source_indices=selected.tolist(), metrics={k: float(v) for k, v in metrics.items()})
            records.append(record)
            print(json.dumps(dict(checkpoint=label, **record), allow_nan=False), flush=True)
            del sample, batch, target, loss, metrics
        grouped = {}
        for record in records:
            grouped.setdefault(record['source_name'], []).append(record['metrics'])
        means = {name: {k: sum(r[k] for r in rows if k in r)/sum(k in r for r in rows)
                        for k in set().union(*(r.keys() for r in rows))} for name, rows in grouped.items()}
        report['checkpoints'][label] = dict(path=path, step=checkpoint_step, clips=len(records), loading=loading,
            model_config=cfg.to_dict(), loss_config=criterion.to_config(),
            amp_dtype='bfloat16' if device.type == 'cuda' else 'float32', by_source=means, records=records)
        (args.output/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        del step, model, criterion
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report['stage'] = 'passed'
    (args.output/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
