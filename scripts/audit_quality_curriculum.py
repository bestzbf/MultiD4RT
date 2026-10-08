"""Verify recovery quotas, complete admitted pools, and real native pair GT."""
import argparse
import json
from pathlib import Path

import torch

from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples, read_data_config
from multicamera4rc.train import _sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--original-curriculum', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads(args.curriculum.read_text())
    original = json.loads(args.original_curriculum.read_text())
    report = dict(passed=False, stages=[], real_samples=[])
    for i, stage in enumerate(plan['stages']):
        config = read_data_config(stage['data_config'])
        before = create_dataset(original['stages'][i]['data_config'])
        after = create_dataset(config)
        assert config['image_mode'] == 'native' and config['supervision'] == 'dense'
        by_name = {s['display_name']: s for s in after.sources}
        for source in before.sources:
            groups = ([by_name['dynamic_replica_left'], by_name['dynamic_replica_right']]
                      if source['name'] == 'dynamic_replica' else [by_name[source['display_name']]])
            pools, frame_pools = {}, {}
            for group in groups:
                assert not set(pools) & dict(group['pool']).keys()
                pools.update(group['pool'])
                frame_pools.update(group['frame_pool'])
                assert source['dense_kind'] == group['dense_kind']
                assert source['view_mode'] == group['view_mode']
                assert {(v['cameras'], v['num_times']) for v in source['layouts']} <= {
                    (v['cameras'], v['num_times']) for v in group['layouts']}
            assert pools == dict(source['pool']), 'admitted sequence/frame coverage changed'
            assert frame_pools == source['frame_pool'], 'admitted frame masks changed'
        rows = [dict(name=s['display_name'], sequences=len(s['pool']), frames=sum(n for _, n in s['pool']),
                     quota=after.quotas[j], fraction=after.quotas[j]/len(after), dense_kind=s['dense_kind'])
                for j, s in enumerate(after.sources)]
        report['stages'].append(dict(start_step=stage['start_step'], epoch_size=len(after), sources=rows,
                                      all_admitted_pools_equal=True))
        if i == 0:
            for j, source in enumerate(after.sources):
                if source['name'] != 'dynamic_replica':
                    continue
                index = after.quota_ends[j-1] if j else 0
                batch = collate_adapter_samples([after[index]])
                selected = _sources(batch, 4, 64, prefer_motion=True)
                pairs = batch['targets']['dense_pair_valid'].flatten(2).any(2)
                selected_motion = pairs.gather(1, selected).sum().item()
                available_motion = pairs.sum().item()
                assert selected_motion == min(selected.numel(), available_motion)
                targets = materialize_dense_targets(batch, selected)
                assert targets['trajectory_valid'].any(), 'expected real geometry supervision'
                if source['display_name'] == 'dynamic_replica_left':
                    assert available_motion > 0 and targets['displacement_valid'].any()
                for field in ('trajectory', 'displacement'):
                    valid = targets[field+'_valid']
                    assert torch.isfinite(targets[field][valid]).all()
                report['real_samples'].append(dict(source=source['display_name'], clip_id=batch['clip_id'],
                    image_shape=list(batch['images'].shape), available_motion_sources=available_motion,
                    selected_motion_sources=selected_motion, source_indices=selected.tolist(),
                    valid_trajectory_pixels=int(targets['trajectory_valid'].sum()),
                    valid_displacement_pixels=int(targets['displacement_valid'].sum())))
                del batch, targets
        del before, after
    report['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
