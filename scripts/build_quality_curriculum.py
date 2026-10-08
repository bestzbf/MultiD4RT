"""Prepare an explicit quality-recovery curriculum without editing live plans."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import yaml

from multicamera4rc.datasets.mixture import read_data_config


ANCHOR_LAYOUTS = ((2, 12), (3, 8), (4, 6))


def replay_layouts(layouts, fraction=.5):
    if not 0 <= fraction < 1 or not math.isfinite(fraction):
        raise ValueError('replay fraction must lie in [0,1), retaining variable layouts')
    total = sum(v.get('weight', 1.) for v in layouts)
    if total <= 0:
        raise ValueError('layout weights must have positive total')
    weights = {}
    for row in layouts:
        key = (row['cameras'], row['num_times'])
        weights[key] = weights.get(key, 0.)+(1-fraction)*row.get('weight', 1.)/total
    if fraction:
        for key in ANCHOR_LAYOUTS:
            weights[key] = weights.get(key, 0.)+fraction/len(ANCHOR_LAYOUTS)
    return [dict(cameras=c, num_times=t, weight=w) for (c, t), w in sorted(weights.items()) if w > 0]


def split_replica_manifest(manifest):
    """Partition camera streams without dropping labels, frames or sequences."""
    result = {camera: deepcopy(manifest) for camera in ('left', 'right')}
    for key in ('sequences', 'validation_sequences'):
        records = manifest.get(key, {})
        groups = {camera: {name: value for name, value in records.items() if name.endswith('_'+camera)}
                  for camera in result}
        if set(groups['left']) & set(groups['right']) or set().union(*map(set, groups.values())) != set(records):
            raise ValueError('Replica manifest contains an unrecognized camera stream')
        for camera in result:
            result[camera][key] = groups[camera]
    if any(not value['sequences'] for value in result.values()):
        raise ValueError('Replica quality split requires both admitted camera pools')
    return result


def make_quality_config(original, directory, *, replay_fraction=.5, replica_left_fraction=.75):
    if not 0 < replica_left_fraction < 1 or not math.isfinite(replica_left_fraction):
        raise ValueError('both Replica camera pools must retain positive training weight')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = deepcopy(original)
    sources, proof = [], dict(replay_fraction=replay_fraction, replica_left_fraction=replica_left_fraction)
    for source in config['sources']:
        if source['adapter'] == 'kubric_multicam':
            source['layouts'] = replay_layouts(source['layouts'], replay_fraction)
            source['num_times'] = max(row['num_times'] for row in source['layouts'])
            sources.append(source)
        elif source['adapter'] == 'dynamic_replica':
            manifest = json.loads(Path(source['manifest']).read_text())
            parts = split_replica_manifest(manifest)
            proof['replica_sequences'] = {camera: len(value['sequences']) for camera, value in parts.items()}
            for camera, fraction in (('left', replica_left_fraction), ('right', 1-replica_left_fraction)):
                path = directory/f'dynamic_replica_{camera}_manifest.json'
                path.write_text(json.dumps(parts[camera], separators=(',', ':'))+'\n')
                row = deepcopy(source)
                row.update(name=f'dynamic_replica_{camera}', manifest=str(path.resolve()), weight=source['weight']*fraction)
                # Keep replica_flow for both: valid right-camera flow, when
                # available, remains supervised under its official masks.
                sources.append(row)
        else:
            sources.append(source)
    config.update(sources=sources, invalid_log=str((directory/'invalid_clips.jsonl').resolve()))
    old_total = sum(s['weight'] for s in original['sources'])
    if not math.isclose(sum(s['weight'] for s in sources), old_total):
        raise AssertionError('quality migration changed the total source weight')
    proof.update(source_weight_total=old_total, all_admitted_sequences_retained=True,
                 native_resolution=config.get('image_mode') == 'native')
    return config, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-step', type=int, default=32000)
    parser.add_argument('--replay-fraction', type=float, default=.5)
    parser.add_argument('--replica-left-fraction', type=float, default=.75)
    args = parser.parse_args()
    if not 25000 <= args.start_step < 34000:
        parser.error('this migration starts within the audited 25k-34k interval')
    original = json.loads(args.curriculum.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(schema='quality_curriculum_v1', target_step=50000, stages=[],
                  decoder_memory='target_time', architecture_migration='separate validated experiment',
                  family_percent=original['family_percent'])
    for old_stage, begin, end in ((0, args.start_step, 34000), (1, 34000, 40000), (2, 40000, 50000)):
        directory = args.output/f'stage_{begin}_{end}'
        config, proof = make_quality_config(read_data_config(original['stages'][old_stage]['data_config']), directory,
            replay_fraction=args.replay_fraction, replica_left_fraction=args.replica_left_fraction)
        path = directory/'data.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        result['stages'].append(dict(start_step=begin, end_step=end, data_config=str(path.resolve()), **proof))
    (args.output/'curriculum.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
