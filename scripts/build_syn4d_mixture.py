"""Create a new mixed-data config with Syn4D; never modify the running config."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from multicamera4rc.datasets.mixture import create_dataset, read_data_config


def syn4d_sources(root, *, split='train', metadata_root=None,
                  v1_subset='syn4d_v1_stride_5', v1_strides=(1,)):
    if v1_subset not in ('syn4d_v1_stride_1', 'syn4d_v1_stride_5'):
        raise ValueError('choose exactly one V1 release to avoid duplicate physical sequences')
    if not v1_strides or any(type(v) is not int or v <= 0 for v in v1_strides):
        raise ValueError('V1 sampling strides must be positive integers')
    common = dict(adapter='syn4d', root=str(Path(root).resolve()), split=split,
                  strides=[1], dense_kind='syn4d')
    sources = []
    for subset, name, layouts in (
        (v1_subset, 'syn4d_v1', [(1,16,.2), (2,8,.4), (4,4,.3), (8,2,.1)]),
        ('syn4d_sim', 'syn4d_sim', [(1,16,.2), (2,8,.45), (4,4,.35)]),
    ):
        options = dict(subsets=[subset], frame_policy='reference', missing_mesh='mask')
        if metadata_root:
            options['metadata_root'] = str(Path(metadata_root).resolve())
        sources.append(dict(**common, name=name, weight=1., options=options,
                            layouts=[dict(cameras=c, num_times=t, weight=w) for c,t,w in layouts]))
        if name == 'syn4d_v1':
            sources[-1]['strides'] = list(dict.fromkeys(v1_strides))
    return sources


def integrate(base, root, *, fraction=.1, v1_fraction=.5, metadata_root=None,
              v1_subset='syn4d_v1_stride_5', v1_strides=(1,)):
    if not 0 < fraction < 1 or not 0 < v1_fraction < 1:
        raise ValueError('Syn4D fraction and V1 fraction must be strictly between zero and one')
    result = deepcopy(base)
    if result.get('supervision') != 'dense' or result.get('sampling') != 'balanced_coverage':
        raise ValueError('Syn4D integration expects a dense balanced_coverage base config')
    if any(s['adapter'] == 'syn4d' for s in result['sources']):
        raise ValueError('base config already includes Syn4D; do not double-count sources')
    total = sum(float(s.get('weight', 1.)) for s in result['sources'])
    if total <= 0:
        raise ValueError('base source weights must be positive')
    for source in result['sources']:
        source['weight'] = float(source.get('weight', 1.)) / total * (1-fraction)
    new = syn4d_sources(root, metadata_root=metadata_root,
                       v1_subset=v1_subset, v1_strides=v1_strides)
    new[0]['weight'] = fraction * v1_fraction
    new[1]['weight'] = fraction * (1-v1_fraction)
    result['sources'].extend(new)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--metadata-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fraction', type=float, default=.1, help='Total Syn4D sample probability; default .10')
    parser.add_argument('--v1-fraction', type=float, default=.5, help='V1 share within Syn4D; default .50')
    parser.add_argument('--v1-subset', choices=['syn4d_v1_stride_1', 'syn4d_v1_stride_5'],
                        default='syn4d_v1_stride_5')
    parser.add_argument('--v1-strides', nargs='+', type=int, default=[1],
                        help='Strides in the selected release reference-frame list')
    parser.add_argument('--check', action='store_true', help='Index all sources and verify layouts before saving')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output == Path(args.base).resolve() or output.exists():
        raise ValueError('output must be a new file; existing training configs are preserved')
    config = integrate(read_data_config(args.base), args.root, fraction=args.fraction,
                       v1_fraction=args.v1_fraction, metadata_root=args.metadata_root,
                       v1_subset=args.v1_subset, v1_strides=args.v1_strides)
    if config.get('skip_invalid'):
        config['invalid_log'] = str(output.parent / 'invalid_clips.jsonl')
    if args.check:
        dataset = create_dataset(config)
        print(json.dumps(dict(sources=[dict(name=s['display_name'], sequences=len(s['pool']),
                                           weight=dataset.weights[i], layouts=s['layouts'])
                                       for i,s in enumerate(dataset.sources)])), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == '.json':
        output.write_text(json.dumps(config, indent=2)+'\n')
    else:
        import yaml
        output.write_text('# Prepared integration config; not automatically activated.\n'
                          + yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(dict(output=str(output), syn4d_fraction=args.fraction)), flush=True)


if __name__ == '__main__':
    main()
