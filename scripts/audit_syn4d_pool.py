"""Check every admitted Syn4D sequence's metadata and one native clip per scene."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from multicamera4rc.datasets.adapters.syn4d import Syn4DAdapter
from multicamera4rc.datasets.mixture import read_data_config
from scripts.audit_syn4d import audit_case, official_module
from scripts.continue_quality_recovery import digest
from scripts.cutover_native_motion_stereo import write_json


def reviewed_rejections(source, root, rejected, manifest, *, collection='exclusions'):
    """Permit only exact, documented exclusions; never admit their missing GT."""
    entries = manifest.get(collection, [])
    if manifest.get('version') != 1 or not isinstance(entries, list):
        raise ValueError('invalid reviewed-exclusions manifest')
    expected = {}
    for item in entries:
        if item.get('source') != source:
            continue
        path = Path(item['path'])
        if path.is_absolute() or '..' in path.parts or not item.get('review', '').strip():
            raise ValueError('exclusions need root-relative paths and a review')
        key = (path.as_posix(), item['reason'])
        if key in expected:
            raise ValueError('duplicate reviewed exclusion')
        expected[key] = item
    root = Path(root).resolve()
    actual = {(Path(item['path']).resolve().relative_to(root).as_posix(), item['reason'])
              for item in rejected}
    if actual != set(expected):
        raise ValueError(f'unreviewed or changed Syn4D exclusions for {source}: '
                         f'new={sorted(actual-set(expected))}, stale={sorted(set(expected)-actual)}')
    return [expected[key] for key in sorted(actual)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--official-loader', required=True)
    parser.add_argument('--reviewed-exclusions', type=Path,
                        help='Exact documented exclusions, matched by source/path/reason; strict by default')
    args = parser.parse_args()
    torch.set_num_threads(4)
    oracle = official_module(args.official_loader)
    config = read_data_config(args.data_config)
    manifest = (json.loads(args.reviewed_exclusions.read_text()) if args.reviewed_exclusions
                else dict(version=1, exclusions=[]))
    sources = {s['name'] for s in config['sources'] if s['adapter'] == 'syn4d'}
    if any(item['source'] not in sources for key in ('exclusions', 'missing_objects')
           for item in manifest.get(key, [])):
        raise ValueError('reviewed exclusions refer to a source outside this audit')
    report = dict(passed=False, config=args.data_config, config_sha256=digest(args.data_config),
                  official_loader_sha256=digest(args.official_loader), sources=[], cases=[])
    if args.reviewed_exclusions:
        report['reviewed_exclusions'] = dict(path=str(args.reviewed_exclusions),
                                            sha256=digest(args.reviewed_exclusions), manifest=manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for source in config['sources']:
        if source['adapter'] != 'syn4d':
            continue
        adapter = Syn4DAdapter(source['root'], split='all', **source.get('options', {}))
        excluded = reviewed_rejections(source['name'], source['root'], adapter.rejected, manifest)
        scenes, splits, camera_counts, missing_masks = {}, Counter(), Counter(), Counter()
        missing_objects = {}
        for index, sequence in enumerate(adapter.sequences):
            record = adapter.index[sequence]
            info = adapter.get_sequence_info(sequence)
            splits[info['split']] += 1
            camera_counts[info['num_cameras']] += 1
            scenes.setdefault(record['scene'].name, []).append(sequence)
            # All cameras must describe the same actors and world transformations.
            frames = [info['frame_indices'][0], info['frame_indices'][-1]]
            meshes, missing = adapter._geometry(record, sorted(record['cameras']), frames)
            for _, (vertices, faces) in meshes.items():
                if not np.isfinite(vertices).all() or not len(faces):
                    raise ValueError('invalid animated geometry in '+sequence)
            for item in missing:
                missing_masks[item['mask']] += 1
                if item['mask'] not in ('_00_body', '_00_clothing'):
                    entry = missing_objects.setdefault(item['path'], dict(path=item['path'],
                        reason='object mesh is unavailable; motion masked', occurrences=[]))
                    entry['occurrences'].append(dict(sequence=sequence, mask=item['mask']))
            if (index+1) % 50 == 0:
                print(json.dumps(dict(source=source['name'], metadata_checked=index+1,
                                      metadata_total=len(adapter))), flush=True)
        reviewed_objects = reviewed_rejections(source['name'], source['root'],
            list(missing_objects.values()), manifest, collection='missing_objects')
        requested = source.get('options', {}).get('scenes')
        if not scenes or (requested is not None and set(requested) != set(scenes)):
            raise ValueError('not every requested Syn4D scene is indexable')
        row = dict(name=source['name'], sequences=len(adapter), split_counts=dict(splits),
                   camera_count_histogram=dict(camera_counts), missing_mesh_masks=dict(missing_masks),
                   scenes={k: len(v) for k,v in scenes.items()}, rejected=adapter.rejected,
                   reviewed_exclusions=excluded, reviewed_missing_objects=reviewed_objects,
                   missing_object_occurrences=list(missing_objects.values()))
        report['sources'].append(row)
        for scene, sequences in scenes.items():
            # Every scene gets an independent released-loader comparison, including
            # nonzero reference time and, where available, separated raw frames.
            sequence = next((s for s in sequences if adapter.get_sequence_info(s)['split'] == 'train'), sequences[0])
            info = adapter.get_sequence_info(sequence)
            stride = min(5, max(1, (info['num_frames']-1)//2))
            start = min(7, info['num_frames']-1-2*stride)
            result = audit_case(adapter, sequence, min(2, info['num_cameras']), 3, start, oracle, stride=stride)
            if result['native_hw'] not in ([720,1280], [1024,1024]):
                raise ValueError('new resolution needs its own giant-memory proof: '+str(result['native_hw']))
            report['cases'].append(result)
            write_json(args.output, report)
            print(json.dumps(dict(scene=scene, **result)), flush=True)
    if not report['sources'] or not report['cases']:
        raise ValueError('no Syn4D data audited')
    report['passed'] = True
    write_json(args.output, report)
    print(json.dumps(dict(passed=True, sources=report['sources'], scenes=len(report['cases']))), flush=True)


if __name__ == '__main__':
    main()
