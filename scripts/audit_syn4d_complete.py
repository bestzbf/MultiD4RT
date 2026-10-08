"""Audit complete human-enabled Syn4D scene pools in independent CPU workers."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from multicamera4rc.datasets.adapters.syn4d import Syn4DAdapter
from multicamera4rc.datasets.mixture import read_data_config
from scripts.audit_syn4d import audit_case, official_module
from scripts.audit_syn4d_pool import reviewed_rejections
from scripts.continue_quality_recovery import digest
from scripts.cutover_native_motion_stereo import write_json


def audit_scene(source, scene, official_loader, extra_sequences, threads):
    torch.set_num_threads(threads)
    began = time.monotonic()
    options = dict(source.get('options', {}), scenes=[scene])
    adapter = Syn4DAdapter(source['root'], split='all', **options)
    if not len(adapter):
        raise ValueError('empty scene: '+scene)
    splits, cameras, missing_masks = Counter(), Counter(), Counter()
    missing_objects = {}
    for sequence in adapter.sequences:
        record, info = adapter.index[sequence], adapter.get_sequence_info(sequence)
        splits[info['split']] += 1
        cameras[info['num_cameras']] += 1
        meshes, missing = adapter._geometry(record, sorted(record['cameras']),
            [info['frame_indices'][0], info['frame_indices'][-1]])
        for vertices, faces in meshes.values():
            if not np.isfinite(vertices).all() or not len(faces):
                raise ValueError('invalid animated geometry: '+sequence)
        for item in missing:
            missing_masks[item['mask']] += 1
            if item['mask'] in ('_00_body', '_00_clothing'):
                raise FileNotFoundError('complete pool requires both human meshes: '+str(item))
            entry = missing_objects.setdefault(item['path'], dict(path=item['path'],
                reason='object mesh is unavailable; motion masked', occurrences=[]))
            entry['occurrences'].append(dict(sequence=sequence, mask=item['mask']))
    primary = next((s for s in adapter.sequences if adapter.get_sequence_info(s)['split'] == 'train'),
                   adapter.sequences[0])
    selected = [primary]+[s for s in extra_sequences if s in adapter.index and s != primary]
    oracle = official_module(official_loader)
    cases = []
    for sequence in selected:
        info = adapter.get_sequence_info(sequence)
        stride = min(5, max(1, (info['num_frames']-1)//2))
        start = min(7, info['num_frames']-1-2*stride)
        result = audit_case(adapter, sequence, min(2, info['num_cameras']), 3, start, oracle, stride=stride)
        if result['native_hw'] not in ([720, 1280], [1024, 1024]):
            raise ValueError('unverified native resolution: '+str(result['native_hw']))
        cases.append(result)
    return dict(source=source['name'], scene=scene, sequences=len(adapter),
        split_counts=dict(splits), camera_count_histogram=dict(cameras),
        missing_mesh_masks=dict(missing_masks), missing_object_occurrences=list(missing_objects.values()),
        rejected=adapter.rejected, cases=cases, seconds=time.monotonic()-began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--official-loader', required=True)
    parser.add_argument('--reviewed-exclusions', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.workers, args.cpu_threads) < 1:
        raise ValueError('workers/threads must be positive')
    config = read_data_config(args.data_config)
    manifest = json.loads(args.reviewed_exclusions.read_text())
    sources = [s for s in config['sources'] if s['adapter'] == 'syn4d']
    if not sources or any(not s.get('options', {}).get('scenes') for s in sources):
        raise ValueError('audit requires explicit Syn4D scene lists')
    if any(x['source'] not in {s['name'] for s in sources}
           for key in ('exclusions', 'missing_objects') for x in manifest.get(key, [])):
        raise ValueError('review manifest names an unconfigured source')
    extra = sorted({seq for item in manifest.get('missing_objects', [])
                    for seq in item.get('affected_sequences', [])}
                   | set(manifest.get('extra_sequences', [])))
    report = dict(passed=False, stage='auditing_scenes', config=args.data_config,
        config_sha256=digest(args.data_config), official_loader_sha256=digest(args.official_loader),
        audit_code_sha256=digest(__file__), reviewed_exclusions=manifest,
        reviewed_exclusions_sha256=digest(args.reviewed_exclusions), sources=[], cases=[], errors=[],
        total_scenes=sum(len(s['options']['scenes']) for s in sources), completed_scenes=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene_dir = args.output.parent/'scene_audits'; scene_dir.mkdir(exist_ok=True)
    write_json(args.output, report)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(audit_scene, source, scene, args.official_loader, extra, args.cpu_threads):
                   (source['name'], scene) for source in sources for scene in source['options']['scenes']}
        for future in as_completed(futures):
            name, scene = futures[future]
            try:
                row = future.result()
                rows.append(row)
                report['cases'].extend(row['cases'])
                write_json(scene_dir/(name+'_'+scene+'.json'), dict(passed=True, **row))
            except Exception:
                failure = dict(source=name, scene=scene, traceback=traceback.format_exc())
                report['errors'].append(failure)
                write_json(scene_dir/(name+'_'+scene+'.json'), dict(passed=False, **failure))
            report['completed_scenes'] += 1
            write_json(args.output, report)
            print(json.dumps(dict(source=name, scene=scene, completed=report['completed_scenes'],
                                  total=report['total_scenes'], errors=len(report['errors']))), flush=True)
    if report['errors']:
        raise RuntimeError('one or more scene audits failed; see '+str(args.output))
    if not set(extra) <= {case['sequence'] for case in report['cases']}:
        raise ValueError('an explicitly requested extra sequence was not audited')
    for source in sources:
        selected = [r for r in rows if r['source'] == source['name']]
        if {r['scene'] for r in selected} != set(source['options']['scenes']):
            raise ValueError('scene audit coverage differs')
        splits, cameras, masks = Counter(), Counter(), Counter()
        rejected, missing_objects = [], {}
        for row in selected:
            splits.update(row['split_counts']); cameras.update(row['camera_count_histogram'])
            masks.update(row['missing_mesh_masks']); rejected.extend(row['rejected'])
            for item in row['missing_object_occurrences']:
                entry = missing_objects.setdefault(item['path'], dict(item, occurrences=[]))
                entry['occurrences'].extend(item['occurrences'])
        excluded = reviewed_rejections(source['name'], source['root'], rejected, manifest)
        missing = reviewed_rejections(source['name'], source['root'], list(missing_objects.values()),
                                      manifest, collection='missing_objects')
        report['sources'].append(dict(name=source['name'], sequences=sum(r['sequences'] for r in selected),
            split_counts=dict(splits), camera_count_histogram=dict(cameras), missing_mesh_masks=dict(masks),
            scenes={r['scene']:r['sequences'] for r in selected}, rejected=rejected,
            reviewed_exclusions=excluded, reviewed_missing_objects=missing,
            missing_object_occurrences=list(missing_objects.values())))
    human = Counter()
    for case in report['cases']:
        human.update(case['human_motion_pixel_times'])
    if any(human[name] <= 0 for name in ('_00_body', '_00_clothing')):
        raise ValueError('audit did not exercise both human motion labels')
    report.update(passed=True, stage='complete', human_motion_pixel_times=dict(human))
    write_json(args.output, report)
    print(json.dumps(dict(passed=True, cases=len(report['cases']), sources=report['sources'])), flush=True)


if __name__ == '__main__':
    main()
