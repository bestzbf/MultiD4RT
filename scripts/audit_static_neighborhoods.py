"""Check one admitted anchor in every static sequence, without model inference.

This covers sequence grouping and measured-neighbor selection across the full
pool. It does not claim an offline scan of every RGB/depth pixel in the pool.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import random
import time

from multicamera4rc.datasets.mixture import create_dataset, read_data_config
from multicamera4rc.datasets.static_views import select_static_views

DATASET=None


def initialize(config):
    global DATASET
    import torch
    torch.set_num_threads(1)
    DATASET=create_dataset(config)


def check(item):
    si,sequence=item;source=DATASET.sources[si]
    n=source['adapter'].get_num_frames(sequence)
    frames=source['frame_pool'].get(sequence,range(n))
    anchor=frames[(len(frames)-1)//2]
    views,info=select_static_views(source['adapter'],sequence,frames,anchor,
        source['camera_choices'],random.Random(f'static-neighborhood:{source["display_name"]}:{sequence}'))
    assert views[0]==anchor and len(views)==len(set(views))
    assert set(views)<=set(frames) and info['time_idx']==0
    return dict(source=source['display_name'],sequence=sequence,anchor=anchor,
        actual_cameras=len(views),available_neighbors=info['available_neighbors'],
        selection=info['selection'],view_indices=views)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-config',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();config=read_data_config(args.data_config)
    config['sources']=[s for s in config['sources'] if s.get('view_mode')=='static_multiview']
    dataset=create_dataset(config)
    tasks=[(si,seq) for si,source in enumerate(dataset.sources) for seq,_ in source['pool']]
    report=dict(data_config=args.data_config,anchors='one admitted middle frame per sequence',
        all_rgb_depth_pixels_offline_scanned=False,records=[])
    began=time.monotonic();args.output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize,initargs=(config,)) as executor:
        for record in executor.map(check,tasks,chunksize=4):
            report['records'].append(record)
            if len(report['records'])%100==0:
                print(json.dumps(dict(completed=len(report['records']),total=len(tasks),elapsed=time.monotonic()-began)),flush=True)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
    assert len(report['records'])==len(tasks) and tasks
    report['sources']={}
    for source in dataset.sources:
        name=source['display_name'];rows=[r for r in report['records'] if r['source']==name]
        report['sources'][name]=dict(sequences=len(rows),actual_camera_counts=dict(Counter(r['actual_cameras'] for r in rows)),
            isolated_anchors=sum(r['available_neighbors']==0 for r in rows),
            anchors_supporting_c4=sum(r['available_neighbors']>=3 for r in rows),
            anchors_supporting_c8=sum(r['available_neighbors']>=7 for r in rows))
    report.update(passed=True,elapsed_seconds=time.monotonic()-began)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,sources=report['sources'],elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
