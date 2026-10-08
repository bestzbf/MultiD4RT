"""Upgrade an admitted full mixture without changing its coverage index map."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from multicamera4rc.datasets.mixture import create_dataset, read_data_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-config',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    old_config=read_data_config(args.data_config);config=deepcopy(old_config)
    layouts={'blended':[4,8],'taskonomy':[4,8],'hypersim':[2,4]}
    changed=[]
    for source in config['sources']:
        name=source['name']
        if name not in layouts: continue
        if source['num_times']!=1 or source.get('dense_kind')!='geometry':
            raise ValueError(f'{name}: expected admitted single-time geometry source')
        source.update(cameras=layouts[name],view_mode='static_multiview')
        changed.append(name)
    if set(changed)!=set(layouts): raise ValueError('expected all three static sources in the existing mixture')
    config['invalid_log']=str(args.output.resolve()/'runtime_invalid_clips.jsonl')
    output=args.output.resolve()/'full_mixture_static_multiview.yaml'
    output.write_text(yaml.safe_dump(config,sort_keys=False))
    before=create_dataset(old_config);after=create_dataset(output)
    assert len(before)==len(after) and before.coverage_size==after.coverage_size
    assert before.coverage_entries==after.coverage_entries and before.coverage_ends==after.coverage_ends
    assert before.weights==after.weights
    reports=[]
    for previous,current in zip(before.sources,after.sources):
        for key in ('display_name','pool','frame_pool','num_times','dense_kind','strides'):
            assert previous[key]==current[key],f'{current["display_name"]}: changed {key}'
        reports.append(dict(name=current['display_name'],sequences=len(current['pool']),
            frames=sum(n for _,n in current['pool']),view_mode=current['view_mode'],
            cameras=list(current['camera_choices']),num_times=current['num_times'],
            pool_and_index_mapping_unchanged=True))
    digest=hashlib.sha256(json.dumps([after.coverage_entries,after.coverage_ends,after.weights],
        separators=(',',':')).encode()).hexdigest()
    report=dict(passed=True,previous_data_config=str(Path(args.data_config).resolve()),data_config=str(output),
        coverage_windows=after.coverage_size,epoch_size=len(after),coverage_index_sha256=digest,
        sampler_cursor_compatible=True,source_order_weights_and_pools_unchanged=True,sources=reports)
    (args.output/'static_data_plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
