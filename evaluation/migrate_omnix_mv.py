"""Rebind immutable cached predictions after native-output parity validation.

Archives from the previous equivalent memory schedule are preserved. Every
copied array is checked against the old archive, scores must be identical,
and each reused case retains the old archive/run fingerprint in diagnostics.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .common import digest,write_json
from .mvtracker.official import OfficialMVTracker
from .mvtracker.plan import (cache_for_scene,digest_json,prediction_path,read_plan,
                             read_prediction,save_prediction,score_predictions,write_report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--parity',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/data6/benchmarks'))
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Migration output must be new')
    old=json.loads((args.source/'run.json').read_text())
    parity=json.loads(args.parity.read_text())
    if old['method']!='omnix' or parity['stage']!='passed' or parity['checkpoint_sha256']!=old['checkpoint_sha256']:
        raise ValueError('Parity is not for this released native checkpoint')
    if not {(2,4),(4,8),(4,24)}.issubset({(c['cameras'],c['frames']) for c in parity['cases']}):
        raise ValueError('Real native parity must cover the complete C4T24 protocol')
    if any(max(c['maximum_absolute_errors'].values())!=0 for c in parity['cases']):
        raise ValueError('Cache reuse requires bit-identical actual native release outputs')
    module=Path(__file__).with_name('omnix_queries.py')
    snapshot=Path(__file__).parent/'references/omnix_queries_v1.py'
    if parity['adapter_sha256']!=digest(module) or old['code_sha256']['omnix_queries.py']!=digest(snapshot):
        raise ValueError('Either memory adapter changed since the recorded run')
    if any(c['adapter_provenance']!=parity['cases'][0]['adapter_provenance'] for c in parity['cases']):
        raise ValueError('Parity cases used different upstream sources')
    new=deepcopy(old)
    new['code_sha256']['omnix_queries.py']=digest(module)
    new['load_report']['query_memory_adapter']=parity['cases'][0]['adapter_provenance']
    new['load_report']['parity_report_sha256']=digest(args.parity)
    write_json(args.output/'run.json',new)
    plan=read_plan(args.plan,args.root)
    rows=[]
    for scene in plan['scenes']:
        source=prediction_path(args.source,scene)
        if not source.exists(): continue
        cache_for_scene(plan,scene)
        tracks,visible,run=read_prediction(source,plan,scene,old)
        target=prediction_path(args.output,scene)
        save_prediction(target,plan,scene,tracks,visible,new)
        restored,restored_visible,_=read_prediction(target,plan,scene,new)
        if not np.array_equal(restored,tracks) or not np.array_equal(restored_visible,visible):
            raise ValueError('Migration altered predictions')
        evidence=dict(source=str(source),source_archive_sha256=digest(source),
                      source_run_sha256=digest_json(old),destination_archive_sha256=digest(target),
                      native_parity_report_sha256=digest(args.parity),inference_reused=True)
        diagnostic=source.with_suffix('.diagnostics.json')
        info=json.loads(diagnostic.read_text()) if diagnostic.exists() else {}
        info['cache_reuse']=evidence
        write_json(target.with_suffix('.diagnostics.json'),info)
        rows.append(dict(id=scene['id'],**evidence))
    official=OfficialMVTracker(args.root/'code/mvtracker')
    before=score_predictions(plan,official,args.source,allow_partial=True)
    after=score_predictions(plan,official,args.output,allow_partial=True)
    if before['per_sequence']!=after['per_sequence'] or before['summaries']!=after['summaries']:
        raise ValueError('Cached predictions no longer give identical official scores')
    write_report(args.output/'score',after)
    write_json(args.output/'cache_migration.json',dict(stage='passed',reused_scenes=len(rows),
        original_run=old,new_run_sha256=digest_json(new),records=rows,
        note='Original immutable v1 arrays reused after bit-identical v2-vs-pristine-native parity; no inference or test label changes'))
    print(json.dumps(dict(stage='passed',reused_scenes=len(rows))),flush=True)


if __name__=='__main__': main()
