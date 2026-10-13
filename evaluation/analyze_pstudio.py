"""Source geometry vs later track errors; does not alter benchmark scoring."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .common import load_module,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args()
    directory=args.campaign/'tapvid16_pstudio_full_eval'
    plan=json.loads((directory/'plan.json').read_text())
    scorer=load_module('_tapvid3d_diagnostics',Path(__file__).parent/'references/tapvid3d_metrics.py')
    cases=defaultdict(list)
    for case in plan['cases']: cases[case['sequence']].append(case)
    reports={m:json.loads((directory/m/'results.json').read_text()) for m in ['multid4rt','omnix']}
    output={}
    for method,report in reports.items():
        records={r['id']:r for r in report['records']}
        rows=[]
        for sequence,clips in cases.items():
            with np.load(clips[0]['path'],allow_pickle=True) as d:
                tracks=d['tracks_XYZ'].astype(np.float32)
                visibility=d['visibility'].astype(bool)
                k=d['fx_fy_cx_cy'].astype(np.float64)
            for case in clips:
                row=records[case['id']]
                matches=list((directory/method).glob(f'worker*/{case["id"]}.npz'))
                if len(matches)!=1: raise ValueError('Missing or duplicate cached inference')
                with np.load(matches[0],allow_pickle=False) as d:
                    predicted=d['tracks_XYZ'].astype(np.float64)
                    points=d['point_ids']
                start=case['start'];truth=tracks[start:start+16,points];visible=visibility[start:start+16,points]
                scaled=predicted*row['metrics']['global_scale']
                epe=np.linalg.norm(scaled-truth,axis=-1)
                displacement_error=np.linalg.norm((scaled-scaled[:1])-(truth-truth[:1]),axis=-1)
                source_apd=scorer.compute_tapvid3d_metrics(~visible[:1],truth[:1],
                    np.zeros_like(visible[:1]),scaled[:1],k,order='t n',scaling='none')['average_pts_within_thresh']
                rows.append(dict(id=case['id'],sequence=sequence,source_apd3d=float(np.mean(source_apd)),
                    source_epe_m=float(epe[0].mean()),future_epe_m=float(epe[1:][visible[1:]].mean()),
                    future_displacement_error_m=float(displacement_error[1:][visible[1:]].mean())))
        keys=['source_apd3d','source_epe_m','future_epe_m','future_displacement_error_m']
        output[method]=dict(mean={k:float(np.mean([r[k] for r in rows])) for k in keys},records=rows)
    write_json(directory/'error_diagnostics.json',dict(stage='passed',methods=output,
        note='Diagnostic only: same cached predictions and unchanged official clip-wide scale; displacement includes camera-pose error; no GT anchoring or retraining'))


if __name__=='__main__': main()
