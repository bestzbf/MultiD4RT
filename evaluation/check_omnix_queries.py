"""Actual checkpoint parity for query-only memory use on real test RGB."""
import argparse
from pathlib import Path

import torch

from .common import digest, load_model, write_json
from .multiview_adapter import TrackingInputs, resize_inputs
from .mvtracker.official import OfficialMVTracker
from .mvtracker.protocols import PROTOCOLS
from .omnix_queries import parity_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--omnix-runner',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/data6/benchmarks'))
    p.add_argument('--device',default='cuda:1')
    args=p.parse_args()
    torch.set_num_threads(4)
    device=torch.device(args.device)
    model,report=load_model('omnix',args.checkpoint,device,args.omnix_runner)
    official=OfficialMVTracker(args.root/'code/mvtracker')
    protocol=PROTOCOLS['dexycb']
    datapoint,valid=official.dataset(protocol,args.root/'datasets/mvtracker')[0]
    if not valid: raise ValueError('Official loader rejected parity scene')
    inputs=TrackingInputs.from_official(datapoint,protocol)
    images,_,preprocessing=resize_inputs(inputs,266,14,device)
    video=images.reshape(1,4,-1,*images.shape[-3:])
    results=[]
    for c,t in [(2,4),(4,8),(4,24)]:
        result=parity_case(model,video,c,t)
        results.append(result)
        print(result,flush=True)
    write_json(args.output,dict(stage='passed',checkpoint_sha256=report['checkpoint_sha256'],
        adapter_sha256=digest(Path(__file__).with_name('omnix_queries.py')),
        actual_inputs=preprocessing,cases=results))


if __name__=='__main__': main()
