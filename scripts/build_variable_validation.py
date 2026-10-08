"""Create a disjoint fixed validation plan for new camera/time layouts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml

from multicamera4rc.datasets.mixture import create_dataset, read_data_config
from multicamera4rc.evaluate import make_validation_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curriculum",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads(args.curriculum.read_text());data=read_data_config(plan["stages"][2]["data_config"])
    template=next(s for s in data["sources"] if s["adapter"] == "kubric_multicam")
    admitted=json.loads(Path(template["manifest"]).read_text())["sequences"]
    layouts=[(1,8),(1,16),(1,24),(1,32),(2,16),(2,24),(4,12),
             (5,4),(6,4),(7,4),(8,6),(9,4),(10,4)]
    config=dict(image_mode="native",supervision="dense",sampling="weighted",num_times=8,
                num_queries=0,epoch_size=100,seed=49,include_visibility=True,sources=[])
    for C,T in layouts:
        s=deepcopy(template)
        s.pop("manifest");s.pop("layouts")
        s.update(name=f"kubric_C{C}T{T}",cameras=C,num_times=T,strides=[1],weight=1.)
        config["sources"].append(s)
    train=create_dataset(config)
    for s in train.sources:
        if not {n for n,_ in s["pool"]} <= set(admitted):raise ValueError("validation setup introduced unadmitted train scenes")
    held=deepcopy(config)
    for s in held["sources"]:s["split"]="val"
    val=create_dataset(held)
    train_ids={n for s in train.sources for n,_ in s["pool"]};val_ids={n for s in val.sources for n,_ in s["pool"]}
    assert not train_ids & val_ids
    fixed=make_validation_plan(val,seed=9263,scenes_per_variant=1)
    (args.output/"data.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    (args.output/"fixed_plan.json").write_text(json.dumps(fixed,indent=2)+"\n")
    result=dict(passed=True,sequence_overlap=0,train_scenes=len(train_ids),val_scenes=len(val_ids),
                layouts=layouts,cases=len(fixed["cases"]),
                plan_sha256=hashlib.sha256(json.dumps(fixed,sort_keys=True).encode()).hexdigest())
    (args.output/"proof.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":main()
