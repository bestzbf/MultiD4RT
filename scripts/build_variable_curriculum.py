"""Retain admitted native GT pools and prepare balanced C/T training to 50k."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import yaml

from multicamera4rc.datasets.mixture import read_data_config


FAMILY_PERCENT = dict(kubric=35, dynamic_replica=12, pointodyssey=12, vkitti2=11,
    tartanair_v1=4, tartanair_v2=7, hypersim=4, scannet=3, scannetpp=3,
    wildrgbd=3, blended=4, taskonomy=2)
CAMERA_WEIGHTS = {1:.15, 2:.17, 3:.14, 4:.18, 5:.06, 6:.07, 7:.06, 8:.07, 9:.05, 10:.05}
NATIVE_MOTION_TIME_CAPS = {"dynamic_replica":16, "pointodyssey":24, "vkitti2":24}
MONO_TIMES = [
    {4:.10, 8:.45, 12:.30, 16:.15},
    {8:.25, 12:.25, 16:.30, 24:.20},
    {8:.10, 12:.15, 16:.25, 24:.30, 32:.20},
]
STEREO_TIMES = [
    {4:.35, 6:.50, 8:.15},
    {4:.20, 6:.35, 8:.30, 12:.15},
    {4:.15, 6:.20, 8:.35, 12:.30},
]


def family(name):
    if name.startswith("kubric"): return "kubric"
    if name.startswith("vkitti2"): return "vkitti2"
    if name.startswith("tartanair_v2"): return "tartanair_v2"
    if name.startswith("tartanair"): return "tartanair_v1"
    return name


def kubric_layouts(stage):
    if stage == 0:
        choices = {1:MONO_TIMES[0], 2:{4:.2,8:.4,12:.4}, 3:{4:.25,6:.35,8:.4},
            4:{4:.4,6:.6}, 5:{2:.2,4:.8}, 6:{2:.2,4:.8}}
        choices.update({c:{2:.4,4:.6} for c in range(7,11)})
    elif stage == 1:
        choices = {1:MONO_TIMES[1], 2:{6:.2,8:.3,12:.3,16:.2}, 3:{4:.2,8:.5,12:.3},
            4:{4:.15,6:.35,8:.35,12:.15}, 5:{2:.15,4:.55,6:.3}, 6:{2:.15,4:.55,6:.3},
            7:{2:.1,4:.7,6:.2}, 8:{2:.1,4:.7,6:.2}, 9:{2:.2,4:.8}, 10:{2:.2,4:.8}}
    else:
        choices = {1:MONO_TIMES[2], 2:{6:.1,8:.2,12:.3,16:.25,24:.15},
            3:{4:.1,8:.3,12:.35,16:.25}, 4:{4:.1,6:.25,8:.35,12:.30},
            5:{2:.1,4:.45,6:.45}, 6:{2:.1,4:.45,6:.45}, 7:{2:.1,4:.55,6:.35},
            8:{2:.1,4:.55,6:.35}, 9:{2:.15,4:.85}, 10:{2:.15,4:.85}}
    return [dict(cameras=c, num_times=t, weight=CAMERA_WEIGHTS[c]*p)
            for c, times in choices.items() for t, p in times.items()]


def make_config(original, stage, invalid_log):
    config = deepcopy(original)
    config.update(sampling="balanced_coverage", epoch_size=100000, invalid_log=str(invalid_log))
    sources = []
    for s in config["sources"]:
        name = s["name"]; group = family(name)
        if group == "kubric":
            if any(family(v["name"]) == "kubric" for v in sources): continue
            s.update(name="kubric_variable", cameras=10, num_times=max(MONO_TIMES[stage]),
                     layouts=kubric_layouts(stage), strides=[1,2,4], weight=35.)
            # All existing Kubric manifests describe the same admitted scenes.
            # C/T eligibility is checked against native metadata per sequence.
        else:
            share = 1.
            if group in ("vkitti2", "tartanair_v1"):
                share = .6 if s["cameras"] == 2 else .2
            elif group == "tartanair_v2":
                share = .1 if s["cameras"] == 1 else .15
            s["weight"] = FAMILY_PERCENT[group]*share
            if s.get("view_mode", "temporal") != "static_multiview":
                choices = MONO_TIMES[stage] if s["cameras"] == 1 else STEREO_TIMES[stage]
                if s["cameras"] == 1 and group in NATIVE_MOTION_TIME_CAPS:
                    maximum=NATIVE_MOTION_TIME_CAPS[group]
                    capped={}
                    for t,p in choices.items():
                        actual=min(t,maximum);capped[actual]=capped.get(actual,0.)+p
                    choices=capped
                s.update(num_times=max(choices), layouts=[dict(cameras=s["cameras"], num_times=t, weight=p)
                                                        for t,p in choices.items()])
                # Adjacent renderer flow is never fabricated across gaps.
                if s["dense_kind"] in ("replica_flow", "vkitti_scene_flow"):
                    s["strides"] = [1]
                else:
                    s["strides"] = [1,2,4]
        sources.append(s)
    config["sources"] = sources
    sums = {g:sum(s["weight"] for s in sources if family(s["name"]) == g) for g in FAMILY_PERCENT}
    if sums != FAMILY_PERCENT: raise ValueError(f"source family allocation mismatch: {sums}")
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-config", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(); directory = args.output.resolve(); directory.mkdir(parents=True, exist_ok=True)
    original = read_data_config(args.data_config)
    kubric = [s for s in original["sources"] if family(s["name"]) == "kubric"]
    if not kubric: raise ValueError("admitted mixture has no Kubric source")
    manifests = [json.loads(Path(s["manifest"]).read_text())["sequences"] for s in kubric]
    if any(v != manifests[0] for v in manifests): raise ValueError("Kubric admitted pools differ")
    plan = dict(schema="variable_ct_curriculum_v1", target_step=50000,
                family_percent=FAMILY_PERCENT, native_resolution=True, no_query_chunking=True,
                max_source_time_pairs=64, retain_all_admitted_sequences=True, stages=[])
    plan["native_motion_time_caps"] = NATIVE_MOTION_TIME_CAPS
    for stage, (begin, end) in enumerate([(25000,32000),(32000,40000),(40000,50000)]):
        path = directory/f"stage_{stage+1}_{begin}_{end}.yaml"
        config = make_config(original, stage, directory/"runtime_invalid_clips.jsonl")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        plan["stages"].append(dict(start_step=begin, end_step=end, data_config=str(path),
            mono_time_probability=MONO_TIMES[stage], stereo_time_probability=STEREO_TIMES[stage],
            kubric_camera_probability=CAMERA_WEIGHTS, max_kubric_observations=max(v["cameras"]*v["num_times"] for v in config["sources"][0]["layouts"])))
    (directory/"curriculum.json").write_text(json.dumps(plan, indent=2)+"\n")
    print(json.dumps(plan), flush=True)


if __name__ == "__main__": main()
