"""Upgrade an admitted mixture; preserve splits and audit new V2 frame pools."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import yaml

from multicamera4rc.datasets.mixture import create_dataset, read_data_config
from multicamera4rc.datasets.registry import create_adapter


def pair_manifest(left, right, adapter, output, *, tartan=False):
    lm, rm = [json.loads(Path(s["manifest"]).read_text()) for s in (left, right)]
    result = dict(schema="native_stereo_manifest_v1", root=left["root"],
                  sequences={}, validation_sequences={}, exclusions=[])
    for split in ("sequences", "validation_sequences"):
        for seq in adapter.list_sequences():
            ln, rn = (seq+"/left", seq+"/right") if tartan else (seq, seq)
            if ln not in lm[split] or rn not in rm[split]: continue
            lr, rr = lm[split][ln], rm[split][rn]
            if lr["physical_scene"] != rr["physical_scene"]:
                raise ValueError("stereo physical scene split mismatch")
            if tartan:
                streams = [a.records[s]["frames"] for a, s in zip(adapter.streams, (ln, rn))]
                allowed = [set(fs[i] for i in r.get("valid_frames", range(len(fs))))
                           for fs, r in zip(streams, (lr, rr))]
                frames = [i for i, f in enumerate(adapter.frames[seq]) if f in allowed[0] & allowed[1]]
            else:
                n = adapter.get_num_frames(seq)
                frames = sorted(set(lr.get("valid_frames", range(n))) & set(rr.get("valid_frames", range(n))))
            if not frames: continue
            r = dict(length=len(frames), physical_scene=lr["physical_scene"])
            if len(frames) != adapter.get_num_frames(seq): r["valid_frames"] = frames
            result[split][seq] = r
    train = {r["physical_scene"] for r in result["sequences"].values()}
    val = {r["physical_scene"] for r in result["validation_sequences"].values()}
    if train & val: raise ValueError("stereo train/validation scene leakage")
    result.update(train_physical_scenes=sorted(train), validation_physical_scenes=sorted(val),
                  scene_overlap=0, screen="intersection of admitted left/right frames in identical physical-scene splits")
    output.write_text(json.dumps(result, separators=(",", ":"))+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-config", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--v2-root", default="/data8/mugen/TartanAir_V2_subset")
    args = p.parse_args(); directory = args.output.resolve(); directory.mkdir(parents=True, exist_ok=True)
    original = read_data_config(args.data_config); config = deepcopy(original)
    names = {s["name"]: s for s in config["sources"]}
    point = names["pointodyssey"]
    point.update(adapter="pointodyssey_tracks", dense_kind="pointodyssey_tracks", weight=.75)
    additions = []
    for a, b, new, adapter_name, weight, tartan in (
        ("vkitti2_Camera_0", "vkitti2_Camera_1", "vkitti2_stereo", "vkitti2_stereo", .75, False),
        ("tartanair_left", "tartanair_right", "tartanair_v1_stereo", "tartanair_v1_stereo", .2, True)):
        left, right = names[a], names[b]
        adapter = create_adapter(adapter_name, root=left["root"], split="train", cache_dir=str(directory/"index_cache"))
        manifest = directory/(new+"_manifest.json")
        pair_manifest(left, right, adapter, manifest, tartan=tartan)
        left["weight"] /= 2; right["weight"] /= 2
        additions.append(dict(name=new, adapter=adapter_name, root=left["root"], split="train",
            cameras=2, num_times=6, weight=weight, strides=[1], dense_kind="geometry" if tartan else "vkitti_scene_flow",
            manifest=str(manifest), options=dict(cache_dir=str(directory/"index_cache"), verbose=False)))
        print(json.dumps(dict(prepared=new, sequences=len(json.loads(manifest.read_text())["sequences"]))), flush=True)
    # All six native directions use a stereo pair. Distinct directions are
    # not treated as overlapping cameras; a 90-degree cube face can have no
    # common pixels with another face. Split each environment as one unit.
    mono_options = dict(cache_dir=str(directory/"index_cache"), direction="front",
                        stereo=False, camera="lcam", verbose=False, depth_max=200., depth_percentile=98.)
    mono_adapter = create_adapter("tartanair_v2_zip", root=args.v2_root, **mono_options)
    scenes = sorted({r["physical_scene"] for r in mono_adapter.records.values()},
                    key=lambda s: hashlib.sha256(("49:"+s).encode()).hexdigest())
    held = set(scenes[:max(1, math.ceil(.05*len(scenes)))])
    v2_manifests = []
    for direction in ("front", "back", "left", "right", "top", "bottom"):
        name = "tartanair_v2_"+direction; manifest = directory/(name+"_manifest.json")
        options = dict(cache_dir=str(directory/"index_cache"), direction=direction, verbose=False,
                       depth_max=200., depth_percentile=98.)
        adapter = create_adapter("tartanair_v2_zip", root=args.v2_root, **options)
        records = {n:dict(length=adapter.get_num_frames(n), physical_scene=r["physical_scene"])
                   for n, r in adapter.records.items()}
        data = dict(schema="native_v2_stereo_manifest_v1", root=args.v2_root,
            sequences={n:r for n, r in records.items() if r["physical_scene"] not in held},
            validation_sequences={n:r for n, r in records.items() if r["physical_scene"] in held},
            scene_overlap=0, global_validation_physical_scenes=sorted(held),
            validation_physical_scenes=sorted({r["physical_scene"] for r in records.values() if r["physical_scene"] in held}),
            screen="every RGB/depth zip membership and native quaternion row, intersected by physical frame across cameras")
        manifest.write_text(json.dumps(data, separators=(",", ":"))+"\n")
        additions.append(dict(name=name, adapter="tartanair_v2_zip", root=args.v2_root, split="train",
            cameras=2, num_times=6, weight=.1, strides=[1], dense_kind="geometry", manifest=str(manifest), options=options))
        v2_manifests.append(data)
        print(json.dumps(dict(prepared=name, sequences=len(data["sequences"]),
                              frames=sum(r["length"] for r in data["sequences"].values()))), flush=True)
    # All 74 available environments include the left-front stream. Preserve
    # valid monocular data even when its right camera / other faces are absent.
    name = "tartanair_v2_front_mono"; manifest = directory/(name+"_manifest.json")
    records = {n:dict(length=mono_adapter.get_num_frames(n), physical_scene=r["physical_scene"])
               for n, r in mono_adapter.records.items()}
    data = dict(schema="native_v2_mono_manifest_v1", root=args.v2_root,
        sequences={n:r for n, r in records.items() if r["physical_scene"] not in held},
        validation_sequences={n:r for n, r in records.items() if r["physical_scene"] in held},
        scene_overlap=0, global_validation_physical_scenes=sorted(held),
        validation_physical_scenes=sorted(held),
        screen="all left-front RGB/depth zip membership and native NED quaternion rows; identical environment split as stereo")
    manifest.write_text(json.dumps(data, separators=(",", ":"))+"\n")
    additions.append(dict(name=name, adapter="tartanair_v2_zip", root=args.v2_root, split="train",
        cameras=1, num_times=8, weight=.2, strides=[1], dense_kind="geometry", manifest=str(manifest), options=mono_options))
    v2_manifests.append(data)
    assert not {r["physical_scene"] for m in v2_manifests for r in m["sequences"].values()} & {
        r["physical_scene"] for m in v2_manifests for r in m["validation_sequences"].values()}
    print(json.dumps(dict(prepared=name, sequences=len(data["sequences"]),
                          environments=len(scenes), frames=sum(r["length"] for r in data["sequences"].values()))), flush=True)
    config["sources"].extend(additions)
    config["invalid_log"] = str(directory/"runtime_invalid_clips.jsonl")
    dataset = create_dataset(dict(config, epoch_size=1))
    config["epoch_size"] = 2*dataset.coverage_size
    out = directory/"full_mixture_native_motion_stereo.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(passed=True, data_config=str(out), sources=[dict(name=s["display_name"], adapter=s["name"],
        sequences=len(s["pool"]), frames=sum(n for _, n in s["pool"]), cameras=s["cameras"],
        num_times=s["num_times"], dense_kind=s["dense_kind"], weight=w) for s, w in zip(dataset.sources, dataset.weights)],
        coverage_windows=dataset.coverage_size, epoch_size=config["epoch_size"],
        sampler_cursor_compatible=False, reason="new stereo/V2 pools change the coverage index; restart data cursor, retain optimizer/RNG/LR")
    (directory/"native_motion_data_plan.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
