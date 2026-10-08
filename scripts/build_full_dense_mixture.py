"""Screen the complete sequence pools and write a scene-disjoint dense mixture.

The screen reads every calibration row and checks native file membership.
It does NOT claim an offline scan of every RGB/depth pixel in multi-TB packs:
pixel validity and native label cross-checks also run on every loaded batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import yaml

from multicamera4rc.datasets.registry import create_adapter


def calibrated(K, E, *, allow_skew=False, allow_left_world=False):
    K, E = np.asarray(K, np.float64), np.asarray(E, np.float64)
    if K.ndim != 3 or K.shape[1:] != (3, 3) or E.shape != (len(K), 4, 4):
        raise ValueError("calibration must have native frame axis and 3x3/4x4 matrices")
    valid = np.isfinite(K).all((1, 2)) & np.isfinite(E).all((1, 2))
    valid &= (K[:, 0, 0] > 0) & (K[:, 1, 1] > 0)
    valid &= np.max(abs(K[:, 2]-[0, 0, 1]), axis=1) < 1e-6
    valid &= np.max(abs(E[:, 3]-[0, 0, 0, 1]), axis=1) < 1e-6
    R = np.nan_to_num(E[:, :3, :3])
    valid &= np.max(abs(R @ R.transpose(0, 2, 1)-np.eye(3)), axis=(1, 2)) < 5e-3
    determinant = np.linalg.det(R)
    valid &= (abs(abs(determinant)-1) < 5e-3 if allow_left_world else abs(determinant-1) < 5e-3)
    if not allow_skew:
        valid &= (abs(K[:, 0, 1]) < 1e-6) & (abs(K[:, 1, 0]) < 1e-6)
    return valid


def _physical(name, family):
    if family.startswith("vkitti2"): return name.split("_")[0]
    if family == "scannet": return re.sub(r"_\d\d$", "", name)
    if family == "blended": return name.split("_")[0]
    return name


def _screen(adapter, source):
    import h5py
    records, exclusions = {}, []
    family = source["name"]
    names = adapter.list_sequences()
    began = time.monotonic()
    handle = h5py.File(source["root"], "r") if source["adapter"] == "mugen_h5" else None
    try:
        for i, name in enumerate(names):
            try:
                n = adapter.get_num_frames(name)
                physical = _physical(name, family)
                if source["adapter"] == "mugen_h5":
                    r = adapter._get_record(name); g = handle[r.group_path]
                    if g["depths"].shape != g["images"].shape[:-1] or g["depths"].dtype.kind != "f":
                        raise ValueError("packed depth must be floating camera-Z with RGB-matching shape")
                    K, E = np.asarray(g["intrinsics"]), np.asarray(g["extrinsics"])
                    valid = calibrated(K, E)
                    if len(valid) != n: raise ValueError("calibration does not cover all RGB/depth frames")
                elif source["adapter"] == "dynamic_replica":
                    r = adapter._get_record(name); physical = r.base_seq_name
                    valid = calibrated(r.intrinsics, r.extrinsics)
                    # Read directory membership once instead of millions of
                    # individual network-filesystem stat calls.
                    image_dir = adapter.split_root/r.image_rel_paths[0]
                    depth_dir = adapter.split_root/r.depth_rel_paths[0]
                    rgb = {p.name for p in image_dir.parent.iterdir()}
                    dep = {p.name for p in depth_dir.parent.iterdir()}
                    valid &= np.array([Path(p).name in rgb and Path(d).name in dep
                        for p, d in zip(r.image_rel_paths, r.depth_rel_paths)])
                    for directory, predicate in (("flow_forward", lambda f: f < max(r.frame_numbers)),
                            ("flow_forward_mask", lambda f: f < max(r.frame_numbers)),
                            ("flow_backward", lambda f: f > min(r.frame_numbers)),
                            ("flow_backward_mask", lambda f: f > min(r.frame_numbers))):
                        directory_path = r.sequence_dir/directory
                        files = {p.name for p in directory_path.iterdir()} if directory_path.is_dir() else set()
                        valid &= np.array([not predicate(f) or f"{r.sequence_dir.name}_{f:04d}.png" in files for f in r.frame_numbers])
                elif source["adapter"] == "pointodyssey_geometry":
                    r = adapter.get_record(name)
                    if r.anno_path is None: raise ValueError("missing native annotation calibration")
                    hp = r.anno_path.with_suffix(".h5")
                    if hp.is_file():
                        with h5py.File(hp, "r") as f:
                            K, E = np.asarray(f["intrinsics"]), np.asarray(f["extrinsics"])
                    else:
                        with np.load(r.anno_path, allow_pickle=False) as f:
                            K, E = f["intrinsics"], f["extrinsics"]
                    valid = calibrated(K, E, allow_left_world=True)
                    handedness = np.linalg.det(E[valid, :3, :3]) < 0
                    if handedness.any() and not handedness.all():
                        raise ValueError("world handedness changes inside a PointOdyssey sequence")
                    if len(valid) != n or r.depth_paths is None: raise ValueError("PointOdyssey depth/calibration count mismatch")
                    valid &= np.array([p is not None for p in r.depth_paths])
                elif source["adapter"] == "vkitti2":
                    Es, Ks = adapter._load_sequence_metadata(name)
                    valid = np.array([f in Es and f in Ks for f in range(n)])
                    ids = np.flatnonzero(valid)
                    valid[ids] &= calibrated(np.array([Ks[f] for f in ids]), np.array([Es[f] for f in ids]))
                    base = adapter.seq_to_dir[name]/"frames"
                    rgb = {p.name for p in (base/"rgb"/adapter.camera).iterdir()}
                    dep = {p.name for p in (base/"depth"/adapter.camera).iterdir()}
                    for f in range(n):
                        valid[f] &= f"rgb_{f:05d}.jpg" in rgb and f"depth_{f:05d}.png" in dep
                    # Motion files must cover every native adjacent source;
                    # the true sequence endpoints naturally have one direction.
                    for directory, prefix, rng in (("forwardSceneFlow", "sceneFlow", range(n-1)),
                            ("backwardSceneFlow", "backwardSceneFlow", range(1, n)),
                            ("forwardFlow", "flow", range(n-1)), ("backwardFlow", "backwardFlow", range(1, n))):
                        files = {p.name for p in (base/directory/adapter.camera).iterdir()}
                        missing = [f for f in rng if f"{prefix}_{f:05d}.png" not in files]
                        if missing: raise ValueError(f"missing {directory} labels at {missing[:5]}")
                elif source["adapter"] in ("tartanair_v1_zip", "hypersim_zip"):
                    r = adapter.records[name]; physical = r["physical_scene"]
                    if "E" in r:
                        valid = calibrated(np.repeat(r["K"][None], n, 0), r["E"], allow_skew=True)
                    else:
                        p = r["poses"][r["frames"]]
                        valid = np.isfinite(p).all(1) & (abs(np.linalg.norm(p[:, 3:], axis=1)-1) < 5e-3)
                else:
                    # Kubric's revision-4 validation stamps are checked by its
                    # adapter. They certify the native renderer schema/labels.
                    valid = np.ones(n, bool)
                frames = np.flatnonzero(valid).tolist()
                if not frames: raise ValueError("no calibrated native frames")
                record = {"length": len(frames), "physical_scene": physical}
                if source["adapter"] == "pointodyssey_geometry":
                    record["native_world_handedness"] = "left" if handedness.all() else "right"
                if len(frames) != n: record["valid_frames"] = frames
                records[name] = record
                if len(frames) != n:
                    exclusions.append(dict(sequence=name, reason="invalid calibration/file membership", excluded_frames=n-len(frames)))
            except (ValueError, KeyError, OSError, IndexError) as error:
                exclusions.append(dict(sequence=name, reason=str(error), excluded_sequence=True))
            if (i+1) % 500 == 0:
                print(json.dumps({"screen": family, "processed": i+1, "total": len(names), "elapsed": time.monotonic()-began}), flush=True)
    finally:
        if handle is not None: handle.close()
    return records, exclusions


def specifications(cache, metadata):
    opts = dict(cache_dir=str(cache), verbose=False)
    kubric = dict(**opts, index_workers=4, required_schema_version="movi_tracking_v3",
        require_native_tracks=True, require_depth=True, require_validation_stamp=True,
        validation_stamp_dir="/data8/kubric_multicam_tracking_v3_state/validated", min_validation_revision=4,
        split_strategy="variant_balanced", val_ratio=.05, random_cameras=True,
        depth_is_ray_distance=True, extrinsics_are_opengl=False)
    sources = []
    for C, T in ((2, 12), (3, 8), (4, 6)):
        sources.append(dict(name=f"kubric_C{C}", adapter="kubric_multicam", root="/data8/kubric_multicam_tracking_v3",
            split="train", cameras=C, num_times=T, weight=1., strides=[1, 2], options=kubric))
    sources.extend([
        dict(name="dynamic_replica", adapter="dynamic_replica", root="/data5/d4rt_dataset/Dynamic_Replica", num_times=8,
             weight=1.5, options=dict(**opts, cameras="both", load_trajectories=False, pixel_coordinates="array_index_v1", rgb_from_trajectory=False, index_workers=2)),
        dict(name="pointodyssey", adapter="pointodyssey_geometry", root="/data5/d4rt_dataset/PointOdyssey", num_times=8, weight=.3, options=opts),
    ])
    for camera in ("Camera_0", "Camera_1"):
        sources.append(dict(name=f"vkitti2_{camera}", adapter="vkitti2", root="/data8/mugen/vKITTI2", num_times=8, weight=.75,
            options=dict(**opts, camera=camera, strict_calibration=True, load_precomputed=False, load_normals=False, load_flow=False)))
    for camera in ("left", "right"):
        sources.append(dict(name=f"tartanair_{camera}", adapter="tartanair_v1_zip", root="/data8/mugen/TartanAir_V1", num_times=8,
            weight=.2, options=dict(**opts, camera=camera, depth_max=200., depth_percentile=98.)))
    sources.append(dict(name="hypersim", adapter="hypersim_zip", root="/data8/mugen/Hypersim/raw", num_times=1, weight=.3,
        cameras=[2, 4], view_mode="static_multiview",
        options=dict(**opts, camera_metadata=str(metadata/"metadata_camera_parameters.csv"), split_metadata=str(metadata/"metadata_images_split_scene_v1.csv"))))
    for name, file, T, weight in (("scannet", "scannet.h5", 8, .35), ("scannetpp", "scannetpp.h5", 8, .35),
            ("wildrgbd", "wildrgbd.h5", 8, .35), ("blended", "blended_mvg.h5", 1, .2),
            ("taskonomy", "taskonomy.h5", 1, .2), ("megadepth", "megadepth.h5", 1, .1), ("co3d", "co3d.h5", 1, .1)):
        sources.append(dict(name=name, adapter="mugen_h5", root=f"/data7/mugen/{file}", num_times=T, weight=weight,
            options=dict(dataset_name=name, strip_padding=True, generate_static_tracks=False, prefer_precomputed_tracks=False,
                require_generated_tracks=False, assume_static_scene=False, train_only_val_ratio=0., verbose=False)))
        if name in ("blended", "taskonomy"):
            sources[-1].update(cameras=[4, 8], view_mode="static_multiview")
    for source in sources:
        source.setdefault("split", "train"); source.setdefault("cameras", 1); source.setdefault("strides", [1])
        source.setdefault("dense_kind", {"kubric_multicam": "kubric", "dynamic_replica": "replica_flow", "vkitti2": "vkitti_scene_flow"}.get(source["adapter"], "geometry"))
    return sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--metadata", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    sources = specifications(args.output/"index_cache", args.metadata)
    reports, train_sources = [], []
    for source in sources:
        manifest_path = args.output/f"{source['name']}_manifest.json"
        exclusion_path = args.output/f"{source['name']}_exclusions.json"
        if manifest_path.exists() and exclusion_path.exists():
            data = json.loads(manifest_path.read_text()); excluded = json.loads(exclusion_path.read_text())
            print(json.dumps({"resume_screen": source["name"], "sequences": len(data["sequences"])}), flush=True)
        else:
            options = dict(source["options"])
            if source["adapter"] == "kubric_multicam":
                options.update(min_cameras=source["cameras"], max_cameras=source["cameras"], min_times=source["num_times"],
                    max_times=source["num_times"], encoder_tubelet_size=1, sampler_units="time", pixel_coordinates="array_index_v1")
            adapter = create_adapter(source["adapter"], root=source["root"], split=source["split"], **options)
            records, excluded = _screen(adapter, source)
            # Authored train/test splits are kept. Additional validation for
            # unsplit data is assigned to entire physical scenes, all variants
            # and camera streams together (same hash seed for all sources).
            if source["adapter"] not in ("kubric_multicam", "dynamic_replica", "pointodyssey_geometry", "hypersim_zip"):
                physical = sorted({r["physical_scene"] for r in records.values()}, key=lambda s: hashlib.sha256(("49:"+s).encode()).hexdigest())
                held = set(physical[:max(1, math.ceil(.05*len(physical)))])
            else: held = set()
            validation = {s: r for s, r in records.items() if r["physical_scene"] in held}
            records = {s: r for s, r in records.items() if r["physical_scene"] not in held}
            data = dict(schema="full_dense_manifest_v1", root=source["root"], adapter=source["adapter"],
                sequences=records, validation_sequences=validation,
                screen="every sequence schema/calibration row/native membership; runtime pixel masks and independent native label cross-checks",
                estimated_geometry=source["name"] in ("co3d", "megadepth"))
            manifest_path.write_text(json.dumps(data, separators=(",", ":"))+"\n")
            exclusion_path.write_text(json.dumps(excluded, indent=2)+"\n")
            del adapter
        frames = sum(r["length"] for r in data["sequences"].values())
        T = source["num_times"]
        paired = source["dense_kind"] in ("replica_flow", "vkitti_scene_flow")
        windows = sum(math.ceil((r["length"]-1)/(min(T, r["length"])-1)) if paired and min(T, r["length"]) > 1
                      else math.ceil(r["length"]/T) for r in data["sequences"].values())
        report = dict(name=source["name"], sequences=len(data["sequences"]), frames=frames, coverage_windows=windows,
            validation_sequences=len(data["validation_sequences"]), exclusions=len(excluded), manifest=str(manifest_path),
            cameras=source["cameras"], num_times=source["num_times"], view_mode=source.get("view_mode", "temporal"),
            estimated_geometry=data["estimated_geometry"], train_admitted=not data["estimated_geometry"])
        reports.append(report); print(json.dumps(report), flush=True)
        if not data["sequences"]: raise ValueError(f"{source['name']}: no valid training sequences")
        source["manifest"] = str(manifest_path)
        if report["train_admitted"]: train_sources.append(source)
    coverage = sum(r["coverage_windows"] for r in reports if r["train_admitted"])
    config = dict(image_mode="native", supervision="dense", sampling="coverage", num_times=8,
        num_queries=0, epoch_size=2*coverage, seed=49, include_visibility=True, sources=train_sources,
        skip_invalid=True, invalid_log=str(args.output/"runtime_invalid_clips.jsonl"))
    (args.output/"full_mixture_native.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    summary = dict(sources=reports, coverage_windows=coverage, epoch_size=2*coverage,
        sampling="Each epoch includes every admitted frame window once, plus weighted extra windows to preserve dynamic supervision.",
        excluded_sources={"hm3d": "truncated H5", "dtu": "test-only", "tnt": "test-only/no depth",
            "mapillary": "camera/view/time and label provenance not verified", "dl3dv": "depth/pose label provenance not verified",
            "waymo": "sparse LiDAR depth; dense GT unavailable in this pack",
            "vkitti_pack": "unverified version/duplicate native source",
            "co3d": "estimated SfM/MVS depth; original quality scores/masks absent, not certified as native GT",
            "megadepth": "estimated SfM/MVS depth; geometry candidate screened, not admitted as native GT"},
        native_motion_stereo_upgrade="scripts/enable_native_motion_stereo.py adds audited V2 pinhole stereo, V1/vKITTI2 stereo and PointOdyssey tracks to this admitted baseline",
        limitation="This full structural screen does not certify every pixel offline. Every loaded pixel is masked; estimated SfM/MVS labels remain estimated geometry, not native motion GT.")
    (args.output/"full_mixture_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"ready": str(args.output/"full_mixture_native.yaml"), "coverage_windows": coverage}), flush=True)


if __name__ == "__main__": main()
