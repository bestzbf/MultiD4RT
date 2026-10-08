"""Real native GT, stereo covisibility, and full-pool particle schema audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import cv2
import h5py
import numpy as np
from PIL import Image
import torch

from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.partial_dense import vkitti_pairs, _world
if __package__:
    from .audit_static_multiview import cross_view_depth
else:  # Direct script invocation with the project installed or PYTHONPATH=.
    from audit_static_multiview import cross_view_depth


def stats(a):
    a = np.asarray(a).ravel(); a = a[np.isfinite(a)]
    return dict(count=len(a), median=float(np.median(a)), p95=float(np.quantile(a, .95)),
                max=float(a.max())) if len(a) else dict(count=0)


def check_point_schema(source):
    adapter = source["adapter"]; rows = []
    for seq, n in source["pool"]:
        r = adapter.get_record(seq)
        if r.fast_anno_paths:
            arrays = {k:np.load(r.fast_anno_paths[k], mmap_mode="r")
                      for k in ("trajs_2d", "trajs_3d", "valids", "visibs")}
            shapes = {k:list(a.shape) for k, a in arrays.items()}
        else:
            with h5py.File(r.anno_path.with_suffix(".h5"), "r") as f:
                shapes = {k:list(f[k].shape) for k in ("trajs_2d", "trajs_3d", "valids", "visibs")}
        N = shapes["trajs_2d"][1]
        native3d = len(shapes["trajs_3d"]) == 3
        assert (shapes["trajs_3d"] == [r.num_frames, N, 3] if native3d else not shapes["trajs_3d"]) and N > 0
        assert shapes["trajs_2d"] == [r.num_frames, N, 2]
        assert shapes["valids"] == shapes["visibs"] == [r.num_frames, N]
        rows.append(dict(sequence=seq, native_frames=r.num_frames, admitted_frames=n, tracks=N, has_native_3d=native3d))
    return dict(passed=True, sequences=len(rows), native_3d_sequences=sum(r["has_native_3d"] for r in rows),
                depth_lifted_2d_sequences=sum(not r["has_native_3d"] for r in rows), rows=rows,
                screen="all admitted native annotation shapes and persistent columns")


def stereo_flow_agreement(clip, pairs, instances, times):
    """Independently reproject source pixels; compare native same-object flow."""
    background, foreground = [], []
    for source, other in ((0, times), (times, 0)):
        a, b = pairs[source][0], pairs[other][0]
        if a is None or b is None: continue
        assert clip.metadata["frame_indices"][a[0]] == clip.metadata["frame_indices"][b[0]]
        depth = clip.depths[source]; h, w = depth.shape
        y, x = np.mgrid[0:h:8, 0:w:8]; uv = np.stack((x, y), -1)
        world = _world(uv, depth[y, x], clip.intrinsics[source], clip.extrinsics[source])
        E = clip.extrinsics[other]; camera = world @ E[:3, :3].T+E[:3, 3]
        projection = camera @ clip.intrinsics[other].T
        coords = projection[..., :2]/np.maximum(projection[..., 2:], 1e-8)
        finite = np.isfinite(coords).all(-1)
        nearest = np.rint(np.nan_to_num(coords, nan=-1e6, posinf=1e6, neginf=-1e6)).astype(np.int64)
        inside = finite & (camera[..., 2] > 0) & (nearest[..., 0] >= 0) & (nearest[..., 0] < w)
        inside &= (nearest[..., 1] >= 0) & (nearest[..., 1] < h)
        u, v = np.clip(nearest[..., 0], 0, w-1), np.clip(nearest[..., 1], 0, h-1)
        target_z = clip.depths[other][v, u]
        valid = inside & (depth[y, x] > 0) & (target_z > 0)
        valid &= abs(target_z-camera[..., 2]) <= .05*np.maximum(camera[..., 2], 1e-8)
        valid &= a[2][y, x] & b[2][v, u]
        valid &= instances[source][y, x] == instances[other][v, u]
        error = np.linalg.norm(a[1][y, x]-b[1][v, u], axis=-1)
        object_mask = instances[source][y, x] > 0
        background.extend(error[valid & ~object_mask]); foreground.extend(error[valid & object_mask])
    result = dict(background_delta_difference_m=stats(background), object_delta_difference_m=stats(foreground),
        policy="stride8 native depth reprojection, nearest pixel, <=5pct Z, identical instance ID, both native forward scene-flow masks")
    for values in (background, foreground):
        if len(values): assert np.quantile(values, .95) < .05
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-config", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequences-per-source", type=int, default=4)
    p.add_argument("--sources", nargs="+", help="Restrict this audit to named mixture sources")
    args = p.parse_args(); torch.set_num_threads(2); cv2.setNumThreads(1)
    data = create_dataset(args.data_config)
    if args.sources and set(args.sources)-{s["display_name"] for s in data.sources}:
        raise ValueError("requested audit source is absent from mixture")
    entries = {(si, seq):(data.coverage_ends[i-1] if i else 0, end)
        for i, ((si, seq, _, _), end) in enumerate(zip(data.coverage_entries, data.coverage_ends))}
    report = dict(passed=False, data_config=args.data_config, samples=[], limitations=
        "Full-pool schema/membership/calibration is screened; independent per-pixel GT checks below are representative windows, not all dataset pixels.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for si, source in enumerate(data.sources):
        if args.sources and source["display_name"] not in args.sources: continue
        if source["display_name"] != "pointodyssey" and source["cameras"] != 2 and source["name"] != "tartanair_v2_zip": continue
        if source["name"] == "kubric_multicam": continue
        if source["display_name"] == "pointodyssey": report["pointodyssey_pool"] = check_point_schema(source)
        ids = sorted(set(np.linspace(0, len(source["pool"])-1, args.sequences_per_source, dtype=int)))
        for position in ids:
            seq, length = source["pool"][position]; first, end = entries[si, seq]
            for index in sorted({first, end-1}):
                began = time.monotonic(); sample = data[index]
                b = collate_adapter_samples([sample]); S = len(sample["images"])
                selected = torch.arange(min(4, S))[None]
                target = materialize_dense_targets(b, selected)
                for k in ("depth", "K", "c2w", "trajectory", "displacement"):
                    assert torch.isfinite(target[k]).all(), k
                assert torch.allclose(target["c2w"][0, 0], torch.eye(4), atol=1e-6)
                C = sample["camera_id"].unique().numel(); T = sample["query_time_idx"].numel()
                assert S == C*T and len(set(zip(sample["camera_id"].tolist(), sample["time_idx"].tolist()))) == S
                frames = sample["time_idx"].tolist()[:T]
                # Load raw native observations independently of target materialization.
                ordinal = source["frame_pool"].get(seq, list(range(length)))
                window_stride = T-1 if source["dense_kind"] == "vkitti_scene_flow" and T > 1 else T
                start = min((index-first)*window_stride, length-T)
                native_indices = ordinal[start:start+T]
                raw = source["adapter"].load_clip(seq, native_indices*C)
                h, w = raw.images[0].shape[:2]
                for s in range(S):
                    torch.testing.assert_close(sample["targets"]["depth"][s, :h, :w][sample["targets"]["depth_valid"][s, :h, :w]],
                        torch.tensor(raw.depths[s])[sample["targets"]["depth_valid"][s, :h, :w]], atol=0, rtol=0)
                    torch.testing.assert_close(sample["targets"]["K"][s], torch.tensor(raw.intrinsics[s]).float(), atol=0, rtol=0)
                    expected = np.asarray(raw.images[s], np.float32)
                    if raw.images[s].dtype == np.uint8 or expected.max() > 1: expected = expected/255.
                    torch.testing.assert_close(sample["images"][s, :, :h, :w], torch.from_numpy(expected).permute(2, 0, 1), atol=0, rtol=0)
                item = dict(source=source["display_name"], sequence=seq, index=index, input_shape=list(b["images"].shape),
                    cameras=C, times=T, time_idx=sample["time_idx"].tolist(), depth_pixels=int(target["depth_valid"].sum()),
                    source_pixels=int(target["source_pixel_valid"].sum()), scene_scale=float(target["scene_scale"][0]),
                    dense_motion_point_times=int(target["displacement_valid"].sum()), rgb_depth_K_unchanged=True)
                if C == 2:
                    pairs = cross_view_depth(target, source=0)
                    left_to_right = next(r for r in pairs if r["target_camera"] == T)
                    reverse = next(r for r in cross_view_depth(target, source=T) if r["target_camera"] == 0)
                    centers = target["c2w"][0, :, :3, 3].cpu().numpy()*float(target["scene_scale"][0])
                    item.update(stereo_baseline_m=float(np.linalg.norm(centers[T]-centers[0])),
                                left_to_right=left_to_right, right_to_left=reverse)
                    assert min(left_to_right["matched_within_5pct"], reverse["matched_within_5pct"]) >= 128
                if source["dense_kind"] == "vkitti_scene_flow":
                    pairs = vkitti_pairs(raw, source["adapter"])
                    bg_errors, obj_pixels, obj_motion, instances = [], 0, [], {}
                    for s, row in enumerate(pairs):
                        if row[0] is None: continue
                        t, dv, valid = row[0]
                        assert s//T == t//T  # Same-camera adjacent pairing.
                        cam = raw.metadata["camera_names"][s]
                        path = source["adapter"].seq_to_dir[seq]/"frames"/"instanceSegmentation"/cam/f"instancegt_{frames[s%T]:05d}.png"
                        with Image.open(path) as image: instance = np.asarray(image)
                        instances[s] = instance
                        assert instance.shape == valid.shape
                        bg = valid & (instance == 0); obj = valid & (instance > 0)
                        bg_errors.extend(np.linalg.norm(dv[bg], axis=-1)[::16])
                        obj_motion.extend(np.linalg.norm(dv[obj], axis=-1)[::16]); obj_pixels += int(obj.sum())
                    item.update(background_world_motion_m=stats(bg_errors), object_world_motion_m=stats(obj_motion),
                                object_supervised_pixels=obj_pixels, adjacent_pair_camera_match=True)
                    assert item["background_world_motion_m"].get("p95", 0.) < .03
                    if C == 2: item["stereo_native_scene_flow_agreement"] = stereo_flow_agreement(raw, pairs, instances, T)
                if source["dense_kind"] == "pointodyssey_tracks":
                    t = target; n = len(t["sparse_queries"][0]); uv = t["sparse_queries"][0]
                    track = t["sparse_track_id"][0]; obs = uv[:, 0].long()
                    expected = raw.trajs_3d_world[:, track.numpy()].transpose(1, 0, 2)
                    E0 = raw.extrinsics[0].astype(np.float64)
                    expected = expected @ E0[:3, :3].T+E0[:3, 3]
                    expected /= float(t["scene_scale"][0])
                    known = t["sparse_trajectory_valid"][0]
                    np.testing.assert_allclose(t["sparse_trajectory"][0].numpy()[known], expected[known.numpy()], atol=3e-5, rtol=2e-5)
                    assert not t["displacement_valid"].any() and not t["static_mask"].any()
                    assert n == int(t["sparse_query_valid"].sum()) and n > 0
                    hidden = t["sparse_visibility_valid"].bool() & (t["sparse_visibility"] == 0)
                    if not raw.metadata["has_native_world_trajectories"]:
                        assert not (known & ~raw.visibs[:, track.numpy()].T).any()
                    item.update(native_queries=n, native_point_times=int(t["sparse_trajectory_valid"].sum()),
                        occluded_native_point_times=int(hidden.sum()),
                        moving_native_point_times=int(((t["sparse_displacement"].norm(dim=-1) > .002) & t["sparse_displacement_valid"]).sum()),
                        source_checks=raw.metadata["native_track_checks"], native_world_handedness=raw.metadata["native_world_handedness"],
                        trajectory_source=raw.metadata["trajectory_source"], hidden_3d_supervision=raw.metadata["hidden_3d_supervision"],
                        full_native_track_pool=True, invented_dense_motion=False)
                elif source["dense_kind"] == "geometry":
                    assert not target["displacement_valid"].any() and not target["static_mask"].any()
                item["elapsed_seconds"] = time.monotonic()-began
                report["samples"].append(item); args.output.write_text(json.dumps(report, indent=2)+"\n")
                print(json.dumps(item), flush=True)
                del sample, b, target, raw
    report["passed"] = True; args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(passed=True, samples=len(report["samples"]))), flush=True)


if __name__ == "__main__": main()
