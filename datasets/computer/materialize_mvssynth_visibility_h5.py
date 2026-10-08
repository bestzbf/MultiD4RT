#!/usr/bin/env python3
"""Materialize MVS-Synth compact static tracks into dense H5 visibility caches.

The MVS-Synth precompute path can store compact ``static_points_world`` to keep
the initial cache tiny.  This script loads those compact points through
``MVSSynthAdapter`` with the requested visibility policy, projects them through
all frames once, and writes ``precomputed.h5`` containing direct
``trajs_2d``/``trajs_3d_world``/``valids``/``visibs`` arrays.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.mvssynth import MVSSynthAdapter


PER_FRAME_KEYS = {"trajs_2d", "trajs_3d_world", "valids", "visibs"}
PER_POINT_KEYS = {
    "static_points_world",
    "static_point_ref_frames",
    "static_point_ref_uv",
    "static_point_ref_depth",
}


def _optional_positive_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and value > 0.0 else None


def _write_dataset(handle: h5py.File, key: str, value: Any) -> None:
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O"}:
        return
    if key in PER_FRAME_KEYS and arr.ndim >= 2 and arr.shape[0] > 1:
        handle.create_dataset(key, data=arr, chunks=(1,) + arr.shape[1:])
    elif key not in PER_POINT_KEYS and arr.ndim >= 2 and arr.shape[0] > 1:
        handle.create_dataset(key, data=arr, chunks=(1,) + arr.shape[1:])
    else:
        handle.create_dataset(key, data=arr)


def _materialize_one(args: tuple) -> tuple[str, str | None]:
    (
        seq_name,
        root,
        precompute_root,
        output_root,
        overwrite,
        adapter_kwargs,
    ) = args
    out_dir = Path(output_root) / seq_name
    out_path = out_dir / "precomputed.h5"
    tmp_path = out_dir / "precomputed.h5.tmp"
    if out_path.exists() and not overwrite:
        return seq_name, None
    if overwrite:
        out_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)

    try:
        adapter = MVSSynthAdapter(
            root=root,
            precompute_root=precompute_root,
            load_precomputed=True,
            load_normals=False,
            verbose=False,
            **adapter_kwargs,
        )
        info = adapter.get_sequence_info(seq_name)
        frame_indices = list(range(int(info["num_frames"])))
        clip = adapter.load_clip(seq_name, frame_indices)
        if (
            clip.trajs_2d is None
            or clip.trajs_3d_world is None
            or clip.valids is None
            or clip.visibs is None
        ):
            return seq_name, "missing dense tracks after adapter projection"

        npz_path = Path(precompute_root) / seq_name / "precomputed.npz"
        base: dict[str, Any] = {}
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as raw:
                for key in raw.files:
                    if key in PER_FRAME_KEYS or key == "normals":
                        continue
                    base[key] = raw[key]

        base.update(
            {
                "trajs_2d": np.asarray(clip.trajs_2d, dtype=np.float32),
                "trajs_3d_world": np.asarray(clip.trajs_3d_world, dtype=np.float32),
                "valids": np.asarray(clip.valids, dtype=np.bool_),
                "visibs": np.asarray(clip.visibs, dtype=np.bool_),
                "num_frames": np.array(len(frame_indices), dtype=np.int32),
                "num_points": np.array(clip.trajs_2d.shape[1], dtype=np.int32),
                "compact_static_tracks": np.array(False, dtype=np.bool_),
                "materialized_dense_visibility": np.array(True, dtype=np.bool_),
            }
        )
        metadata = clip.metadata if isinstance(clip.metadata, dict) else {}
        origin_shift = metadata.get("origin_shift")
        if origin_shift is not None:
            origin_shift = np.asarray(origin_shift, dtype=np.float32).reshape(-1)
            if origin_shift.size == 3:
                base["origin_shift"] = origin_shift
        policy_scalars = {
            "visibility_depth_support_radius": adapter.visibility_depth_support_radius,
            "visibility_visible_abs_thresh": adapter.visibility_visible_abs_thresh,
            "visibility_occlusion_abs_thresh": adapter.visibility_occlusion_abs_thresh,
            "visibility_occlusion_rel_thresh": adapter.visibility_occlusion_rel_thresh,
            "visibility_occlusion_min_support": adapter.visibility_occlusion_min_support,
            "visibility_occlusion_far_depth_start": adapter.visibility_occlusion_far_depth_start or 0.0,
            "visibility_occlusion_far_depth_end": adapter.visibility_occlusion_far_depth_end or 0.0,
            "visibility_occlusion_far_abs_thresh": adapter.visibility_occlusion_far_abs_thresh or 0.0,
            "visibility_occlusion_far_rel_thresh": adapter.visibility_occlusion_far_rel_thresh or 0.0,
            "visibility_occlusion_far_min_support": adapter.visibility_occlusion_far_min_support or 0,
            "visibility_far_uncertain_as_visible_depth": adapter.visibility_far_uncertain_as_visible_depth or 0.0,
            "visibility_depth_max": adapter.visibility_depth_max or 0.0,
        }
        for key, value in policy_scalars.items():
            dtype = np.int32 if isinstance(value, int) else np.float32
            base[key] = np.array(value, dtype=dtype)

        out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(tmp_path, "w") as handle:
            for key in sorted(base.keys()):
                _write_dataset(handle, key, base[key])
        os.replace(tmp_path, out_path)
        return seq_name, None
    except Exception:
        tmp_path.unlink(missing_ok=True)
        return seq_name, traceback.format_exc()


def _parse_sequences(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--precompute-root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visibility-depth-support-radius", type=int, default=2)
    parser.add_argument("--visibility-visible-abs-thresh", type=float, default=0.25)
    parser.add_argument("--visibility-occlusion-abs-thresh", type=float, default=1.0)
    parser.add_argument("--visibility-occlusion-rel-thresh", type=float, default=0.08)
    parser.add_argument("--visibility-occlusion-min-support", type=int, default=9)
    parser.add_argument("--visibility-occlusion-far-depth-start", type=float, default=None)
    parser.add_argument("--visibility-occlusion-far-depth-end", type=float, default=None)
    parser.add_argument("--visibility-occlusion-far-abs-thresh", type=float, default=None)
    parser.add_argument("--visibility-occlusion-far-rel-thresh", type=float, default=None)
    parser.add_argument("--visibility-occlusion-far-min-support", type=int, default=None)
    parser.add_argument("--visibility-far-uncertain-as-visible-depth", type=float, default=None)
    parser.add_argument("--visibility-depth-max", type=float, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    precompute_root = Path(args.precompute_root)
    output_root = Path(args.output_root) if args.output_root else precompute_root
    adapter = MVSSynthAdapter(
        root=str(root),
        precompute_root=str(precompute_root),
        load_precomputed=True,
        load_normals=False,
        verbose=False,
    )
    sequences = _parse_sequences(args.sequences) or adapter.list_sequences()
    missing = [seq for seq in sequences if seq not in adapter._name_to_record]
    if missing:
        raise ValueError(f"Unknown MVS-Synth sequence ids: {missing}")

    adapter_kwargs = {
        "visibility_depth_support_radius": args.visibility_depth_support_radius,
        "visibility_visible_abs_thresh": args.visibility_visible_abs_thresh,
        "visibility_occlusion_abs_thresh": args.visibility_occlusion_abs_thresh,
        "visibility_occlusion_rel_thresh": args.visibility_occlusion_rel_thresh,
        "visibility_occlusion_min_support": args.visibility_occlusion_min_support,
        "visibility_occlusion_far_depth_start": _optional_positive_float(args.visibility_occlusion_far_depth_start),
        "visibility_occlusion_far_depth_end": _optional_positive_float(args.visibility_occlusion_far_depth_end),
        "visibility_occlusion_far_abs_thresh": _optional_positive_float(args.visibility_occlusion_far_abs_thresh),
        "visibility_occlusion_far_rel_thresh": _optional_positive_float(args.visibility_occlusion_far_rel_thresh),
        "visibility_occlusion_far_min_support": args.visibility_occlusion_far_min_support,
        "visibility_far_uncertain_as_visible_depth": _optional_positive_float(args.visibility_far_uncertain_as_visible_depth),
        "visibility_depth_max": _optional_positive_float(args.visibility_depth_max),
    }

    jobs = [
        (
            seq,
            str(root),
            str(precompute_root),
            str(output_root),
            bool(args.overwrite),
            adapter_kwargs,
        )
        for seq in sequences
    ]
    print(f"[mvssynth-h5] sequences={len(jobs)} output={output_root}")
    t0 = time.perf_counter()
    failures: list[tuple[str, str]] = []
    if args.workers <= 1:
        for job in tqdm(jobs, desc="mvssynth-h5"):
            seq, err = _materialize_one(job)
            if err:
                failures.append((seq, err))
                tqdm.write(f"[ERR] {seq}: {err.splitlines()[-1]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_materialize_one, job): job[0] for job in jobs}
            with tqdm(total=len(futures), desc="mvssynth-h5") as pbar:
                for fut in as_completed(futures):
                    seq, err = fut.result()
                    pbar.update(1)
                    if err:
                        failures.append((seq, err))
                        tqdm.write(f"[ERR] {seq}: {err.splitlines()[-1]}")
    elapsed = time.perf_counter() - t0
    print(f"[mvssynth-h5] done {len(jobs) - len(failures)}/{len(jobs)} in {elapsed:.1f}s")
    if failures:
        print("[mvssynth-h5] failures:")
        for seq, err in failures:
            print(f"  {seq}:\n{err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
