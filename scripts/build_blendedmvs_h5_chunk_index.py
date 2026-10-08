#!/usr/bin/env python3
"""Build BlendedMVS precomputed.h5 chunk-index pickles for COS Range reads."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ARRAY_KEYS = ("normals", "trajs_2d", "trajs_3d_world", "valids", "visibs")
SCALAR_KEYS = ("num_frames", "num_points", "ref_frame", "track_semantics_version")


def _to_jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _to_jsonable_scalar(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _dataset_index(ds: h5py.Dataset) -> dict[str, Any]:
    if ds.chunks is None:
        raise ValueError(f"{ds.name}: expected chunked dataset")
    if len(ds.chunks) == 0 or int(ds.chunks[0]) != 1:
        raise ValueError(f"{ds.name}: expected per-frame chunks, got {ds.chunks}")

    offsets: list[tuple[int, int]] = []
    filter_masks: list[int] = []
    coord = [0] * ds.ndim
    for frame_idx in range(int(ds.shape[0])):
        coord[0] = frame_idx
        info = ds.id.get_chunk_info_by_coord(tuple(coord))
        offsets.append((int(info.byte_offset), int(info.size)))
        filter_masks.append(int(info.filter_mask))

    return {
        "offsets": offsets,
        "filter_masks": filter_masks,
        "dtype": str(ds.dtype),
        "chunk_shape": tuple(int(v) for v in ds.chunks),
        "shape": tuple(int(v) for v in ds.shape),
        "compression": ds.compression,
        "compression_opts": ds.compression_opts,
    }


def build_index(h5_path: Path) -> dict[str, Any]:
    index: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        for key in ARRAY_KEYS:
            if key in f:
                index[key] = _dataset_index(f[key])
        for key in SCALAR_KEYS:
            if key in f:
                index[key] = {
                    "scalar": True,
                    "value": _to_jsonable_scalar(f[key][()]),
                }

    missing = [key for key in ("trajs_2d", "trajs_3d_world", "valids", "visibs") if key not in index]
    if missing:
        raise KeyError(f"{h5_path}: missing required keys {missing}")
    return index


def index_path_for(h5_path: Path) -> Path:
    return h5_path.with_name(f"{h5_path.name}_chunk_index.pkl")


def write_index(h5_path: Path, overwrite: bool = False, dry_run: bool = False) -> dict[str, Any]:
    out_path = index_path_for(h5_path)
    if out_path.exists() and not overwrite:
        return {"scene": h5_path.parent.name, "status": "exists", "path": str(out_path)}

    t0 = time.perf_counter()
    index = build_index(h5_path)
    payload = pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - t0
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{out_path.name}.",
            suffix=".tmp",
            dir=str(out_path.parent),
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            os.replace(tmp_name, out_path)
        finally:
            Path(tmp_name).unlink(missing_ok=True)

    return {
        "scene": h5_path.parent.name,
        "status": "dry_run" if dry_run else "built",
        "path": str(out_path),
        "bytes": len(payload),
        "elapsed_s": round(elapsed, 3),
        "keys": sorted(index.keys()),
    }


def load_scenes(args: argparse.Namespace) -> list[str]:
    scenes: list[str] = []
    scenes.extend(args.scenes or [])
    for file_path in args.scenes_file or []:
        scenes.extend(
            line.strip()
            for line in Path(file_path).read_text().splitlines()
            if line.strip()
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for scene in scenes:
        if scene not in seen:
            seen.add(scene)
            deduped.append(scene)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data_cos/hdu_datasets/BlendedMVS")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--scenes-file", action="append")
    args = parser.parse_args()

    root = Path(args.root)
    scenes = load_scenes(args)
    if scenes:
        h5_paths = [root / scene / "precomputed.h5" for scene in scenes]
    else:
        h5_paths = sorted(root.glob("*/precomputed.h5"))
    h5_paths = [path for path in h5_paths if path.exists()]
    if args.limit > 0:
        h5_paths = h5_paths[: args.limit]
    if not h5_paths:
        raise SystemExit(f"No precomputed.h5 files found under {root}")

    print(
        f"root={root} h5_files={len(h5_paths)} workers={args.workers} "
        f"overwrite={args.overwrite} dry_run={args.dry_run}",
        flush=True,
    )
    counts: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_path = {
            executor.submit(write_index, path, args.overwrite, args.dry_run): path
            for path in h5_paths
        }
        for idx, future in enumerate(as_completed(future_to_path), 1):
            path = future_to_path[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "scene": path.parent.name,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(result)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            print(json.dumps(result, sort_keys=True), flush=True)
            if idx % 20 == 0 or idx == len(h5_paths):
                print(f"progress {idx}/{len(h5_paths)} counts={counts}", flush=True)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
