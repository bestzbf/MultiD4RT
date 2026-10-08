#!/usr/bin/env python3
"""Materialize Co3D overlay semantics into base-root precomputed.h5 files.

This avoids rewriting compressed ``precomputed.npz`` files. Instead we write
merged ``precomputed.h5`` files beside the base data tree, which
``load_precomputed_fast()`` already prefers over ``.npz``.

Semantics:
- overlay ``precomputed.npz`` exists:
  merge base payload + overlay track fields into base ``precomputed.h5``
- overlay ``precomputed.failed.json`` exists:
  write base payload without stale track fields into base ``precomputed.h5``
- neither exists:
  do nothing; base ``precomputed.npz`` remains authoritative
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import h5py
import numpy as np


_FAILURE_MARKER_NAME = "precomputed.failed.json"
_TRACK_KEYS = {
    "trajs_2d",
    "trajs_3d_world",
    "valids",
    "visibs",
    "ref_frame",
    "num_points",
    "track_source",
}


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def _normalize_for_h5(value: Any) -> tuple[Any, dict[str, Any]]:
    """Convert numpy string/object payloads into HDF5-compatible values."""
    kwargs: dict[str, Any] = {}
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)

    if arr.dtype.kind == "U":
        kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
        return arr.astype(object), kwargs

    if arr.dtype.kind == "O":
        if arr.ndim == 0:
            item = arr.item()
            if item is None:
                kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
                return np.asarray("", dtype=object), kwargs
            if isinstance(item, bytes):
                item = item.decode("utf-8", errors="replace")
            kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
            return np.asarray(item, dtype=object), kwargs

        flat = []
        for item in arr.reshape(-1):
            if isinstance(item, bytes):
                flat.append(item.decode("utf-8", errors="replace"))
            elif item is None:
                flat.append("")
            else:
                flat.append(str(item))
        kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
        return np.asarray(flat, dtype=object).reshape(arr.shape), kwargs

    return arr, kwargs


def _write_h5(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with h5py.File(tmp_path, "w") as f:
        for key, value in payload.items():
            arr, kwargs = _normalize_for_h5(value)
            if isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[0] > 1:
                kwargs.setdefault("chunks", (1,) + arr.shape[1:])
                kwargs.setdefault("compression", "lzf")
            f.create_dataset(key, data=arr, **kwargs)
    os.replace(tmp_path, path)


def _merge_payload(base_payload: dict[str, Any], overlay_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_payload)
    for key, value in overlay_payload.items():
        merged[key] = value
    return merged


def _strip_payload(base_payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in base_payload.items() if k not in _TRACK_KEYS}


def _process_one(job: tuple[str, str, str, bool]) -> dict[str, Any]:
    base_root_str, overlay_root_str, sequence, overwrite = job
    base_root = Path(base_root_str)
    overlay_root = Path(overlay_root_str)

    base_npz = base_root / sequence / "precomputed.npz"
    out_h5 = base_root / sequence / "precomputed.h5"
    overlay_npz = overlay_root / sequence / "precomputed.npz"
    overlay_marker = overlay_root / sequence / _FAILURE_MARKER_NAME

    result = {
        "sequence": sequence,
        "base_npz": str(base_npz),
        "out_h5": str(out_h5),
        "overlay_npz_exists": overlay_npz.exists(),
        "overlay_marker_exists": overlay_marker.exists(),
    }

    if not base_npz.exists():
        result["status"] = "skip_base_missing"
        return result

    if out_h5.exists() and not overwrite:
        result["status"] = "skip_h5_exists"
        return result

    if overlay_npz.exists():
        base_payload = _load_npz(base_npz)
        overlay_payload = _load_npz(overlay_npz)
        payload = _merge_payload(base_payload, overlay_payload)
        _write_h5(out_h5, payload)
        result["status"] = "merged_h5"
        return result

    if overlay_marker.exists():
        base_payload = _load_npz(base_npz)
        payload = _strip_payload(base_payload)
        _write_h5(out_h5, payload)
        result["status"] = "stripped_h5"
        return result

    result["status"] = "skip_no_overlay"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize merged Co3D precomputed.h5 files into the base dataset tree."
    )
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--overlay-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    overlay_root = Path(args.overlay_root)
    tasks = []
    for p in sorted(overlay_root.rglob("precomputed.npz")):
        tasks.append((args.base_root, args.overlay_root, str(p.relative_to(overlay_root).parent), args.overwrite))
    for p in sorted(overlay_root.rglob(_FAILURE_MARKER_NAME)):
        tasks.append((args.base_root, args.overlay_root, str(p.relative_to(overlay_root).parent), args.overwrite))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    t0 = time.perf_counter()
    results = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for idx, result in enumerate(pool.imap_unordered(_process_one, tasks), start=1):
                results.append(result)
                if idx % 100 == 0 or idx == len(tasks):
                    print(f"[{idx}/{len(tasks)}] last={result['status']} {result['sequence']}", flush=True)
    else:
        for idx, job in enumerate(tasks, start=1):
            result = _process_one(job)
            results.append(result)
            if idx % 100 == 0 or idx == len(tasks):
                print(f"[{idx}/{len(tasks)}] last={result['status']} {result['sequence']}", flush=True)

    summary = {
        "base_root": args.base_root,
        "overlay_root": args.overlay_root,
        "workers": args.workers,
        "overwrite": args.overwrite,
        "limit": args.limit,
        "total_tasks": len(tasks),
        "elapsed_sec": time.perf_counter() - t0,
    }
    for status in sorted({r["status"] for r in results}):
        summary[status] = sum(1 for r in results if r["status"] == status)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
