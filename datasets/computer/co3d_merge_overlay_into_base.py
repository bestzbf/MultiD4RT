#!/usr/bin/env python3
"""Merge Co3D track overlay files back into the base dataset tree.

This is safer than blindly copying overlay ``precomputed.npz`` files over the
base files because the base files may also contain ``normals``. The merge keeps
non-track fields from the base file and replaces only track-related fields from
the overlay.

For sequences with ``precomputed.failed.json`` in the overlay root, the script
can strip the stale track arrays from the base file while keeping ``normals``.
This makes the base tree self-contained for depth/normals-only fallback.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np


_TRACK_KEYS = {
    "trajs_2d",
    "trajs_3d_world",
    "valids",
    "visibs",
    "ref_frame",
    "num_points",
    "track_source",
}
_FAILURE_MARKER_NAME = "precomputed.failed.json"


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def _write_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(str(tmp_path), **payload)
    tmp_path.replace(path)


def _sequence_uid(root: Path, path: Path) -> str:
    return str(path.relative_to(root).parent)


def _merge_one(base_root: Path, overlay_npz: Path, apply: bool) -> dict[str, Any]:
    seq = _sequence_uid(overlay_npz.parents[2], overlay_npz)
    base_npz = base_root / seq / "precomputed.npz"
    base_h5 = base_npz.with_suffix(".h5")
    result: dict[str, Any] = {
        "sequence": seq,
        "action": "merge_overlay",
        "base_npz": str(base_npz),
        "overlay_npz": str(overlay_npz),
        "base_exists": base_npz.exists(),
        "base_h5_exists": base_h5.exists(),
    }
    if base_h5.exists():
        result["status"] = "skipped_base_h5_exists"
        return result

    if not base_npz.exists():
        result["status"] = "skipped_base_missing"
        return result

    if not apply:
        result["status"] = "would_merge"
        return result

    base_payload = _load_npz(base_npz)
    overlay_payload = _load_npz(overlay_npz)

    already_merged = True
    for key, value in overlay_payload.items():
        if key not in base_payload or not np.array_equal(base_payload[key], value):
            already_merged = False
            break
    if already_merged:
        result["status"] = "already_merged"
        return result

    merged = dict(base_payload)
    for key, value in overlay_payload.items():
        merged[key] = value

    result["base_keys"] = sorted(base_payload.keys())
    result["overlay_keys"] = sorted(overlay_payload.keys())
    result["merged_keys"] = sorted(merged.keys())
    result["preserved_normals"] = "normals" in base_payload and "normals" in merged
    _write_npz(base_npz, merged)
    result["status"] = "merged"

    return result


def _strip_failed_one(base_root: Path, overlay_marker: Path, apply: bool) -> dict[str, Any]:
    seq = _sequence_uid(overlay_marker.parents[2], overlay_marker)
    base_npz = base_root / seq / "precomputed.npz"
    base_h5 = base_npz.with_suffix(".h5")
    result: dict[str, Any] = {
        "sequence": seq,
        "action": "strip_failed_tracks",
        "base_npz": str(base_npz),
        "overlay_marker": str(overlay_marker),
        "base_exists": base_npz.exists(),
        "base_h5_exists": base_h5.exists(),
    }
    if base_h5.exists():
        result["status"] = "skipped_base_h5_exists"
        return result

    if not base_npz.exists():
        result["status"] = "skipped_base_missing"
        return result

    if not apply:
        result["status"] = "would_strip"
        return result

    base_payload = _load_npz(base_npz)
    removed_track_keys = sorted(set(base_payload.keys()) & _TRACK_KEYS)
    if not removed_track_keys:
        result["status"] = "already_stripped"
        return result
    stripped = {k: v for k, v in base_payload.items() if k not in _TRACK_KEYS}

    result["base_keys"] = sorted(base_payload.keys())
    result["stripped_keys"] = sorted(stripped.keys())
    result["removed_track_keys"] = removed_track_keys
    result["preserved_normals"] = "normals" in base_payload and "normals" in stripped
    _write_npz(base_npz, stripped)
    result["status"] = "stripped"

    return result


def _run_task(task: tuple[str, str, str, bool]) -> dict[str, Any]:
    action, base_root_str, path_str, apply = task
    base_root = Path(base_root_str)
    path = Path(path_str)
    if action == "merge":
        return _merge_one(base_root, path, apply)
    if action == "strip":
        return _strip_failed_one(base_root, path, apply)
    raise ValueError(f"Unknown task action: {action}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Co3D track overlay files into the base dataset tree."
    )
    parser.add_argument("--base-root", required=True, help="Base Co3D dataset root")
    parser.add_argument("--overlay-root", required=True, help="Overlay root containing fixed tracks")
    parser.add_argument("--output-json", default=None, help="Optional summary JSON path")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for --apply")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in-place. Default is dry-run.",
    )
    args = parser.parse_args()

    base_root = Path(args.base_root)
    overlay_root = Path(args.overlay_root)

    overlay_npz_files = sorted(overlay_root.rglob("precomputed.npz"))
    overlay_failure_markers = sorted(overlay_root.rglob(_FAILURE_MARKER_NAME))

    tasks = [
        ("merge", str(base_root), str(overlay_npz), args.apply)
        for overlay_npz in overlay_npz_files
    ] + [
        ("strip", str(base_root), str(overlay_marker), args.apply)
        for overlay_marker in overlay_failure_markers
    ]

    results = []
    if args.apply and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_task, task) for task in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for task in tasks:
            results.append(_run_task(task))

    summary = {
        "mode": "apply" if args.apply else "dry_run",
        "base_root": str(base_root),
        "overlay_root": str(overlay_root),
        "overlay_npz_count": len(overlay_npz_files),
        "overlay_failure_marker_count": len(overlay_failure_markers),
        "merged": sum(1 for r in results if r["status"] == "merged"),
        "would_merge": sum(1 for r in results if r["status"] == "would_merge"),
        "already_merged": sum(1 for r in results if r["status"] == "already_merged"),
        "stripped": sum(1 for r in results if r["status"] == "stripped"),
        "would_strip": sum(1 for r in results if r["status"] == "would_strip"),
        "already_stripped": sum(1 for r in results if r["status"] == "already_stripped"),
        "skipped_base_missing": sum(1 for r in results if r["status"] == "skipped_base_missing"),
        "skipped_base_h5_exists": sum(1 for r in results if r["status"] == "skipped_base_h5_exists"),
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2)
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
