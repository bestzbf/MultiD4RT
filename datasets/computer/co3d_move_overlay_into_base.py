#!/usr/bin/env python3
"""Move Co3D overlay files into the base dataset tree as sibling overlay files.

This is a space-efficient alternative to rewriting merged precomputed files:

- overlay ``precomputed.npz`` becomes ``<base>/<seq>/precomputed.overlay.npz``
- overlay ``precomputed.failed.json`` becomes ``<base>/<seq>/precomputed.failed.json``

`Co3Dv2Adapter` can then read the overlay directly from the base tree without
an external `track_precompute_root`.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path


_OVERLAY_NPZ_NAME = "precomputed.overlay.npz"
_FAILURE_MARKER_NAME = "precomputed.failed.json"


def _move_one(job: tuple[str, str, str, bool, bool]) -> dict[str, str]:
    overlay_root_str, base_root_str, rel_path_str, overwrite, is_marker = job
    overlay_root = Path(overlay_root_str)
    base_root = Path(base_root_str)
    rel_path = Path(rel_path_str)

    src = overlay_root / rel_path
    sequence_dir = base_root / rel_path.parent
    dst_name = _FAILURE_MARKER_NAME if is_marker else _OVERLAY_NPZ_NAME
    dst = sequence_dir / dst_name

    result = {
        "src": str(src),
        "dst": str(dst),
        "status": "unknown",
    }

    if not src.exists():
        result["status"] = "skip_src_missing"
        return result

    sequence_dir.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if not overwrite:
            result["status"] = "skip_dst_exists"
            return result
        dst.unlink()

    os.replace(src, dst)
    result["status"] = "moved_marker" if is_marker else "moved_overlay"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Move Co3D overlay files into the base dataset tree.")
    parser.add_argument("--overlay-root", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    overlay_root = Path(args.overlay_root)
    tasks = [
        (args.overlay_root, args.base_root, str(p.relative_to(overlay_root)), args.overwrite, False)
        for p in sorted(overlay_root.rglob("precomputed.npz"))
    ] + [
        (args.overlay_root, args.base_root, str(p.relative_to(overlay_root)), args.overwrite, True)
        for p in sorted(overlay_root.rglob(_FAILURE_MARKER_NAME))
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    results = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for idx, result in enumerate(pool.imap_unordered(_move_one, tasks), start=1):
                results.append(result)
                if idx % 500 == 0 or idx == len(tasks):
                    print(f"[{idx}/{len(tasks)}] last={result['status']} {result['dst']}", flush=True)
    else:
        for idx, task in enumerate(tasks, start=1):
            result = _move_one(task)
            results.append(result)
            if idx % 500 == 0 or idx == len(tasks):
                print(f"[{idx}/{len(tasks)}] last={result['status']} {result['dst']}", flush=True)

    summary = {
        "overlay_root": args.overlay_root,
        "base_root": args.base_root,
        "workers": args.workers,
        "overwrite": args.overwrite,
        "limit": args.limit,
        "total_tasks": len(tasks),
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
