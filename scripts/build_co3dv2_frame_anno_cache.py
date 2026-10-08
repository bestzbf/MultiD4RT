#!/usr/bin/env python3
"""Build fast Co3Dv2 frame-annotation npz caches.

Co3Dv2 stores per-frame camera/path metadata in one large
``frame_annotations.jgz`` per category.  Loading those JSON files inside
training workers can take many seconds on COS-backed mounts.  This script
converts them once into the stacked npz layout consumed by
``Co3Dv2Adapter._lookup_frame_anno_from_npz``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.adapters.co3dv2 import ALL_CATEGORIES


def _string_array(values: list[str]) -> np.ndarray:
    max_len = max((len(v) for v in values), default=1)
    return np.asarray(values, dtype=f"<U{max_len}")


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"{path} contains {type(payload).__name__}, expected list")
    return payload


def _build_category(root: Path, out_dir: Path, category: str, force: bool) -> dict[str, Any]:
    src_path = root / category / "frame_annotations.jgz"
    out_path = out_dir / f"frame_anno_{category}_v1.npz"
    if out_path.exists() and not force:
        return {
            "category": category,
            "status": "skipped",
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
        }
    if not src_path.is_file():
        raise FileNotFoundError(src_path)

    t0 = time.perf_counter()
    annotations = _load_annotations(src_path)
    annotations.sort(key=lambda item: (str(item["sequence_name"]), int(item["frame_number"])))

    seq_names: list[str] = []
    offsets: list[int] = [0]
    fns: list[int] = []
    r_values: list[Any] = []
    t_values: list[Any] = []
    focal_values: list[Any] = []
    pp_values: list[Any] = []
    imgsz_values: list[Any] = []
    dscale_values: list[float] = []
    img_stems: list[str] = []

    current_seq: str | None = None
    for entry in annotations:
        seq_name = str(entry["sequence_name"])
        frame_number = int(entry["frame_number"])
        image = entry["image"]
        depth = entry["depth"]
        viewpoint = entry["viewpoint"]

        if current_seq != seq_name:
            if current_seq is not None:
                offsets.append(len(fns))
            seq_names.append(seq_name)
            current_seq = seq_name

        fns.append(frame_number)
        r_values.append(viewpoint["R"])
        t_values.append(viewpoint["T"])
        focal_values.append(viewpoint["focal_length"])
        pp_values.append(viewpoint["principal_point"])
        imgsz_values.append(image["size"])
        dscale_values.append(float(depth.get("scale_adjustment", 1.0)))
        img_stems.append(Path(str(image["path"])).stem)

    offsets.append(len(fns))

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    np.savez(
        tmp_path,
        seq_names=_string_array(seq_names),
        offsets=np.asarray(offsets, dtype=np.int64),
        fns=np.asarray(fns, dtype=np.int32),
        R=np.asarray(r_values, dtype=np.float32),
        T=np.asarray(t_values, dtype=np.float32),
        focal=np.asarray(focal_values, dtype=np.float32),
        pp=np.asarray(pp_values, dtype=np.float32),
        imgsz=np.asarray(imgsz_values, dtype=np.int32),
        dscale=np.asarray(dscale_values, dtype=np.float32),
        img_stems=_string_array(img_stems),
    )
    if not tmp_path.exists() and tmp_path.with_suffix(tmp_path.suffix + ".npz").exists():
        tmp_path = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    os.replace(tmp_path, out_path)
    elapsed = time.perf_counter() - t0
    return {
        "category": category,
        "status": "built",
        "path": str(out_path),
        "frames": len(fns),
        "sequences": len(seq_names),
        "bytes": out_path.stat().st_size,
        "elapsed_s": round(elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data_cos/hdu_datasets/Co3Dv2")
    parser.add_argument(
        "--out-dir",
        default="/home/zbf/16t/e/ZBF_Data/0/.index_cache_5datasets_blendedmvs_hdu/co3dv2_frame_anno",
    )
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    categories = list(args.categories) if args.categories else list(ALL_CATEGORIES)
    workers = max(1, int(args.workers))

    print(f"root={root}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"categories={len(categories)} workers={workers} force={args.force}", flush=True)

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_category = {
            executor.submit(_build_category, root, out_dir, category, args.force): category
            for category in categories
        }
        for idx, future in enumerate(as_completed(future_to_category), 1):
            category = future_to_category[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{idx:02d}/{len(categories):02d}] {category}: "
                    f"{result['status']} frames={result.get('frames', '-')} "
                    f"seqs={result.get('sequences', '-')} "
                    f"size={result.get('bytes', 0) / 1024**2:.1f}MB "
                    f"time={result.get('elapsed_s', 0):.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "category": category,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                print(f"[{idx:02d}/{len(categories):02d}] {category}: ERROR {exc}", flush=True)

    report_path = out_dir / "build_report.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report:
        for item in sorted(results + failures, key=lambda x: x["category"]):
            report.write(json.dumps(item, sort_keys=True) + "\n")

    elapsed = time.perf_counter() - t0
    built = sum(1 for item in results if item["status"] == "built")
    skipped = sum(1 for item in results if item["status"] == "skipped")
    total_bytes = sum(int(item.get("bytes", 0)) for item in results)
    print(
        f"done built={built} skipped={skipped} failures={len(failures)} "
        f"size={total_bytes / 1024**3:.2f}GB elapsed={elapsed:.1f}s report={report_path}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
