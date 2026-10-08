#!/usr/bin/env python
"""Build the fast Co3D frame-annotation cache used by Co3Dv2Adapter."""

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


def _unicode_array(values: list[str]) -> np.ndarray:
    width = max(1, max((len(v) for v in values), default=1))
    return np.asarray(values, dtype=f"<U{width}")


def _discover_categories(root: Path) -> list[str]:
    return sorted(
        path.parent.name
        for path in root.glob("*/frame_annotations.jgz")
        if path.is_file()
    )


def _load_frame_annotations(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as fh:
        return json.load(fh)


def _build_one(root: Path, output_dir: Path, category: str, force: bool) -> tuple[str, int, float, Path, str | None]:
    src = root / category / "frame_annotations.jgz"
    dst = output_dir / f"frame_anno_{category}_v1.npz"
    if dst.is_file() and not force:
        return category, -1, 0.0, dst, None
    if not src.is_file():
        return category, 0, 0.0, dst, f"missing {src}"

    t0 = time.time()
    annotations = _load_frame_annotations(src)
    annotations.sort(key=lambda item: (str(item["sequence_name"]), int(item["frame_number"])))

    seq_names: list[str] = []
    offsets: list[int] = [0]
    fns: list[int] = []
    rotations: list[Any] = []
    translations: list[Any] = []
    focals: list[Any] = []
    principal_points: list[Any] = []
    image_sizes: list[Any] = []
    depth_scales: list[float] = []
    image_stems: list[str] = []

    current_seq: str | None = None
    for anno in annotations:
        seq = str(anno["sequence_name"])
        if seq != current_seq:
            if current_seq is not None:
                offsets.append(len(fns))
            seq_names.append(seq)
            current_seq = seq

        frame_number = int(anno["frame_number"])
        image = anno["image"]
        depth = anno["depth"]
        viewpoint = anno["viewpoint"]

        fns.append(frame_number)
        rotations.append(viewpoint["R"])
        translations.append(viewpoint["T"])
        focals.append(viewpoint["focal_length"])
        principal_points.append(viewpoint["principal_point"])
        image_sizes.append(image["size"])
        depth_scales.append(float(depth.get("scale_adjustment", 1.0)))
        image_stems.append(Path(str(image["path"])).stem or f"frame{frame_number:06d}")

    offsets.append(len(fns))

    tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            seq_names=_unicode_array(seq_names),
            offsets=np.asarray(offsets, dtype=np.int64),
            fns=np.asarray(fns, dtype=np.int32),
            R=np.asarray(rotations, dtype=np.float32),
            T=np.asarray(translations, dtype=np.float32),
            focal=np.asarray(focals, dtype=np.float32),
            pp=np.asarray(principal_points, dtype=np.float32),
            imgsz=np.asarray(image_sizes, dtype=np.int32),
            dscale=np.asarray(depth_scales, dtype=np.float32),
            img_stems=_unicode_array(image_stems),
        )
    os.replace(tmp, dst)
    return category, len(fns), time.time() - t0, dst, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Co3Dv2 dataset root")
    parser.add_argument("--cache-dir", required=True, help="Index cache root")
    parser.add_argument("--output-dir", default="", help="Override output directory")
    parser.add_argument("--categories", default="", help="Comma-separated category list")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.cache_dir) / "co3dv2_frame_anno"
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    if not categories:
        categories = _discover_categories(root)
    if not categories:
        raise SystemExit(f"No Co3D categories found under {root}")

    workers = max(1, min(int(args.workers), len(categories)))
    print(
        f"[build_co3dv2_frame_cache] root={root} output_dir={output_dir} "
        f"categories={len(categories)} workers={workers} force={args.force}",
        flush=True,
    )

    failures: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_build_one, root, output_dir, category, bool(args.force))
            for category in categories
        ]
        for fut in as_completed(futures):
            category, n_frames, elapsed, dst, err = fut.result()
            if err is not None:
                failures.append(f"{category}: {err}")
                print(f"[build_co3dv2_frame_cache] FAIL {category}: {err}", flush=True)
            elif n_frames < 0:
                print(f"[build_co3dv2_frame_cache] skip {category}: {dst}", flush=True)
            else:
                size_mb = dst.stat().st_size / 1024**2
                print(
                    f"[build_co3dv2_frame_cache] wrote {category}: "
                    f"frames={n_frames} size={size_mb:.1f}MB time={elapsed:.1f}s",
                    flush=True,
                )

    print(f"[build_co3dv2_frame_cache] done in {time.time() - t0:.1f}s", flush=True)
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
