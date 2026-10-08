#!/usr/bin/env python3
"""Generate h5 chunk index files for COS Range-request staging.

For each h5 file, produces a companion .h5_chunk_index.pkl containing:
    {dataset_key: [(byte_offset, byte_size), ...]}  # one entry per frame

This allows staging to fetch only the needed frame chunks via COS Range requests
instead of downloading the entire h5 file (300-760MB for scannetpp, 46MB for kubric).

Usage:
    # Generate index for a single file
    python generate_h5_chunk_index.py /data_cos/hdu_datasets/scannetpp/data/7e7d2e8640/precomputed.h5

    # Generate for all scannetpp scenes
    python generate_h5_chunk_index.py --dataset scannetpp --root /data_cos/hdu_datasets/scannetpp/data

    # Generate for all kubric sequences
    python generate_h5_chunk_index.py --dataset kubric --root /data_cos/hdu_datasets/Kubric
"""

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py


def build_chunk_index(h5_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Extract per-frame chunk offsets from an h5 file.

    Returns {key: {"offsets": [(byte_offset, byte_size), ...], "dtype": str, "chunk_shape": tuple}}
    for each chunked dataset.
    """
    index: dict[str, dict] = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            ds = f[key]
            if ds.chunks is None:
                # Store scalar values directly
                index[key] = {"scalar": True, "value": ds[()].tolist() if hasattr(ds[()], 'tolist') else ds[()]}
                continue
            num_chunks = ds.id.get_num_chunks()
            offsets = []
            for i in range(num_chunks):
                info = ds.id.get_chunk_info(i)
                offsets.append((info.byte_offset, info.size))
            index[key] = {
                "offsets": offsets,
                "dtype": str(ds.dtype),
                "chunk_shape": ds.chunks,
                "shape": ds.shape,
                "compression": ds.compression,
            }
    return index


def generate_index_for_file(h5_path: Path, output_path: Path | None = None) -> Path:
    """Generate and save chunk index for a single h5 file."""
    if output_path is None:
        output_path = h5_path.with_name(h5_path.name + "_chunk_index.pkl")

    index = build_chunk_index(h5_path)
    with open(output_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def process_scannetpp(root: Path, workers: int = 8):
    """Generate chunk indices for all scannetpp precomputed.h5 files."""
    h5_files = sorted(root.glob("*/precomputed.h5"))
    print(f"Found {len(h5_files)} scannetpp h5 files")

    done = 0
    errors = 0
    t0 = time.time()

    def process_one(h5_path: Path):
        idx_path = h5_path.with_name("precomputed.h5_chunk_index.pkl")
        if idx_path.exists():
            return "skip"
        try:
            generate_index_for_file(h5_path, idx_path)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, p): p for p in h5_files}
        for fut in as_completed(futures):
            result = fut.result()
            if result == "ok":
                done += 1
            elif result == "skip":
                pass
            else:
                errors += 1
                print(f"  ERROR {futures[fut].parent.name}: {result}")
            total = done + errors
            if total % 50 == 0:
                print(f"  Progress: {total}/{len(h5_files)} ({time.time()-t0:.1f}s)")

    print(f"Done: {done} generated, {errors} errors, {time.time()-t0:.1f}s")


def process_kubric(root: Path, workers: int = 8):
    """Generate chunk indices for all kubric h5 files."""
    h5_files = sorted(root.glob("*/*.h5"))
    print(f"Found {len(h5_files)} kubric h5 files")

    done = 0
    errors = 0
    t0 = time.time()

    def process_one(h5_path: Path):
        idx_path = h5_path.with_name(h5_path.name + "_chunk_index.pkl")
        if idx_path.exists():
            return "skip"
        try:
            generate_index_for_file(h5_path, idx_path)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, p): p for p in h5_files}
        for fut in as_completed(futures):
            result = fut.result()
            if result == "ok":
                done += 1
            elif result == "skip":
                pass
            else:
                errors += 1
            total = done + errors
            if total % 100 == 0:
                print(f"  Progress: {total}/{len(h5_files)} ({time.time()-t0:.1f}s)")

    print(f"Done: {done} generated, {errors} errors, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="Single h5 file to index")
    parser.add_argument("--dataset", choices=["scannetpp", "kubric"])
    parser.add_argument("--root", type=str)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.path:
        p = Path(args.path)
        out = generate_index_for_file(p)
        idx = pickle.loads(out.read_bytes())
        print(f"Generated: {out}")
        print(f"Keys: {list(idx.keys())}")
        for k, v in idx.items():
            print(f"  {k}: {len(v)} chunks, first=({v[0][0]}, {v[0][1]})")
    elif args.dataset == "scannetpp":
        process_scannetpp(Path(args.root), workers=args.workers)
    elif args.dataset == "kubric":
        process_kubric(Path(args.root), workers=args.workers)
    else:
        parser.print_help()
