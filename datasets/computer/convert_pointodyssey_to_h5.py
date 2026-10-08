#!/usr/bin/env python3
"""
Convert PointOdyssey anno.npz → anno.h5 for fast per-frame random access.

anno.npz keys: trajs_2d [T,N,2], trajs_3d [T,N,3], valids [T,N],
               visibs [T,N], intrinsics [T,3,3], extrinsics [T,4,4]

All are [T,...] so chunked HDF5 gives O(frames) random access vs full
zlib decompression for npz.

Typical speedup: 3-4s → <0.1s per clip.

Usage:
    cd /data2/d4rt/code/datasets/computer
    python convert_pointodyssey_to_h5.py --root /data2/d4rt/datasets/PointOdyssey --workers 4
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np


def convert_one(npz_path: Path) -> str:
    h5_path = npz_path.with_suffix('.h5')
    if h5_path.exists():
        return f"SKIP  {npz_path.parent.name}"

    t0 = time.perf_counter()
    try:
        data = np.load(npz_path, allow_pickle=True)
        with h5py.File(h5_path, 'w') as f:
            for key in data.files:
                arr = data[key]
                if arr.ndim >= 2 and arr.shape[0] > 1:
                    chunks = (1,) + arr.shape[1:]
                    f.create_dataset(key, data=arr, chunks=chunks, compression='lzf')
                else:
                    f.create_dataset(key, data=arr)
        elapsed = time.perf_counter() - t0
        size_mb = os.path.getsize(h5_path) / 1e6
        npz_mb  = os.path.getsize(npz_path) / 1e6
        return f"OK    {npz_path.parent.name}  {elapsed:.1f}s  npz={npz_mb:.0f}MB → h5={size_mb:.0f}MB"
    except Exception as e:
        if h5_path.exists():
            h5_path.unlink()
        return f"ERR   {npz_path.parent.name}: {e}\n{traceback.format_exc()}"


def _worker(path_str: str) -> str:
    return convert_one(Path(path_str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PointOdyssey anno.npz → anno.h5")
    parser.add_argument('--root', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    npz_files = sorted(root.rglob('anno.npz'))
    if not npz_files:
        print(f"No anno.npz found under {root}")
        return

    if args.overwrite:
        for p in npz_files:
            h5 = p.with_suffix('.h5')
            if h5.exists():
                h5.unlink()

    total_gb = sum(os.path.getsize(p) for p in npz_files) / 1e9
    print(f"Found {len(npz_files)} anno.npz files  ({total_gb:.1f} GB total)")
    print(f"Workers: {args.workers}")
    print()

    t_start = time.perf_counter()
    n = len(npz_files)
    done = 0

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(_worker, [str(p) for p in npz_files]):
                done += 1
                print(f"[{done:4d}/{n}] {result}", flush=True)
    else:
        for npz in npz_files:
            done += 1
            print(f"[{done:4d}/{n}] {convert_one(npz)}", flush=True)

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s")


if __name__ == '__main__':
    main()
