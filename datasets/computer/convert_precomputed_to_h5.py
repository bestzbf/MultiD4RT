#!/usr/bin/env python3
"""
Convert precomputed.npz → precomputed.h5 for fast per-frame random access.

npz uses zlib compression which requires full decompression to access any frame.
h5 with LZF compression + per-frame chunking allows O(1) frame access.

Typical speedup: 200s → <1s per clip load for ScanNet/VirtualKitti.

Usage:
    cd /data2/d4rt/code/datasets/computer

    python convert_precomputed_to_h5.py --root /data2/d4rt/datasets/scannet/scannet --workers 1
    python convert_precomputed_to_h5.py --root /data2/d4rt/datasets/BlendedMVS --workers 4
    python convert_precomputed_to_h5.py --root /data2/d4rt/datasets/MVS-Synth/GTAV_1080 --workers 4
    python convert_precomputed_to_h5.py --root /data2/d4rt/datasets/TartanAir --workers 2
    python convert_precomputed_to_h5.py --root /data2/d4rt/datasets/VirtualKitti --workers 4

Notes:
    - ScanNet files can be 3 GB each; use --workers 1 to avoid OOM.
    - Existing .h5 files are skipped unless --overwrite is passed.
    - Output: precomputed.h5 alongside each precomputed.npz.
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


PER_POINT_KEYS = {
    "static_points_world",
    "static_point_ref_frames",
    "static_point_ref_uv",
    "static_point_ref_depth",
}


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
                if key not in PER_POINT_KEYS and arr.ndim >= 2 and arr.shape[0] > 1:
                    # Per-frame chunking: one frame per chunk → O(1) random access.
                    # No compression: write speed 8x faster than lzf, read avoids
                    # CPU decompression overhead. Size penalty ~8% vs lzf.
                    chunks = (1,) + arr.shape[1:]
                    f.create_dataset(key, data=arr, chunks=chunks)
                else:
                    # Scalar or 1-D metadata: store as-is
                    f.create_dataset(key, data=arr)
        elapsed = time.perf_counter() - t0
        size_mb = os.path.getsize(h5_path) / 1e6
        npz_mb  = os.path.getsize(npz_path) / 1e6
        return f"OK    {npz_path.parent.name}  {elapsed:.1f}s  npz={npz_mb:.0f}MB → h5={size_mb:.0f}MB"
    except Exception as e:
        if h5_path.exists():
            h5_path.unlink()
        return f"ERR   {npz_path.parent.name}: {e}\n{traceback.format_exc()}"


def _convert_one_str(args: tuple) -> str:
    """Wrapper for multiprocessing (Path must be serialisable)."""
    return convert_one(Path(args))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert precomputed.npz → precomputed.h5"
    )
    parser.add_argument('--root', required=True, help="Dataset root directory")
    parser.add_argument(
        '--workers', type=int, default=2,
        help="Parallel workers (default 2; use 1 for large files to avoid OOM)"
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help="Re-convert even if .h5 already exists"
    )
    args = parser.parse_args()

    root = Path(args.root)
    npz_files = sorted(root.rglob('precomputed.npz'))
    if not npz_files:
        print(f"No precomputed.npz found under {root}")
        return

    if args.overwrite:
        for p in npz_files:
            h5 = p.with_suffix('.h5')
            if h5.exists():
                h5.unlink()

    total_gb = sum(os.path.getsize(p) for p in npz_files) / 1e9
    print(f"Found {len(npz_files)} precomputed.npz files  ({total_gb:.1f} GB total)")
    print(f"Workers : {args.workers}")
    print(f"Output  : precomputed.h5 alongside each precomputed.npz")
    print()

    t_start = time.perf_counter()
    done = 0
    n = len(npz_files)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(
                _convert_one_str, [str(p) for p in npz_files]
            ):
                done += 1
                print(f"[{done:4d}/{n}] {result}", flush=True)
    else:
        for npz in npz_files:
            result = convert_one(npz)
            done += 1
            print(f"[{done:4d}/{n}] {result}", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
