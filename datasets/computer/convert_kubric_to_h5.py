#!/usr/bin/env python3
"""
Convert Kubric .npy annotation → .h5 for fast per-frame random access.

Original .npy (pickle dict) contains:
  coords        [N, T, 2]  float32   2D track coordinates
  coords_depth  [N, T]     float32   depth at track points
  visibility    [N, T]     bool
  depth         [T, H, W, 1] uint16  (large; not stored in h5)
  segmentations [T, H, W, 1] uint8   (not stored in h5)
  metadata      dict
  camera        dict

We store only the fields used by KubricAdapter.load_clip():
  trajs_2d      [T, N, 2]  (transposed from coords)
  coords_depth  [T, N]     (transposed)
  visibility    [T, N]     (transposed)

Camera params come from _with_rank.npz which is tiny (0.1MB), no need to convert.

Speedup: 0.5-1.3s → ~0.05s per clip.

Usage:
    cd /data2/d4rt/code/datasets/computer
    python convert_kubric_to_h5.py --root /data2/d4rt/datasets/kubric --workers 8
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import h5py
import numpy as np


CompressionMode = Literal["lzf", "none"]


def convert_one(npy_path: Path, compression: CompressionMode = "lzf") -> str:
    h5_path = npy_path.with_suffix('.h5')
    if h5_path.exists():
        return f"SKIP  {npy_path.stem}"

    t0 = time.perf_counter()
    try:
        ann = np.load(npy_path, allow_pickle=True).item()

        coords_nt2 = np.asarray(ann["coords"], dtype=np.float32)            # [N,T,2]
        coords_depth_nt = np.asarray(ann["coords_depth"], dtype=np.float32) # [N,T]
        visibility_nt   = np.asarray(ann["visibility"], dtype=bool)         # [N,T]

        # Transpose to [T,N,*] for per-frame chunking
        trajs_2d     = np.ascontiguousarray(np.transpose(coords_nt2, (1, 0, 2)))   # [T,N,2]
        coords_depth = np.ascontiguousarray(np.transpose(coords_depth_nt, (1, 0))) # [T,N]
        visibility   = np.ascontiguousarray(np.transpose(visibility_nt, (1, 0)))   # [T,N]

        T = trajs_2d.shape[0]
        h5_compression = None if compression == "none" else compression

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('trajs_2d',     data=trajs_2d,
                             chunks=(1, trajs_2d.shape[1], 2), compression=h5_compression)
            f.create_dataset('coords_depth', data=coords_depth,
                             chunks=(1, coords_depth.shape[1]), compression=h5_compression)
            f.create_dataset('visibility',   data=visibility,
                             chunks=(1, visibility.shape[1]), compression=h5_compression)
            f.create_dataset('num_frames', data=np.int64(T))

        elapsed = time.perf_counter() - t0
        size_mb = os.path.getsize(h5_path) / 1e6
        npy_mb  = os.path.getsize(npy_path) / 1e6
        return f"OK    {npy_path.parent.name}/{npy_path.stem}  {elapsed:.1f}s  npy={npy_mb:.0f}MB → h5={size_mb:.0f}MB"
    except Exception as e:
        if h5_path.exists():
            h5_path.unlink()
        return f"ERR   {npy_path.stem}: {e}\n{traceback.format_exc()}"


def _worker(args: tuple[str, CompressionMode]) -> str:
    path_str, compression = args
    return convert_one(Path(path_str), compression=compression)


def _progress_iter(iterable, total: int):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, unit="file", dynamic_ncols=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kubric .npy → .h5")
    parser.add_argument('--root', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunksize', type=int, default=4)
    parser.add_argument('--compression', choices=('lzf', 'none'), default='lzf')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    # Only annotation .npy files: named exactly like their parent sequence dir
    # (e.g. kubric/0001/0001/0001.npy), not _trajs_2d.npy / _visibility.npy / depths/*.npy
    npy_files = sorted(
        p for p in root.rglob('*.npy')
        if 'depths' not in p.parts
        and not p.stem.endswith('_trajs_2d')
        and not p.stem.endswith('_visibility')
        and p.stem == p.parent.name   # file stem matches parent directory name
    )
    if not npy_files:
        print(f"No annotation .npy found under {root}")
        return

    if args.overwrite:
        for p in npy_files:
            h5 = p.with_suffix('.h5')
            if h5.exists():
                h5.unlink()

    total_gb = sum(os.path.getsize(p) for p in npy_files) / 1e9
    print(f"Found {len(npy_files)} annotation .npy files  ({total_gb:.1f} GB total)")
    print(f"Workers: {args.workers}")
    print(f"Chunksize: {args.chunksize}")
    print(f"Compression: {args.compression}")
    print()

    t_start = time.perf_counter()
    n = len(npy_files)
    done = 0
    worker_args = [(str(p), args.compression) for p in npy_files]

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.imap_unordered(_worker, worker_args, chunksize=max(1, args.chunksize))
            for result in _progress_iter(results, n):
                done += 1
                print(f"[{done:4d}/{n}] {result}", flush=True)
    else:
        for npy in npy_files:
            done += 1
            print(f"[{done:4d}/{n}] {convert_one(npy, compression=args.compression)}", flush=True)

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s")


if __name__ == '__main__':
    main()
