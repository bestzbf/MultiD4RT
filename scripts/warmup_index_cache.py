#!/usr/bin/env python
"""
Warmup index cache for all datasets in a config file.

Usage:
    python scripts/warmup_index_cache.py --config configs/mixture_5datasets_cos_planned.yaml

This script creates all dataset adapters (with cache_dir + index_workers),
triggering _build_index for each one.  Adapters are created in parallel
threads so multiple datasets warm up simultaneously.

After running once, subsequent training starts will load the cached .pkl
files in < 0.01s instead of spending minutes/hours on COS stat() calls.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from datasets.registry import create_adapter


def warmup_one_dataset(
    ds_config: dict,
    split: str,
    index_cache_dir: str,
    index_workers: int,
) -> tuple[str, float, int, str | None]:
    """Create an adapter, which triggers _build_index if cache is cold.

    Returns:
        (dataset_name, elapsed_seconds, num_sequences, error_message_or_None)
    """
    name = ds_config["name"]
    root = ds_config["root"]
    adapter_kwargs = dict(ds_config.get("adapter_kwargs", {}))

    t0 = time.time()
    try:
        adapter = create_adapter(
            name=name,
            root=root,
            split=split,
            cache_dir=index_cache_dir,
            index_workers=index_workers,
            **adapter_kwargs,
        )
        elapsed = time.time() - t0
        n_seq = len(adapter.list_sequences()) if hasattr(adapter, "list_sequences") else len(adapter)
        return name, elapsed, n_seq, None
    except Exception as e:
        elapsed = time.time() - t0
        return name, elapsed, 0, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Warmup index cache for all datasets in a YAML config."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to mixture YAML config file",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--index-cache-dir", default=None,
        help="Override index_cache_dir from config",
    )
    parser.add_argument(
        "--index-workers", type=int, default=None,
        help="Override index_workers from config",
    )
    parser.add_argument(
        "--parallel-datasets", type=int, default=4,
        help="Number of datasets to warm up in parallel (default: 4)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", [])
    if not datasets:
        print("No datasets found in config.")
        return

    index_cache_dir = args.index_cache_dir or config.get("index_cache_dir")
    if not index_cache_dir:
        print("ERROR: No index_cache_dir specified (neither in config nor --index-cache-dir)")
        sys.exit(1)

    index_workers = args.index_workers or config.get("index_workers", 8)

    # Ensure cache dir exists
    Path(index_cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Index Cache Warmup")
    print(f"{'=' * 60}")
    print(f"Config:          {args.config}")
    print(f"Split:           {args.split}")
    print(f"Cache dir:       {index_cache_dir}")
    print(f"Index workers:   {index_workers} (per dataset)")
    print(f"Parallel:        {args.parallel_datasets} datasets")
    print(f"Datasets:        {len(datasets)}")
    for ds in datasets:
        print(f"  - {ds['name']:20s}  {ds['root']}")
    print(f"{'=' * 60}")
    print()

    total_t0 = time.time()

    n_parallel = min(args.parallel_datasets, len(datasets))
    results = []

    if n_parallel > 1:
        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            futures = {}
            for ds in datasets:
                fut = executor.submit(
                    warmup_one_dataset,
                    ds, args.split, index_cache_dir, index_workers,
                )
                futures[fut] = ds["name"]

            for fut in as_completed(futures):
                name, elapsed, n_seq, err = fut.result()
                if err:
                    print(f"  FAIL  {name:20s}  {elapsed:7.1f}s  error: {err}")
                else:
                    print(f"  OK    {name:20s}  {elapsed:7.1f}s  {n_seq} sequences")
                results.append((name, elapsed, n_seq, err))
    else:
        for ds in datasets:
            name, elapsed, n_seq, err = warmup_one_dataset(
                ds, args.split, index_cache_dir, index_workers,
            )
            if err:
                print(f"  FAIL  {name:20s}  {elapsed:7.1f}s  error: {err}")
            else:
                print(f"  OK    {name:20s}  {elapsed:7.1f}s  {n_seq} sequences")
            results.append((name, elapsed, n_seq, err))

    total_elapsed = time.time() - total_t0
    n_ok = sum(1 for _, _, _, e in results if e is None)
    n_fail = sum(1 for _, _, _, e in results if e is not None)

    print()
    print(f"{'=' * 60}")
    print(f"Done in {total_elapsed:.1f}s  |  {n_ok} OK, {n_fail} failed")
    print(f"Cache files in: {index_cache_dir}")
    print(f"{'=' * 60}")

    # List cache files
    cache_path = Path(index_cache_dir)
    if cache_path.exists():
        pkl_files = sorted(cache_path.glob("*.pkl"))
        if pkl_files:
            print()
            print("Cached index files:")
            for p in pkl_files:
                size_kb = p.stat().st_size / 1024
                print(f"  {p.name:50s}  {size_kb:8.1f} KB")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
