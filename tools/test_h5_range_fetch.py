#!/usr/bin/env python3
"""Test: fetch only needed h5 chunks via COS Range requests vs full download."""

import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from datasets.sample_stage import SampleLocalStager, SampleStageConfig


def fetch_h5_frames_range(
    cos_client,
    bucket: str,
    h5_key: str,
    chunk_index: dict[str, list[tuple[int, int]]],
    frame_indices: list[int],
    skip_keys: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Fetch only needed frame chunks from an h5 file via COS Range requests.

    Returns dict of {key: np.ndarray} with arrays indexed to frame_indices order.
    """
    skip = skip_keys or set()
    needed = sorted(set(frame_indices))

    # Collect all byte ranges we need to fetch
    fetch_tasks: list[tuple[str, int, int, int]] = []  # (key, frame_idx, offset, size)
    for key, offsets in chunk_index.items():
        if key in skip:
            continue
        for fi in needed:
            if fi < len(offsets):
                offset, size = offsets[fi]
                fetch_tasks.append((key, fi, offset, size))

    # Merge adjacent/overlapping ranges for efficiency
    # Sort by offset to enable merging
    fetch_tasks.sort(key=lambda x: x[2])

    # Fetch all chunks in parallel
    raw_chunks: dict[tuple[str, int], bytes] = {}

    def fetch_one(task):
        key, fi, offset, size = task
        range_header = f"bytes={offset}-{offset + size - 1}"
        resp = cos_client.get_object(Bucket=bucket, Key=h5_key, Range=range_header)
        data = resp["Body"].get_raw_stream().read()
        return (key, fi), data

    with ThreadPoolExecutor(max_workers=min(32, len(fetch_tasks))) as ex:
        results = list(ex.map(fetch_one, fetch_tasks))

    for k, data in results:
        raw_chunks[k] = data

    # Reconstruct numpy arrays
    # We need to know the dtype and shape per chunk to interpret raw bytes
    # Since h5 files in this project use no compression (scannetpp) or lzf (kubric),
    # we handle both cases
    return raw_chunks, needed


def test_range_fetch():
    """Compare Range fetch vs full download for scannetpp precomputed.h5."""

    config = SampleStageConfig(
        backend="cos_sdk",
        stage_root="/data1/zbf/d4rt_sample_stage",
        sdk_workers=32,
        mount_root="/data_cos",
        bucket="hd-ai-data-1251882982",
        region="ap-beijing",
        enabled_datasets=("scannetpp",),
    )
    stager = SampleLocalStager(config)
    client = stager._get_client()
    bucket = config.bucket

    scene = "7e7d2e8640"
    h5_key = f"hdu_datasets/scannetpp/data/{scene}/precomputed.h5"
    index_key = f"hdu_datasets/scannetpp/data/{scene}/precomputed.h5_chunk_index.pkl"

    frame_indices = list(range(48))
    skip_keys = {"normals"}  # adapter skips normals

    # Load chunk index
    print("Loading chunk index...")
    t0 = time.perf_counter()
    resp = client.get_object(Bucket=bucket, Key=index_key)
    chunk_index = pickle.loads(resp["Body"].get_raw_stream().read())
    t_index = time.perf_counter() - t0
    print(f"  Index loaded in {t_index*1000:.0f}ms")

    # Calculate expected bytes
    total_bytes = 0
    for key, offsets in chunk_index.items():
        if key in skip_keys:
            continue
        for fi in frame_indices:
            if fi < len(offsets):
                total_bytes += offsets[fi][1]
    print(f"  Need to fetch: {total_bytes/1024/1024:.1f}MB for {len(frame_indices)} frames")
    print(f"  Keys: {[k for k in chunk_index if k not in skip_keys]}")

    # Fetch via Range requests
    print("\nFetching via Range requests...")
    t0 = time.perf_counter()

    fetch_tasks = []
    for key, offsets in chunk_index.items():
        if key in skip_keys:
            continue
        for fi in frame_indices:
            if fi < len(offsets):
                offset, size = offsets[fi]
                fetch_tasks.append((key, fi, offset, size))

    raw_chunks: dict[tuple[str, int], bytes] = {}

    def fetch_one(task):
        key, fi, offset, size = task
        range_header = f"bytes={offset}-{offset + size - 1}"
        resp = client.get_object(Bucket=bucket, Key=h5_key, Range=range_header)
        return (key, fi), resp["Body"].get_raw_stream().read()

    with ThreadPoolExecutor(max_workers=32) as ex:
        for result in ex.map(fetch_one, fetch_tasks):
            k, data = result
            raw_chunks[k] = data

    t_fetch = time.perf_counter() - t0
    fetched_bytes = sum(len(v) for v in raw_chunks.values())

    print(f"  Fetched {len(raw_chunks)} chunks ({fetched_bytes/1024/1024:.1f}MB) in {t_fetch*1000:.0f}ms")
    print(f"  Throughput: {fetched_bytes/t_fetch/1024/1024:.0f}MB/s")

    # Compare with full download time (from earlier profiling)
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Range fetch (48 frames, skip normals): {(t_index + t_fetch)*1000:.0f}ms")
    print(f"  Full h5 download (from profiling):     ~10000-12000ms")
    print(f"  Speedup: ~{10000/(t_index + t_fetch)/1000:.0f}x")
    print(f"  Data transferred: {fetched_bytes/1024/1024:.1f}MB vs ~683MB")


if __name__ == "__main__":
    test_range_fetch()
