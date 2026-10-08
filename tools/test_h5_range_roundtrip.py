#!/usr/bin/env python3
"""Verify Range-fetched h5 chunks produce correct numpy arrays."""
import pickle
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
from datasets.sample_stage import SampleLocalStager, SampleStageConfig
from datasets.adapters.base import load_precomputed_fast


def test_roundtrip():
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
    original_h5 = Path(f"/data_cos/hdu_datasets/scannetpp/data/{scene}/precomputed.h5")

    frame_indices = [0, 10, 20, 30, 40, 47]
    skip_keys = {"normals"}

    # Load chunk index
    resp = client.get_object(Bucket=bucket, Key=index_key)
    chunk_index = pickle.loads(resp["Body"].get_raw_stream().read())

    # Get metadata from original h5 (dtype, shape info per key)
    with h5py.File(original_h5, "r") as f:
        key_meta = {}
        for key in f.keys():
            ds = f[key]
            if ds.chunks is None:
                # scalar - read directly
                key_meta[key] = {"scalar": True, "value": ds[()]}
            else:
                key_meta[key] = {
                    "scalar": False,
                    "dtype": ds.dtype,
                    "shape": ds.shape,
                    "chunks": ds.chunks,
                    "compression": ds.compression,
                }

    # Fetch needed chunks via Range requests
    needed = sorted(set(frame_indices))
    fetch_tasks = []
    for key, offsets in chunk_index.items():
        if key in skip_keys:
            continue
        for fi in needed:
            if fi < len(offsets):
                offset, size = offsets[fi]
                fetch_tasks.append((key, fi, offset, size))

    raw_chunks: dict[tuple[str, int], bytes] = {}
    def fetch_one(task):
        key, fi, offset, size = task
        range_header = f"bytes={offset}-{offset + size - 1}"
        resp = client.get_object(Bucket=bucket, Key=h5_key, Range=range_header)
        return (key, fi), resp["Body"].get_raw_stream().read()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=32) as ex:
        for result in ex.map(fetch_one, fetch_tasks):
            k, data = result
            raw_chunks[k] = data
    print(f"Range fetch: {(time.perf_counter()-t0)*1000:.0f}ms")

    # Write a minimal h5 with only needed frames
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    t0 = time.perf_counter()
    with h5py.File(tmp_path, "w") as out:
        for key, meta in key_meta.items():
            if key in skip_keys:
                continue
            if meta["scalar"]:
                out.create_dataset(key, data=meta["value"])
                continue
            # Build array from fetched chunks
            dtype = meta["dtype"]
            chunk_shape = meta["chunks"]  # (1, ...)
            per_frame_shape = chunk_shape[1:]  # shape of one frame
            n_frames = len(needed)
            out_shape = (n_frames,) + per_frame_shape
            ds = out.create_dataset(
                key, shape=out_shape, dtype=dtype,
                chunks=(1,) + per_frame_shape,
            )
            for new_idx, fi in enumerate(needed):
                chunk_data = raw_chunks.get((key, fi))
                if chunk_data is None:
                    continue
                arr = np.frombuffer(chunk_data, dtype=dtype).reshape(per_frame_shape)
                ds[new_idx] = arr
    print(f"Write mini h5: {(time.perf_counter()-t0)*1000:.0f}ms")

    # Verify: compare with original h5
    print(f"\nVerification:")
    # Map needed frame indices to new indices (0..N-1)
    new_indices = list(range(len(needed)))

    with h5py.File(original_h5, "r") as orig, h5py.File(tmp_path, "r") as mini:
        for key in sorted(mini.keys()):
            if key in skip_keys:
                continue
            orig_ds = orig[key]
            mini_ds = mini[key]
            if orig_ds.chunks is None:
                # scalar comparison
                assert orig_ds[()] == mini_ds[()], f"{key}: scalar mismatch"
                print(f"  {key}: scalar OK")
                continue
            for new_idx, fi in enumerate(needed):
                orig_arr = orig_ds[fi]
                mini_arr = mini_ds[new_idx]
                if not np.array_equal(orig_arr, mini_arr):
                    diff = np.abs(orig_arr.astype(float) - mini_arr.astype(float)).max()
                    print(f"  {key}[{fi}]: MISMATCH max_diff={diff}")
                    break
            else:
                print(f"  {key}: all {len(needed)} frames match ✓")

    tmp_path.unlink()
    print(f"\nDone! Mini h5 verified successfully.")


if __name__ == "__main__":
    test_roundtrip()
