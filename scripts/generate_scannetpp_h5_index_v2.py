#!/usr/bin/env python3
"""Generate h5 chunk index WITHOUT downloading the full h5 file.

Uses h5py with a custom file-like wrapper that fetches bytes on-demand via
COS SDK Range requests. h5py only reads the metadata B-tree (a few KB) to
extract chunk offsets — chunk data itself is never downloaded.

This avoids downloading 300-760MB of h5 data when we only need ~70KB of pkl.
"""
import argparse
import io
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.sample_stage import SampleLocalStager, SampleStageConfig


class COSRangeFile(io.RawIOBase):
    """File-like object that reads from COS via Range requests.

    Prefetches file header (first N MB) in one request to cover most
    HDF5 metadata (superblock, B-trees for small files).
    """

    HEADER_PREFETCH_BYTES = 8 * 1024 * 1024  # 8MB prefetch

    def __init__(self, client, bucket: str, key: str, file_size: int):
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size = file_size
        self._pos = 0
        # Prefetch header
        prefetch_size = min(self.HEADER_PREFETCH_BYTES, file_size)
        resp = client.get_object(
            Bucket=bucket, Key=key,
            Range=f"bytes=0-{prefetch_size - 1}",
        )
        self._header_cache = resp["Body"].get_raw_stream().read()
        self._header_end = len(self._header_cache)
        # Also prefetch tail (last 1MB) for files that store metadata at end
        if file_size > self._header_end + 1024 * 1024:
            tail_start = file_size - 1024 * 1024
            resp = client.get_object(
                Bucket=bucket, Key=key,
                Range=f"bytes={tail_start}-{file_size - 1}",
            )
            self._tail_cache = resp["Body"].get_raw_stream().read()
            self._tail_start = tail_start
        else:
            self._tail_cache = b""
            self._tail_start = file_size

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if self._pos >= self._size:
            return b""
        if n < 0 or self._pos + n > self._size:
            n = self._size - self._pos
        if n == 0:
            return b""
        # Serve from header cache if possible
        if self._pos + n <= self._header_end:
            data = self._header_cache[self._pos:self._pos + n]
            self._pos += n
            return data
        # Serve from tail cache if possible
        if self._pos >= self._tail_start and self._pos + n <= self._size:
            rel_start = self._pos - self._tail_start
            data = self._tail_cache[rel_start:rel_start + n]
            self._pos += n
            return data
        # Fall back to Range request
        range_header = f"bytes={self._pos}-{self._pos + n - 1}"
        resp = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
            Range=range_header,
        )
        data = resp["Body"].get_raw_stream().read()
        self._pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def get_cos_client():
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
    return stager._get_client(), config.bucket


def build_chunk_index_from_cos(client, bucket: str, h5_key: str) -> dict:
    """Build chunk index without downloading the full h5 file.

    Strategy: HDF5 files have a superblock at the start and B-tree metadata
    scattered throughout. h5py needs random access to read structure. We use
    a file-like with seek+read that serves bytes via COS Range requests.
    h5py's 'rdcc' cache minimizes repeated reads.

    For HDF5 format: the metadata footprint is typically <1MB even for large
    files, so only a tiny fraction of the file is actually downloaded.
    """
    head = client.head_object(Bucket=bucket, Key=h5_key)
    file_size = int(head["Content-Length"])
    f = COSRangeFile(client, bucket, h5_key, file_size)
    # BufferedReader wraps our RawIOBase so h5py can seek/read efficiently
    buffered = io.BufferedReader(f, buffer_size=64 * 1024)
    index = {}
    with h5py.File(buffered, "r") as hf:
        for key in hf.keys():
            ds = hf[key]
            if ds.chunks is None:
                value = ds[()]
                index[key] = {
                    "scalar": True,
                    "value": value.tolist() if hasattr(value, "tolist") else int(value),
                }
                continue
            n = ds.id.get_num_chunks()
            offsets = [(ds.id.get_chunk_info(i).byte_offset, ds.id.get_chunk_info(i).size) for i in range(n)]
            index[key] = {
                "offsets": offsets,
                "dtype": str(ds.dtype),
                "chunk_shape": ds.chunks,
                "shape": ds.shape,
                "compression": ds.compression,
            }
    return index


def list_scannetpp_scenes(client, bucket):
    prefix = "hdu_datasets/scannetpp/data/"
    scenes = set()
    marker = ""
    while True:
        resp = client.list_objects(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter="/",
            Marker=marker,
            MaxKeys=1000,
        )
        for cp in resp.get("CommonPrefixes", []):
            scene_name = cp["Prefix"].rstrip("/").split("/")[-1]
            scenes.add(scene_name)
        if resp.get("IsTruncated") == "true":
            marker = resp.get("NextMarker", "")
        else:
            break
    return sorted(scenes)


def process_one_scene(scene_name: str, client, bucket: str, force: bool = False) -> str:
    h5_key = f"hdu_datasets/scannetpp/data/{scene_name}/precomputed.h5"
    pkl_key = f"hdu_datasets/scannetpp/data/{scene_name}/precomputed.h5_chunk_index.pkl"

    if not force:
        try:
            client.head_object(Bucket=bucket, Key=pkl_key)
            return "skip"
        except Exception:
            pass

    try:
        client.head_object(Bucket=bucket, Key=h5_key)
    except Exception:
        return "no_h5"

    try:
        index = build_chunk_index_from_cos(client, bucket, h5_key)
        pkl_bytes = pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL)
        client.put_object(Bucket=bucket, Key=pkl_key, Body=pkl_bytes)
        return "ok"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client, bucket = get_cos_client()

    print("Listing scannetpp scenes...")
    scenes = list_scannetpp_scenes(client, bucket)
    print(f"Found {len(scenes)} scenes")

    if args.test:
        scenes = scenes[:5]
        print(f"Test mode: {len(scenes)} scenes")

    thread_local = threading.local()

    def get_thread_client():
        c = getattr(thread_local, "client", None)
        if c is None:
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
            c = stager._get_client()
            thread_local.client = c
        return c

    def worker(scene_name):
        c = get_thread_client()
        return scene_name, process_one_scene(scene_name, c, bucket, force=args.force)

    t0 = time.time()
    done = skipped = errors = no_h5 = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, s): s for s in scenes}
        for fut in as_completed(futures):
            scene_name, result = fut.result()
            if result == "ok":
                done += 1
            elif result == "skip":
                skipped += 1
            elif result == "no_h5":
                no_h5 += 1
            else:
                errors += 1
                print(f"  ERROR {scene_name}: {result}", flush=True)

            total = done + skipped + errors + no_h5
            if total % 20 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(scenes) - total) / rate if rate > 0 else 0
                print(f"  [{total}/{len(scenes)}] done={done} skip={skipped} err={errors} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.0f}s: done={done} skip={skipped} err={errors} no_h5={no_h5}")


if __name__ == "__main__":
    main()
