#!/usr/bin/env python3
"""Generate h5 chunk index for all ScanNetPP scenes via COS SDK.

Workflow per scene:
1. Download precomputed.h5 from COS to local tmpdir
2. Extract chunk info (fast, <300ms)
3. Save pkl locally
4. Upload pkl to COS via coscli or SDK
5. Delete local h5

Usage:
    python generate_scannetpp_h5_index.py --workers 8
"""
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.sample_stage import SampleLocalStager, SampleStageConfig


def build_chunk_index(h5_path: Path) -> dict[str, list[tuple[int, int]]]:
    index = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            ds = f[key]
            if ds.chunks is None:
                continue
            n = ds.id.get_num_chunks()
            index[key] = [(ds.id.get_chunk_info(i).byte_offset, ds.id.get_chunk_info(i).size) for i in range(n)]
    return index


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


def list_scannetpp_scenes(client, bucket):
    """List all scannetpp scenes that have precomputed.h5."""
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


def process_one_scene(scene_name: str, tmpdir: str, client, bucket, force: bool = False):
    """Download h5, generate index, upload pkl, cleanup."""
    h5_key = f"hdu_datasets/scannetpp/data/{scene_name}/precomputed.h5"
    pkl_key = f"hdu_datasets/scannetpp/data/{scene_name}/precomputed.h5_chunk_index.pkl"

    # Check if pkl already exists on COS
    if not force:
        try:
            client.head_object(Bucket=bucket, Key=pkl_key)
            return "skip"  # Already exists
        except Exception:
            pass

    # Check if h5 exists
    try:
        client.head_object(Bucket=bucket, Key=h5_key)
    except Exception:
        return "no_h5"

    local_h5 = Path(tmpdir) / f"{scene_name}.h5"
    local_pkl = Path(tmpdir) / f"{scene_name}.pkl"

    try:
        # Download h5
        resp = client.get_object(Bucket=bucket, Key=h5_key)
        with open(local_h5, "wb") as f:
            stream = resp["Body"].get_raw_stream()
            while True:
                chunk = stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        # Generate index
        index = build_chunk_index(local_h5)
        with open(local_pkl, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Upload pkl to COS
        client.put_object_from_local_file(
            Bucket=bucket,
            Key=pkl_key,
            LocalFilePath=str(local_pkl),
        )
        return "ok"
    except Exception as e:
        return f"error: {e}"
    finally:
        local_h5.unlink(missing_ok=True)
        local_pkl.unlink(missing_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--test", action="store_true", help="Test with 3 scenes only")
    parser.add_argument("--force", action="store_true", help="Overwrite existing pkl files")
    args = parser.parse_args()

    client, bucket = get_cos_client()

    print("Listing scannetpp scenes...")
    scenes = list_scannetpp_scenes(client, bucket)
    print(f"Found {len(scenes)} scenes")

    if args.test:
        scenes = scenes[:3]
        print(f"Test mode: processing {len(scenes)} scenes")

    tmpdir = tempfile.mkdtemp(prefix="scannetpp_h5_index_", dir="/data1/zbf/scannetpp_index_tmp")
    print(f"Temp dir: {tmpdir}")

    t0 = time.time()
    done = 0
    skipped = 0
    errors = 0
    no_h5 = 0

    # Use per-thread COS clients to avoid thread-safety issues
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
        return scene_name, process_one_scene(scene_name, tmpdir, c, bucket, force=args.force)

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
                print(f"  ERROR {scene_name}: {result}")

            total = done + skipped + errors + no_h5
            if total % 20 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(scenes) - total) / rate if rate > 0 else 0
                print(f"  [{total}/{len(scenes)}] done={done} skip={skipped} err={errors} no_h5={no_h5} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.0f}s")
    print(f"  Generated: {done}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  No h5 file: {no_h5}")
    print(f"  Errors: {errors}")

    # Cleanup
    os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
