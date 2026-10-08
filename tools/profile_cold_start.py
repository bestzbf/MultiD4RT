#!/usr/bin/env python3
"""Profile cold-start ScanNetPP staging - find where the 10s delay comes from."""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.sample_stage import SampleLocalStager, SampleStageConfig


class FakeScanNetPPAdapter:
    def __init__(self, root: str):
        self.dataset_name = "scannetpp"
        self.root = Path(root)
        self.data_root = self.root
        self._scene_cache = {}
        self._staged_depth_map_tmp = None
        self._depth_chunk_cache = {}

    def _get_scene_data(self, scene_name: str):
        if scene_name in self._scene_cache:
            return self._scene_cache[scene_name]
        import numpy as np
        scene_dir = self.data_root / scene_name
        full_indices = np.arange(48, dtype=np.int32)
        data = {
            "scene_dir": scene_dir,
            "full_indices": full_indices,
            "frame_stems": [f"{i:06d}" for i in range(48)],
        }
        self._scene_cache[scene_name] = data
        return data


def profile_cold_start(scene_name: str):
    config = SampleStageConfig(
        backend="cos_sdk",
        stage_root="/data1/zbf/d4rt_sample_stage",
        sdk_workers=32,
        cache_max_bytes=100 * 1024**3,
        mount_root="/data_cos",
        bucket="hd-ai-data-1251882982",
        region="ap-beijing",
        enabled_datasets=("scannetpp",),
    )
    stager = SampleLocalStager(config)
    adapter = FakeScanNetPPAdapter(root="/data_cos/hdu_datasets/scannetpp/data")
    frame_indices = list(range(48))

    download_times: list[tuple[str, float, int]] = []
    evict_times: list[float] = []
    depth_stage_time = [0.0]

    original_download = stager._download_to_cache
    original_evict = stager._maybe_evict_cache
    original_depth_stage = stager._stage_scannetpp_depth

    def timed_download(src_path, cache_path, rel_key):
        t0 = time.perf_counter()
        r = original_download(src_path, cache_path, rel_key)
        e = time.perf_counter() - t0
        sz = cache_path.stat().st_size if cache_path.exists() else 0
        download_times.append((str(rel_key), e, sz))
        return r

    def timed_evict():
        t0 = time.perf_counter()
        r = original_evict()
        e = time.perf_counter() - t0
        evict_times.append(e)
        return r

    def timed_depth_stage(original_scene_dir, staged_scene_dir, frame_indices):
        t0 = time.perf_counter()
        r = original_depth_stage(original_scene_dir, staged_scene_dir, frame_indices)
        depth_stage_time[0] = time.perf_counter() - t0
        return r

    stager._download_to_cache = timed_download
    stager._maybe_evict_cache = timed_evict
    stager._stage_scannetpp_depth = timed_depth_stage

    print(f"\n{'='*80}")
    print(f"Cold-start staging test: scene={scene_name}")
    print(f"{'='*80}\n")

    overall_start = time.perf_counter()
    with stager.stage_sample(adapter, scene_name, frame_indices, sample_tag="cold") as _:
        overall_elapsed = time.perf_counter() - overall_start

    print(f"Total staging time: {overall_elapsed*1000:.0f}ms")
    print(f"Files downloaded: {len(download_times)}")
    print(f"Evict calls: {len(evict_times)}, total={sum(evict_times)*1000:.0f}ms, max={max(evict_times)*1000 if evict_times else 0:.0f}ms")
    print(f"Depth stage time: {depth_stage_time[0]*1000:.0f}ms")
    print()

    if download_times:
        total_download = sum(t for _, t, _ in download_times)
        total_bytes = sum(sz for _, _, sz in download_times)
        print(f"Download stats:")
        print(f"  Total download time (summed): {total_download*1000:.0f}ms")
        print(f"  Total bytes: {total_bytes/1024/1024:.1f}MB")
        print(f"  Avg throughput: {total_bytes/total_download/1024/1024:.1f}MB/s")
        print()

        # Group by file type
        by_type: dict[str, list[tuple[float, int]]] = {}
        for path, elapsed, sz in download_times:
            if path.endswith(".jpg"):
                key = "JPG frames"
            elif path.endswith(".h5"):
                key = "precomputed.h5"
            elif path.endswith(".txt"):
                key = "colmap .txt"
            elif path.endswith(".json"):
                key = "pose .json"
            elif path.endswith(".pkl"):
                key = "chunk_index.pkl"
            else:
                key = "other"
            by_type.setdefault(key, []).append((elapsed, sz))

        print(f"By file type:")
        for key, items in sorted(by_type.items(), key=lambda x: -sum(t for t, _ in x[1])):
            count = len(items)
            tot_ms = sum(t for t, _ in items) * 1000
            avg_ms = tot_ms / count
            max_ms = max(t for t, _ in items) * 1000
            tot_sz = sum(sz for _, sz in items) / 1024 / 1024
            print(f"  {key:20s} n={count:3d}  total={tot_ms:6.0f}ms  avg={avg_ms:5.0f}ms  max={max_ms:5.0f}ms  size={tot_sz:6.2f}MB")
        print()

        print(f"Slowest 15 downloads:")
        for path, elapsed, sz in sorted(download_times, key=lambda x: -x[1])[:15]:
            print(f"  {elapsed*1000:6.0f}ms  {sz/1024:8.1f}KB  {path}")
        print()

    overhead = overall_elapsed * 1000 - depth_stage_time[0] * 1000 - sum(evict_times) * 1000
    print(f"Overhead breakdown (for {overall_elapsed*1000:.0f}ms total):")
    print(f"  Depth stage (range requests):  {depth_stage_time[0]*1000:6.0f}ms")
    print(f"  Eviction scans:                 {sum(evict_times)*1000:6.0f}ms")
    print(f"  Parallel download wall time:    ~{(overall_elapsed - depth_stage_time[0] - sum(evict_times))*1000:6.0f}ms")
    print(f"  (Summed sequential download:     {sum(t for _, t, _ in download_times)*1000:6.0f}ms)")
    print(f"  Speedup from parallelism:       {sum(t for _, t, _ in download_times)/(overall_elapsed - depth_stage_time[0] - sum(evict_times)):.1f}x" if overall_elapsed > depth_stage_time[0] + sum(evict_times) else "")


if __name__ == "__main__":
    # Load uncached scene list
    uncached = Path("/tmp/uncached_scenes.txt").read_text().strip().split("\n")
    print(f"Found {len(uncached)} uncached scenes")

    # Test 2-3 cold scenes to get consistent timing
    test_scenes = uncached[:3]
    for scene in test_scenes:
        profile_cold_start(scene)
