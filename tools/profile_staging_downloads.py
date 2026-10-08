#!/usr/bin/env python3
"""Profile which files are slow during ScanNetPP staging."""

import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.sample_stage import SampleLocalStager, SampleStageConfig


class FakeScanNetPPAdapter:
    """Minimal fake adapter for testing."""
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


def profile_staging():
    """Profile a single ScanNetPP staging operation."""

    # Use production config - match the training script settings
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

    # Create fake adapter with correct path
    adapter = FakeScanNetPPAdapter(root="/data_cos/hdu_datasets/scannetpp/data")

    # Use a real scene from the logs - use random frames to simulate real usage
    scene_name = "7e7d2e8640"
    frame_indices = list(range(48))  # 48 frames starting at 0

    # Track download times
    download_times = []
    original_download = stager._download_to_cache

    def timed_download(src_path, cache_path, rel_key):
        start = time.time()
        result = original_download(src_path, cache_path, rel_key)
        elapsed = time.time() - start
        download_times.append((str(rel_key), elapsed))
        print(f"Downloaded {rel_key} in {elapsed*1000:.0f}ms")
        return result

    stager._download_to_cache = timed_download

    # Stage the sample
    print(f"\nStaging scene={scene_name} frames={len(frame_indices)}")
    print("=" * 80)

    overall_start = time.time()

    with stager.stage_sample(adapter, scene_name, frame_indices, sample_tag="profile") as staged_adapter:
        overall_elapsed = time.time() - overall_start
        print("=" * 80)
        print(f"\nTotal staging time: {overall_elapsed*1000:.0f}ms")
        print(f"Files downloaded: {len(download_times)}")

        if download_times:
            print(f"\nSlowest downloads:")
            for path, elapsed in sorted(download_times, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {elapsed*1000:6.0f}ms  {path}")

            total_download = sum(t for _, t in download_times)
            print(f"\nTotal download time: {total_download*1000:.0f}ms")
            print(f"Overhead (staging - downloads): {(overall_elapsed - total_download)*1000:.0f}ms")

            # Group by file type
            print(f"\nBreakdown by file type:")
            by_type: dict[str, list[float]] = {}
            for path, elapsed in download_times:
                if path.endswith(".jpg"):
                    key = "JPG frames"
                elif path.endswith(".h5"):
                    key = "precomputed.h5"
                elif path.endswith(".txt"):
                    key = "colmap .txt"
                elif path.endswith(".json"):
                    key = "pose .json"
                elif path.endswith(".bin"):
                    key = "depth.bin"
                else:
                    key = "other"
                by_type.setdefault(key, []).append(elapsed)

            for key, times in sorted(by_type.items(), key=lambda x: -sum(x[1])):
                total_ms = sum(times) * 1000
                avg_ms = total_ms / len(times)
                max_ms = max(times) * 1000
                print(f"  {key:20s} count={len(times):3d}  total={total_ms:6.0f}ms  avg={avg_ms:5.0f}ms  max={max_ms:5.0f}ms")


if __name__ == "__main__":
    profile_staging()
