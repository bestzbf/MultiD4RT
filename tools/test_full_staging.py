#!/usr/bin/env python3
"""Test the full scannetpp staging with h5 Range requests."""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from datasets.sample_stage import SampleLocalStager, SampleStageConfig


class FakeScanNetPPAdapter:
    def __init__(self, root: str):
        self.dataset_name = "scannetpp"
        self.root = Path(root)
        self.data_root = self.root
        self._scene_cache = {}
        self._staged_depth_map_tmp = None
        self._staged_h5_frame_map = None
        self._depth_chunk_cache = {}
        self.precomputed_name = "precomputed.npz"

    def _get_scene_data(self, scene_name: str):
        if scene_name in self._scene_cache:
            return self._scene_cache[scene_name]
        scene_dir = self.data_root / scene_name
        full_indices = np.arange(960, dtype=np.int32)  # 7e7d2e8640 has 960 frames
        data = {
            "scene_dir": scene_dir,
            "full_indices": full_indices,
            "frame_stems": [f"{i:06d}" for i in range(960)],
        }
        self._scene_cache[scene_name] = data
        return data


def test_full_staging():
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
    adapter = FakeScanNetPPAdapter(root="/data_cos/hdu_datasets/scannetpp/data")

    scene_name = "7e7d2e8640"
    # Simulate realistic frame indices: start=100, 48 consecutive frames
    frame_indices = list(range(100, 148))

    print(f"Testing full staging: scene={scene_name} frames={frame_indices[0]}..{frame_indices[-1]}")
    print("=" * 70)

    t0 = time.perf_counter()
    with stager.stage_sample(adapter, scene_name, frame_indices, sample_tag="test_h5range") as staged:
        t_stage = time.perf_counter() - t0
        print(f"\nStaging complete: {t_stage*1000:.0f}ms")

        # Verify mini h5 exists and is readable
        sd = adapter._scene_cache.get(scene_name)
        if sd:
            precomputed_dir = sd.get("_precomputed_dir", sd["scene_dir"])
            h5_path = precomputed_dir / "precomputed.h5"
            print(f"Mini h5 path: {h5_path}")
            print(f"Mini h5 exists: {h5_path.exists()}")
            if h5_path.exists():
                import h5py
                with h5py.File(h5_path, "r") as f:
                    print(f"Mini h5 keys: {list(f.keys())}")
                    for key in f.keys():
                        ds = f[key]
                        print(f"  {key}: shape={ds.shape} dtype={ds.dtype}")

            # Check frame map
            h5_map = sd.get("_staged_h5_frame_map")
            print(f"\nFrame map present: {h5_map is not None}")
            if h5_map:
                print(f"  Map size: {len(h5_map)} entries")
                print(f"  Sample: frame 100 -> idx {h5_map.get(100)}, frame 147 -> idx {h5_map.get(147)}")

    print(f"\n{'='*70}")
    print(f"RESULT: staging took {t_stage*1000:.0f}ms (was 10000-12000ms before)")


if __name__ == "__main__":
    test_full_staging()
