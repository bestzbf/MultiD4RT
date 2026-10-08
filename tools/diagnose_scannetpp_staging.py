"""Diagnose why scannetpp staging takes 21 seconds in production."""
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import MethodType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.sample_stage import SampleLocalStager, SampleStageConfig


class FakeScanNetPPAdapter:
    dataset_name = "scannetpp"

    def __init__(self, root: Path, scene_id: str, num_frames: int = 48):
        self.root = root  # Required by supports() check
        self.data_root = root
        self.split = "train"
        self._scene_cache = {}
        self._depth_chunk_cache = {}
        self._staged_depth_map_tmp = None
        self._num_frames = num_frames
        self._scene_id = scene_id

    def get_num_frames(self, sequence_name: str) -> int:
        return self._num_frames

    def _get_scene_data(self, scene_name: str):
        if scene_name in self._scene_cache:
            return self._scene_cache[scene_name]
        import numpy as np
        scene_dir = self.data_root / scene_name
        full_indices = np.arange(self._num_frames, dtype=np.int32)
        data = {
            "scene_dir": scene_dir,
            "full_indices": full_indices,
            "frame_stems": [f"{i:06d}" for i in range(self._num_frames)],
        }
        self._scene_cache[scene_name] = data
        return data


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_fake_scannetpp_scene(root: Path, scene_id: str, num_frames: int = 48) -> None:
    scene_dir = root / scene_id
    colmap_dir = scene_dir / "iphone" / "colmap"
    frames_dir = scene_dir / "iphone" / "frames"

    _write_text(colmap_dir / "cameras.txt", "# camera data")
    _write_text(colmap_dir / "images.txt", "# images data")
    _write_text(scene_dir / "iphone" / "pose_intrinsic_imu.json", "{}")
    _write_text(scene_dir / "precomputed.h5", "fake-h5-data")

    for i in range(num_frames):
        _write_text(frames_dir / f"{i:06d}.jpg", f"rgb-{i}")


def run_diagnosis():
    print("="*70)
    print("ScanNetPP Staging Performance Diagnosis")
    print("="*70)
    print("\nSimulating production environment:")
    print("  - Scene: a492fe77aa")
    print("  - Frames: 48")
    print("  - Real staging time: 21049ms")
    print("  - Files to download: 4 metadata + 48 JPG = 52 files")
    print("  - Expected time per file: 21049ms / 52 = 405ms")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mount_root = base / "mount"
        dataset_root = mount_root / "hdu_datasets" / "scannetpp" / "data"
        stage_root = base / "stage"
        passwd_file = base / "passwd.txt"
        passwd_file.write_text("id:key", encoding="utf-8")

        scene_id = "a492fe77aa"
        num_frames = 48
        _make_fake_scannetpp_scene(dataset_root, scene_id, num_frames=num_frames)
        adapter = FakeScanNetPPAdapter(dataset_root, scene_id, num_frames=num_frames)

        # Test with different worker counts
        for sdk_workers in [1, 4, 8, 16, 32]:
            stager = SampleLocalStager(
                SampleStageConfig(
                    backend="cos_sdk",
                    stage_root=str(stage_root),
                    sdk_workers=sdk_workers,
                    mount_root=str(mount_root),
                    passwd_file=str(passwd_file),
                    enabled_datasets=("scannetpp",),
                )
            )

            download_calls: list[tuple[str, float]] = []

            def fake_download(self, src_path: Path, cache_path: Path, rel_key: Path) -> None:
                t0 = time.perf_counter()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Simulate realistic COS latency: 400ms per file
                time.sleep(0.4)
                shutil.copy2(src_path, cache_path)
                elapsed = time.perf_counter() - t0
                download_calls.append((rel_key.as_posix(), elapsed))

            stager._download_to_cache = MethodType(fake_download, stager)

            # Override _ensure_cached to always download (cold cache)
            def fake_ensure_cached(self, src_path: Path) -> Path:
                src_path = Path(src_path)
                rel_key = Path(self._to_cos_key(src_path))
                cache_path = self.cache_data_root / rel_key
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._download_to_cache(src_path, cache_path, rel_key)
                return cache_path

            stager._ensure_cached = MethodType(fake_ensure_cached, stager)

            # Mock depth staging
            def fake_prepare_depth(self, adapter, sequence_name, frame_indices, staged_dataset_root):
                time.sleep(0.1)  # Simulate depth staging overhead

            stager._prepare_scannetpp_depth_stage = MethodType(fake_prepare_depth, stager)

            download_calls.clear()
            start_time = time.perf_counter()
            with stager.stage_sample(
                adapter=adapter,
                sequence_name=scene_id,
                frame_indices=list(range(num_frames)),
                sample_tag="scannetpp_test",
            ):
                pass
            elapsed = time.perf_counter() - start_time

            expected_sequential = 52 * 0.4
            speedup = expected_sequential / elapsed
            print(f"Workers={sdk_workers:2d}: {elapsed:6.2f}s (speedup: {speedup:4.2f}x, files: {len(download_calls)})")

        print()
        print("="*70)
        print("Analysis:")
        print("="*70)
        print("1. With 400ms latency per file and 32 workers:")
        print("   - Sequential time: 52 * 400ms = 20.8s")
        print("   - With perfect parallelization: 20.8s / 32 = 0.65s")
        print("   - Actual production time: 21s")
        print()
        print("2. This suggests:")
        print("   - Files are being downloaded sequentially or with low parallelism")
        print("   - OR: COS is throttling requests")
        print("   - OR: Network bandwidth is saturated")
        print()
        print("3. Potential solutions:")
        print("   a) Add scene-level prefetch for scannetpp (like blendedmvs)")
        print("      - Download entire scene once, cache with .d4rt_scene_complete marker")
        print("      - Subsequent samples from same scene: 0ms staging time")
        print()
        print("   b) Increase cache size (currently 100GB shared across 5 datasets)")
        print("      - Check cache hit rate: are we re-downloading same scenes?")
        print()
        print("   c) Verify SDK workers are actually being used")
        print("      - Check if ThreadPoolExecutor is working correctly")
        print()
        print("   d) Profile actual COS download time")
        print("      - Add timing logs to _download_to_cache")
        print("      - Check if time is spent in download vs file I/O")


if __name__ == "__main__":
    run_diagnosis()
