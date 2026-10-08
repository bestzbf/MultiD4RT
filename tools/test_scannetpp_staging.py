"""Standalone test to diagnose scannetpp staging performance."""
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


def test_scannetpp_staging():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mount_root = base / "mount"
        dataset_root = mount_root / "hdu_datasets" / "scannetpp" / "data"
        stage_root = base / "stage"
        passwd_file = base / "passwd.txt"
        passwd_file.write_text("id:key", encoding="utf-8")

        scene_id = "a492fe77aa"
        num_frames = 48
        print(f"Creating fake scannetpp scene: {scene_id} with {num_frames} frames")
        _make_fake_scannetpp_scene(dataset_root, scene_id, num_frames=num_frames)
        adapter = FakeScanNetPPAdapter(dataset_root, scene_id, num_frames=num_frames)

        stager = SampleLocalStager(
            SampleStageConfig(
                backend="cos_sdk",
                stage_root=str(stage_root),
                sdk_workers=32,
                mount_root=str(mount_root),
                passwd_file=str(passwd_file),
                enabled_datasets=("scannetpp",),
            )
        )

        download_calls: list[tuple[str, float]] = []

        def fake_download(self, src_path: Path, cache_path: Path, rel_key: Path) -> None:
            t0 = time.perf_counter()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Simulate network latency (10ms per file)
            time.sleep(0.01)
            shutil.copy2(src_path, cache_path)
            elapsed = time.perf_counter() - t0
            download_calls.append((rel_key.as_posix(), elapsed))

        stager._download_to_cache = MethodType(fake_download, stager)

        # Override _ensure_cached to always call download (simulate cold cache)
        original_ensure_cached = stager._ensure_cached
        def fake_ensure_cached(self, src_path: Path) -> Path:
            src_path = Path(src_path)
            rel_key = Path(self._to_cos_key(src_path))
            cache_path = self.cache_data_root / rel_key
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Always download (simulate cold cache)
            self._download_to_cache(src_path, cache_path, rel_key)
            return cache_path

        stager._ensure_cached = MethodType(fake_ensure_cached, stager)

        # Mock depth staging to avoid COS SDK calls
        def fake_prepare_depth(self, adapter, sequence_name, frame_indices, staged_dataset_root):
            # Simulate depth staging time
            time.sleep(0.1)

        stager._prepare_scannetpp_depth_stage = MethodType(fake_prepare_depth, stager)

        print(f"\nStarting staging test...")
        start_time = time.perf_counter()
        with stager.stage_sample(
            adapter=adapter,
            sequence_name=scene_id,
            frame_indices=list(range(num_frames)),
            sample_tag="scannetpp_test",
        ):
            pass
        elapsed = time.perf_counter() - start_time

        print(f"\n{'='*60}")
        print(f"ScanNetPP Staging Performance Test Results")
        print(f"{'='*60}")
        print(f"Scene: {scene_id}")
        print(f"Frames: {num_frames}")
        print(f"Files downloaded: {len(download_calls)}")
        print(f"Total time: {elapsed:.3f}s")
        print(f"Time per file: {elapsed/len(download_calls)*1000:.1f}ms")
        print(f"\nBreakdown:")

        # Group by file type
        metadata_files = [call for call in download_calls if not call[0].endswith(".jpg")]
        jpg_files = [call for call in download_calls if call[0].endswith(".jpg")]

        print(f"  Metadata files: {len(metadata_files)}")
        for path, t in metadata_files:
            print(f"    {path}: {t*1000:.1f}ms")

        print(f"  JPG frames: {len(jpg_files)}")
        if jpg_files:
            total_jpg_time = sum(t for _, t in jpg_files)
            print(f"    Total JPG time: {total_jpg_time:.3f}s")
            print(f"    Avg per JPG: {total_jpg_time/len(jpg_files)*1000:.1f}ms")

        # Expected files: 4 metadata + 48 JPG = 52
        expected_files = 4 + num_frames
        assert len(download_calls) == expected_files, f"Expected {expected_files} downloads, got {len(download_calls)}"

        # Check metadata files
        paths = [call[0] for call in download_calls]
        assert any("cameras.txt" in p for p in paths), "Missing cameras.txt"
        assert any("images.txt" in p for p in paths), "Missing images.txt"
        assert any("pose_intrinsic_imu.json" in p for p in paths), "Missing pose_intrinsic_imu.json"
        assert any("precomputed.h5" in p for p in paths), "Missing precomputed.h5"

        print(f"\n✓ All assertions passed")
        print(f"\n{'='*60}")
        print(f"Analysis:")
        print(f"{'='*60}")
        print(f"With 10ms simulated latency per file:")
        print(f"  Expected time (sequential): {expected_files * 0.01:.3f}s")
        print(f"  Actual time: {elapsed:.3f}s")
        print(f"  Parallelization factor: {expected_files * 0.01 / elapsed:.2f}x")
        print(f"\nIn real COS environment with 21s staging time:")
        print(f"  Avg time per file: {21000 / expected_files:.0f}ms")
        print(f"  This suggests network latency or COS throttling is the bottleneck")


if __name__ == "__main__":
    test_scannetpp_staging()
