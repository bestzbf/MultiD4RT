"""
Sample-local staging for COS-backed datasets.

This module implements an optional builder-side staging step backed by a
node-local shared raw-file cache:

1. Build the exact file manifest for one sample
2. Materialize those files into a persistent local cache directory
3. Hard-link the cached files into a temporary sample directory
4. Let the existing adapter.load_clip() read from local disk
5. Delete the temporary sample directory after the sample is built

This keeps QuerySample spool semantics unchanged while finally exploiting
the sampler's short-term frame / sequence locality across nearby samples.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import hashlib
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Optional

from qcloud_cos import CosConfig, CosS3Client


@dataclasses.dataclass(frozen=True)
class SampleStageConfig:
    backend: str
    stage_root: str
    sdk_workers: int = 8
    request_timeout_s: float = 20.0
    request_retries: int = 1
    cache_max_bytes: int = 100 * 1024**3
    cache_low_watermark_ratio: float = 0.9
    cache_touch_interval_s: float = 30.0
    cache_scan_interval_s: float = 30.0
    eviction_mode: str = "background"
    window_radius: int = 0
    mount_root: str = "/data_cos"
    extra_mount_roots: tuple[str, ...] = ()
    bucket: str = "hd-ai-data-1251882982"
    region: str = "ap-beijing"
    passwd_file: str = "/etc/passwd-s3fs-data_cos"
    enabled_datasets: tuple[str, ...] = ()
    scene_prefetch_datasets: tuple[str, ...] = ()
    pinned_manifest_root: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SampleStageConfig":
        datasets = raw.get("enabled_datasets", ())
        if isinstance(datasets, str):
            datasets = tuple(
                item.strip()
                for item in datasets.split(",")
                if item.strip()
            )
        else:
            datasets = tuple(str(item).strip() for item in datasets if str(item).strip())
        scene_prefetch_datasets = raw.get("scene_prefetch_datasets", ())
        if isinstance(scene_prefetch_datasets, str):
            scene_prefetch_datasets = tuple(
                item.strip()
                for item in scene_prefetch_datasets.split(",")
                if item.strip()
            )
        else:
            scene_prefetch_datasets = tuple(
                str(item).strip()
                for item in scene_prefetch_datasets
                if str(item).strip()
            )
        pinned_manifest_root = str(
            raw.get(
                "pinned_manifest_root",
                os.getenv("D4RT_ROLLING_WARM_READY_DIR", "")
                or os.getenv("D4RT_ROLLING_WARM_PROGRESS_DIR", ""),
            )
        ).strip()
        eviction_mode = str(
            raw.get("eviction_mode", raw.get("cache_eviction_mode", "background"))
        ).strip().lower()
        if eviction_mode in {"", "on", "true", "1"}:
            eviction_mode = "background"
        elif eviction_mode in {"off", "none", "disable"}:
            eviction_mode = "disabled"
        elif eviction_mode not in {"background", "external", "disabled"}:
            eviction_mode = "background"
        return cls(
            backend=str(raw.get("backend", "")).strip().lower(),
            stage_root=str(raw.get("stage_root", "")).strip(),
            sdk_workers=max(1, int(raw.get("sdk_workers", 8))),
            request_timeout_s=max(1.0, float(raw.get("request_timeout_s", 20.0))),
            request_retries=max(0, int(raw.get("request_retries", 1))),
            cache_max_bytes=max(0, int(raw.get("cache_max_bytes", 100 * 1024**3))),
            cache_low_watermark_ratio=min(
                0.99,
                max(0.50, float(raw.get("cache_low_watermark_ratio", 0.9))),
            ),
            cache_touch_interval_s=max(
                0.0, float(raw.get("cache_touch_interval_s", 30.0))
            ),
            cache_scan_interval_s=max(
                1.0, float(raw.get("cache_scan_interval_s", 30.0))
            ),
            eviction_mode=eviction_mode,
            window_radius=max(0, int(raw.get("window_radius", 0))),
            mount_root=str(raw.get("mount_root", "/data_cos")).strip(),
            extra_mount_roots=tuple(
                str(item).strip()
                for item in (raw.get("extra_mount_roots") or ())
                if str(item).strip()
            ),
            bucket=str(raw.get("bucket", "hd-ai-data-1251882982")).strip(),
            region=str(raw.get("region", "ap-beijing")).strip(),
            passwd_file=str(raw.get("passwd_file", "/etc/passwd-s3fs-data_cos")).strip(),
            enabled_datasets=datasets,
            scene_prefetch_datasets=scene_prefetch_datasets,
            pinned_manifest_root=pinned_manifest_root,
        )


class SampleLocalStager:
    """Builder-local staging helper."""

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_tls', None)  # threading.local() is not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tls = threading.local()

    def __init__(self, config: SampleStageConfig):
        if config.backend != "cos_sdk":
            raise ValueError(f"Unsupported sample stage backend: {config.backend!r}")
        self.config = config
        self.mount_root = Path(config.mount_root)
        self.all_mount_roots: list[Path] = [self.mount_root] + [
            Path(p) for p in config.extra_mount_roots
        ]
        self.stage_root = Path(config.stage_root)
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.cache_root = self.stage_root / "shared_raw_cache"
        self.cache_data_root = self.cache_root / "data"
        # The original flat lock directory can accumulate millions of lock
        # files on long runs.  Keep new locks in a sharded v2 tree so each
        # directory stays small and metadata lookups do not dominate staging.
        self.lock_root = self.cache_root / "locks_v2"
        self.work_root = self.stage_root / "work"
        self.cache_data_root.mkdir(parents=True, exist_ok=True)
        self.lock_root.mkdir(parents=True, exist_ok=True)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.cache_usage_path = self.cache_root / "cache_usage_v1.txt"
        self.cache_usage_scan_stamp_path = self.cache_root / "cache_usage_full_scan.stamp"
        self.cache_usage_lock_path = self.lock_root / "cache_usage.lock"
        self._tls = threading.local()
        self._last_cache_scan_s = time.time()
        self._eviction_thread: threading.Thread | None = None
        self._pinned_manifest_root = Path(config.pinned_manifest_root) if config.pinned_manifest_root else None
        self._last_stage_stats: dict[str, Any] = {}

    def _get_client(self) -> CosS3Client:
        client = getattr(self._tls, "cos_client", None)
        if client is None:
            parts = Path(self.config.passwd_file).read_text().strip().split(":")
            if len(parts) == 2:
                secret_id, secret_key = parts
            elif len(parts) == 3:
                _bucket, secret_id, secret_key = parts
            else:
                raise ValueError(
                    f"Unsupported COS passwd file format: {self.config.passwd_file}"
                )
            cos_config = CosConfig(
                Region=self.config.region,
                SecretId=secret_id,
                SecretKey=secret_key,
                Scheme="https",
                Timeout=max(1, int(self.config.request_timeout_s)),
            )
            client = CosS3Client(cos_config)
            self._tls.cos_client = client
        return client

    def _reset_client(self) -> None:
        self._tls.__dict__.pop("cos_client", None)

    def _get_object(self, key: str, range_header: str | None = None) -> Any:
        kwargs: dict[str, Any] = {
            "Bucket": self.config.bucket,
            "Key": key,
        }
        if range_header is not None:
            kwargs["Range"] = range_header

        last_exc: Exception | None = None
        for attempt in range(self.config.request_retries + 1):
            try:
                return self._get_client().get_object(**kwargs)
            except Exception as exc:
                last_exc = exc
                self._reset_client()
                if attempt < self.config.request_retries:
                    time.sleep(min(2.0, 0.25 * (2 ** attempt)))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"COS get_object failed unexpectedly: {key}")

    def supports(self, adapter: Any) -> bool:
        dataset_name = str(getattr(adapter, "dataset_name", "")).strip()
        if dataset_name not in self.config.enabled_datasets:
            return False
        root = getattr(adapter, "root", None)
        if root is None:
            return False
        root_str = str(root)
        for mr in self.all_mount_roots:
            mount = str(mr).rstrip("/") + "/"
            if root_str.startswith(mount) or root_str == str(mr):
                return True
        return False

    @staticmethod
    def _stage_blendedmvs_precomputed_enabled() -> bool:
        return os.getenv(
            "D4RT_BLENDEDMVS_STAGE_PRECOMPUTED", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

    @contextlib.contextmanager
    def stage_sample(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
        sample_tag: str,
    ) -> Iterator[Any]:
        if not self.supports(adapter):
            yield adapter
            return

        t_stage0 = time.perf_counter()
        manifest = self._build_manifest(adapter, sequence_name, frame_indices)
        if not manifest:
            yield adapter
            return
        t_prefetch0 = time.perf_counter()
        self._maybe_prefetch_scene(adapter, sequence_name)
        prefetch_s = time.perf_counter() - t_prefetch0

        direct_cache_rebase = self._use_direct_cache_rebase(adapter)
        temp_dir: Path | None = None
        try:
            t_materialize0 = time.perf_counter()
            if direct_cache_rebase:
                materialize_stats = self._materialize_manifest_cache_only(manifest)
                staged_dataset_root = self.cache_data_root / Path(
                    self._to_cos_key(Path(adapter.root))
                )
            else:
                temp_dir = Path(
                    tempfile.mkdtemp(
                        prefix=f"sample_stage_{sample_tag}_",
                        dir=self.work_root,
                    )
                )
                materialize_stats = self._materialize_manifest(manifest, temp_dir)
                staged_dataset_root = temp_dir / self._to_cos_key(Path(adapter.root))
            materialize_s = time.perf_counter() - t_materialize0
            self._last_stage_stats = {
                "dataset": str(getattr(adapter, "dataset_name", "")),
                "sequence": sequence_name,
                "requested_frames": len(frame_indices),
                "manifest_files": len(manifest),
                "materialize_s": materialize_s,
                **materialize_stats,
            }
            # scannetpp: warm _scene_cache from staged files so _get_scene_data
            # reads local cameras.txt/images.txt instead of COS mount
            depth_s = 0.0
            if str(getattr(adapter, "dataset_name", "")) == "scannetpp":
                source_data_root = self._scannetpp_source_data_root(adapter)
                t_depth0 = time.perf_counter()
                if sequence_name not in adapter._scene_cache:
                    adapter.data_root = staged_dataset_root
                    try:
                        adapter._get_scene_data(sequence_name)
                    finally:
                        adapter.data_root = source_data_root
                        # restore scene_dir to COS path; drop if path restoration fails
                        sd = adapter._scene_cache.get(sequence_name)
                        if sd is not None:
                            try:
                                cos_scene_dir = source_data_root / Path(sd["scene_dir"]).relative_to(staged_dataset_root)
                                adapter._scene_cache[sequence_name] = dict(sd, scene_dir=cos_scene_dir)
                            except ValueError:
                                adapter._scene_cache.pop(sequence_name, None)
                depth_frame_indices = self._expand_frame_indices(
                    adapter,
                    sequence_name,
                    frame_indices,
                )
                self._prepare_scannetpp_depth_stage(
                    adapter,
                    sequence_name,
                    depth_frame_indices,
                    staged_dataset_root,
                )
                depth_s = time.perf_counter() - t_depth0
                self._last_stage_stats["scannetpp_depth_stage_s"] = depth_s
            stage_ready_s = time.perf_counter() - t_stage0
            slow_threshold = float(os.environ.get("D4RT_SAMPLE_STAGE_SLOW_THRESHOLD_S", "10.0"))
            profile_enabled = os.environ.get("D4RT_SAMPLE_STAGE_PROFILE", "").strip().lower() in {
                "1", "true", "yes", "on"
            }
            if profile_enabled or stage_ready_s >= slow_threshold:
                dataset_name = str(getattr(adapter, "dataset_name", ""))
                print(
                    f"[SampleStageProfile] dataset={dataset_name} seq={sequence_name} "
                    f"frames={len(frame_indices)} total={stage_ready_s:.3f}s "
                    f"prefetch={prefetch_s:.3f}s materialize={materialize_s:.3f}s "
                    f"depth={depth_s:.3f}s files={len(manifest)} "
                    f"cold={materialize_stats.get('cold_files', 0)} "
                    f"file_max={materialize_stats.get('file_max_s', 0.0):.3f}s "
                    f"file_sum={materialize_stats.get('file_sum_s', 0.0):.3f}s "
                    f"sdk_workers={self.config.sdk_workers} "
                    f"timeout={self.config.request_timeout_s:.1f}s "
                    f"retries={self.config.request_retries}",
                    flush=True,
                )
            with self._rebase_adapter(adapter, sequence_name, staged_dataset_root):
                yield adapter
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _build_manifest(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        dataset_name = str(adapter.dataset_name)
        manifest_frame_indices = self._expand_frame_indices(
            adapter, sequence_name, frame_indices
        )
        if dataset_name == "kubric":
            return self._manifest_kubric(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "pointodyssey":
            return self._manifest_pointodyssey(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "dynamic_replica":
            return self._manifest_dynamic_replica(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "co3dv2":
            return self._manifest_co3dv2(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "blendedmvs":
            return self._manifest_blendedmvs(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "scannetpp":
            return self._manifest_scannetpp(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "mvssynth":
            return self._manifest_mvssynth(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "scannet":
            return self._manifest_scannet(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "tartanair":
            return self._manifest_tartanair(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "vkitti2":
            return self._manifest_vkitti2(adapter, sequence_name, manifest_frame_indices)
        if dataset_name == "waymo":
            return self._manifest_waymo(adapter, sequence_name, manifest_frame_indices)
        return []

    def _expand_frame_indices(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[int]:
        radius = self.config.window_radius
        if not frame_indices:
            return frame_indices
        unique_indices = sorted(set(int(i) for i in frame_indices))
        if radius <= 0:
            return unique_indices
        try:
            num_frames = int(adapter.get_num_frames(sequence_name))
        except Exception:
            return unique_indices
        if num_frames <= 0:
            return unique_indices

        positive_gaps = [
            b - a for a, b in zip(unique_indices, unique_indices[1:]) if b > a
        ]
        if positive_gaps and min(positive_gaps) > 1 and len(set(positive_gaps)) == 1:
            # Strided clips should not pull every intermediate frame into the
            # raw cache.  Extend only along the observed stride so staging stays
            # proportional to the frames the adapter will actually load.
            stride = positive_gaps[0]
            extra_steps = radius // stride
            if extra_steps <= 0:
                return unique_indices
            expanded = set(unique_indices)
            first = unique_indices[0]
            last = unique_indices[-1]
            for step in range(1, extra_steps + 1):
                before = first - step * stride
                after = last + step * stride
                if before >= 0:
                    expanded.add(before)
                if after < num_frames:
                    expanded.add(after)
            return sorted(expanded)

        low = max(0, unique_indices[0] - radius)
        high = min(num_frames - 1, unique_indices[-1] + radius)
        return list(range(low, high + 1))

    def _manifest_kubric(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = [record.rank_path]
        if record.h5_path is not None:
            paths.append(record.h5_path)
            if record.depth_names is not None or getattr(record, "has_depth_dir", False):
                paths.extend(
                    record.depth_dir / adapter._depth_name_for_index(record, i)
                    for i in frame_indices
                )
        else:
            paths.append(record.ann_path)
        paths.extend(
            record.frame_dir / adapter._frame_name_for_index(record, i)
            for i in frame_indices
        )
        return self._dedupe_keep_order(paths)

    def _manifest_pointodyssey(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter.get_record(sequence_name)
        paths: list[Path] = []
        if record.scene_info_path is not None:
            paths.append(Path(record.scene_info_path))
        if self._pointodyssey_stage_anno_h5(record):
            anno_path = Path(record.anno_path) if record.anno_path is not None else None
            if anno_path is not None:
                h5_path = anno_path.with_suffix(".h5")
                paths.append(h5_path if h5_path.exists() else anno_path)
        for idx in frame_indices:
            paths.append(Path(record.rgb_paths[idx]))
            if record.depth_paths is not None:
                paths.append(Path(record.depth_paths[idx]))
            if record.normal_paths is not None:
                paths.append(Path(record.normal_paths[idx]))
        return self._dedupe_keep_order(paths)

    def _manifest_dynamic_replica(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        rgb_from_trajectory = bool(getattr(adapter, "rgb_from_trajectory", False))
        prefer_trajectory_npz = bool(getattr(adapter, "prefer_trajectory_npz", False))
        load_trajectories = bool(getattr(adapter, "load_trajectories", False))
        skip_depth_when_tracks = bool(getattr(adapter, "skip_depth_when_tracks", False))
        has_record_tracks = bool(getattr(record, "has_trajectories", False))
        use_track_files = load_trajectories and has_record_tracks
        load_depths = not (use_track_files and skip_depth_when_tracks)
        for idx in frame_indices:
            traj_rel = record.traj_rel_paths[idx]
            use_traj_npz = bool(prefer_trajectory_npz and use_track_files and traj_rel is not None)
            if not (
                rgb_from_trajectory
                and use_track_files
                and traj_rel is not None
                and not use_traj_npz
            ):
                paths.append(adapter.split_root / record.image_rel_paths[idx])
            if load_depths:
                paths.append(adapter.split_root / record.depth_rel_paths[idx])
            if traj_rel is not None:
                traj_path = adapter.split_root / traj_rel
                paths.append(traj_path.with_suffix(".npz") if use_traj_npz else traj_path)
        return self._dedupe_keep_order(paths)

    def _manifest_scannet(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        rec = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in frame_indices:
            paths.append(rec.image_dir / f"{idx:05d}.jpg")
            paths.append(rec.depth_dir / f"{idx:05d}.png")
            paths.append(rec.pose_dir  / f"{idx:05d}.txt")
        # precomputed.h5 stays on COS mount — h5py slices frames without full download
        return self._dedupe_keep_order(paths)

    def _manifest_scannetpp(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        # data_root is intentionally mutated while a sample is staged.  Timeout
        # threads may outlive the failed sample, so source manifests must be
        # derived from the immutable dataset root instead.
        scene_dir = self._scannetpp_source_data_root(adapter) / sequence_name
        colmap_dir = scene_dir / "iphone" / "colmap"
        frames_dir = scene_dir / "iphone" / "frames"
        rgb_mode = str(
            getattr(
                adapter,
                "rgb_read_mode",
                os.getenv("SCANNETPP_RGB_READ_MODE", "auto"),
            )
            or "auto"
        ).strip().lower()
        if rgb_mode in {"cache256", "decoded_cache", "rgb_cache"}:
            rgb_mode = "cache"
        frame_index_path = scene_dir / "iphone" / "frame_index.pkl"
        if self._has_cached_or_source(frame_index_path):
            paths: list[Path] = [frame_index_path]
        else:
            paths = [
                colmap_dir / "cameras.txt",
                colmap_dir / "images.txt",
                scene_dir / "iphone" / "pose_intrinsic_imu.json",
            ]
        if rgb_mode == "cache" and getattr(adapter, "target_hw", None) is not None:
            from datasets.adapters.scannetpp import rgb_cache_frame_path

            target_hw = tuple(int(v) for v in adapter.target_hw)
            cache_paths = [
                rgb_cache_frame_path(scene_dir, target_hw, i)
                for i in frame_indices
            ]
            missing_cache_paths = [
                path for path in cache_paths if not self._has_cached_or_source(path)
            ]
            if missing_cache_paths:
                cache_paths = self._ensure_scannetpp_rgb_cache_frames(
                    scene_dir,
                    target_hw,
                    frame_indices,
                )
            paths += cache_paths
        elif rgb_mode == "cache":
            raise ValueError(
                "ScanNet++ RGB cache mode requires adapter.target_hw so the "
                "frame cache directory can be resolved; refusing to stage "
                f"full iphone/rgb.mkv for sample sequence={sequence_name!r}."
            )
        elif rgb_mode == "video":
            paths.append(scene_dir / "iphone" / "rgb.mkv")
        else:
            paths += [frames_dir / f"{i:06d}.jpg" for i in frame_indices]
        # depth.bin fetched via Range requests — not in manifest
        # precomputed.h5 stays on the original mount and is read via COS Range.
        return self._dedupe_keep_order(paths)

    def _valid_scannetpp_rgb_cache_file(
        self,
        path: Path,
        target_hw: tuple[int, int],
    ) -> bool:
        try:
            return (
                path.is_file()
                and path.stat().st_size
                == int(target_hw[0]) * int(target_hw[1]) * 3
            )
        except OSError:
            return False

    def _ensure_scannetpp_rgb_cache_frames(
        self,
        scene_dir: Path,
        target_hw: tuple[int, int],
        frame_indices: list[int],
    ) -> list[Path]:
        from datasets.adapters.scannetpp import (
            _read_rgb_jpg,
            rgb_cache_frame_path,
            write_rgb_cache_frame,
        )

        cache_paths = [
            rgb_cache_frame_path(scene_dir, target_hw, i)
            for i in frame_indices
        ]
        missing: list[tuple[int, Path, Path, Path]] = []
        for frame_idx, source_cache_path in zip(frame_indices, cache_paths):
            rel_key = Path(self._to_cos_key(source_cache_path))
            local_cache_path = self.cache_data_root / rel_key
            if self._valid_scannetpp_rgb_cache_file(local_cache_path, target_hw):
                self._touch_cache_entry(local_cache_path)
                continue
            if self._valid_scannetpp_rgb_cache_file(source_cache_path, target_hw):
                local_cache_path = self._ensure_cached(source_cache_path)
                if self._valid_scannetpp_rgb_cache_file(local_cache_path, target_hw):
                    continue
            missing.append((int(frame_idx), source_cache_path, rel_key, local_cache_path))

        if not missing:
            return cache_paths

        frames_dir = scene_dir / "iphone" / "frames"
        total_written = 0
        total_lock = threading.Lock()

        def build_one(item: tuple[int, Path, Path, Path]) -> None:
            nonlocal total_written
            frame_idx, source_cache_path, rel_key, local_cache_path = item
            with self._path_lock(rel_key):
                if self._valid_scannetpp_rgb_cache_file(local_cache_path, target_hw):
                    self._touch_cache_entry(local_cache_path)
                    return
                if self._valid_scannetpp_rgb_cache_file(source_cache_path, target_hw):
                    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = local_cache_path.with_name(
                        f".{local_cache_path.name}.part.{os.getpid()}.{threading.get_ident()}"
                    )
                    try:
                        shutil.copy2(source_cache_path, tmp_path)
                        os.replace(tmp_path, local_cache_path)
                    finally:
                        tmp_path.unlink(missing_ok=True)
                    bytes_written = local_cache_path.stat().st_size
                else:
                    image = _read_rgb_jpg(frames_dir / f"{frame_idx:06d}.jpg")
                    bytes_written = write_rgb_cache_frame(
                        local_cache_path,
                        image,
                        target_hw,
                    )
                self._touch_cache_entry(local_cache_path, force=True)
            with total_lock:
                total_written += int(bytes_written)

        workers = max(
            1,
            min(
                int(os.getenv("SCANNETPP_RGB_CACHE_BUILD_WORKERS", "4") or "4"),
                len(missing),
            ),
        )
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(build_one, item) for item in missing]
                for future in as_completed(futures):
                    future.result()
        else:
            for item in missing:
                build_one(item)

        if total_written:
            self._adjust_cache_usage(total_written)
            self._maybe_evict_cache()
        return cache_paths

    def _prepare_scannetpp_depth_stage(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
        staged_dataset_root: Path,
    ) -> None:
        """Fetch depth frames via Range requests and prepare staging metadata."""
        source_scene_dir = self._scannetpp_source_data_root(adapter) / sequence_name
        # Use full_indices from cache if available, else read from COS path directly
        # Never use sd["scene_dir"] here — it may point to a staged path
        sd = adapter._scene_cache.get(sequence_name)
        if sd is not None and "full_indices" in sd:
            full_indices_arr = sd["full_indices"]
        else:
            from datasets.adapters.scannetpp import _join_frames
            full_indices_arr = _join_frames(source_scene_dir)["full_indices"]
        needed_full_indices = sorted(set(int(full_indices_arr[i]) for i in frame_indices))
        depth_stage = self._stage_scannetpp_depth(
            source_scene_dir,
            staged_dataset_root / sequence_name,
            needed_full_indices,
        )
        if depth_stage.get("mode") == "chunks":
            adapter._staged_depth_chunks_tmp = depth_stage["chunk_dir"]
            adapter.__dict__.pop("_staged_depth_map_tmp", None)
        else:
            adapter._staged_depth_map_tmp = {
                fi: pos for pos, fi in enumerate(needed_full_indices)
            }
            adapter.__dict__.pop("_staged_depth_chunks_tmp", None)

    def _stage_scannetpp_depth(
        self,
        original_scene_dir: Path,
        staged_scene_dir: Path,
        frame_indices: list[int],
    ) -> dict[str, Any]:
        """Fetch only needed depth frames via COS Range requests and write a minimal depth.bin."""
        import pickle, struct
        from concurrent.futures import ThreadPoolExecutor

        rel_key = self._to_cos_key(original_scene_dir)
        depth_key = f"{rel_key}/iphone/depth.bin"
        direct_chunks = os.environ.get(
            "D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

        # Cache chunk index locally; it is small but used for every ScanNet++ sample.
        for attempt in range(2):
            index_cache_path = self._ensure_cached(
                original_scene_dir / "iphone" / "depth_chunk_index.pkl"
            )
            try:
                chunk_offsets: list[tuple[int, int]] = pickle.loads(index_cache_path.read_bytes())
                break
            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                try:
                    index_cache_path.unlink()
                except OSError:
                    pass
                if attempt == 0:
                    continue
                raise

        needed = sorted(set(frame_indices))

        def fetch_chunk(fi: int) -> tuple[int, bytes | None]:
            offset, chunk_size = chunk_offsets[fi]
            chunk_cache_path = (
                self.cache_data_root
                / Path(f"{depth_key}.chunks")
                / f"{fi:08d}.bin"
            )
            if chunk_cache_path.is_file():
                self._touch_cache_entry(chunk_cache_path)
                if direct_chunks:
                    return fi, None
                try:
                    return fi, chunk_cache_path.read_bytes()
                except FileNotFoundError:
                    pass

            lock_key = Path(f"{depth_key}.chunks") / f"{fi:08d}.lock"
            with self._path_lock(lock_key):
                if chunk_cache_path.is_file():
                    self._touch_cache_entry(chunk_cache_path)
                    if direct_chunks:
                        return fi, None
                    try:
                        return fi, chunk_cache_path.read_bytes()
                    except FileNotFoundError:
                        pass

                chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = chunk_cache_path.with_name(
                    f".{chunk_cache_path.name}.part.{os.getpid()}.{threading.get_ident()}"
                )
                try:
                    depth_bin_path = original_scene_dir / "iphone" / "depth.bin"
                    if depth_bin_path.is_file():
                        with open(depth_bin_path, "rb") as df:
                            df.seek(offset)
                            raw = df.read(chunk_size)
                    else:
                        range_header = f"bytes={offset}-{offset + chunk_size - 1}"
                        r = self._get_object(depth_key, range_header=range_header)
                        raw = r["Body"].get_raw_stream().read()
                    with open(tmp_path, "wb") as f:
                        f.write(raw)
                    os.replace(tmp_path, chunk_cache_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
                self._touch_cache_entry(chunk_cache_path, force=True)
                self._maybe_evict_cache()
                return fi, None if direct_chunks else raw

        with ThreadPoolExecutor(max_workers=min(self.config.sdk_workers, len(needed))) as ex:
            chunks = dict(ex.map(lambda fi: fetch_chunk(fi), needed))

        if direct_chunks:
            return {
                "mode": "chunks",
                "chunk_dir": self.cache_data_root / Path(f"{depth_key}.chunks"),
            }

        # write a minimal depth.bin with only the needed frames (re-indexed 0..N-1)
        dst_depth = staged_scene_dir / "iphone" / "depth.bin"
        dst_depth.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_depth, "wb") as f:
            for fi in needed:
                chunk = chunks[fi]
                if chunk is None:
                    chunk = (
                        self.cache_data_root
                        / Path(f"{depth_key}.chunks")
                        / f"{fi:08d}.bin"
                    ).read_bytes()
                f.write(struct.pack("<I", len(chunk)))
                f.write(chunk)
        return {"mode": "minimal"}

    def _manifest_blendedmvs(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in frame_indices:
            frame_id = record.frame_ids[idx]
            paths.append(record.rgb_path(frame_id, adapter.use_masked))
            paths.append(record.depth_path(frame_id))
            paths.append(record.cam_path(frame_id))
        precompute_root = getattr(adapter, "precompute_root", None)
        stage_precomputed = self._stage_blendedmvs_precomputed_enabled()
        if stage_precomputed and precompute_root is not None:
            precomputed_npz = Path(precompute_root) / sequence_name / "precomputed.npz"
            precomputed_h5 = precomputed_npz.with_suffix(".h5")
            max_bytes = int(
                float(os.getenv("D4RT_BLENDEDMVS_STAGE_PRECOMPUTED_MAX_GB", "2.0"))
                * 1024**3
            )
            if precomputed_h5.exists():
                try:
                    if max_bytes <= 0 or precomputed_h5.stat().st_size <= max_bytes:
                        paths.append(precomputed_h5)
                except OSError:
                    pass
            elif precomputed_npz.exists():
                try:
                    if max_bytes <= 0 or precomputed_npz.stat().st_size <= max_bytes:
                        paths.append(precomputed_npz)
                except OSError:
                    pass
        return self._dedupe_keep_order(paths)

    def _manifest_co3dv2(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in frame_indices:
            frame_idx = max(0, min(int(idx), record.num_frames - 1))
            frame_number = record.frame_numbers[frame_idx]
            anno = adapter._get_frame_anno(
                record.category,
                record.sequence_name,
                frame_number,
            )
            paths.append(adapter.root / anno["image"]["path"])
            paths.append(adapter.root / anno["depth"]["path"])
            depth_mask_path = anno.get("depth", {}).get("mask_path")
            if depth_mask_path:
                paths.append(adapter.root / depth_mask_path)
        return self._dedupe_keep_order(paths)

    def _manifest_mvssynth(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in frame_indices:
            paths.append(record.image_path(idx))
            paths.append(record.depth_path(idx))
            paths.append(record.pose_path(idx))
        if frame_indices and frame_indices[0] != 0:
            # When precomputed tracks exist, MVSSynthAdapter reads pose 0 to
            # match the full-sequence origin shift used at precompute time.
            paths.append(record.pose_path(0))
        # precomputed.h5/npz stays on the original COS mount so h5py can slice
        # requested frames without downloading the whole sequence cache.
        return self._dedupe_keep_order(paths)

    def _manifest_tartanair(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        seq_dir = adapter.seq_to_dir[sequence_name]
        image_dir = seq_dir / f"image_{adapter.camera}"
        depth_dir = seq_dir / f"depth_{adapter.camera}"
        paths: list[Path] = [seq_dir / f"pose_{adapter.camera}.txt"]
        for idx in frame_indices:
            paths.append(image_dir / f"{idx:06d}_{adapter.camera}.png")
            paths.append(depth_dir / f"{idx:06d}_{adapter.camera}_depth.npy")

        # precomputed.h5/npz stays on NFS; h5py reads only requested frame
        # chunks directly, same as vkitti2. Files are GB-class and not worth copying.
        return self._dedupe_keep_order(paths)

    def _manifest_vkitti2(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        seq_dir = adapter.seq_to_dir[sequence_name]
        frames_dir = seq_dir / "frames"
        image_dir = adapter._get_actual_dir(frames_dir, "rgb")
        depth_dir = adapter._get_actual_dir(frames_dir, "depth")
        flow_dir = adapter._get_actual_dir(frames_dir, "forwardFlow")
        paths: list[Path] = [
            seq_dir / "extrinsic.txt",
            seq_dir / "intrinsic.txt",
        ]
        for idx in frame_indices:
            paths.append(adapter._frame_path(image_dir, "rgb", idx, ".jpg"))
            paths.append(adapter._frame_path(depth_dir, "depth", idx, ".png"))
            if bool(getattr(adapter, "load_flow", True)):
                flow_path = adapter._frame_path(flow_dir, "flow", idx, ".png")
                if self._has_cached_or_source(flow_path):
                    paths.append(flow_path)
        # Keep precomputed.h5 on the original precompute_root. VKITTI H5 files
        # are sequence-sized (about GB-class), while h5py reads only requested
        # frame chunks from local /data5 in the current training setup.
        return self._dedupe_keep_order(paths)

    def _manifest_waymo(
        self,
        adapter: Any,
        sequence_name: str,
        frame_indices: list[int],
    ) -> list[Path]:
        seq_dir = adapter.root / sequence_name
        camera_name = int(adapter.camera_name)
        paths: list[Path] = [seq_dir / "calib.json"]
        for idx in frame_indices:
            paths.append(seq_dir / f"{idx:05d}_{camera_name}.jpg")
            paths.append(seq_dir / f"{idx:05d}_{camera_name}.npz")

        # Keep sequence-sized precomputed.h5/npz on the original mount.  The
        # adapter reads only requested frame chunks via h5py, while raw jpg/npz
        # frame files are cheap and useful to stage from slow NFS.
        return self._dedupe_keep_order(paths)

    @staticmethod
    def _dedupe_keep_order(paths: list[Path]) -> list[Path]:
        out: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            if path not in seen:
                out.append(path)
                seen.add(path)
        return out

    def _to_cos_key(self, path: Path) -> str:
        path = Path(path)
        for mr in self.all_mount_roots:
            try:
                rel_path = path.relative_to(mr)
                return str(rel_path).replace(os.sep, "/")
            except ValueError:
                try:
                    rel_path = path.resolve().relative_to(mr)
                    return str(rel_path).replace(os.sep, "/")
                except ValueError:
                    continue
        raise ValueError(f"Path {path} is not under any mount root: {self.all_mount_roots}")

    def _has_cached_or_source(self, path: Path) -> bool:
        rel_key = Path(self._to_cos_key(path))
        if (self.cache_data_root / rel_key).is_file():
            return True
        try:
            return Path(path).is_file()
        except OSError:
            return False

    def _materialize_manifest(self, manifest: list[Path], temp_dir: Path) -> dict[str, float | int]:
        stats: dict[str, float | int] = {
            "cold_files": 0,
            "file_max_s": 0.0,
            "file_sum_s": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        stats_lock = threading.Lock()


        def stage_one(src_path: Path) -> None:
            rel_key = Path(self._to_cos_key(src_path))
            cache_path = self.cache_data_root / rel_key
            was_hit = cache_path.is_file()
            dst = temp_dir / rel_key
            dst.parent.mkdir(parents=True, exist_ok=True)
            mode, size = self._link_cached_file(src_path, dst)
            with stats_lock:
                stats["cache_hits" if was_hit else "cache_misses"] += 1
                stats[mode] = int(stats.get(mode, 0)) + 1
                stats["bytes"] = int(stats.get("bytes", 0)) + int(size)

        with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
            futures = [executor.submit(stage_one, path) for path in manifest]
            for future in as_completed(futures):
                future.result()
        return stats

    def _materialize_manifest_cache_only(
        self,
        manifest: list[Path],
    ) -> dict[str, float | int]:
        stats: dict[str, float | int] = {
            "cold_files": 0,
            "file_max_s": 0.0,
            "file_sum_s": 0.0,
        }
        stats_lock = threading.Lock()

        def stage_one(src_path: Path) -> None:
            rel_key = Path(self._to_cos_key(src_path))
            cache_path = self.cache_data_root / rel_key
            cold = not cache_path.is_file()
            t0 = time.perf_counter()
            self._ensure_cached(src_path)
            elapsed = time.perf_counter() - t0
            with stats_lock:
                stats["file_sum_s"] = float(stats["file_sum_s"]) + elapsed
                stats["file_max_s"] = max(float(stats["file_max_s"]), elapsed)
                if cold:
                    stats["cold_files"] = int(stats["cold_files"]) + 1

        with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
            futures = [executor.submit(stage_one, path) for path in manifest]
            for future in as_completed(futures):
                future.result()
        return stats

    def _use_direct_cache_rebase(self, adapter: Any) -> bool:
        enabled = os.environ.get(
            "D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return False
        dataset_name = str(getattr(adapter, "dataset_name", "")).strip()
        if dataset_name == "scannetpp":
            # Safe only when depth uses immutable per-frame chunk files.  Without
            # this mode, staging writes a per-sample minimal depth.bin and must
            # stay in a temporary directory.
            return os.environ.get(
                "D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
        return dataset_name in {
            "dynamic_replica",
            "co3dv2",
            "blendedmvs",
            "kubric",
            "pointodyssey",
            "tartanair",
            "vkitti2",
            "waymo",
        }

    def _path_under_any_mount(self, path: Path) -> bool:
        path = Path(path)
        for mr in self.all_mount_roots:
            try:
                path.relative_to(mr)
                return True
            except ValueError:
                try:
                    path.resolve().relative_to(mr.resolve())
                    return True
                except ValueError:
                    continue
        return False

    def _scannetpp_source_data_root(self, adapter: Any) -> Path:
        data_root = getattr(adapter, "data_root", None)
        if data_root is not None:
            data_root_path = Path(data_root)
            if self._path_under_any_mount(data_root_path):
                return data_root_path
        root = getattr(adapter, "root", None)
        if root is not None:
            return Path(root)
        return Path(data_root)

    def _maybe_prefetch_scene(self, adapter: Any, sequence_name: str) -> None:
        dataset_name = str(getattr(adapter, "dataset_name", "")).strip()
        if dataset_name not in self.config.scene_prefetch_datasets:
            return
        # Non-blocking: check marker first, skip if already prefetched
        marker_path = self._get_scene_marker_path(adapter, sequence_name)
        if marker_path is not None and marker_path.is_file():
            return
        # Fire-and-forget background prefetch (don't block the builder)
        threading.Thread(
            target=self._do_scene_prefetch,
            args=(adapter, dataset_name, sequence_name),
            daemon=True,
        ).start()

    def _get_scene_marker_path(self, adapter: Any, sequence_name: str) -> Optional[Path]:
        dataset_name = str(getattr(adapter, "dataset_name", "")).strip()
        try:
            if dataset_name == "blendedmvs":
                record = adapter._get_record(sequence_name)
                return self.cache_data_root / Path(self._to_cos_key(record.scene_dir)) / ".d4rt_scene_complete"
            elif dataset_name == "co3dv2":
                record = adapter._get_record(sequence_name)
                return self.cache_data_root / Path(self._to_cos_key(adapter.root / record.category / record.sequence_name)) / ".d4rt_scene_complete"
            elif dataset_name == "kubric":
                record = adapter._get_record(sequence_name)
                return self.cache_data_root / Path(self._to_cos_key(record.scene_dir)) / ".d4rt_scene_complete"
            elif dataset_name == "dynamic_replica":
                record = adapter._get_record(sequence_name)
                return self.cache_data_root / Path(self._to_cos_key(adapter.split_root / record.image_rel_paths[0])).parent.parent / ".d4rt_scene_complete"
            elif dataset_name == "scannetpp":
                return self.cache_data_root / Path(self._to_cos_key(adapter.data_root / sequence_name)) / ".d4rt_scene_complete"
        except Exception:
            return None
        return None

    def _do_scene_prefetch(self, adapter: Any, dataset_name: str, sequence_name: str) -> None:
        try:
            if dataset_name == "blendedmvs":
                self._prefetch_blendedmvs_scene(adapter, sequence_name)
            elif dataset_name == "co3dv2":
                self._prefetch_co3dv2_scene(adapter, sequence_name)
            elif dataset_name == "kubric":
                self._prefetch_kubric_scene(adapter, sequence_name)
            elif dataset_name == "dynamic_replica":
                self._prefetch_dynamic_replica_scene(adapter, sequence_name)
            elif dataset_name == "scannetpp":
                self._prefetch_scannetpp_scene(adapter, sequence_name)
        except Exception as e:
            print(f"[ScenePrefetch] {dataset_name}/{sequence_name} failed: {e}", flush=True)

    def _prefetch_blendedmvs_scene(self, adapter: Any, sequence_name: str) -> None:
        record = adapter._get_record(sequence_name)
        scene_rel_key = Path(self._to_cos_key(record.scene_dir))
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._blendedmvs_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=blendedmvs\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _blendedmvs_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for frame_id in record.frame_ids:
            paths.append(record.rgb_path(frame_id, adapter.use_masked))
            paths.append(record.depth_path(frame_id))
            paths.append(record.cam_path(frame_id))
        return self._dedupe_keep_order(paths)

    def _prefetch_co3dv2_scene(self, adapter: Any, sequence_name: str) -> None:
        record = adapter._get_record(sequence_name)
        scene_rel_key = Path(
            self._to_cos_key(adapter.root / record.category / record.sequence_name)
        )
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._co3dv2_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=co3dv2\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _co3dv2_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for fn in record.frame_numbers:
            anno = adapter._get_frame_anno(record.category, record.sequence_name, fn)
            paths.append(adapter.root / anno["image"]["path"])
            paths.append(adapter.root / anno["depth"]["path"])
            depth_mask_path = anno.get("depth", {}).get("mask_path")
            if depth_mask_path:
                paths.append(adapter.root / depth_mask_path)
        return self._dedupe_keep_order(paths)

    def _prefetch_kubric_scene(self, adapter: Any, sequence_name: str) -> None:
        record = adapter._get_record(sequence_name)
        scene_rel_key = Path(self._to_cos_key(record.scene_dir))
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._kubric_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=kubric\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _kubric_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = [record.rank_path]
        if record.h5_path is not None:
            paths.append(record.h5_path)
            if record.depth_names is not None or getattr(record, "has_depth_dir", False):
                paths.extend(
                    record.depth_dir / adapter._depth_name_for_index(record, i)
                    for i in range(record.num_frames)
                )
        else:
            paths.append(record.ann_path)
        paths.extend(
            record.frame_dir / adapter._frame_name_for_index(record, i)
            for i in range(record.num_frames)
        )
        return self._dedupe_keep_order(paths)

    def _prefetch_dynamic_replica_scene(self, adapter: Any, sequence_name: str) -> None:
        record = adapter._get_record(sequence_name)
        scene_rel_key = Path(
            self._to_cos_key(adapter.split_root / record.image_rel_paths[0])
        ).parent.parent
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._dynamic_replica_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=dynamic_replica\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _dynamic_replica_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        record = adapter._get_record(sequence_name)
        paths: list[Path] = []
        for idx in range(record.num_frames):
            paths.append(adapter.split_root / record.image_rel_paths[idx])
            paths.append(adapter.split_root / record.depth_rel_paths[idx])
            traj_rel = record.traj_rel_paths[idx]
            if traj_rel is not None:
                paths.append(adapter.split_root / traj_rel)
        return self._dedupe_keep_order(paths)

    def _prefetch_scannetpp_scene(self, adapter: Any, sequence_name: str) -> None:
        scene_dir = adapter.data_root / sequence_name
        scene_rel_key = Path(self._to_cos_key(scene_dir))
        marker_path = self.cache_data_root / scene_rel_key / ".d4rt_scene_complete"
        if marker_path.is_file():
            return

        scene_paths = self._scannetpp_scene_paths(adapter, sequence_name)
        lock_key = scene_rel_key / ".d4rt_scene_prefetch_lock"
        with self._path_lock(lock_key):
            if marker_path.is_file():
                return

            with ThreadPoolExecutor(max_workers=self.config.sdk_workers) as executor:
                futures = [executor.submit(self._ensure_cached, path) for path in scene_paths]
                for future in as_completed(futures):
                    future.result()

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_marker = marker_path.with_name(
                f".{marker_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                tmp_marker.write_text(
                    f"dataset=scannetpp\nsequence={sequence_name}\nfiles={len(scene_paths)}\n",
                    encoding="utf-8",
                )
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            self._maybe_evict_cache()

    def _scannetpp_scene_paths(self, adapter: Any, sequence_name: str) -> list[Path]:
        scene_dir = adapter.data_root / sequence_name
        colmap_dir = scene_dir / "iphone" / "colmap"
        frames_dir = scene_dir / "iphone" / "frames"
        paths: list[Path] = [
            colmap_dir / "cameras.txt",
            colmap_dir / "images.txt",
            scene_dir / "iphone" / "pose_intrinsic_imu.json",
        ]
        num_frames = adapter.get_num_frames(sequence_name)
        paths += [frames_dir / f"{i:06d}.jpg" for i in range(num_frames)]
        if getattr(adapter, "use_precomputed_tracks", False):
            index_path = scene_dir / "precomputed.h5_chunk_index.pkl"
            if index_path.exists():
                paths.append(index_path)
        return self._dedupe_keep_order(paths)

    def _link_cached_file(self, src_path: Path, dst: Path) -> tuple[str, int]:
        for attempt in range(2):
            cache_path = self._ensure_cached(src_path)
            try:
                os.link(cache_path, dst)
                return "linked", cache_path.stat().st_size
            except FileNotFoundError:
                if attempt == 0:
                    continue
                raise
            except OSError as exc:
                if exc.errno in {errno.EXDEV, errno.EPERM, errno.EACCES}:
                    try:
                        shutil.copy2(cache_path, dst)
                        return "copied", cache_path.stat().st_size
                    except FileNotFoundError:
                        if attempt == 0:
                            continue
                        raise
                raise
        raise FileNotFoundError(src_path)

    def _ensure_cached(self, src_path: Path) -> Path:
        src_path = Path(src_path)
        rel_key = Path(self._to_cos_key(src_path))
        cache_path = self.cache_data_root / rel_key
        total_t0 = time.perf_counter()
        if cache_path.is_file():
            self._touch_cache_entry(cache_path)
            return cache_path

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_wait_s = 0.0
        download_s = 0.0
        downloaded = False

        # Non-blocking lock: poll with short sleeps instead of blocking forever.
        # This avoids the convoy effect where 17 workers wait 12s+ for one
        # worker to finish downloading a file from NFS.
        lock_t0 = time.perf_counter()
        poll_interval = 0.05
        max_poll_s = 30.0
        acquired = False
        digest = hashlib.sha1(rel_key.as_posix().encode("utf-8")).hexdigest()
        lock_dir = self.lock_root / digest[:2] / digest[2:4]
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{digest}.lock"

        while True:
            # Check if file appeared (another worker finished downloading)
            if cache_path.is_file():
                lock_wait_s = time.perf_counter() - lock_t0
                self._touch_cache_entry(cache_path)
                return cache_path

            # Try non-blocking lock
            lock_f = open(lock_path, "a+b")
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                lock_f.close()
                elapsed = time.perf_counter() - lock_t0
                if elapsed > max_poll_s:
                    # Fallback: blocking acquire
                    lock_f = open(lock_path, "a+b")
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                    acquired = True
                    break
                time.sleep(poll_interval)
                # Exponential backoff up to 0.5s
                poll_interval = min(poll_interval * 1.5, 0.5)

        lock_wait_s = time.perf_counter() - lock_t0
        try:
            if cache_path.is_file():
                self._touch_cache_entry(cache_path)
                total_s = time.perf_counter() - total_t0
                slow_s = float(os.environ.get("D4RT_SAMPLE_STAGE_SLOW_ENSURE_S", "8.0"))
                if total_s >= slow_s:
                    try:
                        size = cache_path.stat().st_size
                    except OSError:
                        size = 0
                    print(
                        f"[SampleStageEnsureSlow] key={rel_key.as_posix()} "
                        f"total={total_s:.3f}s lock_wait={lock_wait_s:.3f}s "
                        f"download=0.000s downloaded=0 size={size} "
                        f"timeout={self.config.request_timeout_s:.1f}s "
                        f"retries={self.config.request_retries}",
                        flush=True,
                    )
                return cache_path
            download_t0 = time.perf_counter()
            self._download_to_cache(src_path, cache_path, rel_key)
            download_s = time.perf_counter() - download_t0
            downloaded = True
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            lock_f.close()

        self._touch_cache_entry(cache_path, force=True)
        self._maybe_evict_cache()
        total_s = time.perf_counter() - total_t0
        slow_s = float(os.environ.get("D4RT_SAMPLE_STAGE_SLOW_ENSURE_S", "8.0"))
        if total_s >= slow_s:
            try:
                size = cache_path.stat().st_size
            except OSError:
                size = 0
            print(
                f"[SampleStageEnsureSlow] key={rel_key.as_posix()} "
                f"total={total_s:.3f}s lock_wait={lock_wait_s:.3f}s "
                f"download={download_s:.3f}s downloaded={int(downloaded)} "
                f"size={size} timeout={self.config.request_timeout_s:.1f}s "
                f"retries={self.config.request_retries}",
                flush=True,
            )
        return cache_path

    def _download_to_cache(self, src_path: Path, cache_path: Path, rel_key: Path) -> None:
        key = rel_key.as_posix()
        tmp_path = cache_path.with_name(f".{cache_path.name}.part.{os.getpid()}.{threading.get_ident()}")
        t0 = time.perf_counter()
        try:
            if src_path.is_file():
                shutil.copy2(src_path, tmp_path)
            else:
                response = self._get_object(key)
                stream = response["Body"].get_raw_stream()
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            with self._cache_usage_lock():
                os.replace(tmp_path, cache_path)
                accounted_bytes = self._read_cache_usage_unlocked()
                if accounted_bytes is not None:
                    self._write_cache_usage_unlocked(
                        accounted_bytes + cache_path.stat().st_size
                    )
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            elapsed = time.perf_counter() - t0
            slow_s = float(os.environ.get("D4RT_SAMPLE_STAGE_SLOW_DOWNLOAD_S", "5.0"))
            if elapsed >= slow_s:
                size = cache_path.stat().st_size if cache_path.exists() else 0
                print(
                    f"[SampleStageDownloadSlow] key={key} "
                    f"time={elapsed:.3f}s size={size} "
                    f"timeout={self.config.request_timeout_s:.1f}s "
                    f"retries={self.config.request_retries}",
                    flush=True,
                )

    def _touch_cache_entry(self, cache_path: Path, force: bool = False) -> None:
        interval = self.config.cache_touch_interval_s
        if interval <= 0:
            return
        now = time.time()
        if not force:
            try:
                if now - cache_path.stat().st_mtime < interval:
                    return
            except OSError:
                return
        try:
            os.utime(cache_path, (now, now), follow_symlinks=False)
        except OSError:
            pass

    def evict_cache_once(
        self,
        emit_log: bool = True,
        *,
        force_low_watermark: bool = False,
    ) -> dict[str, float | int] | None:
        """Run one cache eviction pass.

        Normally eviction only runs above the hard cap.  ``force_low_watermark``
        is for startup pre-cleaning: it trims to the low watermark even when the
        cache is merely close to the cap, so training does not pay that cost in
        the first few hundred batches.
        """
        max_bytes = self.config.cache_max_bytes
        if max_bytes <= 0:
            return None

        eviction_lock = self.lock_root / "eviction.lock"
        try:
            if time.time() - eviction_lock.stat().st_mtime < self.config.cache_scan_interval_s:
                return None
        except OSError:
            pass

        with open(eviction_lock, "a+b") as lock_f:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            try:
                started = time.perf_counter()
                try:
                    if time.time() - eviction_lock.stat().st_mtime < self.config.cache_scan_interval_s:
                        return None
                except OSError:
                    pass
                stats = self._evict_cache_locked(
                    force_low_watermark=force_low_watermark
                )
                os.utime(eviction_lock, None)
                elapsed = time.perf_counter() - started
                slow_s = float(os.environ.get("D4RT_SAMPLE_STAGE_SLOW_EVICT_S", "2.0"))
                if emit_log and elapsed >= slow_s:
                    print(
                        f"[SampleStageEvictSlow] elapsed={elapsed:.3f}s "
                        f"force_low_watermark={int(force_low_watermark)} "
                        f"max_bytes={self.config.cache_max_bytes} "
                        f"low_watermark={self.config.cache_low_watermark_ratio:.3f} "
                        f"files={stats['files']} "
                        f"total_before={stats['total_before']} "
                        f"total_after={stats['total_after']} "
                        f"deleted_files={stats['deleted_files']} "
                        f"deleted_bytes={stats['deleted_bytes']} "
                        f"scan={stats['scan_s']:.3f}s "
                        f"delete={stats['delete_s']:.3f}s",
                        flush=True,
                    )
                return stats
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def _path_lock(self, rel_key: Path) -> Iterator[None]:
        digest = hashlib.sha1(rel_key.as_posix().encode("utf-8")).hexdigest()
        lock_dir = self.lock_root / digest[:2] / digest[2:4]
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{digest}.lock"
        with open(lock_path, "a+b") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def _cache_usage_lock(self) -> Iterator[None]:
        self.cache_usage_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_usage_lock_path, "a+b") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _read_cache_usage_unlocked(self) -> int | None:
        try:
            raw = self.cache_usage_path.read_text(encoding="utf-8").strip().split()
        except OSError:
            return None
        if not raw:
            return None
        try:
            value = int(raw[0])
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def _write_cache_usage_unlocked(self, total_bytes: int) -> None:
        tmp_path = self.cache_usage_path.with_name(
            f".{self.cache_usage_path.name}.part.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            tmp_path.write_text(
                f"{max(0, int(total_bytes))} {time.time():.6f}\n",
                encoding="utf-8",
            )
            os.replace(tmp_path, self.cache_usage_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _adjust_cache_usage(self, delta_bytes: int) -> None:
        if delta_bytes == 0:
            return
        with self._cache_usage_lock():
            current = self._read_cache_usage_unlocked()
            if current is None:
                # Existing caches from older runs are calibrated by the janitor's
                # next full scan. Avoid creating a misleading partial counter.
                return
            self._write_cache_usage_unlocked(max(0, current + int(delta_bytes)))

    def _usage_full_rescan_due_unlocked(self) -> bool:
        try:
            interval_s = float(
                os.environ.get("D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S", "600")
            )
        except ValueError:
            interval_s = 600.0
        if interval_s <= 0:
            return True
        try:
            return time.time() - self.cache_usage_scan_stamp_path.stat().st_mtime >= interval_s
        except OSError:
            return True

    def _mark_usage_full_rescan_unlocked(self) -> None:
        try:
            self.cache_usage_scan_stamp_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_usage_scan_stamp_path.touch()
        except OSError:
            pass

    def _maybe_evict_cache(self) -> None:
        if self.config.eviction_mode != "background":
            return
        max_bytes = self.config.cache_max_bytes
        if max_bytes <= 0:
            return

        now = time.time()
        if now - self._last_cache_scan_s < self.config.cache_scan_interval_s:
            return
        self._last_cache_scan_s = now

        eviction_lock = self.lock_root / "eviction.lock"
        try:
            if now - eviction_lock.stat().st_mtime < self.config.cache_scan_interval_s:
                return
        except OSError:
            pass

        thread = self._eviction_thread
        if thread is not None and thread.is_alive():
            return

        def evict_in_background() -> None:
            self.evict_cache_once(emit_log=True)

        self._eviction_thread = threading.Thread(
            target=evict_in_background,
            name="SampleStageEvict",
            daemon=True,
        )
        self._eviction_thread.start()

    def _evict_cache_locked(
        self,
        *,
        force_low_watermark: bool = False,
    ) -> dict[str, float | int]:
        with self._cache_usage_lock():
            accounted_bytes = self._read_cache_usage_unlocked()
            full_rescan_due = self._usage_full_rescan_due_unlocked()
            if (
                not force_low_watermark
                and
                accounted_bytes is not None
                and accounted_bytes <= self.config.cache_max_bytes
                and not full_rescan_due
            ):
                return {
                    "files": -1,
                    "total_before": accounted_bytes,
                    "total_after": accounted_bytes,
                    "deleted_files": 0,
                    "deleted_bytes": 0,
                    "scan_s": 0.0,
                    "delete_s": 0.0,
                }
        # Lock released — scan and delete run without holding it.
        return self._evict_cache_scan_locked(
            force_low_watermark=force_low_watermark
        )

    def _evict_cache_scan_locked(
        self,
        *,
        force_low_watermark: bool = False,
    ) -> dict[str, float | int]:
        entries: list[tuple[float, int, str]] = []
        total_bytes = 0
        scan_t0 = time.perf_counter()
        cache_root_str = os.fspath(self.cache_data_root)
        pinned_paths = self._load_pinned_manifest_paths()
        pinned_dirs = tuple(
            path for path in pinned_paths if path.endswith(os.sep)
        )
        pinned_exact = (
            pinned_paths - set(pinned_dirs) if pinned_dirs else pinned_paths
        )
        for dirpath, _, filenames in os.walk(cache_root_str):
            for name in filenames:
                if name.startswith(".") or ".part." in name:
                    continue
                path_str = os.path.join(dirpath, name)
                try:
                    stat = os.stat(path_str, follow_symlinks=False)
                except OSError:
                    continue
                total_bytes += stat.st_size
                abs_path = os.path.abspath(path_str)
                if abs_path in pinned_exact or (
                    pinned_dirs and abs_path.startswith(pinned_dirs)
                ):
                    continue
                if name == "frame_index.pkl" and self._is_persistent_sidecar_str(
                    path_str, cache_root_str
                ):
                    continue
                entries.append((stat.st_mtime, stat.st_size, path_str))
        scan_s = time.perf_counter() - scan_t0

        stats: dict[str, float | int] = {
            "files": len(entries),
            "total_before": total_bytes,
            "total_after": total_bytes,
            "deleted_files": 0,
            "deleted_bytes": 0,
            "scan_s": scan_s,
            "delete_s": 0.0,
        }

        target_bytes = int(
            self.config.cache_max_bytes * self.config.cache_low_watermark_ratio
        )
        if total_bytes <= self.config.cache_max_bytes and (
            not force_low_watermark or total_bytes <= target_bytes
        ):
            with self._cache_usage_lock():
                self._write_cache_usage_unlocked(total_bytes)
                self._mark_usage_full_rescan_unlocked()
            return stats

        entries.sort(key=lambda item: item[0])
        delete_t0 = time.perf_counter()
        for _, size, path_str in entries:
            if total_bytes <= target_bytes:
                break
            path = Path(path_str)
            self._invalidate_scene_marker_for_path(path)
            try:
                os.unlink(path_str)
            except FileNotFoundError:
                continue
            except OSError:
                continue
            total_bytes -= size
            stats["deleted_files"] = int(stats["deleted_files"]) + 1
            stats["deleted_bytes"] = int(stats["deleted_bytes"]) + size
        stats["total_after"] = total_bytes
        stats["delete_s"] = time.perf_counter() - delete_t0
        with self._cache_usage_lock():
            self._write_cache_usage_unlocked(total_bytes)
            self._mark_usage_full_rescan_unlocked()
        return stats

    def _load_pinned_manifest_paths(self) -> set[str]:
        root = self._pinned_manifest_root
        if root is None:
            return set()
        try:
            generation_dirs = sorted(
                (p for p in root.glob("g*") if p.is_dir()),
                key=lambda p: p.name,
                reverse=True,
            )
        except OSError:
            return set()
        pinned: set[str] = set()
        for gen_dir in generation_dirs[:2]:
            try:
                manifests = sorted(gen_dir.glob("block_*.manifest"))
            except OSError:
                continue
            for manifest in manifests:
                try:
                    for raw in manifest.read_text(encoding="utf-8").splitlines():
                        line = raw.strip()
                        if line:
                            is_dir = line.endswith("/")
                            abs_line = os.path.abspath(line)
                            if is_dir:
                                abs_line = abs_line.rstrip(os.sep) + os.sep
                            pinned.add(abs_line)
                except OSError:
                    continue
        return pinned

    def _is_persistent_sidecar_str(self, cache_path: str, cache_root: str) -> bool:
        """Fast string variant for hot eviction scans."""
        try:
            rel_path = os.path.relpath(cache_path, cache_root)
        except ValueError:
            return False
        rel_parts = rel_path.split(os.sep)
        return (
            len(rel_parts) >= 6
            and rel_parts[0] == "hdu_datasets"
            and rel_parts[1] == "scannetpp"
            and rel_parts[2] == "data"
            and rel_parts[-2:] == ["iphone", "frame_index.pkl"]
        )

    def _is_persistent_sidecar(self, cache_path: Path) -> bool:
        """Keep generated metadata sidecars that avoid expensive COS reads."""
        try:
            rel_parts = cache_path.relative_to(self.cache_data_root).parts
        except ValueError:
            return False
        return (
            len(rel_parts) >= 6
            and rel_parts[0] == "hdu_datasets"
            and rel_parts[1] == "scannetpp"
            and rel_parts[2] == "data"
            and rel_parts[-2:] == ("iphone", "frame_index.pkl")
        )

    def _invalidate_scene_marker_for_path(self, cache_path: Path) -> None:
        try:
            rel_parts = cache_path.relative_to(self.cache_data_root).parts
        except ValueError:
            return
        if len(rel_parts) < 4:
            return
        if cache_path.name == ".d4rt_scene_complete":
            return

        # BlendedMVS: hdu_datasets/BlendedMVS/<scene>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "BlendedMVS" and len(rel_parts) >= 3:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2], ".d4rt_scene_complete"
            )
            marker_path.unlink(missing_ok=True)
            return

        # Co3Dv2: hdu_datasets/Co3Dv2/<category>/<sequence>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "Co3Dv2" and len(rel_parts) >= 4:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3],
                ".d4rt_scene_complete",
            )
            marker_path.unlink(missing_ok=True)
            return

        # Kubric: hdu_datasets/Kubric/<scene_id>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "Kubric" and len(rel_parts) >= 3:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2],
                ".d4rt_scene_complete",
            )
            marker_path.unlink(missing_ok=True)
            return

        # DynamicReplica: hdu_datasets/Dynamic_Replica/<split>/<seq>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "Dynamic_Replica" and len(rel_parts) >= 4:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3],
                ".d4rt_scene_complete",
            )
            marker_path.unlink(missing_ok=True)
            return

        # ScanNet++: hdu_datasets/scannetpp/data/<scene>/...
        if rel_parts[0] == "hdu_datasets" and rel_parts[1] == "scannetpp" and len(rel_parts) >= 4:
            marker_path = self.cache_data_root.joinpath(
                rel_parts[0], rel_parts[1], rel_parts[2], rel_parts[3],
                ".d4rt_scene_complete",
            )
            marker_path.unlink(missing_ok=True)

    @contextlib.contextmanager
    def _rebase_adapter(
        self,
        adapter: Any,
        sequence_name: str,
        staged_dataset_root: Path,
    ) -> Iterator[None]:
        dataset_name = str(adapter.dataset_name)
        if dataset_name == "kubric":
            old_root = adapter.root
            old_record = adapter._name_to_record[sequence_name]
            new_record = dataclasses.replace(
                old_record,
                scene_dir=staged_dataset_root / old_record.scene_dir.relative_to(old_root),
                ann_path=staged_dataset_root / old_record.ann_path.relative_to(old_root),
                rank_path=staged_dataset_root / old_record.rank_path.relative_to(old_root),
                h5_path=(
                    staged_dataset_root / old_record.h5_path.relative_to(old_root)
                    if old_record.h5_path is not None
                    else None
                ),
                trajs_2d_path=(
                    staged_dataset_root / old_record.trajs_2d_path.relative_to(old_root)
                    if old_record.trajs_2d_path is not None
                    else None
                ),
            )
            adapter.root = staged_dataset_root
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter._name_to_record[sequence_name] = old_record
            return

        if dataset_name == "pointodyssey":
            old_root = adapter.root
            old_record = adapter.name_to_record[sequence_name]
            new_record = dataclasses.replace(
                old_record,
                sequence_root=self._rebase_pathlike(
                    old_record.sequence_root, old_root, staged_dataset_root
                ),
                rgb_paths=self._rebase_pathlike_list(
                    old_record.rgb_paths, old_root, staged_dataset_root
                ),
                depth_paths=self._rebase_pathlike_list(
                    old_record.depth_paths, old_root, staged_dataset_root
                ),
                normal_paths=self._rebase_pathlike_list(
                    old_record.normal_paths, old_root, staged_dataset_root
                ),
                mask_paths=self._rebase_pathlike_list(
                    old_record.mask_paths, old_root, staged_dataset_root
                ),
                # Keep heavy annotation files on /data_cos so h5py can slice
                # the needed frames directly instead of downloading the whole
                # per-scene anno.h5 into the sample stage.
                anno_path=(
                    self._rebase_pathlike(
                        old_record.anno_path, old_root, staged_dataset_root
                    )
                    if self._pointodyssey_stage_anno_h5(old_record)
                    else old_record.anno_path
                ),
                info_path=old_record.info_path,
                scene_info_path=self._rebase_pathlike(
                    old_record.scene_info_path, old_root, staged_dataset_root
                ),
                fast_dir=old_record.fast_dir,
                fast_anno_paths=old_record.fast_anno_paths,
                encoded_cache_paths=old_record.encoded_cache_paths,
            )
            adapter.root = staged_dataset_root
            adapter.name_to_record[sequence_name] = new_record
            adapter._scene_info_cache.clear()
            adapter._encoded_cache_store.clear()
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.name_to_record[sequence_name] = old_record
                adapter._scene_info_cache.clear()
                adapter._encoded_cache_store.clear()
            return

        if dataset_name == "dynamic_replica":
            old_root = adapter.root
            old_split_root = adapter.split_root
            old_record = adapter._name_to_record[sequence_name]
            new_record = dataclasses.replace(
                old_record,
                sequence_dir=staged_dataset_root / old_record.sequence_dir.relative_to(old_root),
            )
            adapter.root = staged_dataset_root
            adapter.split_root = staged_dataset_root / adapter.split
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.split_root = old_split_root
                adapter._name_to_record[sequence_name] = old_record
            return

        if dataset_name == "blendedmvs":
            old_root = adapter.root
            old_precompute_root = getattr(adapter, "precompute_root", None)
            old_record = adapter._name_to_record[sequence_name]
            new_record = old_record.__class__(
                scene_id=old_record.scene_id,
                scene_dir=staged_dataset_root / old_record.scene_dir.relative_to(old_root),
                frame_ids=old_record.frame_ids,
            )
            new_precompute_root = old_precompute_root
            if (
                old_precompute_root is not None
                and self._stage_blendedmvs_precomputed_enabled()
            ):
                try:
                    new_precompute_root = (
                        staged_dataset_root / Path(old_precompute_root).relative_to(old_root)
                    )
                except ValueError:
                    new_precompute_root = old_precompute_root
            adapter.root = staged_dataset_root
            adapter.precompute_root = new_precompute_root
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.precompute_root = old_precompute_root
                adapter._name_to_record[sequence_name] = old_record
            return

        if dataset_name == "scannetpp":
            source_data_root = self._scannetpp_source_data_root(adapter)
            old_data_root = source_data_root
            old_scene_cache = adapter._scene_cache.get(sequence_name)
            restore_scene_cache = old_scene_cache
            if old_scene_cache is not None:
                old_scene_cache = self._normalize_scannetpp_scene_cache(
                    old_scene_cache,
                    sequence_name,
                    source_data_root,
                )
                restore_scene_cache = old_scene_cache
                new_scene_data = dict(old_scene_cache)
                old_precomputed_dir = old_scene_cache.get(
                    "_precomputed_dir", old_scene_cache["scene_dir"]
                )
                local_precomputed_dir = self._local_precomputed_dir(
                    old_precomputed_dir,
                    old_data_root,
                    staged_dataset_root,
                )
                new_scene_data["scene_dir"] = (
                    staged_dataset_root
                    / old_scene_cache["scene_dir"].relative_to(old_data_root)
                )
                local_h5 = local_precomputed_dir / adapter.precomputed_name
                local_h5 = local_h5.with_suffix(".h5")
                local_index = adapter._precomputed_h5_chunk_index_path(
                    local_precomputed_dir
                )
                if local_h5.is_file() and local_index.is_file():
                    new_scene_data["_precomputed_dir"] = local_precomputed_dir
                    new_scene_data["_precomputed_index_dir"] = local_precomputed_dir
                else:
                    new_scene_data["_precomputed_dir"] = old_precomputed_dir
                # _staged_depth_map set by _prepare_scannetpp_depth_stage in stage_sample
                staged_depth_map = getattr(adapter, "_staged_depth_map_tmp", None)
                staged_depth_chunks = getattr(adapter, "_staged_depth_chunks_tmp", None)
                if staged_depth_chunks is not None:
                    new_scene_data["_staged_depth_chunks_dir"] = Path(staged_depth_chunks)
                elif staged_depth_map is not None:
                    new_scene_data["_staged_depth_map"] = staged_depth_map
                adapter._scene_cache[sequence_name] = new_scene_data
                adapter._depth_chunk_cache.pop(sequence_name, None)
            adapter.data_root = staged_dataset_root
            try:
                yield
            finally:
                adapter.data_root = old_data_root
                if restore_scene_cache is not None:
                    adapter._scene_cache[sequence_name] = restore_scene_cache
                else:
                    adapter._scene_cache.pop(sequence_name, None)
                adapter._depth_chunk_cache.pop(sequence_name, None)
                adapter.__dict__.pop("_staged_depth_map_tmp", None)
                adapter.__dict__.pop("_staged_depth_chunks_tmp", None)
            return

        if dataset_name == "scannet":
            old_root = adapter.root  # /data_cos/hdu_datasets/scannet
            old_scans_dir = adapter._scans_dir
            old_precompute_root = adapter.precompute_root
            old_record = adapter._name_to_record[sequence_name]
            def _rb(p: Path) -> Path:
                try:
                    return staged_dataset_root / p.relative_to(old_root)
                except ValueError:
                    return p
            from datasets.adapters.scannet import _SequenceRecord as _ScanNetRecord
            new_record = _ScanNetRecord(
                scene_id=old_record.scene_id,
                scene_dir=_rb(old_record.scene_dir),
                processed_dir=_rb(old_record.processed_dir),
                image_dir=_rb(old_record.image_dir),
                depth_dir=_rb(old_record.depth_dir),
                pose_dir=_rb(old_record.pose_dir),
                precomputed_path=old_record.precomputed_path,  # keep on COS for h5py slicing
                mesh_path=old_record.mesh_path,
                num_frames=old_record.num_frames,
                color_hw=(old_record.color_height, old_record.color_width),
                depth_hw=(old_record.depth_height, old_record.depth_width),
                has_axis_alignment=old_record.has_axis_alignment,
            )
            adapter.root = staged_dataset_root
            adapter._scans_dir = staged_dataset_root / old_scans_dir.relative_to(old_root)
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter._scans_dir = old_scans_dir
                adapter.precompute_root = old_precompute_root
                adapter._name_to_record[sequence_name] = old_record
            return

        if dataset_name == "co3dv2":
            old_root = adapter.root
            old_precompute_root = adapter.precompute_root
            old_track_precompute_root = adapter.track_precompute_root
            old_record = adapter._uid_to_record[sequence_name]
            new_record = dataclasses.replace(
                old_record,
                sequence_dir=(
                    staged_dataset_root
                    / old_record.category
                    / old_record.sequence_name
                ),
                # Co3Dv2 COS planned mode stages only raw RGB/depth/mask files.
                # Keep precomputed tracks disabled in staged samples so h5/npz
                # checks cannot fall back to the slow /data_cos mount.
                has_precomputed=False,
            )
            adapter.root = staged_dataset_root
            adapter.precompute_root = None
            adapter.track_precompute_root = None
            adapter._uid_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.precompute_root = old_precompute_root
                adapter.track_precompute_root = old_track_precompute_root
                adapter._uid_to_record[sequence_name] = old_record
            return

        if dataset_name == "mvssynth":
            old_root = adapter.root
            old_precompute_root = adapter.precompute_root
            old_record = adapter._name_to_record[sequence_name]
            new_record = old_record.__class__(
                sequence_id=old_record.sequence_id,
                sequence_dir=staged_dataset_root / old_record.sequence_dir.relative_to(old_root),
                num_frames=old_record.num_frames,
            )
            adapter.root = staged_dataset_root
            # Keep precomputed_root on the original mount, matching ScanNet++:
            # raw per-frame files are staged, while h5/npz caches are sliced in
            # place instead of copied into each sample directory.
            adapter.precompute_root = old_precompute_root
            adapter._name_to_record[sequence_name] = new_record
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.precompute_root = old_precompute_root
                adapter._name_to_record[sequence_name] = old_record
            return

        if dataset_name == "tartanair":
            old_root = adapter.root
            old_precompute_root = adapter.precompute_root
            old_seq_dir = adapter.seq_to_dir[sequence_name]
            new_seq_dir = staged_dataset_root / old_seq_dir.relative_to(old_root)
            adapter.root = staged_dataset_root
            adapter.seq_to_dir[sequence_name] = new_seq_dir
            # TartanAir staging copies only per-frame RGB/depth/pose files.
            # Keep sequence-sized precomputed.h5/npz on the original root so
            # h5py can slice requested frames instead of expecting a copied H5
            # inside the transient staged sample directory.
            adapter.precompute_root = old_precompute_root
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.precompute_root = old_precompute_root
                adapter.seq_to_dir[sequence_name] = old_seq_dir
            return

        if dataset_name == "vkitti2":
            old_root = adapter.root
            old_seq_dir = adapter.seq_to_dir[sequence_name]
            new_seq_dir = staged_dataset_root / old_seq_dir.relative_to(old_root)
            adapter.root = staged_dataset_root
            adapter.seq_to_dir[sequence_name] = new_seq_dir
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.seq_to_dir[sequence_name] = old_seq_dir
            return

        if dataset_name == "waymo":
            old_root = adapter.root
            old_precompute_root = adapter.precompute_root
            adapter.root = staged_dataset_root
            # Staging copies only raw per-frame files.  Keep H5/NPZ track cache
            # on the original root so the adapter can slice large sequence
            # files in place.
            adapter.precompute_root = old_precompute_root
            try:
                yield
            finally:
                adapter.root = old_root
                adapter.precompute_root = old_precompute_root
            return

        yield

    def _local_precomputed_dir(
        self,
        old_precomputed_dir: Path,
        old_data_root: Path,
        staged_dataset_root: Path,
    ) -> Path:
        try:
            cache_dir = self.cache_data_root / Path(self._to_cos_key(old_precomputed_dir))
            local_h5 = cache_dir / "precomputed.h5"
            local_index = cache_dir / "precomputed.h5_chunk_index.pkl"
            if local_h5.is_file() and local_index.is_file():
                return cache_dir
        except Exception:
            pass
        return staged_dataset_root / old_precomputed_dir.relative_to(old_data_root)

    def _normalize_scannetpp_scene_cache(
        self,
        scene_cache: dict[str, Any],
        sequence_name: str,
        source_data_root: Path,
    ) -> dict[str, Any]:
        """Drop staged-local paths from a cached ScanNet++ scene record."""
        normalized = dict(scene_cache)
        source_scene_dir = source_data_root / sequence_name
        scene_dir = Path(normalized.get("scene_dir", source_scene_dir))
        if not self._path_under_any_mount(scene_dir):
            normalized["scene_dir"] = source_scene_dir

        for key in ("_precomputed_dir", "_precomputed_index_dir"):
            value = normalized.get(key)
            if value is not None and not self._path_under_any_mount(Path(value)):
                normalized[key] = source_scene_dir

        normalized.pop("_staged_depth_map", None)
        normalized.pop("_staged_depth_chunks_dir", None)
        return normalized

    @staticmethod
    def _pointodyssey_stage_anno_h5(record: Any) -> bool:
        enabled = os.getenv("D4RT_POINTODYSSEY_STAGE_ANNO_H5", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        return enabled and getattr(record, "anno_path", None) is not None

    @staticmethod
    def _rebase_pathlike(
        value: Any,
        old_root: Path,
        staged_dataset_root: Path,
    ) -> Any:
        if value is None:
            return None
        path_value = Path(value)
        try:
            return staged_dataset_root / path_value.relative_to(old_root)
        except Exception:
            return value

    def _rebase_pathlike_list(
        self,
        values: Any,
        old_root: Path,
        staged_dataset_root: Path,
    ) -> Any:
        if values is None:
            return None
        return [
            self._rebase_pathlike(value, old_root, staged_dataset_root)
            for value in values
        ]


def build_sample_stager(raw_config: Optional[dict[str, Any]]) -> Optional[SampleLocalStager]:
    if not raw_config:
        return None
    config = SampleStageConfig.from_dict(raw_config)
    if not config.backend:
        return None
    return SampleLocalStager(config)
