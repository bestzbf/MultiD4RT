from __future__ import annotations

import os
import glob
import hashlib
import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import BaseAdapter, UnifiedClip


def _resolve_positive_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(1, out)


def _resolve_nonnegative_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, out)


def _resolve_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sleep_before_retry(delay_s: float, attempt: int) -> None:
    if delay_s <= 0:
        return
    time.sleep(float(delay_s) * float(attempt + 1))


def _read_tartanair_image(path: Path, retries: int, retry_delay_s: float) -> np.ndarray:
    last_error: Exception | None = None
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        last_error = FileNotFoundError(f"Missing or unreadable TartanAir image: {path}")
        if attempt + 1 < attempts:
            _sleep_before_retry(retry_delay_s, attempt)
    raise last_error if last_error is not None else FileNotFoundError(path)


def _read_tartanair_depth(
    path: Path,
    retries: int,
    retry_delay_s: float,
    depth_max: float = 200.0,
) -> np.ndarray:
    last_error: Exception | None = None
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        try:
            depth = np.load(str(path)).astype(np.float32)
            depth = np.where(depth > float(depth_max), np.nan, depth)
            return depth
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                _sleep_before_retry(retry_delay_s, attempt)
    raise RuntimeError(f"Failed to load TartanAir depth after {attempts} attempts: {path}") from last_error


def quat_to_rotation_matrix(quat):
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = quat
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ], dtype=np.float32)


def _project_static_points_to_clip(
    points_world: np.ndarray,
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_consistency_thresh: float = 0.05,
    depth_max: float = 200.0,
    depth_support_radius: int = 2,
    visible_abs_thresh: float = 0.25,
    occlusion_abs_thresh: float = 1.0,
    occlusion_rel_thresh: float = 0.08,
    occlusion_min_support: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project static ENU world points into a requested TartanAir clip."""
    points_world = np.asarray(points_world, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    T = len(depths)
    N = points_world.shape[0]
    trajs_2d = np.zeros((T, N, 2), dtype=np.float32)
    valids = np.zeros((T, N), dtype=bool)
    visibs = np.zeros((T, N), dtype=bool)
    if N == 0:
        return trajs_2d, valids, visibs

    ones = np.ones((N, 1), dtype=np.float32)
    points_h = np.concatenate([points_world, ones], axis=-1)
    for t in range(T):
        if depths[t] is None:
            continue
        depth_t = np.asarray(depths[t], dtype=np.float32)
        H, W = depth_t.shape
        K_t = intrinsics[t]
        E_t = extrinsics[t]
        points_cam = (E_t @ points_h.T).T[:, :3]
        z = points_cam[:, 2]
        safe_z = np.where(z > 0, z, 1.0)
        uv = np.stack(
            [
                points_cam[:, 0] / safe_z * K_t[0, 0] + K_t[0, 2],
                points_cam[:, 1] / safe_z * K_t[1, 1] + K_t[1, 2],
            ],
            axis=-1,
        ).astype(np.float32)
        trajs_2d[t] = uv

        uv_finite = np.isfinite(uv).all(axis=-1)
        in_front = (z > 0) & np.isfinite(z) & (z < depth_max)
        in_bounds = (
            uv_finite
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        sample_mask = in_front & in_bounds
        visible = np.zeros((N,), dtype=bool)
        occluded = np.zeros((N,), dtype=bool)
        if sample_mask.any():
            ids = np.flatnonzero(sample_mask)
            px = np.clip(np.round(uv[ids, 0]).astype(np.int32), 0, W - 1)
            py = np.clip(np.round(uv[ids, 1]).astype(np.int32), 0, H - 1)
            radius = max(int(depth_support_radius), 0)
            min_support = max(int(occlusion_min_support), 1)
            z_sel = z[ids].astype(np.float32)
            visible_margin = np.maximum(
                float(visible_abs_thresh),
                float(depth_consistency_thresh) * np.maximum(z_sel, 0.0),
            )
            occlusion_margin = np.maximum(
                float(occlusion_abs_thresh),
                float(occlusion_rel_thresh) * np.maximum(z_sel, 0.0),
            )
            visible_support = np.zeros((len(ids),), dtype=np.int16)
            nearer_support = np.zeros((len(ids),), dtype=np.int16)
            for dy in range(-radius, radius + 1):
                yy = py + dy
                y_ok = (yy >= 0) & (yy < H)
                if not y_ok.any():
                    continue
                yy = np.clip(yy, 0, H - 1)
                for dx in range(-radius, radius + 1):
                    xx = px + dx
                    ok = y_ok & (xx >= 0) & (xx < W)
                    if not ok.any():
                        continue
                    xx = np.clip(xx, 0, W - 1)
                    sampled = depth_t[yy, xx].astype(np.float32)
                    depth_ok = ok & (sampled > 0) & np.isfinite(sampled) & (sampled < depth_max)
                    visible_support += (depth_ok & (np.abs(sampled - z_sel) <= visible_margin)).astype(np.int16)
                    nearer_support += (depth_ok & (sampled + occlusion_margin < z_sel)).astype(np.int16)
            visible[ids] = visible_support > 0
            occluded[ids] = (~visible[ids]) & (nearer_support >= min_support)

        valids[t] = visible | occluded
        visibs[t] = visible

    return trajs_2d, valids, visibs


class TartanAirAdapter(BaseAdapter):
    dataset_name: str = "tartanair"

    def __init__(
        self,
        root: str,
        split: str = "train",
        camera: str = "left",
        precompute_root: Optional[str] = None,
        load_precomputed: bool = True,
        verbose: bool = True,
        strict: bool = True,
        cache_dir: Optional[str] = None,
        max_static_track_points: Optional[int] = 8192,
        visibility_depth_support_radius: int = 2,
        visibility_visible_abs_thresh: float = 0.25,
        visibility_occlusion_abs_thresh: float = 1.0,
        visibility_occlusion_rel_thresh: float = 0.08,
        visibility_occlusion_min_support: int = 9,
        load_normals: bool = True,
        io_workers: Optional[int] = None,
        io_read_retries: Optional[int] = None,
        io_retry_delay_s: Optional[float] = None,
        pose_cache_items: Optional[int] = None,
        index_workers: int = 1,
    ):
        """
        Args:
            root:             TartanAir root directory (contains scene/Easy|Hard/P00X/).
            split:            Dataset split tag (informational only).
            camera:           Which camera to use: 'left' or 'right'.
            precompute_root:  If set, load precomputed normals/tracks from
                              <precompute_root>/<seq_name>/precomputed.npz.
            verbose:          Print loading info.
            cache_dir:        If set, cache sequence index to avoid slow COS iterdir().
            max_static_track_points:
                              Cap compact static points projected per loaded clip.
                              Points whose reference frame is inside the clip are
                              preferred so source coverage stays dense.
            visibility_*:     Runtime visibility policy for compact static points.
                              Visible is deliberately permissive; invisible labels
                              are emitted only for high-confidence occlusion.
            load_normals:     Whether to load normal maps from precomputed cache.
                              Disable when normal supervision is not used.
            io_workers:       Per-sample frame reader threads.  Use >1 for NFS
                              or staged local cache with per-file latency.
            io_read_retries:  Retry count for transient image/depth/precomputed
                              read failures on NFS-like mounts.
            pose_cache_items: Number of parsed pose files cached per process.
            index_workers:    Accepted for config parity with other adapters.
                              TartanAir indexing is directory/pose-file bound
                              and uses the disk cache for repeated runs.
        """
        self.root = Path(root)
        self.split = split
        self.camera = camera
        self.strict = bool(strict)
        self.load_precomputed = bool(load_precomputed)
        self.precompute_root = (
            Path(precompute_root) if precompute_root and self.load_precomputed
            else (self.root if self.load_precomputed else None)
        )
        self.verbose = verbose
        self.max_static_track_points = (
            int(max_static_track_points)
            if max_static_track_points is not None and int(max_static_track_points) > 0
            else None
        )
        self.visibility_depth_support_radius = max(int(visibility_depth_support_radius), 0)
        self.visibility_visible_abs_thresh = float(visibility_visible_abs_thresh)
        self.visibility_occlusion_abs_thresh = float(visibility_occlusion_abs_thresh)
        self.visibility_occlusion_rel_thresh = float(visibility_occlusion_rel_thresh)
        self.visibility_occlusion_min_support = max(int(visibility_occlusion_min_support), 1)
        self.load_normals = _resolve_bool(load_normals, True)
        if io_workers is None:
            io_workers = os.getenv("D4RT_TARTANAIR_IO_WORKERS", "4")
        if io_read_retries is None:
            io_read_retries = os.getenv("D4RT_TARTANAIR_IO_READ_RETRIES", "3")
        if io_retry_delay_s is None:
            raw_delay = os.getenv("D4RT_TARTANAIR_IO_RETRY_DELAY_S", "0.05")
            try:
                io_retry_delay_s = float(raw_delay)
            except ValueError:
                io_retry_delay_s = 0.05
        if pose_cache_items is None:
            pose_cache_items = os.getenv("D4RT_TARTANAIR_POSE_CACHE_ITEMS", "128")
        self.io_workers = _resolve_positive_int(io_workers, 1)
        self.io_read_retries = _resolve_nonnegative_int(io_read_retries, 2)
        self.io_retry_delay_s = max(0.0, float(io_retry_delay_s))
        self.pose_cache_items = _resolve_nonnegative_int(pose_cache_items, 128)
        self.index_workers = _resolve_positive_int(index_workers, 1)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._pose_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._pose_cache_lock = threading.RLock()
        self._precomputed_info_cache: dict[str, dict[str, Any]] = {}

        # TartanAir standard intrinsics (640x480)
        self.K = np.array([
            [320.0, 0.0,   320.0],
            [0.0,   320.0, 240.0],
            [0.0,   0.0,   1.0]
        ], dtype=np.float32)

        _cache_path = self._index_cache_path(self.cache_dir)
        _cache_hit = bool(_cache_path is not None and _cache_path.exists())
        if not _cache_hit and not self.root.exists():
            raise FileNotFoundError(f"TartanAir root not found: {self.root}")

        if self.cache_dir is not None:
            from datasets.index_cache import load_or_build

            assert _cache_path is not None
            index: list[tuple[str, int]] = load_or_build(self._build_index, _cache_path)
        else:
            index = self._build_index()

        self.sequences: list[str] = [name for name, _ in index]
        self.seq_to_dir: dict[str, Path] = {name: self.root / name for name in self.sequences}
        self._num_frames_map: dict[str, int] = {name: nf for name, nf in index}

        if len(self.sequences) == 0:
            raise RuntimeError(f"No valid TartanAir sequences found under {self.root}")

        if self.verbose:
            print(f"[TartanAirAdapter] split={self.split}, camera={self.camera}, "
                  f"num_sequences={len(self.sequences)}, "
                  f"precompute={'yes' if self.precompute_root else 'no'}, "
                  f"load_normals={self.load_normals}, "
                  f"io_workers={self.io_workers}, "
                  f"io_read_retries={self.io_read_retries}")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_pose_cache"] = OrderedDict()
        state["_pose_cache_lock"] = None
        state["_precomputed_info_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pose_cache = OrderedDict()
        self._pose_cache_lock = threading.RLock()
        self._precomputed_info_cache = {}

    def __len__(self) -> int:
        return len(self.sequences)

    def list_sequences(self) -> list[str]:
        return self.sequences.copy()

    def get_sequence_name(self, index: int) -> str:
        return self.sequences[index]

    def _index_cache_path(self, cache_dir: Optional[Path]) -> Optional[Path]:
        if cache_dir is None:
            return None
        cache_key = {
            "dataset": self.dataset_name,
            "split": self.split,
            "camera": self.camera,
            "root": str(self.root),
            "precompute_root": str(self.precompute_root) if self.precompute_root else "",
            "cache_schema": 2,
        }
        suffix = hashlib.sha1(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:12]
        return cache_dir / f"{self.dataset_name}_{self.split}_{self.camera}_{suffix}.pkl"

    def _path_exists_retry(self, path: Path) -> bool:
        attempts = max(1, self.io_read_retries + 1)
        for attempt in range(attempts):
            try:
                if path.exists():
                    return True
            except OSError:
                pass
            if attempt + 1 < attempts:
                _sleep_before_retry(self.io_retry_delay_s, attempt)
        return False

    def _iterdir_retry(self, path: Path) -> list[Path]:
        attempts = max(1, self.io_read_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return sorted(path.iterdir())
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    _sleep_before_retry(self.io_retry_delay_s, attempt)
        raise RuntimeError(f"Failed to list TartanAir directory: {path}") from last_error

    def _build_index(self) -> list[tuple[str, int]]:
        results: list[tuple[str, int]] = []
        for scene_dir in self._iterdir_retry(self.root):
            if not scene_dir.is_dir():
                continue
            try:
                scene_entries = self._iterdir_retry(scene_dir)
            except Exception as exc:
                if self.strict:
                    raise
                if self.verbose:
                    print(f"[TartanAirAdapter] Warning: skip scene {scene_dir}: {exc}")
                continue
            # Two-level layout: scene/difficulty/P00X  (TartanAirV1)
            difficulty_dirs = sorted(
                d for d in scene_entries
                if d.is_dir() and d.name not in (f"image_{self.camera}", f"depth_{self.camera}", "flow")
            )
            candidates = []
            for d in difficulty_dirs:
                try:
                    sequence_dirs = self._iterdir_retry(d)
                except Exception as exc:
                    if self.strict:
                        raise
                    if self.verbose:
                        print(f"[TartanAirAdapter] Warning: skip difficulty {d}: {exc}")
                    continue
                for p in sequence_dirs:
                    if p.is_dir() and self._path_exists_retry(p / f"image_{self.camera}"):
                        candidates.append((f"{scene_dir.name}/{d.name}/{p.name}", p))
            if not candidates and self._path_exists_retry(scene_dir / f"image_{self.camera}"):
                candidates = [(scene_dir.name, scene_dir)]
            for seq_name, seq_dir in candidates:
                pose_file = seq_dir / f"pose_{self.camera}.txt"
                try:
                    with open(pose_file, 'r') as f:
                        nf = sum(1 for line in f if line.strip())
                except Exception:
                    nf = 0
                if nf > 0:
                    results.append((seq_name, nf))
        return results

    def _precomputed_paths(self, sequence_name: str) -> tuple[Path, Path]:
        base = (
            self.precompute_root / sequence_name
            if self.precompute_root is not None
            else self.seq_to_dir[sequence_name]
        )
        npz_path = base / "precomputed.npz"
        return npz_path, npz_path.with_suffix(".h5")

    def _get_precomputed_info(self, sequence_name: str) -> dict[str, Any]:
        cached = self._precomputed_info_cache.get(sequence_name)
        if cached is not None:
            return cached

        info = {
            "has_precomputed": False,
            "backend": None,
            "has_normals": False,
            "has_tracks": False,
            "has_visibility": False,
            "has_trajs_3d_world": False,
        }
        if not self.load_precomputed or self.precompute_root is None:
            self._precomputed_info_cache[sequence_name] = info
            return info

        npz_path, h5_path = self._precomputed_paths(sequence_name)
        try:
            if self._path_exists_retry(h5_path):
                import h5py

                with h5py.File(h5_path, "r") as f:
                    keys = set(f.keys())
                info["has_precomputed"] = True
                info["backend"] = "h5"
            elif self._path_exists_retry(npz_path):
                with np.load(npz_path, allow_pickle=True) as raw:
                    keys = set(raw.files)
                info["has_precomputed"] = True
                info["backend"] = "npz"
            else:
                keys = set()

            if keys:
                has_static_points = "static_points_world" in keys
                info["has_normals"] = "normals" in keys and self.load_normals
                info["has_tracks"] = (
                    "trajs_2d" in keys
                    or "trajs_3d_world" in keys
                    or has_static_points
                )
                info["has_visibility"] = (
                    "valids" in keys and "visibs" in keys
                ) or has_static_points
                info["has_trajs_3d_world"] = "trajs_3d_world" in keys or has_static_points
        except Exception as exc:
            if self.verbose:
                print(f"[TartanAirAdapter] Warning: failed to inspect precomputed cache "
                      f"for {sequence_name}: {exc}")

        self._precomputed_info_cache[sequence_name] = info
        return info

    def _get_valid_frame_list(self, seq_dir: Path, seq_name: str) -> Optional[list[int]]:
        """Return sorted list of usable frame indices for this sequence.

        When a precomputed cache exists, returns only the contiguous run of frames
        starting from ref_frame that have at least MIN_VALID_PTS valid+visible points.
        This ensures the sampler always picks frames with dense track coverage.
        Falls back to all image frames when no cache is found.
        """
        MIN_VALID_PTS = 5
        cache_dir = (
            self.precompute_root / seq_name
            if self.precompute_root is not None
            else seq_dir
        )
        h5_path = cache_dir / "precomputed.h5"
        npz_path = cache_dir / "precomputed.npz"
        try:
            per_frame = None
            per_frame_is_ref_counts = False
            if h5_path.exists():
                import h5py
                with h5py.File(h5_path, 'r') as f:
                    ref_frame = int(f['ref_frame'][()])
                    if "valid_visible_counts" in f:
                        per_frame = np.asarray(f["valid_visible_counts"][()], dtype=np.int64)
                    elif "static_point_ref_counts" in f:
                        per_frame = np.asarray(f["static_point_ref_counts"][()], dtype=np.int64)
                        per_frame_is_ref_counts = True
                    else:
                        valids = f['valids'][()]
                        visibs = f['visibs'][()]
                        per_frame = (valids & visibs).sum(axis=1)
            elif npz_path.exists():
                raw = np.load(npz_path, allow_pickle=True)
                ref_frame = int(raw['ref_frame'])
                if "valid_visible_counts" in raw.files:
                    per_frame = np.asarray(raw["valid_visible_counts"], dtype=np.int64)
                elif "static_point_ref_counts" in raw.files:
                    per_frame = np.asarray(raw["static_point_ref_counts"], dtype=np.int64)
                    per_frame_is_ref_counts = True
                else:
                    valids = raw['valids']
                    visibs = raw['visibs']
                    per_frame = (valids & visibs).sum(axis=1)
            else:
                return None

            if per_frame_is_ref_counts:
                good = np.where(per_frame >= MIN_VALID_PTS)[0].tolist()
                return good if good else None

            # Find the contiguous run from ref_frame where valid pts >= threshold
            n = len(per_frame)
            end = ref_frame
            while end < n and per_frame[end] >= MIN_VALID_PTS:
                end += 1
            good = list(range(ref_frame, end))
            if len(good) == 0:
                # fallback: any frame with valid pts
                good = np.where(per_frame >= MIN_VALID_PTS)[0].tolist()
            return good if good else None
        except Exception as e:
            if self.verbose:
                print(f"[TartanAirAdapter] Warning: could not read valid frames for {seq_name}: {e}")
        return None

    def _load_precomputed(self, sequence_name: str, frame_indices: list[int]) -> Optional[dict]:
        """Load precomputed normals/tracks for frame_indices. Prefers .h5 over .npz."""
        if not self.load_precomputed or self.precompute_root is None:
            return None
        from datasets.adapters.base import load_precomputed_fast
        cache_path, h5_path = self._precomputed_paths(sequence_name)
        if not (self._path_exists_retry(h5_path) or self._path_exists_retry(cache_path)):
            return None
        attempts = max(1, self.io_read_retries + 1)
        last_error: Exception | None = None
        skip_keys = set() if self.load_normals else {"normals"}
        for attempt in range(attempts):
            try:
                return load_precomputed_fast(cache_path, frame_indices, skip_keys=skip_keys)
            except Exception as e:
                last_error = e
                if attempt + 1 < attempts:
                    _sleep_before_retry(self.io_retry_delay_s, attempt)
                    continue
        if self.verbose:
            print(f"[TartanAirAdapter] Warning: failed to load precomputed cache "
                  f"for {sequence_name}: {last_error}")
        return None

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        precomputed = self._get_precomputed_info(sequence_name)
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": sequence_name,
            "num_frames": self.get_num_frames(sequence_name),
            "has_depth": True,
            "has_flow": False,
            "has_normals": precomputed["has_normals"],
            "has_tracks": precomputed["has_tracks"],
            "has_visibility": precomputed["has_visibility"],
            "has_trajs_3d_world": precomputed["has_trajs_3d_world"],
            "precomputed_backend": precomputed["backend"],
        }

    def get_num_frames(self, sequence_name: str) -> int:
        return self._num_frames_map.get(sequence_name, 0)

    def _parse_pose_file(self, pose_file: Path) -> np.ndarray:
        attempts = max(1, self.io_read_retries + 1)
        last_error: Exception | None = None
        lines: list[str] | None = None
        for attempt in range(attempts):
            try:
                with open(pose_file, "r") as f:
                    lines = f.readlines()
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    _sleep_before_retry(self.io_retry_delay_s, attempt)
        if lines is None:
            raise RuntimeError(f"Failed to read TartanAir pose file: {pose_file}") from last_error

        mats: list[np.ndarray] = []
        for line_no, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 7:
                if self.strict:
                    raise ValueError(f"Invalid TartanAir pose line {line_no} in {pose_file}: {line!r}")
                continue
            tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            R = quat_to_rotation_matrix([qx, qy, qz, qw])
            t = np.array([tx, ty, tz], dtype=np.float32)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R.T
            w2c[:3, 3] = -R.T @ t
            mats.append(w2c)
        if not mats:
            raise RuntimeError(f"No valid TartanAir poses in {pose_file}")
        return np.stack(mats, axis=0).astype(np.float32)

    def _load_pose_matrices(self, sequence_name: str, pose_file: Path) -> np.ndarray:
        if self.pose_cache_items <= 0:
            return self._parse_pose_file(pose_file)

        key = f"{sequence_name}|{pose_file}"
        with self._pose_cache_lock:
            cached = self._pose_cache.get(key)
            if cached is not None:
                self._pose_cache.move_to_end(key)
                return cached

        poses = self._parse_pose_file(pose_file)
        with self._pose_cache_lock:
            cached = self._pose_cache.get(key)
            if cached is not None:
                self._pose_cache.move_to_end(key)
                return cached
            self._pose_cache[key] = poses
            while len(self._pose_cache) > self.pose_cache_items:
                self._pose_cache.popitem(last=False)
            return poses

    def _select_static_points_for_clip(
        self,
        sequence_name: str,
        frame_indices: list[int],
        num_points: int,
        ref_frames: Optional[np.ndarray],
    ) -> np.ndarray:
        cap = self.max_static_track_points
        if cap is None or num_points <= cap:
            return np.arange(num_points, dtype=np.int64)

        import hashlib

        key = f"{sequence_name}:{frame_indices[0] if frame_indices else -1}:{frame_indices[-1] if frame_indices else -1}:{num_points}"
        seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        if ref_frames is not None and len(ref_frames) == num_points:
            requested = np.asarray(frame_indices, dtype=np.int32)
            local = np.flatnonzero(np.isin(ref_frames.astype(np.int32, copy=False), requested))
            if local.size >= cap:
                return np.sort(rng.choice(local, size=cap, replace=False).astype(np.int64))
            if local.size > 0:
                remaining = cap - local.size
                other = np.flatnonzero(~np.isin(ref_frames.astype(np.int32, copy=False), requested))
                if other.size > remaining:
                    other = rng.choice(other, size=remaining, replace=False)
                return np.sort(np.concatenate([local, other.astype(np.int64, copy=False)]))

        return np.sort(rng.choice(num_points, size=cap, replace=False).astype(np.int64))

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        seq_dir = self.seq_to_dir[sequence_name]

        img_dir   = seq_dir / f"image_{self.camera}"
        depth_dir = seq_dir / f"depth_{self.camera}"
        pose_file = seq_dir / f"pose_{self.camera}.txt"

        actual_indices = frame_indices

        pose_mats_ned = self._load_pose_matrices(sequence_name, pose_file)
        if actual_indices:
            lo, hi = min(actual_indices), max(actual_indices)
            if lo < 0 or hi >= pose_mats_ned.shape[0]:
                raise IndexError(
                    f"[{sequence_name}] frame_indices out of range: "
                    f"min={lo}, max={hi}, num_poses={pose_mats_ned.shape[0]}"
                )

        def _load_frame(idx: int) -> tuple[int, np.ndarray, Optional[np.ndarray], str]:
            # Direct path construction — avoids slow glob on COS
            img_path = img_dir / f"{idx:06d}_{self.camera}.png"
            depth_path = depth_dir / f"{idx:06d}_{self.camera}_depth.npy"
            image = _read_tartanair_image(
                img_path,
                retries=self.io_read_retries,
                retry_delay_s=self.io_retry_delay_s,
            )
            try:
                depth = _read_tartanair_depth(
                    depth_path,
                    retries=self.io_read_retries,
                    retry_delay_s=self.io_retry_delay_s,
                    depth_max=200.0,
                )
            except Exception as exc:
                if self.strict:
                    raise
                if self.verbose:
                    print(f"[TartanAirAdapter] Warning: failed to load depth {depth_path}: {exc}")
                depth = None
            return idx, image, depth, str(img_path)

        if self.io_workers > 1 and len(actual_indices) > 1:
            with ThreadPoolExecutor(max_workers=min(self.io_workers, len(actual_indices))) as ex:
                frame_rows = list(ex.map(_load_frame, actual_indices))
        else:
            frame_rows = [_load_frame(idx) for idx in actual_indices]

        images = [row[1] for row in frame_rows]
        depths = [row[2] for row in frame_rows]
        frame_paths = [row[3] for row in frame_rows]
        flows = [None for _ in actual_indices]
        extrinsics_arr = pose_mats_ned[np.asarray(actual_indices, dtype=np.int64)].copy()
        intrinsics_arr = np.broadcast_to(
            self.K[None],
            (len(actual_indices), 3, 3),
        ).copy()

        R_ned2enu = np.array([[0,1,0],[1,0,0],[0,0,-1]], dtype=np.float32)
        R_ned2cv  = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=np.float32)
        R_edn2enu = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float32)

        # ---- precomputed normals / tracks ----
        normals_out    = None
        trajs_2d_out   = None
        trajs_3d_out   = None
        valids_out     = None
        visibs_out     = None
        has_precomputed = False
        trajs_3d_source = None

        cache = self._load_precomputed(sequence_name, actual_indices)
        if cache is not None:
            try:
                if self.load_normals and "normals" in cache:
                    normals_out  = [n.astype(np.float32) for n in cache["normals"]]
                if "trajs_2d" in cache:
                    trajs_2d_out = cache["trajs_2d"]
                if "trajs_3d_world" in cache:
                    trajs_3d_cache = cache["trajs_3d_world"].astype(np.float32)
                    sh = trajs_3d_cache.shape
                    pts = trajs_3d_cache.reshape(-1, 3)
                    conv = int(np.asarray(cache.get("trajs_3d_world_convention", 0)).item())
                    if conv == 1:
                        trajs_3d_out = pts.reshape(sh).astype(np.float32)
                        trajs_3d_source = "precomputed_trajs_3d_world_adapter_world"
                    else:
                        trajs_3d_out = (R_edn2enu @ pts.T).T.reshape(sh).astype(np.float32)
                        trajs_3d_source = "precomputed_trajs_3d_world_edn_to_enu"
                if "valids" in cache:
                    valids_out   = cache["valids"]
                if "visibs" in cache:
                    visibs_out   = cache["visibs"]
                has_precomputed = True
            except Exception as e:
                if self.verbose:
                    print(f"[TartanAirAdapter] Warning: precomputed indexing failed "
                          f"for {sequence_name}: {e}")

        # Fallback for older caches that do not contain trajs_3d_world:
        # recompute from trajs_2d + depth (OpenCV cam coords -> NED world).
        if trajs_3d_out is None and trajs_2d_out is not None and len(depths) > 0:
            T_frames = len(actual_indices)
            N = trajs_2d_out.shape[1]
            trajs_3d_ned = np.zeros((T_frames, N, 3), dtype=np.float32)
            for ti in range(T_frames):
                uv = trajs_2d_out[ti]  # [N,2] pixel coords in original resolution
                K_t = intrinsics_arr[ti]
                E_t = extrinsics_arr[ti]  # w2c NED
                depth_t = depths[ti] if ti < len(depths) else None
                if depth_t is None:
                    continue
                H, W = depth_t.shape
                xi = np.clip(np.round(uv[:,0]).astype(np.int32), 0, W-1)
                yi = np.clip(np.round(uv[:,1]).astype(np.int32), 0, H-1)
                d = depth_t[yi, xi]
                # OpenCV unproject: Z=depth
                fx, fy = K_t[0,0], K_t[1,1]
                cx, cy = K_t[0,2], K_t[1,2]
                pts_cam = np.stack([(uv[:,0]-cx)/fx*d, (uv[:,1]-cy)/fy*d, d], axis=-1)
                # OpenCV cam -> NED cam -> NED world
                R_ned2cv = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=np.float32)
                pts_cam_ned = (R_ned2cv.T @ pts_cam.T).T  # OpenCV -> NED camera
                R = E_t[:3,:3]; t_w = E_t[:3,3]
                R_c2w = R.T; t_c2w = -R.T @ t_w
                pts_world = (R_c2w @ pts_cam_ned.T).T + t_c2w
                trajs_3d_ned[ti] = pts_world
            sh = trajs_3d_ned.shape
            pts = trajs_3d_ned.reshape(-1, 3).astype(np.float32)
            trajs_3d_out = (R_ned2enu @ pts.T).T.reshape(sh).astype(np.float32)
            trajs_3d_source = "depth_recomputed_ned_to_enu"

        # TartanAir uses NED world frame and NED camera frame.
        # Convert to: ENU world frame + OpenCV camera frame (x=right, y=down, z=fwd).
        #
        # R_ned2enu: NED world → ENU world  [[0,1,0],[1,0,0],[0,0,-1]]
        # R_ned2cv:  NED cam  → OpenCV cam  [[0,1,0],[0,0,1],[1,0,0]]
        #
        # Full extrinsic: p_cam_cv = R_ned2cv @ w2c_ned @ R_ned2enu^T @ p_enu
        T_left  = np.eye(4, dtype=np.float32); T_left[:3,:3]  = R_ned2cv
        T_right = np.eye(4, dtype=np.float32); T_right[:3,:3] = R_ned2enu.T  # = R_enu2ned

        extrinsics_arr = T_left @ extrinsics_arr @ T_right  # [T,4,4]

        if (
            trajs_2d_out is None
            and cache is not None
            and "static_points_world" in cache
            and len(depths) > 0
        ):
            pts = np.asarray(cache["static_points_world"], dtype=np.float32)
            conv = int(np.asarray(cache.get("static_points_world_convention", 1)).item())
            pts_enu = pts if conv == 1 else (R_edn2enu @ pts.T).T.astype(np.float32)
            ref_frames = None
            if "static_point_ref_frames" in cache:
                ref_frames = np.asarray(cache["static_point_ref_frames"], dtype=np.int32)
            keep = self._select_static_points_for_clip(
                sequence_name=sequence_name,
                frame_indices=actual_indices,
                num_points=pts_enu.shape[0],
                ref_frames=ref_frames,
            )
            pts_enu = pts_enu[keep]
            thresh = float(np.asarray(cache.get("track_depth_consistency_thresh", 0.05)).item())
            depth_max = float(np.asarray(cache.get("track_depth_max", 200.0)).item())
            trajs_2d_out, valids_out, visibs_out = _project_static_points_to_clip(
                points_world=pts_enu,
                depths=depths,
                intrinsics=intrinsics_arr,
                extrinsics=extrinsics_arr,
                depth_consistency_thresh=thresh,
                depth_max=depth_max,
                depth_support_radius=self.visibility_depth_support_radius,
                visible_abs_thresh=self.visibility_visible_abs_thresh,
                occlusion_abs_thresh=self.visibility_occlusion_abs_thresh,
                occlusion_rel_thresh=self.visibility_occlusion_rel_thresh,
                occlusion_min_support=self.visibility_occlusion_min_support,
            )
            trajs_3d_out = np.broadcast_to(
                pts_enu[None],
                (len(actual_indices), pts_enu.shape[0], 3),
            ).copy()
            trajs_3d_source = "compact_static_points_world_projected_on_load"
            has_precomputed = True

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths if depths else None,
            normals=normals_out,
            trajs_2d=trajs_2d_out,
            trajs_3d_world=trajs_3d_out,
            valids=valids_out,
            visibs=visibs_out,
            intrinsics=intrinsics_arr,
            extrinsics=extrinsics_arr,
            flows=flows if any(f is not None for f in flows) else None,
            metadata={
                "dataset_name": self.dataset_name,
                "split": self.split,
                "has_depth": len(depths) > 0,
                "has_flow": any(f is not None for f in flows),
                "has_normals": has_precomputed,
                "has_tracks": has_precomputed,
                "has_visibility": has_precomputed,
                "trajs_3d_source": trajs_3d_source,
                "world_convention": "enu",
                "pose_convention": "world_to_camera",
                "intrinsics_convention": "pinhole",
                "depth_unit": "meters",
            }
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        info = self.get_sequence_info(sequence_name)
        info["ok"] = info["num_frames"] > 0
        return info
