from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from .base import BaseAdapter, UnifiedClip, load_precomputed_fast


class RAFTFlowEstimator:
    def __init__(self, device: Optional[str] = None):
        import torch
        import torchvision.transforms.functional as F_vision
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self._torch = torch
        self._F_vision = F_vision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading RAFT flow model to {self.device}...")
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = raft_large(weights=weights, progress=False).to(self.device)
        self.model.eval()
        print("RAFT model loaded")

    def compute_flow(self, img1_np: np.ndarray, img2_np: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            t1 = self._F_vision.to_tensor(img1_np).unsqueeze(0).to(self.device)
            t2 = self._F_vision.to_tensor(img2_np).unsqueeze(0).to(self.device)
            t1, t2 = self.transforms(t1, t2)
            flows = self.model(t1, t2)
            return flows[-1][0].cpu().numpy().transpose(1, 2, 0)


_SPLIT_ALIASES = {
    "train": "training",
    "training": "training",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "testing",
    "testing": "testing",
}

_CAMERA_NAME_TO_ID = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
}

# Waymo camera axes: X=forward, Y=left, Z=up.
# OpenCV camera axes: X=right, Y=down, Z=forward.
_R_WAYMO_TO_STD = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _normalize_split(split: str) -> str:
    return _SPLIT_ALIASES.get(str(split), str(split))


def _normalize_camera_name(camera_name: int | str) -> int:
    if isinstance(camera_name, str):
        name = camera_name.strip()
        if name.isdigit():
            return int(name)
        return _CAMERA_NAME_TO_ID.get(name.upper(), 1)
    return int(camera_name)


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


def _sleep_before_retry(delay_s: float, attempt: int) -> None:
    if delay_s <= 0:
        return
    time.sleep(float(delay_s) * float(attempt + 1))


def _read_waymo_image(path: Path, retries: int, retry_delay_s: float) -> np.ndarray:
    attempts = max(1, int(retries) + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                _sleep_before_retry(retry_delay_s, attempt)
    raise RuntimeError(f"Failed to load Waymo image after {attempts} attempts: {path}") from last_error


def _make_v2c_std(c2v_waymo: np.ndarray) -> np.ndarray:
    """Return vehicle-to-camera transform in OpenCV convention."""
    r4 = np.eye(4, dtype=np.float64)
    r4[:3, :3] = _R_WAYMO_TO_STD
    return r4 @ np.linalg.inv(c2v_waymo.astype(np.float64))


def _build_depth_dense(
    image_shape: tuple[int, ...],
    npz_data: Any,
    v2c_std: np.ndarray,
    dilation_radius: int,
) -> np.ndarray:
    """Build a depth map from Waymo projected LiDAR points.

    The per-frame ``pixels`` field comes from the Waymo projection and is more
    faithful than reprojecting points with a pinhole model.  The depth value is
    recomputed after converting vehicle points into the OpenCV camera frame.
    """
    h, w = image_shape[:2]
    pts_veh = np.asarray(npz_data["pts3d"], dtype=np.float64)
    if len(pts_veh) == 0:
        return np.zeros((h, w), dtype=np.float32)

    pts_h = np.concatenate([pts_veh, np.ones((len(pts_veh), 1), dtype=np.float64)], axis=1)
    pts_cam = (v2c_std @ pts_h.T).T[:, :3]
    depth_z = pts_cam[:, 2].astype(np.float32)

    pixels = np.asarray(npz_data["pixels"], dtype=np.int32)
    u = pixels[:, 0]
    v = pixels[:, 1]
    valid = (depth_z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    depth = np.zeros((h, w), dtype=np.float32)
    if valid.any():
        # Fill far-to-near so nearer surfaces overwrite farther surfaces.
        order = np.argsort(depth_z[valid])[::-1]
        valid_ids = np.flatnonzero(valid)[order]
        depth[v[valid_ids], u[valid_ids]] = depth_z[valid_ids]

    if dilation_radius > 0:
        ksize = 2 * int(dilation_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        depth = cv2.dilate(depth, kernel)
    return depth


class WaymoAdapter(BaseAdapter):
    dataset_name: str = "waymo"

    def __init__(
        self,
        root: str,
        split: str = "training",
        camera_name: int | str = 1,
        extract_flow: bool = False,
        precompute_root: Optional[str] = None,
        verbose: bool = True,
        depth_dilation_radius: int = 6,
        cache_dir: Optional[str] = None,
        strict: bool = True,
        io_workers: Optional[int] = None,
        io_read_retries: Optional[int] = None,
        io_retry_delay_s: Optional[float] = None,
        calib_cache_items: Optional[int] = None,
        index_workers: int = 1,
    ) -> None:
        """Adapter for the extracted Waymo layout used by D4RT.

        Expected layout:
          root/training/<sequence>.tfrecord/calib.json
          root/training/<sequence>.tfrecord/00000_1.jpg
          root/training/<sequence>.tfrecord/00000_1.npz
          precompute_root/training/<sequence>.tfrecord/precomputed.h5

        ``cache_dir`` and ``io_*`` mirror the other D4RT adapters: index
        caching avoids repeated NFS/COS directory walks, while read retries and
        per-frame workers tolerate slow metadata visibility on network mounts.
        """
        self.root = Path(root)
        self.split = _normalize_split(split)
        self.camera_name = _normalize_camera_name(camera_name)
        self.extract_flow = bool(extract_flow)
        self.precompute_root = Path(precompute_root) if precompute_root else self.root
        self.verbose = bool(verbose)
        self.depth_dilation_radius = int(depth_dilation_radius)
        self.strict = bool(strict)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if io_workers is None:
            io_workers = os.getenv("D4RT_WAYMO_IO_WORKERS", "4")
        if io_read_retries is None:
            io_read_retries = os.getenv("D4RT_WAYMO_IO_READ_RETRIES", "3")
        if io_retry_delay_s is None:
            raw_delay = os.getenv("D4RT_WAYMO_IO_RETRY_DELAY_S", "0.05")
            try:
                io_retry_delay_s = float(raw_delay)
            except ValueError:
                io_retry_delay_s = 0.05
        if calib_cache_items is None:
            calib_cache_items = os.getenv("D4RT_WAYMO_CALIB_CACHE_ITEMS", "128")
        self.io_workers = _resolve_positive_int(io_workers, 1)
        self.io_read_retries = _resolve_nonnegative_int(io_read_retries, 2)
        self.io_retry_delay_s = max(0.0, float(io_retry_delay_s))
        self.calib_cache_items = _resolve_nonnegative_int(calib_cache_items, 128)
        self.index_workers = _resolve_positive_int(index_workers, 1)
        self._calib_cache: "OrderedDict[str, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._calib_cache_lock = threading.RLock()
        self._precomputed_info_cache: dict[str, dict[str, Any]] = {}

        if self.extract_flow:
            self.flow_estimator = RAFTFlowEstimator()

        cache_path = self._index_cache_path(self.cache_dir)
        cache_hit = bool(cache_path is not None and cache_path.exists())
        if not cache_hit and not self._path_exists_retry(self.root):
            raise FileNotFoundError(f"Waymo root not found: {self.root}")

        if self.cache_dir is not None:
            from datasets.index_cache import load_or_build

            assert cache_path is not None
            index: list[tuple[str, int]] = load_or_build(self._build_index, cache_path)
        else:
            index = self._build_index()

        self.sequences: list[str] = [name for name, _ in index]
        self._num_frames_map: dict[str, int] = {name: int(nf) for name, nf in index}
        if not self.sequences:
            raise RuntimeError(f"No valid Waymo sequences found under {self.root} for split={self.split}")

        if self.verbose:
            print(
                f"[WaymoAdapter] split={self.split}, num_sequences={len(self.sequences)}, "
                f"camera={self.camera_name}, precompute_root={self.precompute_root}, "
                f"extract_flow={self.extract_flow}, depth_dilation_radius={self.depth_dilation_radius}, "
                f"io_workers={self.io_workers}, io_read_retries={self.io_read_retries}"
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_calib_cache"] = OrderedDict()
        state["_calib_cache_lock"] = None
        state["_precomputed_info_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._calib_cache = OrderedDict()
        self._calib_cache_lock = threading.RLock()
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
            "root": str(self.root),
            "precompute_root": str(self.precompute_root),
            "split": self.split,
            "camera_name": self.camera_name,
            "cache_schema": 2,
        }
        suffix = hashlib.sha1(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:12]
        return cache_dir / f"{self.dataset_name}_{self.split}_{self.camera_name}_{suffix}.pkl"

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

    def _glob_retry(self, root: Path, pattern: str) -> list[Path]:
        attempts = max(1, self.io_read_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return sorted(root.glob(pattern))
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    _sleep_before_retry(self.io_retry_delay_s, attempt)
        raise RuntimeError(f"Failed to glob Waymo path: {root}/{pattern}") from last_error

    def _build_index(self) -> list[tuple[str, int]]:
        split_root = self.root / self.split
        search_roots = [split_root] if self._path_exists_retry(split_root) else [self.root]
        sequences: list[tuple[str, int]] = []
        for search_root in search_roots:
            for calib_path in self._glob_retry(search_root, "**/calib.json"):
                seq_dir = calib_path.parent
                frame_paths = self._glob_retry(seq_dir, f"*_{self.camera_name}.jpg")
                if not frame_paths:
                    if self.strict:
                        raise FileNotFoundError(f"No camera {self.camera_name} frames under {seq_dir}")
                    continue
                sequences.append((str(seq_dir.relative_to(self.root)), len(frame_paths)))
        return sequences

    def _cache_path(self, sequence_name: str, stem: str = "precomputed") -> Path:
        return self.precompute_root / sequence_name / f"{stem}.npz"

    def _load_precomputed(self, sequence_name: str, frame_indices: list[int]) -> Optional[dict]:
        cache_path = self._cache_path(sequence_name, "precomputed")
        if not self._path_exists_retry(cache_path) and not self._path_exists_retry(cache_path.with_suffix(".h5")):
            return None
        attempts = max(1, self.io_read_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return load_precomputed_fast(cache_path, frame_indices)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    _sleep_before_retry(self.io_retry_delay_s, attempt)
        if self.verbose:
            print(f"[WaymoAdapter] Warning: failed to load cache for {sequence_name}: {last_error}")
        return None

    def _load_precomputed_normals(self, sequence_name: str, frame_indices: list[int]) -> Optional[dict]:
        cache_path = self._cache_path(sequence_name, "precomputed_normals")
        if not self._path_exists_retry(cache_path) and not self._path_exists_retry(cache_path.with_suffix(".h5")):
            return None
        attempts = max(1, self.io_read_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return load_precomputed_fast(cache_path, frame_indices)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    _sleep_before_retry(self.io_retry_delay_s, attempt)
        if self.verbose:
            print(f"[WaymoAdapter] Warning: failed to load normals cache for {sequence_name}: {last_error}")
        return None

    def _count_frames(self, sequence_name: str) -> int:
        cached = self._num_frames_map.get(sequence_name)
        if cached is not None:
            return int(cached)
        seq_dir = self.root / sequence_name
        return len(self._glob_retry(seq_dir, f"*_{self.camera_name}.jpg"))

    def get_num_frames(self, sequence_name: str) -> int:
        return self._count_frames(sequence_name)

    def _get_precomputed_info(self, sequence_name: str) -> dict[str, Any]:
        cached = self._precomputed_info_cache.get(sequence_name)
        if cached is not None:
            return cached

        track_path = self._cache_path(sequence_name, "precomputed")
        track_h5_path = track_path.with_suffix(".h5")
        normals_path = self._cache_path(sequence_name, "precomputed_normals")
        normals_h5_path = normals_path.with_suffix(".h5")
        has_tracks = self._path_exists_retry(track_path) or self._path_exists_retry(track_h5_path)
        has_normals = self._path_exists_retry(normals_path) or self._path_exists_retry(normals_h5_path)
        backend = "h5" if self._path_exists_retry(track_h5_path) else ("npz" if self._path_exists_retry(track_path) else None)

        if not has_normals and has_tracks:
            try:
                if self._path_exists_retry(track_h5_path):
                    import h5py

                    with h5py.File(track_h5_path, "r") as data:
                        has_normals = "normals" in data
                elif self._path_exists_retry(track_path):
                    with np.load(track_path, mmap_mode="r") as data:
                        has_normals = "normals" in data.files
            except Exception:
                has_normals = False

        info = {
            "has_tracks": bool(has_tracks),
            "has_normals": bool(has_normals),
            "has_visibility": bool(has_tracks),
            "backend": backend,
        }
        self._precomputed_info_cache[sequence_name] = info
        return info

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        precomputed = self._get_precomputed_info(sequence_name)

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": sequence_name,
            "num_frames": self._count_frames(sequence_name),
            "has_depth": True,
            "has_flow": self.extract_flow,
            "has_normals": precomputed["has_normals"],
            "has_tracks": precomputed["has_tracks"],
            "has_visibility": precomputed["has_visibility"],
            "precomputed_backend": precomputed["backend"],
        }

    def _load_calibration(self, sequence_name: str) -> tuple[np.ndarray, np.ndarray]:
        if self.calib_cache_items > 0:
            with self._calib_cache_lock:
                cached = self._calib_cache.get(sequence_name)
                if cached is not None:
                    self._calib_cache.move_to_end(sequence_name)
                    return cached

        seq_dir = self.root / sequence_name
        calib_path = seq_dir / "calib.json"
        attempts = max(1, self.io_read_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with open(calib_path, encoding="utf-8") as handle:
                    calib_list = json.load(handle)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    _sleep_before_retry(self.io_retry_delay_s, attempt)
        else:
            raise RuntimeError(f"Failed to read Waymo calibration: {calib_path}") from last_error
        calib_map = {int(cam_name): info for cam_name, info in calib_list}
        if self.camera_name not in calib_map:
            raise KeyError(f"camera {self.camera_name} missing from {seq_dir / 'calib.json'}")
        cam_calib = calib_map[self.camera_name]
        intr = cam_calib["intrinsics"]
        intrinsic = np.array(
            [
                [intr[0], 0.0, intr[2]],
                [0.0, intr[1], intr[3]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        c2v = np.asarray(cam_calib["extrinsics"], dtype=np.float64).reshape(4, 4)
        result = (intrinsic, _make_v2c_std(c2v))
        if self.calib_cache_items > 0:
            with self._calib_cache_lock:
                self._calib_cache[sequence_name] = result
                while len(self._calib_cache) > self.calib_cache_items:
                    self._calib_cache.popitem(last=False)
        return result

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        seq_dir = self.root / sequence_name
        target_indices = [int(i) for i in frame_indices]
        if target_indices:
            lo, hi = min(target_indices), max(target_indices)
            n = self._count_frames(sequence_name)
            if lo < 0 or hi >= n:
                raise IndexError(
                    f"[{sequence_name}] frame_indices out of range: "
                    f"min={lo}, max={hi}, num_frames={n}"
                )
        intrinsic_mat, v2c_std = self._load_calibration(sequence_name)

        def _load_frame(frame_idx: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, str]:
            img_path = seq_dir / f"{frame_idx:05d}_{self.camera_name}.jpg"
            npz_path = seq_dir / f"{frame_idx:05d}_{self.camera_name}.npz"
            if not self._path_exists_retry(img_path) or not self._path_exists_retry(npz_path):
                raise FileNotFoundError(f"Missing Waymo frame files: {img_path} / {npz_path}")

            image = _read_waymo_image(
                img_path,
                retries=self.io_read_retries,
                retry_delay_s=self.io_retry_delay_s,
            )

            attempts = max(1, self.io_read_retries + 1)
            last_error: Exception | None = None
            for attempt in range(attempts):
                try:
                    with np.load(npz_path) as npz:
                        v2w = np.asarray(npz["pose"], dtype=np.float64).reshape(4, 4)
                        w2c = (v2c_std @ np.linalg.inv(v2w)).astype(np.float32)
                        depth = _build_depth_dense(
                            image.shape,
                            npz,
                            v2c_std,
                            dilation_radius=self.depth_dilation_radius,
                        )
                    return frame_idx, image, w2c, depth, str(img_path)
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < attempts:
                        _sleep_before_retry(self.io_retry_delay_s, attempt)
            raise RuntimeError(f"Failed to load Waymo frame npz after {attempts} attempts: {npz_path}") from last_error

        if self.io_workers > 1 and len(target_indices) > 1:
            with ThreadPoolExecutor(max_workers=min(self.io_workers, len(target_indices))) as ex:
                frame_rows = list(ex.map(_load_frame, target_indices))
        else:
            frame_rows = [_load_frame(idx) for idx in target_indices]

        images = [row[1] for row in frame_rows]
        extrinsics = [row[2] for row in frame_rows]
        depths = [row[3] for row in frame_rows]
        frame_paths = [row[4] for row in frame_rows]
        intrinsics = [intrinsic_mat for _ in frame_rows]

        flows = []
        if self.extract_flow and images:
            if self.verbose:
                print(f"[WaymoAdapter] Computing RAFT flow for {sequence_name}...")
            for idx in range(len(images)):
                if idx < len(images) - 1:
                    flows.append(self.flow_estimator.compute_flow(images[idx], images[idx + 1]))
                else:
                    flows.append(None)

        normals_out = None
        trajs_2d_out = None
        trajs_3d_out = None
        valids_out = None
        visibs_out = None
        has_tracks = False
        has_normals = False
        track_metadata: dict[str, Any] = {}

        cache = self._load_precomputed(sequence_name, target_indices)
        if cache is not None:
            try:
                trajs_2d_out = np.asarray(cache["trajs_2d"], dtype=np.float32)
                trajs_3d_out = np.asarray(cache["trajs_3d_world"], dtype=np.float32)
                valids_out = np.asarray(cache["valids"], dtype=bool)
                visibs_out = np.asarray(cache["visibs"], dtype=bool)
                has_tracks = True
                for key in (
                    "track_is_dynamic",
                    "track_object_ids",
                    "track_label_types",
                    "track_ref_frames",
                ):
                    if key in cache:
                        track_metadata[key] = cache[key]
                if "track_is_dynamic" in cache:
                    track_metadata["track_types"] = np.asarray(cache["track_is_dynamic"], dtype=np.int8)
                if "normals" in cache:
                    normals_out = [np.asarray(n, dtype=np.float32) for n in cache["normals"]]
                    has_normals = True
            except Exception as exc:
                if self.verbose:
                    print(f"[WaymoAdapter] Warning: precomputed indexing failed for {sequence_name}: {exc}")

        if normals_out is None:
            normals_cache = self._load_precomputed_normals(sequence_name, target_indices)
            if normals_cache is not None and "normals" in normals_cache:
                try:
                    normals_out = [np.asarray(n, dtype=np.float32) for n in normals_cache["normals"]]
                    has_normals = True
                except Exception as exc:
                    if self.verbose:
                        print(f"[WaymoAdapter] Warning: normals indexing failed for {sequence_name}: {exc}")

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
            intrinsics=np.stack(intrinsics, axis=0)
            if intrinsics
            else np.zeros((len(images), 3, 3), dtype=np.float32),
            extrinsics=np.stack(extrinsics, axis=0)
            if extrinsics
            else np.eye(4, dtype=np.float32)[None].repeat(len(images), axis=0),
            flows=flows if any(flow is not None for flow in flows) else None,
            metadata={
                "dataset_name": self.dataset_name,
                "split": self.split,
                "sequence_root": str(seq_dir),
                "frame_indices": target_indices,
                "num_frames_in_sequence": self._count_frames(sequence_name),
                "has_depth": bool(depths),
                "has_flow": any(flow is not None for flow in flows),
                "has_normals": has_normals,
                "has_tracks": has_tracks,
                "has_visibility": has_tracks,
                "visibility_validity_mode": "defined" if has_tracks else "visibility_supported",
                "keep_out_of_view_valids": bool(has_tracks),
                "pose_convention": "world_to_camera",
                "intrinsics_convention": "pinhole",
                "depth_unit": "meters",
                **track_metadata,
            },
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        info = self.get_sequence_info(sequence_name)
        info["ok"] = True
        return info
