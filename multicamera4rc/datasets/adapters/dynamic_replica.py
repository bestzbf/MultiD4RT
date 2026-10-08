from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import (
    BaseAdapter,
    UnifiedClip,
    canonical_anchor_frame_index,
    canonical_scene_anchor_metadata,
    canonical_sequence_time_metadata,
)

try:
    import cv2
except Exception:  # pragma: no cover - optional fast path
    cv2 = None

try:
    _TRAJECTORY_CACHE_SIZE = max(
        1, int(os.getenv("D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE", "64"))
    )
except ValueError:
    _TRAJECTORY_CACHE_SIZE = 64

try:
    _DEFAULT_RGB_CACHE_SIZE = max(
        0, int(os.getenv("D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE", "96"))
    )
except ValueError:
    _DEFAULT_RGB_CACHE_SIZE = 96

_TRAJECTORY_CACHE_LOCK = threading.Lock()
_TrajectoryCacheValue = tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
_TRAJECTORY_CACHE: "OrderedDict[tuple[int, int, int, int], _TrajectoryCacheValue]" = OrderedDict()


def _resolve_load_max_track_points() -> Optional[int]:
    raw = os.getenv(
        "D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS",
        os.getenv("D4RT_MAX_TRACK_POINTS", ""),
    ).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# Camera conversion helpers (identical to Co3Dv2, shared PyTorch3D format)
# ---------------------------------------------------------------------------

def _ndc_to_pinhole(
    focal_length: list[float],
    principal_point: list[float],
    image_size: list[int],
) -> np.ndarray:
    """Convert PyTorch3D NDC intrinsics to a standard 3×3 pinhole matrix.

    Dynamic_Replica uses the same ``ndc_isotropic`` format as Co3Dv2:

    - ``focal_length`` normalised by ``min(W, H) / 2``
    - ``principal_point`` has (0, 0) at image centre, x pointing left,
      y pointing up (opposite to OpenCV pixel convention)

    Args:
        focal_length:    [fx_ndc, fy_ndc]
        principal_point: [px_ndc, py_ndc]   ((0, 0) = centre)
        image_size:      [H, W]  (annotation stores height first)

    Returns:
        K: (3, 3) float32 pinhole intrinsics in pixel coordinates.
    """
    H, W = float(image_size[0]), float(image_size[1])
    half_s = min(H, W) / 2.0

    fx = focal_length[0] * half_s
    fy = focal_length[1] * half_s

    # NDC pp sign: pp_ndc_x = (W/2 - cx_px) / half_s  →  cx_px = W/2 - pp_ndc_x * half_s
    cx = W / 2.0 - principal_point[0] * half_s
    cy = H / 2.0 - principal_point[1] * half_s

    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _p3d_to_opencv_extrinsics(
    R_p3d: list[list[float]],
    T_p3d: list[float],
) -> np.ndarray:
    """Convert PyTorch3D camera pose to a 4×4 OpenCV-style w2c matrix.

    PyTorch3D uses row-vector convention (+x left, +y up, +z into scene).
    We convert to OpenCV column-vector convention (+x right, +y down, +z forward).

        R_cv = diag(-1, -1, 1) @ R_p3d.T
        T_cv = diag(-1, -1, 1) @ T_p3d

    Returns:
        E: (4, 4) float32 world-to-camera matrix (OpenCV convention, w2c).
    """
    R = np.array(R_p3d, dtype=np.float64)
    T = np.array(T_p3d, dtype=np.float64)

    D = np.diag([-1.0, -1.0, 1.0])
    R_cv = (D @ R.T).astype(np.float32)
    T_cv = (D @ T).astype(np.float32)

    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R_cv
    E[:3, 3] = T_cv
    return E


# ---------------------------------------------------------------------------
# Low-level file I/O
# ---------------------------------------------------------------------------

def _load_jgz(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return json.load(f)


def _load_rgb(path: Path) -> np.ndarray:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.asarray(Image.open(path).convert("RGB"))


def _coerce_rgb_array(value: Any) -> np.ndarray:
    """Convert a trajectory-embedded image tensor/array to HWC uint8 RGB."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"trajectory img must be rank-3, got shape={arr.shape}")
    if arr.shape[-1] not in {3, 4} and arr.shape[0] in {3, 4}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"trajectory img must have 3 RGB channels, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32, copy=False)
        if arr_f.size and float(np.nanmax(arr_f)) <= 1.0:
            arr_f = arr_f * 255.0
        arr = np.rint(np.clip(arr_f, 0.0, 255.0)).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _load_depth_raw(path: Path) -> np.ndarray:
    """Load the uint16 payload of a Dynamic Replica geometric-depth PNG."""
    if cv2 is not None:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            return np.asarray(raw, dtype=np.uint16)
    return np.array(Image.open(path), dtype=np.uint16)


def _decode_depth_raw(raw: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Decode IEEE-754 float16 bit patterns stored in the uint16 PNG.

    Dynamic Replica reuses the Co3D ``*.geometric.png`` encoding: each uint16
    pixel contains the bit representation of one float16 camera-Z value.  The
    historical affine arguments are retained for API/checkpoint compatibility
    but are intentionally ignored.
    """
    del scale, offset
    payload = np.ascontiguousarray(raw, dtype=np.uint16)
    depth = payload.view(np.float16).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0.0] = 0.0
    return depth


def _load_depth(path: Path, scale: float, offset: float) -> np.ndarray:
    """Load a 16-bit PNG depth map and convert to metric depth.

    Dynamic Replica encodes camera-Z as IEEE-754 float16 bit patterns inside
    uint16 PNG pixels, identical to Co3D ``*.geometric.png``.  ``scale`` and
    ``offset`` are ignored and retained only for compatibility.

    Returns:
        (H, W) float32 depth array in world units.  Zero indicates invalid depth.
    """
    return _decode_depth_raw(_load_depth_raw(path), scale, offset)


def _collect_raw_z_pairs(
    raw_depth: np.ndarray,
    traj_3d_world: np.ndarray,
    traj_2d: np.ndarray,
    visibs: np.ndarray,
    E: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Collect (raw_pixel, z_cam) pairs at visible trajectory positions."""
    H, W = raw_depth.shape
    vis = visibs.astype(bool)
    if not vis.any():
        return None

    pts_cam = (E[:3, :3] @ traj_3d_world[vis].T).T + E[:3, 3]
    z_cam = pts_cam[:, 2]

    u = traj_2d[vis, 0]
    v = traj_2d[vis, 1]
    xs = np.round(u).astype(int)
    ys = np.round(v).astype(int)
    in_bounds = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not in_bounds.any():
        return None

    xs = xs[in_bounds]; ys = ys[in_bounds]
    raw_at = raw_depth[ys, xs]
    z_exp = z_cam[in_bounds]
    valid = (raw_at > 0) & (raw_at < 32768) & (z_exp > 0)
    if valid.sum() < 10:
        return None

    return raw_at[valid].astype(np.float64), z_exp[valid].astype(np.float64)


def _calibrate_depth_linear(
    raw_depth: np.ndarray,
    traj_3d_world: np.ndarray,
    traj_2d: np.ndarray,
    visibs: np.ndarray,
    E: np.ndarray,
    fallback: tuple[float, float] = (0.00142, 14800.0),
) -> tuple[float, float] | None:
    """Fit depth = (raw - offset) * scale via least-squares regression.

    Dynamic_Replica depth is linearly encoded with a large per-sequence offset
    (~14800-15200) and a per-sequence scale (~0.0014-0.0018).

    Returns:
        (scale, offset) tuple, or fallback on failure.
    """
    result = _collect_raw_z_pairs(raw_depth, traj_3d_world, traj_2d, visibs, E)
    if result is None:
        return fallback

    raw_v, z_v = result
    # Least-squares: z = a * raw + b  =>  scale=a, offset=-b/a
    A = np.stack([raw_v, np.ones_like(raw_v)], axis=1)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, z_v, rcond=None)
    except np.linalg.LinAlgError:
        return fallback

    a, b = float(coeffs[0]), float(coeffs[1])
    if a <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return fallback

    offset = -b / a
    return (a, offset)


def _calibrate_depth_linear_multiframe(
    raw_depths: list[np.ndarray],
    trajs_3d_world: np.ndarray,
    trajs_2d: np.ndarray,
    visibs: np.ndarray,
    extrinsics: np.ndarray,
    max_frames: int = 5,
    fallback: tuple[float, float] = (0.00142, 14800.0),
) -> tuple[float, float]:
    """Jointly fit depth = (raw - offset) * scale across multiple frames.

    Pools all (raw, z_gt) pairs from up to max_frames and fits one regression.

    Returns:
        (scale, offset) tuple.
    """
    T = len(raw_depths)
    num_frames = min(max_frames, T)

    all_raw = []
    all_z = []
    for t in range(num_frames):
        result = _collect_raw_z_pairs(
            raw_depths[t], trajs_3d_world[t], trajs_2d[t], visibs[t], extrinsics[t]
        )
        if result is not None:
            all_raw.append(result[0])
            all_z.append(result[1])

    if len(all_raw) == 0:
        return fallback

    raw_v = np.concatenate(all_raw)
    z_v = np.concatenate(all_z)

    A = np.stack([raw_v, np.ones_like(raw_v)], axis=1)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, z_v, rcond=None)
    except np.linalg.LinAlgError:
        return fallback

    a, b = float(coeffs[0]), float(coeffs[1])
    if a <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return fallback

    offset = -b / a
    return (a, offset)


def _load_trajectory_pth(path: Path) -> dict[str, np.ndarray]:
    """Load a per-frame trajectory .pth file (PyTorch zip-archive format).

    Each file stores data for the N consistently-indexed mesh vertices:
        traj_3d_world : (N, 3) float32  – 3D world positions at this frame
        traj_2d       : (N, 3) float32  – 2D pixel positions (cols 0–1 are u,v)
        verts_inds_vis: (N,)   bool     – per-vertex visibility flag
        img           : (H, W, 3) uint8 – RGB frame duplicated in the .pth

    Returns:
        dict with keys "traj_3d_world", "traj_2d", "verts_inds_vis", "img".

    Raises:
        ImportError  if PyTorch is not installed.
        FileNotFoundError  if the .pth file does not exist.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load Dynamic_Replica trajectory files. "
            "Install it or set load_trajectories=False."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    return {
        "traj_3d_world": data["traj_3d_world"].numpy().astype(np.float32),  # (N, 3)
        "traj_2d": data["traj_2d"].numpy().astype(np.float32),              # (N, 3)
        "verts_inds_vis": data["verts_inds_vis"].numpy().astype(bool),      # (N,)
        "img": _coerce_rgb_array(data["img"]),
    }


def _trajectory_npz_path(path: Path) -> Path:
    return path.with_suffix(".npz")


def _load_trajectory_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory npz file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {
            "traj_3d_world": data["traj_3d_world"].astype(np.float32),
            "traj_2d": data["traj_2d"].astype(np.float32),
            "verts_inds_vis": data["verts_inds_vis"].astype(bool),
        }


def _trajectory_cache_key(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))


def _load_trajectory_pth_cached(
    path_str: str,
) -> _TrajectoryCacheValue:
    """Process-local cache for per-frame trajectory files.

    Direct DataLoader workers may revisit the same trajectory through different
    sampled windows. Use inode-level identity so repeated windows reuse a
    decoded trajectory without depending on a copied path.
    """
    path = Path(path_str)
    cache_key = _trajectory_cache_key(path)
    with _TRAJECTORY_CACHE_LOCK:
        cached = _TRAJECTORY_CACHE.get(cache_key)
        if cached is not None:
            _TRAJECTORY_CACHE.move_to_end(cache_key)
            return cached

    data = _load_trajectory_pth(path)
    value = (
        data["traj_3d_world"],
        data["traj_2d"],
        data["verts_inds_vis"],
        data["img"],
    )
    with _TRAJECTORY_CACHE_LOCK:
        existing = _TRAJECTORY_CACHE.get(cache_key)
        if existing is not None:
            _TRAJECTORY_CACHE.move_to_end(cache_key)
            return existing
        _TRAJECTORY_CACHE[cache_key] = value
        _TRAJECTORY_CACHE.move_to_end(cache_key)
        while len(_TRAJECTORY_CACHE) > _TRAJECTORY_CACHE_SIZE:
            _TRAJECTORY_CACHE.popitem(last=False)
    return value


def _load_trajectory_cached(
    path_str: str,
    *,
    prefer_npz: bool,
) -> _TrajectoryCacheValue:
    path = Path(path_str)
    if prefer_npz:
        npz_path = _trajectory_npz_path(path)
        try:
            cache_key = _trajectory_cache_key(npz_path)
            with _TRAJECTORY_CACHE_LOCK:
                cached = _TRAJECTORY_CACHE.get(cache_key)
                if cached is not None:
                    _TRAJECTORY_CACHE.move_to_end(cache_key)
                    return cached

            data = _load_trajectory_npz(npz_path)
            value: _TrajectoryCacheValue = (
                data["traj_3d_world"],
                data["traj_2d"],
                data["verts_inds_vis"],
                None,
            )
            with _TRAJECTORY_CACHE_LOCK:
                existing = _TRAJECTORY_CACHE.get(cache_key)
                if existing is not None:
                    _TRAJECTORY_CACHE.move_to_end(cache_key)
                    return existing
                _TRAJECTORY_CACHE[cache_key] = value
                _TRAJECTORY_CACHE.move_to_end(cache_key)
                while len(_TRAJECTORY_CACHE) > _TRAJECTORY_CACHE_SIZE:
                    _TRAJECTORY_CACHE.popitem(last=False)
            return value
        except FileNotFoundError:
            pass

    return _load_trajectory_pth_cached(path_str)


# ---------------------------------------------------------------------------
# Sequence index record
# ---------------------------------------------------------------------------

@dataclass
class _DRSequenceRecord:
    """Index entry for one Dynamic_Replica sequence (= one camera track)."""

    sequence_name: str        # full dir name, e.g. "009850-3_obj_source_left"
    base_seq_name: str        # base scene, e.g. "009850-3_obj"
    camera_name: str          # "left" or "right"
    split: str                # "train", "valid", or "test"
    sequence_dir: Path        # absolute path to the sequence directory
    frame_numbers: tuple[int, ...]  # sorted ascending list of annotation frame numbers
    image_size: tuple[int, int]     # (H, W) from the first frame annotation
    image_rel_paths: tuple[str, ...]
    depth_rel_paths: tuple[str, ...]
    traj_rel_paths: tuple[Optional[str], ...]
    intrinsics: np.ndarray          # (T, 3, 3) float32
    extrinsics: np.ndarray          # (T, 4, 4) float32
    has_trajectories: bool    # True only for "left" camera sequences

    @property
    def num_frames(self) -> int:
        return len(self.frame_numbers)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DynamicReplicaAdapter(BaseAdapter):
    """Dataset adapter for the Dynamic_Replica dataset.

    Dataset layout::

        <root>/
            <split>/                              # train / valid / test
                frame_annotations_<split>.jgz    # per-frame camera + path annotations
                <base_seq>_source_left/
                    images/
                        <base_seq>_source_left-XXXX.png       (RGBA, uint8)
                    depths/
                        <base_seq>_source_left_XXXX.geometric.png  (uint16)
                    trajectories/
                        XXXXXX.pth                             (PyTorch zip)
                    masks/
                    ...
                <base_seq>_source_right/
                    images/ depths/ ...           (no trajectories)

    Sequence IDs
    ------------
    Each adapter sequence corresponds to one camera track (one sub-directory).
    The unique ID is the directory name, e.g. ``"009850-3_obj_source_left"``.

    Camera convention
    -----------------
    Annotations use PyTorch3D NDC intrinsics and row-vector extrinsics.
    These are converted to OpenCV pixel-space intrinsics and world-to-camera
    (w2c) extrinsics (same conversion as the Co3Dv2 adapter).

    Depth encoding
    --------------
    ``depth = raw_uint16.view(float16).astype(float32)``.  Depth is camera-Z in
    PyTorch3D world units (consistent with camera extrinsics); non-finite and
    non-positive payloads are returned as invalid zero depth.

    Trajectory files
    ----------------
    Only left-camera sequences have trajectory data.  Each per-frame ``.pth``
    file stores the 3D world positions and 2D pixel positions of N=22,671
    consistently-indexed mesh vertices (``verts_inds_vis`` encodes visibility).

    Loading trajectories requires PyTorch.  Pass ``load_trajectories=False``
    to skip trajectory loading (all trajectory fields will be ``None``).

    Supervision availability
    ------------------------
    - ``has_depth``      : True
    - ``has_normals``    : False
    - ``has_tracks``     : True  (left camera only; False if load_trajectories=False
                                   or right camera)
    - ``has_visibility`` : True  (when tracks are available)
    """

    dataset_name: str = "dynamic_replica"
    _CACHE_SCHEMA_VERSION = 2

    def __init__(
        self,
        root: str,
        split: str = "train",
        load_trajectories: bool = True,
        min_frames: int = 2,
        strict: bool = False,
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        index_workers: int = 8,
        rgb_from_trajectory: bool = True,
        skip_depth_when_tracks: bool = False,
        prefer_trajectory_npz: Optional[bool] = None,
        io_workers: Optional[int] = None,
        pixel_coordinates: str = "legacy_native",
        cameras: str = "left",
    ) -> None:
        """
        Parameters
        ----------
        root :
            Root directory of the Dynamic_Replica dataset.
        split :
            Which split to load: ``"train"``, ``"valid"``, or ``"test"``.
        load_trajectories :
            If True (default), load per-frame trajectory ``.pth`` files for
            left-camera sequences (requires PyTorch).  Set to False to skip
            trajectory loading for faster I/O.
        min_frames :
            Sequences with fewer than this many frames are skipped.
        strict :
            If True, raise on any indexing error.
            If False, skip broken sequences with a warning.
        verbose :
            Print a summary after index construction.
        cache_dir :
            Optional local cache directory. When set, the adapter stores the
            fully parsed per-sequence metadata index and per-sequence depth
            calibration on local disk, which avoids repeated remote reads of
                              ``frame_annotations_<split>.jgz`` in the local root.
        rgb_from_trajectory :
            If True, left-camera frames with trajectory ``.pth`` files use the
            embedded ``img`` tensor instead of reading the duplicate image PNG.
        skip_depth_when_tracks :
            If True, left-camera clips with trajectory supervision do not load
            depth PNG files. The track path can train from trajectory 2D/3D
            labels directly; skipping depth removes 48 extra file reads per
            48-frame sample at the cost of disabling DynamicReplica depth-edge
            sampling and depth-map consistency filtering.
        prefer_trajectory_npz :
            Prefer same-name ``.npz`` trajectory files when available.  These
            contain the same trajectory arrays without the duplicated RGB image,
            so RGB is loaded from the image PNG and trajectory I/O is smaller
            than the legacy ``.pth`` path.
        io_workers :
            Number of per-sample threads used to read RGB/trajectory frames.
            Keep this small when many DataLoader workers run in parallel.
        pixel_coordinates :
            ``array_index_v1`` converts native boundary-coordinate UV and the
            intrinsic principal point by -0.5, matching integer array centers.
            Raw index/trajectory caches stay native; legacy consumers default
            to ``legacy_native`` and are not silently migrated.
        """
        if pixel_coordinates not in ("legacy_native", "array_index_v1"):
            raise ValueError("Replica pixel_coordinates must be legacy_native or array_index_v1")
        self.pixel_coordinates = pixel_coordinates
        if cameras not in ("left", "right", "both"):
            raise ValueError("Replica cameras must be left, right or both")
        self.root = Path(root)
        self.split = split
        self.split_root = self.root / split

        # Check cache first to skip slow root.exists() on remote storage
        _cache_hit = False
        if cache_dir is not None:
            # Build minimal cache path to check existence
            _cache_key = {
                "dataset": "dynamic_replica",
                "split": split,
                "root": str(root),
                "schema": 2,  # _CACHE_SCHEMA_VERSION
            }
            _cache_suffix = hashlib.sha1(
                json.dumps(_cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            _cache_path = Path(cache_dir) / f"dynamic_replica_{split}_{_cache_suffix}.pkl"
            if _cache_path.exists():
                _cache_hit = True

        if not _cache_hit and not self.root.exists():
            raise FileNotFoundError(f"Dynamic_Replica root not found: {self.root}")

        if not _cache_hit and not self.split_root.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_root}. "
                f"Valid splits: train, valid, test."
            )

        self.load_trajectories = load_trajectories
        self.min_frames = min_frames
        self.strict = strict
        self.verbose = verbose
        self.index_workers = index_workers
        self.rgb_from_trajectory = bool(rgb_from_trajectory)
        self.skip_depth_when_tracks = bool(skip_depth_when_tracks)
        if io_workers is None:
            raw_io_workers = os.getenv("D4RT_DYNAMIC_REPLICA_IO_WORKERS", "1")
            try:
                io_workers = int(raw_io_workers)
            except ValueError:
                io_workers = 1
        self.io_workers = max(1, int(io_workers))
        if prefer_trajectory_npz is None:
            prefer_trajectory_npz = os.getenv(
                "D4RT_DYNAMIC_REPLICA_PREFER_TRAJ_NPZ", "1"
            ).strip().lower() in {"1", "true", "yes", "on"}
        self.prefer_trajectory_npz = bool(prefer_trajectory_npz)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache_suffix = self._build_cache_suffix()
        self._depth_calibration_cache: dict[str, tuple[float, float]] = {}
        self._canonical_anchor_cache: dict[str, np.ndarray] = {}
        self.canonical_cache_items = max(
            1, int(os.getenv("D4RT_DYNAMIC_REPLICA_CANONICAL_CACHE_SIZE", "1"))
        )
        self._canonical_payload_cache: "OrderedDict[str, dict[str, np.ndarray]]" = OrderedDict()
        self._canonical_payload_cache_lock = threading.RLock()
        self.rgb_cache_items = _DEFAULT_RGB_CACHE_SIZE
        self.load_max_track_points = _resolve_load_max_track_points()
        self._rgb_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._rgb_cache_lock = threading.RLock()
        self._depth_calibration_cache_dir = (
            self.cache_dir / f"{self.dataset_name}_{self.split}_{self._cache_suffix}_depth_calibration"
            if self.cache_dir is not None
            else None
        )

        if self.cache_dir is not None:
            from multicamera4rc.datasets.index_cache import load_or_build
            _cache_path = (
                self.cache_dir
                / f"{self.dataset_name}_{self.split}_{self._cache_suffix}.pkl"
            )
            self._records: list[_DRSequenceRecord] = load_or_build(self._build_index, _cache_path)
        else:
            self._records: list[_DRSequenceRecord] = self._build_index()
        self._name_to_record: dict[str, _DRSequenceRecord] = {
            r.sequence_name: r for r in self._records
        }
        self._left_records: list[_DRSequenceRecord] = [
            r for r in self._records if cameras == "both" or r.camera_name == cameras
        ]

        if not self._records:
            raise RuntimeError(
                f"No valid Dynamic_Replica sequences found under {self.split_root}."
            )

        if self.verbose:
            n_left = sum(1 for r in self._records if r.camera_name == "left")
            n_right = sum(1 for r in self._records if r.camera_name == "right")
            print(
                f"[DynamicReplicaAdapter] split={split!r}, "
                f"sequences={len(self._records)} "
                f"(left={n_left}, right={n_right}), "
                f"load_trajectories={load_trajectories} "
                f"rgb_from_trajectory={self.rgb_from_trajectory} "
                f"skip_depth_when_tracks={self.skip_depth_when_tracks} "
                f"prefer_trajectory_npz={self.prefer_trajectory_npz} "
                f"io_workers={self.io_workers} "
                f"load_max_track_points={self.load_max_track_points}"
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Avoid pickling decoded frame arrays into multiprocessing workers.
        # Each worker rebuilds its own bounded cache on demand.
        state["_rgb_cache"] = OrderedDict()
        state["_rgb_cache_lock"] = None
        state["_canonical_payload_cache"] = OrderedDict()
        state["_canonical_payload_cache_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._rgb_cache = OrderedDict()
        self._rgb_cache_lock = threading.RLock()
        self._canonical_payload_cache = OrderedDict()
        self._canonical_payload_cache_lock = threading.RLock()
        if not hasattr(self, "rgb_cache_items"):
            self.rgb_cache_items = _DEFAULT_RGB_CACHE_SIZE
        if not hasattr(self, "load_max_track_points"):
            self.load_max_track_points = _resolve_load_max_track_points()

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._left_records)

    def list_sequences(self) -> list[str]:
        return [r.sequence_name for r in self._left_records]

    def get_sequence_name(self, index: int) -> str:
        return self._left_records[index].sequence_name

    def get_num_frames(self, sequence_name: str) -> int:
        """Fast path: read from cached record, no annotation reload."""
        return self._get_record(sequence_name).num_frames

    def get_canonical_anchor_frame_index(self, sequence_name: str) -> int:
        return canonical_anchor_frame_index(
            self._get_record(sequence_name).num_frames
        )

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self._get_record(sequence_name)
        H, W = r.image_size
        has_tracks = r.has_trajectories and self.load_trajectories
        has_depth = not (has_tracks and self.skip_depth_when_tracks)
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": r.sequence_name,
            "base_seq_name": r.base_seq_name,
            "camera_name": r.camera_name,
            "sequence_dir": str(r.sequence_dir),
            "num_frames": r.num_frames,
            "height": H,
            "width": W,
            "has_depth": has_depth,
            "has_normals": False,
            "has_tracks": has_tracks,
            "has_visibility": has_tracks,
            "has_trajs_3d_world": has_tracks,
            "extrinsics_convention": "w2c",
            "depth_unit": "dynamic_replica_world_units",
        }

    def _load_canonical_payload(
        self, record: _DRSequenceRecord
    ) -> dict[str, np.ndarray]:
        """Read and cache all persistent mesh-vertex trajectories in a sequence."""

        with self._canonical_payload_cache_lock:
            cached = self._canonical_payload_cache.get(record.sequence_name)
            if cached is not None:
                self._canonical_payload_cache.move_to_end(record.sequence_name)
                return cached

        rows: list[_TrajectoryCacheValue] = []
        for relative in record.traj_rel_paths:
            if relative is None:
                raise RuntimeError(
                    f"[{self.dataset_name}/{record.sequence_name}] full-sequence "
                    "canonical trajectory file is missing"
                )
            rows.append(
                _load_trajectory_cached(
                    str(self.split_root / relative),
                    prefer_npz=self.prefer_trajectory_npz,
                )
            )
        point_counts = {int(row[0].shape[0]) for row in rows}
        if len(point_counts) != 1:
            raise RuntimeError(
                f"[{self.dataset_name}/{record.sequence_name}] persistent track "
                "column count changes across the sequence"
            )
        world = np.stack([row[0] for row in rows], axis=0).astype(np.float32)
        trajs_2d = np.stack([row[1][:, :2] for row in rows], axis=0).astype(
            np.float32
        )
        raw_visibility = np.stack([row[2] for row in rows], axis=0).astype(bool)
        # A finite mesh vertex exists in the world even while outside the
        # camera. Do not erase full-scene geometry using a projection mask.
        valids = np.isfinite(world).all(axis=-1)
        height, width = record.image_size
        in_bounds = (
            np.isfinite(trajs_2d).all(axis=-1)
            & (trajs_2d[..., 0] >= 0.0)
            & (trajs_2d[..., 0] < width)
            & (trajs_2d[..., 1] >= 0.0)
            & (trajs_2d[..., 1] < height)
        )
        visibs = raw_visibility & in_bounds & valids
        num_frames, num_tracks = world.shape[:2]
        payload = {
            "trajs_2d": np.ascontiguousarray(trajs_2d),
            "trajs_3d_world": np.ascontiguousarray(world),
            "valids": np.ascontiguousarray(valids),
            "visibs": np.ascontiguousarray(visibs),
            "visibility_valids": np.ascontiguousarray(valids),
            "physical_times": np.arange(num_frames, dtype=np.float64),
            "track_ids": np.arange(num_tracks, dtype=np.int64),
            "slot_camera": np.zeros(num_frames, dtype=np.int64),
            "extrinsics": np.ascontiguousarray(record.extrinsics),
        }
        with self._canonical_payload_cache_lock:
            existing = self._canonical_payload_cache.get(record.sequence_name)
            if existing is not None:
                return existing
            self._canonical_payload_cache[record.sequence_name] = payload
            while len(self._canonical_payload_cache) > self.canonical_cache_items:
                self._canonical_payload_cache.popitem(last=False)
        return payload

    def load_clip(
        self,
        sequence_name: str,
        frame_indices: list[int],
        *,
        canonical_anchor_required: bool = False,
    ) -> UnifiedClip:
        """Load a clip from a Dynamic_Replica sequence.

        Parameters
        ----------
        sequence_name :
            Directory name, e.g. ``"009850-3_obj_source_left"``.
        frame_indices :
            Positions into the sequence's sorted frame list.
            Must be in ``[0, num_frames)``.

        Returns
        -------
        UnifiedClip
            Unified intermediate representation.
            ``normals`` is always ``None``.
            ``trajs_2d``, ``trajs_3d_world``, ``visibs``, ``valids`` are set
            for left-camera sequences when ``load_trajectories=True``,
            otherwise ``None``.
        """
        total_t0 = time.perf_counter()
        timing = {
            "traj_load_s": 0.0,
            "rgb_load_s": 0.0,
            "depth_load_s": 0.0,
            "stack_s": 0.0,
            "total_s": 0.0,
        }
        r = self._get_record(sequence_name)
        self._check_indices(frame_indices, r.num_frames, sequence_name)
        clip_has_tracks = r.has_trajectories and self.load_trajectories
        load_depths = not (clip_has_tracks and self.skip_depth_when_tracks)

        images: list[np.ndarray] = []
        dep_paths: list[Path] = []
        frame_paths: list[str] = []
        frame_numbers_clip: list[int] = []
        traj_paths_clip: list[Optional[Path]] = []
        traj_cache_clip: list[
            Optional[_TrajectoryCacheValue]
        ] = []
        selected_positions: list[int] = []

        def _load_frame(idx: int) -> tuple[
            int,
            np.ndarray,
            Path,
            str,
            int,
            Optional[Path],
            Optional[_TrajectoryCacheValue],
            float,
            float,
        ]:
            fn = r.frame_numbers[idx]
            img_path = self.split_root / r.image_rel_paths[idx]
            dep_path = self.split_root / r.depth_rel_paths[idx]
            traj_rel = r.traj_rel_paths[idx]
            traj_path = self.split_root / traj_rel if traj_rel is not None else None
            traj_load_s = 0.0
            rgb_load_s = 0.0
            loaded_traj: Optional[_TrajectoryCacheValue] = None
            use_traj_rgb = (
                self.rgb_from_trajectory
                and self.load_trajectories
                and r.has_trajectories
                and traj_path is not None
            )
            if use_traj_rgb and traj_path is not None:
                t0 = time.perf_counter()
                loaded_traj = _load_trajectory_cached(
                    str(traj_path),
                    prefer_npz=self.prefer_trajectory_npz,
                )
                traj_load_s += time.perf_counter() - t0
                if loaded_traj[3] is not None:
                    image = loaded_traj[3]
                else:
                    t0 = time.perf_counter()
                    image = self._load_rgb_cached(img_path)
                    rgb_load_s += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                image = self._load_rgb_cached(img_path)
                rgb_load_s += time.perf_counter() - t0

            return (
                idx,
                image,
                dep_path,
                str(img_path),
                fn,
                traj_path,
                loaded_traj,
                traj_load_s,
                rgb_load_s,
            )

        if self.io_workers > 1 and len(frame_indices) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(frame_indices), self.io_workers)) as ex:
                frame_rows = list(ex.map(_load_frame, frame_indices))
        else:
            frame_rows = [_load_frame(idx) for idx in frame_indices]

        for (
            idx,
            image,
            dep_path,
            frame_path,
            fn,
            traj_path,
            loaded_traj,
            traj_load_s,
            rgb_load_s,
        ) in frame_rows:
            selected_positions.append(idx)
            images.append(image)
            frame_paths.append(frame_path)
            frame_numbers_clip.append(fn)
            timing["traj_load_s"] += traj_load_s
            timing["rgb_load_s"] += rgb_load_s
            if load_depths:
                dep_paths.append(dep_path)
            if traj_path is not None:
                traj_paths_clip.append(traj_path)
            else:
                traj_paths_clip.append(None)
            traj_cache_clip.append(loaded_traj)

        t0 = time.perf_counter()
        intrinsics = r.intrinsics[selected_positions].copy()
        pixel_coordinates = getattr(self, "pixel_coordinates", "legacy_native")
        if pixel_coordinates == "array_index_v1":
            if canonical_anchor_required:
                raise ValueError("Replica array_index_v1 is supported by the track-native path only")
            # Cached annotations remain native. Convert exactly once per load,
            # together with UV, so old runs and raw caches retain their contract.
            intrinsics[:, :2, 2] -= .5
        extrinsics = r.extrinsics[selected_positions].copy()
        timing["stack_s"] += time.perf_counter() - t0

        # Load trajectories if enabled and available
        has_tracks = (
            self.load_trajectories
            and r.has_trajectories
            and all(p is not None for p in traj_paths_clip)
        )

        trajs_2d: Optional[np.ndarray] = None
        trajs_3d_world: Optional[np.ndarray] = None
        valids: Optional[np.ndarray] = None
        visibs: Optional[np.ndarray] = None

        if has_tracks:
            traj_3d_list: list[np.ndarray] = []
            traj_2d_list: list[np.ndarray] = []
            vis_list: list[np.ndarray] = []
            track_keep_idx: Optional[np.ndarray] = None
            track_points_original: Optional[int] = None

            for p, cached in zip(traj_paths_clip, traj_cache_clip):
                if cached is None:
                    t0 = time.perf_counter()
                    traj_3d, traj_2d, traj_vis, _ = _load_trajectory_cached(
                        str(p),
                        prefer_npz=self.prefer_trajectory_npz,
                    )
                    timing["traj_load_s"] += time.perf_counter() - t0
                else:
                    traj_3d, traj_2d, traj_vis, _ = cached
                if track_points_original is None:
                    track_points_original = int(traj_3d.shape[0])
                    track_keep_idx = self._select_track_points(
                        sequence_name=sequence_name,
                        frame_numbers=frame_numbers_clip,
                        num_points=track_points_original,
                    )
                if track_keep_idx is not None:
                    traj_3d = traj_3d[track_keep_idx]
                    traj_2d = traj_2d[track_keep_idx]
                    traj_vis = traj_vis[track_keep_idx]
                traj_3d_list.append(traj_3d)        # (N, 3)
                traj_2d_list.append(traj_2d[:, :2]) # (N, 2) – drop 3rd col
                vis_list.append(traj_vis)           # (N,)

            t0 = time.perf_counter()
            trajs_3d_world = np.stack(traj_3d_list, axis=0)  # (T, N, 3)
            trajs_2d = np.stack(traj_2d_list, axis=0)        # (T, N, 2)
            if pixel_coordinates == "array_index_v1":
                trajs_2d -= .5
            raw_vis = np.stack(vis_list, axis=0)              # (T, N) bool, occlusion-free per raycast

            # valids: independent geometric validity — has finite 2D coord, finite 3D world coord,
            #   and 2D projection lands inside the image frame. Used as the BCE loss mask.
            H, W = r.image_size
            in_bounds = (
                (trajs_2d[..., 0] >= 0) & (trajs_2d[..., 0] < W) &
                (trajs_2d[..., 1] >= 0) & (trajs_2d[..., 1] < H)
            )
            if pixel_coordinates == "array_index_v1":
                in_bounds &= (trajs_2d[..., 0] <= W-1) & (trajs_2d[..., 1] <= H-1)
            finite_2d = np.isfinite(trajs_2d).all(axis=-1)
            finite_3d = np.isfinite(trajs_3d_world).all(axis=-1)
            valids = finite_2d & finite_3d & in_bounds

            # visibs: a point is "visible" iff (raycast says not occluded) AND (in-frame).
            # Out-of-frame raycast hits default to True in the raw label, so we must AND inbounds.
            visibs = raw_vis & in_bounds
            timing["stack_s"] += time.perf_counter() - t0

        if load_depths:
            t0 = time.perf_counter()
            raw_depths = [_load_depth_raw(p) for p in dep_paths]
            timing["depth_load_s"] += time.perf_counter() - t0
        else:
            raw_depths = []

        if load_depths:
            t0 = time.perf_counter()
            depths = [_decode_depth_raw(raw, 1.0, 0.0) for raw in raw_depths]
            timing["stack_s"] += time.perf_counter() - t0
        else:
            depths = None

        timing["total_s"] = time.perf_counter() - total_t0

        clip_metadata = {
            "pixel_coordinates": pixel_coordinates,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "base_seq_name": r.base_seq_name,
            "camera_name": r.camera_name,
            "sequence_dir": str(r.sequence_dir),
            "frame_indices": list(frame_indices),
            "frame_numbers": frame_numbers_clip,
            "num_frames_in_sequence": r.num_frames,
            "has_depth": load_depths,
            "has_normals": False,
            "has_tracks": has_tracks,
            "has_visibility": has_tracks,
            "has_trajs_3d_world": has_tracks,
            "infer_dynamic_from_authoritative_world_trajectories": has_tracks,
            "dynamic_motion_label_schema": (
                "world_displacement_three_state_strict_static_unknown_band_"
                "distinct_time_v2"
            ),
            "track_points_original": (
                int(track_points_original)
                if has_tracks and "track_points_original" in locals() and track_points_original is not None
                else None
            ),
            "track_points_loaded": (
                int(trajs_2d.shape[1]) if trajs_2d is not None else None
            ),
            "normal_convention": None,
            "normal_supervision_compatible": False,
            "pose_convention": "w2c",
            "extrinsics_convention": "w2c",
            "intrinsics_convention": "pinhole",
            "depth_encoding": (
                "uint16_ieee_float16_bits_camera_z" if load_depths else None
            ),
            "depth_unit": "dynamic_replica_world_units" if load_depths else None,
            "_load_timing": timing,
            "track_source_indices": (
                track_keep_idx.tolist()
                if has_tracks and track_keep_idx is not None
                else (
                    list(range(int(trajs_2d.shape[1])))
                    if trajs_2d is not None else None
                )
            ),
        }
        canonical_payload: dict[str, np.ndarray] = {}
        if canonical_anchor_required:
            if not has_tracks:
                raise RuntimeError(
                    f"[{self.dataset_name}/{sequence_name}] Canonical Stage-1 "
                    "requires persistent trajectory supervision"
                )
            canonical_payload = self._load_canonical_payload(r)
            clip_metadata.update(
                canonical_sequence_time_metadata(
                    num_frames=r.num_frames,
                    reference_time=0.5 * float(r.num_frames - 1),
                    time_domain_id=(
                        f"{self.dataset_name}/{sequence_name}/frame_time_v1"
                    ),
                    track_pool_id=(
                        f"{self.dataset_name}/{sequence_name}/mesh_vertex_pool_v1"
                    ),
                    track_pool_stable=True,
                )
            )
            clip_metadata["canonical_payload_source"] = (
                "dynamic_replica_full_persistent_mesh_vertices_v1"
            )
            anchor_frame_index = self.get_canonical_anchor_frame_index(sequence_name)
            anchor_matches = [
                slot
                for slot, position in enumerate(frame_indices)
                if int(position) == anchor_frame_index
            ]
            if len(anchor_matches) != 1:
                raise RuntimeError(
                    f"[{self.dataset_name}/{sequence_name}] canonical anchor "
                    "must occur exactly once in the clip; got "
                    f"{len(anchor_matches)} occurrences"
                )
            anchor_extrinsic = np.asarray(
                r.extrinsics[anchor_frame_index], dtype=np.float32
            )
            if (
                anchor_extrinsic.shape != (4, 4)
                or not np.isfinite(anchor_extrinsic).all()
            ):
                raise RuntimeError(
                    f"[{self.dataset_name}/{sequence_name}] canonical anchor pose "
                    "is missing or non-finite"
                )
            clip_metadata.update(
                canonical_scene_anchor_metadata(
                    extrinsic=anchor_extrinsic,
                    time=float(anchor_frame_index),
                    camera_id=0,
                    frame_id=(
                        f"{self.dataset_name}/{sequence_name}/"
                        f"camera={r.camera_name}/position={anchor_frame_index}/"
                        f"frame={r.frame_numbers[anchor_frame_index]}"
                    ),
                    anchor_slot=anchor_matches[0],
                    num_slots=len(frame_indices),
                    require_in_clip=True,
                )
            )

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths,
            normals=None,
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d_world,
            valids=valids,
            visibs=visibs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata=clip_metadata,
            visibility_valids=valids.copy() if valids is not None else None,
            canonical_trajs_2d=canonical_payload.get("trajs_2d"),
            canonical_trajs_3d_world=canonical_payload.get("trajs_3d_world"),
            canonical_valids=canonical_payload.get("valids"),
            canonical_visibs=canonical_payload.get("visibs"),
            canonical_visibility_valids=canonical_payload.get(
                "visibility_valids"
            ),
            canonical_physical_times=canonical_payload.get("physical_times"),
            canonical_track_ids=canonical_payload.get("track_ids"),
            canonical_slot_camera=canonical_payload.get("slot_camera"),
            canonical_slot_physical_camera_id=canonical_payload.get(
                "slot_camera"
            ),
            canonical_extrinsics=canonical_payload.get("extrinsics"),
        )

    def _rgb_cache_key(self, path: Path) -> tuple:
        """Build a stable cache key that changes when the file changes."""
        try:
            st = path.stat()
            return ("rgb", int(st.st_dev), int(st.st_ino), int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            return ("rgb", str(path))

    def _load_rgb_cached(self, path: Path) -> np.ndarray:
        if self.rgb_cache_items <= 0:
            return _load_rgb(path)

        key = self._rgb_cache_key(path)
        with self._rgb_cache_lock:
            cached = self._rgb_cache.get(key)
            if cached is not None:
                self._rgb_cache.move_to_end(key)
                return cached.copy()

        arr = _load_rgb(path)
        with self._rgb_cache_lock:
            cached = self._rgb_cache.get(key)
            if cached is not None:
                self._rgb_cache.move_to_end(key)
                return cached.copy()
            self._rgb_cache[key] = arr
            while len(self._rgb_cache) > self.rgb_cache_items:
                self._rgb_cache.popitem(last=False)
            return arr.copy()

    def _select_track_points(
        self,
        *,
        sequence_name: str,
        frame_numbers: list[int],
        num_points: int,
    ) -> Optional[np.ndarray]:
        max_points = self.load_max_track_points
        if max_points is None or num_points <= max_points:
            return None
        payload = (
            f"dynamic_replica|{sequence_name}|{max_points}|"
            + ",".join(str(int(v)) for v in frame_numbers)
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        return np.sort(
            rng.choice(num_points, size=max_points, replace=False).astype(np.int64)
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        """Run consistency checks on a sequence.

        Checks
        ------
        1. Image and depth files exist for the first 5 frames.
        2. Intrinsics: shape (T, 3, 3), finite, positive focal lengths.
        3. Extrinsics: shape (T, 4, 4), finite, rotation near-orthonormal,
           last row == [0, 0, 0, 1].
        4. Depth values finite; at least some positive values per frame.
        5. Round-trip reprojection error < 1 px (depth → world → camera → pixel).
        6. If trajectories available: shapes [T, N, 2], [T, N, 3], [T, N];
           traj_2d coordinates inside image bounds.

        Returns
        -------
        dict with ``dataset_name``, ``sequence_name``, ``ok``, ``messages``.
        """
        r = self._get_record(sequence_name)
        msgs: list[str] = []
        ok = True

        probe_count = min(5, r.num_frames)

        # ``skip_depth_when_tracks`` is an intentional trajectory-only mode:
        # depth files are not loaded and must not be required by this check.
        has_tracks = bool(r.has_trajectories and self.load_trajectories)
        check_depth = not (has_tracks and self.skip_depth_when_tracks)

        # ── 1. File existence ────────────────────────────────────────────
        for i in range(probe_count):
            fn = r.frame_numbers[i]
            required_files = [("image", r.image_rel_paths[i])]
            if check_depth:
                required_files.append(("depth", r.depth_rel_paths[i]))
            for key, rel in required_files:
                p = self.split_root / rel
                if not p.exists():
                    ok = False
                    msgs.append(f"frame {fn}: missing {key} file: {p}")

        if not ok:
            return {
                "dataset_name": self.dataset_name,
                "sequence_name": sequence_name,
                "ok": False,
                "messages": msgs,
            }

        # ── Load probe clip ──────────────────────────────────────────────
        try:
            clip = self.load_clip(sequence_name, list(range(probe_count)))
        except Exception as exc:
            return {
                "dataset_name": self.dataset_name,
                "sequence_name": sequence_name,
                "ok": False,
                "messages": msgs + [f"load_clip failed: {repr(exc)}"],
            }

        T = probe_count

        # ── 2. Intrinsics ───────────────────────────────────────────────
        if clip.intrinsics.shape != (T, 3, 3):
            ok = False
            msgs.append(f"intrinsics shape {clip.intrinsics.shape} != ({T},3,3)")
        elif not np.isfinite(clip.intrinsics).all():
            ok = False
            msgs.append("intrinsics contains non-finite values")
        else:
            fx = clip.intrinsics[:, 0, 0]
            fy = clip.intrinsics[:, 1, 1]
            if not (np.all(fx > 0) and np.all(fy > 0)):
                ok = False
                msgs.append(f"non-positive focal lengths: fx={fx}, fy={fy}")

        # ── 3. Extrinsics ───────────────────────────────────────────────
        if clip.extrinsics.shape != (T, 4, 4):
            ok = False
            msgs.append(f"extrinsics shape {clip.extrinsics.shape} != ({T},4,4)")
        elif not np.isfinite(clip.extrinsics).all():
            ok = False
            msgs.append("extrinsics contains non-finite values")
        else:
            bottom = clip.extrinsics[:, 3, :]
            if not np.allclose(bottom, [0, 0, 0, 1], atol=1e-5):
                ok = False
                msgs.append("extrinsics last row != [0,0,0,1]")
            R = clip.extrinsics[:, :3, :3].astype(np.float64)
            RRt = R @ np.transpose(R, (0, 2, 1))
            err = float(np.linalg.norm(RRt - np.eye(3)[None], axis=(1, 2)).max())
            if err > 1e-4:
                ok = False
                msgs.append(f"rotation matrices non-orthonormal: max ||RR^T-I||_F={err:.2e}")

        # ── 4/5. Depth and reprojection ─────────────────────────────────
        # In trajectory-only mode ``UnifiedClip.depths`` is None by design;
        # there is no depth contract to validate or reproject.
        if clip.depths is not None:
            for t in range(T):
                dep = clip.depths[t]
                if not np.isfinite(dep).all():
                    ok = False
                    msgs.append(f"depth[{t}] contains non-finite values")
                elif not (dep >= 0).all():
                    ok = False
                    msgs.append(f"depth[{t}] contains negative values")

            # ── 5. Reprojection ──────────────────────────────────────────
            for t in range(T):
                dep = clip.depths[t]
                K = clip.intrinsics[t].astype(np.float64)
                E = clip.extrinsics[t].astype(np.float64)

                ys, xs = np.where(dep > 0)
                if ys.size == 0:
                    msgs.append(f"reproj check skipped frame {t}: no valid depth")
                    continue

                mid = ys.size // 2
                u_px, v_px = int(xs[mid]), int(ys[mid])
                d = float(dep[v_px, u_px])

                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                xc = (u_px - cx) / fx * d
                yc = (v_px - cy) / fy * d
                p_cam = np.array([xc, yc, d, 1.0])

                try:
                    p_world = np.linalg.inv(E) @ p_cam
                except np.linalg.LinAlgError:
                    ok = False
                    msgs.append(f"extrinsics[{t}] singular")
                    continue

                p_cam2 = E @ p_world
                if abs(p_cam2[2]) < 1e-8:
                    msgs.append(f"reproj frame {t}: reprojected depth ≈ 0, skipped")
                    continue

                u2 = K[0, 0] * p_cam2[0] / p_cam2[2] + K[0, 2]
                v2 = K[1, 1] * p_cam2[1] / p_cam2[2] + K[1, 2]
                reproj_err = float(np.hypot(u2 - u_px, v2 - v_px))
                if reproj_err > 1.0:
                    ok = False
                    msgs.append(
                        f"reproj error frame {t}: err={reproj_err:.4f}px "
                        f"(reprojected ({u2:.2f},{v2:.2f}) vs original ({u_px},{v_px}))"
                    )

        # ── 6. Trajectories (if loaded) ──────────────────────────────────
        if clip.trajs_2d is not None:
            N = clip.trajs_2d.shape[1]
            if clip.trajs_2d.shape != (T, N, 2):
                ok = False
                msgs.append(f"trajs_2d shape {clip.trajs_2d.shape} != ({T},{N},2)")
            if clip.trajs_3d_world is not None and clip.trajs_3d_world.shape != (T, N, 3):
                ok = False
                msgs.append(f"trajs_3d_world shape {clip.trajs_3d_world.shape} != ({T},{N},3)")
            if clip.visibs is not None and clip.visibs.shape != (T, N):
                ok = False
                msgs.append(f"visibs shape {clip.visibs.shape} != ({T},{N})")
            if clip.valids is not None and clip.valids.shape != (T, N):
                ok = False
                msgs.append(f"valids shape {clip.valids.shape} != ({T},{N})")

            # traj_2d should be in roughly image pixel range
            if clip.images:
                H_img, W_img = clip.images[0].shape[:2]
                vis_mask = clip.visibs if clip.visibs is not None else np.ones((T, N), bool)
                if vis_mask.any():
                    uv_vis = clip.trajs_2d[vis_mask]
                    in_bounds = (
                        (uv_vis[:, 0] >= 0) & (uv_vis[:, 0] < W_img)
                        & (uv_vis[:, 1] >= 0) & (uv_vis[:, 1] < H_img)
                    )
                    frac_in = float(in_bounds.mean())
                    if frac_in < 0.5:
                        ok = False
                        msgs.append(
                            f"trajs_2d visible points mostly out of bounds "
                            f"(frac_in_bounds={frac_in:.3f})"
                        )

        if ok and not msgs:
            msgs.append("all checks passed")

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "ok": ok,
            "messages": msgs,
            "num_frames": r.num_frames,
            "probe_frames": probe_count,
        }

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> list[_DRSequenceRecord]:
        anno_file = self._annotation_file()
        if not anno_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {anno_file}. "
                f"Expected at: <root>/<split>/frame_annotations_<split>.jgz"
            )

        raw = _load_jgz(anno_file)
        grouped: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
        records: list[_DRSequenceRecord] = []
        skipped: list[str] = []

        for entry in raw:
            try:
                base_seq_name = str(entry["sequence_name"])
                camera_name = str(entry["camera_name"])
                if camera_name not in ("left", "right"):
                    raise ValueError(f"unsupported camera_name={camera_name!r}")

                image_rel_path = str(entry["image"]["path"])
                path_parts = Path(image_rel_path).parts
                if not path_parts:
                    raise ValueError("empty image path in annotation")
                sequence_name = path_parts[0]

                frame_number = int(entry["frame_number"])
                frame_map = grouped.setdefault(
                    (sequence_name, base_seq_name, camera_name),
                    {},
                )
                frame_map[frame_number] = {
                    "image_rel_path": image_rel_path,
                    "depth_rel_path": str(entry["depth"]["path"]),
                    "traj_rel_path": (
                        None
                        if entry.get("trajectories", {}).get("path") is None
                        else str(entry["trajectories"]["path"])
                    ),
                    "image_size": tuple(int(v) for v in entry["image"]["size"]),
                    "intrinsics": _ndc_to_pinhole(
                        entry["viewpoint"]["focal_length"],
                        entry["viewpoint"]["principal_point"],
                        entry["image"]["size"],
                    ),
                    "extrinsics": _p3d_to_opencv_extrinsics(
                        entry["viewpoint"]["R"],
                        entry["viewpoint"]["T"],
                    ),
                }
            except Exception as exc:
                if self.strict:
                    raise
                skipped.append(f"annotation entry {entry!r}: {exc}")
                if self.verbose:
                    print(f"[DynamicReplicaAdapter][WARN] skip malformed entry: {exc}")

        for (sequence_name, base_seq_name, camera_name), frame_map in sorted(grouped.items()):
            try:
                if not frame_map:
                    continue

                ordered_items = sorted(frame_map.items())
                frame_numbers = tuple(fn for fn, _ in ordered_items)
                if len(frame_numbers) < self.min_frames:
                    continue

                sequence_dir = self.split_root / sequence_name
                if self.strict:
                    images_dir = sequence_dir / "images"
                    if not images_dir.exists():
                        raise FileNotFoundError(f"missing images dir: {images_dir}")

                first_frame = ordered_items[0][1]
                image_rel_paths = tuple(item["image_rel_path"] for _, item in ordered_items)
                depth_rel_paths = tuple(item["depth_rel_path"] for _, item in ordered_items)
                traj_rel_paths = tuple(item["traj_rel_path"] for _, item in ordered_items)
                intrinsics = np.stack(
                    [item["intrinsics"] for _, item in ordered_items],
                    axis=0,
                ).astype(np.float32, copy=False)
                extrinsics = np.stack(
                    [item["extrinsics"] for _, item in ordered_items],
                    axis=0,
                ).astype(np.float32, copy=False)

                records.append(
                    _DRSequenceRecord(
                        sequence_name=sequence_name,
                        base_seq_name=base_seq_name,
                        camera_name=camera_name,
                        split=self.split,
                        sequence_dir=sequence_dir,
                        frame_numbers=frame_numbers,
                        image_size=first_frame["image_size"],
                        image_rel_paths=image_rel_paths,
                        depth_rel_paths=depth_rel_paths,
                        traj_rel_paths=traj_rel_paths,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        has_trajectories=(
                            camera_name == "left"
                            and any(path is not None for path in traj_rel_paths)
                        ),
                    )
                )
            except Exception as exc:
                if self.strict:
                    raise
                skipped.append(f"{sequence_name}: {exc}")
                if self.verbose:
                    print(f"[DynamicReplicaAdapter][WARN] skip {sequence_name}: {exc}")

        if self.verbose and skipped:
            print(
                f"[DynamicReplicaAdapter] skipped {len(skipped)} malformed entries/sequences "
                "(non-strict mode)"
            )

        return records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _annotation_file(self) -> Path:
        return self.split_root / f"frame_annotations_{self.split}.jgz"

    def _build_cache_suffix(self) -> str:
        anno_file = self._annotation_file()
        stat_size = None
        stat_mtime_ns = None
        try:
            stat = anno_file.stat()
        except OSError:
            stat = None
        if stat is not None:
            stat_size = stat.st_size
            stat_mtime_ns = stat.st_mtime_ns

        cache_key = {
            "dataset": self.dataset_name,
            "split": self.split,
            "root": str(self.root),
            "min_frames": self.min_frames,
            "strict": self.strict,
            "schema": self._CACHE_SCHEMA_VERSION,
            "annotation_file": str(anno_file),
            "annotation_size": stat_size,
            "annotation_mtime_ns": stat_mtime_ns,
        }
        return hashlib.sha1(
            json.dumps(cache_key, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

    def _depth_calibration_cache_path(self, sequence_name: str) -> Optional[Path]:
        if self._depth_calibration_cache_dir is None:
            return None
        return self._depth_calibration_cache_dir / f"{sequence_name}.json"

    def _load_cached_depth_calibration(
        self,
        sequence_name: str,
    ) -> Optional[tuple[float, float]]:
        cache_path = self._depth_calibration_cache_path(sequence_name)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text())
            scale = float(payload["scale"])
            offset = float(payload["offset"])
        except Exception:
            return None
        if scale <= 0 or not np.isfinite(scale) or not np.isfinite(offset):
            return None
        return (scale, offset)

    def _store_depth_calibration(
        self,
        sequence_name: str,
        scale: float,
        offset: float,
    ) -> None:
        cache_path = self._depth_calibration_cache_path(sequence_name)
        if cache_path is None:
            return
        tmp_path: Optional[Path] = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f".tmp{os.getpid()}")
            tmp_path.write_text(json.dumps({"scale": scale, "offset": offset}))
            os.replace(tmp_path, cache_path)
        except OSError:
            try:
                if tmp_path is not None:
                    tmp_path.unlink()
            except Exception:
                pass

    def _get_depth_calibration(
        self,
        sequence_name: str,
        raw_depths: list[np.ndarray],
        trajs_3d_world: np.ndarray,
        trajs_2d: np.ndarray,
        visibs: np.ndarray,
        extrinsics: np.ndarray,
    ) -> tuple[float, float]:
        cached = self._depth_calibration_cache.get(sequence_name)
        if cached is not None:
            return cached

        cached = self._load_cached_depth_calibration(sequence_name)
        if cached is not None:
            self._depth_calibration_cache[sequence_name] = cached
            return cached

        calibration = _calibrate_depth_linear_multiframe(
            raw_depths=raw_depths,
            trajs_3d_world=trajs_3d_world,
            trajs_2d=trajs_2d,
            visibs=visibs,
            extrinsics=extrinsics,
            max_frames=5,
        )
        self._depth_calibration_cache[sequence_name] = calibration
        self._store_depth_calibration(sequence_name, *calibration)
        return calibration

    def _get_record(self, sequence_name: str) -> _DRSequenceRecord:
        if sequence_name not in self._name_to_record:
            raise KeyError(
                f"Unknown sequence_name: '{sequence_name}'. "
                f"Use list_sequences() to enumerate valid names."
            )
        return self._name_to_record[sequence_name]

    def _check_indices(
        self,
        frame_indices: list[int],
        num_frames: int,
        sequence_name: str,
    ) -> None:
        if not frame_indices:
            raise ValueError("frame_indices is empty")
        lo, hi = min(frame_indices), max(frame_indices)
        if lo < 0 or hi >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={lo}, max={hi}, num_frames={num_frames}"
            )
