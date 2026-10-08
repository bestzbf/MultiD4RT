from __future__ import annotations

import json
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from datasets.computer.depth_to_tracks import (
    TRACK_SEMANTICS_VERSION,
    _project_world_points_to_frames,
    recompute_track_projection_masks,
)

from .base import BaseAdapter, UnifiedClip


MVSSYNTH_TRACK_DEPTH_MAX_M = 100.0


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


def _resolve_nonnegative_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, out)


def _sleep_before_retry(delay_s: float, attempt: int) -> None:
    if delay_s <= 0:
        return
    time.sleep(float(delay_s) * float(attempt + 1))


# ---------------------------------------------------------------------------
# EXR depth reader
# ---------------------------------------------------------------------------

def _read_exr_depth(
    path: Path,
    retries: int = 0,
    retry_delay_s: float = 0.0,
) -> np.ndarray:
    """Read OpenEXR depth map and convert inf to 0.

    Requires opencv-python with OpenEXR support.
    Set OPENCV_IO_ENABLE_OPENEXR=1 environment variable if needed.

    Returns:
        depth: (H, W) float32 array. Invalid pixels (inf) are set to 0.
    """
    import cv2

    attempts = max(1, int(retries) + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            dep = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if dep is None:
                raise RuntimeError(f"cv2.imread returned None for EXR file: {path}")
            dep = dep.astype(np.float32)
            dep[~np.isfinite(dep)] = 0.0
            return dep
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                _sleep_before_retry(retry_delay_s, attempt)
    raise RuntimeError(f"Failed to read EXR file after {attempts} attempts: {path}") from last_error


def _read_rgb(
    path: Path,
    retries: int = 0,
    retry_delay_s: float = 0.0,
) -> np.ndarray:
    """Load RGB image as uint8 (H, W, 3)."""
    import cv2

    attempts = max(1, int(retries) + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"cv2.imread returned None for image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                _sleep_before_retry(retry_delay_s, attempt)
    raise RuntimeError(f"Failed to read image after {attempts} attempts: {path}") from last_error


# ---------------------------------------------------------------------------
# Sequence record
# ---------------------------------------------------------------------------

class _SequenceRecord:
    """Index entry for one MVS-Synth sequence."""

    __slots__ = ("sequence_id", "sequence_dir", "num_frames")

    def __init__(self, sequence_id: str, sequence_dir: Path, num_frames: int):
        self.sequence_id = sequence_id
        self.sequence_dir = sequence_dir
        self.num_frames = num_frames

    def image_path(self, frame_idx: int) -> Path:
        return self.sequence_dir / "images" / f"{frame_idx:04d}.png"

    def depth_path(self, frame_idx: int) -> Path:
        return self.sequence_dir / "depths" / f"{frame_idx:04d}.exr"

    def pose_path(self, frame_idx: int) -> Path:
        return self.sequence_dir / "poses" / f"{frame_idx:04d}.json"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MVSSynthAdapter(BaseAdapter):
    """Dataset adapter for MVS-Synth (GTAV_1080 variant).

    Expected dataset layout::

        <root>/
            num_images.json              # [100, 100, ..., 100] (120 sequences)
            0000/
                images/
                    0000.png, 0001.png, ..., 0099.png
                depths/
                    0000.exr, 0001.exr, ..., 0099.exr
                poses/
                    0000.json, 0001.json, ..., 0099.json
            0001/
                ...
            ...
            0119/
                ...

    Each pose JSON contains::

        {
            "f_x": 1157.84...,
            "f_y": 1157.84...,
            "c_x": 960,
            "c_y": 540,
            "extrinsic": [[4x4 matrix]]  # world-to-camera (w2c)
        }

    Depth maps are OpenEXR float32 with inf values for invalid/sky pixels.

    Supervision availability
    ------------------------
    - ``has_depth``      : True
    - ``has_normals``    : False
    - ``has_tracks``     : False
    - ``has_visibility`` : False

    Extrinsics convention
    ---------------------
    MVS-Synth stores **world-to-camera** (w2c) transforms in a left-handed
    GTA V world (X=right, Y=forward, Z=up).  The adapter converts to a
    right-handed OpenCV-compatible world via the basis change
    P = [[0,1,0],[0,0,-1],[1,0,0]] (new_X=forward, new_Y=down, new_Z=right).
    After this transform det(R) ≈ +1 and the camera follows the RDF
    (Right-Down-Forward) convention expected by rerun's Pinhole archetype.
    """

    dataset_name: str = "mvssynth"

    def __init__(
        self,
        root: str,
        split: str = "train",
        strict: bool = True,
        verbose: bool = True,
        precompute_root: Optional[str] = None,
        load_precomputed: bool = True,
        load_normals: bool = False,
        visibility_depth_support_radius: int = 2,
        visibility_visible_abs_thresh: float = 0.25,
        visibility_occlusion_abs_thresh: float = 1.0,
        visibility_occlusion_rel_thresh: float = 0.08,
        visibility_occlusion_min_support: int = 9,
        visibility_occlusion_far_depth_start: Optional[float] = None,
        visibility_occlusion_far_depth_end: Optional[float] = None,
        visibility_occlusion_far_abs_thresh: Optional[float] = None,
        visibility_occlusion_far_rel_thresh: Optional[float] = None,
        visibility_occlusion_far_min_support: Optional[int] = None,
        visibility_far_uncertain_as_visible_depth: Optional[float] = None,
        visibility_depth_max: Optional[float] = None,
        cache_dir: Optional[str] = None,
        io_workers: Optional[int] = None,
        io_read_retries: Optional[int] = None,
        io_retry_delay_s: Optional[float] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        root :
            Root directory of the MVS-Synth GTAV_1080 dataset.
        split :
            Split name (ignored, MVS-Synth doesn't have splits).
        strict :
            If True, raise on any indexing error.
            If False, skip broken sequences with a warning.
        verbose :
            Print summary after index construction.
        load_precomputed :
            If True, load precomputed tracks/normals from ``precompute_root``.
        load_normals :
            If True, load precomputed full-resolution normal maps. Defaults to
            False because MVS-Synth normals are large and slow to slice from
            COS-backed HDF5; tracks remain enabled either way.
        cache_dir :
            Optional local index-cache directory. When present, cached sequence
            metadata avoids repeated directory walks on NFS/COS mounts.
        io_workers :
            Number of per-sample threads used to read RGB/depth/pose frames.
        io_read_retries :
            Number of extra read/stat attempts for delayed NFS/COS visibility.
        io_retry_delay_s :
            Base delay between retries; each retry waits a little longer.
        """
        self.split = split
        self.root = Path(root)
        self.strict = strict
        self.verbose = verbose
        if io_workers is None:
            io_workers = os.getenv("D4RT_MVSSYNTH_IO_WORKERS", "4")
        if io_read_retries is None:
            io_read_retries = os.getenv("D4RT_MVSSYNTH_IO_READ_RETRIES", "3")
        if io_retry_delay_s is None:
            io_retry_delay_s = os.getenv("D4RT_MVSSYNTH_IO_RETRY_DELAY_S", "0.05")
        self.io_workers = _resolve_positive_int(io_workers, 1)
        self.io_read_retries = _resolve_nonnegative_int(io_read_retries, 2)
        self.io_retry_delay_s = _resolve_nonnegative_float(io_retry_delay_s, 0.05)
        self.load_precomputed = bool(load_precomputed)
        self.precompute_root = (
            Path(precompute_root) if precompute_root else self.root
        ) if self.load_precomputed else None
        self.load_normals = bool(load_normals)
        self._precomputed_info_cache: dict[str, dict[str, Any]] = {}

        # Check cache first to skip slow root.exists() on remote storage
        cache_path = self._index_cache_path(Path(cache_dir) if cache_dir is not None else None)
        _cache_hit = bool(cache_path is not None and self._path_exists_retry(cache_path))
        if not _cache_hit and not self._path_exists_retry(self.root):
            raise FileNotFoundError(f"MVS-Synth root not found: {self.root}")

        self.visibility_depth_support_radius = max(int(visibility_depth_support_radius), 0)
        self.visibility_visible_abs_thresh = float(visibility_visible_abs_thresh)
        self.visibility_occlusion_abs_thresh = float(visibility_occlusion_abs_thresh)
        self.visibility_occlusion_rel_thresh = float(visibility_occlusion_rel_thresh)
        self.visibility_occlusion_min_support = max(int(visibility_occlusion_min_support), 1)
        def _optional_positive_float(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            value_f = float(value)
            return value_f if np.isfinite(value_f) and value_f > 0.0 else None

        self.visibility_occlusion_far_depth_start = _optional_positive_float(
            visibility_occlusion_far_depth_start
        )
        self.visibility_occlusion_far_depth_end = _optional_positive_float(
            visibility_occlusion_far_depth_end
        )
        self.visibility_occlusion_far_abs_thresh = _optional_positive_float(
            visibility_occlusion_far_abs_thresh
        )
        self.visibility_occlusion_far_rel_thresh = _optional_positive_float(
            visibility_occlusion_far_rel_thresh
        )
        self.visibility_occlusion_far_min_support = (
            max(int(visibility_occlusion_far_min_support), 1)
            if visibility_occlusion_far_min_support is not None
            else None
        )
        self.visibility_far_uncertain_as_visible_depth = _optional_positive_float(
            visibility_far_uncertain_as_visible_depth
        )
        self.visibility_depth_max = (
            float(visibility_depth_max)
            if visibility_depth_max is not None and float(visibility_depth_max) > 0
            else None
        )

        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

        if cache_dir is not None:
            from datasets.index_cache import load_or_build
            assert cache_path is not None
            self._records = load_or_build(self._build_index, cache_path)
        else:
            self._records = self._build_index()
        self._name_to_record: dict[str, _SequenceRecord] = {
            r.sequence_id: r for r in self._records
        }

        if not self._records:
            raise RuntimeError(
                f"No valid MVS-Synth sequences found under {self.root}"
            )

        if self.verbose:
            print(
                f"[MVSSynthAdapter] root={self.root.name}, "
                f"sequences={len(self._records)}, "
                f"precompute_root={self.precompute_root}, "
                f"io_workers={self.io_workers}, "
                f"io_read_retries={self.io_read_retries}"
            )

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [r.sequence_id for r in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].sequence_id

    def _index_cache_path(self, cache_dir: Optional[Path]) -> Optional[Path]:
        if cache_dir is None:
            return None
        cache_key = {
            "dataset": self.dataset_name,
            "root": str(self.root),
            "precompute_root": str(self.precompute_root),
            "split": self.split,
            "cache_schema": 2,
        }
        suffix = hashlib.sha1(
            json.dumps(cache_key, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return cache_dir / f"{self.dataset_name}_{self.split}_{suffix}.pkl"

    def _retry_call(self, label: str, fn: Callable[[], Any]) -> Any:
        attempts = max(1, self.io_read_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    _sleep_before_retry(self.io_retry_delay_s, attempt)
        raise RuntimeError(f"Failed to {label} after {attempts} attempts") from last_error

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

    def _path_is_dir_retry(self, path: Path) -> bool:
        attempts = max(1, self.io_read_retries + 1)
        for attempt in range(attempts):
            try:
                if path.is_dir():
                    return True
            except OSError:
                pass
            if attempt + 1 < attempts:
                _sleep_before_retry(self.io_retry_delay_s, attempt)
        return False

    def _iterdir_retry(self, path: Path) -> list[Path]:
        return self._retry_call(f"list directory {path}", lambda: sorted(path.iterdir()))

    def _glob_retry(self, path: Path, pattern: str) -> list[Path]:
        return self._retry_call(
            f"glob {path}/{pattern}",
            lambda: sorted(path.glob(pattern)),
        )

    def _read_json_retry(self, path: Path) -> Any:
        def _read() -> Any:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        return self._retry_call(f"read json {path}", _read)

    def _read_rgb_retry(self, path: Path) -> np.ndarray:
        return _read_rgb(
            path,
            retries=self.io_read_retries,
            retry_delay_s=self.io_retry_delay_s,
        )

    def _read_exr_depth_retry(self, path: Path) -> np.ndarray:
        return _read_exr_depth(
            path,
            retries=self.io_read_retries,
            retry_delay_s=self.io_retry_delay_s,
        )

    def _get_precomputed_info(self, sequence_name: str) -> dict[str, Any]:
        cached = self._precomputed_info_cache.get(sequence_name)
        if cached is not None:
            return cached

        has_tracks = False
        has_normals = False
        backend = None
        if self.precompute_root is not None:
            npz_path = self.precompute_root / sequence_name / "precomputed.npz"
            h5_path = npz_path.with_suffix(".h5")
            has_h5 = self._path_exists_retry(h5_path)
            has_npz = self._path_exists_retry(npz_path)
            has_tracks = has_h5 or has_npz
            backend = "h5" if has_h5 else ("npz" if has_npz else None)
            if has_tracks:
                try:
                    if has_h5:
                        import h5py

                        with h5py.File(h5_path, "r") as f:
                            has_normals = "normals" in f
                    elif has_npz:
                        with np.load(npz_path, mmap_mode="r", allow_pickle=False) as data:
                            has_normals = "normals" in data.files
                except Exception:
                    has_normals = False

        info = {
            "has_tracks": bool(has_tracks),
            "has_normals": bool(has_normals and self.load_normals),
            "has_visibility": bool(has_tracks),
            "backend": backend,
        }
        self._precomputed_info_cache[sequence_name] = info
        return info

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self._get_record(sequence_name)

        # Probe first frame for image size
        img_path = r.image_path(0)
        img = self._read_rgb_retry(img_path)
        H, W = img.shape[:2]
        precomputed = self._get_precomputed_info(sequence_name)

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": r.sequence_id,
            "sequence_root": str(r.sequence_dir),
            "num_frames": r.num_frames,
            "height": H,
            "width": W,
            "has_depth": True,
            "has_normals": precomputed["has_normals"],
            "has_tracks": precomputed["has_tracks"],
            "has_visibility": precomputed["has_visibility"],
            "has_trajs_3d_world": precomputed["has_tracks"],
            "precomputed_backend": precomputed["backend"],
            "extrinsics_convention": "w2c",
            "depth_unit": "meters",
        }

    def get_num_frames(self, sequence_name: str) -> int:
        return self._get_record(sequence_name).num_frames

    def load_clip(
        self, sequence_name: str, frame_indices: list[int]
    ) -> UnifiedClip:
        """Load a clip from an MVS-Synth sequence.

        Parameters
        ----------
        sequence_name :
            Sequence ID (e.g., "0000", "0042").
        frame_indices :
            Frame indices within [0, num_frames).

        Returns
        -------
        UnifiedClip
            Unified intermediate representation.
            ``normals``, ``trajs_2d``, ``trajs_3d_world``, ``valids``,
            and ``visibs`` are loaded from precomputed cache when available;
            otherwise they are ``None``.
        """
        r = self._get_record(sequence_name)
        self._check_indices(frame_indices, r.num_frames, sequence_name)

        images: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        intrinsics_list: list[np.ndarray] = []
        extrinsics_list: list[np.ndarray] = []
        frame_paths: list[str] = []

        # GTA V uses a left-handed world coordinate system
        # (X=right, Y=forward, Z=up, det(R)=-1).
        # Convert to a right-handed OpenCV-compatible world:
        #   new_X = old_Y  (forward)
        #   new_Y = -old_Z (down)
        #   new_Z = old_X  (right)
        # Basis-change matrix P (new_world = P @ old_world), det(P)=-1:
        #   P = [[0,1,0],[0,0,-1],[1,0,0]]
        # For a w2c matrix E=[R|t]:
        #   E_new = E @ P^{-1} = E @ P^T
        # det(R_new) = det(R)*det(P) = (-1)*(-1) = +1
        # Camera axes in new world: cam_Y ≈ down, cam_Z ≈ forward (RDF).
        P = np.array(
            [[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32
        )

        def _load_frame(idx: int) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            str,
        ]:
            img_path = r.image_path(idx)
            dep_path = r.depth_path(idx)
            pose_path = r.pose_path(idx)

            image = self._read_rgb_retry(img_path)
            # Raw depth is in centimeters; convert to meters
            depth = self._read_exr_depth_retry(dep_path) / 100.0

            pose = self._read_json_retry(pose_path)

            fx = float(pose["f_x"])
            fy = float(pose["f_y"])
            cx = float(pose["c_x"])
            cy = float(pose["c_y"])
            K = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

            E = np.array(pose["extrinsic"], dtype=np.float32)  # w2c
            # Translation is in centimeters; convert to meters
            E[:3, 3] /= 100.0
            E[:3, :3] = E[:3, :3] @ P.T
            return image, depth, K, E, str(img_path)

        if self.io_workers > 1 and len(frame_indices) > 1:
            with ThreadPoolExecutor(max_workers=min(self.io_workers, len(frame_indices))) as ex:
                frame_rows = list(ex.map(_load_frame, frame_indices))
        else:
            frame_rows = [_load_frame(idx) for idx in frame_indices]

        for image, depth, K, E, frame_path in frame_rows:
            images.append(image)
            depths.append(depth)
            intrinsics_list.append(K)
            extrinsics_list.append(E)
            frame_paths.append(frame_path)

        intrinsics = np.stack(intrinsics_list, axis=0)   # (T, 3, 3)
        extrinsics = np.stack(extrinsics_list, axis=0)   # (T, 4, 4) w2c

        # Load precomputed normals / tracks if available.
        # This must happen before origin shifting so that, for cached tracks,
        # we can reuse the exact full-sequence frame-0 shift used at
        # precompute time.  Sub-clips may start at a later frame.
        normals_out, trajs_2d_out, trajs_3d_out, valids_out, visibs_out = \
            None, None, None, None, None
        has_normals_out, has_tracks_out = False, False
        precomputed_origin_shift: np.ndarray | None = None
        precomputed_track_semantics_version: Optional[int] = None
        active_track_semantics_version: Optional[int] = None
        precomputed_track_semantics_refreshed = False
        has_precomputed_cache = False
        if self.precompute_root is not None:
            cache = self._load_precomputed(sequence_name, frame_indices)
            if cache is not None:
                has_precomputed_cache = True
                if self.load_normals and "normals" in cache:
                    normals_out = [n.astype(np.float32) for n in cache["normals"]]
                    has_normals_out = True
                if "trajs_2d" in cache:
                    trajs_2d_out = cache["trajs_2d"]
                if "trajs_3d_world" in cache:
                    # compute_tracks() receives already-converted data from this
                    # adapter: meters, right-handed world, and origin-shifted.  The
                    # cached 3-D tracks are therefore already in the same world
                    # space as the extrinsics below; applying the raw GTA unit/basis
                    # conversion again would corrupt them.
                    trajs_3d_out = cache["trajs_3d_world"].astype(np.float32)
                if "valids" in cache:
                    valids_out = cache["valids"]
                if "visibs" in cache:
                    visibs_out = cache["visibs"]
                raw_semantics_version = cache.get("track_semantics_version", 0)
                precomputed_track_semantics_version = int(
                    np.asarray(raw_semantics_version).item()
                )
                active_track_semantics_version = precomputed_track_semantics_version
                has_tracks_out = (
                    trajs_2d_out is not None
                    or trajs_3d_out is not None
                    or "static_points_world" in cache
                )
                if "origin_shift" in cache:
                    origin_shift = np.asarray(
                        cache["origin_shift"], dtype=np.float32
                    ).reshape(-1)
                    if origin_shift.size != 3:
                        raise RuntimeError(
                            f"[{sequence_name}] invalid origin_shift shape in "
                            f"precomputed cache: {origin_shift.shape}"
                        )
                    precomputed_origin_shift = origin_shift

        # Shift world origin to a camera center so that the scene is near the
        # origin and rerun can auto-fit the 3-D view.
        # Depth images live in camera space and are unaffected by this shift.
        # For a w2c matrix E = [R | t]:  cam_center = E_inv[:3,3] = -R^T t
        # After shifting by cam0:  new_t = t + R @ cam0  (reprojection unchanged)
        #
        # Cached tracks are computed over the full sequence, whose first frame
        # is frame 0.  When loading a sub-clip, use the same full-sequence shift
        # instead of the sub-clip's first camera center.
        if precomputed_origin_shift is not None:
            cam0 = precomputed_origin_shift
        elif has_precomputed_cache and frame_indices[0] != 0:
            pose0 = self._read_json_retry(r.pose_path(0))
            E0 = np.array(pose0["extrinsic"], dtype=np.float32)
            E0[:3, 3] /= 100.0
            E0[:3, :3] = E0[:3, :3] @ P.T
            cam0 = np.linalg.inv(E0.astype(np.float64))[:3, 3].astype(np.float32)
        else:
            E0_inv = np.linalg.inv(extrinsics[0].astype(np.float64))
            cam0 = E0_inv[:3, 3].astype(np.float32)
        for i in range(len(extrinsics)):
            extrinsics[i, :3, 3] += extrinsics[i, :3, :3] @ cam0

        # Old MVS-Synth caches predate the current track semantics: they often
        # have visibs == valids and retain very far projected points.  Refresh
        # from depth so valids mean "label defined", visibs mean "visible", and
        # the dataset-level p99.5-ish 100 m depth cap masks the extreme tail.
        if (
            trajs_3d_out is not None
            and precomputed_track_semantics_version is not None
            and precomputed_track_semantics_version < TRACK_SEMANTICS_VERSION
        ):
            refreshed = recompute_track_projection_masks(
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                trajs_3d_world=trajs_3d_out,
                depth_max=(
                    min(MVSSYNTH_TRACK_DEPTH_MAX_M, self.visibility_depth_max)
                    if self.visibility_depth_max is not None
                    else MVSSYNTH_TRACK_DEPTH_MAX_M
                ),
                depth_support_radius=self.visibility_depth_support_radius,
                visible_abs_thresh=self.visibility_visible_abs_thresh,
                occlusion_abs_thresh=self.visibility_occlusion_abs_thresh,
                occlusion_rel_thresh=self.visibility_occlusion_rel_thresh,
                occlusion_min_support=self.visibility_occlusion_min_support,
                occlusion_far_depth_start=self.visibility_occlusion_far_depth_start,
                occlusion_far_depth_end=self.visibility_occlusion_far_depth_end,
                occlusion_far_abs_thresh=self.visibility_occlusion_far_abs_thresh,
                occlusion_far_rel_thresh=self.visibility_occlusion_far_rel_thresh,
                occlusion_far_min_support=self.visibility_occlusion_far_min_support,
                far_uncertain_as_visible_depth=self.visibility_far_uncertain_as_visible_depth,
            )
            trajs_2d_out = refreshed["trajs_2d"]
            valids_out = refreshed["valids"]
            visibs_out = refreshed["visibs"]
            active_track_semantics_version = TRACK_SEMANTICS_VERSION
            precomputed_track_semantics_refreshed = True

        if (
            trajs_2d_out is None
            and has_precomputed_cache
            and cache is not None
            and "static_points_world" in cache
        ):
            static_points = np.asarray(cache["static_points_world"], dtype=np.float32)
            conv = int(
                np.asarray(cache.get("static_points_world_convention", 1)).item()
            )
            if conv != 1:
                raise RuntimeError(
                    f"[{sequence_name}] unsupported static_points_world_convention={conv}"
                )
            depth_consistency_thresh = float(
                np.asarray(cache.get("track_depth_consistency_thresh", 0.05)).item()
            )
            depth_max = float(
                np.asarray(cache.get("track_depth_max", MVSSYNTH_TRACK_DEPTH_MAX_M)).item()
            )
            if self.visibility_depth_max is not None:
                depth_max = min(depth_max, self.visibility_depth_max)
            trajs_2d_out, valids_out, visibs_out = _project_world_points_to_frames(
                points_world=static_points,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                depth_consistency_thresh=depth_consistency_thresh,
                depth_max=depth_max,
                depth_support_radius=self.visibility_depth_support_radius,
                visible_abs_thresh=self.visibility_visible_abs_thresh,
                occlusion_abs_thresh=self.visibility_occlusion_abs_thresh,
                occlusion_rel_thresh=self.visibility_occlusion_rel_thresh,
                occlusion_min_support=self.visibility_occlusion_min_support,
                occlusion_far_depth_start=self.visibility_occlusion_far_depth_start,
                occlusion_far_depth_end=self.visibility_occlusion_far_depth_end,
                occlusion_far_abs_thresh=self.visibility_occlusion_far_abs_thresh,
                occlusion_far_rel_thresh=self.visibility_occlusion_far_rel_thresh,
                occlusion_far_min_support=self.visibility_occlusion_far_min_support,
                far_uncertain_as_visible_depth=self.visibility_far_uncertain_as_visible_depth,
            )
            trajs_3d_out = np.broadcast_to(
                static_points[None],
                (len(frame_indices), static_points.shape[0], 3),
            ).copy()
            active_track_semantics_version = TRACK_SEMANTICS_VERSION
            has_tracks_out = True

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths,
            normals=normals_out,
            trajs_2d=trajs_2d_out,
            trajs_3d_world=trajs_3d_out,
            valids=valids_out,
            visibs=visibs_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata={
                "dataset_name": self.dataset_name,
                "sequence_root": str(r.sequence_dir),
                "frame_indices": list(frame_indices),
                "num_frames_in_sequence": r.num_frames,
                "raw_image_hw": [images[0].shape[0], images[0].shape[1]] if images else None,
                "has_depth": True,
                "has_normals": has_normals_out,
                "has_tracks": has_tracks_out,
                "has_visibility": has_tracks_out,
                "has_trajs_3d_world": has_tracks_out,
                "pose_convention": "w2c",
                "extrinsics_convention": "w2c",
                "intrinsics_convention": "pinhole",
                "precomputed_track_semantics_version": precomputed_track_semantics_version,
                "track_semantics_version": active_track_semantics_version,
                "precomputed_track_semantics_refreshed": precomputed_track_semantics_refreshed,
                "track_depth_max_m": MVSSYNTH_TRACK_DEPTH_MAX_M,
                "visibility_depth_max_m": self.visibility_depth_max,
                "visibility_occlusion_far_depth_start_m": self.visibility_occlusion_far_depth_start,
                "visibility_occlusion_far_depth_end_m": self.visibility_occlusion_far_depth_end,
                "visibility_occlusion_far_abs_thresh_m": self.visibility_occlusion_far_abs_thresh,
                "visibility_occlusion_far_rel_thresh": self.visibility_occlusion_far_rel_thresh,
                "visibility_occlusion_far_min_support": self.visibility_occlusion_far_min_support,
                "visibility_far_uncertain_as_visible_depth_m": self.visibility_far_uncertain_as_visible_depth,
                "depth_unit": "meters",
                "depth_note": (
                    "OpenEXR float32; inf values converted to 0 (invalid/sky)"
                ),
                # Camera center used for the origin shift, in pre-shift world
                # coordinates.  Saved into precomputed caches so sub-clips can
                # keep cached trajs_3d_world and extrinsics in the same frame.
                "origin_shift": cam0.tolist(),
            },
        )

    def _load_precomputed(self, sequence_name: str, frame_indices: list[int]) -> Optional[dict]:
        """Load precomputed data for frame_indices. Prefers .h5 over .npz."""
        from datasets.adapters.base import load_precomputed_fast
        path = self.precompute_root / sequence_name / "precomputed.npz"
        h5_path = path.with_suffix('.h5')
        has_npz = self._path_exists_retry(path)
        has_h5 = self._path_exists_retry(h5_path)
        if not has_npz and not has_h5:
            return None
        try:
            if has_h5:
                import h5py

                def _read_h5_num_frames() -> int:
                    with h5py.File(h5_path, 'r') as f:
                        if 'trajs_2d' in f:
                            return int(f['trajs_2d'].shape[0])
                        if 'num_frames' in f:
                            return int(np.asarray(f['num_frames'][()]).item())
                        return 0

                n = self._retry_call(f"read precomputed h5 metadata {h5_path}", _read_h5_num_frames)
            else:
                def _read_npz_num_frames() -> int:
                    with np.load(path, allow_pickle=False) as raw:
                        return int(raw["num_frames"])

                n = self._retry_call(f"read precomputed npz metadata {path}", _read_npz_num_frames)
        except Exception as exc:
            if self.verbose:
                print(f"[MVSSynthAdapter] Warning: failed to inspect cache for {sequence_name}: {exc}")
            return None
        if n < max(frame_indices) + 1:
            return None
        skip_keys = set() if self.load_normals else {"normals"}
        attempts = max(1, self.io_read_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return load_precomputed_fast(path, frame_indices, skip_keys=skip_keys)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    _sleep_before_retry(self.io_retry_delay_s, attempt)
        if self.verbose:
            print(f"[MVSSynthAdapter] Warning: failed to load cache for {sequence_name}: {last_error}")
        return None

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        """Run consistency checks on a sequence.

        Checks
        ------
        1. Image, depth, and pose files exist for first 5 frames.
        2. Intrinsics shape (T, 3, 3), finite, positive focal lengths.
        3. Extrinsics shape (T, 4, 4), finite, rotation near-orthonormal,
           last row == [0, 0, 0, 1].
        4. Depth values finite after inf→0 conversion; at least some
           positive values per frame.
        5. Round-trip reprojection: unproject center pixel, re-project,
           check error < 1 px.

        Returns
        -------
        dict with ``dataset_name``, ``sequence_name``, ``ok`` (bool),
        ``messages`` (list[str]).
        """
        r = self._get_record(sequence_name)
        msgs: list[str] = []
        ok = True

        # ── 1. File existence (first 5 frames) ──────────────────────────
        probe_count = min(5, r.num_frames)
        for i in range(probe_count):
            for key, path in [
                ("image", r.image_path(i)),
                ("depth", r.depth_path(i)),
                ("pose", r.pose_path(i)),
            ]:
                if not self._path_exists_retry(path):
                    ok = False
                    msgs.append(f"frame {i}: missing {key} file: {path}")

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
            # Last row == [0,0,0,1]
            bottom = clip.extrinsics[:, 3, :]
            if not np.allclose(bottom, [0, 0, 0, 1], atol=1e-5):
                ok = False
                msgs.append("extrinsics last row != [0,0,0,1]")
            # Rotation orthonormality
            R = clip.extrinsics[:, :3, :3].astype(np.float64)
            RRt = R @ np.transpose(R, (0, 2, 1))
            err = float(np.linalg.norm(RRt - np.eye(3)[None], axis=(1, 2)).max())
            if err > 1e-4:
                ok = False
                msgs.append(f"rotation matrices non-orthonormal: max ||RR^T-I||_F={err:.2e}")
            # After the P basis-change, det(R) ≈ +1 (proper rotation)
            dets = np.linalg.det(R)
            if not np.all(np.abs(np.abs(dets) - 1.0) < 1e-4):
                ok = False
                msgs.append(f"rotation det not ±1: range [{dets.min():.6f},{dets.max():.6f}]")

        # ── 4. Depth ────────────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            if not np.isfinite(dep).all():
                ok = False
                msgs.append(f"depth[{t}] contains non-finite values")
            if (dep > 0).sum() == 0:
                ok = False
                msgs.append(f"depth[{t}] has no positive values")

        # ── 5. Reprojection ─────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            K = clip.intrinsics[t].astype(np.float64)
            E = clip.extrinsics[t].astype(np.float64)  # w2c

            ys, xs = np.where(dep > 0)
            if ys.size == 0:
                msgs.append(f"reproj check skipped frame {t}: no valid depth")
                continue

            # Sample center-ish pixel
            mid = ys.size // 2
            u, v = int(xs[mid]), int(ys[mid])
            d = float(dep[v, u])

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            xc = (u - cx) / fx * d
            yc = (v - cy) / fy * d
            p_cam = np.array([xc, yc, d, 1.0])

            try:
                E_inv = np.linalg.inv(E)
            except np.linalg.LinAlgError:
                ok = False
                msgs.append(f"extrinsics[{t}] singular")
                continue

            p_world = E_inv @ p_cam
            p_cam2 = E @ p_world
            p_2d = K @ p_cam2[:3]
            u2 = p_2d[0] / p_2d[2]
            v2 = p_2d[1] / p_2d[2]
            reproj_err = float(np.hypot(u2 - u, v2 - v))
            if reproj_err > 1.0:
                ok = False
                msgs.append(
                    f"reproj error frame {t}: ({u2:.2f},{v2:.2f}) vs ({u},{v}), "
                    f"err={reproj_err:.4f}px"
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
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self) -> list[_SequenceRecord]:
        # Check for num_images.json
        num_images_path = self.root / "num_images.json"
        if self._path_exists_retry(num_images_path):
            num_images_list = self._read_json_retry(num_images_path)
        else:
            num_images_list = None

        records: list[_SequenceRecord] = []
        skipped: list[str] = []

        if num_images_list is not None:
            # COS directory stat/list calls are expensive.  The official
            # num_images.json already defines the compact 0000..0119 index, so
            # build records from it and let load_clip report any missing files.
            for seq_idx, num_frames in enumerate(num_images_list):
                seq_id = f"{seq_idx:04d}"
                if int(num_frames) <= 0:
                    if self.strict:
                        raise RuntimeError(f"No frames found in {self.root / seq_id}")
                    skipped.append(f"{seq_id}: no frames")
                    continue
                records.append(
                    _SequenceRecord(
                        sequence_id=seq_id,
                        sequence_dir=self.root / seq_id,
                        num_frames=int(num_frames),
                    )
                )
            if self.verbose and skipped:
                print(
                    f"[MVSSynthAdapter] skipped {len(skipped)} sequences "
                    "(non-strict mode)"
                )
            return records

        # Enumerate sequence directories (0000, 0001, ..., 0119)
        for seq_dir in self._iterdir_retry(self.root):
            if not self._path_is_dir_retry(seq_dir):
                continue
            if not seq_dir.name.isdigit():
                continue

            try:
                rec = self._index_sequence(seq_dir, num_images_list)
                if rec is not None:
                    records.append(rec)
            except Exception as exc:
                if self.strict:
                    raise
                skipped.append(f"{seq_dir.name}: {exc}")
                if self.verbose:
                    print(f"[MVSSynthAdapter][WARN] skip {seq_dir.name}: {exc}")

        if self.verbose and skipped:
            print(
                f"[MVSSynthAdapter] skipped {len(skipped)} sequences "
                "(non-strict mode)"
            )

        return records

    def _index_sequence(
        self, seq_dir: Path, num_images_list: Optional[list[int]]
    ) -> Optional[_SequenceRecord]:
        seq_id = seq_dir.name

        # Check required subdirectories
        for subdir in ["images", "depths", "poses"]:
            if not self._path_is_dir_retry(seq_dir / subdir):
                raise FileNotFoundError(f"Missing {subdir}/ in {seq_dir}")

        # Determine num_frames
        if num_images_list is not None:
            seq_idx = int(seq_id)
            if seq_idx < len(num_images_list):
                num_frames = num_images_list[seq_idx]
            else:
                raise ValueError(
                    f"Sequence {seq_id} index out of range in num_images.json"
                )
        else:
            # Count pose files
            pose_files = self._glob_retry(seq_dir / "poses", "*.json")
            num_frames = len(pose_files)

        if num_frames == 0:
            raise RuntimeError(f"No frames found in {seq_dir}")

        return _SequenceRecord(
            sequence_id=seq_id,
            sequence_dir=seq_dir,
            num_frames=num_frames,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_record(self, sequence_name: str) -> _SequenceRecord:
        if sequence_name not in self._name_to_record:
            raise KeyError(
                f"Unknown sequence_name: '{sequence_name}'. "
                f"Use list_sequences() to enumerate valid names."
            )
        return self._name_to_record[sequence_name]

    def _check_indices(
        self, frame_indices: list[int], num_frames: int, sequence_name: str
    ) -> None:
        if not frame_indices:
            raise ValueError("frame_indices is empty")
        lo, hi = min(frame_indices), max(frame_indices)
        if lo < 0 or hi >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={lo}, max={hi}, num_frames={num_frames}"
            )
