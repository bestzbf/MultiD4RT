from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from multicamera4rc.datasets.computer.depth_to_tracks import (
    TRACK_SEMANTICS_VERSION,
    recompute_track_projection_masks,
)

from .base import BaseAdapter, UnifiedClip


# ---------------------------------------------------------------------------
# PFM I/O
# ---------------------------------------------------------------------------

def _read_pfm(path: Path) -> np.ndarray:
    """Read a PFM (Portable Float Map) file and return a float32 2-D array.

    PFM layout (text header + binary payload):
      Line 1 : "PF" (3-channel) or "Pf" (single-channel)
      Line 2 : "<width> <height>"
      Line 3 : "<scale>"   negative  → little-endian, positive → big-endian
      Payload : height × width × channels float32 values, stored bottom-up.

    BlendedMVS depth maps are single-channel Pf files.
    """
    with open(path, "rb") as f:
        magic = f.readline().decode("latin-1").strip()
        if magic not in ("PF", "Pf"):
            raise ValueError(f"Not a valid PFM file (magic='{magic}'): {path}")

        dims = f.readline().decode("latin-1").strip()
        width, height = map(int, dims.split())

        scale_raw = float(f.readline().decode("latin-1").strip())
        little_endian = scale_raw < 0
        scale = abs(scale_raw)

        channels = 3 if magic == "PF" else 1
        dtype = np.dtype("<f4" if little_endian else ">f4")
        data = np.frombuffer(f.read(), dtype=dtype)

    expected = height * width * channels
    if data.size != expected:
        raise ValueError(
            f"PFM payload size mismatch (got {data.size}, expected {expected}): {path}"
        )

    data = data.reshape((height, width, channels) if channels > 1 else (height, width))
    data = np.flipud(data)  # PFM stored bottom-up

    if scale != 1.0:
        data = data * scale

    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Camera-file parser
# ---------------------------------------------------------------------------

def _parse_cam_file(path: Path) -> dict[str, Any]:
    """Parse a BlendedMVS / MVSNet-style camera parameter file.

    File format::

        extrinsic
        r00 r01 r02 t0
        r10 r11 r12 t1
        r20 r21 r22 t2
        0.0 0.0 0.0 1.0

        intrinsic
        fx  0   cx
        0   fy  cy
        0   0   1.0

        depth_min depth_interval depth_num depth_max

    The ``extrinsic`` matrix is the **world-to-camera** (w2c) transform, i.e.
    ``p_cam = E @ p_world_homogeneous``.

    Returns
    -------
    dict with keys:
        ``extrinsic``       : (4, 4) float32 world-to-camera matrix
        ``intrinsic``       : (3, 3) float32 camera intrinsic matrix
        ``depth_min``       : float
        ``depth_interval``  : float
        ``depth_num``       : float  (number of depth hypotheses / max range)
        ``depth_max``       : float
    """
    with open(path, "r") as f:
        # Filter out blank lines so indices are deterministic.
        lines = [ln.strip() for ln in f if ln.strip()]

    # lines[0] == "extrinsic"
    # lines[1..4] == 4 rows of the 4×4 matrix
    extrinsic = np.array(
        [[float(v) for v in lines[i].split()] for i in range(1, 5)],
        dtype=np.float32,
    )

    # lines[5] == "intrinsic"
    # lines[6..8] == 3 rows of the 3×3 matrix
    intrinsic = np.array(
        [[float(v) for v in lines[i].split()] for i in range(6, 9)],
        dtype=np.float32,
    )

    # lines[9] == depth params
    depth_params = [float(v) for v in lines[9].split()]

    return {
        "extrinsic": extrinsic,          # (4, 4)  w2c
        "intrinsic": intrinsic,          # (3, 3)
        "depth_min": depth_params[0],
        "depth_interval": depth_params[1],
        "depth_num": depth_params[2],
        "depth_max": depth_params[3],
    }


# ---------------------------------------------------------------------------
# Sequence record
# ---------------------------------------------------------------------------

class _SequenceRecord:
    """Index entry for a single BlendedMVS scene."""

    __slots__ = (
        "scene_id",
        "scene_dir",
        "frame_ids",     # list[str]  e.g. ["00000000", "00000001", ...]
        "num_frames",
    )

    def __init__(
        self,
        scene_id: str,
        scene_dir: Path,
        frame_ids: list[str],
    ) -> None:
        self.scene_id = scene_id
        self.scene_dir = scene_dir
        self.frame_ids = frame_ids
        self.num_frames = len(frame_ids)

    def rgb_path(self, frame_id: str, masked: bool = False) -> Path:
        suffix = "_masked.jpg" if masked else ".jpg"
        return self.scene_dir / "blended_images" / f"{frame_id}{suffix}"

    def depth_path(self, frame_id: str) -> Path:
        return self.scene_dir / "rendered_depth_maps" / f"{frame_id}.pfm"

    def cam_path(self, frame_id: str) -> Path:
        return self.scene_dir / "cams" / f"{frame_id}_cam.txt"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

_CAM_FILENAME_RE = re.compile(r"^(\d+)_cam\.txt$")


class BlendedMVSAdapter(BaseAdapter):
    """Dataset adapter for BlendedMVS.

    Expected dataset layout::

        <root>/
            BlendedMVS_training.txt     # one scene ID per line (training split)
            validation_list.txt         # one scene ID per line (val split)
            <scene_id>/
                blended_images/
                    00000000.jpg
                    00000000_masked.jpg
                    00000001.jpg
                    ...
                cams/
                    00000000_cam.txt
                    00000001_cam.txt
                    ...
                rendered_depth_maps/
                    00000000.pfm
                    00000001.pfm
                    ...

    Frames are indexed by the camera files present in ``cams/``.
    Each frame maps to one RGB image, one depth map, and one camera file.

    Supervision availability
    ------------------------
    - ``has_depth``      : True   (rendered depth maps in PFM format)
    - ``has_normals``    : False  (not provided by BlendedMVS)
    - ``has_tracks``     : False  (no point-track annotations)
    - ``has_visibility`` : False  (no per-point visibility annotation)

    Extrinsics convention
    ---------------------
    BlendedMVS camera files store **world-to-camera** (w2c) transforms:
    ``p_cam = E @ p_world_homogeneous``.
    This matches the ``"w2c"`` convention used by the rest of the D4RT adapter
    family (see ScanNetAdapter).
    """

    dataset_name: str = "blendedmvs"

    # split-name → list file inside the dataset root
    _SPLIT_LIST_FILES: dict[str, str] = {
        "train": "BlendedMVS_training.txt",
        "val": "validation_list.txt",
        "valid": "validation_list.txt",
        "validation": "validation_list.txt",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        use_masked: bool = False,
        strict: bool = True,
        verbose: bool = True,
        precompute_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
        index_workers: int = 8,
        io_workers: int = 1,
        depth_cache_dir: Optional[str] = None,
        load_precomputed: bool = True,
        load_normals: bool = True,
        refresh_track_projection_masks: bool = False,
        enable_covisibility_sampling: bool = False,
        covisibility_max_baseline_m: float = 0.5,
        covisibility_max_angle_deg: float = 30.0,
        covisibility_min_neighbors: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        root :
            Root directory of the BlendedMVS dataset.
        split :
            One of ``"train"``, ``"val"`` / ``"valid"`` / ``"validation"``.
        use_masked :
            When *True*, load ``XXXXXXXX_masked.jpg`` instead of
            ``XXXXXXXX.jpg`` as the RGB input.
        strict :
            When *True*, raise on any scene that cannot be indexed.
            When *False*, skip broken scenes with a warning.
        verbose :
            Print summary information during construction.
        enable_covisibility_sampling :
            When *True*, pre-compute camera geometry neighbor graphs for
            covisibility-aware sampling. Each frame gets a list of neighbors
            within geometric constraints (baseline + viewing angle).
        covisibility_max_baseline_m :
            Maximum camera center distance (meters) for two views to be
            considered co-visible neighbors.
        covisibility_max_angle_deg :
            Maximum viewing direction angle (degrees) for two views to be
            considered co-visible neighbors.
        covisibility_min_neighbors :
            Minimum number of neighbors per frame. If geometric constraints
            yield fewer, expands to the K nearest cameras by distance.
        """
        self.root = Path(root)
        # Check cache first to skip slow root.exists() on remote storage
        _cache_hit = False
        if cache_dir is not None:
            from multicamera4rc.datasets.index_cache import load_or_build
            cache_precompute_root = Path(precompute_root) if precompute_root else self.root
            cache_key = {
                "dataset": "blendedmvs",
                "split": split.lower(),
                "root": str(self.root.resolve()),
                "precompute_root": str(cache_precompute_root.resolve()),
                "use_masked": use_masked,
                "strict": strict,
                "cache_schema": 2,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            _cache_path = Path(cache_dir) / f"blendedmvs_{split.lower()}_{cache_suffix}.pkl"
            if _cache_path.exists():
                _cache_hit = True
        if not _cache_hit and not self.root.exists():
            raise FileNotFoundError(f"BlendedMVS root not found: {self.root}")

        split_key = split.lower()
        if split_key not in self._SPLIT_LIST_FILES:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Valid splits: {sorted(self._SPLIT_LIST_FILES.keys())}"
            )
        self.split = split_key
        self.use_masked = use_masked
        self.strict = strict
        self.verbose = verbose
        self.index_workers = index_workers
        self.io_workers = max(1, int(io_workers))
        self.load_precomputed = bool(load_precomputed)
        self.refresh_track_projection_masks = bool(refresh_track_projection_masks)
        if isinstance(load_normals, str):
            self.load_normals = load_normals.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.load_normals = bool(load_normals)
        self._precomputed_info_cache: dict[str, dict[str, Any]] = {}
        resolved_depth_cache_dir = depth_cache_dir or os.getenv(
            "D4RT_BLENDEDMVS_DEPTH_CACHE_DIR", ""
        )
        self.depth_cache_dir = (
            Path(resolved_depth_cache_dir) if resolved_depth_cache_dir else None
        )
        self.precompute_root = (
            Path(precompute_root) if precompute_root else self.root
        ) if self.load_precomputed else None

        # Covisibility sampling parameters
        self.enable_covisibility_sampling = bool(enable_covisibility_sampling)
        self.covisibility_max_baseline_m = float(covisibility_max_baseline_m)
        self.covisibility_max_angle_deg = float(covisibility_max_angle_deg)
        self.covisibility_min_neighbors = int(covisibility_min_neighbors)
        self._covisibility_graphs: dict[str, list[list[int]]] = {}
        self._covisibility_cache_dir = Path(cache_dir) if cache_dir else None

        if cache_dir is not None:
            from multicamera4rc.datasets.index_cache import load_or_build
            cache_precompute_root = Path(precompute_root) if precompute_root else self.root
            cache_key = {
                "dataset": self.dataset_name,
                "split": self.split,
                "root": str(self.root.resolve()),
                "precompute_root": str(cache_precompute_root.resolve()),
                "use_masked": self.use_masked,
                "strict": self.strict,
                "cache_schema": 2,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            _cache_path = Path(cache_dir) / f"{self.dataset_name}_{self.split}_{cache_suffix}.pkl"
            self._records: list[_SequenceRecord] = load_or_build(self._build_index, _cache_path)
        else:
            self._records: list[_SequenceRecord] = self._build_index()
        self._name_to_record: dict[str, _SequenceRecord] = {
            r.scene_id: r for r in self._records
        }

        if len(self._records) == 0:
            raise RuntimeError(
                f"No valid BlendedMVS scenes found for split='{split}' under {self.root}"
            )

        if self.verbose:
            print(
                f"[BlendedMVSAdapter] split={self.split!r}, "
                f"num_sequences={len(self._records)}, "
                f"use_masked={self.use_masked}"
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [r.scene_id for r in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].scene_id

    def get_num_frames(self, sequence_name: str) -> int:
        """Fast path: read from cached record, no image reading."""
        return self._get_record(sequence_name).num_frames

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
            if h5_path.exists():
                backend = "h5"
                try:
                    import h5py

                    with h5py.File(h5_path, "r") as f:
                        required = {"trajs_2d", "trajs_3d_world", "valids", "visibs"}
                        has_tracks = required.issubset(f.keys())
                        has_normals = "normals" in f
                except Exception:
                    has_tracks = False
                    has_normals = False
            elif npz_path.exists():
                backend = "npz"
                try:
                    with np.load(npz_path, mmap_mode="r", allow_pickle=False) as data:
                        required = {"trajs_2d", "trajs_3d_world", "valids", "visibs"}
                        has_tracks = required.issubset(data.files)
                        has_normals = "normals" in data.files
                except Exception:
                    has_tracks = False
                    has_normals = False

        info = {
            "has_tracks": bool(has_tracks),
            "has_normals": bool(has_normals and self.load_normals),
            "has_visibility": bool(has_tracks),
            "backend": backend if has_tracks else None,
        }
        self._precomputed_info_cache[sequence_name] = info
        return info

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self._get_record(sequence_name)
        # Probe image size from first frame.
        h, w = self._read_image_size(r.rgb_path(r.frame_ids[0], self.use_masked))
        precomputed = self._get_precomputed_info(sequence_name)
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": r.scene_id,
            "sequence_root": str(r.scene_dir),
            "num_frames": r.num_frames,
            "height": h,
            "width": w,
            "has_depth": True,
            "has_normals": precomputed["has_normals"],
            "has_tracks": precomputed["has_tracks"],
            "has_visibility": precomputed["has_visibility"],
            "has_trajs_3d_world": precomputed["has_tracks"],
            "precomputed_backend": precomputed["backend"],
            "extrinsics_convention": "w2c",
            "depth_unit": "meters",
        }

    def get_covisibility_neighbors(self, sequence_name: str, frame_index: int) -> Optional[list[int]]:
        """Return neighbor frame indices for covisibility-aware sampling.

        Returns None if covisibility sampling is disabled, otherwise returns
        a list of frame indices that are geometrically co-visible with the
        given frame (within baseline and angle constraints).
        """
        if not self.enable_covisibility_sampling:
            return None

        if sequence_name not in self._covisibility_graphs:
            self._build_covisibility_graph(sequence_name)

        graph = self._covisibility_graphs.get(sequence_name)
        if graph is None or frame_index >= len(graph):
            return None
        return graph[frame_index]

    def _build_covisibility_graph(self, sequence_name: str) -> None:
        """Build camera geometry neighbor graph for one scene."""
        # Try loading from disk cache first
        if self._covisibility_cache_dir is not None:
            cache_key = {
                "sequence": sequence_name,
                "max_baseline_m": self.covisibility_max_baseline_m,
                "max_angle_deg": self.covisibility_max_angle_deg,
                "min_neighbors": self.covisibility_min_neighbors,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            cache_path = self._covisibility_cache_dir / f"covisibility_{sequence_name}_{cache_suffix}.pkl"

            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        self._covisibility_graphs[sequence_name] = pickle.load(f)
                    return
                except Exception:
                    pass  # Rebuild if cache is corrupted

        r = self._get_record(sequence_name)
        n = r.num_frames

        # Load all camera extrinsics (w2c)
        extrinsics = []
        for frame_id in r.frame_ids:
            cam_data = _parse_cam_file(r.cam_path(frame_id))
            extrinsics.append(cam_data["extrinsic"])  # (4, 4) w2c

        # Extract camera centers in world space: c_world = -R^T @ t
        centers = []
        view_dirs = []
        for E in extrinsics:
            R = E[:3, :3]
            t = E[:3, 3]
            c = -R.T @ t  # camera center in world
            view_dir = R[2, :]  # third row of R = camera's +Z axis in world (viewing direction)
            centers.append(c)
            view_dirs.append(view_dir)

        centers = np.array(centers, dtype=np.float32)  # (n, 3)
        view_dirs = np.array(view_dirs, dtype=np.float32)  # (n, 3)

        # Compute pairwise baseline distances
        dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)  # (n, n)

        # Compute pairwise viewing angles (cosine similarity of view directions)
        view_dirs_norm = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8)
        cos_angles = view_dirs_norm @ view_dirs_norm.T  # (n, n)
        angles_deg = np.rad2deg(np.arccos(np.clip(cos_angles, -1.0, 1.0)))

        # Build neighbor lists
        graph: list[list[int]] = []
        max_baseline = self.covisibility_max_baseline_m
        max_angle = self.covisibility_max_angle_deg
        min_neighbors = self.covisibility_min_neighbors

        for i in range(n):
            # Find neighbors within geometric constraints
            mask = (dists[i] <= max_baseline) & (angles_deg[i] <= max_angle) & (np.arange(n) != i)
            neighbors = np.where(mask)[0].tolist()

            # Ensure minimum neighbor count by expanding to K nearest
            if len(neighbors) < min_neighbors:
                # Sort by distance, exclude self
                sorted_indices = np.argsort(dists[i])
                neighbors = [idx for idx in sorted_indices if idx != i][:min_neighbors]

            graph.append(neighbors)

        self._covisibility_graphs[sequence_name] = graph

        # Save to disk cache
        if self._covisibility_cache_dir is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(graph, f)
            except Exception:
                pass  # Non-fatal if cache write fails

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        """Load a clip from a BlendedMVS scene.

        Parameters
        ----------
        sequence_name :
            Scene ID (24-character hex string).
        frame_indices :
            List of integer frame indices into the sorted camera file list for
            this scene.  Must be within ``[0, num_frames)``.

        Returns
        -------
        UnifiedClip
            Unified intermediate representation.
            ``normals``, ``trajs_2d``, ``trajs_3d_world``, ``valids``, and
            ``visibs`` are all ``None`` (not available in BlendedMVS).
        """
        import time as _time

        timing: dict[str, float] = {}
        t_total_start = _time.perf_counter()
        t0 = _time.perf_counter()
        r = self._get_record(sequence_name)
        timing["scene_data_s"] = _time.perf_counter() - t0
        self._check_indices(frame_indices, r.num_frames, sequence_name)

        def _load_one(idx: int):
            fid = r.frame_ids[idx]

            rgb_p = r.rgb_path(fid, self.use_masked)
            dep_p = r.depth_path(fid)
            cam_p = r.cam_path(fid)

            t_rgb = _time.perf_counter()
            image = self._read_image(rgb_p)
            rgb_s = _time.perf_counter() - t_rgb
            t_depth = _time.perf_counter()
            depth = self._read_depth(dep_p, r.scene_id, fid)
            depth_s = _time.perf_counter() - t_depth
            t_cam = _time.perf_counter()
            cam = _parse_cam_file(cam_p)
            cam_s = _time.perf_counter() - t_cam
            return (
                image,
                depth,
                cam["intrinsic"],    # (3, 3) float32
                cam["extrinsic"],    # (4, 4) float32 w2c
                str(rgb_p),
                (cam["depth_min"], cam["depth_max"]),
                rgb_s,
                depth_s,
                cam_s,
            )

        t0 = _time.perf_counter()
        if self.io_workers > 1 and len(frame_indices) > 1:
            with ThreadPoolExecutor(max_workers=min(len(frame_indices), self.io_workers)) as ex:
                rows = list(ex.map(_load_one, frame_indices))
        else:
            rows = [_load_one(idx) for idx in frame_indices]
        timing["frame_load_s"] = _time.perf_counter() - t0

        (
            images,
            depths,
            intrinsics_list,
            extrinsics_list,
            frame_paths,
            depth_ranges,
            rgb_times,
            depth_times,
            cam_times,
        ) = (
            list(items) for items in zip(*rows)
        )
        timing["rgb_load_s"] = float(sum(rgb_times))
        timing["depth_load_s"] = float(sum(depth_times))
        timing["cam_load_s"] = float(sum(cam_times))

        t0 = _time.perf_counter()
        intrinsics = np.stack(intrinsics_list, axis=0)   # (T, 3, 3) float32
        extrinsics = np.stack(extrinsics_list, axis=0)   # (T, 4, 4) float32 w2c
        timing["process_s"] = _time.perf_counter() - t0

        # Load precomputed normals / tracks if available.
        # Old BlendedMVS caches used visibs == valids, which destroys visibility
        # supervision. Refresh those semantics online from cached world points.
        normals_out, trajs_2d_out, trajs_3d_out, valids_out, visibs_out = \
            None, None, None, None, None
        has_normals_out, has_tracks_out = False, False
        precomputed_track_semantics_version: Optional[int] = None
        active_track_semantics_version: Optional[int] = None
        precomputed_track_semantics_refreshed = False
        if self.precompute_root is not None:
            t0 = _time.perf_counter()
            cache = self._load_precomputed(sequence_name, frame_indices)
            timing["precomputed_s"] = _time.perf_counter() - t0
            if cache is not None:
                if self.load_normals and "normals" in cache:
                    normals_out = [n.astype(np.float32) for n in cache["normals"]]
                trajs_2d_out    = cache["trajs_2d"]
                trajs_3d_out    = cache["trajs_3d_world"]
                valids_out      = cache["valids"]
                visibs_out      = cache["visibs"]
                raw_semantics_version = cache.get("track_semantics_version", 0)
                precomputed_track_semantics_version = int(np.asarray(raw_semantics_version).item())
                active_track_semantics_version = precomputed_track_semantics_version

                if (
                    trajs_3d_out is not None
                    and (
                        self.refresh_track_projection_masks
                        or precomputed_track_semantics_version < TRACK_SEMANTICS_VERSION
                    )
                ):
                    refreshed = recompute_track_projection_masks(
                        depths=depths,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        trajs_3d_world=trajs_3d_out,
                        depth_consistency_thresh=0.03,
                        # Visibility is defined at the projected pixel.  A
                        # neighbouring surface must not make an occluded point
                        # visible, so use the nearest pixel rather than an
                        # any-match 3x3 support window.
                        depth_support_radius=0,
                        visible_abs_thresh=0.01,
                        occlusion_abs_thresh=0.01,
                        occlusion_rel_thresh=0.03,
                        occlusion_min_support=1,
                        occlusion_overrides_visible=True,
                    )
                    trajs_2d_out = refreshed["trajs_2d"]
                    valids_out = refreshed["valids"]
                    visibs_out = refreshed["visibs"]
                    active_track_semantics_version = TRACK_SEMANTICS_VERSION
                    precomputed_track_semantics_refreshed = True

                has_normals_out = normals_out is not None
                has_tracks_out  = True
                timing["process_s"] += _time.perf_counter() - t0
        else:
            timing["precomputed_s"] = 0.0

        timing["total_s"] = _time.perf_counter() - t_total_start

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
                "split": self.split,
                "sequence_root": str(r.scene_dir),
                "frame_indices": list(frame_indices),
                "frame_ids": [r.frame_ids[i] for i in frame_indices],
                "num_frames_in_sequence": r.num_frames,
                "raw_image_hw": list(images[0].shape[:2]) if images else None,
                "has_depth": True,
                "has_normals": has_normals_out,
                "has_tracks": has_tracks_out,
                "has_visibility": has_tracks_out,
                "has_trajs_3d_world": has_tracks_out,
                "normal_convention": "camera_space_opencv_towards_camera" if has_normals_out else None,
                "normal_supervision_compatible": has_normals_out,
                "pose_convention": "w2c",
                "extrinsics_convention": "w2c",
                "intrinsics_convention": "pinhole",
                "precomputed_track_semantics_version": precomputed_track_semantics_version,
                "track_semantics_version": active_track_semantics_version,
                "precomputed_track_semantics_refreshed": precomputed_track_semantics_refreshed,
                "depth_unit": "meters",
                "depth_ranges": depth_ranges,  # [(depth_min, depth_max), ...]
                "use_masked": self.use_masked,
                "_load_timing": timing,
            },
        )

    def _load_precomputed(self, sequence_name: str, frame_indices: list[int]) -> Optional[dict]:
        """Load precomputed data for frame_indices. Prefers .h5 over .npz."""
        from multicamera4rc.datasets.adapters.base import load_precomputed_fast
        path = self.precompute_root / sequence_name / "precomputed.npz"
        h5_path = path.with_suffix('.h5')
        skip_keys = set() if self.load_normals else {"normals"}
        if not path.exists() and not h5_path.exists():
            return None
        if h5_path.exists():
            import h5py
            with h5py.File(h5_path, 'r') as f:
                n = f['trajs_2d'].shape[0] if 'trajs_2d' in f else 0
        else:
            n = int(np.load(path, allow_pickle=False)["num_frames"])
        if n < max(frame_indices) + 1:
            return None
        skip_keys = set() if self.load_normals else {"normals"}
        return load_precomputed_fast(path, frame_indices, skip_keys=skip_keys)

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        """Run consistency checks on a sequence.

        Checks
        ------
        1. Camera, image, and depth files exist for all frames.
        2. Intrinsics shape correct: (T, 3, 3).
        3. Extrinsics shape correct: (T, 4, 4).
        4. All intrinsic/extrinsic values are finite.
        5. Extrinsic rotation matrices are non-singular.
        6. Depth values are finite and contain at least some positive values.
        7. Round-trip reprojection: unproject center pixel with its depth,
           re-project, check reprojection error < 1 pixel (sampled from first
           5 frames).

        Returns
        -------
        dict with keys:
            ``dataset_name``, ``sequence_name``, ``ok`` (bool),
            ``messages`` (list[str])
        """
        r = self._get_record(sequence_name)
        msgs: list[str] = []
        ok = True

        # ── 1. File existence (all frames) ──────────────────────────────
        missing_rgb = []
        missing_dep = []
        missing_cam = []
        for fid in r.frame_ids:
            if not r.rgb_path(fid, self.use_masked).exists():
                missing_rgb.append(fid)
            if not r.depth_path(fid).exists():
                missing_dep.append(fid)
            if not r.cam_path(fid).exists():
                missing_cam.append(fid)

        for modal, lst in [("rgb", missing_rgb), ("depth", missing_dep), ("cam", missing_cam)]:
            if lst:
                ok = False
                msgs.append(
                    f"missing {modal} files ({len(lst)}): {lst[:5]}"
                    + (" ..." if len(lst) > 5 else "")
                )

        if not ok:
            return {"dataset_name": self.dataset_name, "sequence_name": sequence_name,
                    "ok": False, "messages": msgs}

        # ── Load a short probe clip (up to 5 frames) ────────────────────
        probe_count = min(5, r.num_frames)
        probe_indices = list(range(probe_count))
        try:
            clip = self.load_clip(sequence_name, probe_indices)
        except Exception as exc:
            return {
                "dataset_name": self.dataset_name,
                "sequence_name": sequence_name,
                "ok": False,
                "messages": msgs + [f"load_clip failed: {repr(exc)}"],
            }

        T = probe_count

        # ── 2 & 3. Shape checks ─────────────────────────────────────────
        if clip.intrinsics.shape != (T, 3, 3):
            ok = False
            msgs.append(f"intrinsics shape {clip.intrinsics.shape} != ({T}, 3, 3)")

        if clip.extrinsics.shape != (T, 4, 4):
            ok = False
            msgs.append(f"extrinsics shape {clip.extrinsics.shape} != ({T}, 4, 4)")

        # ── 4. Finite checks ─────────────────────────────────────────────
        if not np.isfinite(clip.intrinsics).all():
            ok = False
            msgs.append("intrinsics contains non-finite values")

        if not np.isfinite(clip.extrinsics).all():
            ok = False
            msgs.append("extrinsics contains non-finite values")

        # ── 5. Rotation non-singularity ──────────────────────────────────
        for t in range(T):
            det = float(np.linalg.det(clip.extrinsics[t, :3, :3]))
            if abs(det) < 1e-6:
                ok = False
                msgs.append(
                    f"extrinsics[{t}] rotation det ≈ {det:.2e} (nearly singular)"
                )

        # ── 6. Depth sanity ──────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            if not np.isfinite(dep).all():
                ok = False
                msgs.append(f"depth[{t}] contains non-finite values")
            if (dep > 0).sum() == 0:
                ok = False
                msgs.append(f"depth[{t}] has no positive values")

        # ── 7. Round-trip reprojection ───────────────────────────────────
        for t in range(T):
            img = clip.images[t]
            H, W = img.shape[:2]
            K = clip.intrinsics[t]          # (3, 3) w2c
            E = clip.extrinsics[t]          # (4, 4) w2c
            dep = clip.depths[t]            # (H, W)

            cy_px, cx_px = H // 2, W // 2
            d = float(dep[cy_px, cx_px])

            if not (np.isfinite(d) and d > 0):
                msgs.append(
                    f"reprojection check skipped for frame {t}: "
                    "depth at image center is invalid"
                )
                continue

            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            # Unproject center pixel to camera space.
            x_c = (cx_px - cx) / fx * d
            y_c = (cy_px - cy) / fy * d
            p_cam = np.array([x_c, y_c, d, 1.0], dtype=np.float64)

            # To world (E is w2c → invert).
            try:
                E_inv = np.linalg.inv(E.astype(np.float64))
            except np.linalg.LinAlgError:
                ok = False
                msgs.append(f"extrinsics[{t}] is singular — cannot invert")
                continue

            p_world = E_inv @ p_cam
            # Back to camera.
            p_cam2 = E.astype(np.float64) @ p_world
            # Project.
            p_proj = K.astype(np.float64) @ p_cam2[:3]
            u = p_proj[0] / p_proj[2]
            v = p_proj[1] / p_proj[2]

            reproj_err = max(abs(u - cx_px), abs(v - cy_px))
            if reproj_err > 1.0:
                ok = False
                msgs.append(
                    f"reprojection error too large at frame {t}: "
                    f"({u:.2f}, {v:.2f}) vs center ({cx_px}, {cy_px}), "
                    f"max err={reproj_err:.4f} px"
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
        list_filename = self._SPLIT_LIST_FILES[self.split]
        list_path = self.root / list_filename

        if not list_path.exists():
            raise FileNotFoundError(
                f"Split list file not found: {list_path}\n"
                f"Expected a text file with one scene ID per line."
            )

        with open(list_path, "r") as f:
            scene_ids = [ln.strip() for ln in f if ln.strip()]

        if not scene_ids:
            raise RuntimeError(f"Split list is empty: {list_path}")

        records: list[_SequenceRecord] = []
        skipped: list[str] = []

        def _index_one(scene_id: str) -> tuple[str, Optional[_SequenceRecord], Optional[Exception]]:
            scene_dir = self.root / scene_id
            try:
                return scene_id, self._index_scene(scene_id, scene_dir), None
            except Exception as exc:
                return scene_id, None, exc

        n_workers = min(self.index_workers, len(scene_ids))
        if n_workers > 1:
            order = {sid: i for i, sid in enumerate(scene_ids)}
            tmp: list[tuple[int, Optional[_SequenceRecord]]] = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_index_one, sid): sid for sid in scene_ids}
                for fut in as_completed(futures):
                    sid, rec, exc = fut.result()
                    if exc is not None:
                        if self.strict:
                            raise exc
                        skipped.append(f"{sid}: {exc}")
                        if self.verbose:
                            print(f"[BlendedMVSAdapter][WARN] skip {sid}: {exc}")
                    elif rec is not None:
                        tmp.append((order[sid], rec))
            records = [r for _, r in sorted(tmp)]
        else:
            for scene_id in scene_ids:
                sid, rec, exc = _index_one(scene_id)
                if exc is not None:
                    if self.strict:
                        raise exc
                    skipped.append(f"{sid}: {exc}")
                    if self.verbose:
                        print(f"[BlendedMVSAdapter][WARN] skip {sid}: {exc}")
                elif rec is not None:
                    records.append(rec)

        if self.verbose and skipped:
            print(f"[BlendedMVSAdapter] skipped {len(skipped)} scenes (non-strict mode)")

        return records

    def _index_scene(
        self, scene_id: str, scene_dir: Path
    ) -> Optional[_SequenceRecord]:
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

        cam_dir = scene_dir / "cams"
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"Missing cams/ in scene: {scene_dir}")

        # Enumerate camera files; sort by frame index.
        cam_files = [
            p for p in cam_dir.iterdir()
            if p.is_file() and _CAM_FILENAME_RE.match(p.name)
        ]
        if not cam_files:
            raise RuntimeError(f"No camera files found in {cam_dir}")

        cam_files.sort(key=lambda p: int(_CAM_FILENAME_RE.match(p.name).group(1)))
        frame_ids = [_CAM_FILENAME_RE.match(p.name).group(1) for p in cam_files]

        return _SequenceRecord(
            scene_id=scene_id,
            scene_dir=scene_dir,
            frame_ids=frame_ids,
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

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        """Load an RGB image as uint8 (H, W, 3)."""
        return np.asarray(Image.open(path).convert("RGB"))

    def _read_depth(self, path: Path, scene_id: str, frame_id: str) -> np.ndarray:
        if self.depth_cache_dir is None:
            return _read_pfm(path)

        cache_path = self.depth_cache_dir / scene_id / f"{frame_id}.npy"
        if cache_path.is_file():
            try:
                return np.load(cache_path, allow_pickle=False)
            except (OSError, ValueError):
                cache_path.unlink(missing_ok=True)

        depth = _read_pfm(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(
            f".{cache_path.name}.part.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with open(tmp_path, "wb") as f:
                np.save(f, depth, allow_pickle=False)
            os.replace(tmp_path, cache_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return depth

    @staticmethod
    def _read_image_size(path: Path) -> tuple[int, int]:
        img = Image.open(path)
        return img.height, img.width
