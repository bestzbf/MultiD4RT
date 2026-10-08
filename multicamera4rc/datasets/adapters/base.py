from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


def canonical_anchor_frame_index(num_frames: int) -> int:
    """Return the immutable lower-middle frame used by Stage-1 adapters.

    The policy is deliberately sequence-level and contains no clip-dependent
    state.  Keeping it here prevents small adapter-specific midpoint choices
    from silently creating multiple canonical gauges for one sequence.
    """

    if isinstance(num_frames, (bool, np.bool_)):
        raise ValueError("num_frames must be a positive integer")
    try:
        value = int(num_frames)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("num_frames must be a positive integer") from error
    if value <= 0 or float(num_frames) != float(value):
        raise ValueError("num_frames must be a positive integer")
    return (value - 1) // 2


def canonical_scene_anchor_metadata(
    *,
    extrinsic: np.ndarray,
    time: float,
    camera_id: int,
    frame_id: str,
    anchor_slot: Optional[int] = None,
    num_slots: Optional[int] = None,
    require_in_clip: bool = False,
) -> dict[str, Any]:
    """Validate and encode an authoritative sequence anchor.

    No pose is synthesized or repaired here.  In particular, a missing pose
    must never be represented by an identity matrix.  ``anchor_slot`` refers
    to the adapter output slot before any model-side layout remapping.
    """

    pose = np.asarray(extrinsic, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("canonical anchor extrinsic must be finite [4,4]")
    if not np.allclose(
        pose[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1.0e-6
    ):
        raise ValueError(
            "canonical anchor extrinsic must be a homogeneous [4,4] matrix"
        )

    if isinstance(time, (bool, np.bool_)):
        raise ValueError("canonical anchor time must be a finite scalar")
    try:
        anchor_time = float(time)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("canonical anchor time must be a finite scalar") from error
    if not np.isfinite(anchor_time):
        raise ValueError("canonical anchor time must be a finite scalar")

    if isinstance(camera_id, (bool, np.bool_)):
        raise ValueError("canonical anchor camera_id must be a non-negative integer")
    try:
        anchor_camera = int(camera_id)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "canonical anchor camera_id must be a non-negative integer"
        ) from error
    if anchor_camera < 0 or float(camera_id) != float(anchor_camera):
        raise ValueError("canonical anchor camera_id must be a non-negative integer")

    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError("canonical anchor frame_id must be a non-empty string")

    metadata: dict[str, Any] = {
        "canonical_scene_anchor_extrinsic": np.ascontiguousarray(
            pose.astype(np.float32)
        ),
        "canonical_scene_anchor_time": anchor_time,
        "canonical_scene_anchor_camera_id": anchor_camera,
        "canonical_scene_anchor_frame_id": frame_id.strip(),
        "canonical_scene_anchor_extrinsics_convention": "w2c",
        "canonical_scene_anchor_policy": "sequence_lower_middle_frame_v1",
    }

    if num_slots is None:
        if anchor_slot is not None or require_in_clip:
            raise ValueError("num_slots is required when encoding an anchor slot")
        return metadata
    if isinstance(num_slots, (bool, np.bool_)):
        raise ValueError("num_slots must be a positive integer")
    slot_count = int(num_slots)
    if slot_count <= 0 or float(num_slots) != float(slot_count):
        raise ValueError("num_slots must be a positive integer")

    if anchor_slot is None:
        if require_in_clip:
            raise ValueError(
                "canonical Stage-1 clip does not contain its fixed anchor view"
            )
        return metadata
    if isinstance(anchor_slot, (bool, np.bool_)):
        raise ValueError("canonical anchor slot must be an integer")
    slot = int(anchor_slot)
    if float(anchor_slot) != float(slot) or not 0 <= slot < slot_count:
        raise ValueError("canonical anchor slot lies outside the clip")
    mask = np.zeros(slot_count, dtype=np.bool_)
    mask[slot] = True
    metadata["canonical_anchor_slot"] = slot
    metadata["canonical_anchor_mask"] = mask.tolist()
    return metadata


def canonical_sequence_time_metadata(
    *,
    num_frames: int,
    reference_time: float,
    time_domain_id: str,
    track_pool_id: str,
    fps: float | None = None,
    track_pool_stable: bool = False,
) -> dict[str, Any]:
    """Build the explicit sequence-level time/identity contract.

    ``num_frames`` is the complete logical sequence length.  If ``fps`` is
    supplied, physical time is expressed in seconds; otherwise logical frame
    indices are used.  The helper intentionally does not inspect a clip, so
    every crop from one sequence receives byte-identical timeline metadata.
    Adapters should set ``track_pool_stable=True`` only when the returned
    track columns are a persistent full-sequence pool (never for clip-seeded
    depth sampling).
    """

    if isinstance(num_frames, (bool, np.bool_)):
        raise ValueError("num_frames must be a positive integer")
    count = int(num_frames)
    if count < 2 or float(num_frames) != float(count):
        raise ValueError("canonical sequence requires at least two frames")
    if fps is None:
        scale = 1.0
        unit = "frames"
    else:
        scale = float(fps)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("fps must be finite and positive")
        unit = "seconds"
    times = np.arange(count, dtype=np.float64) / scale
    time_range = [float(times[0]), float(times[-1])]
    reference = float(reference_time)
    if not np.isfinite(reference):
        raise ValueError("reference_time must be finite")
    if not isinstance(time_domain_id, str) or not time_domain_id.strip():
        raise ValueError("time_domain_id must be a non-empty string")
    if not isinstance(track_pool_id, str) or not track_pool_id.strip():
        raise ValueError("track_pool_id must be a non-empty string")
    return {
        "canonical_sequence_physical_times": times.tolist(),
        "canonical_sequence_time_range": time_range,
        "canonical_sequence_reference_time": reference,
        "canonical_sequence_time_domain_id": time_domain_id.strip(),
        "canonical_sequence_track_pool_id": track_pool_id.strip(),
        "canonical_sequence_track_pool_stable": bool(track_pool_stable),
        "canonical_sequence_time_unit": unit,
        "canonical_time_spec": {
            "physical_times": times.tolist(),
            "time_range": time_range,
            "reference_time": reference,
            "time_domain_id": time_domain_id.strip(),
            "track_pool_id": track_pool_id.strip(),
        },
    }


def h5_read_frame_slice(
    h5file: Any,
    frame_indices: list[int],
    keys: Optional[list[str]] = None,
) -> dict:
    """Read only the requested frame indices from an already-open h5py.File.

    h5py fancy indexing requires sorted, unique indices.  This helper handles
    the sort → read → reorder dance so callers don't duplicate it.

    Per-frame arrays (``ds.shape[0] > 1``) are indexed to ``frame_indices``
    order; scalar / metadata arrays are loaded in full.

    Args:
        h5file:        An open ``h5py.File`` (or any mapping with h5py Dataset
                       values that support ``ds[list_of_ints]`` and ``ds[()]``).
        frame_indices: Requested frame positions (may be unsorted or repeated).
        keys:          If provided, only these keys are read.  Defaults to all
                       keys in ``h5file``.

    Returns:
        dict mapping key → numpy array, already in ``frame_indices`` order.
    """
    sorted_idx = sorted(set(frame_indices))
    idx_map = {v: pos for pos, v in enumerate(sorted_idx)}
    reorder = [idx_map[i] for i in frame_indices]
    needs_reorder = reorder != list(range(len(frame_indices)))

    result: dict = {}
    per_point_keys = {
        "static_points_world",
        "static_point_ref_frames",
        "static_point_ref_uv",
        "static_point_ref_depth",
    }
    for key in (keys if keys is not None else h5file.keys()):
        ds = h5file[key]
        if key in per_point_keys:
            result[key] = ds[()]
        elif ds.ndim >= 2 and ds.shape[0] > 1:
            invalid = [i for i in sorted_idx if i < 0 or i >= ds.shape[0]]
            if invalid:
                raise IndexError(
                    f"frame indices {invalid} are out of range for HDF5 dataset "
                    f"{key!r} with {ds.shape[0]} frames"
                )
            if not sorted_idx:
                data = ds[np.empty(0, dtype=np.int64)]
            elif ds.chunks is not None and ds.chunks[0] == 1:
                # h5py sparse fancy indexing is very slow for frame-chunked
                # compressed arrays; direct per-frame reads touch the same
                # chunks without the selection overhead.
                data = np.stack([ds[i] for i in sorted_idx], axis=0)
            else:
                data = ds[sorted_idx]
            result[key] = data[reorder] if needs_reorder else data
        else:
            result[key] = ds[()]            # scalar / metadata
    return result


def load_precomputed_fast(
    npz_path: Path,
    frame_indices: list[int],
    skip_keys: Optional[set[str]] = None,
) -> Optional[dict]:
    """
    Load precomputed tracks/normals for specific frame indices.

    Prefers .h5 (chunked HDF5, O(frames) random access) over .npz
    (requires full zlib decompression of the entire array).
    Falls back to .npz if .h5 is not found.

    Returns a dict with arrays already indexed to frame_indices order,
    or None if neither .h5 nor .npz exists.

    Args:
        skip_keys: Set of array keys to skip entirely.  Use this to avoid
                   decompressing expensive arrays that the caller does not
                   need (e.g. ``{"normals"}`` saves ~2 s for a 50 MB npz).

    Run computer/convert_precomputed_to_h5.py once to generate .h5 files.
    """
    npz_path = Path(npz_path)
    h5_path = npz_path.with_suffix('.h5')
    _skip = skip_keys or set()

    if h5_path.exists():
        import h5py
        with h5py.File(h5_path, 'r') as f:
            keys = [k for k in f.keys() if k not in _skip]
            return h5_read_frame_slice(f, frame_indices, keys=keys)

    elif npz_path.exists():
        raw = np.load(npz_path, allow_pickle=True)
        result = {}
        per_point_keys = {
            "static_points_world",
            "static_point_ref_frames",
            "static_point_ref_uv",
            "static_point_ref_depth",
        }
        for k in raw.files:
            if k in _skip:
                continue
            arr = raw[k]
            if (
                arr.ndim >= 2
                and arr.shape[0] > 1
                and k not in ('origin_shift',)
                and k not in per_point_keys
            ):
                result[k] = arr[np.array(frame_indices)]
            else:
                result[k] = arr[()]
        return result

    return None


@dataclass
class UnifiedClip:
    dataset_name: str
    sequence_name: str
    frame_paths: Optional[list[str]]
    images: list[np.ndarray]                  # [T][H,W,3]
    depths: Optional[list[np.ndarray]]        # [T][H,W]
    normals: Optional[list[np.ndarray]]       # [T][H,W,3]
    trajs_2d: Optional[np.ndarray]            # [T,N,2]
    trajs_3d_world: Optional[np.ndarray]      # [T,N,3]
    valids: Optional[np.ndarray]              # [T,N]
    visibs: Optional[np.ndarray]              # [T,N]
    intrinsics: np.ndarray                    # [T,3,3]
    extrinsics: np.ndarray                    # [T,4,4]
    flows: Optional[list[np.ndarray]] = None  # [T][H,W,2], optical flow
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional three-state visibility contract. ``valids`` says the 3D track
    # exists; this mask says whether visible/invisible is a reliable label.
    visibility_valids: Optional[np.ndarray] = None  # [T,N]
    # Optional per-track motion semantics. ``track_is_dynamic`` is aligned to
    # the N dimension of every trajectory tensor and remains constant in time.
    # ``track_dynamic_known`` preserves unknown labels for partially supported
    # tracks; when omitted alongside an explicit label array, every label is
    # authoritative for backward compatibility.
    track_is_dynamic: Optional[np.ndarray] = None   # [N]
    track_dynamic_known: Optional[np.ndarray] = None  # [N]

    # Optional authoritative sequence-level payload used only by Canonical
    # World-Bank Stage-1.  The regular fields above always remain clip-local
    # (and may therefore be cropped/subsampled); these fields are deliberately
    # separate so a partial clip can never be mistaken for a complete spline
    # domain.  All arrays use the persistent track-column ordering published by
    # the adapter and are never altered by spatial transforms.
    canonical_trajs_2d: Optional[np.ndarray] = None       # [L,N,2]
    canonical_trajs_3d_world: Optional[np.ndarray] = None  # [L,N,3]
    canonical_valids: Optional[np.ndarray] = None         # [L,N]
    canonical_visibs: Optional[np.ndarray] = None         # [L,N]
    canonical_visibility_valids: Optional[np.ndarray] = None  # [L,N]
    canonical_physical_times: Optional[np.ndarray] = None # [L]
    canonical_track_ids: Optional[np.ndarray] = None      # [N]
    canonical_track_is_dynamic: Optional[np.ndarray] = None  # [N]
    canonical_track_dynamic_known: Optional[np.ndarray] = None  # [N]
    canonical_slot_camera: Optional[np.ndarray] = None    # [L]
    canonical_slot_physical_camera_id: Optional[np.ndarray] = None  # [L]
    canonical_extrinsics: Optional[np.ndarray] = None     # [L,4,4]

    @property
    def num_frames(self) -> int:
        return len(self.images)

    @property
    def image_size(self) -> tuple[int, int]:
        h, w = self.images[0].shape[:2]
        return h, w


class BaseAdapter(ABC):
    dataset_name: str = "base"

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def list_sequences(self) -> list[str]:
        pass

    @abstractmethod
    def get_sequence_name(self, index: int) -> str:
        pass

    @abstractmethod
    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        pass

    def get_num_frames(self, sequence_name: str) -> int:
        """Return number of frames for sequence_name without expensive I/O.

        Subclasses should override this if they can return num_frames directly
        from their in-memory index without loading annotation files.
        Default falls back to get_sequence_info (may be slow for some adapters).
        """
        return self.get_sequence_info(sequence_name)['num_frames']

    def get_canonical_anchor_frame_index(self, sequence_name: str) -> int:
        """Return the immutable source-frame anchor for ``sequence_name``.

        Canonical Stage-1 is fail-closed: adapters must opt in only when they
        can read an authoritative full-sequence pose for this frame.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not provide an authoritative "
            "sequence-level canonical anchor"
        )

    def get_canonical_anchor_sampler_index(self, sequence_name: str) -> int:
        """Return the anchor expressed in this adapter's sampler units."""

        return self.get_canonical_anchor_frame_index(sequence_name)

    @abstractmethod
    def load_clip(
        self,
        sequence_name: str,
        frame_indices: list[int],
        *,
        canonical_anchor_required: bool = False,
    ) -> UnifiedClip:
        """Load a clip, failing closed on the optional canonical contract.

        Adapters that do not implement an authoritative sequence-level anchor
        may keep their narrower runtime signature: the mixture calls this
        keyword only after ``get_canonical_anchor_sampler_index`` succeeds.
        """
        pass

    @abstractmethod
    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        pass
