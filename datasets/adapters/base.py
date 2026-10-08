from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


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
            # Guard against frame_indices that exceed this dataset's length.
            clipped = [i for i in sorted_idx if i < ds.shape[0]]
            data = ds[clipped]              # reads only the needed chunks
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

    @abstractmethod
    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        pass

    @abstractmethod
    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        pass
