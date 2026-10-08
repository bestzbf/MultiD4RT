from __future__ import annotations

import contextlib
import io
import fcntl
import hashlib
import json
import os
import pickle
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy.lib.format as _npy_fmt  # for lightweight numpy header reads
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import BaseAdapter, UnifiedClip, h5_read_frame_slice


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr", ".npy"}

# PointOdyssey official export code applies a fixed basis change R1 when it
# writes the released extrinsics:
#   RT_saved = R3 @ R2 @ RT @ R1
# but the released normal PNG/JPG files are copied directly from the rendered
# normal pass after visualization encoding, without the same camera-basis
# correction. After decoding those images back to unit vectors, we therefore
# need to rotate them into the released camera space using:
#   normal_cam = normal_saved_basis @ (R_saved @ R1^T)^T
# where R_saved is the 3x3 rotation block of the released world-to-camera
# extrinsic. See PointOdyssey official `utils/gen_tracking*.py` and
# `utils/openexr_utils.py`.
POINTODYSSEY_R1 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

FAST_ANNOTATION_DIRNAME = "anno_fast"
FAST_REQUIRED_ANNO_FILES = ("trajs_2d", "trajs_3d", "valids", "intrinsics", "extrinsics")
FAST_FRAME_MANIFEST_FILENAME = "frame_manifest.json"
FAST_ENCODED_META_FILENAME = "frame_pack_meta.json"
FAST_ENCODED_RGB_BIN = "rgb_frames.bin"
FAST_ENCODED_RGB_OFFSETS = "rgb_frames_offsets.npy"
FAST_ENCODED_DEPTH_BIN = "depth_frames.bin"
FAST_ENCODED_DEPTH_OFFSETS = "depth_frames_offsets.npy"
FAST_ENCODED_NORMAL_BIN = "normal_frames.bin"
FAST_ENCODED_NORMAL_OFFSETS = "normal_frames_offsets.npy"
FAST_ENCODED_NORMAL_VALIDS = "normal_frames_valids.npy"


class _CosRangeFile(io.RawIOBase):
    """Small read-only file object backed by COS Range requests.

    h5py can open Python file-like objects.  Using this avoids calling h5py on
    the COS FUSE mount when we only need HDF5 metadata/chunk offsets.
    """

    def __init__(
        self,
        *,
        size: int,
        read_range: Any,
        block_size: int = 4 * 1024 * 1024,
        max_blocks: int = 32,
    ) -> None:
        super().__init__()
        self._size = max(0, int(size))
        self._read_range = read_range
        self._block_size = max(64 * 1024, int(block_size))
        self._max_blocks = max(1, int(max_blocks))
        self._pos = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            new_pos = int(offset)
        elif whence == os.SEEK_CUR:
            new_pos = self._pos + int(offset)
        elif whence == os.SEEK_END:
            new_pos = self._size + int(offset)
        else:
            raise ValueError(f"Unsupported seek whence: {whence}")
        self._pos = max(0, new_pos)
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed COS range file")
        if self._pos >= self._size:
            return b""
        if size is None or size < 0:
            end = self._size
        else:
            end = min(self._size, self._pos + int(size))
        start = self._pos
        if end <= start:
            return b""

        chunks: list[bytes] = []
        cur = start
        while cur < end:
            block_start = (cur // self._block_size) * self._block_size
            block = self._get_block(block_start)
            offset = cur - block_start
            take = min(end - cur, len(block) - offset)
            if take <= 0:
                break
            chunks.append(block[offset:offset + take])
            cur += take
        self._pos = cur
        return b"".join(chunks)

    def readinto(self, b: Any) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def _get_block(self, block_start: int) -> bytes:
        cached = self._cache.get(block_start)
        if cached is not None:
            self._cache.move_to_end(block_start)
            return cached
        block_end = min(self._size, block_start + self._block_size)
        data = self._read_range(block_start, block_end)
        self._cache[block_start] = data
        self._cache.move_to_end(block_start)
        while len(self._cache) > self._max_blocks:
            self._cache.popitem(last=False)
        return data


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Decompress an HDF5 LZF chunk payload."""
    try:
        import imagecodecs

        decoded = imagecodecs.lzf_decode(data, out=expected_size)
        if len(decoded) != expected_size:
            raise ValueError(
                f"LZF decoded {len(decoded)} bytes, expected {expected_size}"
            )
        return bytes(decoded)
    except ImportError as exc:
        raise RuntimeError(
            "PointOdyssey COS Range anno.h5 reading needs imagecodecs to decode "
            "LZF-compressed HDF5 chunks."
        ) from exc


@dataclass
class SequenceRecord:
    dataset_name: str
    split: str
    sequence_name: str
    sequence_root: Path

    rgb_paths: list[Path]
    depth_paths: Optional[list[Path]]
    normal_paths: Optional[list[Path]]
    mask_paths: Optional[list[Path]]

    anno_path: Optional[Path]
    info_path: Optional[Path]
    scene_info_path: Optional[Path]

    num_frames: int
    image_size: Optional[tuple[int, int]]
    has_tracks: Optional[bool] = None

    # Fast-loading paths (populated for PointOdyssey_fast sequences)
    fast_dir: Optional[Path] = None
    fast_anno_paths: Optional[dict[str, Path]] = None
    encoded_cache_paths: Optional[dict[str, Path]] = None


class PointOdysseyAdapter(BaseAdapter):
    dataset_name: str = "pointodyssey"

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_anno_cos_tls", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._anno_cos_tls = threading.local()

    def __init__(
        self,
        root: str,
        split: str = "train",
        strict: bool = True,
        verbose: bool = True,
        index_workers: int = 8,
        fast_root: Optional[str] = None,
        require_tracks: bool = False,
        cache_dir: Optional[str] = None,
        runtime_sanitize: bool = True,
        io_workers: int = 1,
        anno_frame_cache_dir: Optional[str] = None,
        load_normals: bool = True,
        anno_read_mode: str = "auto",
        anno_cos_mount_root: str = "/data_cos",
        anno_cos_bucket: str = "hd-ai-data-1251882982",
        anno_cos_region: str = "ap-beijing",
        anno_cos_passwd_file: str = "/etc/passwd-s3fs-data_cos",
        anno_cos_timeout_s: int = 20,
        anno_cos_range_workers: int = 4,
        anno_cos_range_retries: int = 2,
        anno_cos_range_merge_gap_bytes: int = 1024 * 1024,
        anno_index_window_radius: int = 128,
    ):
        self.root = Path(root)
        self.split = split
        self.strict = strict
        self.verbose = verbose
        self.index_workers = index_workers
        self.fast_root = Path(fast_root) if fast_root is not None else None
        self.require_tracks = require_tracks
        self.runtime_sanitize = bool(runtime_sanitize)
        self.io_workers = max(1, int(io_workers))
        self.load_normals = bool(load_normals)
        self.anno_read_mode = str(anno_read_mode or "auto").strip().lower()
        self.anno_cos_mount_root = Path(anno_cos_mount_root)
        self.anno_cos_bucket = str(anno_cos_bucket)
        self.anno_cos_region = str(anno_cos_region)
        if (
            anno_cos_passwd_file == "/etc/passwd-s3fs-data_cos"
            and not Path(anno_cos_passwd_file).exists()
            and Path("/etc/passwd-cosfs").exists()
        ):
            anno_cos_passwd_file = "/etc/passwd-cosfs"
        self.anno_cos_passwd_file = str(anno_cos_passwd_file)
        self.anno_cos_timeout_s = int(anno_cos_timeout_s)
        self.anno_cos_range_workers = max(1, int(anno_cos_range_workers))
        self.anno_cos_range_retries = max(0, int(anno_cos_range_retries))
        self.anno_cos_range_merge_gap_bytes = max(
            0, int(anno_cos_range_merge_gap_bytes)
        )
        self.anno_index_window_radius = max(0, int(anno_index_window_radius))
        self.anno_local_cache_max_bytes = int(
            float(os.getenv("POINTODYSSEY_ANNO_LOCAL_CACHE_MAX_GB", "24")) * 1024**3
        )
        self._anno_cos_tls = threading.local()
        self._anno_h5_chunk_index_cache: dict[str, dict[str, Any]] = {}
        self._anno_h5_index_cache_root = self._resolve_anno_h5_index_cache_root()
        self._h5_range_cache_root = self._resolve_h5_range_cache_root()
        self._last_anno_range_stats: dict[str, int] = {}
        resolved_anno_frame_cache_dir = anno_frame_cache_dir or os.getenv(
            "D4RT_POINTODYSSEY_ANNO_FRAME_CACHE_DIR", ""
        )
        self.anno_frame_cache_dir = (
            Path(resolved_anno_frame_cache_dir)
            if resolved_anno_frame_cache_dir
            else None
        )

        split_root = self.root / self.split

        # Check cache first to skip slow root.exists() on remote storage
        _cache_hit = False
        if cache_dir is not None:
            from datasets.index_cache import load_or_build
            cache_key = {
                "dataset": self.dataset_name,
                "split": split,
                "root": str(self.root.resolve()),
                "fast_root": str(self.fast_root.resolve()) if self.fast_root is not None else None,
                "require_tracks": self.require_tracks,
                "strict": self.strict,
                "cache_schema": 3,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            _cache_path = Path(cache_dir) / f"{self.dataset_name}_{split}_{cache_suffix}.pkl"
            if _cache_path.exists():
                _cache_hit = True

        if not _cache_hit and not split_root.exists():
            raise FileNotFoundError(
                f"Split root not found: {split_root}\n"
                f"Expected root like /path/to/PointOdyssey and split like 'train'/'test'."
            )

        _cache_path: Optional[Path] = None
        if cache_dir is not None:
            from datasets.index_cache import load_or_build
            cache_key = {
                "dataset": self.dataset_name,
                "split": split,
                "root": str(self.root.resolve()),
                "fast_root": str(self.fast_root.resolve()) if self.fast_root is not None else None,
                "require_tracks": self.require_tracks,
                "strict": self.strict,
                "cache_schema": 3,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            _cache_path = Path(cache_dir) / f"{self.dataset_name}_{split}_{cache_suffix}.pkl"
            self.records: list[SequenceRecord] = load_or_build(self._build_index, _cache_path)
        else:
            self.records: list[SequenceRecord] = self._build_index()
        if self.runtime_sanitize:
            self.records = self._sanitize_records_for_runtime(self.records)
        if self.require_tracks:
            cache_missing_track_flags = any(
                getattr(r, "has_tracks", None) is None for r in self.records
            )
            records_for_cache = self.records
            original_count = len(self.records)
            self.records = [r for r in self.records if self._record_has_tracks(r)]
            if cache_missing_track_flags and _cache_path is not None:
                self._rewrite_index_cache_with_track_flags(_cache_path, records_for_cache)
            if self.verbose:
                print(
                    f"[PointOdysseyAdapter] require_tracks=True kept "
                    f"{len(self.records)}/{original_count} sequences"
                )
        self.name_to_record: dict[str, SequenceRecord] = {}
        for r in self.records:
            if r.sequence_name in self.name_to_record:
                raise ValueError(f"Duplicate sequence_name found: {r.sequence_name}")
            self.name_to_record[r.sequence_name] = r

        if len(self.records) == 0:
            raise RuntimeError(f"No valid PointOdyssey sequences found under {split_root}")

        self._scene_info_cache: dict[str, Optional[dict[str, Any]]] = {}
        # Cache for open memmap handles (avoids re-opening on every load_clip call)
        self._encoded_cache_store: dict[str, dict[str, Any]] = {}

        if self.verbose:
            fast_count = sum(1 for r in self.records if r.fast_anno_paths is not None)
            enc_count = sum(1 for r in self.records if r.encoded_cache_paths is not None)
            print(
                f"[PointOdysseyAdapter] split={self.split}, "
                f"num_sequences={len(self.records)}, strict={self.strict}, "
                f"fast_anno={fast_count}, encoded_cache={enc_count}"
            )

    def _record_has_tracks(self, record: SequenceRecord) -> bool:
        cached = getattr(record, "has_tracks", None)
        if cached is not None:
            return bool(cached)

        if os.getenv("POINTODYSSEY_ASSUME_TRACKS", "0").strip().lower() in {"1", "true", "yes", "on"}:
            return True

        seq_root = str(getattr(record, "sequence_root", ""))
        if seq_root.startswith("/data_cos/") and os.getenv(
            "D4RT_POINTODYSSEY_PROBE_REMOTE_TRACKS", "0"
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            # Server-side COS caches are expensive to probe at startup. They are
            # expected to be pre-filtered by the cache warmer unless explicitly
            # requested otherwise.
            return True

        has_tracks = False
        try:
            if record.fast_anno_paths is not None and "trajs_3d" in record.fast_anno_paths:
                trajs_3d = np.load(record.fast_anno_paths["trajs_3d"], mmap_mode="r")
                has_tracks = not (trajs_3d.ndim == 0 or trajs_3d.shape[0] == 0)
            elif record.anno_path is not None:
                h5_path = Path(record.anno_path).with_suffix(".h5")
                if h5_path.exists():
                    import h5py

                    with h5py.File(h5_path, "r") as f:
                        if "trajs_3d" in f:
                            trajs_3d = f["trajs_3d"]
                            has_tracks = not (
                                trajs_3d.ndim == 0 or trajs_3d.shape[0] == 0
                            )
                else:
                    anno = np.load(record.anno_path, allow_pickle=True)
                    if "trajs_3d" in anno:
                        trajs_3d = anno["trajs_3d"]
                        has_tracks = not (trajs_3d.ndim == 0 or trajs_3d.shape[0] == 0)
        except (FileNotFoundError, KeyError, OSError, ValueError, IndexError):
            has_tracks = False

        try:
            record.has_tracks = has_tracks
        except Exception:
            pass
        return has_tracks

    @staticmethod
    def _rewrite_index_cache_with_track_flags(
        cache_path: Path,
        records: list[SequenceRecord],
    ) -> None:
        """Persist lazily discovered has_tracks flags into the local index cache."""
        cache_path = Path(cache_path)
        tmp = cache_path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
        except OSError:
            pass
        finally:
            tmp.unlink(missing_ok=True)

    def _sanitize_records_for_runtime(
        self,
        records: list[SequenceRecord],
    ) -> list[SequenceRecord]:
        """Reconcile cached records with the actually mounted dataset tree."""
        sanitized: list[SequenceRecord] = []
        dropped = 0

        for record in records:
            seq_root = Path(record.sequence_root)
            if not seq_root.is_dir():
                dropped += 1
                continue

            if record.fast_dir is not None and not Path(record.fast_dir).is_dir():
                record.fast_dir = None
                record.fast_anno_paths = None
                record.encoded_cache_paths = None

            if record.fast_anno_paths is not None:
                fast_anno_paths = {
                    k: p for k, p in record.fast_anno_paths.items() if Path(p).exists()
                }
                if all(k in fast_anno_paths for k in FAST_REQUIRED_ANNO_FILES):
                    record.fast_anno_paths = fast_anno_paths
                else:
                    record.fast_anno_paths = None

            if record.encoded_cache_paths is not None:
                encoded_cache_paths = {
                    k: p for k, p in record.encoded_cache_paths.items() if Path(p).exists()
                }
                required_encoded = {
                    "rgb_bin_path",
                    "rgb_offsets_path",
                    "depth_bin_path",
                    "depth_offsets_path",
                    "normal_bin_path",
                    "normal_offsets_path",
                    "normal_valids_path",
                    "meta_path",
                }
                if required_encoded.issubset(encoded_cache_paths):
                    record.encoded_cache_paths = encoded_cache_paths
                else:
                    record.encoded_cache_paths = None

            if record.anno_path is not None and not Path(record.anno_path).exists():
                npz_path = seq_root / "anno.npz"
                if npz_path.exists():
                    record.anno_path = npz_path
                else:
                    record.anno_path = None

            has_rgb = record.encoded_cache_paths is not None or (seq_root / "rgbs").is_dir()
            has_anno = record.fast_anno_paths is not None or record.anno_path is not None
            if not has_rgb or not has_anno:
                dropped += 1
                continue

            if record.depth_paths is not None and not (seq_root / "depths").is_dir():
                record.depth_paths = None
            if record.normal_paths is not None and not (seq_root / "normals").is_dir():
                record.normal_paths = None
            if record.mask_paths is not None and not (seq_root / "masks").is_dir():
                record.mask_paths = None
            if record.info_path is not None and not Path(record.info_path).exists():
                record.info_path = None
            if record.scene_info_path is not None and not Path(record.scene_info_path).exists():
                record.scene_info_path = None

            sanitized.append(record)

        if self.verbose and dropped:
            print(f"[PointOdysseyAdapter] dropped {dropped} cached records with missing mandatory data")
        return sanitized

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.records)

    def list_sequences(self) -> list[str]:
        return [r.sequence_name for r in self.records]

    def get_sequence_name(self, index: int) -> str:
        return self.records[index].sequence_name

    def get_record(self, sequence_name: str) -> SequenceRecord:
        if sequence_name not in self.name_to_record:
            raise KeyError(f"Unknown sequence_name: {sequence_name}")
        return self.name_to_record[sequence_name]

    def get_num_frames(self, sequence_name: str) -> int:
        """Fast path: read from cached record."""
        return self.get_record(sequence_name).num_frames

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self.get_record(sequence_name)
        has_tracks = self._record_has_tracks(r)
        return {
            "dataset_name": r.dataset_name,
            "split": r.split,
            "sequence_name": r.sequence_name,
            "sequence_root": str(r.sequence_root),
            "num_frames": r.num_frames,
            "image_size": r.image_size,
            "has_depth": r.depth_paths is not None,
            "has_normals": r.normal_paths is not None,
            "has_masks": r.mask_paths is not None,
            "has_anno": r.anno_path is not None or r.fast_anno_paths is not None,
            "has_info": r.info_path is not None,
            "has_scene_info": r.scene_info_path is not None,
            "has_tracks": has_tracks,
            "has_visibility": has_tracks,
            "has_trajs_3d_world": has_tracks,
            "fast_anno": r.fast_anno_paths is not None,
            "encoded_cache": r.encoded_cache_paths is not None,
        }

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        r = self.get_record(sequence_name)

        out: dict[str, Any] = {
            "dataset_name": r.dataset_name,
            "split": r.split,
            "sequence_name": r.sequence_name,
            "sequence_root": str(r.sequence_root),
            "num_frames": r.num_frames,
            "image_size": r.image_size,
            "rgb_count": len(r.rgb_paths),
            "depth_count": len(r.depth_paths) if r.depth_paths is not None else None,
            "normal_count": len(r.normal_paths) if r.normal_paths is not None else None,
            "mask_count": len(r.mask_paths) if r.mask_paths is not None else None,
            "has_anno": r.anno_path is not None or r.fast_anno_paths is not None,
            "has_info": r.info_path is not None,
            "has_scene_info": r.scene_info_path is not None,
            "fast_anno": r.fast_anno_paths is not None,
            "encoded_cache": r.encoded_cache_paths is not None,
        }

        if len(r.rgb_paths) != r.num_frames:
            out["error"] = f"rgb count {len(r.rgb_paths)} != num_frames {r.num_frames}"
            return out

        for modal_name, modal_paths in [
            ("depth", r.depth_paths),
            ("normal", r.normal_paths),
            ("mask", r.mask_paths),
        ]:
            if modal_paths is not None and len(modal_paths) != r.num_frames:
                out["error"] = (
                    f"{modal_name} count {len(modal_paths)} != num_frames {r.num_frames}"
                )
                return out

        anno = self._load_anno(sequence_name)
        required = ["trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics"]
        missing = [k for k in required if k not in anno]
        if missing:
            out["error"] = f"missing anno keys: {missing}"
            return out

        out["anno_shapes"] = {
            "trajs_2d": tuple(anno["trajs_2d"].shape),
            "trajs_3d": tuple(anno["trajs_3d"].shape),
            "valids": tuple(anno["valids"].shape),
            "visibs": tuple(anno["visibs"].shape),
            "intrinsics": tuple(anno["intrinsics"].shape),
            "extrinsics": tuple(anno["extrinsics"].shape),
        }
        out["ok"] = True
        return out

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        timing: dict[str, float] = {}
        t_total_start = time.perf_counter()
        r = self.get_record(sequence_name)

        frame_indices_np = np.asarray(frame_indices, dtype=np.int64)
        if frame_indices_np.ndim != 1:
            raise ValueError("frame_indices must be a 1D list/array of ints.")
        if len(frame_indices_np) == 0:
            raise ValueError("frame_indices must not be empty.")
        if frame_indices_np.min() < 0 or frame_indices_np.max() >= r.num_frames:
            raise IndexError(
                f"frame_indices out of range for sequence '{sequence_name}', "
                f"valid range = [0, {r.num_frames - 1}]"
            )

        def _timed_load_anno() -> dict[str, Any]:
            t0 = time.perf_counter()
            out = self._load_anno(sequence_name, frame_indices_np)
            timing["precomputed_s"] = time.perf_counter() - t0
            range_stats = getattr(self, "_last_anno_range_stats", None)
            if range_stats:
                for key, value in range_stats.items():
                    timing[f"precomputed_range_{key}"] = float(value)
            return out

        if self.io_workers > 1 and len(frame_indices_np) > 1:
            with ThreadPoolExecutor(max_workers=1) as anno_executor:
                anno_future = anno_executor.submit(_timed_load_anno)
                t0 = time.perf_counter()
                scene_info = self._load_scene_info(sequence_name)
                timing["scene_data_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                images, depths, normals = self._load_frame_payload(
                    sequence_name, r, frame_indices_np
                )
                timing["frame_load_s"] = time.perf_counter() - t0
                anno = anno_future.result()
        else:
            anno = _timed_load_anno()
            t0 = time.perf_counter()
            scene_info = self._load_scene_info(sequence_name)
            timing["scene_data_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            images, depths, normals = self._load_frame_payload(
                sequence_name, r, frame_indices_np
            )
            timing["frame_load_s"] = time.perf_counter() - t0

        # PointOdyssey depth uses 16-bit visualization units:
        # depth_uint16 = depth_meters * 65535 / 1000.
        t_process0 = time.perf_counter()
        if depths is not None:
            depths = [d * (1000.0 / 65535.0) if d is not None else None for d in depths]

        # _load_anno pre-slices all paths to frame_indices_np, so index with [:].
        trajs_2d = anno["trajs_2d"][:].astype(np.float32)               # [T,N,2]

        trajs_3d_raw = anno["trajs_3d"]
        if trajs_3d_raw.ndim == 0 or trajs_3d_raw.shape[0] == 0:
            trajs_3d_world = None
        else:
            trajs_3d_world = trajs_3d_raw[:].astype(np.float32)         # [T,N,3]

        valids = anno["valids"][:].astype(bool)                         # [T,N]
        visibs = anno["visibs"][:].astype(bool)                         # [T,N]
        intrinsics = anno["intrinsics"][:].astype(np.float32)           # [T,3,3]
        extrinsics = anno["extrinsics"][:].astype(np.float32)           # [T,4,4]
        if normals is not None:
            normals = self._convert_released_normals_to_camera_space(normals, extrinsics)
        timing["process_s"] = time.perf_counter() - t_process0
        timing["total_s"] = time.perf_counter() - t_total_start

        clip = UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=[str(r.rgb_paths[i]) for i in frame_indices_np],
            images=images,
            depths=depths,
            normals=normals,
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d_world,
            valids=valids,
            visibs=visibs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata={
                "dataset_name": self.dataset_name,
                "split": r.split,
                "sequence_root": str(r.sequence_root),
                "frame_indices": frame_indices_np.tolist(),
                "num_frames_in_sequence": r.num_frames,
                "raw_image_hw": list(r.image_size) if r.image_size is not None else None,
                "has_depth": depths is not None,
                "has_normals": normals is not None,
                "has_tracks": trajs_3d_world is not None,
                "has_visibility": trajs_3d_world is not None,
                "has_trajs_3d_world": trajs_3d_world is not None,
                "has_masks": r.mask_paths is not None,
                "pose_convention": "world_to_camera",
                "trajs_3d_convention": "world",
                "intrinsics_convention": "pinhole",
                "normal_source_convention": "pointodyssey_released_normal_pass_decoded",
                "normal_convention": "camera_space_opencv_towards_camera" if normals is not None else None,
                "normal_supervision_compatible": normals is not None,
                "depth_unit": "unknown",
                "raw_mask_semantics": "unknown",
                "scene_info_keys": list(scene_info.keys()) if scene_info is not None else None,
                "_load_timing": timing,
            },
        )
        return clip

    def _load_frame_payload(
        self,
        sequence_name: str,
        r: SequenceRecord,
        frame_indices_np: np.ndarray,
    ) -> tuple[list[np.ndarray], Optional[list[np.ndarray]], Optional[list[np.ndarray]]]:
        if r.encoded_cache_paths is not None:
            # Fast path: decode frames from packed binary cache
            cache = self._get_encoded_cache(sequence_name, r)
            frame_ids = [int(i) for i in frame_indices_np]
            images = self._parallel_map(
                lambda i: self._decode_rgb_from_cache(cache, i), frame_ids
            )
            depths = self._parallel_map(
                lambda i: self._decode_depth_from_cache(cache, i), frame_ids
            )
            normals = None
            if self.load_normals:
                normals_raw = self._parallel_map(
                    lambda i: self._decode_normal_from_cache(cache, i), frame_ids
                )
                # Replace None (invalid normal) with zeros matching the image shape
                h, w = images[0].shape[:2]
                normals = [
                    n if n is not None else np.zeros((h, w, 3), dtype=np.float32)
                    for n in normals_raw
                ]
        else:
            # Fallback: read individual frame files
            frame_ids = [int(i) for i in frame_indices_np]
            images = self._parallel_map(lambda i: self._read_rgb(r.rgb_paths[i]), frame_ids)
            depths = None
            if r.depth_paths is not None:
                try:
                    depths = self._parallel_map(
                        lambda i: self._read_depth(r.depth_paths[i]), frame_ids
                    )
                except (FileNotFoundError, OSError, IOError):
                    r.depth_paths = None
                    depths = None
            normals = None
            if self.load_normals and r.normal_paths is not None:
                try:
                    normals = self._parallel_map(
                        lambda i: self._read_normal(r.normal_paths[i]), frame_ids
                    )
                except (FileNotFoundError, OSError, IOError):
                    r.normal_paths = None
                    normals = None

        return images, depths, normals

    def _parallel_map(self, fn, values: list[int]) -> list[Any]:
        if self.io_workers <= 1 or len(values) <= 1:
            return [fn(value) for value in values]
        with ThreadPoolExecutor(max_workers=min(len(values), self.io_workers)) as executor:
            return list(executor.map(fn, values))

    def probe_sequence_meta(self, sequence_name: str) -> dict[str, Any]:
        r = self.get_record(sequence_name)
        out: dict[str, Any] = {
            "sequence_name": sequence_name,
            "anno": None,
            "info": None,
            "scene_info_json_keys": None,
        }

        if r.fast_anno_paths is not None:
            out["anno"] = {
                k: {
                    "shape": tuple(np.load(p, mmap_mode="r", allow_pickle=False).shape),
                    "dtype": str(np.load(p, mmap_mode="r", allow_pickle=False).dtype),
                }
                for k, p in r.fast_anno_paths.items()
            }
        elif r.anno_path is not None:
            z = np.load(r.anno_path, allow_pickle=True)
            out["anno"] = {
                k: {
                    "shape": getattr(z[k], "shape", None),
                    "dtype": str(getattr(z[k], "dtype", None)),
                }
                for k in z.files
            }

        if r.info_path is not None:
            z = np.load(r.info_path, allow_pickle=True)
            out["info"] = {
                k: {
                    "shape": getattr(z[k], "shape", None),
                    "dtype": str(getattr(z[k], "dtype", None)),
                }
                for k in z.files
            }

        if r.scene_info_path is not None:
            with open(r.scene_info_path, "r") as f:
                meta = json.load(f)
            out["scene_info_json_keys"] = list(meta.keys())

        return out

    # ------------------------------------------------------------------ #
    #  Index building                                                      #
    # ------------------------------------------------------------------ #

    def _build_index(self) -> list[SequenceRecord]:
        split_root = self.root / self.split
        scene_dirs = sorted(
            [p for p in split_root.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )

        results: list[Optional[SequenceRecord]] = [None] * len(scene_dirs)
        skipped: list[str] = []

        def _index_one(args):
            i, scene_dir = args
            try:
                rec = self._index_scene(scene_dir)
                return i, rec, None
            except Exception as e:
                return i, None, (scene_dir.name, e)

        n_workers = min(self.index_workers, len(scene_dirs))
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_index_one, (i, d))
                    for i, d in enumerate(scene_dirs)
                ]
                for fut in as_completed(futures):
                    i, rec, err = fut.result()
                    if err is not None:
                        name, e = err
                        if self.strict:
                            raise e
                        skipped.append(f"{name}: {e}")
                        if self.verbose:
                            print(f"[WARN] skip scene {name}: {e}")
                    elif rec is not None:
                        results[i] = rec
        else:
            for i, scene_dir in enumerate(scene_dirs):
                i, rec, err = _index_one((i, scene_dir))
                if err is not None:
                    name, e = err
                    if self.strict:
                        raise e
                    skipped.append(f"{name}: {e}")
                    if self.verbose:
                        print(f"[WARN] skip scene {name}: {e}")
                elif rec is not None:
                    results[i] = rec

        valid_records = [r for r in results if r is not None]

        if self.verbose and skipped:
            print(f"[PointOdysseyAdapter] skipped {len(skipped)} scenes in non-strict mode.")

        return valid_records

    def _index_scene(self, scene_dir: Path) -> Optional[SequenceRecord]:
        fast_dir = scene_dir / FAST_ANNOTATION_DIRNAME
        if fast_dir.is_dir():
            return self._index_scene_fast(scene_dir, fast_dir)

        if self.fast_root is not None:
            external_fast_dir = self.fast_root / self.split / scene_dir.name / FAST_ANNOTATION_DIRNAME
            if external_fast_dir.is_dir():
                return self._index_scene_fast(scene_dir, external_fast_dir)

        return self._index_scene_original(scene_dir)

    def _index_scene_fast(self, scene_dir: Path, fast_dir: Path) -> Optional[SequenceRecord]:
        """Index a PointOdyssey_fast sequence (all data in anno_fast/).

        Key optimizations vs naive approach:
        - Single os.listdir() replaces N individual stat() calls
        - Frame count from numpy file header (~128 bytes) — no mmap, no JSON parse
        - Path lists generated from known PointOdyssey naming convention
        - Image size from the tiny motion-boundary meta JSON
        """
        # Single directory scan → O(1) set lookups replace O(N) stat() calls
        present = {p.name for p in fast_dir.iterdir()}

        pack_meta_path = fast_dir / FAST_ENCODED_META_FILENAME

        # Frame count: read only the numpy file header (128–256 bytes, no data load)
        num_frames = 0
        trajs_npy_name = "trajs_2d.npy"
        if trajs_npy_name in present:
            with open(fast_dir / trajs_npy_name, "rb") as _f:
                try:
                    _ver = _npy_fmt.read_magic(_f)
                    _header_reader = (
                        _npy_fmt.read_array_header_1_0 if _ver == (1, 0)
                        else _npy_fmt.read_array_header_2_0
                    )
                    _shape, _, _ = _header_reader(_f)
                    num_frames = int(_shape[0])
                except Exception:
                    pass
        if num_frames == 0 and pack_meta_path.name in present:
            with open(pack_meta_path, "r") as _f:
                num_frames = int(json.load(_f).get("frames", 0))
        if num_frames == 0:
            return None

        # PointOdyssey canonical suffixes — avoids reading the large manifest JSON
        depth_suffix = ".png"
        normal_suffix = ".jpg"

        # Generate path lists from known naming convention (no I/O)
        # Generate path lists as strings — 5× cheaper than pathlib Path division,
        # since fast sequences load frames via encoded cache anyway (paths are labels only).
        _sd = str(scene_dir)
        rgb_paths = [f"{_sd}/rgbs/rgb_{i:05d}.jpg" for i in range(num_frames)]
        depth_paths = [f"{_sd}/depths/depth_{i:05d}{depth_suffix}" for i in range(num_frames)]
        normal_paths = [f"{_sd}/normals/normal_{i:05d}{normal_suffix}" for i in range(num_frames)]

        # Image size from the tiny motion-boundary meta JSON (has height/width)
        image_size: Optional[tuple[int, int]] = None
        for stride in (1, 2, 3, 4):
            mb_name = f"motion_boundary_stride_{stride:02d}_meta.json"
            if mb_name in present:
                with open(fast_dir / mb_name, "r") as f:
                    mb = json.load(f)
                image_size = (int(mb["height"]), int(mb["width"]))
                break

        # Fast annotation .npy files (loaded on demand with mmap)
        fast_anno_paths: dict[str, Path] = {}
        for key in FAST_REQUIRED_ANNO_FILES:
            fname = f"{key}.npy"
            if fname in present:
                fast_anno_paths[key] = fast_dir / fname
        if "visibs.npy" in present:
            fast_anno_paths["visibs"] = fast_dir / "visibs.npy"

        # Encoded frame cache (set membership, no stat() calls)
        encoded_cache_paths: Optional[dict[str, Path]] = None
        enc_names = {
            "rgb_bin_path": FAST_ENCODED_RGB_BIN,
            "rgb_offsets_path": FAST_ENCODED_RGB_OFFSETS,
            "depth_bin_path": FAST_ENCODED_DEPTH_BIN,
            "depth_offsets_path": FAST_ENCODED_DEPTH_OFFSETS,
            "normal_bin_path": FAST_ENCODED_NORMAL_BIN,
            "normal_offsets_path": FAST_ENCODED_NORMAL_OFFSETS,
            "normal_valids_path": FAST_ENCODED_NORMAL_VALIDS,
        }
        if all(name in present for name in enc_names.values()):
            encoded_cache_paths = {k: fast_dir / name for k, name in enc_names.items()}
            encoded_cache_paths["meta_path"] = pack_meta_path

        return SequenceRecord(
            dataset_name=self.dataset_name,
            split=self.split,
            sequence_name=scene_dir.name,
            sequence_root=scene_dir,
            rgb_paths=rgb_paths,
            depth_paths=depth_paths,
            normal_paths=normal_paths,
            mask_paths=None,
            anno_path=(scene_dir / "anno.npz") if (scene_dir / "anno.npz").exists() else None,
            info_path=(scene_dir / "info.npz") if (scene_dir / "info.npz").exists() else None,
            scene_info_path=(scene_dir / "scene_info.json") if (scene_dir / "scene_info.json").exists() else None,
            num_frames=num_frames,
            image_size=image_size,
            fast_dir=fast_dir,
            fast_anno_paths=fast_anno_paths if fast_anno_paths else None,
            encoded_cache_paths=encoded_cache_paths,
        )

    def _index_scene_original(self, scene_dir: Path) -> Optional[SequenceRecord]:
        """Index a classic PointOdyssey sequence (rgbs/ depths/ normals/ subdirs)."""
        from PIL import Image

        rgb_dir = scene_dir / "rgbs"
        if not rgb_dir.exists():
            return None

        rgb_files = self._sorted_files(rgb_dir)
        if len(rgb_files) == 0:
            return None

        depth_files = self._sorted_files(scene_dir / "depths") if (scene_dir / "depths").exists() else None
        normal_files = self._sorted_files(scene_dir / "normals") if (scene_dir / "normals").exists() else None
        mask_files = self._sorted_files(scene_dir / "masks") if (scene_dir / "masks").exists() else None

        rgb_map = self._build_frame_map(rgb_files, scene_dir=scene_dir, modal_name="rgbs")
        depth_map = (
            self._build_frame_map(depth_files, scene_dir=scene_dir, modal_name="depths")
            if depth_files is not None
            else None
        )
        normal_map = (
            self._build_frame_map(normal_files, scene_dir=scene_dir, modal_name="normals")
            if normal_files is not None
            else None
        )
        mask_map = (
            self._build_frame_map(mask_files, scene_dir=scene_dir, modal_name="masks")
            if mask_files is not None
            else None
        )

        frame_ids = sorted(rgb_map.keys(), key=lambda x: int(x))

        rgb_paths = [rgb_map[fid] for fid in frame_ids]
        depth_paths = self._align_optional_modal_paths(
            scene_dir=scene_dir, modal_name="depths",
            rgb_map=rgb_map, modal_map=depth_map, frame_ids=frame_ids,
        )
        normal_paths = self._align_optional_modal_paths(
            scene_dir=scene_dir, modal_name="normals",
            rgb_map=rgb_map, modal_map=normal_map, frame_ids=frame_ids,
        )
        mask_paths = self._align_optional_modal_paths(
            scene_dir=scene_dir, modal_name="masks",
            rgb_map=rgb_map, modal_map=mask_map, frame_ids=frame_ids,
        )

        with Image.open(rgb_paths[0]) as _img:
            image_size = (_img.height, _img.width)

        return SequenceRecord(
            dataset_name=self.dataset_name,
            split=self.split,
            sequence_name=scene_dir.name,
            sequence_root=scene_dir,
            rgb_paths=rgb_paths,
            depth_paths=depth_paths,
            normal_paths=normal_paths,
            mask_paths=mask_paths,
            anno_path=(scene_dir / "anno.npz") if (scene_dir / "anno.npz").exists() else None,
            info_path=(scene_dir / "info.npz") if (scene_dir / "info.npz").exists() else None,
            scene_info_path=(scene_dir / "scene_info.json") if (scene_dir / "scene_info.json").exists() else None,
            num_frames=len(rgb_paths),
            image_size=image_size,
        )

    # ------------------------------------------------------------------ #
    #  Annotation loading                                                  #
    # ------------------------------------------------------------------ #

    def _load_anno(
        self,
        sequence_name: str,
        frame_indices: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Load annotation arrays for *sequence_name*.

        Parameters
        ----------
        frame_indices:
            If provided (1-D int64 array), per-frame arrays are sliced to
            exactly these positions before returning.  All three backing
            formats (fast_anno mmap, .h5, .npz) return pre-sliced arrays so
            the caller can unconditionally index with ``[:]``.
            Pass ``None`` to load all frames (used by sanity_check /
            probe_sequence_meta).
        """
        r = self.get_record(sequence_name)
        idx: list[int] | None = (
            [int(i) for i in frame_indices] if frame_indices is not None else None
        )

        # Fast path: individual .npy files with memory-mapped lazy loading.
        if r.fast_anno_paths:
            anno: dict[str, Any] = {}
            for key in FAST_REQUIRED_ANNO_FILES:
                if key in r.fast_anno_paths:
                    arr = np.load(r.fast_anno_paths[key], mmap_mode="r", allow_pickle=False)
                    if idx is not None and arr.ndim >= 1:
                        anno[key] = arr[idx]
                    elif idx is not None:
                        anno[key] = arr[()]
                    else:
                        anno[key] = arr
            if "visibs" in r.fast_anno_paths:
                arr = np.load(r.fast_anno_paths["visibs"], mmap_mode="r", allow_pickle=False)
                if idx is not None and arr.ndim >= 1:
                    anno["visibs"] = arr[idx]
                elif idx is not None:
                    anno["visibs"] = arr[()]
                else:
                    anno["visibs"] = arr
            elif "valids" in anno:
                anno["visibs"] = anno["valids"]
            return anno

        if r.anno_path is None:
            raise FileNotFoundError(f"Missing anno.npz for sequence: {sequence_name}")

        # HDF5 chunked format: O(frames) random access vs full zlib decompress for npz.
        h5_path = r.anno_path.with_suffix(".h5")
        if h5_path.exists():
            self._last_anno_range_stats = {}
            if idx is not None and self._should_use_anno_local_h5_cache(h5_path):
                try:
                    local_h5_path, was_hit = self._ensure_anno_local_h5_cache(h5_path)
                    import h5py
                    with h5py.File(local_h5_path, "r") as f:
                        anno = h5_read_frame_slice(f, idx)
                    self._last_anno_range_stats = {
                        "cache_hits": 1 if was_hit else 0,
                        "cache_misses": 0 if was_hit else 1,
                        "range_tasks": 0,
                        "range_bytes": 0,
                        "local_h5": 1,
                    }
                except Exception:
                    if self.anno_read_mode in {"local_h5", "local_cache"}:
                        raise
                    anno = self._load_anno_h5_with_fallback(sequence_name, h5_path, idx)
            elif idx is not None and self._should_use_anno_cos_range(h5_path):
                anno = self._load_anno_h5_with_fallback(sequence_name, h5_path, idx)
            elif idx is not None and self.anno_frame_cache_dir is not None:
                anno = self._load_anno_from_frame_cache(sequence_name, h5_path, idx)
            else:
                import h5py
                with h5py.File(h5_path, "r") as f:
                    if idx is not None:
                        anno = h5_read_frame_slice(f, idx)
                    else:
                        anno = {k: f[k][()] for k in f.keys()}
        else:
            self._last_anno_range_stats = {}
            z = np.load(r.anno_path, allow_pickle=True)
            if idx is not None:
                idx_np = np.asarray(idx)
                anno = {
                    k: (z[k][idx_np] if z[k].ndim >= 1 and z[k].shape[0] > 1 else z[k][()])
                    for k in z.files
                }
            else:
                anno = {k: z[k] for k in z.files}

        return anno

    def _load_anno_h5_with_fallback(
        self,
        sequence_name: str,
        h5_path: Path,
        idx: list[int],
    ) -> dict[str, Any]:
        try:
            return self._load_anno_h5_cos_range(h5_path, idx)
        except Exception:
            if self.anno_read_mode in {"cos_range", "range"}:
                raise
            if self.anno_frame_cache_dir is not None:
                return self._load_anno_from_frame_cache(sequence_name, h5_path, idx)
            import h5py
            with h5py.File(h5_path, "r") as f:
                return h5_read_frame_slice(f, idx)

    def _should_use_anno_local_h5_cache(self, h5_path: Path) -> bool:
        mode = self.anno_read_mode
        if mode in {"h5py", "direct", "npz", "cos_range", "range"}:
            return False
        if mode not in {"auto", "local_h5", "local_cache"}:
            return False
        if mode == "auto" and not self._path_is_under_anno_cos_mount(h5_path):
            return False
        return True

    def _anno_local_h5_cache_path(self, h5_path: Path) -> Optional[Path]:
        raw = os.getenv("D4RT_H5_RANGE_CACHE_ROOT", "").strip()
        if not raw:
            default = Path("/tmp/d4rt_sample_stage/shared_raw_cache/data")
            if default.exists():
                raw = str(default)
        if not raw:
            return None
        root = Path(raw) / ".d4rt_pointodyssey_anno_h5"
        cache_key = (
            self._anno_cos_key(h5_path)
            if self._path_is_under_anno_cos_mount(h5_path)
            else str(h5_path)
        )
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", h5_path.parent.name)[:80]
        return root / f"{safe_name}_{digest}.h5"

    def _ensure_anno_local_h5_cache(self, h5_path: Path) -> tuple[Path, bool]:
        cache_path = self._anno_local_h5_cache_path(h5_path)
        if cache_path is None:
            raise RuntimeError("D4RT_H5_RANGE_CACHE_ROOT is not configured")
        if cache_path.is_file():
            try:
                os.utime(cache_path, None)
            except OSError:
                pass
            return cache_path, True

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with open(lock_path, "a+b") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                if cache_path.is_file():
                    try:
                        os.utime(cache_path, None)
                    except OSError:
                        pass
                    return cache_path, True
                self._download_anno_h5_to_cache(h5_path, cache_path)
                self._evict_anno_local_h5_cache(keep_path=cache_path)
                return cache_path, False
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _evict_anno_local_h5_cache(self, keep_path: Optional[Path] = None) -> None:
        max_bytes = getattr(self, "anno_local_cache_max_bytes", 0)
        if max_bytes <= 0:
            return
        cache_dir = self._anno_local_h5_cache_path(Path("/tmp/dummy/anno.h5"))
        if cache_dir is None:
            return
        cache_root = cache_dir.parent
        lock_path = cache_root / ".eviction.lock"
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "a+b") as lock_f:
                try:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
                try:
                    entries: list[tuple[float, int, Path]] = []
                    total_bytes = 0
                    for path in cache_root.glob("*.h5"):
                        if keep_path is not None and path == keep_path:
                            continue
                        try:
                            stat = path.stat()
                        except OSError:
                            continue
                        entries.append((stat.st_mtime, stat.st_size, path))
                        total_bytes += stat.st_size
                    if keep_path is not None:
                        try:
                            total_bytes += keep_path.stat().st_size
                        except OSError:
                            pass
                    if total_bytes <= max_bytes:
                        return
                    target_bytes = int(max_bytes * 0.9)
                    entries.sort(key=lambda item: item[0])
                    for _, size, path in entries:
                        if total_bytes <= target_bytes:
                            break
                        path.unlink(missing_ok=True)
                        total_bytes -= size
                finally:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except OSError:
            return

    def _download_anno_h5_to_cache(self, h5_path: Path, cache_path: Path) -> None:
        tmp_path = cache_path.with_name(
            f".{cache_path.name}.part.{os.getpid()}.{threading.get_ident()}"
        )
        last_exc: Optional[BaseException] = None
        if self._path_is_under_anno_cos_mount(h5_path):
            cos_key = self._anno_cos_key(h5_path)
            for attempt in range(self.anno_cos_range_retries + 1):
                try:
                    resp = self._get_anno_cos_client().get_object(
                        Bucket=self.anno_cos_bucket,
                        Key=cos_key,
                    )
                    stream = resp["Body"].get_raw_stream()
                    with open(tmp_path, "wb") as f:
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    os.replace(tmp_path, cache_path)
                    return
                except BaseException as exc:
                    last_exc = exc
                    tmp_path.unlink(missing_ok=True)
                    if attempt < self.anno_cos_range_retries:
                        time.sleep(min(2.0, 0.25 * (2 ** attempt)))
                        continue
                    raise
                finally:
                    tmp_path.unlink(missing_ok=True)
        else:
            try:
                with open(h5_path, "rb") as src, open(tmp_path, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                os.replace(tmp_path, cache_path)
                return
            finally:
                tmp_path.unlink(missing_ok=True)
        if last_exc is not None:
            raise last_exc

    def _path_is_under_anno_cos_mount(self, path: Path) -> bool:
        mount = str(self.anno_cos_mount_root).rstrip("/") + "/"
        path_str = str(path)
        return path_str == str(self.anno_cos_mount_root) or path_str.startswith(mount)

    def _should_use_anno_cos_range(self, h5_path: Path) -> bool:
        mode = self.anno_read_mode
        if mode in {"h5py", "direct", "npz", "local_h5", "local_cache"}:
            return False
        if mode not in {"auto", "cos_range", "range"}:
            return False
        if mode == "auto" and not self._path_is_under_anno_cos_mount(h5_path):
            return False
        return True

    def _get_anno_cos_client(self) -> Any:
        client = getattr(self._anno_cos_tls, "client", None)
        if client is None:
            from qcloud_cos import CosConfig, CosS3Client

            parts = Path(self.anno_cos_passwd_file).read_text().strip().split(":")
            if len(parts) == 2:
                secret_id, secret_key = parts
            elif len(parts) == 3:
                _bucket, secret_id, secret_key = parts
            else:
                raise ValueError(
                    "Unsupported COS passwd file format: "
                    f"{self.anno_cos_passwd_file}"
                )
            config = CosConfig(
                Region=self.anno_cos_region,
                SecretId=secret_id,
                SecretKey=secret_key,
                Scheme="https",
                Timeout=self.anno_cos_timeout_s,
            )
            client = CosS3Client(config)
            self._anno_cos_tls.client = client
        return client

    def _anno_cos_key(self, path: Path) -> str:
        try:
            return path.relative_to(self.anno_cos_mount_root).as_posix()
        except ValueError:
            mount = str(self.anno_cos_mount_root).rstrip("/") + "/"
            path_str = str(path)
            if path_str.startswith(mount):
                return path_str[len(mount):]
            raise

    def _read_anno_cos_range(self, cos_key: str, start: int, end: int) -> bytes:
        range_header = f"bytes={start}-{end - 1}"
        last_exc: Optional[BaseException] = None
        for attempt in range(self.anno_cos_range_retries + 1):
            try:
                resp = self._get_anno_cos_client().get_object(
                    Bucket=self.anno_cos_bucket,
                    Key=cos_key,
                    Range=range_header,
                )
                return resp["Body"].get_raw_stream().read()
            except BaseException as exc:
                last_exc = exc
                if attempt < self.anno_cos_range_retries:
                    time.sleep(min(2.0, 0.25 * (2 ** attempt)))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise IOError(f"COS range read failed: {cos_key} {range_header}")

    def _anno_cos_object_size(self, cos_key: str) -> int:
        try:
            resp = self._get_anno_cos_client().head_object(
                Bucket=self.anno_cos_bucket,
                Key=cos_key,
            )
            for key in ("Content-Length", "content-length", "ContentLength"):
                if key in resp:
                    return int(resp[key])
        except Exception:
            pass

        resp = self._get_anno_cos_client().get_object(
            Bucket=self.anno_cos_bucket,
            Key=cos_key,
            Range="bytes=0-0",
        )
        try:
            resp["Body"].get_raw_stream().read()
        finally:
            pass
        content_range = (
            resp.get("Content-Range")
            or resp.get("content-range")
            or resp.get("ContentRange")
            or ""
        )
        if "/" in content_range:
            return int(str(content_range).rsplit("/", 1)[1])
        for key in ("Content-Length", "content-length", "ContentLength"):
            if key in resp:
                return int(resp[key])
        raise IOError(f"Could not determine COS object size for {cos_key}")

    @contextlib.contextmanager
    def _open_anno_h5_metadata(self, h5_path: Path) -> Any:
        import h5py

        if self._path_is_under_anno_cos_mount(h5_path):
            cos_key = self._anno_cos_key(h5_path)
            object_size = self._anno_cos_object_size(cos_key)
            block_mb = float(os.getenv("POINTODYSSEY_ANNO_METADATA_RANGE_BLOCK_MB", "4"))
            cache_blocks = int(os.getenv("POINTODYSSEY_ANNO_METADATA_RANGE_CACHE_BLOCKS", "32"))
            range_file = _CosRangeFile(
                size=object_size,
                read_range=lambda start, end: self._read_anno_cos_range(
                    cos_key,
                    int(start),
                    int(end),
                ),
                block_size=int(max(0.25, block_mb) * 1024**2),
                max_blocks=cache_blocks,
            )
            try:
                with h5py.File(range_file, "r") as f:
                    yield f
            finally:
                range_file.close()
            return

        with h5py.File(h5_path, "r") as f:
            yield f

    def _resolve_anno_h5_index_cache_root(self) -> Optional[Path]:
        raw = os.getenv("D4RT_H5_RANGE_CACHE_ROOT", "").strip()
        if not raw:
            default = Path("/tmp/d4rt_sample_stage/shared_raw_cache/data")
            if default.exists():
                raw = str(default)
        if not raw:
            return None
        root = Path(raw) / ".d4rt_h5_chunk_indexes" / "pointodyssey"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return root

    def _resolve_h5_range_cache_root(self) -> Optional[Path]:
        raw = os.getenv("D4RT_H5_RANGE_CACHE_ROOT", "").strip()
        if not raw:
            default = Path("/tmp/d4rt_sample_stage/shared_raw_cache/data")
            if default.exists():
                raw = str(default)
        if not raw:
            return None
        root = Path(raw) / ".d4rt_h5_range_chunks"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return root

    def _anno_h5_index_cache_path(self, h5_path: Path) -> Optional[Path]:
        root = getattr(self, "_anno_h5_index_cache_root", None)
        if root is None:
            return None
        cache_key = (
            self._anno_cos_key(h5_path)
            if self._path_is_under_anno_cos_mount(h5_path)
            else str(h5_path)
        )
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", h5_path.parent.name)[:80]
        return root / f"{safe_name}_{digest}.pkl"

    def _load_anno_h5_chunk_index(
        self,
        h5_path: Path,
        required_frames: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        cache_key = str(h5_path)
        cached = self._anno_h5_chunk_index_cache.get(cache_key)
        if cached is not None:
            if required_frames:
                self._ensure_anno_h5_chunk_index_frames(h5_path, cached, required_frames)
            return cached

        cache_path = self._anno_h5_index_cache_path(h5_path)
        if cache_path is not None and cache_path.is_file():
            try:
                with open(cache_path, "rb") as f:
                    index = pickle.load(f)
                if required_frames:
                    self._ensure_anno_h5_chunk_index_frames(h5_path, index, required_frames)
                self._anno_h5_chunk_index_cache[cache_key] = index
                return index
            except (OSError, ValueError, KeyError, EOFError, pickle.PickleError):
                cache_path.unlink(missing_ok=True)

        index = self._build_anno_h5_chunk_index(h5_path, required_frames=required_frames)
        self._anno_h5_chunk_index_cache[cache_key] = index

        if cache_path is not None:
            tmp_path = cache_path.with_name(
                f".{cache_path.name}.part.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as f:
                    pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, cache_path)
            except OSError:
                pass
            finally:
                tmp_path.unlink(missing_ok=True)

        return index

    def _ensure_anno_h5_chunk_index_frames(
        self,
        h5_path: Path,
        index: dict[str, Any],
        required_frames: list[int],
    ) -> None:
        keys = ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")
        needs_update = False
        for key in keys:
            entry = index.get(key)
            if not isinstance(entry, dict) or "offsets" not in entry:
                needs_update = True
                break
            offsets = entry["offsets"]
            if any(
                int(frame_idx) >= len(offsets) or offsets[int(frame_idx)] is None
                for frame_idx in required_frames
            ):
                needs_update = True
                break
        if not needs_update:
            return

        frames = self._expand_anno_h5_index_frames(h5_path, required_frames)
        changed = False
        with self._open_anno_h5_metadata(h5_path) as f:
            for key in keys:
                if key not in f:
                    continue
                ds = f[key]
                entry = index.get(key)
                if ds.chunks is None or ds.ndim < 1:
                    if entry is None:
                        index[key] = {
                            "scalar": True,
                            "value": ds[()],
                            "dtype": ds.dtype.str,
                            "shape": tuple(int(v) for v in ds.shape),
                        }
                        changed = True
                    continue
                if entry is None:
                    entry = {
                        "offsets": [None] * int(ds.shape[0]),
                        "filter_masks": [0] * int(ds.shape[0]),
                        "dtype": ds.dtype.str,
                        "chunk_shape": tuple(int(v) for v in ds.chunks),
                        "shape": tuple(int(v) for v in ds.shape),
                        "compression": ds.compression,
                    }
                    index[key] = entry
                    changed = True
                offsets = entry.get("offsets")
                filter_masks = entry.get("filter_masks")
                if offsets is None or filter_masks is None:
                    continue
                coord_tail = tuple(0 for _ in range(ds.ndim - 1))
                for frame_idx in frames:
                    if frame_idx < 0 or frame_idx >= len(offsets) or offsets[frame_idx] is not None:
                        continue
                    info = ds.id.get_chunk_info_by_coord((frame_idx,) + coord_tail)
                    offsets[frame_idx] = (int(info.byte_offset), int(info.size))
                    filter_masks[frame_idx] = int(info.filter_mask)
                    changed = True
        if changed:
            cache_path = self._anno_h5_index_cache_path(h5_path)
            if cache_path is not None:
                tmp_path = cache_path.with_name(
                    f".{cache_path.name}.part.{os.getpid()}.{threading.get_ident()}"
                )
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(tmp_path, "wb") as f:
                        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp_path, cache_path)
                except OSError:
                    pass
                finally:
                    tmp_path.unlink(missing_ok=True)

    def _expand_anno_h5_index_frames(
        self,
        h5_path: Path,
        required_frames: list[int],
    ) -> list[int]:
        radius = self.anno_index_window_radius
        if radius <= 0:
            return sorted(set(int(i) for i in required_frames))
        # Use the sequence size if already known; otherwise fall back to the
        # requested window only.
        total = None
        try:
            cache_path = self._anno_h5_index_cache_path(h5_path)
            if cache_path is not None and cache_path.is_file():
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                for entry in cached.values():
                    if isinstance(entry, dict) and "shape" in entry and entry["shape"]:
                        total = int(entry["shape"][0])
                        break
        except Exception:
            total = None
        frames: set[int] = set()
        for frame_idx in required_frames:
            start = max(0, int(frame_idx) - radius)
            end = int(frame_idx) + radius
            if total is not None:
                end = min(total - 1, end)
            frames.update(range(start, end + 1))
        return sorted(frames)

    def _build_anno_h5_chunk_index(
        self,
        h5_path: Path,
        required_frames: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        keys = ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")
        index: dict[str, Any] = {"_source": str(h5_path), "_kind": "pointodyssey_anno_h5"}
        with self._open_anno_h5_metadata(h5_path) as f:
            for key in keys:
                if key not in f:
                    continue
                ds = f[key]
                if ds.chunks is None or ds.ndim < 1:
                    index[key] = {
                        "scalar": True,
                        "value": ds[()],
                        "dtype": ds.dtype.str,
                        "shape": tuple(int(v) for v in ds.shape),
                    }
                    continue
                offsets: list[Optional[tuple[int, int]]] = [None] * int(ds.shape[0])
                filter_masks = [0] * int(ds.shape[0])
                coord_tail = tuple(0 for _ in range(ds.ndim - 1))
                frames = (
                    self._expand_anno_h5_index_frames(h5_path, required_frames)
                    if required_frames is not None
                    else list(range(int(ds.shape[0])))
                )
                for frame_idx in frames:
                    if frame_idx < 0 or frame_idx >= len(offsets):
                        continue
                    info = ds.id.get_chunk_info_by_coord((frame_idx,) + coord_tail)
                    offsets[frame_idx] = (int(info.byte_offset), int(info.size))
                    filter_masks[frame_idx] = int(info.filter_mask)
                index[key] = {
                    "offsets": offsets,
                    "filter_masks": filter_masks,
                    "dtype": ds.dtype.str,
                    "chunk_shape": tuple(int(v) for v in ds.chunks),
                    "shape": tuple(int(v) for v in ds.shape),
                    "compression": ds.compression,
                }
        return index

    def _h5_chunk_cache_path(
        self,
        cos_key: str,
        key: str,
        frame_idx: int,
        offset: int,
        size: int,
    ) -> Optional[Path]:
        root = getattr(self, "_h5_range_cache_root", None)
        if root is None:
            return None
        digest = hashlib.sha1(f"{cos_key}:{key}".encode("utf-8")).hexdigest()
        return root / digest / f"{int(frame_idx):08d}_{int(offset)}_{int(size)}.chunk"

    def _read_h5_chunk_cache(
        self,
        cos_key: str,
        key: str,
        frame_idx: int,
        offset: int,
        size: int,
    ) -> Optional[bytes]:
        path = self._h5_chunk_cache_path(cos_key, key, frame_idx, offset, size)
        if path is None or not path.is_file():
            return None
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        if len(raw) != int(size):
            path.unlink(missing_ok=True)
            return None
        try:
            os.utime(path, None)
        except OSError:
            pass
        return raw

    def _write_h5_chunk_cache(
        self,
        cos_key: str,
        key: str,
        frame_idx: int,
        offset: int,
        size: int,
        raw: bytes,
    ) -> None:
        path = self._h5_chunk_cache_path(cos_key, key, frame_idx, offset, size)
        if path is None or path.is_file() or len(raw) != int(size):
            return
        tmp: Optional[Path] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.part.{os.getpid()}.{threading.get_ident()}")
            tmp.write_bytes(raw)
            os.replace(tmp, path)
        except OSError:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def _load_anno_h5_cos_range(
        self,
        h5_path: Path,
        frame_indices: list[int],
    ) -> dict[str, Any]:
        sorted_idx = sorted(set(int(i) for i in frame_indices))
        if not sorted_idx:
            raise ValueError("frame_indices is empty")
        index = self._load_anno_h5_chunk_index(h5_path, required_frames=sorted_idx)
        required_keys = ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")
        keys = [key for key in required_keys if key in index]
        if not keys:
            raise KeyError(f"No supported PointOdyssey anno keys in h5 index: {h5_path}")

        tasks: list[dict[str, Any]] = []
        chunks_by_key: dict[str, dict[int, np.ndarray]] = {key: {} for key in keys}
        entries: dict[str, dict[str, Any]] = {}
        range_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "range_tasks": 0,
            "range_bytes": 0,
            "index_cache_hit": 1,
        }
        cos_key = self._anno_cos_key(h5_path)

        for key in keys:
            entry = index[key]
            entries[key] = entry
            self._validate_anno_h5_range_entry(key, entry, sorted_idx)
            dtype = np.dtype(entry["dtype"])
            chunk_shape = tuple(int(v) for v in entry["chunk_shape"])
            for start, end, chunks in self._merge_anno_h5_range_chunks(entry, sorted_idx):
                missing_chunks: list[tuple[int, int, int, int]] = []
                for frame_idx, offset, size, filter_mask in chunks:
                    raw = self._read_h5_chunk_cache(cos_key, key, frame_idx, offset, size)
                    if raw is None:
                        missing_chunks.append((frame_idx, offset, size, filter_mask))
                        range_stats["cache_misses"] += 1
                        continue
                    chunks_by_key[key][int(frame_idx)] = self._decode_anno_h5_chunk(
                        raw,
                        entry=entry,
                        dtype=dtype,
                        chunk_shape=chunk_shape,
                        filter_mask=int(filter_mask),
                    )[0].copy()
                    range_stats["cache_hits"] += 1
                if missing_chunks:
                    task_start = min(int(offset) for _, offset, _, _ in missing_chunks)
                    task_end = max(
                        int(offset) + int(size) for _, offset, size, _ in missing_chunks
                    )
                    tasks.append({
                        "key": key,
                        "start": task_start,
                        "end": task_end,
                        "chunks": missing_chunks,
                    })
                    range_stats["range_tasks"] += 1
                    range_stats["range_bytes"] += task_end - task_start

        def fetch_task(task: dict[str, Any]) -> dict[str, Any]:
            data = self._read_anno_cos_range(cos_key, int(task["start"]), int(task["end"]))
            return {**task, "data": data}

        max_workers = min(self.anno_cos_range_workers, max(1, len(tasks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_task, task) for task in tasks]
            for future in as_completed(futures):
                task = future.result()
                entry = entries[task["key"]]
                dtype = np.dtype(entry["dtype"])
                chunk_shape = tuple(int(v) for v in entry["chunk_shape"])
                start = int(task["start"])
                data = task["data"]
                for frame_idx, offset, size, filter_mask in task["chunks"]:
                    rel = int(offset) - start
                    raw = data[rel:rel + int(size)]
                    if len(raw) != int(size):
                        raise IOError(
                            f"Short COS range read for {task['key']} frame {frame_idx}: "
                            f"got {len(raw)} bytes, expected {size}"
                        )
                    chunks_by_key[task["key"]][int(frame_idx)] = self._decode_anno_h5_chunk(
                        raw,
                        entry=entry,
                        dtype=dtype,
                        chunk_shape=chunk_shape,
                        filter_mask=int(filter_mask),
                    )[0].copy()
                    self._write_h5_chunk_cache(
                        cos_key,
                        task["key"],
                        frame_idx,
                        offset,
                        size,
                        raw,
                    )
        self._last_anno_range_stats = range_stats

        pos = {frame_idx: idx for idx, frame_idx in enumerate(sorted_idx)}
        reorder = [pos[int(frame_idx)] for frame_idx in frame_indices]
        result: dict[str, Any] = {}
        for key in keys:
            arr_sorted = np.stack(
                [chunks_by_key[key][frame_idx] for frame_idx in sorted_idx],
                axis=0,
            )
            result[key] = arr_sorted[reorder]
        return result

    def _validate_anno_h5_range_entry(
        self,
        key: str,
        entry: dict[str, Any],
        sorted_idx: list[int],
    ) -> None:
        compression = entry.get("compression")
        if compression not in (None, "None", "lzf"):
            raise RuntimeError(f"Unsupported compressed h5 chunks for {key}: {compression}")
        chunk_shape = tuple(int(v) for v in entry["chunk_shape"])
        if not chunk_shape or chunk_shape[0] != 1:
            raise RuntimeError(f"Unsupported h5 chunk_shape for {key}: {chunk_shape}")
        offsets = entry["offsets"]
        max_idx = sorted_idx[-1]
        if max_idx >= len(offsets):
            raise IndexError(f"Frame {max_idx} out of range for {key} ({len(offsets)})")
        if any(offsets[int(frame_idx)] is None for frame_idx in sorted_idx):
            raise RuntimeError(f"Missing h5 chunk offsets for {key}")

    def _merge_anno_h5_range_chunks(
        self,
        entry: dict[str, Any],
        sorted_idx: list[int],
    ) -> list[tuple[int, int, list[tuple[int, int, int, int]]]]:
        offsets = entry["offsets"]
        filter_masks = entry.get("filter_masks") or [0] * len(offsets)
        chunks = [
            (
                frame_idx,
                int(offsets[frame_idx][0]),
                int(offsets[frame_idx][1]),
                int(filter_masks[frame_idx]),
            )
            for frame_idx in sorted_idx
        ]
        chunks.sort(key=lambda item: item[1])
        spans: list[tuple[int, int, list[tuple[int, int, int, int]]]] = []
        max_gap = self.anno_cos_range_merge_gap_bytes
        cur_start: Optional[int] = None
        cur_end: Optional[int] = None
        cur_chunks: list[tuple[int, int, int, int]] = []
        for frame_idx, offset, size, filter_mask in chunks:
            end = offset + size
            if cur_start is None or cur_end is None or offset > cur_end + max_gap:
                if cur_start is not None and cur_end is not None:
                    spans.append((cur_start, cur_end, cur_chunks))
                cur_start = offset
                cur_end = end
                cur_chunks = [(frame_idx, offset, size, filter_mask)]
            else:
                cur_end = max(cur_end, end)
                cur_chunks.append((frame_idx, offset, size, filter_mask))
        if cur_start is not None and cur_end is not None:
            spans.append((cur_start, cur_end, cur_chunks))
        return spans

    def _decode_anno_h5_chunk(
        self,
        raw: bytes,
        entry: dict[str, Any],
        dtype: np.dtype,
        chunk_shape: tuple[int, ...],
        filter_mask: int,
    ) -> np.ndarray:
        expected_nbytes = int(np.prod(chunk_shape)) * dtype.itemsize
        compression = entry.get("compression")
        if compression in (None, "None") or (filter_mask & 1):
            payload = raw
        elif compression == "lzf":
            payload = _lzf_decompress(raw, expected_nbytes)
        else:
            raise RuntimeError(f"Unsupported h5 compression: {compression}")
        if len(payload) != expected_nbytes:
            raise IOError(
                f"Decoded h5 chunk has {len(payload)} bytes, expected {expected_nbytes}"
            )
        return np.frombuffer(payload, dtype=dtype).reshape(chunk_shape)

    def _anno_frame_cache_path(self, sequence_name: str, frame_idx: int) -> Path:
        safe_name = sequence_name.replace("/", "__")
        return (
            self.anno_frame_cache_dir
            / self.split
            / safe_name
            / f"{frame_idx:06d}.pkl"
        )

    def _load_cached_anno_frame(
        self,
        sequence_name: str,
        frame_idx: int,
    ) -> Optional[dict[str, np.ndarray]]:
        cache_path = self._anno_frame_cache_path(sequence_name, frame_idx)
        if not cache_path.is_file():
            return None
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except (OSError, ValueError, KeyError, pickle.PickleError, EOFError):
            cache_path.unlink(missing_ok=True)
            return None

    def _write_cached_anno_frame(
        self,
        sequence_name: str,
        frame_idx: int,
        payload: dict[str, np.ndarray],
    ) -> None:
        cache_path = self._anno_frame_cache_path(sequence_name, frame_idx)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(
            f".{cache_path.name}.part.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _load_anno_from_frame_cache(
        self,
        sequence_name: str,
        h5_path: Path,
        idx: list[int],
    ) -> dict[str, Any]:
        frame_payloads: dict[int, dict[str, np.ndarray]] = {}
        missing = []
        for frame_idx in sorted(set(int(i) for i in idx)):
            cached = self._load_cached_anno_frame(sequence_name, frame_idx)
            if cached is None:
                missing.append(frame_idx)
            else:
                frame_payloads[frame_idx] = cached

        if missing:
            import h5py
            with h5py.File(h5_path, "r") as f:
                for frame_idx in missing:
                    payload = {}
                    for key in f.keys():
                        ds = f[key]
                        if ds.ndim >= 1 and ds.shape[0] > 1:
                            if frame_idx >= ds.shape[0]:
                                continue
                            payload[key] = ds[frame_idx]
                        else:
                            payload[key] = ds[()]
                    frame_payloads[frame_idx] = payload
                    self._write_cached_anno_frame(sequence_name, frame_idx, payload)

        first_payload = frame_payloads[int(idx[0])]
        out: dict[str, Any] = {}
        for key, first_value in first_payload.items():
            if getattr(first_value, "ndim", 0) >= 0:
                out[key] = np.stack(
                    [frame_payloads[int(frame_idx)][key] for frame_idx in idx],
                    axis=0,
                )
            else:
                out[key] = first_value
        return out

    def _load_scene_info(self, sequence_name: str) -> Optional[dict[str, Any]]:
        if sequence_name in self._scene_info_cache:
            return self._scene_info_cache[sequence_name]

        r = self.get_record(sequence_name)
        if r.scene_info_path is None:
            self._scene_info_cache[sequence_name] = None
            return None

        if not Path(r.scene_info_path).exists():
            r.scene_info_path = None
            self._scene_info_cache[sequence_name] = None
            return None

        with open(r.scene_info_path, "r") as f:
            meta = json.load(f)

        self._scene_info_cache[sequence_name] = meta
        return meta

    # ------------------------------------------------------------------ #
    #  Encoded frame cache                                                 #
    # ------------------------------------------------------------------ #

    def _get_encoded_cache(self, sequence_name: str, r: SequenceRecord) -> dict[str, Any]:
        """Open (or retrieve cached) memmap handles for a sequence's binary cache."""
        if sequence_name in self._encoded_cache_store:
            return self._encoded_cache_store[sequence_name]

        paths = r.encoded_cache_paths
        depth_suffix = (
            Path(r.depth_paths[0]).suffix.lower() if r.depth_paths else ".png"
        )
        normal_suffix = (
            Path(r.normal_paths[0]).suffix.lower() if r.normal_paths else ".jpg"
        )

        cache: dict[str, Any] = {
            "rgb_bin":      np.memmap(paths["rgb_bin_path"],    mode="r", dtype=np.uint8),
            "rgb_offsets":  np.load(paths["rgb_offsets_path"],  mmap_mode="r", allow_pickle=False),
            "depth_bin":    np.memmap(paths["depth_bin_path"],  mode="r", dtype=np.uint8),
            "depth_offsets": np.load(paths["depth_offsets_path"], mmap_mode="r", allow_pickle=False),
            "normal_bin":   np.memmap(paths["normal_bin_path"], mode="r", dtype=np.uint8),
            "normal_offsets": np.load(paths["normal_offsets_path"], mmap_mode="r", allow_pickle=False),
            "normal_valids": np.load(paths["normal_valids_path"], mmap_mode="r", allow_pickle=False),
            "depth_suffix":  depth_suffix,
            "normal_suffix": normal_suffix,
        }
        self._encoded_cache_store[sequence_name] = cache
        return cache

    @staticmethod
    def _slice_cache_bytes(bin_arr: np.ndarray, offsets: np.ndarray, frame_pos: int) -> np.ndarray:
        start = int(offsets[frame_pos])
        end = int(offsets[frame_pos + 1])
        return np.asarray(bin_arr[start:end], dtype=np.uint8)

    def _decode_rgb_from_cache(self, cache: dict, frame_pos: int) -> np.ndarray:
        encoded = self._slice_cache_bytes(cache["rgb_bin"], cache["rgb_offsets"], frame_pos)
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode RGB frame {frame_pos} from encoded cache")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _decode_depth_from_cache(self, cache: dict, frame_pos: int) -> np.ndarray:
        encoded = self._slice_cache_bytes(cache["depth_bin"], cache["depth_offsets"], frame_pos)
        suffix = cache["depth_suffix"]
        if suffix == ".npy":
            depth = np.load(io.BytesIO(encoded.tobytes()))
        else:
            depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"Failed to decode depth frame {frame_pos} from encoded cache")
        return depth.astype(np.float32)

    def _decode_normal_from_cache(self, cache: dict, frame_pos: int) -> Optional[np.ndarray]:
        is_valid = bool(np.asarray(cache["normal_valids"][frame_pos]).item())
        if not is_valid:
            return None
        encoded = self._slice_cache_bytes(cache["normal_bin"], cache["normal_offsets"], frame_pos)
        if len(encoded) == 0:
            return None
        suffix = cache["normal_suffix"]
        if suffix == ".npy":
            arr = np.load(io.BytesIO(encoded.tobytes())).astype(np.float32)
            return self._decode_normal_map(arr)
        normal = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if normal is None:
            return None
        if normal.ndim == 2:
            return normal.astype(np.float32)
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        return self._decode_normal_map(normal)

    # ------------------------------------------------------------------ #
    #  File-based frame reading (original format fallback)                 #
    # ------------------------------------------------------------------ #

    def _read_rgb(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read RGB image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_depth(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            return arr.astype(np.float32)

        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"Failed to read depth: {path}")
        return depth.astype(np.float32)

    def _read_normal(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            return self._decode_normal_map(arr.astype(np.float32))

        normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if normal is None:
            raise IOError(f"Failed to read normal: {path}")

        if normal.ndim == 2:
            return normal.astype(np.float32)

        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        return self._decode_normal_map(normal)

    def _decode_normal_map(self, normal: np.ndarray) -> np.ndarray:
        """
        Decode PointOdyssey normal maps to unit vectors in [-1, 1].

        Official PointOdyssey export stores Blender normal pass values in
        [-1, 1], converts them to uint16 PNG, and later re-saves those PNGs
        as JPG for the released dataset. The loader therefore needs to invert
        that visualization encoding instead of treating the file as plain RGB.

        Normalization uses einsum for the squared-norm (single pass) and a
        float mask multiply instead of np.where to avoid branched allocations.
        """
        arr = normal.astype(np.float32, copy=False)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return arr

        valid_threshold_sq: float
        if np.issubdtype(normal.dtype, np.integer):
            max_val = 65535.0 if normal.dtype == np.uint16 else 255.0
            arr = arr * (2.0 / max_val) - 1.0
            # JPG re-encoding perturbs true zero-vectors; use a larger threshold.
            valid_threshold_sq = (5e-2) ** 2
        else:
            if arr.size == 0:
                return arr
            if arr.min() >= 0.0 and arr.max() <= 1.0:
                arr = arr * 2.0 - 1.0
            elif arr.min() < -1.0 or arr.max() > 1.0:
                arr = arr * (2.0 / 255.0) - 1.0
            valid_threshold_sq = (1e-6) ** 2

        # Squared L2 norm per pixel — single pass, no keepdims overhead.
        norm_sq = np.einsum("hwc,hwc->hw", arr, arr)  # (H, W)
        safe_norm = np.sqrt(norm_sq.clip(min=valid_threshold_sq))
        # Zero background pixels (norm below threshold) via float multiply.
        valid = (norm_sq >= valid_threshold_sq).astype(np.float32)
        return (arr / safe_norm[:, :, np.newaxis] * valid[:, :, np.newaxis]).astype(
            np.float32, copy=False
        )

    def _convert_released_normals_to_camera_space(
        self,
        normals: list[np.ndarray],
        extrinsics: np.ndarray,
    ) -> list[np.ndarray]:
        """Rotate released PointOdyssey normals into the released camera space.

        The released normal images are copied from the rendered Blender normal
        pass after visualization encoding, while the released extrinsics are
        transformed by the official export code using R3 @ R2 @ RT @ R1.
        After decoding a normal map back to unit vectors, each frame therefore
        needs the per-frame rotation R_saved @ R1^T to align with the released
        world-to-camera extrinsic convention used elsewhere in D4RT.

        The input normals are already unit vectors (_decode_normal_map).
        A rotation preserves vector length, so renormalization is unnecessary.

        Note: a batched np.stack + reshape + matmul approach is ~4× slower
        than the per-frame loop here because np.stack must allocate a single
        (T, H, W, 3) contiguous buffer (~300 MB for T=48, H=540, W=960),
        which exceeds CPU cache and dominates over Python loop overhead.
        """
        converted: list[np.ndarray] = []
        for normal, extrinsic in zip(normals, extrinsics):
            arr = normal.astype(np.float32, copy=False)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                converted.append(arr)
                continue
            transform = extrinsic[:3, :3].astype(np.float32, copy=False) @ POINTODYSSEY_R1.T
            # (H, W, 3) @ (3, 3).T — rotation preserves unit length; no renorm needed.
            converted.append((arr @ transform.T).astype(np.float32, copy=False))
        return converted

    # ------------------------------------------------------------------ #
    #  Index helpers                                                       #
    # ------------------------------------------------------------------ #

    def _sorted_files(self, d: Path) -> list[Path]:
        files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]

        def sort_key(p: Path):
            fid = self._try_frame_id_from_path(p)
            if fid is not None:
                return (0, int(fid), p.name)
            return (1, p.stem, p.name)

        return sorted(files, key=sort_key)

    def _build_frame_map(
        self,
        files: list[Path],
        scene_dir: Path,
        modal_name: str,
    ) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for p in files:
            fid = self._frame_id_from_path(p)
            if fid in out:
                raise ValueError(
                    f"{scene_dir.name}: duplicate frame id '{fid}' in {modal_name}: "
                    f"{out[fid].name} and {p.name}"
                )
            out[fid] = p
        return out

    def _align_optional_modal_paths(
        self,
        scene_dir: Path,
        modal_name: str,
        rgb_map: dict[str, Path],
        modal_map: Optional[dict[str, Path]],
        frame_ids: list[str],
    ) -> Optional[list[Path]]:
        if modal_map is None:
            return None

        missing = [fid for fid in frame_ids if fid not in modal_map]
        extra = [fid for fid in modal_map.keys() if fid not in rgb_map]

        if missing or extra:
            msg = (
                f"{scene_dir.name}: {modal_name} not aligned with rgb. "
                f"missing={missing[:10]} (count={len(missing)}), "
                f"extra={extra[:10]} (count={len(extra)})"
            )
            if self.strict:
                raise ValueError(msg)
            if self.verbose:
                print(f"[WARN] {msg}")
            return None

        return [modal_map[fid] for fid in frame_ids]

    def _try_frame_id_from_path(self, p: Path) -> Optional[str]:
        m = re.search(r"(\d+)$", p.stem)
        if m is None:
            return None
        return m.group(1)

    def _frame_id_from_path(self, p: Path) -> str:
        fid = self._try_frame_id_from_path(p)
        if fid is None:
            raise ValueError(f"Cannot parse frame id from filename: {p.name}")
        return fid
