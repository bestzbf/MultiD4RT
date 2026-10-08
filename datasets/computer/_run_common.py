"""
Shared precompute logic used by all run_*.py scripts.

For each sequence in an adapter, loads all frames, computes normals
and tracks, and writes a compressed .npz cache file.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from computer.depth_to_normals import compute_normals_sequence
from computer.depth_to_tracks import compute_tracks
from adapters.base import BaseAdapter


def _valid_mask_from_masked_rgb(
    image: np.ndarray,
    *,
    threshold: int,
    erosion_px: int,
) -> np.ndarray:
    """Return an eroded valid-pixel mask from a BlendedMVS *_masked.jpg."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected RGB masked image, got shape {arr.shape}")
    valid = arr.max(axis=-1) > int(threshold)
    erosion_px = max(int(erosion_px), 0)
    if erosion_px > 0:
        import cv2

        kernel = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), dtype=np.uint8)
        valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    return valid


# --------------------------------------------------------------------------- #
# Per-sequence worker (runs in subprocess)
# --------------------------------------------------------------------------- #

def _process_sequence(args: tuple) -> tuple[str, str | None]:
    """
    Process one sequence.  Returns (seq_name, error_message_or_None).
    Must be a top-level function for multiprocessing pickle.
    """
    seq_name, adapter_state, output_root, num_points, overwrite, track_kwargs, compute_normals = args
    track_kwargs = dict(track_kwargs or {})
    source_mask_from_masked_images = bool(
        track_kwargs.pop("source_mask_from_masked_images", False)
    )
    source_mask_threshold = int(track_kwargs.pop("source_mask_threshold", 30))
    source_mask_erosion_px = int(track_kwargs.pop("source_mask_erosion_px", 0))

    out_path = Path(output_root) / seq_name / "precomputed.npz"
    if out_path.exists() and not overwrite:
        return seq_name, None   # already done
    if overwrite:
        h5_path = out_path.with_suffix(".h5")
        if h5_path.exists():
            h5_path.unlink()

    try:
        # Reconstruct adapter in subprocess
        adapter_cls, adapter_kwargs = adapter_state
        adapter = adapter_cls(**adapter_kwargs)

        info = adapter.get_sequence_info(seq_name)
        num_frames = info["num_frames"]

        # Load all frames for this sequence
        all_indices = list(range(num_frames))
        clip = adapter.load_clip(seq_name, all_indices)

        depths     = clip.depths       # list[np.ndarray] [H,W]
        intrinsics = clip.intrinsics   # [T,3,3]
        extrinsics = clip.extrinsics   # [T,4,4] w2c

        if depths is None:
            return seq_name, "no depth"

        H, W = depths[0].shape
        source_valid_masks = None
        if source_mask_from_masked_images:
            if getattr(adapter, "dataset_name", "") != "blendedmvs":
                return seq_name, "source_mask_from_masked_images is only supported for BlendedMVS"
            source_valid_masks = [
                _valid_mask_from_masked_rgb(
                    image,
                    threshold=source_mask_threshold,
                    erosion_px=source_mask_erosion_px,
                )
                for image in clip.images
            ]

        # ---- normals ----
        normals = (
            compute_normals_sequence(depths, intrinsics)   # [T,H,W,3]
            if compute_normals
            else None
        )

        # ---- tracks ----
        tracks = compute_tracks(
            depths=depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            num_points=num_points,
            source_valid_masks=source_valid_masks,
            **track_kwargs,
        )

        # ---- save ----
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "ref_frame": np.array(tracks["ref_frame"], dtype=np.int32),
            "ref_frames": np.asarray(
                tracks.get("ref_frames", np.array([tracks["ref_frame"]], dtype=np.int32)),
                dtype=np.int32,
            ),
            "num_frames": np.array(num_frames, dtype=np.int32),
            "num_points": np.array(tracks["num_points"], dtype=np.int32),
            "track_semantics_version": np.array(
                tracks.get("track_semantics_version", 0),
                dtype=np.int32,
            ),
            "pre_dedup_num_points": np.array(
                tracks.get("pre_dedup_num_points", tracks["num_points"]),
                dtype=np.int32,
            ),
            "num_ref_segments": np.array(
                tracks.get("num_ref_segments", 1),
                dtype=np.int32,
            ),
            "points_per_ref_frame": np.array(
                tracks.get("points_per_ref_frame", 0),
                dtype=np.int32,
            ),
            "ref_frame_stride": np.array(
                tracks.get("ref_frame_stride", 0),
                dtype=np.int32,
            ),
            "world_dedup_voxel_size": np.array(
                tracks.get("world_dedup_voxel_size", 0.0),
                dtype=np.float32,
            ),
            "compact_static_tracks": np.array(
                tracks.get("compact_static_tracks", False),
                dtype=np.bool_,
            ),
            "track_depth_max": np.array(
                tracks.get("track_depth_max", np.nan),
                dtype=np.float32,
            ),
            "track_depth_consistency_thresh": np.array(
                tracks.get("track_depth_consistency_thresh", np.nan),
                dtype=np.float32,
            ),
            "source_depth_sampling_mode": np.array(
                tracks.get("source_depth_sampling_mode", 0),
                dtype=np.int32,
            ),
            "source_depth_bin_edges": np.asarray(
                tracks.get("source_depth_bin_edges", np.empty(0, dtype=np.float32)),
                dtype=np.float32,
            ),
            "source_depth_bin_weights": np.asarray(
                tracks.get("source_depth_bin_weights", np.empty(0, dtype=np.float32)),
                dtype=np.float32,
            ),
            "source_depth_num_bins": np.array(
                tracks.get("source_depth_num_bins", 0),
                dtype=np.int32,
            ),
            "source_mask_from_masked_images": np.array(
                source_mask_from_masked_images,
                dtype=np.bool_,
            ),
            "source_mask_threshold": np.array(
                source_mask_threshold,
                dtype=np.int32,
            ),
            "source_mask_erosion_px": np.array(
                source_mask_erosion_px,
                dtype=np.int32,
            ),
        }
        if normals is not None:
            save_kwargs["normals"] = normals.astype(np.float16)   # [T,H,W,3] half to save space
        for key in (
            "trajs_2d",
            "trajs_3d_world",
            "valids",
            "visibs",
            "static_points_world",
            "static_point_ref_frames",
            "static_point_ref_uv",
            "static_point_ref_depth",
            "static_point_ref_counts",
            "static_points_world_convention",
            "trajs_3d_world_convention",
            "valid_visible_counts",
        ):
            if key in tracks:
                arr = tracks[key]
                if key in {"trajs_2d", "trajs_3d_world", "static_points_world", "static_point_ref_uv", "static_point_ref_depth"}:
                    arr = np.asarray(arr, dtype=np.float32)
                elif key in {"static_point_ref_frames", "static_point_ref_counts", "static_points_world_convention", "trajs_3d_world_convention", "valid_visible_counts"}:
                    arr = np.asarray(arr, dtype=np.int32)
                save_kwargs[key] = arr
        origin_shift = clip.metadata.get("origin_shift")
        if origin_shift is not None:
            origin_shift = np.asarray(origin_shift, dtype=np.float32).reshape(-1)
            if origin_shift.size == 3:
                save_kwargs["origin_shift"] = origin_shift

        np.savez_compressed(out_path, **save_kwargs)
        return seq_name, None

    except Exception:
        return seq_name, traceback.format_exc()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_precompute(
    adapter: BaseAdapter,
    output_root: Path,
    num_points: int = 8000,
    workers: int = 4,
    overwrite: bool = False,
    track_kwargs: dict | None = None,
    compute_normals: bool = True,
) -> None:
    """
    Run precomputation for all sequences in an adapter.

    Args:
        adapter:     Any BaseAdapter instance.
        output_root: Directory where precomputed.npz files will be written.
                     Structure: output_root / <sequence_name> / precomputed.npz
        num_points:  Number of track points per sequence.
        workers:     Number of parallel worker processes.
        overwrite:   Re-compute even if output already exists.
        compute_normals:
            If False, write track-only caches. Useful for diagnostics where
            full-resolution normals would dominate memory and disk.
    """
    sequences = adapter.list_sequences()
    total = len(sequences)
    print(f"[precompute] {adapter.dataset_name}: {total} sequences → {output_root}")

    track_kwargs = dict(track_kwargs or {})

    # Build a picklable description of the adapter so workers can reconstruct it
    adapter_state = _make_adapter_state(adapter)

    job_args = [
        (seq, adapter_state, str(output_root), num_points, overwrite, track_kwargs, compute_normals)
        for seq in sequences
    ]

    done = 0
    failed = []

    if workers <= 1:
        # Single-process mode (easier to debug)
        for args in tqdm(job_args, desc=adapter.dataset_name):
            seq_name, err = _process_sequence(args)
            done += 1
            if err:
                failed.append((seq_name, err))
                tqdm.write(f"  [SKIP] {seq_name}: {err.splitlines()[-1]}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_sequence, a): a[0] for a in job_args}
            with tqdm(total=total, desc=adapter.dataset_name) as pbar:
                for fut in as_completed(futures):
                    seq_name, err = fut.result()
                    done += 1
                    pbar.update(1)
                    if err:
                        failed.append((seq_name, err))
                        tqdm.write(f"  [SKIP] {seq_name}: {err.splitlines()[-1]}")

    print(f"[precompute] done {done - len(failed)}/{total}, failed {len(failed)}")
    if failed:
        print("[precompute] Failed sequences:")
        for seq_name, err in failed:
            print(f"  {seq_name}:\n    {err.splitlines()[-1]}")


# --------------------------------------------------------------------------- #
# Adapter state serialisation (for subprocess pickling)
# --------------------------------------------------------------------------- #

def _make_adapter_state(adapter: BaseAdapter) -> tuple:
    """
    Return (adapter_class, kwargs_dict) so a subprocess can re-instantiate.
    Covers the four supported adapters.
    """
    cls = type(adapter)
    name = cls.__name__

    if name == "ScanNetAdapter":
        return cls, {
            "root": str(adapter.root),
            "depth_scale": adapter.depth_scale,
            "default_pose_convention": adapter.default_pose_convention,
        }
    elif name == "Co3Dv2Adapter":
        return cls, {
            "root": str(adapter.root),
            "categories": adapter.categories if hasattr(adapter, "categories") else None,
            "subset_name": adapter.subset_name if hasattr(adapter, "subset_name") else "fewview_train",
            "split": adapter.split if hasattr(adapter, "split") else "train",
        }
    elif name == "BlendedMVSAdapter":
        return cls, {
            "root": str(adapter.root),
            "split": adapter.split if hasattr(adapter, "split") else "train",
            "use_masked": adapter.use_masked if hasattr(adapter, "use_masked") else False,
            "load_precomputed": False,
            "load_normals": False,
            "verbose": False,
        }
    elif name == "MVSSynthAdapter":
        return cls, {
            "root": str(adapter.root),
            "load_precomputed": False,
            "verbose": False,
        }
    elif name == "TartanAirAdapter":
        return cls, {
            "root": str(adapter.root),
            "split": adapter.split if hasattr(adapter, "split") else "train",
            "camera": adapter.camera if hasattr(adapter, "camera") else "left",
            "load_precomputed": False,
            "verbose": False,
        }
    elif name == "VKITTI2Adapter":
        return cls, {
            "root": str(adapter.root),
            "split": adapter.split if hasattr(adapter, "split") else "train",
            "camera": adapter.camera if hasattr(adapter, "camera") else "Camera_0",
            "verbose": False,
        }
    elif name == "WaymoAdapter":
        # Waymo is TFRecord-based; disable on-the-fly RAFT flow for precompute
        return cls, {
            "root": str(adapter.root),
            "split": adapter.split if hasattr(adapter, "split") else "training",
            "extract_flow": False,
            "verbose": False,
        }
    else:
        raise ValueError(f"Unsupported adapter type for multiprocessing: {name}")
