"""
Regenerate Co3D precomputed tracks from ``pointcloud.ply``.

The previous Co3D recompute path sampled points from one reference frame's
depth map and then projected those points to all other frames.  That works
for datasets with dense, view-consistent depth, but it fails for Co3D:

- the depth maps are sparse;
- a single frame only observes a thin visible surface patch of the object;
- relaxing the depth check turns most in-bounds projections into false
  positives, so the saved ``valids`` become nearly useless;
- the resulting point cloud is typically a small front surface that looks
  planar in rerun.

Co3D already ships a sequence-level ``pointcloud.ply`` in the same world
coordinates as the cameras.  We therefore use the point cloud itself as the
3D track source and only use per-frame depth for normals and optional sparse
consistency filtering.

Usage:
    conda run -n d4rt python datasets/computer/co3d_recompute_tracks.py \\
        --root /data2/d4rt/datasets/Co3Dv2 \\
        --categories apple toaster \\
        --num-points 8000 \\
        --workers 4 \\
        --overwrite

    # All categories (may take a long time):
    conda run -n d4rt python datasets/computer/co3d_recompute_tracks.py \\
        --root /data2/d4rt/datasets/Co3Dv2 \\
        --num-points 8000 \\
        --workers 8 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.co3dv2 import (
    Co3Dv2Adapter,
    _load_depth,
    _load_depth_mask,
    _ndc_to_pinhole,
    _precomputed_tracks_look_temporal,
    _p3d_to_opencv_extrinsics,
    _summarize_precomputed_track_validity,
)
from computer.co3d_pointcloud_to_tracks import (
    project_pointcloud_to_frames,
    read_ply_pointcloud,
)
from computer.depth_to_normals import compute_normals_sequence


_WORKER_ADAPTER: Co3Dv2Adapter | None = None
_WORKER_OUTPUT_ROOT: Path | None = None
_WORKER_NUM_POINTS: int = 8000
_WORKER_OVERWRITE: bool = False
_WORKER_TRACKS_ONLY: bool = False
_FAILURE_MARKER_NAME = "precomputed.failed.json"


def _write_failure_marker(failure_path: Path, seq_name: str, err: str) -> None:
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequence": seq_name,
        "error": err.splitlines()[-1] if err else "unknown error",
        "traceback": err,
    }
    tmp_path = failure_path.with_name(failure_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(failure_path)


def _init_worker(
    adapter_kwargs: dict,
    output_root: str,
    num_points: int,
    overwrite: bool,
    tracks_only: bool,
) -> None:
    """Initialise worker-local state once instead of once per sequence."""
    global _WORKER_ADAPTER, _WORKER_OUTPUT_ROOT
    global _WORKER_NUM_POINTS, _WORKER_OVERWRITE, _WORKER_TRACKS_ONLY

    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_NUM_POINTS = int(num_points)
    _WORKER_OVERWRITE = bool(overwrite)
    _WORKER_TRACKS_ONLY = bool(tracks_only)

    # On Linux/fork the parent adapter can be inherited copy-on-write.
    # On spawn/fallback we construct it once per worker.
    if _WORKER_ADAPTER is None:
        _WORKER_ADAPTER = Co3Dv2Adapter(**adapter_kwargs)


def _load_sequence_geometry(
    adapter: Co3Dv2Adapter,
    sequence_name: str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, tuple[int, int], int, Path]:
    """Load only depths + cameras for a full sequence."""
    r = adapter._get_record(sequence_name)
    adapter._ensure_frame_anno_loaded(r.category)

    def _load_one(frame_number: int) -> tuple[np.ndarray, tuple[int, int], np.ndarray, np.ndarray]:
        anno = adapter._get_frame_anno(r.category, r.sequence_name, frame_number)
        dep = _load_depth(
            adapter.root / anno["depth"]["path"],
            anno["depth"]["scale_adjustment"],
        )
        dep_mask_path = adapter.root / anno["depth"]["mask_path"]
        if dep_mask_path.exists():
            dep[~_load_depth_mask(dep_mask_path)] = 0.0
        H, W = anno["image"]["size"]
        K = _ndc_to_pinhole(
            anno["viewpoint"]["focal_length"],
            anno["viewpoint"]["principal_point"],
            anno["image"]["size"],
        )
        E = _p3d_to_opencv_extrinsics(
            anno["viewpoint"]["R"],
            anno["viewpoint"]["T"],
        )
        return dep, (H, W), K, E

    frame_numbers = list(r.frame_numbers)
    with ThreadPoolExecutor(max_workers=min(len(frame_numbers), 8)) as ex:
        rows = list(ex.map(_load_one, frame_numbers))

    depths, image_sizes, intrinsics_list, extrinsics_list = map(list, zip(*rows))
    image_size = tuple(image_sizes[0]) if image_sizes else (0, 0)
    intrinsics = np.stack(intrinsics_list, axis=0)
    extrinsics = np.stack(extrinsics_list, axis=0)
    return depths, intrinsics, extrinsics, image_size, r.num_frames, r.sequence_dir


def compute_tracks_co3d(
    pointcloud: np.ndarray,
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_size: tuple[int, int],
    num_points: int = 8000,
    depth_consistency_thresh: float = 0.20,
    rng_seed: int = 42,
) -> dict:
    """Project sampled PLY points to all frames.

    The sampled 3D points come directly from the sequence point cloud, so the
    saved tracks cover the full object shape instead of a single view's
    visible surface patch.
    """
    rng = np.random.default_rng(rng_seed)
    pointcloud = np.asarray(pointcloud, dtype=np.float32)

    if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
        raise ValueError(f"pointcloud must have shape [N,3], got {pointcloud.shape}")

    if len(pointcloud) == 0:
        T = len(depths)
        return {
            "trajs_2d": np.zeros((T, 0, 2), dtype=np.float32),
            "trajs_3d_world": np.zeros((T, 0, 3), dtype=np.float32),
            "valids": np.zeros((T, 0), dtype=bool),
            "visibs": np.zeros((T, 0), dtype=bool),
            "ref_frame": 0,
            "num_points": 0,
        }

    if len(pointcloud) > num_points:
        keep = rng.choice(len(pointcloud), num_points, replace=False)
        pointcloud = pointcloud[keep]

    tracks = project_pointcloud_to_frames(
        pointcloud=pointcloud,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size=image_size,
        depths=depths,
        depth_consistency_thresh=depth_consistency_thresh,
    )
    tracks["track_source"] = "pointcloud.ply"
    return tracks


def process_sequence(seq_name: str) -> tuple[str, str | None]:
    """Process one Co3D sequence and write precomputed.npz."""
    adapter = _WORKER_ADAPTER
    output_root = _WORKER_OUTPUT_ROOT
    num_points = _WORKER_NUM_POINTS
    overwrite = _WORKER_OVERWRITE
    tracks_only = _WORKER_TRACKS_ONLY
    if adapter is None or output_root is None:
        raise RuntimeError("Worker state is not initialised")

    out_path = output_root / seq_name / "precomputed.npz"
    failure_path = out_path.with_name(_FAILURE_MARKER_NAME)
    if out_path.exists() and not overwrite:
        if failure_path.exists():
            failure_path.unlink()
        return seq_name, None

    try:
        depths, intrinsics, extrinsics, image_size, num_frames, sequence_root = _load_sequence_geometry(
            adapter, seq_name
        )
        normals = None if tracks_only else compute_normals_sequence(depths, intrinsics)

        ply_path = sequence_root / "pointcloud.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"Missing pointcloud.ply for {seq_name}: {ply_path}")
        pointcloud = read_ply_pointcloud(ply_path)

        tracks = compute_tracks_co3d(
            pointcloud=pointcloud,
            depths=depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_size=image_size,
            num_points=num_points,
            depth_consistency_thresh=0.20,
        )
        stats = _summarize_precomputed_track_validity(tracks["valids"])
        if not _precomputed_tracks_look_temporal(tracks["valids"]):
            raise RuntimeError(
                "recomputed tracks still look degenerate: "
                + json.dumps(stats, sort_keys=True)
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.name + ".tmp.npz")
        np.savez_compressed(
            str(tmp_path),
            trajs_2d=tracks["trajs_2d"].astype(np.float32),
            trajs_3d_world=tracks["trajs_3d_world"].astype(np.float32),
            valids=tracks["valids"],
            visibs=tracks["visibs"],
            ref_frame=np.array(tracks["ref_frame"], dtype=np.int32),
            num_frames=np.array(num_frames, dtype=np.int32),
            num_points=np.array(tracks["num_points"], dtype=np.int32),
            track_source=np.array(tracks["track_source"]),
            **({"normals": normals.astype(np.float16)} if normals is not None else {}),
        )
        tmp_path.replace(out_path)
        if failure_path.exists():
            failure_path.unlink()
        return seq_name, None

    except Exception:
        err = traceback.format_exc()
        _write_failure_marker(failure_path, seq_name, err)
        return seq_name, err


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Co3D precomputed tracks with depth-scale fix."
    )
    parser.add_argument("--root", required=True, help="Co3Dv2 dataset root")
    parser.add_argument("--output-root", default=None, help="Output root (defaults to --root)")
    parser.add_argument("--num-points", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Categories to process (default: all)")
    parser.add_argument("--sequence-list", default=None,
                        help="Optional txt/json file of sequences to recompute")
    parser.add_argument("--tracks-only", action="store_true",
                        help="Write only track arrays/metadata. Use with adapter track_precompute_root overlay.")
    parser.add_argument("--subset-name", default="fewview_train",
                        help="Co3D subset name (default: fewview_train). Use fewview_dev for val split.")
    parser.add_argument("--split", default="train",
                        help="Dataset split (default: train). Use val for validation sequences.")
    args = parser.parse_args()

    adapter = Co3Dv2Adapter(root=args.root, categories=args.categories, verbose=True,
                            subset_name=args.subset_name, split=args.split)
    output_root = Path(args.output_root) if args.output_root else Path(args.root)

    if args.sequence_list:
        seq_path = Path(args.sequence_list)
        if seq_path.suffix.lower() == ".json":
            raw = json.loads(seq_path.read_text())
            if isinstance(raw, dict):
                sequences = []
                for value in raw.values():
                    if isinstance(value, list):
                        sequences.extend(value)
            else:
                sequences = list(raw)
        else:
            sequences = [line.strip() for line in seq_path.read_text().splitlines() if line.strip()]
    else:
        sequences = adapter.list_sequences()
    total = len(sequences)
    print(f"[co3d_recompute] {total} sequences → {output_root}")

    adapter_kwargs = {
        "root": str(args.root),
        "categories": adapter.categories,
        "subset_name": adapter.subset_name,
        "split": adapter.split,
        "verbose": False,
    }

    done = 0
    failed = []

    global _WORKER_ADAPTER
    _WORKER_ADAPTER = adapter

    if args.workers <= 1:
        _init_worker(
            adapter_kwargs,
            str(output_root),
            args.num_points,
            args.overwrite,
            args.tracks_only,
        )
        for seq_name in tqdm(sequences, desc="co3d_recompute"):
            seq_name, err = process_sequence(seq_name)
            done += 1
            if err:
                failed.append((seq_name, err))
                tqdm.write(f"  [FAIL] {seq_name}: {err.splitlines()[-1]}")
    else:
        pool_kwargs = {
            "max_workers": args.workers,
            "initializer": _init_worker,
            "initargs": (
                adapter_kwargs,
                str(output_root),
                args.num_points,
                args.overwrite,
                args.tracks_only,
            ),
        }
        if sys.platform.startswith("linux"):
            pool_kwargs["mp_context"] = mp.get_context("fork")
        with ProcessPoolExecutor(**pool_kwargs) as pool:
            futures = {pool.submit(process_sequence, seq_name): seq_name for seq_name in sequences}
            with tqdm(total=total, desc="co3d_recompute") as pbar:
                for fut in as_completed(futures):
                    seq_name, err = fut.result()
                    done += 1
                    pbar.update(1)
                    if err:
                        failed.append((seq_name, err))
                        tqdm.write(f"  [FAIL] {seq_name}: {err.splitlines()[-1]}")

    print(f"\n[co3d_recompute] done {done - len(failed)}/{total}, failed {len(failed)}")
    if failed:
        print("[co3d_recompute] First 10 failures:")
        for sn, err in failed[:10]:
            print(f"  {sn}: {err.splitlines()[-1]}")


if __name__ == "__main__":
    main()
