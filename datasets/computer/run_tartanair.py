"""
Precompute normals + tracks for TartanAir sequences.

TartanAir has RGB + depth + optical-flow but NO tracks/normals/visibility.
This script derives normals (from depth+intrinsics) and pseudo-static tracks
(from depth+pose, Strategy A) and caches them as precomputed.npz.

The output directory structure mirrors the adapter's sequence names so that
TartanAirAdapter can load the cache with:
    <output_root> / <seq_name> / precomputed.npz

Usage:
    python run_tartanair.py \\
        --root /data2/d4rt/datasets/TartanAir \\
        [--output-root /data2/d4rt/datasets/TartanAir] \\
        [--camera left] \\
        [--points-per-frame 100] \\
        [--track-dedup-voxel-size 0.05] \\
        [--track-depth-max 200] \\
        [--track-depth-sampling log_balanced] \\
        [--track-depth-bin-edges 0,5,10,20,40,80,200] \\
        [--track-depth-bin-weights 6,6,4,2,1,1] \\
        [--workers 4] \\
        [--overwrite]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # /data2/d4rt/code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # /data2/d4rt/code/datasets

from computer._run_common import run_precompute
from adapters.TartanAir import TartanAirAdapter


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "" or raw.lower() in {"none", "auto"}:
        return None
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute TartanAir normals + pseudo-static tracks from depth+pose"
    )
    parser.add_argument("--root", required=True,
                        help="TartanAir root directory (contains P001/, P003/, …)")
    parser.add_argument("--output-root", default=None,
                        help="Where to write precomputed.npz files (default: same as --root)")
    parser.add_argument("--camera", default="left", choices=["left", "right"],
                        help="Which camera to use (default: left)")
    parser.add_argument("--num-points", type=int, default=8000,
                        help="Legacy total track points per sequence when --points-per-frame <= 0")
    parser.add_argument("--points-per-frame", type=int, default=100,
                        help="Sample this many static source points from each frame before 3D dedup (default: 100; set <=0 for legacy segment sampling)")
    parser.add_argument("--ref-frame-stride", type=int, default=1,
                        help="Frame stride for --points-per-frame mode (default: 1, every frame)")
    parser.add_argument("--track-dedup-voxel-size", type=float, default=0.05,
                        help="Metric 3D voxel size for deduplicating per-frame static points (default: 0.05m; <=0 disables)")
    parser.add_argument("--dense-static-tracks", action="store_true",
                        help="Store dense full-sequence trajs_2d/valids/visibs instead of compact static points (not recommended for long TartanAir sequences)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--num-ref-segments", type=int, default=20,
                        help="How many temporal segments to split each sequence into when sampling track reference frames (default: 20)")
    parser.add_argument("--ref-frame-strategy", default="max_valid", choices=["first", "max_valid", "middle"],
                        help="How to choose a reference frame inside each temporal segment")
    parser.add_argument("--track-depth-max", type=float, default=200.0,
                        help="Maximum metric depth used when sampling/projecting precomputed tracks (default: 200 for TartanAir)")
    parser.add_argument("--track-depth-sampling", default="log_balanced",
                        choices=["uniform", "log_balanced", "linear_balanced"],
                        help="How to sample source pixels in each reference frame (default: log_balanced)")
    parser.add_argument("--track-depth-bin-edges", default="0,5,10,20,40,80,200",
                        help="Comma-separated metric depth bin edges for balanced source sampling; use 'none' for dynamic bins")
    parser.add_argument("--track-depth-bin-weights", default="6,6,4,2,1,1",
                        help="Comma-separated per-bin weights; length must be len(edges)-1")
    parser.add_argument("--track-depth-num-bins", type=int, default=7,
                        help="Number of dynamic depth bins when --track-depth-bin-edges=none")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute even if precomputed.npz already exists")
    args = parser.parse_args()

    adapter = TartanAirAdapter(
        root=args.root,
        camera=args.camera,
        load_precomputed=False,
        verbose=False,
    )
    output_root = Path(args.output_root) if args.output_root else Path(args.root)

    run_precompute(
        adapter=adapter,
        output_root=output_root,
        num_points=args.num_points,
        workers=args.workers,
        overwrite=args.overwrite,
        track_kwargs={
            "num_ref_segments": args.num_ref_segments,
            "ref_frame_strategy": args.ref_frame_strategy,
            "depth_max": args.track_depth_max,
            "source_depth_sampling": args.track_depth_sampling,
            "source_depth_bin_edges": _parse_float_list(args.track_depth_bin_edges),
            "source_depth_bin_weights": _parse_float_list(args.track_depth_bin_weights),
            "source_depth_num_bins": args.track_depth_num_bins,
            "points_per_ref_frame": args.points_per_frame if args.points_per_frame > 0 else None,
            "ref_frame_stride": args.ref_frame_stride,
            "world_dedup_voxel_size": args.track_dedup_voxel_size,
            "compact_static_tracks": not args.dense_static_tracks,
        },
    )


if __name__ == "__main__":
    main()
