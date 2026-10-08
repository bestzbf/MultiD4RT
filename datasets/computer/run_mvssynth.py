"""
Precompute normals + tracks for MVS-Synth sequences.

Usage:
    python run_mvssynth.py \\
        --root /data2/d4rt/datasets/MVS-Synth/GTAV_1080 \\
        [--output-root /data2/d4rt/datasets/MVS-Synth/GTAV_1080] \\
        [--num-points 8000] \\
        [--points-per-frame 100] \\
        [--track-dedup-voxel-size 0.05] \\
        [--track-depth-max 100] \\
        [--track-depth-sampling log_balanced] \\
        [--track-depth-bin-edges 0,5,10,20,40,80,100] \\
        [--track-depth-bin-weights 6,6,4,2,1,1] \\
        [--sequences 0000,0001] \\
        [--skip-normals] \\
        [--workers 4] \\
        [--overwrite]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from computer._run_common import run_precompute
from adapters.mvssynth import MVSSynthAdapter


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "" or raw.lower() in {"none", "auto"}:
        return None
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_sequence_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    seqs = [part.strip() for part in raw.split(",") if part.strip()]
    return seqs or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute MVS-Synth normals + tracks")
    parser.add_argument("--root", required=True, help="MVS-Synth GTAV_1080 root directory")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-points", type=int, default=8000,
                        help="Legacy total track points per sequence when --points-per-frame <= 0")
    parser.add_argument("--points-per-frame", type=int, default=100,
                        help="Sample this many static source points from each frame before 3D dedup (default: 100; set <=0 for legacy segment sampling)")
    parser.add_argument("--ref-frame-stride", type=int, default=1,
                        help="Frame stride for --points-per-frame mode (default: 1, every frame)")
    parser.add_argument("--track-dedup-voxel-size", type=float, default=0.05,
                        help="Metric 3D voxel size for deduplicating per-frame static points (default: 0.05m; <=0 disables)")
    parser.add_argument("--dense-static-tracks", action="store_true",
                        help="Store dense full-sequence trajs_2d/valids/visibs instead of compact static points")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--num-ref-segments", type=int, default=20,
                        help="How many temporal segments to split each sequence into in legacy segment mode")
    parser.add_argument("--ref-frame-strategy", default="max_valid", choices=["first", "max_valid", "middle"],
                        help="How to choose a reference frame inside each temporal segment")
    parser.add_argument("--track-depth-max", type=float, default=100.0,
                        help="Maximum metric depth used when sampling/projecting precomputed tracks (default: 100 for MVS-Synth)")
    parser.add_argument("--track-depth-sampling", default="log_balanced",
                        choices=["uniform", "log_balanced", "linear_balanced"],
                        help="How to sample source pixels in each reference frame (default: log_balanced)")
    parser.add_argument("--track-depth-bin-edges", default="0,5,10,20,40,80,100",
                        help="Comma-separated metric depth bin edges for balanced source sampling; use 'none' for dynamic bins")
    parser.add_argument("--track-depth-bin-weights", default="6,6,4,2,1,1",
                        help="Comma-separated per-bin weights; length must be len(edges)-1")
    parser.add_argument("--track-depth-num-bins", type=int, default=7,
                        help="Number of dynamic depth bins when --track-depth-bin-edges=none")
    parser.add_argument("--sequences", default=None,
                        help="Optional comma-separated sequence ids to precompute, e.g. 0000,0001")
    parser.add_argument("--skip-normals", action="store_true",
                        help="Write track-only caches without full-resolution normal maps")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    adapter = MVSSynthAdapter(root=args.root, load_precomputed=False, verbose=False)
    sequences = _parse_sequence_list(args.sequences)
    if sequences is not None:
        missing = [seq for seq in sequences if seq not in adapter._name_to_record]
        if missing:
            raise ValueError(f"Unknown MVS-Synth sequence ids: {missing}")
        adapter._records = [adapter._name_to_record[seq] for seq in sequences]
        adapter._name_to_record = {r.sequence_id: r for r in adapter._records}
    output_root = Path(args.output_root) if args.output_root else Path(args.root)

    run_precompute(
        adapter=adapter,
        output_root=output_root,
        num_points=args.num_points,
        workers=args.workers,
        overwrite=args.overwrite,
        compute_normals=not args.skip_normals,
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
