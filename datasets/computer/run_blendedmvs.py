"""
Precompute normals + tracks for BlendedMVS sequences.

Usage:
    python run_blendedmvs.py \\
        --root /data2/d4rt/datasets/BlendedMVS \\
        [--split all] \\
        [--output-root /data2/d4rt/datasets/BlendedMVS] \\
        [--num-points 8000] \\
        [--points-per-frame 100] \\
        [--track-dedup-voxel-size 0.05] \\
        [--track-depth-max 500] \\
        [--track-depth-sampling log_balanced] \\
        [--sequences scene_a,scene_b] \\
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
from adapters.blendedmvs import BlendedMVSAdapter


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
    parser = argparse.ArgumentParser(description="Precompute BlendedMVS normals + tracks")
    parser.add_argument("--root", required=True, help="BlendedMVS root directory")
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "all"],
        help=(
            "Which split to precompute. "
            "'all' is the safe default so validation scenes do not silently miss precomputed caches."
        ),
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-points", type=int, default=8000,
                        help="Legacy total track points per sequence when --points-per-frame <= 0")
    parser.add_argument("--points-per-frame", type=int, default=100,
                        help="Sample this many source points from each frame before 3D dedup (default: 100; set <=0 for legacy segment sampling)")
    parser.add_argument("--ref-frame-stride", type=int, default=1,
                        help="Frame stride for --points-per-frame mode (default: 1, every frame)")
    parser.add_argument("--track-dedup-voxel-size", type=float, default=0.05,
                        help="Metric 3D voxel size for deduplicating per-frame static points (default: 0.05m; <=0 disables)")
    parser.add_argument("--dense-static-tracks", action="store_true",
                        help="Store dense full-sequence trajs_2d/valids/visibs instead of compact static points")
    parser.add_argument("--num-ref-segments", type=int, default=20,
                        help="How many temporal segments to split each sequence into in legacy segment mode")
    parser.add_argument("--ref-frame-strategy", default="max_valid", choices=["first", "max_valid", "middle"],
                        help="How to choose a reference frame inside each temporal segment")
    parser.add_argument("--track-depth-max", type=float, default=500.0,
                        help="Maximum metric depth used when sampling/projecting tracks (default: 500 for BlendedMVS)")
    parser.add_argument("--track-depth-sampling", default="log_balanced",
                        choices=["uniform", "log_balanced", "linear_balanced"],
                        help="How to sample source pixels in each reference frame (default: log_balanced)")
    parser.add_argument("--track-depth-bin-edges", default="0,2,5,10,20,40,80,160,320,500",
                        help="Comma-separated metric depth bin edges for balanced source sampling; use 'none' for dynamic bins")
    parser.add_argument("--track-depth-bin-weights", default="12,12,8,6,4,2,1,1,1",
                        help="Comma-separated per-bin weights; length must be len(edges)-1")
    parser.add_argument("--track-depth-num-bins", type=int, default=9,
                        help="Number of dynamic depth bins when --track-depth-bin-edges=none")
    parser.add_argument("--sequences", default=None,
                        help="Optional comma-separated scene ids to precompute")
    parser.add_argument("--skip-normals", action="store_true",
                        help="Write track-only caches without full-resolution normal maps")
    parser.add_argument("--track-mask-from-masked-images", action=argparse.BooleanOptionalAction, default=True,
                        help="Use BlendedMVS *_masked.jpg files to reject source pixels before track sampling")
    parser.add_argument("--track-mask-threshold", type=int, default=30,
                        help="A masked RGB pixel is valid when max(channel) is above this value")
    parser.add_argument("--track-mask-erosion-px", type=int, default=3,
                        help="Erode the valid masked-image region before source-point sampling")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else Path(args.root)
    splits = ["train", "val"] if args.split == "all" else [args.split]
    requested_sequences = _parse_sequence_list(args.sequences)

    for split in splits:
        adapter = BlendedMVSAdapter(
            root=args.root,
            split=split,
            use_masked=bool(args.track_mask_from_masked_images),
            verbose=False,
            load_precomputed=False,
            load_normals=False,
        )
        if requested_sequences is not None:
            present = set(adapter.list_sequences())
            selected = [seq for seq in requested_sequences if seq in present]
            if not selected:
                print(f"[run_blendedmvs] split={split}: no requested sequences in split")
                continue
            adapter._records = [adapter._name_to_record[seq] for seq in selected]
            adapter._name_to_record = {r.scene_id: r for r in adapter._records}
        print(f"[run_blendedmvs] split={split} -> output_root={output_root}")
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
                "source_mask_from_masked_images": bool(args.track_mask_from_masked_images),
                "source_mask_threshold": int(args.track_mask_threshold),
                "source_mask_erosion_px": int(args.track_mask_erosion_px),
            },
        )


if __name__ == "__main__":
    main()
