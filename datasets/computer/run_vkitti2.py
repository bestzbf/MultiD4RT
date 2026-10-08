"""
Precompute normals + tracks for Virtual KITTI 2 sequences.

VirtualKitti2 has RGB + depth + optical-flow but NO tracks/normals/visibility.
This script derives normals (from depth+intrinsics) and pseudo-static tracks
(from depth+pose, Strategy A) and caches them as precomputed.npz.

The output directory structure mirrors the adapter's sequence names so that
VKITTI2Adapter can load the cache with:
    <output_root> / <seq_name> / precomputed.npz
  e.g.  <output_root> / Scene01_clone / precomputed.npz

Usage:
    python run_vkitti2.py \\
        --root /data2/d4rt/datasets/VirtualKitti \\
        [--output-root /data2/d4rt/datasets/VirtualKitti] \\
        [--camera Camera_0] \\
        [--num-points 8000] \\
        [--num-ref-segments 20] \\
        [--track-depth-max 400] \\
        [--track-depth-sampling log_balanced] \\
        [--track-depth-bin-edges 0,10,20,40,80,160,400] \\
        [--track-depth-bin-weights 12,12,6,3,1,1] \\
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
from adapters.VirtualKitti import VKITTI2Adapter


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "" or raw.lower() in {"none", "auto"}:
        return None
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute VirtualKitti2 normals + pseudo-static tracks from depth+pose"
    )
    parser.add_argument("--root", required=True,
                        help="VirtualKitti2 root directory (contains Scene01/, Scene02/, …)")
    parser.add_argument("--output-root", default=None,
                        help="Where to write precomputed.npz files (default: same as --root)")
    parser.add_argument("--camera", default="Camera_0", choices=["Camera_0", "Camera_1"],
                        help="Which camera to use (default: Camera_0)")
    parser.add_argument("--num-points", type=int, default=8000,
                        help="Number of track points to sample per sequence (default: 8000)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--num-ref-segments", type=int, default=20,
                        help="How many temporal segments to split each sequence into when sampling track reference frames (default: 20)")
    parser.add_argument("--ref-frame-strategy", default="max_valid", choices=["first", "max_valid", "middle"],
                        help="How to choose a reference frame inside each temporal segment")
    parser.add_argument("--track-depth-max", type=float, default=400.0,
                        help="Maximum metric depth used when sampling/projecting precomputed tracks (default: 400 for VKITTI2)")
    parser.add_argument("--track-depth-sampling", default="log_balanced",
                        choices=["uniform", "log_balanced", "linear_balanced"],
                        help="How to sample source pixels in each reference frame (default: log_balanced)")
    parser.add_argument("--track-depth-bin-edges", default="0,10,20,40,80,160,400",
                        help="Comma-separated metric depth bin edges for balanced source sampling; use 'none' for dynamic bins")
    parser.add_argument("--track-depth-bin-weights", default="12,12,6,3,1,1",
                        help="Comma-separated per-bin weights; length must be len(edges)-1")
    parser.add_argument("--track-depth-num-bins", type=int, default=7,
                        help="Number of dynamic depth bins when --track-depth-bin-edges=none")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute even if precomputed.npz already exists")
    args = parser.parse_args()

    adapter = VKITTI2Adapter(root=args.root, camera=args.camera, verbose=False)
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
        },
    )


if __name__ == "__main__":
    main()
