"""
Precompute normals + tracks for Waymo Open Dataset sequences.

Waymo has RGB + sparse LiDAR depth but NO tracks/normals/visibility.
This script derives normals (from depth+intrinsics) and pseudo-static tracks
(from depth+pose, Strategy A) and caches them as precomputed.npz.

NOTE: Waymo depth maps are LiDAR-projected and therefore very sparse.
      The track quality will be lower than for dense depth datasets.
      RAFT optical flow is disabled during precompute (extract_flow=False)
      to avoid GPU memory issues in multi-process mode.

The output directory structure mirrors the adapter's sequence names so that
WaymoAdapter can load the cache with:
    <output_root> / <seq_name> / precomputed.npz

Usage:
    python run_waymo.py \\
        --root /data2/d4rt/datasets/Waymo \\
        [--output-root /data2/d4rt/datasets/Waymo] \\
        [--num-points 4000] \\
        [--workers 1] \\
        [--overwrite]

NOTE: Waymo TFRecord parsing can be slow; workers=1 is recommended to avoid
      TensorFlow multi-process issues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # /data2/d4rt/code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # /data2/d4rt/code/datasets

from computer._run_common import run_precompute
from adapters.Waymo import WaymoAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute Waymo normals + pseudo-static tracks from sparse LiDAR depth+pose"
    )
    parser.add_argument("--root", required=True,
                        help="Waymo root directory (contains *.tfrecord files)")
    parser.add_argument("--output-root", default=None,
                        help="Where to write precomputed.npz files (default: same as --root)")
    parser.add_argument("--num-points", type=int, default=4000,
                        help="Number of track points to sample per sequence (default: 4000, "
                             "lower than dense-depth datasets due to sparse LiDAR)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1, "
                             "TensorFlow multi-process issues may arise with >1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute even if precomputed.npz already exists")
    args = parser.parse_args()

    # extract_flow=False: skip RAFT during precompute to avoid GPU contention
    adapter = WaymoAdapter(root=args.root, extract_flow=False, verbose=False)
    output_root = Path(args.output_root) if args.output_root else Path(args.root)

    run_precompute(
        adapter=adapter,
        output_root=output_root,
        num_points=args.num_points,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
