"""
Precompute normals + tracks for ScanNet sequences.

Usage:
    python run_scannet.py \\
        --root /data2/d4rt/datasets/scannet/scannet \\
        [--output-root /data2/d4rt/datasets/scannet/scannet] \\
        [--num-points 8000] \\
        [--workers 4] \\
        [--overwrite]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # /data2/d4rt/code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # /data2/d4rt/code/datasets

from computer._run_common import run_precompute
from adapters.scannet import ScanNetAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ScanNet normals + tracks")
    parser.add_argument("--root", required=True, help="ScanNet root (contains scene*/ dirs)")
    parser.add_argument("--output-root", default=None,
                        help="Where to write precomputed.npz files (default: same as --root)")
    parser.add_argument("--num-points", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    adapter = ScanNetAdapter(root=args.root)
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
