"""
Precompute normals + tracks for Co3Dv2 sequences.

Usage:
    python run_co3dv2.py \\
        --root /data2/d4rt/datasets/Co3Dv2 \\
        [--output-root /data2/d4rt/datasets/Co3Dv2] \\
        [--num-points 8000] \\
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
from adapters.co3dv2 import Co3Dv2Adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Co3Dv2 normals + tracks")
    parser.add_argument("--root", required=True, help="Co3Dv2 root directory")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-points", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    adapter = Co3Dv2Adapter(root=args.root)
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
