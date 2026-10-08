from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.adapters.co3dv2 import Co3Dv2Adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter Co3Dv2 sequences and optionally export an allowlist."
    )
    parser.add_argument("--root", required=True, help="Co3Dv2 dataset root.")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional category subset, e.g. apple bottle chair.",
    )
    parser.add_argument("--subset-name", default="fewview_train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--precompute-root", default=None)
    parser.add_argument("--min-viewpoint-quality", type=float, default=None)
    parser.add_argument("--min-pointcloud-quality", type=float, default=None)
    parser.add_argument("--min-pointcloud-n-points", type=int, default=None)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=None)
    parser.add_argument("--min-foreground-ratio", type=float, default=None)
    parser.add_argument("--quality-probe-frames", type=int, default=3)
    parser.add_argument("--require-pointcloud", action="store_true")
    parser.add_argument("--require-precomputed", action="store_true")
    parser.add_argument("--max-sequences-per-category", type=int, default=None)
    parser.add_argument("--sequence-allowlist", default=None)
    parser.add_argument("--sequence-denylist", default=None)
    parser.add_argument("--output-json", default=None, help="Write summary JSON here.")
    parser.add_argument("--output-txt", default=None, help="Write one UID per line here.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    adapter = Co3Dv2Adapter(
        root=args.root,
        categories=args.categories,
        subset_name=args.subset_name,
        split=args.split,
        min_frames=args.min_frames,
        verbose=not args.quiet,
        precompute_root=args.precompute_root,
        min_viewpoint_quality=args.min_viewpoint_quality,
        min_pointcloud_quality=args.min_pointcloud_quality,
        min_pointcloud_n_points=args.min_pointcloud_n_points,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        min_foreground_ratio=args.min_foreground_ratio,
        quality_probe_frames=args.quality_probe_frames,
        require_pointcloud=args.require_pointcloud,
        require_precomputed=args.require_precomputed,
        max_sequences_per_category=args.max_sequences_per_category,
        sequence_allowlist=args.sequence_allowlist,
        sequence_denylist=args.sequence_denylist,
    )
    sequences = adapter.list_sequences()
    summary = adapter.get_filter_summary()

    payload = {
        "root": args.root,
        "subset_name": args.subset_name,
        "split": args.split,
        "num_sequences": len(sequences),
        "filters": {
            "min_frames": args.min_frames,
            "min_viewpoint_quality": args.min_viewpoint_quality,
            "min_pointcloud_quality": args.min_pointcloud_quality,
            "min_pointcloud_n_points": args.min_pointcloud_n_points,
            "min_valid_depth_ratio": args.min_valid_depth_ratio,
            "min_foreground_ratio": args.min_foreground_ratio,
            "quality_probe_frames": args.quality_probe_frames,
            "require_pointcloud": args.require_pointcloud,
            "require_precomputed": args.require_precomputed,
            "max_sequences_per_category": args.max_sequences_per_category,
            "sequence_allowlist": args.sequence_allowlist,
            "sequence_denylist": args.sequence_denylist,
        },
        "summary": summary,
        "sequences": sequences,
    }

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output_txt:
        out_txt = Path(args.output_txt)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(sequences) + ("\n" if sequences else ""))

    print(f"selected sequences: {len(sequences)}")
    if summary.get("dropped_by_reason"):
        print("dropped_by_reason:")
        for reason, count in sorted(summary["dropped_by_reason"].items()):
            print(f"  {reason}: {count}")
    if sequences:
        preview = sequences[:10]
        print("preview:")
        for seq in preview:
            print(f"  {seq}")
        if len(sequences) > len(preview):
            print(f"  ... ({len(sequences) - len(preview)} more)")


if __name__ == "__main__":
    main()
