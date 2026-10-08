#!/usr/bin/env python3
"""
Validate all D4RT datasets for training readiness.

Checks:
1. Precomputed cache files exist
2. Adapters can load data correctly
3. Data format matches D4RT requirements
4. UnifiedClip schema compliance

Usage:
    python validate_all_datasets.py --output validation_report.json
"""

import sys
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.registry import create_adapter, list_datasets


# Dataset configurations
DATASET_CONFIGS = {
    "tartanair": {
        "root": "/data2/d4rt/datasets/TartanAir",
        "precompute_root": "/data2/d4rt/datasets/TartanAir",
        "camera": "left",
    },
    "vkitti2": {
        "root": "/data2/d4rt/datasets/VirtualKitti",
        "precompute_root": "/data2/d4rt/datasets/VirtualKitti",
        "camera": "Camera_0",
    },
    "blendedmvs": {
        "root": "/data2/d4rt/datasets/BlendedMVS",
        "precompute_root": "/data2/d4rt/datasets/BlendedMVS",
    },
    "scannet": {
        "root": "/data2/d4rt/datasets/scannet/scannet",
        "precompute_root": "/data2/d4rt/datasets/scannet/scannet",
    },
    "mvssynth": {
        "root": "/data2/d4rt/datasets/MVS-Synth",
        "precompute_root": "/data2/d4rt/datasets/MVS-Synth",
    },
    "pointodyssey": {
        "root": "/data2/d4rt/datasets/PointOdyssey",
    },
    "kubric": {
        "root": "/data2/d4rt/datasets/kubric1",
    },
    "dynamic_replica": {
        "root": "/data1/d4rt/datasets/Dynamic_Replica",
    },
}


def validate_dataset(dataset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single dataset."""
    result = {
        "dataset_name": dataset_name,
        "status": "unknown",
        "adapter_loaded": False,
        "num_sequences": 0,
        "sequences_with_cache": 0,
        "sample_load_success": False,
        "errors": [],
        "warnings": [],
    }

    try:
        # 1. Load adapter
        adapter = create_adapter(dataset_name, **config)
        result["adapter_loaded"] = True

        sequences = adapter.list_sequences()
        result["num_sequences"] = len(sequences)

        if len(sequences) == 0:
            result["errors"].append("No sequences found")
            result["status"] = "error"
            return result

        # 2. Check precompute cache (if applicable)
        if "precompute_root" in config:
            precompute_root = Path(config["precompute_root"])
            cached = 0
            for seq in sequences:
                cache_path = precompute_root / seq / "precomputed.npz"
                h5_path = precompute_root / seq / "precomputed.h5"
                if cache_path.exists() or h5_path.exists():
                    cached += 1
            result["sequences_with_cache"] = cached

            if cached == 0:
                result["warnings"].append("No precomputed cache found")
            elif cached < len(sequences):
                result["warnings"].append(
                    f"Only {cached}/{len(sequences)} sequences have cache"
                )

        # 3. Try loading a sample clip
        try:
            sample_seq = sequences[0]
            info = adapter.get_sequence_info(sample_seq)
            num_frames = info.get("num_frames", 0)

            if num_frames < 8:
                result["warnings"].append(
                    f"Sample sequence has only {num_frames} frames (< 8)"
                )
            else:
                # Load a small clip
                frame_indices = list(range(min(8, num_frames)))
                clip = adapter.load_clip(sample_seq, frame_indices)

                # Validate UnifiedClip schema
                assert hasattr(clip, "dataset_name")
                assert hasattr(clip, "images")
                assert hasattr(clip, "intrinsics")
                assert hasattr(clip, "extrinsics")
                assert hasattr(clip, "metadata")

                assert len(clip.images) == len(frame_indices)
                assert clip.intrinsics.shape[0] == len(frame_indices)
                assert clip.extrinsics.shape[0] == len(frame_indices)

                result["sample_load_success"] = True
                result["sample_info"] = {
                    "sequence": sample_seq,
                    "num_frames_loaded": len(clip.images),
                    "image_shape": clip.images[0].shape,
                    "has_depth": clip.depths is not None,
                    "has_normals": clip.normals is not None,
                    "has_tracks": clip.trajs_2d is not None,
                    "has_visibility": clip.visibs is not None,
                }

        except Exception as e:
            result["errors"].append(f"Sample load failed: {str(e)}")
            result["sample_load_success"] = False

        # Determine overall status
        if result["errors"]:
            result["status"] = "error"
        elif result["warnings"]:
            result["status"] = "warning"
        else:
            result["status"] = "ok"

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"Adapter creation failed: {str(e)}")
        result["traceback"] = traceback.format_exc()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="validation_report.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("D4RT Dataset Validation")
    print("=" * 80)
    print()

    results = {}

    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"Validating {dataset_name}...", end=" ", flush=True)
        result = validate_dataset(dataset_name, config)
        results[dataset_name] = result

        status_symbol = {
            "ok": "✅",
            "warning": "⚠️",
            "error": "❌",
            "unknown": "❓",
        }[result["status"]]

        print(f"{status_symbol} {result['status'].upper()}")

        if args.verbose or result["status"] in ["error", "warning"]:
            if result["errors"]:
                for err in result["errors"]:
                    print(f"  ERROR: {err}")
            if result["warnings"]:
                for warn in result["warnings"]:
                    print(f"  WARN: {warn}")

        if result.get("sample_info"):
            info = result["sample_info"]
            print(f"  Sequences: {result['num_sequences']}")
            if result.get("sequences_with_cache"):
                print(f"  Cached: {result['sequences_with_cache']}/{result['num_sequences']}")
            print(f"  Sample: {info['sequence']} ({info['num_frames_loaded']} frames)")
            print(f"  Image: {info['image_shape']}")
            print(f"  Has depth: {info['has_depth']}, "
                  f"normals: {info['has_normals']}, "
                  f"tracks: {info['has_tracks']}")
        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    ok_count = sum(1 for r in results.values() if r["status"] == "ok")
    warn_count = sum(1 for r in results.values() if r["status"] == "warning")
    error_count = sum(1 for r in results.values() if r["status"] == "error")

    print(f"✅ OK: {ok_count}")
    print(f"⚠️  Warning: {warn_count}")
    print(f"❌ Error: {error_count}")
    print()

    total_sequences = sum(r["num_sequences"] for r in results.values())
    total_cached = sum(r.get("sequences_with_cache", 0) for r in results.values())
    print(f"Total sequences: {total_sequences}")
    print(f"Total with cache: {total_cached}")
    print()

    # Save report
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
