#!/usr/bin/env python3
"""Test script to verify MVS-Synth adapter output matches documentation requirements."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.adapters.mvssynth import MVSSynthAdapter


def test_schema_compliance():
    """Test that MVS-Synth adapter output matches UnifiedClip schema."""

    print("=" * 80)
    print("MVS-Synth Schema Compliance Test")
    print("=" * 80)

    # Initialize adapter
    adapter = MVSSynthAdapter(
        root="/data2/d4rt/datasets/MVS-Synth/GTAV_1080",
        verbose=False,
    )

    # Load a clip
    seq_name = "0000"
    frame_indices = [0, 10, 20]
    clip = adapter.load_clip(seq_name, frame_indices)

    T = len(frame_indices)
    passed = 0
    total = 0

    # Test 1: dataset_name
    total += 1
    if isinstance(clip.dataset_name, str) and clip.dataset_name:
        print(f"✓ dataset_name: str = '{clip.dataset_name}'")
        passed += 1
    else:
        print(f"✗ dataset_name: expected str, got {type(clip.dataset_name)}")

    # Test 2: sequence_name
    total += 1
    if isinstance(clip.sequence_name, str) and clip.sequence_name:
        print(f"✓ sequence_name: str = '{clip.sequence_name}'")
        passed += 1
    else:
        print(f"✗ sequence_name: expected str, got {type(clip.sequence_name)}")

    # Test 3: frame_paths
    total += 1
    if clip.frame_paths is None or (isinstance(clip.frame_paths, list) and
                                     len(clip.frame_paths) == T and
                                     all(isinstance(p, str) for p in clip.frame_paths)):
        print(f"✓ frame_paths: list[str] | None, len={len(clip.frame_paths) if clip.frame_paths else 0}")
        passed += 1
    else:
        print(f"✗ frame_paths: invalid format")

    # Test 4: images
    total += 1
    if (isinstance(clip.images, list) and len(clip.images) == T and
        all(isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3
            for img in clip.images)):
        H, W = clip.images[0].shape[:2]
        print(f"✓ images: list[np.ndarray], len={T}, shape=[{H},{W},3]")
        passed += 1
    else:
        print(f"✗ images: expected list[np.ndarray] with shape [H,W,3]")

    # Test 5: depths
    total += 1
    if clip.depths is None or (isinstance(clip.depths, list) and len(clip.depths) == T and
                                all(isinstance(d, np.ndarray) and d.ndim == 2
                                    for d in clip.depths)):
        if clip.depths:
            print(f"✓ depths: list[np.ndarray], len={T}, shape={clip.depths[0].shape}")
        else:
            print(f"✓ depths: None")
        passed += 1
    else:
        print(f"✗ depths: invalid format")

    # Test 6-10: Optional precomputed trajectory fields.
    has_tracks = bool(clip.metadata.get("has_tracks", False))
    for field_name in ["normals", "trajs_2d", "trajs_3d_world", "valids", "visibs"]:
        total += 1
        field_value = getattr(clip, field_name)
        if field_name == "normals":
            ok = field_value is None or isinstance(field_value, list)
        elif has_tracks:
            ok = isinstance(field_value, np.ndarray) and field_value.shape[0] == T
        else:
            ok = field_value is None
        if ok:
            detail = "None" if field_value is None else f"{type(field_value).__name__} shape={getattr(field_value, 'shape', '')}"
            print(f"✓ {field_name}: {detail}")
            passed += 1
        else:
            expected = "np.ndarray with T frames" if has_tracks else "None"
            print(f"✗ {field_name}: expected {expected}, got {type(field_value)}")

    # Test 11: intrinsics
    total += 1
    if (isinstance(clip.intrinsics, np.ndarray) and
        clip.intrinsics.shape == (T, 3, 3)):
        K = clip.intrinsics[0]
        print(f"✓ intrinsics: np.ndarray, shape={clip.intrinsics.shape}")
        print(f"  [0]: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
        passed += 1
    else:
        print(f"✗ intrinsics: expected shape ({T},3,3), got {clip.intrinsics.shape}")

    # Test 12: extrinsics
    total += 1
    if (isinstance(clip.extrinsics, np.ndarray) and
        clip.extrinsics.shape == (T, 4, 4)):
        E = clip.extrinsics[0]
        print(f"✓ extrinsics: np.ndarray, shape={clip.extrinsics.shape}")
        print(f"  [0]: translation={E[:3,3]}, det(R)={np.linalg.det(E[:3,:3]):.6f}")
        passed += 1
    else:
        print(f"✗ extrinsics: expected shape ({T},4,4), got {clip.extrinsics.shape}")

    # Test 13: metadata
    total += 1
    if isinstance(clip.metadata, dict):
        print(f"✓ metadata: dict, keys={list(clip.metadata.keys())}")
        passed += 1
    else:
        print(f"✗ metadata: expected dict, got {type(clip.metadata)}")

    print("\n" + "=" * 80)
    print(f"Schema Compliance: {passed}/{total} tests passed")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = test_schema_compliance()
    sys.exit(0 if success else 1)
