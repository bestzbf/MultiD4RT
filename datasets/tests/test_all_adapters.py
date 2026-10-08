#!/usr/bin/env python
"""
Test all available dataset adapters.

Usage:
    python test_all_adapters.py
"""

import sys
sys.path.insert(0, '/data2/d4rt/code')

from datasets.registry import create_adapter, list_datasets


# Dataset configurations
DATASET_CONFIGS = {
    'pointodyssey': {
        'root': '/data2/d4rt/datasets/PointOdyssey',
        'split': 'train',
    },
    'scannet': {
        'root': '/data2/d4rt/datasets/scannet/scannet',
    },
    'co3dv2': {
        'root': '/data2/d4rt/datasets/Co3Dv2',
    },
    'kubric': {
        'root': '/data2/d4rt/datasets/kubric',
    },
    'blendedmvs': {
        'root': '/data2/d4rt/datasets/BlendedMVS',
    },
    'mvssynth': {
        'root': '/data2/d4rt/datasets/MVS-Synth/GTAV_1080',
    },
    'dynamic_replica': {
        'root': '/data1/d4rt/datasets/Dynamic_Replica',
    },
}


def test_adapter(name: str, config: dict):
    """Test a single adapter."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        adapter = create_adapter(name, **config)
        num_seq = len(adapter)
        sequences = adapter.list_sequences()

        print(f"✓ Adapter created: {num_seq} sequences")
        print(f"  First 3 sequences: {sequences[:3]}")

        # Test loading one clip
        if num_seq > 0:
            seq_name = sequences[0]
            info = adapter.get_sequence_info(seq_name)
            num_frames = info['num_frames']

            # Sample a few frames
            frame_indices = list(range(min(8, num_frames)))
            clip = adapter.load_clip(seq_name, frame_indices)

            print(f"  Test clip from '{seq_name}':")
            print(f"    images: {len(clip.images)} frames, shape={clip.images[0].shape}")
            print(f"    depths: {clip.depths is not None}")
            print(f"    normals: {clip.normals is not None}")
            print(f"    trajs_2d: {clip.trajs_2d is not None}")
            print(f"    trajs_3d: {clip.trajs_3d_world is not None}")
            print(f"    has_tracks: {clip.metadata.get('has_tracks', False)}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing all D4RT dataset adapters\n")
    print(f"Registered datasets: {list_datasets()}")

    results = {}
    for name, config in DATASET_CONFIGS.items():
        results[name] = test_adapter(name, config)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} adapters working")
