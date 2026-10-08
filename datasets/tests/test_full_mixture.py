#!/usr/bin/env python
"""
Test MixtureDataset with all available datasets.

Usage:
    python test_full_mixture.py
"""

import sys
sys.path.insert(0, '/data2/d4rt/code')

from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset
from torch.utils.data import DataLoader
from datasets.collate import d4rt_collate_fn


# Available datasets (6 working datasets)
DATASET_CONFIGS = [
    {'name': 'pointodyssey', 'root': '/data2/d4rt/datasets/PointOdyssey', 'split': 'train', 'weight': 0.25},
    {'name': 'scannet', 'root': '/data2/d4rt/datasets/scannet/scannet', 'weight': 0.15},
    {'name': 'co3dv2', 'root': '/data2/d4rt/datasets/Co3Dv2', 'weight': 0.20},
    {'name': 'kubric', 'root': '/data2/d4rt/datasets/kubric', 'weight': 0.15},
    {'name': 'blendedmvs', 'root': '/data2/d4rt/datasets/BlendedMVS', 'weight': 0.15},
    {'name': 'mvssynth', 'root': '/data2/d4rt/datasets/MVS-Synth/GTAV_1080', 'weight': 0.08},
    {'name': 'dynamic_replica', 'root': '/data1/d4rt/datasets/Dynamic_Replica', 'weight': 0.12},
]


def test_full_mixture():
    """Test mixture with all available datasets."""
    print("="*60)
    print("Creating Multi-Dataset Mixture")
    print("="*60)

    # Create adapters
    adapters = []
    weights = []

    for cfg in DATASET_CONFIGS:
        try:
            name = cfg['name']
            weight = cfg.pop('weight')
            adapter = create_adapter(**cfg)
            adapters.append(adapter)
            weights.append(weight)
            print(f"✓ {name}: {len(adapter)} sequences (weight={weight})")
        except Exception as e:
            print(f"✗ {cfg['name']}: {e}")

    if len(adapters) == 0:
        print("\n✗ No adapters available")
        return False

    # Create mixture dataset
    dataset = MixtureDataset(
        adapters=adapters,
        dataset_weights=weights,
        clip_len=8,
        img_size=256,
        num_queries=512,
    )
    print(f"\n✓ MixtureDataset created with {len(adapters)} datasets")

    # Test sampling
    print("\nSampling 20 times to check mixture:")
    dataset_counts = {}
    for i in range(20):
        try:
            sample = dataset[i]
            name = sample.dataset_name
            dataset_counts[name] = dataset_counts.get(name, 0) + 1
        except Exception as e:
            print(f"  Error at index {i}: {e}")
            return False

    for name, count in sorted(dataset_counts.items()):
        print(f"  {name}: {count}/20")

    # Test DataLoader
    print("\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=4, collate_fn=d4rt_collate_fn, num_workers=0)
    batch = next(iter(loader))
    print(f"  Batch video: {batch['video'].shape}")
    print(f"  Batch datasets: {batch['dataset_names']}")

    print("\n✓ Full mixture test passed")
    return True


if __name__ == "__main__":
    success = test_full_mixture()
    exit(0 if success else 1)
