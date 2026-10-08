#!/usr/bin/env python
"""
Test MixtureDataset on real data.

Usage:
    python test_mixture.py
"""

import sys
sys.path.insert(0, '/data2/d4rt/code')

import torch
from torch.utils.data import DataLoader

from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset
from datasets.collate import d4rt_collate_fn


def test_single_adapter():
    """Test with single adapter (PointOdyssey)"""
    print("=" * 60)
    print("Test 1: Single Adapter (PointOdyssey)")
    print("=" * 60)

    try:
        adapter = create_adapter(
            'pointodyssey',
            root='/data2/d4rt/datasets/PointOdyssey',
            split='train',
            verbose=True
        )
        print(f"✓ Adapter created: {len(adapter)} sequences")

        dataset = MixtureDataset(
            adapters=[adapter],
            clip_len=8,
            img_size=256,
            num_queries=512,
            seed=42,
        )
        print(f"✓ Dataset created: length={len(dataset)}")

        # Test single sample
        print("\nTesting single sample...")
        sample = dataset[0]

        print(f"  video: {sample.video.shape}")
        print(f"  coords: {sample.coords.shape}")
        print(f"  t_src: {sample.t_src.shape}")
        print(f"  t_tgt: {sample.t_tgt.shape}")
        print(f"  t_cam: {sample.t_cam.shape}")
        print(f"  intrinsics: {sample.intrinsics.shape}")
        print(f"  extrinsics: {sample.extrinsics.shape}")
        print(f"  local_patches: {sample.local_patches.shape}")
        print(f"  dataset_name: {sample.dataset_name}")
        print(f"  sequence_name: {sample.sequence_name}")

        # Check targets
        print("\nTargets:")
        for key, val in sample.targets.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}, dtype={val.dtype}")

        print("\n✓ Single sample test passed")
        return True

    except Exception as e:
        print(f"\n✗ Single adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test with DataLoader"""
    print("\n" + "=" * 60)
    print("Test 2: DataLoader")
    print("=" * 60)

    try:
        adapter = create_adapter(
            'pointodyssey',
            root='/data2/d4rt/datasets/PointOdyssey',
            split='train',
            verbose=False
        )

        dataset = MixtureDataset(
            adapters=[adapter],
            clip_len=8,
            img_size=256,
            num_queries=512,
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=d4rt_collate_fn,
            num_workers=0,
        )
        print(f"✓ DataLoader created")

        # Test one batch
        print("\nTesting one batch...")
        batch = next(iter(loader))

        print(f"  video: {batch['video'].shape}")
        print(f"  coords: {batch['coords'].shape}")
        print(f"  t_src: {batch['t_src'].shape}")
        print(f"  intrinsics: {batch['intrinsics'].shape}")
        print(f"  local_patches: {batch['local_patches'].shape}")
        print(f"  dataset_names: {batch['dataset_names']}")
        print(f"  sequence_names: {batch['sequence_names']}")

        # Check targets
        print("\nBatch targets:")
        for key, val in batch['targets'].items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}")

        print("\n✓ DataLoader test passed")
        return True

    except Exception as e:
        print(f"\n✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_adapter():
    """Test with multiple adapters"""
    print("\n" + "=" * 60)
    print("Test 3: Multiple Adapters")
    print("=" * 60)

    try:
        adapters = []
        weights = []

        # Try PointOdyssey
        try:
            po = create_adapter(
                'pointodyssey',
                root='/data2/d4rt/datasets/PointOdyssey',
                split='train',
                verbose=False
            )
            adapters.append(po)
            weights.append(0.6)
            print(f"✓ PointOdyssey: {len(po)} sequences")
        except Exception as e:
            print(f"⚠ PointOdyssey not available: {e}")

        # Try ScanNet
        try:
            sn = create_adapter(
                'scannet',
                root='/data2/d4rt/datasets/scannet/scannet',
            )
            adapters.append(sn)
            weights.append(0.4)
            print(f"✓ ScanNet: {len(sn)} sequences")
        except Exception as e:
            print(f"⚠ ScanNet not available: {e}")

        if len(adapters) < 2:
            print("\n⚠ Need at least 2 datasets for multi-adapter test")
            return None

        dataset = MixtureDataset(
            adapters=adapters,
            dataset_weights=weights,
            clip_len=8,
            img_size=256,
            num_queries=512,
        )
        print(f"\n✓ MixtureDataset created with {len(adapters)} adapters")

        # Sample multiple times to check mixture
        print("\nSampling 10 times to check mixture:")
        dataset_counts = {}
        for i in range(10):
            sample = dataset[i]
            name = sample.dataset_name
            dataset_counts[name] = dataset_counts.get(name, 0) + 1

        for name, count in dataset_counts.items():
            print(f"  {name}: {count}/10")

        print("\n✓ Multi-adapter test passed")
        return True

    except Exception as e:
        print(f"\n✗ Multi-adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing D4RT MixtureDataset on real data\n")

    results = []
    results.append(("Single Adapter", test_single_adapter()))
    results.append(("DataLoader", test_dataloader()))
    results.append(("Multi Adapter", test_multi_adapter()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"{status}: {name}")
