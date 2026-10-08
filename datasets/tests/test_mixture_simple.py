#!/usr/bin/env python3
"""Test MixtureDataset with Kubric only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets.adapters.kubric import KubricAdapter
from datasets.mixture import MixtureDataset


def main():
    print("=" * 80)
    print("Testing MixtureDataset (Kubric only)")
    print("=" * 80)

    # Initialize adapter
    kubric = KubricAdapter(root="/data2/d4rt/datasets/kubric")
    print(f"✓ Kubric: {len(kubric)} sequences")

    # Create mixture dataset (single dataset)
    mixture = MixtureDataset(
        adapters=[kubric],
        dataset_weights=[1.0],
        clip_len=8,
        img_size=256,
        use_augs=True,
        num_queries=2048,
        seed=42,
    )
    print(f"\n✓ MixtureDataset created: {mixture.get_dataset_names()}")
    print(f"  Epoch size: {len(mixture)}")

    # Sample a few batches
    print("\n" + "=" * 80)
    print("Sampling 3 batches...")
    print("=" * 80)

    for i in range(3):
        sample = mixture[i]

        print(f"\nBatch {i}:")
        print(f"  Dataset: {sample.metadata['dataset_name']}")
        print(f"  Sequence: {sample.metadata['sequence_name']}")
        print(f"  video: {sample.video.shape}")
        print(f"  coords: {sample.coords.shape}")
        print(f"  t_src: {sample.t_src.shape}")
        print(f"  t_tgt: {sample.t_tgt.shape}")
        print(f"  t_cam: {sample.t_cam.shape}")
        print(f"  mask_3d: {sample.masks['mask_3d'].mean():.3f}")
        print(f"  mask_2d: {sample.masks['mask_2d'].mean():.3f}")
        print(f"  mask_vis: {sample.masks['mask_vis'].mean():.3f}")

    print("\n✓ Phase 3 complete: sampling.py + mixture.py working!")


if __name__ == "__main__":
    main()
