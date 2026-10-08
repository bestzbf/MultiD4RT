"""
Benchmark dataloader speed for D4RT mixed training.

Target: >= 2-3 batches/second to avoid training bottleneck.

Usage:
    python benchmark_dataloader.py
"""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.collate import d4rt_collate_fn
from datasets.mixture import MixtureDataset
from datasets.registry import create_adapter


def benchmark_dataloader(
    batch_size: int = 4,
    num_workers: int = 4,
    num_batches: int = 50,
    clip_len: int = 8,
    img_size: int = 256,
    num_queries: int = 2048,
):
    """Benchmark dataloader throughput."""

    print("=" * 80)
    print("D4RT DataLoader Benchmark")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Clip length: {clip_len}")
    print(f"Image size: {img_size}")
    print(f"Queries per sample: {num_queries}")
    print(f"Measuring {num_batches} batches...")
    print()

    # Dataset roots
    roots = {
        "pointodyssey": "/data2/d4rt/datasets/PointOdyssey",
        "dynamic_replica": "/data1/d4rt/datasets/Dynamic_Replica",
        "kubric": "/data2/d4rt/datasets/kubric",
        "scannet": "/data2/d4rt/datasets/scannet",
        "co3dv2": "/data2/d4rt/datasets/Co3Dv2",
        "blendedmvs": "/data2/d4rt/datasets/BlendedMVS",
        "mvssynth": "/data2/d4rt/datasets/MVS-Synth",
        "tartanair": "/data2/d4rt/datasets/TartanAir",
        "vkitti2": "/data2/d4rt/datasets/VirtualKitti",
        "waymo": "/data2/d4rt/datasets/Waymo",
    }

    # Check which datasets exist
    available_datasets = []
    for name, root in roots.items():
        if Path(root).exists():
            available_datasets.append(name)

    print(f"Available datasets: {len(available_datasets)}/{len(roots)}")
    for name in available_datasets:
        print(f"  ✓ {name}")
    print()

    if len(available_datasets) == 0:
        print("ERROR: No datasets found!")
        return

    # Create adapters
    print("Creating adapters...")
    adapters = []
    for name in available_datasets:
        try:
            adapter = create_adapter(name, root=roots[name], split="train")
            adapters.append(adapter)
            print(f"  ✓ {name}: {len(adapter)} sequences")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    if len(adapters) == 0:
        print("ERROR: No adapters created!")
        return

    print()

    # Create mixture dataset
    print("Creating mixture dataset...")
    dataset = MixtureDataset(
        adapters=adapters,
        dataset_weights=None,  # uniform
        clip_len=clip_len,
        img_size=img_size,
        use_augs=True,
        num_queries=num_queries,
        boundary_ratio=0.3,
        t_tgt_eq_t_cam_ratio=0.4,
        seed=42,
    )
    print(f"  Dataset length: {len(dataset)}")
    print()

    # Create dataloader
    print("Creating dataloader...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=d4rt_collate_fn,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    print()

    # Warmup
    print("Warming up (5 batches)...")
    warmup_start = time.time()
    for i, batch in enumerate(loader):
        if i >= 5:
            break
    warmup_time = time.time() - warmup_start
    print(f"  Warmup time: {warmup_time:.2f}s")
    print()

    # Benchmark
    print(f"Benchmarking ({num_batches} batches)...")
    batch_times = []

    start_time = time.time()
    for i, batch in enumerate(loader):
        batch_end = time.time()

        if i > 0:  # Skip first batch timing
            batch_times.append(batch_end - batch_start)

        batch_start = batch_end

        if i >= num_batches:
            break

        if (i + 1) % 10 == 0:
            avg_time = sum(batch_times[-10:]) / len(batch_times[-10:])
            throughput = 1.0 / avg_time
            print(f"  Batch {i+1}/{num_batches}: {avg_time:.3f}s/batch ({throughput:.2f} batch/s)")

    total_time = time.time() - start_time

    # Results
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)

    avg_batch_time = sum(batch_times) / len(batch_times)
    throughput = 1.0 / avg_batch_time
    samples_per_sec = throughput * batch_size

    print(f"Total time: {total_time:.2f}s")
    print(f"Batches processed: {len(batch_times)}")
    print(f"Average batch time: {avg_batch_time:.3f}s")
    print(f"Throughput: {throughput:.2f} batches/s")
    print(f"Samples/s: {samples_per_sec:.2f}")
    print()

    # Check target
    target_throughput = 2.0
    if throughput >= target_throughput:
        print(f"✅ PASS: {throughput:.2f} >= {target_throughput} batches/s")
    else:
        print(f"⚠️  WARN: {throughput:.2f} < {target_throughput} batches/s")
        print("   Consider:")
        print("   - Increase num_workers")
        print("   - Use precomputed .h5 files")
        print("   - Reduce img_size or num_queries")

    print()

    # Batch info
    print("Sample batch shape:")
    print(f"  video: {batch['video'].shape}")
    print(f"  coords: {batch['coords'].shape}")
    print(f"  intrinsics: {batch['intrinsics'].shape}")
    if batch['local_patches'] is not None:
        print(f"  local_patches: {batch['local_patches'].shape}")

    return throughput


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=2048)

    args = parser.parse_args()

    benchmark_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        clip_len=args.clip_len,
        img_size=args.img_size,
        num_queries=args.num_queries,
    )
