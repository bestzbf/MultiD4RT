#!/usr/bin/env python
"""
测试数据加载速度和depth反投影功能

目标：确保数据加载不会成为训练瓶颈（每秒至少2-3个batch）
"""

import sys
sys.path.insert(0, '/data2/d4rt/code')

import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.mixture import MixtureDataset
from datasets.collate import d4rt_collate_fn


def test_depth_to_tracks():
    """测试depth反投影生成伪轨迹"""
    print("\n" + "="*70)
    print("测试1：Depth反投影生成伪轨迹")
    print("="*70)

    from datasets.computer.depth_to_tracks import compute_tracks

    # 模拟数据
    T, H, W = 8, 256, 256
    depths = [np.random.rand(H, W).astype(np.float32) * 5.0 for _ in range(T)]

    K = np.array([
        [300.0, 0.0, 128.0],
        [0.0, 300.0, 128.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    intrinsics = np.stack([K] * T, axis=0)

    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * T, axis=0)

    print(f"输入: {T}帧, {H}x{W}, 采样8000个点")

    start = time.time()
    result = compute_tracks(
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        num_points=8000,
        boundary_ratio=0.3
    )
    elapsed = time.time() - start

    print(f"\n✓ 计算完成，耗时: {elapsed:.3f}秒")
    print(f"  - trajs_2d: {result['trajs_2d'].shape}")
    print(f"  - trajs_3d_world: {result['trajs_3d_world'].shape}")
    print(f"  - valids: {result['valids'].shape}, 有效率: {result['valids'].mean()*100:.1f}%")
    print(f"  - visibs: {result['visibs'].shape}, 可见率: {result['visibs'].mean()*100:.1f}%")

    return elapsed < 1.0  # 应该在1秒内完成


def test_dataloader_speed_single_dataset():
    """测试单数据集加载速度"""
    print("\n" + "="*70)
    print("测试2：单数据集加载速度（PointOdyssey）")
    print("="*70)

    try:
        from datasets.adapters.pointodyssey import PointOdysseyAdapter

        adapter = PointOdysseyAdapter(
            root='/data2/d4rt/datasets/PointOdyssey',
            split='train',
            verbose=False
        )

        dataset = MixtureDataset(
            adapters=[adapter],
            dataset_weights=[1.0],
            clip_len=8,
            img_size=256,
            num_queries=2048,
            use_augs=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=4,
            collate_fn=d4rt_collate_fn,
            pin_memory=True
        )

        print(f"配置: batch_size=2, num_workers=4")
        print(f"数据集大小: {len(dataset)}")

        # 预热
        print("\n预热中...")
        for i, batch in enumerate(dataloader):
            if i >= 2:
                break

        # 测速
        print("测速中...")
        num_batches = 10
        start = time.time()

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

        elapsed = time.time() - start
        batches_per_sec = num_batches / elapsed

        print(f"\n✓ 加载{num_batches}个batch，耗时: {elapsed:.2f}秒")
        print(f"  速度: {batches_per_sec:.2f} batches/秒")
        print(f"  目标: ≥2.0 batches/秒")

        if batches_per_sec >= 2.0:
            print(f"  ✓ 通过！速度足够")
            return True
        else:
            print(f"  ⚠️  速度偏慢，可能成为瓶颈")
            return False

    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def test_dataloader_speed_mixture():
    """测试混合数据集加载速度"""
    print("\n" + "="*70)
    print("测试3：混合数据集加载速度")
    print("="*70)

    try:
        dataset_configs = [
            {
                'name': 'pointodyssey',
                'root': '/data2/d4rt/datasets/PointOdyssey',
                'split': 'train',
                'weight': 0.5
            },
            {
                'name': 'scannet',
                'root': '/data2/d4rt/datasets/ScanNet',
                'split': 'train',
                'weight': 0.5
            }
        ]

        dataset = MixtureDataset(
            dataset_configs=dataset_configs,
            num_frames=8,
            img_size=256,
            num_queries=2048,
            use_augs=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=4,
            collate_fn=d4rt_collate_fn,
            pin_memory=True
        )

        print(f"配置: 2个数据集混合, batch_size=2, num_workers=4")

        # 预热
        print("\n预热中...")
        for i, batch in enumerate(dataloader):
            if i >= 2:
                break

        # 测速
        print("测速中...")
        num_batches = 10
        start = time.time()

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

        elapsed = time.time() - start
        batches_per_sec = num_batches / elapsed

        print(f"\n✓ 加载{num_batches}个batch，耗时: {elapsed:.2f}秒")
        print(f"  速度: {batches_per_sec:.2f} batches/秒")

        return batches_per_sec >= 2.0

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("D4RT 数据加载器性能测试")
    print("="*70)

    results = []

    # 测试1: depth反投影
    results.append(("Depth反投影", test_depth_to_tracks()))

    # 测试2: 单数据集加载
    results.append(("单数据集加载", test_dataloader_speed_single_dataset()))

    # 测试3: 混合数据集加载
    # results.append(("混合数据集加载", test_dataloader_speed_mixture()))

    # 总结
    print("\n" + "="*70)
    print("测试结果总结")
    print("="*70)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*70)
    if all_passed:
        print("✓ 所有测试通过！数据加载速度满足要求")
    else:
        print("⚠️  部分测试未通过，建议优化")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
