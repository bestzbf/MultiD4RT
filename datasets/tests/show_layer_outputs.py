#!/usr/bin/env python
"""
展示D4RT数据加载器每一层的具体输出

演示完整数据流：
第1层 Adapter → 第2层 Mixture → 第3层 QueryBuilder → Collate
"""

import sys
sys.path.insert(0, '/data2/d4rt/code')

import numpy as np
import torch


def show_layer1_output():
    """第1层：Dataset Adapter 输出"""
    print("\n" + "="*70)
    print("第1层：Dataset Adapter 输出 (UnifiedClip)")
    print("="*70)

    from datasets.adapters.pointodyssey import PointOdysseyAdapter

    # 创建adapter
    adapter = PointOdysseyAdapter(
        root='/data2/d4rt/datasets/PointOdyssey',
        split='train',
        verbose=False
    )

    print(f"\n数据集: {adapter.dataset_name}")
    print(f"序列数量: {len(adapter)}")

    # 获取一个序列
    seq_name = adapter.get_sequence_name(0)
    print(f"测试序列: {seq_name}")

    # 加载clip
    frame_indices = [0, 10, 20, 30, 40, 50, 60, 70]
    clip = adapter.load_clip(seq_name, frame_indices)

    print(f"\n📦 UnifiedClip 输出结构:")
    print(f"  ├─ dataset_name: {clip.dataset_name}")
    print(f"  ├─ sequence_name: {clip.sequence_name}")
    print(f"  ├─ frame_paths: list[{len(clip.frame_paths)}] paths")
    print(f"  ├─ images: list[{len(clip.images)}] arrays")
    print(f"  │   └─ shape: {clip.images[0].shape} (H,W,C)")

    if clip.depths is not None:
        print(f"  ├─ depths: list[{len(clip.depths)}] arrays")
        print(f"  │   └─ shape: {clip.depths[0].shape} (H,W)")
    else:
        print(f"  ├─ depths: None")

    if clip.normals is not None:
        print(f"  ├─ normals: list[{len(clip.normals)}] arrays")
    else:
        print(f"  ├─ normals: None")

    print(f"  ├─ trajs_2d: {clip.trajs_2d.shape if clip.trajs_2d is not None else None} (T,N,2)")
    print(f"  ├─ trajs_3d_world: {clip.trajs_3d_world.shape if clip.trajs_3d_world is not None else None} (T,N,3)")
    print(f"  ├─ valids: {clip.valids.shape if clip.valids is not None else None} (T,N)")
    print(f"  ├─ visibs: {clip.visibs.shape if clip.visibs is not None else None} (T,N)")
    print(f"  ├─ intrinsics: {clip.intrinsics.shape} (T,3,3)")
    print(f"  ├─ extrinsics: {clip.extrinsics.shape} (T,4,4)")
    print(f"  ├─ flows: {clip.flows}")
    print(f"  └─ metadata:")
    for key, value in clip.metadata.items():
        if isinstance(value, (list, dict)):
            print(f"      ├─ {key}: {type(value).__name__}")
        else:
            print(f"      ├─ {key}: {value}")

    return clip


def show_layer2_output(clip):
    """第2层：GeometryTransformPipeline 输出"""
    print("\n" + "="*70)
    print("第2层：GeometryTransformPipeline 输出 (TransformResult)")
    print("="*70)

    from datasets.transforms import GeometryTransformPipeline
    import random

    pipeline = GeometryTransformPipeline(
        img_size=256,
        use_augs=True
    )

    rng = random.Random(42)
    result = pipeline(clip, rng=rng)

    print(f"\n📦 TransformResult 输出结构:")
    print(f"  ├─ dataset_name: {result.dataset_name}")
    print(f"  ├─ sequence_name: {result.sequence_name}")
    print(f"  ├─ images: list[{len(result.images)}] arrays")
    print(f"  │   └─ shape: {result.images[0].shape} (H,W,C) - 已resize到256x256")
    print(f"  │   └─ dtype: {result.images[0].dtype}, range: [{result.images[0].min():.3f}, {result.images[0].max():.3f}]")

    if result.depths is not None:
        print(f"  ├─ depths: list[{len(result.depths)}] arrays")
        print(f"  │   └─ shape: {result.depths[0].shape}")
    else:
        print(f"  ├─ depths: None")

    if result.trajs_2d is not None:
        print(f"  ├─ trajs_2d: {result.trajs_2d.shape} - 已更新坐标")
        print(f"  │   └─ 坐标范围: x=[{result.trajs_2d[...,0].min():.1f}, {result.trajs_2d[...,0].max():.1f}]")
        print(f"  │                y=[{result.trajs_2d[...,1].min():.1f}, {result.trajs_2d[...,1].max():.1f}]")

    print(f"  ├─ intrinsics: {result.intrinsics.shape} - 已更新内参")
    print(f"  │   └─ fx={result.intrinsics[0,0,0]:.1f}, fy={result.intrinsics[0,1,1]:.1f}")
    print(f"  │   └─ cx={result.intrinsics[0,0,2]:.1f}, cy={result.intrinsics[0,1,2]:.1f}")
    print(f"  ├─ extrinsics: {result.extrinsics.shape}")

    if result.valids is not None:
        print(f"  ├─ valids: {result.valids.shape}")
        print(f"  │   └─ valid率: {result.valids.mean()*100:.1f}%")

    if result.visibs is not None:
        print(f"  └─ visibs: {result.visibs.shape}")
        print(f"      └─ visible率: {result.visibs.mean()*100:.1f}%")

    return result


def show_layer3_output(result):
    """第3层：D4RT Query Builder 输出"""
    print("\n" + "="*70)
    print("第3层：D4RT Query Builder 输出 (QuerySample)")
    print("="*70)

    from datasets.query_builder import D4RTQueryBuilder
    import random

    builder = D4RTQueryBuilder(
        num_queries=2048,
        boundary_ratio=0.3,
        t_tgt_eq_t_cam_ratio=0.4
    )

    rng = random.Random(42)
    sample = builder(result, py_rng=rng)

    print(f"\n📦 QuerySample 输出结构:")
    print(f"  ├─ dataset_name: {sample.dataset_name}")
    print(f"  ├─ sequence_name: {sample.sequence_name}")
    print(f"  ├─ video: {sample.video.shape} (S,C,H,W) - Tensor")
    print(f"  │   └─ dtype: {sample.video.dtype}, device: {sample.video.device}")
    print(f"  ├─ coords: {sample.coords.shape} (Q,2) - source坐标")
    print(f"  │   └─ 范围: x=[{sample.coords[:,0].min():.1f}, {sample.coords[:,0].max():.1f}]")
    print(f"  │            y=[{sample.coords[:,1].min():.1f}, {sample.coords[:,1].max():.1f}]")
    print(f"  ├─ t_src: {sample.t_src.shape} (Q,) - source时间索引")
    print(f"  │   └─ 范围: [{sample.t_src.min()}, {sample.t_src.max()}]")
    print(f"  ├─ t_tgt: {sample.t_tgt.shape} (Q,) - target时间索引")
    print(f"  │   └─ 范围: [{sample.t_tgt.min()}, {sample.t_tgt.max()}]")
    print(f"  ├─ t_cam: {sample.t_cam.shape} (Q,) - camera时间索引")
    print(f"  │   └─ t_tgt==t_cam比例: {(sample.t_tgt==sample.t_cam).float().mean()*100:.1f}%")
    print(f"  ├─ intrinsics: {sample.intrinsics.shape} (S,3,3)")
    print(f"  ├─ extrinsics: {sample.extrinsics.shape} (S,4,4)")
    print(f"  ├─ local_patches: {sample.local_patches.shape} (Q,C,P,P)")

    print(f"  ├─ targets: dict with keys:")
    for key, value in sample.targets.items():
        if isinstance(value, torch.Tensor):
            print(f"  │   ├─ {key}: {value.shape}")
            if key.startswith('mask_'):
                valid_ratio = value.float().mean() * 100
                print(f"  │   │   └─ 有效率: {valid_ratio:.1f}%")

    print(f"  └─ metadata:")
    for key, value in sample.metadata.items():
        if isinstance(value, (list, dict)):
            print(f"      ├─ {key}: {type(value).__name__}")
        else:
            print(f"      ├─ {key}: {value}")

    return sample


def show_collate_output(samples):
    """Collate层：批处理输出"""
    print("\n" + "="*70)
    print("Collate层：批处理输出 (Batch Dict)")
    print("="*70)

    from datasets.collate import d4rt_collate_fn

    batch = d4rt_collate_fn(samples)

    print(f"\n📦 Batch 输出结构:")
    print(f"  ├─ video: {batch['video'].shape} (B,S,C,H,W)")
    print(f"  ├─ coords: {batch['coords'].shape} (B,Q,2)")
    print(f"  ├─ t_src: {batch['t_src'].shape} (B,Q)")
    print(f"  ├─ t_tgt: {batch['t_tgt'].shape} (B,Q)")
    print(f"  ├─ t_cam: {batch['t_cam'].shape} (B,Q)")
    print(f"  ├─ intrinsics: {batch['intrinsics'].shape} (B,S,3,3)")
    print(f"  ├─ extrinsics: {batch['extrinsics'].shape} (B,S,4,4)")
    print(f"  ├─ local_patches: {batch['local_patches'].shape} (B,Q,C,P,P)")
    print(f"  ├─ targets: dict")
    for key, value in batch['targets'].items():
        if isinstance(value, torch.Tensor):
            print(f"  │   ├─ {key}: {value.shape}")
    print(f"  ├─ dataset_names: {batch['dataset_names']}")
    print(f"  └─ sequence_names: {batch['sequence_names']}")

    return batch


def main():
    """运行完整演示"""
    print("="*70)
    print("D4RT 数据加载器 - 每一层输出演示")
    print("="*70)

    try:
        # 第1层
        clip = show_layer1_output()

        # 第2层
        result = show_layer2_output(clip)

        # 第3层
        sample = show_layer3_output(result)

        # Collate层
        samples = [sample, sample]  # 模拟batch
        batch = show_collate_output(samples)

        print("\n" + "="*70)
        print("✅ 完整数据流演示成功！")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
