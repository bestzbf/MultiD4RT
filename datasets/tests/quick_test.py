#!/usr/bin/env python
"""快速验证单场景和混合场景功能"""
import sys
sys.path.insert(0, '/data2/d4rt/code')

from datasets.factory import create_training_dataset

# 测试1: 单场景
print("测试单场景模式...")
config_single = {
    'mode': 'single',
    'name': 'pointodyssey',
    'root': '/data2/d4rt/datasets/PointOdyssey',
    'clip_len': 8,
    'img_size': 256,
    'num_queries': 256,
}
ds = create_training_dataset(config_single, split='train')
print(f"✓ 单场景数据集创建成功")

# 测试2: 混合场景
print("\n测试混合场景模式...")
config_mixture = {
    'mode': 'mixture',
    'datasets': [
        {'name': 'pointodyssey', 'root': '/data2/d4rt/datasets/PointOdyssey', 'weight': 0.7},
        {'name': 'kubric', 'root': '/data2/d4rt/datasets/kubric', 'weight': 0.3},
    ],
    'clip_len': 8,
    'img_size': 256,
    'num_queries': 256,
}
ds_mix = create_training_dataset(config_mixture, split='train')
print(f"✓ 混合数据集创建成功")
print(f"  数据集: {ds_mix.get_dataset_names()}")

print("\n✅ 所有功能验证通过！")
