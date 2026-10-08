#!/usr/bin/env python
"""
测试单场景和混合场景训练数据加载器

验证：
1. 单场景训练模式
2. 混合场景训练模式
3. 配置文件加载
"""

import sys
sys.path.insert(0, '/data2/d4rt/code')

import yaml
from datasets.factory import create_training_dataset
from torch.utils.data import DataLoader
from datasets.collate import d4rt_collate_fn


def test_single_mode():
    """测试单场景训练模式"""
    print("=" * 60)
    print("测试 1: 单场景训练模式 (PointOdyssey)")
    print("=" * 60)

    config = {
        'mode': 'single',
        'name': 'pointodyssey',
        'root': '/data2/d4rt/datasets/PointOdyssey',
        'clip_len': 8,
        'img_size': 256,
        'num_queries': 512,
        'seed': 42,
    }

    try:
        dataset = create_training_dataset(config, split='train')
        print(f"✓ 数据集创建成功")

        # 测试采样
        sample = dataset[0]
        print(f"✓ 采样成功")
        print(f"  - dataset_name: {sample.dataset_name}")
        print(f"  - video shape: {sample.video.shape}")
        print(f"  - coords shape: {sample.coords.shape}")

        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixture_mode():
    """测试混合场景训练模式"""
    print("\n" + "=" * 60)
    print("测试 2: 混合场景训练模式")
    print("=" * 60)

    config = {
        'mode': 'mixture',
        'datasets': [
            {
                'name': 'pointodyssey',
                'root': '/data2/d4rt/datasets/PointOdyssey',
                'weight': 0.6
            },
            {
                'name': 'kubric',
                'root': '/data2/d4rt/datasets/kubric',
                'weight': 0.4
            },
        ],
        'clip_len': 8,
        'img_size': 256,
        'num_queries': 512,
        'seed': 42,
    }

    try:
        dataset = create_training_dataset(config, split='train')
        print(f"✓ 混合数据集创建成功")
        print(f"  - 数据集列表: {dataset.get_dataset_names()}")

        # 测试多次采样，验证混合
        dataset_counts = {}
        for i in range(20):
            sample = dataset[i]
            name = sample.dataset_name
            dataset_counts[name] = dataset_counts.get(name, 0) + 1

        print(f"✓ 采样分布 (20次):")
        for name, count in dataset_counts.items():
            print(f"  - {name}: {count}/20 ({count/20*100:.1f}%)")

        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """测试 DataLoader 批处理"""
    print("\n" + "=" * 60)
    print("测试 3: DataLoader 批处理")
    print("=" * 60)

    config = {
        'mode': 'single',
        'name': 'pointodyssey',
        'root': '/data2/d4rt/datasets/PointOdyssey',
        'clip_len': 8,
        'img_size': 256,
        'num_queries': 512,
    }

    try:
        dataset = create_training_dataset(config, split='train')

        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=d4rt_collate_fn,
        )

        batch = next(iter(loader))
        print(f"✓ DataLoader 创建成功")
        print(f"  - video: {batch['video'].shape}")
        print(f"  - coords: {batch['coords'].shape}")
        print(f"  - dataset_names: {batch['dataset_names']}")

        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_file():
    """测试从配置文件加载"""
    print("\n" + "=" * 60)
    print("测试 4: 从配置文件加载")
    print("=" * 60)

    config_path = '/data2/d4rt/code/configs/single_pointodyssey.yaml'

    try:
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        data_config = full_config['data']
        print(f"✓ 配置文件加载成功: {config_path}")
        print(f"  - mode: {data_config['mode']}")
        print(f"  - name: {data_config.get('name', 'N/A')}")

        dataset = create_training_dataset(data_config, split='train')
        print(f"✓ 数据集创建成功")

        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("D4RT 数据加载器功能测试\n")

    results = []
    results.append(("单场景模式", test_single_mode()))
    results.append(("混合场景模式", test_mixture_mode()))
    results.append(("DataLoader批处理", test_dataloader()))
    results.append(("配置文件加载", test_config_file()))

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败")
