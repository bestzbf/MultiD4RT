"""
D4RT 数据加载器使用示例

展示如何使用 registry.py 和 collate.py
"""

# ============================================================================
# 示例 1: 使用 registry 创建单个 adapter
# ============================================================================

from datasets.registry import create_adapter, list_datasets

# 查看所有可用数据集
print("Available datasets:", list_datasets())
# Output: ['pointodyssey', 'scannet', 'co3dv2', 'kubric', 'blendedmvs', 'mvssynth', 'dynamic_replica']

# 通过 registry 创建 adapter
adapter = create_adapter(
    'pointodyssey',
    root='/path/to/PointOdyssey',
    split='train',
    strict=True,
    verbose=True
)

# ============================================================================
# 示例 2: 使用 registry 创建 MixtureDataset
# ============================================================================

from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset

# 创建多个 adapter
adapters = [
    create_adapter('pointodyssey', root='/data/pointodyssey', split='train'),
    create_adapter('scannet', root='/data/scannet', split='train'),
    create_adapter('kubric', root='/data/kubric', split='train'),
]

# 创建混合数据集
dataset = MixtureDataset(
    adapters=adapters,
    dataset_weights=[0.5, 0.3, 0.2],  # PointOdyssey 50%, ScanNet 30%, Kubric 20%
    clip_len=48,
    img_size=256,
    num_queries=2048,
    boundary_ratio=0.3,
    t_tgt_eq_t_cam_ratio=0.4,
)

# ============================================================================
# 示例 3: 使用 collate_fn 创建 DataLoader
# ============================================================================

from torch.utils.data import DataLoader
from datasets.collate import d4rt_collate_fn

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=d4rt_collate_fn,
    pin_memory=True,
)

# 训练循环
for batch in loader:
    # batch 包含:
    # - video: [B, S, 3, H, W]
    # - coords: [B, Q, 2]
    # - t_src, t_tgt, t_cam: [B, Q]
    # - intrinsics: [B, S, 3, 3]
    # - extrinsics: [B, S, 4, 4]
    # - targets: dict with pos_2d, pos_3d, visibility, etc.
    # - local_patches: [B, Q, 3, P, P]
    # - dataset_names: list[str]
    # - sequence_names: list[str]

    video = batch["video"]
    targets = batch["targets"]

    # 前向传播
    # outputs = model(video, batch["coords"], batch["t_src"], batch["t_tgt"], batch["t_cam"])

    # 计算 loss
    # loss = compute_loss(outputs, targets)

    break

# ============================================================================
# 示例 4: 从配置文件创建数据集
# ============================================================================

from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset

def create_dataset_from_config(config: dict) -> MixtureDataset:
    """从配置字典创建数据集"""

    # 创建 adapters
    adapters = []
    weights = []

    for ds_config in config['datasets']:
        adapter = create_adapter(
            name=ds_config['name'],
            root=ds_config['root'],
            split=ds_config.get('split', 'train'),
        )
        adapters.append(adapter)
        weights.append(ds_config.get('weight', 1.0))

    # 创建混合数据集
    dataset = MixtureDataset(
        adapters=adapters,
        dataset_weights=weights,
        clip_len=config.get('clip_len', 48),
        img_size=config.get('img_size', 256),
        num_queries=config.get('num_queries', 2048),
        boundary_ratio=config.get('boundary_ratio', 0.3),
        t_tgt_eq_t_cam_ratio=config.get('t_tgt_eq_t_cam_ratio', 0.4),
    )

    return dataset

# 配置示例
config = {
    'datasets': [
        {'name': 'pointodyssey', 'root': '/data/pointodyssey', 'weight': 0.5},
        {'name': 'scannet', 'root': '/data/scannet', 'weight': 0.3},
        {'name': 'kubric', 'root': '/data/kubric', 'weight': 0.2},
    ],
    'clip_len': 48,
    'img_size': 256,
    'num_queries': 2048,
}

dataset = create_dataset_from_config(config)
