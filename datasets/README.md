> 此目录是历史 D4RT 数据管线。当前多相机模型使用独立的 `multicamera4rc/datasets/`，其 GT、坐标、原分辨率和混合采样约定见[当前数据文档](../docs/data.md)。

# D4RT 混合训练数据加载器

## ✅ 状态：已完成，可用于训练

**测试日期**: 2026-03-28
**性能**: 7.27 batches/s（目标：2-3 batches/s）✅

---

## 实现内容一览

| 模块 | 文件 | 功能 | 状态 |
|------|------|------|------|
| **第1层：Dataset Adapter** | | | |
| 统一数据结构 | `adapters/base.py` | `UnifiedClip`：所有数据集的统一中间表示（images/depths/normals/trajs/intrinsics/extrinsics/metadata） | ✅ |
| PointOdyssey | `adapters/pointodyssey.py` | 读取 RGB + 深度 + 法向量 + 2D/3D 轨迹，支持 .npz/.h5 预计算 | ✅ |
| PointOdyssey Fast | `adapters/pointodyssey_fast.py` | 优化版，用分离的 .npy 文件加速加载，自动过滤损坏数据 | ✅ |
| Dynamic Replica | `adapters/dynamic_replica.py` | 读取 RGB + 深度 + 光流 + 2D/3D 轨迹，PyTorch3D 相机格式转换 | ✅ |
| Kubric | `adapters/kubric.py` | 读取 RGB + 深度 + 2D/3D 轨迹 | ✅ |
| ScanNet | `adapters/scannet.py` | 读取 RGB + 深度，支持多种相机格式（c2w/w2c/NDC） | ✅ |
| Co3Dv2 | `adapters/co3dv2.py` | 读取 RGB + 深度，PyTorch3D NDC 相机格式转换，自动加载预计算 tracks/normals | ✅ |
| BlendedMVS | `adapters/blendedmvs.py` | 读取 RGB + 深度 + 相机参数 | ✅ |
| MVS-Synth | `adapters/mvssynth.py` | 读取 RGB + 深度（EXR格式）+ 相机参数 | ✅ |
| TartanAir | `adapters/TartanAir.py` | 读取 RGB + 深度 + 相机参数 | ✅ |
| VirtualKitti2 | `adapters/VirtualKitti.py` | 读取 RGB + 深度 + 相机参数 | ✅ |
| Waymo | `adapters/Waymo.py` | 读取 RGB + 深度 + 轨迹（需要 tensorflow） | ✅* |
| 数据集注册表 | `registry.py` | `create_adapter(name, root, split)` 统一接口创建任意 adapter | ✅ |
| **第2层：Mixture Sampler** | | | |
| 单数据集采样 | `sampling.py` | `DatasetSampler`：按序列加权采样，支持 random/sequential clip 截取 | ✅ |
| 多数据集混合 | `mixture.py` | `MixtureSampler`：可配置各数据集采样权重，归一化处理 | ✅ |
| 混合数据�� | `mixture.py` | `MixtureDataset`：标准 PyTorch Dataset，组合三层逻辑 | ✅ |
| **第3层：Query Builder** | | | |
| 几何变换 | `transforms.py` | `GeometryTransformPipeline`：随机 crop + resize，同步更新 intrinsics/trajs_2d/visibility | ✅ |
| Query 采样 | `query_builder.py` | 采样 2048 queries，30% 边界过采样，40% t_tgt=t_cam | ✅ |
| Supervision 构造 | `query_builder.py` | 构造 pos_2d/pos_3d/visibility/displacement/normal 及对应 mask | ✅ |
| 双路径支持 | `query_builder.py` | has_tracks=True：从 2D/3D 轨迹构造全部 supervision；has_tracks=False：从 depth 反投影计算 pos_3d，mask_2d/mask_vis/mask_disp 置 False | ✅ |
| **辅助工具** | | | |
| Batch collate | `collate.py` | `d4rt_collate_fn`：正确处理 tensor stacking 和 metadata | ✅ |
| 预计算加速 | `adapters/base.py` | `load_precomputed_fast()`：优先读 .h5（按需加载），回退到 .npz | ✅ |

> *Waymo 需要安装 `tensorflow`，其余数据集无额外依赖。

---

## 快速开始

```python
from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset
from datasets.collate import d4rt_collate_fn
from torch.utils.data import DataLoader

# 1. 创建 adapters
adapters = [
    create_adapter("pointodyssey",    root="/data2/d4rt/datasets/PointOdyssey",               split="train"),
    create_adapter("dynamic_replica", root="/data1/d4rt/datasets/Dynamic_Replica",             split="train"),
    create_adapter("kubric",          root="/data2/d4rt/datasets/kubric",                      split="train"),
    create_adapter("scannet",         root="/data2/d4rt/datasets/scannet/scannet",             split="train"),
    create_adapter("co3dv2",          root="/data2/d4rt/datasets/Co3Dv2",                     split="train"),
    create_adapter("blendedmvs",      root="/data2/d4rt/datasets/BlendedMVS",                 split="train"),
    create_adapter("mvssynth",        root="/data2/d4rt/datasets/MVS-Synth/GTAV_1080",        split="train"),
    create_adapter("tartanair",       root="/data2/d4rt/datasets/TartanAir",                  split="train"),
    create_adapter("vkitti2",         root="/data2/d4rt/datasets/VirtualKitti",               split="train"),
]

# 2. 创建混合数据集
dataset = MixtureDataset(
    adapters=adapters,
    dataset_weights=None,  # 均匀采样；或传入 [w1, w2, ...] 自定义权重
    clip_len=8,            # 论文设定：48，测试用：8
    img_size=256,          # 论文设定：256×256
    num_queries=2048,      # 论文设定：2048
    boundary_ratio=0.3,    # 论文设定：30% 边界过采样
    t_tgt_eq_t_cam_ratio=0.4,  # 论文设定：40% t_tgt=t_cam
    seed=42,
)

# 3. 创建 DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=16,          # 关键：必须用多线程
    collate_fn=d4rt_collate_fn,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
)

# 4. 训练循环
for batch in loader:
    video      = batch["video"]       # [B, T, 3, H, W]
    coords     = batch["coords"]      # [B, Q, 2]  — query 坐标 (u,v) in [0,1]
    t_src      = batch["t_src"]       # [B, Q]     — source 帧索引
    t_tgt      = batch["t_tgt"]       # [B, Q]     — target 帧索引
    t_cam      = batch["t_cam"]       # [B, Q]     — camera 帧索引
    intrinsics = batch["intrinsics"]  # [B, T, 3, 3]
    extrinsics = batch["extrinsics"]  # [B, T, 4, 4]  w2c
    targets    = batch["targets"]     # dict，见下方
    # targets["pos_2d"]       [B, Q, 2]   — 2D 目标位置
    # targets["pos_3d"]       [B, Q, 3]   — 3D 目标位置（相机坐标）
    # targets["visibility"]   [B, Q]      — 目标可见性
    # targets["displacement"] [B, Q, 2]   — 2D 位移
    # targets["normal"]       [B, Q, 3]   — 法向量
    # targets["mask_2d"]      [B, Q]      — 2D supervision 有效 mask
    # targets["mask_3d"]      [B, Q]      — 3D supervision 有效 mask
    # targets["mask_vis"]     [B, Q]      — visibility supervision 有效 mask
    # targets["mask_disp"]    [B, Q]      — displacement supervision 有效 mask
    # targets["mask_normal"]  [B, Q]      — normal supervision 有效 mask
    ...
```

---

## 数据集一览

| 数据集 | 序列数 | 原始 tracks | 预计算 tracks | pos_3d 来源 | has_depth | 数据路径 |
|--------|--------|------------|--------------|------------|-----------|----------|
| PointOdyssey | 131 | ✅ 有 | - | 从 tracks 反投影 | ✅ | `/data2/d4rt/datasets/PointOdyssey` |
| Dynamic Replica | 966 | ✅ 有 | - | 从 tracks 反投影 | ✅ | `/data1/d4rt/datasets/Dynamic_Replica` |
| Kubric | 755 | ✅ 有 | - | 从 tracks 反投影 | ✅ | `/data2/d4rt/datasets/kubric` |
| ScanNet | 21 | ❌ 无 | ✅ 已计算 (21/21) | 从预计算 tracks | ✅ | `/data2/d4rt/datasets/scannet/scannet` |
| Co3Dv2 | 31834 | ❌ 无 | ✅ 已计算 (31834/31834) | 从预计算 tracks | ✅ | `/data2/d4rt/datasets/Co3Dv2` |
| BlendedMVS | 106 | ❌ 无 | ✅ 已计算 (106/106) | 从预计算 tracks | ✅ | `/data2/d4rt/datasets/BlendedMVS` |
| MVS-Synth | 120 | ❌ 无 | ✅ 已计算 (120/120) | 从预计算 tracks | ✅ | `/data2/d4rt/datasets/MVS-Synth/GTAV_1080` |
| TartanAir | 4 | ❌ 无 | ✅ 已计算 (4/4) | 从预计算 tracks | ✅ | `/data2/d4rt/datasets/TartanAir` |
| VirtualKitti2 | 50 | ❌ 无 | ✅ 已计算 (50/50) | 从预计算 tracks | ✅ | `/data2/d4rt/datasets/VirtualKitti` |
| Waymo | — | ❌ 无 | ✅ 已计算 | 从预计算 tracks | ✅ | `/data2/d4rt/datasets/Waymo` |

> **说明**:
> - **原始 tracks**: 数据集本身提供的 2D/3D 轨迹标注
> - **预计算 tracks**: 通过 RAFT/CoTracker 等算法离线计算的轨迹，存储在 `precomputed.npz` / `precomputed.h5` 中
> - **pos_3d 来源**: Query Builder 如何构造 3D position supervision
> - **所有预计算数据现已自动加载**（默认 `precompute_root=root`）
> - Co3Dv2 的 `precomputed.npz` 直接存放在各序列目录下（`<root>/<category>/<seq>/precomputed.npz`），与其他数据集相同

### Co3Dv2 当前默认过滤（2026-04-23）

- 训练配置当前默认同时使用：
  - 本地 sibling overlay：`<root>/<category>/<seq>/precomputed.overlay.npz`
  - `sequence_denylist: /data/zbf/openclaw/d4rt/configs/co3dv2_denylist_degenerate_clips_20260422.txt`
- 2026-04-23 已把外部 overlay 根目录中的文件搬回 Co3D 数据集目录中：
  - 原路径：`/data2/d4rt/datasets/Co3Dv2_track_overlay_20260421/<category>/<seq>/precomputed.npz`
  - 当前路径：`/data2/d4rt/datasets/Co3Dv2/<category>/<seq>/precomputed.overlay.npz`
- 对失败序列，marker 也已搬回 base 目录：
  - `precomputed.failed.json`
- denylist 来自两层审计：
  - 序列级旧缓存审计：`datasets/computer/co3d_audit_precomputed.py`
  - clip 级随机采样审计：`datasets/computer/co3d_audit_clip_quality.py`
- clip 级审计按真实训练设定评估：`clip_len=48`、`sampling_mode=stride`、`num_samples=32`。
- 当前保守阈值为：
  - `sampled_bad_clip_ratio >= 0.9`
  - `sampled_zero_clip_ratio >= 0.3`
- 2026-04-23 这轮全量审计筛出 `780` 条序列；再与人工复核的日志问题样本并集后，当前 denylist 共 `788` 条。
- 在 `fewview_train/train + require_pointcloud=true` 下，Co3Dv2 可训练序列当前为 `31046` 条。
- 归档说明见：
  - `md/Co3Dv2_clip质量审计与denylist记录_2026-04-23.md`

> **注意**: Waymo 需要安装 `tensorflow`（`pip install tensorflow`），其余数据集无额外依赖。

---

## 性能基准

> 测试环境：Linux，PointOdyssey 单数据集，batch_size=4，clip_len=8，img_size=256，num_queries=2048

| num_workers | 吞吐量 | 达标？ |
|-------------|--------|--------|
| 0（单线程） | 1.13 batches/s | ❌ |
| 16 | **7.27 batches/s** | ✅ |

**结论：必须使用 `num_workers >= 8`，推荐 16。**

---

## 架构说明

```
第1层  DatasetAdapter      每个数据集一个，读原始文件 → UnifiedClip
第2层  MixtureSampler      控制多数据集混合采样权重
第3层  D4RTQueryBuilder    统一构造 query supervision（全数据集共享）
```

新增数据集只需在 `adapters/` 下添加一个继承 `BaseAdapter` 的类，并在 `registry.py` 中注册即可。

---

## 注意事项

1. **不要用单线程**：`num_workers=0` 速度极慢（1.13 batches/s），训练会卡在数据加载。
2. **Co3Dv2 序列极多**（31834个），混合训练时建议降低其采样权重，避免其他数据集被压制。
3. **PointOdyssey_fast**（`/data2/d4rt/datasets/PointOdyssey_fast`）：部分序列数据损坏，adapter 已自动过滤，可用 109 个序列。
