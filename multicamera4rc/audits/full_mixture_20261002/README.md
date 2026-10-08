# 全量有效数据混合训练：2026-10-02

后续 TartanAir 天空深度有效域修复及固定验证结果见[修复报告](../tartan_filter_20261002/README.md)。下文保留第一轮混合训练的配置及检查记录；当前续训状态以该报告和实际 active training 指针为准。

已接入 11 个数据集家族、15 个布局/相机流配置。这里的“全量”指所有通过筛选的训练序列和有效帧进入可采样池，不是抽几个序列的小规模配置。官方训练划分保留；没有官方划分的数据按物理场景留出 5% 验证。同场景的左右相机、不同难度或不同天气变体不会跨训练/验证。

| 数据源 | 训练序列/相机流 | 有效帧 | 输入布局 | 实际监督 |
|---|---:|---:|---|---|
| kubric_C2 | 4,752 | 142,560 | C2/T12 | 几何、密集长期轨迹、位移、可见性 |
| kubric_C3 | 4,752 | 142,560 | C3/T8 | 几何、密集长期轨迹、位移、可见性 |
| kubric_C4 | 4,752 | 142,560 | C4/T6 | 几何、密集长期轨迹、位移、可见性 |
| dynamic_replica | 966 | 289,800 | C1/T≤8 | 几何、原生相邻帧运动 |
| pointodyssey | 131 | 236,883 | C1/T≤8 | 几何；运动未知 |
| vkitti2_Camera_0 | 40 | 18,560 | C1/T≤8 | 几何、原生相邻帧运动 |
| vkitti2_Camera_1 | 40 | 18,560 | C1/T≤8 | 几何、原生相邻帧运动 |
| tartanair_left | 354 | 297,279 | C1/T≤8 | 几何；运动未知 |
| tartanair_right | 354 | 297,279 | C1/T≤8 | 几何；运动未知 |
| hypersim | 614 | 59,492 | C1/T1 | 几何；运动未知 |
| scannet | 1,138 | 1,777,906 | C1/T≤8 | 几何；运动未知 |
| scannetpp | 812 | 690,631 | C1/T≤8 | 几何；运动未知 |
| wildrgbd | 18,692 | 6,514,594 | C1/T≤8 | 几何；运动未知 |
| blended | 475 | 109,223 | C1/T1 | 几何；运动未知 |
| taskonomy | 510 | 4,456,268 | C1/T1 | 几何；运动未知 |

Kubric 三种布局复用同一组 4,752 个训练场景；这些行不能相加当作独立场景。DynamicReplica 为 483 个物理序列、966 个相机流。PointOdyssey 为 131 个训练序列；其中 20 个序列的世界坐标手性不同，已做精确坐标约定转换后保留。

一次完整 epoch 包含 5,961,438 个覆盖窗口和同样数量按权重增加的窗口，共 11,922,876 个样本。覆盖窗口逐序列包含所有有效帧；有原生运动的窗口重叠一帧，避免漏掉窗口边界的相邻运动。DistributedSampler 打乱整个 epoch；恢复训练时保存 epoch 和消费位置。额外窗口保持 Kubric/Replica/VKITTI 运动监督的比例，避免图片量较大的几何数据把运动任务淹没。首阶段从 step 5000 继续到 25000，不能把完成这 20,000 步称作已遍历完整 epoch。

## GT 检查及标签边界

原始格式及独立标签核验沿用[前一轮单目 GT 报告](../monocular_20261002/README.md)：Replica 光流端点与顶点 GT、VKITTI scene flow 与独立光流、PointOdyssey 粒子重投影/量化、打包 RGB-D 相邻视图深度均已交叉核对。本轮在此基础上扩大到完整训练池并实现相应 dense recipe。

完整结构筛查读取了所有训练序列的标定行、数据形状和原生文件成员信息。检查 K 正焦距、齐次矩阵、R 正交性/行列式、RGB/深度对应和原生运动文件完整性。PointOdyssey `cab_e1_ego2` 的 2,139 个非法标定帧、Hypersim 的 51 个非法标定帧被排除，剩余有效帧保留。原始数据只读；筛选结果写入独立 manifest。

这不是离线逐像素检查数 TB 数据的声明。每次实际加载都会检查有限值、深度有效域和适用的标签一致性：

- DynamicReplica 从官方 uint16/float16 位编码读取密集光流，结合两个时刻的原生深度、标定得到可见相邻帧的世界位移；检查前后向有效掩码、越界、循环误差和深度边界。没有用 mesh 顶点轨迹补成密集 GT。
- VKITTI2 读取原生 RGB16 scene flow（±10 m）及 optical flow。先在相邻相机坐标中得到终点，再用相邻时刻外参转到统一世界坐标，正确消除相机运动；逐像素用独立 optical flow 检查投影。不会因目标深度遮挡而删掉合法 scene-flow 终点。
- PointOdyssey 原生 16 位深度按 `raw / 65535 * 1000` 解码。世界反射约定用 `E'=E F`、`F=diag(-1,1,1,1)` 转换；`E'_0 inv(E'_t)=E_0 inv(E_t)`，首相机坐标中的 GT 完全不变。这是坐标约定转换，没有拟合或修复非法相机。该适配器只读取密集深度及标定，不使用粒子轨迹。
- Hypersim 使用官方训练划分、场景长度单位、相机投影元数据，把 range 转为 camera-Z；投影带 skew 时按已知单应性校正并更新 K。每个加载像素还用原生 world-position 检查深度与投影。
- TartanAir V1、ScanNet、ScanNet++、WildRGBD、BlendedMVS、Taskonomy 只用现有有效几何标签。没有把移动相机的静态重投影当作未知动态场景的运动 GT。

没有运动标签的批次令 `displacement_valid=false`，不构造零位移/静态监督，不参与运动置信度或速度损失。这些批次训练几何编码器和深度/相机头；训练时跳过没有监督的运动解码计算。动态数据仍使用原来的全部时刻 KV 和完整运动解码器。K/E 只用于 GT 计算与损失，不进入模型输入。

所有有效原生深度像素参与几何监督；运动源图选 Q≤4，每张选中源图的所有有效像素都参与监督，没有 4,096 条稀疏查询，也没有稀疏轨迹插值。未知的非相邻长期运动保持 masked；长期运动由 Kubric 原生 renderer GT 提供。

全无有效深度的可读窗口会写入 `runtime_invalid_clips.jsonl` 并重新采样。标定/标签格式错误或缺失应有的原生运动文件会直接报错，避免静默制造训练标签。

## 原分辨率及计算

`image_mode=native` 保留数据中现有 RGB 的宽高比例与像素，仅向右/下补齐 14 的倍数；padding 不参与监督。Replica 原图 720×1280 → 728×1288，PointOdyssey 540×960 → 546×966，VKITTI 375×1242 → 378×1246，TartanAir 480×640 → 490×644。已有 H5 的图像在打包前已处理过，适配器去掉打包 padding，不能恢复不存在的原始 RAW。Kubric 原生文件本身为 256×256 → 266×266。

同相机 attention 的相机关系偏置对所有有效 key 为常数，会在 softmax 中抵消。因此单相机可以使用数学等价的 SDPA 路径，避免构造巨大的 dense bias；多相机关系偏置仍可学习。已检查前向和梯度等价。attention/visibility query 分块关闭；模型 gradient checkpointing 保留。

## 可复查证据

- `native_gt_preflight.json`：最终 15 个配置 × 3 个真实序列，共 45 个真实 GT 批次，全部通过。
- `giant_native_gpu4_preflight.json`：15 个配置的 giant 模型原分辨率前向/反向全部有限。该检查的序列池是在 PointOdyssey 扩入 20 个反射世界序列之前生成的。
- `world_basis_gpu5_preflight.json`：扩入后的 PointOdyssey `character`、`gso_in_big` 两个真实批次，giant 前向/反向有限。
- `worst_native_ddp.json`：GPU 4/5 两个 rank 同时使用 Replica 原生 T8/Q4，从 5000 恢复模型和优化器，连续两个真实 AdamW 步骤成功。单 GPU Replica 前反向峰值约 159.34 GiB；DDP 压力测试同时包括优化器状态及共享卡上的现有评估进程。
- `checks.json`：最终自动测试 168 passed。包含未知运动 mask、VKITTI 去相机运动和遮挡、Replica 位编码、完整覆盖、DDP 不同任务/unused heads 切换、单相机 attention 梯度等价和反射世界坐标等价。

当前采用 step 5000 作为恢复起点。原固定 24 批验证中，2000→5000 的 depth AbsRel 为 0.01554→0.00708，动态位移 EPE 为 0.02098→0.01509（场景归一化单位）。随后全部 248 个 Kubric 验证场景 × 3 布局、每个 checkpoint 744 批验证也已完成：depth AbsRel 0.01402→0.00886，动态位移 EPE 0.02548→0.01946，轨迹 EPE 0.02441→0.01005，见 `baseline_full_validation.json`。混入新数据后的训练总 loss 不应直接与纯 Kubric 随机批次比较。保留 dense loss；已有 dense_balanced 试验没有更好的固定验证表现。

## 没有混入的数据

CO3D/MegaDepth 的完整结构已检查，但当前包为估计 SfM/MVS 标签，缺少认证原生 GT 所需的质量信息，未接入本轮监督。Waymo 包只有稀疏 LiDAR 深度；HM3D 文件损坏；DTU/TNT 为测试集或缺少深度；DL3DV/Mapillary 标签来源或组织未验证；TartanAir V2 相机模型待验证；另一个 VKITTI 包版本未核实且可能重复。它们没有作为“有效 GT”混入。

## 复现

在 B300zbf 的项目根目录，使用项目 `.venv/bin/python`：

```bash
.venv/bin/python -m scripts.build_full_dense_mixture \
  --output outputs/training_preflight/full_mixture_20261002 \
  --metadata outputs/training_preflight/monocular_gt_audit_20261001_1536
.venv/bin/python -m scripts.validate_dense_mixture \
  --data-config outputs/training_preflight/full_mixture_20261002/full_mixture_native.yaml \
  --output outputs/training_preflight/full_mixture_20261002/native_gt_preflight.json \
  --sequences-per-source 3 --source-count 4 --device cpu
```

`full_mixture_native.yaml` 为实际机器配置快照，manifest 和 index cache 由 builder 在服务器生成。官方 Hypersim 元数据来自 [相机参数](https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv)和[公开帧/训练划分](https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv)。

## 已启动的混合训练

服务器目录 `/data9/zbf/my_dfrt_multicamera4rc`，运行 `full_dense_mixture_gpu45_20261002_032608`，启动记录见 `training_job.json`。使用 GPU 4/5、giant、BF16、全局 BS=2（每卡 1）、每卡 2 个 DataLoader spawn workers、Q≤4；原生分辨率，attention/visibility 不分块。模型与 AdamW 状态从 step 5000 恢复，更换数据池后重置数据游标；head LR=3e-5、backbone LR=3e-6、200 步 warmup、cosine 衰减至 0.1 倍。目标总 step=25000。

每 2500 全局步保存完整训练状态，保留本轮的全部定期 checkpoint；当前磁盘可用约 217.7 GiB，9 份 checkpoint（包含最后 latest）预计约 153.3 GiB。此前训练、压力测试和评估结果保留。后台 supervisor 会记录失败原因及退出码；混合阶段结束后自动在同一固定验证集上比较 base5000 和 mixed25000。

实时运行状态写入服务器 `outputs/training_preflight/active_training.json` 和本轮 `control_dir/progress.json`；`control_dir/supervisor.log` 记录两个 rank 的数据集、输入大小、真实监督像素数、loss、梯度与优化器更新状态。

启动健康快照 `initial_mixed_training.json` 记录了 87 次实际混合优化更新（至 step 5087）。全部 15 个配置均已出现，2-worker spawn 数据加载和跨任务 DDP 切换已实际运行；所有数值指标有限，跳过更新 0 次，无有效深度批次剔除 0 次。该快照用于验证启动健康，不代表长期收敛或完成全库 epoch。
