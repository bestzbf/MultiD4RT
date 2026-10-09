# 参考 D4RT 的单目几何评估

[返回 README](../README.md) · [推理与权重](inference.md)

`multicamera4rc/benchmarks/monocular.py` 将仓库旧 D4RT 的 Sintel / ScanNet 数据及误差计算适配到 MultiCamera4RC。它加载 native checkpoint 保存的架构及时间语义，以 `camera_id=0`、真实抽取帧索引、RGB 作为输入。内外参和深度只用于评估，全部有效评估像素参与深度与点云评分，没有置信度筛选或 4,096 查询采样。

已完成的实测见 [2026-10-08 / 55,500 步报告](../benchmarks/results/monocular_55500/README.md)：ScanNet 100/100、Sintel 14/14，零失败，保留固定计划及全部逐场景误差。

## 范围与输入

- Sintel：Final pass 的 14 序列列表与 `test/eval_sintel_checkpoint.py` 相同。每段取最多 48 个连续真实帧；较短序列直接输入实际帧数，不复制末帧产生虚假的零误差相邻对。GT 使用官方 2015-03-05 修正相机外参的深度包。
- ScanNet：复用 D4RT test100 场景清单，从 processed 有效帧顺序取 48 帧、stride 4。读取已导出的 RGB、毫米深度 PNG、w2c 位姿文件及 H5 内参；不加载预计算 normals / tracks。逐帧核对位姿文件与 H5 缓存一致。放在隔离目录里的场景也须完成所选帧的 GT 核验，不能按模型分数删去。
- 网络 RGB 等比缩放至长边最多 518，再在右/下补齐 14 的 patch 倍数。ScanNet `968×1296 → 387×518 → 392×518`；Sintel `436×1024 → 221×518 → 224×518`。
- 评分使用旧评估器的 `256×256` 网格。GT depth 采用旧 adapter / transform 的 nearest resize 和内参行缩放；预测移除 patch padding 后重采样，内参按数组像素中心映射一次。

原 D4RT 网络输入为拉伸的 `256×256`。本适配保持当前项目的长宽比约定，输入分辨率因此不同；旧汇总与本结果不能作为控制了全部变量的模型对照。

## 指标定义

复用 `multicamera4rc/benchmarks/metrics.py` 和 `multicamera4rc/benchmarks/camera.py` 的误差函数，深度逐帧对齐函数与旧 Sintel 评估器逐值测试一致。旧部分 summary 的文字称作 global sequence alignment，但现有评估函数实际执行逐帧拟合，本入口按实际代码记录。

| 指标 | 定义及单位 |
| --- | --- |
| Depth S | 每帧以 `median(GT/pred)` 对齐尺度；报告 AbsRel、SqRel、RMSE、log RMSE、δ1/δ2/δ3 |
| Depth SS | 每帧 median-centred 比值的稳健尺度及 median-residual 平移，沿用仓库旧函数；不是 disparity least-squares 对齐 |
| 有效 depth | Sintel `0<GT<70m`，ScanNet `0<GT<10m`，排除非有限 GT；预测非有限或非正值会使案例失败，不通过丢掉预测像素改善分数 |
| 点云 scale + mean shift | 所有帧进入参考相机坐标，整个片段只拟合一个深度尺度，再平移对齐中心，逐对应点/坐标平均 L1，单位 m |
| 点云 Sim3 | 整个片段一次 Umeyama 对应点相似变换后的 L1，单位 m；不进行最近邻 ICP，也不逐帧修补形状 |
| ATE | 仓库 `compute_pose_metrics` 的 Sim3 轨迹对齐后平均 Euclidean 位置误差，单位 m；这个实现的 ATE 是均值，不是轨迹 RMSE |
| RPE-T / RPE-R | 同一轨迹对齐后相邻相机对的平移误差 m、旋转误差 degree |
| Pose AUC@30 | 未用 GT 对齐的相邻相对位姿，rotation / translation-direction 两种角误差的最大值所构成 AUC，报告百分比 |

模型训练输出使用场景归一化坐标，所以点云结果主要使用整个片段的 scale + mean shift 及 Sim3。另保存旧 mean-shift-only 形式的 `pointcloud_l1_raw_gauge` 供诊断；该量混合预测场景单位与 GT 米制，不能当作与旧 D4RT 同单位的米制精度。深度逐帧 S/SS 对齐不会被拿来调整点云或相机。

相机直接使用当前模型的 `c2w` head；旧 D4RT 是用点查询及刚体 Umeyama 估计相机。这里复用误差定义，但不将新模型的 head 输出伪装成旧模型的查询解码输出。这个入口评估深度、逐帧点云与相机几何，没有调用 motion decoder；scene flow / 长期三维追踪使用各自的动态评估入口。

旧 translation-direction 误差函数在 GT 零平移时仍归一化零向量，得到约 90°，而该方向实际上无定义。Sintel 的 `bandage_1/2` 各 47 个相邻对属于这一情况；`ambush_2/7` 也存在微米量级的近零平移。正式 AUC 保留旧函数，须结合 [GT 运动审计](../benchmarks/results/monocular_55500/sintel/gt_pose_motion_audit.json) 解读。Sim3 轨迹对齐也可能将静止 GT 的预测轨迹缩成点，ATE/RPE-T 不能单独说明这类片段的位姿质量。

汇总先计算每场景完整 clip 的指标，再对场景等权平均；不能把若干困难场景排除后仍标成 test100。报告包含全部固定帧索引、输入尺寸、逐场景误差、峰值显存、完整 checkpoint 配置与计划指纹；任何失败或缺失场景都会把最终状态标为 `failed`。

## 运行

先核验正式计划的所有样本：

```bash
python -m multicamera4rc monocular \
  --dataset scannet \
  --roots /data4/d4rt_dataset/scannet/scans_test \
          /data4/d4rt_dataset/scannet/invalid_scenes/scans_test \
  --num-frames 48 --frame-stride 4 \
  --long-side 518 --eval-size 256 \
  --plan /path/to/scannet/fixed_plan.json \
  --output /path/to/scannet --audit-only
```

再用完全相同的计划推理：

```bash
CUDA_VISIBLE_DEVICES=0 python -m multicamera4rc monocular \
  --dataset scannet \
  --roots /data4/d4rt_dataset/scannet/scans_test \
          /data4/d4rt_dataset/scannet/invalid_scenes/scans_test \
  --checkpoint /path/to/accepted_checkpoint.pt \
  --long-side 518 --eval-size 256 \
  --plan /path/to/scannet/fixed_plan.json \
  --output /path/to/scannet --memory-fraction .5 --resume-results
```

Sintel 改为 `--dataset sintel --roots /path/to/sintel --frame-stride 1`，使用自己的 plan / output。默认恰好评估 14 个指定场景；需保证 `training/final`、`training/depth` 和 `training/camdata_left` 都已完整准备。`--long-side 0` 保留原生 RGB 分辨率，须建立新的计划及 GT 审计。新的 plan 可以用 `--scene-names` 固定场景清单；计划一旦存在，其案例及指纹是正式评估范围。

代码测试包含 SDK 深度/相机文件、ScanNet 毫米单位和缓存位姿、保持比例、去 padding 与 K 变换、无重复帧、与 D4RT 逐帧 S/SS 逐值一致、全 clip 尺度与逐帧漂移区分，以及实际 tiny checkpoint 推理和恢复。

官方数据说明：[Sintel depth / camera](https://sintel.is.tue.mpg.de/depth)、[Sintel RGB 下载](https://sintel.is.tue.mpg.de/downloads)、[ScanNet 数据与传感器格式](https://github.com/ScanNet/ScanNet)。
