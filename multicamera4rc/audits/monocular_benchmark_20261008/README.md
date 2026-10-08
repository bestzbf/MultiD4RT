# 55,500 步模型的 D4RT 单目几何评估（2026-10-08）

**已完成 ScanNet test100 的 100/100 场景、Sintel Final 的指定 14/14 序列，零失败。** 使用在首次正式 GPU 推理前已经通过当前质量面板、被接受为最佳参考的 55,500 步 checkpoint；两组评估固定同一份权重，未随训练继续更新。[checkpoint 选择证据](checkpoint_selection.json) 保存当时的质量参考与选择时间。

## 实测均值

每场景先计算完整片段指标，再按场景等权平均。AbsRel、SqRel、RMSE、ATE、RPE 和点云误差越低越好；δ 和 AUC 越高越好。RMSE 与点云/位置误差单位为 m，RPE-R 单位为 degree。

| 指标 | ScanNet：100 场景 | Sintel：14 序列 |
| --- | ---: | ---: |
| Depth S AbsRel | 0.020179 | 0.255322 |
| Depth SS AbsRel | 0.017680 | 0.202269 |
| Depth S SqRel | 0.005230 | 1.615330 |
| Depth SS SqRel | 0.004894 | 1.427299 |
| Depth S RMSE (m) | 0.103738 | 5.052541 |
| Depth SS RMSE (m) | 0.099642 | 5.052532 |
| Depth S log RMSE | 0.046253 | 0.405190 |
| Depth SS log RMSE | 0.044759 | 0.423116 |
| Depth S δ1 (%) | 99.2457 | 69.4671 |
| Depth SS δ1 (%) | 99.2384 | 73.6312 |
| Depth S δ2 (%) | 99.7425 | 82.7118 |
| Depth SS δ2 (%) | 99.7514 | 86.7400 |
| Depth S δ3 (%) | 99.8977 | 88.6022 |
| Depth SS δ3 (%) | 99.9011 | 92.0718 |
| Pointcloud L1：clip scale + mean shift (m) | 0.036766 | 1.594105 |
| Pointcloud L1：clip Sim3 (m) | 0.031174 | 1.481982 |
| Camera ATE：Sim3 后平均位置误差 (m) | 0.029777 | 0.055105 |
| Camera RPE-T (m) | 0.012745 | 0.022647 |
| Camera RPE-R (degree) | 0.378413 | 0.294246 |
| Legacy camera AUC@30 (%) | 43.6590 | 20.0952 |

这里 S 是逐帧 median 比值尺度对齐；SS 是仓库旧脚本的 median-centred 稳健尺度与偏移对齐。点云只对整个 clip 对齐一次，没有把逐帧深度拟合用于修补三维重建。点云误差为逐对应点/坐标平均 L1；这不是 Chamfer distance。ATE 沿用旧实现的平均 Euclidean 位置误差，并非轨迹 RMSE。公式、输入差异和命令见[评估协议说明](../../../docs/monocular-evaluation.md)。

## 如何解读

该固定计划下，模型在 ScanNet 的室内 RGB-D 几何上误差较低；Sintel 的深度与三维形状仍有明显误差。其中 `ambush_5`、`ambush_4`、`cave_2` 的 Depth S AbsRel 分别为 `0.7061`、`0.5146`、`0.4739`；这三个序列仍保留在正式 14 序列均值中。

Sintel 的位姿 AUC 还有旧函数定义上的边界：[GT 运动审计](sintel/gt_pose_motion_audit.json) 核对了 588 个相邻对，其中 94 对 GT 平移小于 `1e-8 m`、161 对小于 `1e-5 m`。`bandage_1/2` 都是零平移，旧函数把无定义的 translation-direction 角度算成约 90°，所以两者 AUC 都为零；`ambush_2/7` 的近零平移方向也受数值误差影响。这里保留旧指标，没有按这些结果删序列；不能仅凭总体 AUC 或静止轨迹的 Sim3 ATE 判断所有相机预测。

这次只评估 encoder 的 depth / K / c2w 几何输出，未调用 motion decoder，不能由此推断 scene flow 或长期三维追踪质量。模型输入保持长宽比，最长边 518；旧 D4RT 网络输入拉伸至 256×256。相机也由当前 camera head 输出，旧 D4RT 则从点查询估计。误差函数被复用，但本结果不是控制了这些差异的 D4RT 对照实验。

## 覆盖与 GT 检查

| 项目 | ScanNet | Sintel |
| --- | --- | --- |
| 网络输入 | C1，T48，stride 4 | C1，T20–48，连续帧 |
| 真实帧数 | 4,800 | 602 |
| 原生 RGB H×W | 968×1296 | 436×1024 |
| 等比缩放内容 H×W | 387×518 | 221×518 |
| patch padding 后 H×W | 392×518 | 224×518 |
| 评分网格 | 256×256 | 256×256 |
| 有效 GT 范围 | 0 < depth < 10 m | 0 < depth < 70 m |
| 实际有效评分像素 | 272,317,494 | 38,275,233 |
| 实际相邻位姿对 | 4,700 | 588 |
| 单进程实测峰值显存 | 14.836 GiB | 11.860 GiB |

GT 内外参不输入网络，只用于评分。先核验全部指定帧的深度、位姿有限性和刚体性质，再推理；每个片段的校准及 GT 深度指纹须与审计一致。所有有效评分像素参与，没有 confidence 排序、稀疏查询抽样或结果筛选。

ScanNet 测试名单与旧 D4RT test100 结果完整一致，包括隔离目录的 `scene0727_00`、`scene0736_00`、`scene0791_00`、`scene0802_00`；这四个场景所选 48 帧全部通过 GT 检查。当前混合训练 manifest 与 test100 的物理场景交集为空，见[训练/测试场景证据](scannet_split_proof.json)。[当前混合数据快照](training_dataset_snapshot.json) 没有 Sintel；初始 4RC 权重的预训练数据重叠未在本次审计范围内。

Sintel 原先在服务器没有完整可读的评估副本。本次从官方地址准备 Final RGB 和 2015-03-05 修正相机外参的 depth 包，保存归档大小、SHA256、解压文件 CRC 校验及 14 序列 RGB/depth/camera 文件数，见[数据来源证据](sintel_assets_provenance.json)。短序列保留实际 T，未复制尾帧。

## 代码、运行与复现

- 推理代码 commit：`72b97d43fd0397ec86578fc01fd4ecf9aedf929c`；独立 worktree `/data9/zbf/my_dfrt_monocular_eval_20261008` 在运行结束时无修改。
- checkpoint：`/data8/zbf/multicamera4rc_runs/scaleout_immediate_20261008/continuation_55000_55500_try0/latest.pt`，保存的 step 为 55,500。
- 输出：`/data8/zbf/multicamera4rc_runs/monocular_benchmark_20261008`。
- 正式推理使用物理 GPU0；GPU2/3/4/5 的训练继续运行。19:39 的[健康快照](training_health_after_evaluation.json)确认四个训练进程在原物理卡上，训练已记录到 55,565 步。生产 worktree 仍为 `3a4cc688cc6bb9b4e41b77332a55c21e9de0c7b3`，本次未修改训练状态或调度策略。
- 本地及服务器 **13 项**针对性检查通过，包括文件单位/矩阵、短序列、不重复帧、padding/K、旧 S/SS 逐值一致、尺度漂移，以及实际 tiny checkpoint 推理/恢复。服务器输出见[测试证明](unit_tests_cuda_fix.json)。
- [收尾检查脚本](postflight.py)复核两个报告均通过、精确场景/帧覆盖、有限指标、计划指纹、checkpoint 一致、场景等权均值和生产代码版本。运行快照及源文件 SHA256 在[完整汇总](summary.json)中。

## 结果文件

- [完整汇总 JSON](summary.json)；[全部 114 场景 CSV](per_scene.csv)。
- ScanNet：[固定计划](scannet/fixed_plan.json)、[完整 GT 审计](scannet/gt_audit.json)、[逐场景结果](scannet/evaluation.json)。
- Sintel：[固定计划](sintel/fixed_plan.json)、[完整 GT 审计](sintel/gt_audit.json)、[逐序列结果](sintel/evaluation.json)。

官方资料：[Sintel depth / camera](https://sintel.is.tue.mpg.de/depth)、[Sintel RGB 下载](https://sintel.is.tue.mpg.de/downloads)、[ScanNet](https://github.com/ScanNet/ScanNet)。
