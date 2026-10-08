# 可变相机、可变时间长度与 50k 续训

新增 `balanced_coverage`：数据集采样比例由 `weight` 决定，数据池大小只决定池内覆盖，不再改变数据集占比。旧 `coverage` 的前半个 epoch 按全库窗口数量分配，Taskonomy 的巨大静态图池使其实际占比约 35%，即使显式权重只有 0.2。本次将其固定为 2%。

## 数据集比例

| 数据集族 | 比例 |
|---|---:|
| Kubric | 35% |
| DynamicReplica | 12% |
| PointOdyssey | 12% |
| vKITTI2 | 11% |
| TartanAir V1 | 4% |
| TartanAir V2 | 7% |
| Hypersim | 4% |
| ScanNet | 3% |
| ScanNet++ | 3% |
| WildRGBD | 3% |
| BlendedMVS | 4% |
| Taskonomy | 2% |

vKITTI2/V1 的族内比例为双目 60%、左单目 20%、右单目 20%。V2 为六个方向的同朝向双目各 15%、左前单目 10%。DynamicReplica 保留左右单目，右流缺少原生 flow 的窗口仍只有几何监督。70% 是动态数据集族的采样比例，不等于每个批次都含动态物体标签。

每个 epoch 100,000 个位置，用最大余数法分配各源配额；DDP 随机打乱全部位置，各卡独立选择不同 C/T。每个源通过常数空间的仿射置换轮换原生覆盖窗口，所有准入序列仍可访问。无有效深度窗口只在同一个源内重试并记录，避免重试改变数据集比例。有限训练步数不保证数百万窗口已经全部看过一遍。

## 相机与时间课程

Kubric 全池实际为 4,752 个训练场景，均有 10 台相机；3,564 个场景有 24 帧，1,188 个场景有 48 帧。T32 只对原生长度足够的场景采样；原生长度较短的场景不会被移出训练池，也不会重复帧凑长度。

Kubric 的目标相机分布为 C1 15%、C2 17%、C3 14%、C4 18%、C5 6%、C6 7%、C7 6%、C8 7%、C9 5%、C10 5%。这是 Kubric 内部的目标分布；剔除不适用于当前场景的 C/T 组合后，剩余组合重新归一化，因此实际边际比例可能略有变化。物理相机子集每次重新采样，保留真实 camera ID。

| 总训练步 | 单目 T 概率 | 双目 T 概率 | Kubric 最大 C×T |
|---|---|---|---:|
| 25k→32k | T4/8/12/16 = 10/45/30/15% | T4/6/8 = 35/50/15% | 40 |
| 32k→40k | T8/12/16/24 = 25/25/30/20% | T4/6/8/12 = 20/35/30/15% | 48 |
| 40k→50k | T8/12/16/24/32 = 10/15/25/30/20% | T4/6/8/12 = 15/20/35/30% | 48 |

Kubric 使用联合 C/T 分布，不能独立采样相机数量和时间长度。C5–C10 保留 T2/T4，后期 C5–C8 加入 T6；C9/C10 保持 T2/T4。C2/C3/C4 后期分别可到 T24/T16/T12，具体概率在阶段 YAML 和 builder 中。

表中的单目分布是通用上限。原分辨率运动头的显存预检要求 DynamicReplica 最大 T16，PointOdyssey/vKITTI2 单目最大 T24；更长 T 的概率合并到各自上限，不删除短时间组合。几何单目和 Kubric 的合格 48 帧场景保留 T32。

Kubric、PointOdyssey 的原生三维轨迹支持 stride 1/2/4；几何数据也可使用这些间隔。PointOdyssey T24/stride4 覆盖 93 个原始帧的位置范围。时间输入使用原始整数帧号，保留间隔；速度损失按真实帧号差归一化，单位仍为每帧。没有要求外部 timestamp，也没有把不同数据集的帧率当成相同物理秒数。

DynamicReplica/vKITTI2 保持 stride1。增加上下文 T 不会让一帧 flow 自动变成长期位移 GT：非相邻未知运动仍屏蔽；长期三维追踪标签由 Kubric 和 PointOdyssey 提供。Hypersim/BlendedMVS/Taskonomy 保留静态多视图 T1。

## 网络输入与计算

```text
images          [B, C*T, 3, H, W]   原始像素，右/下仅补齐 patch14
camera_id       [B, C*T]            同一物理相机的帧共享 ID
time_idx        [B, C*T]            同步相机共享相同的原始帧号
query_time_idx  [B, T]
source_obs_idx  [B, Q]
trajectory GT   [B, Q, T, H, W, 3]
```

使用 giant、BF16、GPU4/5、每卡 BS1、全局 BS2。K/E 只用于 GT。模型参数结构未更改，恢复 AdamW、模型与各 rank RNG。原生图像分辨率和长时间输入不降低，attention/visibility 查询不分块。

**2026-10-03 代码复核更正：** 此次审计测量使用的 `7a2038a` 版本是 encoder 融合所有输入，decoder 直接 KV 只取目标时刻全部相机；此前“decoder 的 KV 包含所有时刻”的文字不准确。`measurements` 中的 GT、显存和 DDP 结果对应这一历史模式。后续框架更新新增 `decoder_memory=all_observations`，并保留 `target_time` 的旧断点兼容模式；不能把这些历史显存记录当作新模式的实测。详见[模型结构与版本语义](../../../docs/architecture.md)。

长序列的完整 motion head 显存很大：Q4/T32 PointOdyssey 在 133.85 GiB 的共享卡预检进程上限下 OOM，不能标记为通过。Q2/T32 也超过 147.23 GiB 的共享进程预算。当前先限制 PointOdyssey/vKITTI2 单目到 T24、Replica 到 T16，避免预检过程中挤占现有训练。采用 `--source-count 4 --max-source-time-pairs 64`，使 Q≤min(4,floor(64/T))，至少 1 个参考帧；T≤16 通常 Q4，T24/T32 为 Q2。减少的是每次解码的参考图数量；选中参考图的全部有效像素或全部有效原生粒子仍参与监督，没有像素/粒子查询配额。所有输入图仍训练几何头，没有运动 GT 的批次保留几何快速路径。

25k 续训的 peak LR 为 backbone 1e-6、其余头 1e-5，500 步 warmup 从 0.3 倍开始，接续原 25k 末端的 3e-7/3e-6；之后 cosine 到 50k 时的 0.1 倍。`--lr-end-step 50000` 将学习率终点与阶段停止步分开；首次 25k 接入新采样器时重置数据游标；32k/40k 保留相同的 100,000 位置配额及已消费游标，只改变 C/T 课程，不重复从同一批位置重来，也不重启 LR。完整 checkpoint 每 2,500 步保存；阶段末另存 latest。

## 验证与运行

`measurements` 保存实际服务器配置和检查记录：数据池保留、阶段配额、33 个原生窗口的 RGB/depth/K/源时刻几何身份检查、PointOdyssey T24/stride4 的原生 XYZ 对照、giant 原分辨率 BF16 反传、恢复 AdamW 的双卡更新与 macOS/B300zbf 均 220 项测试结果。GT 核验是代表窗口加已有全池结构筛查，不是逐像素重算整个数据集。

现有不可变 C2/C3/C4 验证计划在 32k、40k、50k 继续与 25k 基线比较。另有零训练/验证场景交集的 49 个固定样本，覆盖 C1T8/16/24/32、C2T16/24、C4T12、C5/6/7T4、C8T6、C9/10T4，在 50k 评估新布局的泛化。相机、帧号、源参考和计划哈希固定。

续训控制器等待原 25k 训练及验证成功后启动三个阶段，不终止任何进程。失败退出、活动训练变更、代码或配置变化都会写入状态并停止接续。新 checkpoint 写入 `/data8/zbf/multicamera4rc_runs/`；旧训练在 `/data9/zbf/my_dfrt_multicamera4rc/outputs/` 保留。

```bash
PYTHONPATH=. .venv/bin/python scripts/build_variable_curriculum.py \
  --data-config outputs/training_preflight/native_motion_stereo_20261002/full_mixture_native_motion_stereo.yaml \
  --output outputs/training_preflight/variable_ct_20261003
PYTHONPATH=. .venv/bin/python scripts/build_variable_validation.py \
  --curriculum outputs/training_preflight/variable_ct_20261003/curriculum.json \
  --output outputs/training_preflight/variable_ct_20261003/variable_validation
```

实际状态见服务器 `outputs/training_preflight/active_training.json`、`queued_continuation.json`、续训 control_dir 的 `progress.json` 和 `supervisor.log`。准备好并排队的 50k 续训与当前仍在执行的 25k 阶段必须区分。
