# BlendedMVS 正式验证面板迁移（2026-10-08）

**已将正式验证从两个 BlendedMVS 场景扩展到全部 26 个留出场景，并完成三份参考 checkpoint 的同范围重算。新控制器已接管，现有四卡训练进程未重启。** 这是[场景级诊断](../blended_scene_20261008/README.md)之后，经用户确认实施的正式迁移。

## 新验证范围

- BlendedMVS：26 个物理场景，52 个固定 T1 组合，请求 C4/C8；一个只有六个有效视角的场景实际使用 C4/C6。
- Hypersim、Taskonomy：保留各自原有四个固定组合。
- 完整静态面板共 60 个组合，所有原始验证片段保留。BlendedMVS 475 个训练场景与 26 个验证场景无重叠，没有因指标退化删除场景。

固定计划见 `fixed_plan.json`，规范化 SHA-256 为 `47f5a1b5444c9143ef718966efd029d831875878f936cf02a0b672b3b62ff79f`，使用 `metrics_version=2`。全部 52 个 BlendedMVS 片段完成 H5 像素/深度、裁剪/标定、最终归一化标签、padding 排除和无伪造运动监督检查，见 `gt_summary.json`。完整跨视角 GT 诊断保留在服务器 `blended_gt.json`，哈希写入 `baseline_proof.json`。

## 参考 checkpoint 已统一重算

每份参考模型实际重跑全部 60 个静态组合，共 180 次推理。原其他数据域的实测报告继续使用；新旧静态报告中 Hypersim、Taskonomy 的指标逐值一致。质量面板从报告重新构建，并核对缓存指标与原报告一致，只有 BlendedMVS 两项汇总因覆盖范围扩充而变化。

| 参考模型 | Blended depth AbsRel | Blended base EPE，归一化坐标 |
| --- | ---: | ---: |
| 45k 历史参考 | 0.06032320 | 0.06776196 |
| 53k 原最佳参考 | 0.06088415 | 0.07345289 |
| 55k 当前已接受参考 | 0.06124025 | 0.07273548 |

完整推理记录为 `evaluation.json`；控制器使用的记录为 `historical_record.json`、`anchor_record.json` 和 `current_record.json`。按原选模规则重新比较，55k 对 53k 的 24 项误差比几何均值为 **0.96525**，最差单项比为 **1.03210**，满足原最佳模型更新条件。因此当前最佳记录更新到已有的已接受 55k checkpoint，未采用之前被拒绝的 55.5k 权重。

对之前 `55500_replay1` 候选的诊断性回放：原面板判定 `rollback_lower_lr`，用相同候选的 26 场景实测值与校准后的参考面板比较，判定为 `continue`。这项回放说明验证覆盖修正能消除此前的那次不必要重试；不是对当前在途四卡候选的提前验收，也不能保证未来不会出现真实退化。

## 控制器交接与训练连续性

运行代码固定为 `94b126c3eac10ee4a321bf2c83ab22c08bf9ae9d`。修改范围限于控制/验证脚本、测试和审计记录；模型、优化器、训练入口和数据加载器源文件均未改变。旧控制器仅收到发给其自身 PID 的终止信号；没有向进程组、启动器或训练 rank 发信号。

`handover_proof.json` 记录旧控制器 PID 2149634 被新控制器 PID 3603577 替换，启动器 PID **2548590** 和四个 rank PID **2549220–2549223** 保持不变。交接时内存中的模型、Adam、随机状态和采样进度持续使用；没有回到 55k 重跑。

`health_after_handover.json` 的北京时间 **16:55:33** 快照显示，训练从交接前 **55,173** 步继续到 **55,178** 步，新增五次更新全部完成，日志 loss/梯度/LR 均有限，没有跳过优化器更新。GPU 顺序为 `[4,5,2,3]`，每卡 BS1、全局 BS4；当前 LR 为 `4.72089e-7`，backbone 为 `4.72089e-8`，沿用既有衰减计划。

当前段继续训练到 **55,500**，随后按新静态面板及既有 Kubric、原生动态、可变 C/T 和 Syn4D 验证流程评估。通过后自动续训到下一边界，目标仍为 **100k**。质量阈值与最多两次降 LR 重试的预算保持不变；被接管的当前训练段计为第零次尝试，若确有退化，只执行原预算中剩余的重试。

新质量记录包含 `static_contract`。控制器在比较当前、最佳、历史参考和新候选前验证计划指纹与指标版本，检查报告中每个实际视角组合与固定计划一致；不允许因范围变化混用旧均值。静态评估器还保存逐场景指标，便于持续观察困难场景。

## 验证与路径

本地及 B300zbf 各通过 45 项相关回归检查。覆盖场景扩展保留原片段、验证版本/视角/参考权重错配拒绝、缓存指标一致性、仅终止控制器、区分 rank 与 DataLoader worker、在途训练成功/退化分支以及剩余重试预算。服务器结果见 `unit_tests.json`。真实交接前的只读预检查也通过，随后验证了原 rank 继续更新。

- 当前控制器：`/data9/zbf/my_dfrt_multicamera4rc/outputs/training_preflight/jobs/blended_validation_migration_20261008`
- 当前在途训练段：`/data8/zbf/multicamera4rc_runs/scaleout_immediate_20261008/continuation_55000_55500_try0`
- 新控制器后续输出：`/data8/zbf/multicamera4rc_runs/blended_validation_migration_20261008/training`
- 固定计划和基线：`/data8/zbf/multicamera4rc_runs/blended_validation_migration_20261008/static`

实时状态仍以 `outputs/training_preflight/active_training.json`、新控制器 `progress.json` 和其记录的训练日志为准。本记录是交接完成时的快照。
