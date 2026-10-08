# 32k→50k 质量恢复训练

这次调整针对 [质量审查](../quality_20261003/README.md) 中的多相机遗忘、有效运动监督占比不足和验证结果未参与选模。旧训练先完成 32,000 步的原子断点，再由 `scripts/continue_quality_recovery.py` 接续。模型保持 giant、GPU4/5、单卡 BS1/全局 BS2、原始比例和 native 分辨率、全像素/全部有效原生粒子监督、无 query chunking。恢复 AdamW moments、两卡 RNG 和原有 25k→50k cosine 学习率计划；改变数据配额时显式重置数据游标。

## 已选择的采样调整

- 数据家族总比例保持：Kubric 35%、DynamicReplica 12%、PointOdyssey 12%、vKITTI2 11%、TartanAir v1/v2 4%/7%、Hypersim 4%、ScanNet/ScanNet++/WildRGBD 各 3%、BlendedMVS 4%、Taskonomy 2%。
- Kubric 内部 50% 沿用可变 C/T 分布，另外 50% 等分到 C2T12、C3T8、C4T6。首阶段 C4T6 占全部训练约 7.72%，此前约 3.78%；所有原来的 C1–C10 组合继续有非零权重。
- DynamicReplica 左流占全局 9%、右流 3%，各保留 483 条已准入序列及原有有效帧。左右都保留 `replica_flow`，可用右流运动 GT 继续参与监督。
- `--prefer-motion-sources` 在有原生相邻帧 GT 时优先选择有有效 pair 的源图；PointOdyssey 仍优先选择有有效原生轨迹的源图。每张选中源图使用所有有效像素或原生粒子，没有添加伪 GT。
- 原第一阶段的时间长度分布延续到 34k。34k 后每次验证通过质量条件，最多提升一级时间长度课程；未通过则保留当前长度继续训练至 50k。

`build_quality_curriculum.py` 生成独立配置和左右流 manifest；`audit_quality_curriculum.py` 比较每阶段迁移前后的实际适配器序列池、有效帧池和布局，并对真实左右流样本检查源图选择、有限轨迹及位移 GT。

## 时间编码实测及后续试验

31k 的同断点、同计划对照共运行三种编码 × 42 个片段（Kubric 24、原生动态 6、静态多视图 12）：

| 直接修改 | 有利变化 | 回归 | 决定 |
|---|---|---|---|
| `absolute`→`clip_start`，保留投影 | PointOdyssey 非 identity 位移 EPE −14.19% | DynamicReplica temporal 轨迹 EPE +12.86%；Kubric depth AbsRel +5.70% | 不直接替换现有训练 |
| `clip_start`，清零 frame/target 两层投影 | BlendedMVS depth AbsRel −49.97% | Kubric base EPE +32.60%；Hypersim depth AbsRel +14.97% | 不采用重置 |

这些是未经适应训练的切换结果，不能推断相对时间经过训练后的上限。相对时间消除了公共帧号平移依赖，仍值得单独进行短程训练对照。两层投影清零选项保留为显式实验入口，需要 `--resume --time-origin clip_start --reset-time-projections`，只清除对应两层权重和 Adam 状态，默认不启用。

后续流程由控制器执行并写入结果，不能把排队试验说成已经改善精度：

1. 32k 固定验证，保存起始模型及误差面板。
2. 32k→32.5k 仅应用采样和源图选择调整。
3. 从同一个 32.5k 断点分别训练 250 步：保留时间编码 / 使用 `clip_start`。两支共享起始 Adam、LR、数据游标和每卡 RNG。
4. 选择上一轮结果，从同一个 32.75k 断点再分别训练 250 步：原损失 / 仅设置 `foreground_fraction=0.5`。其他损失项与置信度设置相同。
5. 从选中的 33k 断点继续到 50k，34k 起每 2k 固定验证。两个被拒绝分支的计算不计入所选模型的累计 step。

`foreground_fraction` 作用于具有可靠 `trajectory_foreground` mask 的 dense 监督；PointOdyssey 原生粒子的回归权重保持原样，全部有效粒子继续参与损失。原生粒子的运动/静态分组是另一个需要 GT 审核和独立消融的改动，本次没有把它混入 dense 前景试验。

保留 `decoder_memory=target_time`；审查中直接切换全部观察 KV 造成明显回归，本次不同时改变 decoder KV。

## 验证和选择规则

每次验证使用相同的 59 个片段：Kubric 24、额外 C/T 17、原生动态 6、静态多视图 12。Kubric 包含独立 C4T6 指标；PointOdyssey 使用原生非 identity 三维轨迹/位移 EPE；原生 flow 数据使用动态像素位移 EPE。静态集考察 depth AbsRel 和基础点 EPE。完整报告也保留 loss、相机与可见性指标。

选择面板共 24 项原始误差，用 candidate/reference 的几何平均比较，避免置信度让总 loss 降低却掩盖几何误差：

- 候选优于配对对照至少 0.5%，且每一项误差的回归不超过 10%。
- 相对初始 32k，几何平均回归不超过 2%、每项不超过 10%，才允许实验迁移或增加时间长度。
- `best_checkpoint.json` 保留通过选择条件的最佳路径及分项误差；`latest_evaluated_checkpoint.json` 单独记录最近选中并完成验证的模型。被拒绝实验不能覆盖 best。
- 验证控制课程提升和 checkpoint 选择，不在失败时静默替换 GT、丢弃数据源或重置优化器。

阈值是保守工程规则，不是统计显著性检验。原生动态验证每源只有两个片段，vKITTI2 两种天气来自同一物理场景；这些集合用于调参与监控，最终结果仍应在更大、独立测试集上报告。

## 执行与追踪

真实配置和控制器日志记录在 B300 的 `outputs/training_preflight/active_training.json` 指向的 `control_dir`；训练及评测结果保存在 `/data8/zbf/multicamera4rc_runs/quality_recovery_*`。部署证据和 31k 对照原始报告分别保存在 `/data9/zbf/my_dfrt_audit_20261003/outputs/quality_curriculum_20261003` 和 `/data8/zbf/multicamera4rc_runs/quality_improvement_20261003`。

控制器只在预检通过、证据哈希一致、旧控制器 PID/start-time/UID/命令匹配、32k 原子断点可恢复时切换。测试 checkout 与旧训练 checkout 分离；训练进程退出后才更新 live 源码。所有 checkpoint 原样保留，可按 `job.json` / 各试验 `*_decision.json` 复查。

2026-10-03 21:57（UTC+8）部署快照确认训练达到 32,015 步，前 15 步实际数据源与新采样计划一致，更新均未跳过。模型使用已验证的代码 `ed19140`；本地和 B300 各 264 项测试通过，真实 giant 双卡四次更新和三阶段数据池审核通过。59 片段的 32k 基线评估也全部完成，时间和前景损失的短程试验尚未完成。

证据文件：[`deployment.json`](deployment.json)、[`preflight_summary.json`](preflight_summary.json)、[`initial_time_comparison.json`](initial_time_comparison.json)、[`baseline_32000.json`](baseline_32000.json)。`best_checkpoint.json` 的选择范围是本次控制器评估并采用的分支；历史 25k/30k 结果另存于基线对照，不能把最近 checkpoint 或负 loss 直接解读为历史最佳。
