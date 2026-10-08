# TartanAir 深度有效域修复：2026-10-02

## 问题和实际策略

混合训练中的三个 loss > 20 尖峰均包含 TartanAir 右相机窗口。原生深度是 camera-Z、单位米，但天空球可用约 10,000 米的有限深度表示无穷远，见 [TartanAir 官方格式说明](https://github.com/castacks/tartanair_tools/blob/master/data_type.md)。它们通过了旧的“有限且 > 0”判断，参与场景尺度的平均点距离，使正常前景被压到很小的归一化深度。

仅过滤 p98 无法排除占图像 20%–27% 的天空。现在左右流都使用以下逐图策略：

1. 有效候选为 `isfinite(depth) & (depth > 0) & (depth <= 200)`。
2. 在这些候选像素上计算 p98；只保留 `depth <= p98` 的候选。阈值相同的平面保留。
3. 把最终 mask 传到几何监督、基点/ray 有效域和场景尺度计算。空窗口继续由现有加载器记录并重新采样。

200 米是本项目已有 `adapters/TartanAir.py` 的范围约定，不是官方对全部场景的有效 GT 上限。它会舍弃超过该距离的真实远景；p98 还会舍弃有效范围内最远约 2% 的像素。过滤仅用于本次 TartanAir 监督，不是推理深度上限，也没有移除训练序列。RGB、原生深度文件、K/E、原生宽高比、原始位姿不改写；未知运动仍 masked，原始读取模式仍可用。

## 验证证据

- `native_filter_audit.json`：左右相机、环境及难度的 68 个真实窗口，以及三个历史尖峰的确切帧窗口。新策略均有有效像素；68 个窗口的保留比例中位数 95.69%，最小 40.40%。
- `giant_filter_replay.json`：同一个 step 10000 checkpoint 在上述三个窗口上的原生分辨率前向及过滤后反向。新标签的 loss 为 1.492 / 1.400 / 1.450；全部梯度有限，峰值显存约 27.57 GiB。这里没有恢复历史尖峰时刻的模型/RNG，不能将日志中的 4228 与这次 1.450 当作同权重实验。
- `giant_filter_replay12500.json`：实际切换起点 step 12500 的同窗口检查，过滤后 loss 1.655 / 1.660 / 3.162，梯度全部有限，最大未裁剪梯度范数 130.53。
- `checks.json`：完整测试 178 passed，包含十项范围/p98 回归检查；原生 ZIP→bridge→dense target 检查过滤 mask 是否真正排除尺度和 ray/base 标签，并检查无伪造运动 GT。
- `midphase_validation.json`：同一不可变 24 片段 Kubric 验证计划对比 step 5000 / 7500 / 10000。训练和验证序列没有重叠。数据规模较小，仅反映这个验证域。

三个历史窗口的场景尺度（平均有效点距离，米）：

| 窗口 | 未过滤 | 只用 p98 | 200m 后 p98 |
|---|---:|---:|---:|
| neighborhood/Hard/P005/right，104–111 | 4365.12 | 4154.54 | 23.29 |
| endofworld/Easy/P009/right，74–81 | 3116.22 | 2870.00 | 4.97 |
| ocean/Easy/P005/right，349–356 | 179.21 | 22.83 | 21.03 |

## 其他发现及训练衔接

旧日志中达到 `loss > 10`、`depth_abs_rel > 10` 或 `trajectory_epe_normalized > 2` 的 134 个批次均包含 TartanAir。旧指标是两张卡的平均值，不能严格证明其他数据源没有贡献。现在逐 rank 记录 loss、深度/轨迹误差、置信度饱和比例、场景尺度和过滤像素数，以便直接定位数据源。

Hypersim `ai_032_004/cam_01` 的一个单帧窗口没有有效深度，旧加载器已记录并跳过。没有发现 NaN、OOM 或跳过 optimizer update。

固定验证中 step 10000 相对 step 7500 已恢复，但仍弱于混合前的 step 5000：

| 指标（越低越好） | 5000 | 7500 | 10000 | 12500（修复前） |
|---|---:|---:|---:|---:|
| depth AbsRel | 0.007084 | 0.011734 | 0.007961 | 0.008688 |
| 动态位移 EPE（归一化） | 0.015094 | 0.021632 | 0.016803 | 0.022837 |
| 轨迹 EPE（归一化） | 0.009012 | 0.021445 | 0.015120 | 0.015327 |
| 相机旋转（度） | 0.222539 | 1.140748 | 0.815356 | 0.935867 |

`pre_filter12500_validation.json` 是最新切换起点的同计划验证。相对 step 5000，12500 的深度误差增加约 22.6%，动态位移误差增加约 51.3%，轨迹误差增加约 70.1%。这是真实的多相机验证域退化，尚不能单独归因于 TartanAir；几何单目数据与动态多相机任务的混合比例及共享参数遗忘也需要考虑。当前保留原比例以观察 GT 修复的独立影响，不能承诺过滤后泛化已经改善。

因此混合训练的随机总 loss 不能代替固定验证。本次先隔离 GT 修复，保留原始采样权重与损失权重，增加 step 15000 的同集检查；检查范围没有扩大为未验证数据集的质量承诺。

训练在原任务保存完整 step 12500 checkpoint 后切换。新任务仍用 GPU 4/5、BF16、giant、全局 BS=2、Q≤4、dense/native/no query chunking，恢复模型、AdamW moments、rank RNG、数据 epoch/消费游标及原 5000→25000 cosine LR 计划。两侧 TartanAir 过滤不改变序列池、epoch 长度或索引映射；这次不重置采样进度或 warmup。旧 checkpoint 和日志保留，实际续训信息另见 `training_job.json` 与 `resumed_health.json`（部署后写入）。

已于北京时间 11:15 切换为 `full_dense_mixture_tartan_filtered_gpu45_20261002_111519`。新的 supervisor 在 step 15000 checkpoint 原子保存完成后，自动使用 GPU5 的 12% 显存额度执行同一个 24 片段验证计划，比较 5000 / 修复前 12500 / 修复后 15000；最终 25000 后再执行同计划对比。验证保持只读，不中断训练、不重设采样进度。

`resumed_health.json` 记录切换后的 133 次真实 optimizer update（至 step 12633）：逐卡及平均指标全部有限、没有 skipped update、loss 中位数 -1.551。实际左右 TartanAir 均已抽到；逐帧核验过滤参数为 200m/p98，p98 cutoff 在 (0,200]，保留像素数不超过有效范围候选数。新日志能直接查看它们各自的误差，不能据这 133 步宣称验证泛化已恢复。消除冷启动后的短窗口速度约 3.38 秒/step；ETA 应结合共享 GPU 的长窗口速度，不能把模型加载/worker 启动算作每一步耗时。

## 重放

```bash
.venv/bin/python -m scripts.audit_tartanair_depth_filter \
  --data-config outputs/training_preflight/tartan_filter_20261002/full_mixture_filtered.yaml \
  --output outputs/training_preflight/tartan_filter_20261002/native_filter_audit.json
```

适配器 `depth_max` 和 `depth_percentile` 可选；新的完整 mixture builder 为左右 TartanAir 显式配置 200 米与 98。`full_mixture_filtered.yaml` 是训练机实际配置快照。
