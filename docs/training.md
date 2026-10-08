# 训练、推理、验证与恢复

[返回 README](../README.md) · [GT 与数据比例](data.md) · [模型结构](architecture.md)

## 入口和配置边界

所有命令从仓库根目录运行，安装后也可用统一 `multicamera4rc` 命令。`python -m multicamera4rc train --help` 等价于 `python -m multicamera4rc.train --help`。

| 入口 | 作用 |
|---|---|
| `train` | 单进程训练或由 torchrun 启动的 DDP worker |
| `launch` | 单机多 GPU 预检、物理 GPU→UUID 映射、启动 torchrun |
| `infer` | manifest/native adapter 推理，每 clip 导出 `.pt` |
| `evaluate` | renderer-derived Kubric dense GT 的固定验证；不是所有数据集的通用 evaluator |
| `demo` / `adapter-demo` | 完整 CPU smoke，含真实训练、断点和推理 I/O |
| `scripts/evaluate_static_multiview.py` | 固定静态多视图几何验证 |
| `scripts/evaluate_native_dynamic.py` | 已准备的原生动态 GT 固定计划，记录全部有效粒子的未加权三维误差 |
| `scripts/evaluate_monocular.py` | 参考旧 D4RT 的 Sintel Final 14 / ScanNet test100 单目几何评估；先审计固定样本，再复用旧误差函数 |

数据 YAML 管理路径、准入池、C/T、GT 类型和源比例；模型架构保存在 checkpoint 的 `config`；运行参数包括 BS、源参考数、LR、混合精度和总目标 step。不要将网络配置键直接放进数据 YAML。

`--manifest` 和 `--data-config` 二选一。`--pretrained` 仅初始化模型权重，`--resume` 恢复模型、AdamW、scaler、global step、数据游标及每 rank RNG；二者不可同时使用。

## 从 4RC 初始化

```bash
python -m multicamera4rc launch \
  --gpus 4,5 --allow-shared \
  --pretrained /path/to/4RC/model.safetensors \
  --data-config multicamera4rc/configs/kubric_variable_native.yaml \
  --decoder-memory all_observations \
  --batch-size 1 --source-count 4 --max-source-time-pairs 64 \
  --workers 0 --no-query-chunking \
  --steps 1000 --save-every 100 --output outputs/variable_trial
```

`launch` 需要明确初始化权重/断点，默认 giant、BF16。单 GPU 也可直接运行：

```bash
CUDA_VISIBLE_DEVICES=4 python -m multicamera4rc train \
  --data-config multicamera4rc/configs/kubric_multicam_native.yaml \
  --pretrained /path/to/4RC/model.safetensors --preset giant \
  --decoder-memory all_observations --device cuda --amp --amp-dtype bfloat16 \
  --batch-size 1 --source-count 2 --workers 0 --no-query-chunking \
  --steps 1000 --save-every 100 --output outputs/single_gpu_trial
```

4RC 的 `.safetensors` 需要 `weights` extra。原有模块默认必须全部匹配，允许缺少本项目新增模块；形状不符总是报错。native checkpoint 默认完整严格加载。旧 D4RT/VideoMAE 权重不适用。新模块仍需训练，加载原 4RC 不等于得到训练好的多相机模型。

## GPU 选择与显存

`launch --gpus 4,5` 指 `nvidia-smi` 物理下标，实际用对应 UUID 设置 `CUDA_VISIBLE_DEVICES`，rank0/1 分别映射到所选 GPU。使用同一命令加 `--dry-run` 可以查看完整命令和资源快照，不启动子进程、不创建输出目录。

有其他计算任务时，默认拒绝启动；已有授权的共享场景加 `--allow-shared`。`--wait-free SECONDS` 是显式前台等待，并非后台调度。launcher 不终止其他进程。

每卡 BS1、双卡全局 BS2。不同 rank 的 C/T/H/W 可以不同；BS>1 需要同形状分桶。AMP 优先 BF16；FP16 有 GradScaler，日志区分迭代 step 与真正 optimizer update，不能忽略 overflow 跳步。

双卡扩到四卡可使用 `--gpus 4,5,2,3 --batch-size 1`，原 rank0/1 的物理卡顺序保留，全局 BS 变为 4。标准 DDP 每个 rank 各持完整模型并消费一个片段；扩卡增加每次更新的片段数，不会将同一片段的相机或像素分块。变尺寸片段的同步等待和通信会影响吞吐，不能将四卡等同于训练步速翻倍。

扩卡恢复按 `旧 batches_consumed × 旧卡数 × 旧每卡 BS` 换算新数据游标，保留同一 epoch 的已消费排列前缀，而不是用累计 optimizer step 重新定位数据。游标必须落在新全局 batch 边界，否则明确报错；数据配置变化仍需显式重置。保留已有 rank 的 RNG，新 rank 使用 `training_seed + rank` 初始化，新的 checkpoint 保存全部 rank RNG；四卡续训可再次精确恢复。模型、AdamW、损失和 LR 计划照常恢复，不随卡数自动放大学习率。

生产控制器 `scripts/resize_training.py --controller /path/to/job.json` 等待指定旧任务的下一份**通过固定验证**的 checkpoint，再核对进程身份、GPU 占用、代码与证据后切换。先执行短程真实 DDP 预检，再从被接受的父 checkpoint 继续；预检失败会回到原有设备。原有数据比例、验证计划和退化回滚门槛保留。`progress.json` 的 `waiting_accepted_checkpoint` 表示仍在等待，并不表示四卡已经启用。

若旧任务正在质量回退，扩卡不必等待新的模型通过验证：控制器配置 `resume_existing_baseline: true` 可以直接使用该恢复任务记录的已评估父 checkpoint。此路径重新检查父模型满足原质量门槛，仅允许重放已审计的数据配比；因为从旧混合数据的父 checkpoint 重新接入回放数据，明确重置采样游标，保留其模型、AdamW 和 LR 计划。被拒绝的候选模型不会借扩卡被采用。

`--source-count 4` 是每片段 4 张源参考图，不是 4 个像素。全部输入仍做 encoder 和几何预测，选定参考图的全部有效像素/粒子接受对应 GT。

`--max-source-time-pairs 64` 限制本次源数量：`Q ≤ max(1,floor(64/Tq))`，再受 source-count 和实际有效源数限制。T≤16 通常 Q4，T24/T32 为 Q2。该限制控制 motion head 的高分辨率输出规模，不减少 KV 观察数。

`--no-query-chunking` 关闭 attention query 与 visibility 像素 query 分块。几何 DPT 的 `geometry_chunk_size` 独立，默认每 8 张观察计算；可在模型配置设 `None`，但要重新验证峰值显存。激活检查点默认开启，以重算降低训练显存；`--no-checkpointing` 包括恢复训练时都生效。

全时刻 KV 模式的显存/时间要重新测量，不能直接沿用 `target_time` 旧任务的预算。旧预检曾对 PointOdyssey T32 出现 OOM，因此既有课程使用 Replica≤T16、Point/vKITTI 单目≤T24；详见[历史测量](../multicamera4rc/audits/variable_ct_20261003/README.md)。

## 恢复训练与显式架构切换

```bash
# 同一模型/数据配置继续到总共 50,000 step；保存的 LR 计划会继续生效。
python -m multicamera4rc launch \
  --gpus 4,5 --allow-shared \
  --resume outputs/PREVIOUS/latest.pt --output outputs/CONTINUATION \
  --data-config /path/to/the_same_data.yaml \
  --steps 50000 --batch-size 1 --source-count 4 --max-source-time-pairs 64 \
  --workers 0 --no-query-chunking --loss-profile auto
```

`--steps 50000` 是总目标，不是在现有 step 上再加 50k。单独增大 steps 不会自动改写断点中的 LR 衰减终点。`--loss-profile auto` 恢复保存的完整损失配置。

旧 native config 未记录 `decoder_memory` 时，统一恢复为 `target_time`。需要全时刻 KV 时，给上述命令显式加 `--decoder-memory all_observations` 并使用新输出目录记录迁移。虽然参数及 AdamW 形状兼容，计算行为已变，应该先完成少量真实前反向及相同固定 plan 的验证，再进行长期训练。

旧 native config 未记录 `time_origin` 时保留 `absolute`。新模型默认 `clip_start`；迁移旧模型需要显式 `--time-origin clip_start`。该模式只对网络时间编码减去片段原点，GT 和返回的帧标签不变，抽帧间隔仍保留。不要在同一次对照中同时更换 KV、时间原点、loss 和数据比例。

更换采样索引/准入池时使用 `--reset-data-cursor`，保留 optimizer、step、RNG 并从新数据索引起点开始；同一数据配置续训不应反复重置游标。当前三阶段 balanced_coverage 课程配额/索引一致，所以只在 25k 首次接入时重置，32k/40k 不重置。

显式开始新的 LR 阶段：

```bash
python -m multicamera4rc launch \
  --gpus 4,5 --allow-shared \
  --resume /path/to/step_00025000.pt --output outputs/new_stage \
  --data-config /path/to/stage_1_25000_32000.yaml \
  --reset-data-cursor --override-lr \
  --lr 1e-5 --backbone-lr 1e-6 --lr-schedule cosine \
  --warmup-steps 500 --warmup-start-ratio 0.3 --min-lr-ratio 0.1 \
  --lr-end-step 50000 --steps 32000 --save-every 2500 \
  --batch-size 1 --source-count 4 --max-source-time-pairs 64 \
  --workers 0 --no-query-chunking --loss-profile auto
```

仅传 `--lr` 不会覆盖已恢复 AdamW 的学习率，要用 `--override-lr`。`--lr-end-step` 将学习率终点与当前阶段停止点分开；下一阶段保持既有计划，不重复 warmup。checkpoint 保存时序列化完整模型 config、loss config、LR plan、rank RNG 和数据游标。

## 当前 dense 损失

本项目组合 4RC 式空间/时间约束和参考的 L2/前景加权，并不声称逐行复现原 4RC 未公开训练实现。实际定义在 [losses.py](../multicamera4rc/losses.py)。

| 项目 | dense 权重 | 监督范围 |
|---|---:|---|
| depth | 1 | 所有输入观察的有效 Z-depth |
| ray | 1 | 有效世界方向与原点，辅助射线输出 |
| camera | 1 | 归一化 K 和 c2w 矩阵误差 |
| trajectory | 1 | 源图有效像素/原生粒子的目标世界位置 |
| displacement | 1 | 可用时间对的世界位移 |
| depth_gradient | 1 | 有效邻接像素的深度差 |
| velocity | 1 | 已知运动时间对，差分除以真实帧号间隔 |
| visibility | 0.1 | GT 有效位置的 BCE logits |
| static | 0.1 | 仅有可靠 static_mask 的零位移约束 |

向量回归使用 L2 范数，动态前景权重 10，按有效权重和归一化；非前景不是被丢弃。带置信度时典型项为 `c*error - 0.2*log(c)`，loss 内将 c 限制到 `[1,10000]`。源时刻身份槽不使用 motion confidence 的加权/负 log 奖励。camera 项采用独立矩阵 L1 归约，不是把所有项目都统一成 L2。

PointOdyssey 的 `sparse_trajectory/displacement/velocity/visibility` 对全部可用粒子应用对应损失。unknown/invalid mask 排除不存在的 GT；dense 模式 visibility 使用 GT 有效性，不允许网络以预测不可见逃避轨迹误差。

总 loss 因负 log confidence 项可能为负，不能仅凭正负或跨配置总 loss 判断收敛。应看未加置信度的 depth AbsRel、基础点/轨迹/位移/速度 EPE、动态前景 EPE、相机误差、visibility precision/recall/F1，以及固定验证变化。

`metrics_version=2` 修复了 AMP 内相机角度计算：FP32 相对旋转加 atan2，避免 BF16 将小角度量化成 0。旧训练日志、旧静态验证的角度不能直接与新数值横比，需要重算相同 checkpoint。原 Kubric evaluator 已在 AMP 外计算，重算只会有小角度公式差异。新增 `*_epe_temporal_normalized` 排除源时刻恒等查询；`native_*_epe_*` 使用已有双线性采样结果计算 PointOdyssey 的三维误差，不重复采样，也不改变损失梯度。`sparse_trajectory` 等旧键仍是加权 loss，不能当作 EPE。

`dense_balanced` 是保留的实验配置，不是现有训练默认；切换会改变归约与权重，需要单独消融。`--loss-config` 可传完整 JSON/YAML，保存到 checkpoint 后可按 `auto` 恢复。

## 推理和导出

```bash
python -m multicamera4rc infer \
  --manifest /path/to/clips.json --checkpoint /path/to/latest.pt \
  --source-obs 0 3 --query-times 0 5 10 \
  --device cuda --amp --amp-dtype bfloat16 --no-query-chunking \
  --output outputs/inference
```

原生数据用 `--data-config` 替换 manifest，可用 `--max-clips` 限制数量。`--no-visibility` 跳过可见性头。默认保留 checkpoint 的 memory scope；只有主动消融才覆盖。

输出包括 `inference.json`（实际 model config/精度/设备/权重路径）和 `clip_000000.pt` 等预测张量，含原 clip ID、相机/时间映射。所有 tensor，包括 ray 多尺度列表，导出到 CPU。默认不是 PLY/视频导出器；下游可从 base_points/trajectory 及源 RGB 组织点云。

## 固定验证

```bash
python -m multicamera4rc evaluate \
  --data-config multicamera4rc/configs/kubric_multicam_native.yaml \
  --checkpoints before=/path/to/before.pt after=/path/to/after.pt \
  --plan outputs/fixed_validation_plan.json \
  --output outputs/fixed_validation --device cuda --source-count 4
```

首次建立 plan，之后复用同一文件；固定序列、时间、相机采样种子和源参考，保存计划 SHA-256。强制检查 train/val 序列不重叠，按片段等权汇总。该入口需 Kubric renderer 稠密 GT；不要把混合训练 YAML 直接传入并假设所有 adapter 都支持它。验证也可经 `torch.distributed.run --module multicamera4rc.evaluate` 分片。

Kubric 评估现在同时记录 checkpoint 的完整 loss config、总 loss/分项 loss、无置信度误差及 `by_sequence`；同一场景的多个 C/T 组合不是独立场景。比较总 loss 前先检查两份 loss config 是否一致。原生动态评估要求 `sampling: coverage` 和显式固定 index 计划；数据配置需预先选择官方 val/valid 或按物理场景隔离的 held-out 池。脚本检查 index 对应的实际序列，不能把训练集 index 的结果标为验证。参考本次[质量审查](../multicamera4rc/audits/quality_20261003/README.md)中的数据和证据范围。

如果比较 memory scope，必须显式标明使用原 checkpoint 语义还是 `--decoder-memory all_observations` 的覆盖；`evaluation.json` 保存每个模型实际 config。固定验证上的变化才适合用于判断 Kubric C4 是否退化，随机训练 batch 的 loss 不可直接横比。

2026-10-08 起，正式 BlendedMVS 面板使用全部 26 个物理留出场景、52 个固定视角组合；连同原有 Hypersim/Taskonomy 组合，静态面板共 60 个片段。原两场景面板的局部退化曾触发不必要的回退，范围扩充后必须同步重算当前、最佳及历史参考 checkpoint，不能把新面板误差与旧均值比较。静态报告保留 `by_scene` 及原始 `records`，用于跟踪平均值背后的局部变化。详细基线、GT 检查和部署证据见[迁移记录](../multicamera4rc/audits/blended_validation_migration_20261008/README.md)。

`scripts/migrate_static_validation.py --controller /path/to/job.json --verify-only` 在交接前检查已重算的参考报告、完整视角覆盖、固定计划指纹、训练进程身份和验证证明。去掉 `--verify-only` 才执行交接：仅替换明确记录的旧控制器，保留其启动器和全部训练 rank，等待现有训练段写完完整 checkpoint 后再按新面板评估。新控制器将 `static_contract` 写入质量记录并拒绝跨范围比较；若真实质量退化仍需回退，已经完成的在途段计入原重试预算，不额外增加尝试次数。交接本身保留学习率、训练数据及质量阈值。

当前另有显式 `learning_rate_trial` 配置：`start_step=55500`、`end_step=56000`、`multiplier=1.5`、`warmup_steps=100`。它只在边界第一次验证通过，且相对上一份/最佳参考的 24 项误差比几何均值均 ≤1.02、最大单项比均 ≤1.10 时触发。否则沿用原计划。两组 LR 以 checkpoint 中实际 LR 为起点平滑提高，随后 cosine 衰减至总目标 100k；保存 AdamW、每 rank RNG 和同一数据游标。

升 LR 的 56k 候选必须通过原质量门槛，并且相对 55.5k 的综合误差比 ≤1、最差单项比 ≤1.10。未达标时，第一个重试完整恢复 55.5k 及其原始 LR 计划；原 LR 重试也失败才执行一次原 LR ×0.5 的恢复，仍限两个重试。后续阶段不重复触发升 LR。控制器记录 `pending_boundary_validation`、`running`、`skipped` 或 `completed`，证据及实际进度见[学习率试验记录](../multicamera4rc/audits/lr_trial_20261008/README.md)。

## B300zbf 的既有 50k 任务

现有训练工作区是 `/data9/zbf/my_dfrt_multicamera4rc`，原 `/data9/zbf/my_dfrt` 保留。50k 续训输出位于 `/data8/zbf/multicamera4rc_runs/`，避免原磁盘空间不足。既有控制器等待旧 25k 训练及固定/静态验证通过，再执行 25k→32k→40k→50k，各阶段使用已核验的配置和 checkpoint。

框架审查时，原任务固定在 `7a2038a`，使用旧 `target_time` 模式。2026-10-03，该任务在 29,022 步后因 DynamicReplica 全无效光流占位文件触发加载异常，最近完整断点为 27,500 步。修复后恢复训练时显式保留 `target_time`、AdamW 状态、原学习率计划和采样游标，不能静默切换至新模型默认的 `all_observations`。

`scripts/recover_variable_continuation.py` 从已停止且标记失败的训练阶段恢复，保留后续课程与固定验证，默认每 1,000 步保存一次。它检查活跃任务身份、干净的 Git 提交、原配置/证据哈希、恢复证明和完整 checkpoint，并移除原阶段首次启动用的 LR/cursor 重置参数。恢复证明 JSON 必须含 `passed: true` 和对应 `commit`；生成前应完成失败样本回放及回归测试。脚本不会终止任何进程，也不会无限自动重试坏数据。

```bash
python scripts/recover_variable_continuation.py \
  --repo /data9/zbf/my_dfrt_multicamera4rc \
  --failed-job /path/to/failed_control/job.json \
  --checkpoint /path/to/failed_stage/step_00027500.pt \
  --commit VERIFIED_COMMIT --proof /path/to/recovery_proof.json
```

控制器会检查 Git HEAD 和证据文件哈希；先在独立工作区验证修复，确认旧任务已经退出后再部署到训练工作区。文件更新不能冒充已重新验证的续训，Git 同步也不能代替从 checkpoint 恢复进程。

实时状态请读取服务器以下文件，不将 README 当作实时训练日志：

```text
outputs/training_preflight/active_training.json
outputs/training_preflight/queued_continuation.json
<control_dir>/progress.json
<control_dir>/supervisor.log
<run>/launch.json
<run>/torchrun/（逐 rank stdout/stderr）
```

50k 课程生成器为 `scripts/build_variable_curriculum.py`，验证布局生成器为 `scripts/build_variable_validation.py`；配置、GT、giant 和 DDP 历史证据见[课程审计](../multicamera4rc/audits/variable_ct_20261003/README.md)。历史证据均有版本和场景边界，不构成全数据逐像素正确性或新模型精度的承诺。

## 32k 起的质量恢复

`scripts/build_quality_curriculum.py` 基于已准入的可变 C/T 课程生成独立采样配置：Kubric 增加 C2T12/C3T8/C4T6 回放；DynamicReplica 左右流分别占全局 9%/3%；其他家族比例不变。`scripts/audit_quality_curriculum.py` 逐阶段验证实际序列池、有效帧池及真实左右流监督。全部已准入数据仍可采样，有限训练步数并不保证逐帧遍历全部数据。

训练参数 `--prefer-motion-sources` 优先选择具有有效原生运动 GT 的源图，保留全部有效像素/原生粒子。该设置写入 checkpoint；未显式传入时恢复保存值，`--no-prefer-motion-sources` 可显式关闭。

`scripts/continue_quality_recovery.py` 在 32k 原子断点处接续旧控制器，并执行采样恢复、相对时间编码试验、前景归一化试验和后续 50k 训练。每个试验从相同断点分出控制/候选两支，各训练 250 步，分别恢复 Adam、LR、rank RNG 和数据游标；只改变待检验参数。候选训练或验证失败时保留已通过验证的控制支，并记录错误。控制支本身失败则停止并保留日志，不无限重试数据错误。

选择使用固定 59 片段的原始误差面板，单独约束 C4T6 和原生动态三维追踪；总 loss 不参与选模。`best_checkpoint.json` 保存推荐 checkpoint 的路径和完整分项误差；`latest_evaluated_checkpoint.json` 保存最近选中并评估的模型。相对初始模型出现明显回归时暂缓增加时间长度，仍按当前长度继续训练。具体阈值、实测和部署记录见[质量恢复审计](../multicamera4rc/audits/quality_recovery_20261003/README.md)。这些验证集合参与调参，最终效果需另用独立测试集报告。

## GT 修复后的 100k 续训

2026-10-06 检查发现，训练在 45,200 步结束了第一段 Kubric GT 修复适应，尚未达到 50k。过滤穹顶并改变场景尺度后，Kubric 几何明显改善，但固定验证中 PointOdyssey、vKITTI2 的位移误差相对修复前分别增长约 87%、54%。原控制器遇到任一未改 GT 数据源超过 25% 的退化便直接退出，没有后续恢复阶段；这是该次 GPU4/5 空闲的原因。

恢复从保留的 45,200 步完整检查点开始，使用修复后的 GT，恢复 Adam、各 rank RNG 和采样游标。显式将主干/其他参数峰值学习率设为 `1e-7/1e-6`，用 50 步从 50% 预热，余弦衰减终点设为 **100,000**，最低比例 20%。先运行 100 步并执行相同固定验证。该 LR 计划随 checkpoint 保存，后续阶段不重复重置。

[`scripts/continue_extended_training.py`](../scripts/continue_extended_training.py) 等待上述短恢复及验证结束，再自动接续至 100k。它确认前一控制器退出、干净代码提交、证据哈希、优化器/RNG 及修正的 GT 协议。第一段进一步训练 500 步，此后按 2,000 步边界固定验证；训练每 1,000 步及每段结束保存 checkpoint。

若历史动态/静态误差仍高于原基线 25%，启用包含全部原有源的运动回放配置：Kubric 从 35% 调为 28%，PointOdyssey 从 12% 调为 16%，vKITTI2 全族从 11% 调为 14%；其他源比例、相机/时间布局不变。首次切换配额显式重置数据游标，保留模型、Adam 和 RNG。此比例是针对当前回归的恢复设置，需以接下来的固定验证判断效果。Syn4D 的独立接入配置不自动加入此轮，正式 giant 原分辨率布局仍需专门预检。

后续质量检查同时保留三组比较：上一通过检查点、保留的最佳检查点、修复前基线。历史回归持续记录；新增退化若相对前两者任一项超过 25%，或几何平均误差超过 10%，则保留失败候选，从该段起点的完整通过检查点回退，降低 LR 重试，最多两次。不会删除错误记录、恢复错误 GT 或用更大的总 loss 阈值掩盖退化。数据/运行错误、非有限验证值或重试耗尽仍明确失败，避免无限重复无效训练。

实时入口仍为 `outputs/training_preflight/active_training.json`。排队中的长程控制器另记录于 `outputs/training_preflight/queued_extended_training.json`；实际运行目录和目标以这两个文件及对应 `progress.json` 为准。审计见 [100k 续训记录](../multicamera4rc/audits/extended_100k_20261006/README.md)。
