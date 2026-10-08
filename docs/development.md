# 代码组织与开发

[返回 README](../README.md)

## 模块边界

当前可安装包只包含 `multicamera4rc*`。模型内部不导入顶层旧 `models/`、`datasets/`、`utils/`，避免同名包混淆。历史入口保留，迁移文档集中到 [legacy-d4rt.md](legacy-d4rt.md)。

```text
README.md                       当前项目导航、安装、最小用法
docs/
  architecture.md               实际模型结构与维度
  data.md                       数据/GT/准入/采样
  training.md                   运行与恢复
  development.md                本文
  legacy-d4rt.md                历史 D4RT 用法
multicamera4rc/
  __main__.py                   统一 CLI 分发，原 python -m 入口继续支持
  __init__.py                   公开模型 API
  config.py / checkpoint.py     架构校验、配置恢复、权重兼容、原子保存
  metadata.py / embeddings.py   相机/帧/源/目标索引与时间编码
  encoder.py / attention.py     共享 encoder、同相机 attention 关系
  model.py                      encode/decode 组装与世界轨迹锚定
  geometry.py                   相机解码、可微反投影
  motion.py / visibility.py     全观察/历史 memory、运动 DPT、可见性
  tasks.py                      世界 scene flow、任意源像素轨迹采样
  data.py                       自备 JSON/JSONL manifest I/O
  datasets/
    registry.py / adapters/     懒加载原生文件读取，统一 UnifiedClip
    mixture.py / sampling.py    多源配额、原生窗口覆盖、联合 C/T
    bridge.py / multicam_slots.py  像素变换、共同坐标、观察槽合同
    dense.py / partial_dense.py GT 构造、未知标签 mask、动态/几何监督
    static_views.py             静态共视邻域
  train.py / distributed.py     训练循环、loss 图 DDP、RNG/指标聚合
  schedule.py                   可保存、恢复的学习率阶段
  losses.py / metrics.py        可用 GT 损失、无 confidence 权重评估
  infer.py / evaluate.py        推理与固定 Kubric 验证
  launch.py                     单机 GPU 预检与 torchrun
  configs/                      可读数据/损失示例，不嵌入训练权重
  vendor/                       原 4RC 组件及来源、许可
  audits/                       版本化 GT/训练检查证据
scripts/                        仓库级审计、数据池建设、训练调度工具
tests/multicamera4rc/            当前包的可重复 CPU/小型 DDP 测试
models/ datasets/ losses/ utils/ train_mixture.py test/
                                历史 D4RT 源码与入口
```

`scripts/` 中保留部分旧数据工具；它们不是安装包 API。当前工作流的工具按职责使用：

| 工作 | 主要脚本 |
|---|---|
| 全量 GT 准入与混合池 | `build_full_dense_mixture.py`、`validate_dense_mixture.py` |
| 静态共视数据 | `enable_static_multiview.py`、`audit_static_multiview.py`、`evaluate_static_multiview.py` |
| 双目/原生动态标签 | `enable_native_motion_stereo.py`、`audit_native_motion_stereo.py`、`preflight_native_motion_stereo.py` |
| 可变 C/T、50k | `build_variable_curriculum.py`、`audit_variable_curriculum.py`、`preflight_variable_ddp.py`、`build_variable_validation.py` |
| 质量恢复与选模 | `build_quality_curriculum.py`、`audit_quality_curriculum.py`、`continue_quality_recovery.py`；纯误差比较在 `multicamera4rc/quality.py` |
| 已审计运行的接续 | `cutover_native_motion_stereo.py`、`queue_variable_continuation.py` |
| decoder 架构迁移预检 | `preflight_decoder_memory.py`，真实 Kubric giant 单次前反向，不做 optimizer 更新 |

这些脚本从仓库根目录可用 `python -m scripts.NAME`，也保留安装项目后或 `PYTHONPATH=.` 的直接脚本调用方式。不会为了整理目录改写历史 checkpoint 的参数前缀。

## 此次审查落实的改进

1. **修正 decoder memory 范围。** 原实现只有目标时刻 KV，新默认是全观察 KV；按目标合并 cross-attention query，共用 KV 投影。self-attention/AdaLN 仍独立，时间子集不丢掉未查询的输入图。
2. **统一断点语义。** `config_from_checkpoint` 是训练、推理、验证和预检的共同恢复入口。旧断点按旧模式执行；显式 scope 切换保存在新 checkpoint/日志中，避免同权重在不同入口悄悄变架构。
3. **显式计算配置。** 暴露 `geometry_chunk_size`，修正恢复时 `--no-checkpointing` 被覆盖的问题；推理增加 AMP、关闭 query 分块和实际配置记录。
4. **稳定运行入口。** 新增统一 CLI 与安装命令，正确处理原 API 返回 Path 的情形；调整审计脚本的相对导入，支持模块入口。
5. **建立文档与测试边界。** 根 README 以当前项目为主，旧文档单独保存；新文档可正常纳入 Git；pytest 默认限定当前包，历史测试需显式指定。

没有移动/重命名 `backbone/head/cam_dec/motion_decoder/track_head` 等权重模块，也没有改动 GT 数学定义、课程配额和现场正在运行的任务。先前课程报告关于全时刻 KV 的错误表述已明确更正，原测量文件保留。

## 验证

```bash
python -m pip install -e '.[data,test,weights]'
python -m pytest -q
python -m multicamera4rc demo --output outputs/dev_smoke --train-steps 2
python -m multicamera4rc adapter-demo --output outputs/dev_adapter_smoke --train-steps 2
python -m pip wheel . --no-deps --wheel-dir outputs/wheels
```

新增 memory 回归检查：shared KV 与重复 KV 的输出/所有参数及输入梯度等价；checkpoint 重算等价；mask 后 KV 无梯度；未查询时刻直接参与全观察 memory；旧断点恢复逐元素保持历史输出；scope 显式迁移不改变参数键和形状。既有测试覆盖相机重命名、非参考观察重排、目标子集、源锚、几何/动态 GT、真实 I/O、优化器恢复和 DDP。

小模型通过不等于 giant 显存通过。涉及 attention 范围、源数、时间长度、图像尺寸、精度或 checkpointing 的修改，还需在隔离工作区用真实数据执行 giant 前向/反向，记录准确配置及峰值显存。模型质量需使用同一固定 held-out plan 比较。

## 扩展原则

增加模型选项时先写入 `MultiCamera4RCConfig` 并校验，明确缺字段的旧 checkpoint 默认值；不要只给某一个 CLI 加隐式行为。参数改名需提供明确映射并检查 optimizer 恢复，不能依赖 `strict=False` 吞掉不匹配。

新增数据先实现 adapter 并验证坐标/像素/帧号合同，再在 bridge/GT 层明确转换，最后加入准入配置；不要在模型 forward 中修补某一个数据集的标签问题。

GT 核验报告保留版本、数据路径、抽样范围和失败案例。大型权重、运行日志、缓存位于忽略的 `outputs/` 等目录；提交的是源码、配置示例和必要的轻量证据。

当前固定在旧 commit 的远端任务有 HEAD/文件哈希保护。仓库代码审查使用独立 worktree，提交到 GitHub 后也不自动覆盖活跃任务目录。
