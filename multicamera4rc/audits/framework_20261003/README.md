# 框架与模型结构审查：2026-10-03

本次更新修正 decoder 直接 KV 的时间范围，增加共享 all-observation memory，并将旧断点加载语义集中管理。文档入口：[README](../../../README.md)、[模型结构](../../../docs/architecture.md)、[代码组织](../../../docs/development.md)。

## 验证结果

| 检查 | 实际结果 |
|---|---|
| 修改前 CPU 基线 | 220 项通过 |
| 修改后 macOS / Python 3.12 / PyTorch 2.14 | 226 项通过，31.55 秒 |
| 修改后 B300zbf / Python 3.10 / PyTorch 2.10.0+cu128 | 226 项通过，80.65 秒 |
| 统一 console `demo` | 2 次小模型训练、保存、推理，退出码 0 |
| 统一模块 `adapter-demo` | 实际 H5 写入/读取、2 次训练、保存、推理，退出码 0 |
| train/infer/evaluate/launch/demo/adapter-demo `--help` | 6 个入口成功 |
| 6 个审计/课程工具的模块与直接脚本 `--help` | 12 个入口成功 |
| 安装/打包 | editable 安装与 wheel 构建成功；包含 CLI、配置和 vendor 许可，未打包顶层旧模型 |
| 独立 wheel 导入 | 离开仓库目录，从隔离安装位置执行 README 示例成功，轨迹为 `[1,2,3,28,42,3]` |
| 可变布局示例配置 | B300 读取 4,752 个训练场景，256 个采样计划覆盖 C1–C10 的全部 11 种示例布局 |
| Markdown 本地文件链接 | 新文档、根 README、包 README 均通过 |

两地回归各有一条来自 PyTorch 的 `torch.jit.script` 弃用提示，不影响测试结果。

## 新模式的真实 giant 前反向

证据：[giant_all_observations_c4t6.json](giant_all_observations_c4t6.json)。从旧 25k native 断点加载相同权重，显式切换 `decoder_memory=all_observations`：

- B300 物理 GPU5，BF16，原生 Kubric `scene_E_001009`；输入 `[1,24,3,266,266]`，C4T6，Q2，实际源观察下标 12/19。
- 几何 + motion + visibility 的 dense loss 前向/反向完成，参与反传的 1,058 个参数张量梯度有限；camera relation bias 收到非零梯度。
- attention/visibility query 不分块，激活检查点开启；几何 DPT 沿观察的 chunk 为 8。
- 峰值已分配显存 **19.205 GiB**，峰值保留显存 **20.520 GiB**；共享卡环境单次前反向及梯度检查约 **25.54 秒**，不含权重/数据加载。
- 证据包含实际模型配置、帧号、相机 ID、损失/误差、源码 SHA-256。与提交前本地 `multicamera4rc/*.py` 及所有子目录 Python 源码逐文件对照一致。

该检查只验证一个真实布局的数值/梯度与显存，没有执行 optimizer 更新，不能据此判断新 scope 比旧 scope 的精度更好，也不能将其显存推广到原分辨率 PointOdyssey/Replica 长序列。

可重复运行：

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python -m scripts.preflight_decoder_memory \
  --data-config multicamera4rc/configs/kubric_multicam_native.yaml \
  --checkpoint /path/to/step_00025000.pt \
  --decoder-memory all_observations --cameras 4 --times 6 --source-count 2 \
  --memory-fraction .16 --output outputs/giant_memory_preflight.json
```

## 现场训练边界

原训练目录 `/data9/zbf/my_dfrt_multicamera4rc` 保持 `7a2038a`，未热更新。旧 25k 训练及验证结束后，原控制器已自动进入 `training_stage_1`，目标 50k，继续使用旧 `target_time` 模式。新框架在 `/data9/zbf/my_dfrt_framework_review_20261003` 独立验证，未停止原任务、改动其证据或重置训练游标。

旧课程报告关于“decoder KV 包含所有时间”的说法已更正；[历史测量](../variable_ct_20261003/README.md)继续明确标为旧模式，保留原证据。
