# v0.3.0 推理与评估发布版

[返回项目](../../README.md) · [机器可读验证](validation.json)

精简自 v0.2.0，移除训练入口、损失函数、优化器与恢复/LR 控制、训练专用配置和测试、重复的旧 D4RT 代码树及运行过程记录。保留模型、checkpoint 兼容、原生数据适配、GT recipes、独立评估、正式单目结果与第三方许可。原 v0.2.0 tag 和发布附件继续保留。

## 验证

- 207 项测试通过，3 项可选 PyTorch3D 对照跳过；原生/SciPy 指标对照通过。
- 从 wheel 安装，在源码目录外完成 15 项检查：全部命令帮助、两个推理 demo、单目 GT 审计与几何评分、静态多视角评分、原生 renderer 稠密评分和动态评分。
- 51 个模型核心及 vendor 文件与 v0.2.0 完全相同，SHA-256 记录在 validation.json。
- 指标函数实现保持一致；原生评估改为直接计算物理误差，不再调用训练损失。
- 已完成的 ScanNet 100 与 Sintel 14 场景正式结果、固定计划和 GT 审计保持原始内容；清理过程没有重新计算或改善分数。

测试运行于 Python 3.12 / PyTorch 的 CPU 环境，未在本次清理中验证 CUDA。随机 tiny checkpoint 检查安装、数据读取、维度和调用链，不代表真实模型精度。预训练权重、数据集和网格资产不随发布包附带。

## 发布附件

源码包 `multicamera4rc-0.3.0.tar.gz`、wheel `multicamera4rc-0.3.0-py3-none-any.whl`、`release-validation.json` 和 `SHA256SUMS`。源码包包含文档、测试和冻结结果；wheel 包含可运行模型、适配器和七个 CLI 入口。

```bash
python -m pip install 'multicamera4rc-0.3.0-py3-none-any.whl[data,weights]'
multicamera4rc --help
```
