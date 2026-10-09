# 代码结构与验证

[返回 README](../README.md)

| 路径 | 内容 |
|---|---|
| `multicamera4rc/model.py`、`config.py` | 模型接口、配置及 checkpoint 兼容语义 |
| `encoder.py`、`attention.py`、`embeddings.py` | 共享主干、相机关系与时间编码 |
| `geometry.py`、`motion.py`、`visibility.py` | 几何、基础点加位移与可见性 |
| `infer.py`、`tasks.py`、`checkpoint.py` | 推理 CLI、下游 API、权重读写 |
| `runtime.py`、`evaluation.py`、`metrics.py` | 设备/评估上下文与独立物理指标 |
| `datasets/` | 原生数据格式、坐标桥接与 GT recipes |
| `benchmarks/` | 安装后可用的单目、动态与静态多视角评估 |
| `vendor/` | 保留来源和许可的上游模型组件 |
| `tests/` | 模型、推理、坐标/GT 和评估协议回归检查 |
| 根目录 `benchmarks/results/` | 冻结的正式单目评估结果 |

`scripts/evaluate_*.py` 是旧命令路径的薄兼容入口，实际实现位于可安装的 `multicamera4rc.benchmarks`。不再保留重复的旧 D4RT 模型树、训练控制器或运行过程记录。

```bash
python -m pip install -e '.[test,data]'
python -m pytest
```

测试覆盖帧标签/相机身份、可变相机与时间布局、无效观察、输出锚定、checkpoint 映射、数据 GT 单位与坐标、网格运动、指标对齐，以及无需训练模块的推理和评估。PyTorch3D 对照项在未安装该可选库时跳过；SciPy 对照仍会运行。

发布验证还应从 wheel 在源码目录外运行 CLI、推理 demo 和原生适配 demo，防止测试意外依赖开发仓库。模型核心文件保持 v0.2.0 原样；数据 GT mask 和正式单目结果不因清理而改变。
