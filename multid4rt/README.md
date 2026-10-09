# MultiD4RT

Multi-camera 4D Reconstruction and Tracking.

[项目入口](../README.md) · [模型结构](../docs/architecture.md) · [推理说明](../docs/inference.md) · [可视化](../docs/visualization.md)

```python
from multid4rt import MultiD4RT, MultiD4RTConfig, pack_synchronized_images
```

输入 RGB、相机 ID 和共享帧号；输出深度、相机参数、世界基础点、三维轨迹、位移与逐相机可见性。命令行为 `multid4rt infer`、`multid4rt visualize` 和 `multid4rt demo`。
