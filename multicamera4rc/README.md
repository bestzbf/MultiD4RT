# MultiCamera4RC 包

同步多相机、未知内外参的几何重建、scene flow 和三维追踪。当前项目的安装、最小输入示例及运行入口统一维护在[仓库 README](../README.md)。

| 说明 | 入口 |
|---|---|
| 模型与完整维度 | [architecture.md](../docs/architecture.md) |
| 数据、GT、采样比例 | [data.md](../docs/data.md) |
| 训练、推理、断点、50k 课程 | [training.md](../docs/training.md) |
| 模块职责与开发流程 | [development.md](../docs/development.md) |
| 历史 GT/训练检查证据 | [audits](audits/) |

默认新模型使用 `decoder_memory=all_observations`。历史 native checkpoint 缺少此字段时恢复为 `target_time`；旧服务器任务使用旧模式。两者共享参数形状，但 attention 范围不同。明确切换才改变模型行为，详见模型文档的兼容说明。

稳定 Python 入口：

```python
from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig, pack_synchronized_images
from multicamera4rc.checkpoint import config_from_checkpoint, save_checkpoint
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.tasks import scene_flow, sample_tracks
```

CLI：`python -m multicamera4rc --help`。所有训练、推理与验证入口都使用同一套 checkpoint 配置恢复语义。
