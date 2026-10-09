# MultiCamera4RC

基于 4RC 的同步、未知内外参多相机 4D 模型，输出几何重建、scene flow、三维轨迹与逐相机可见性。输入仅需 RGB、片段内相机 ID 和共享整数帧号；不要求 timestamp 或 GT 相机参数。

**v0.3.0 是推理与评估发布版。** 保留原模型和 checkpoint 兼容行为，移除训练入口、优化器、损失函数、学习率控制与训练过程记录。完整历史快照仍在 [v0.2.0](https://github.com/bestzbf/MultiCamera4RC/releases/tag/v0.2.0)。预训练权重和数据集需要另行准备。

## 安装与运行

```bash
git clone --branch v0.3.0 https://github.com/bestzbf/MultiCamera4RC.git
cd MultiCamera4RC
python -m pip install -e '.[data,weights]'
multicamera4rc --help
```

可以先在 CPU 上验证完整输入输出流程：

```bash
multicamera4rc demo --output outputs/demo
multicamera4rc adapter-demo --output outputs/adapter_demo
```

这两个 demo 使用随机初始化的 tiny 模型，只验证接口与张量形状，不能用于判断重建质量。

## 模型接口

同步输入 `[B,C,T,3,H,W]` 按相机优先展平为 `[B,S,3,H,W]`，其中 `S=C*T`。相机数、时刻数可以变化，也支持缺失观察。

```python
import torch
from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig, pack_synchronized_images

model = MultiCamera4RC(MultiCamera4RCConfig.tiny()).eval()
images, camera_id, time_idx = pack_synchronized_images(torch.rand(1, 2, 3, 3, 28, 42))
with torch.inference_mode():
    pred = model(images, camera_id, time_idx, source_obs_idx=[0, 3])
print(pred['trajectory'].shape)  # [1,2,3,28,42,3]
print(pred['visibility'].shape)  # [1,2,3,2,28,42]
```

`H,W` 必须是 14 的倍数且均至少 28。原生适配器默认保留原始图像比例并补边；manifest 输入需要事先准备成相同、合法的大小，默认不 resize。

主干继承 4RC/DINOv2；giant 为 patch14、1536 维、40 层。相机身份通过同相机 attention bias 编码，时间通过共享帧编码、time token 和 decoder AdaLN 进入网络。几何分支预测深度与相机，运动分支输出基础点加三维位移。新模型 decoder 的 KV 包含全部输入时刻和相机；加载旧 checkpoint 时保留其保存的时间与 KV 语义。

## 推理与评估

```bash
multicamera4rc infer --manifest observations.json --checkpoint model.pt --output outputs/prediction --device cuda --amp --amp-dtype bfloat16
multicamera4rc monocular --dataset scannet --roots /path/to/scans_test --output outputs/scannet --plan outputs/scannet/plan.json --audit-only
multicamera4rc monocular --dataset scannet --roots /path/to/scans_test --output outputs/scannet --plan outputs/scannet/plan.json --checkpoint model.pt --device cuda
```

安装后提供 `infer`、`evaluate`、`monocular`、`native-dynamic`、`static-multiview`、`demo`、`adapter-demo` 七个命令；各命令支持 `--help`。评估直接计算几何、运动及可见性指标，不依赖训练损失。

- [模型结构与完整维度](docs/architecture.md)
- [推理、manifest 和 checkpoint](docs/inference.md)
- [原生数据适配与 GT 约定](docs/data.md)
- [单目评估协议](docs/monocular-evaluation.md)
- [Syn4D 数据适配](docs/syn4d.md)
- [完整 55,500 步单目结果](benchmarks/results/monocular_55500/README.md)
- [代码结构与验证](docs/development.md)
- [发布检查记录](releases/v0.3.0/README.md)

第三方许可与来源见 [NOTICE](NOTICE)。本仓库未将上游组件重新声明为统一许可。
