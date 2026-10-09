# MultiD4RT 推理

[返回 README](../README.md)

## 图像与 manifest

输入为同一片段内的 RGB、相机 ID 和共享整数帧号。manifest 可以是 JSON 片段列表、包含 `clips` 的 JSON，或每行一个片段的 JSONL。图片路径相对于 manifest：

```json
[{"id": "example", "observations": [
  {"image": "cam0_t0.png", "camera_id": 0, "time_idx": 0},
  {"image": "cam0_t5.png", "camera_id": 0, "time_idx": 5},
  {"image": "cam1_t0.png", "camera_id": 1, "time_idx": 0},
  {"image": "cam1_t5.png", "camera_id": 1, "time_idx": 5}
]}]
```

同一片段的图片需准备成相同大小，高宽为 14 的倍数且均至少 28。默认不 resize；保持比例准备图像，可以补边。`--image-size H W` 为显式 resize，调用方应选择保持比例的尺寸。

同一物理相机共享 ID，各相机使用同一时间轴；帧号保留抽帧间隔。需要缺失观察时可在 manifest 中设置 `valid: false`，观察 0 必须有效。相机 0 的参考角色由观察顺序决定。

## 运行

```bash
multid4rt infer --manifest observations.json --checkpoint model.pt --output outputs/prediction --source-obs 0 2 --query-times 0 5 --device cuda --amp --amp-dtype bfloat16 --no-query-chunking
```

`source_obs` 是展平后的输入下标，`query_times` 是帧标签。每个请求时刻至少需要一张有效输入图。不传这两个参数时返回所有源观察和所有输入时刻。`--no-visibility` 可省略可见性分支。

结果保存为 CPU 张量：`inference.json` 记录运行配置，`clip_000000.pt` 等文件保留 `clip_id` 与预测字段。完整形状见 [模型结构](architecture.md)。输出的世界尺度沿用 checkpoint 约定。

## Checkpoint 与接口

`MultiD4RT.from_checkpoint()` 恢复保存的模型配置和权重，也兼容更名前的原生 checkpoint。参数键、张量形状、时间编码与 decoder KV 语义保持一致。`--decoder-memory`、`--time-origin` 是显式覆盖选项，可能改变旧权重的输出。

```python
from multid4rt import MultiD4RT
from multid4rt.tasks import scene_flow, sample_tracks

model = MultiD4RT.from_checkpoint('model.pt').eval()
```

同一输入可先 `model.encode()`，再通过 `model.decode()` 查询不同源图和目标时刻。`scene_flow` 返回被跟踪点的世界位移；`sample_tracks` 使用 `[source_obs,u,v]` 的像素查询提取稠密预测。

`multid4rt demo --output outputs/demo` 使用随机 tiny 权重检查 IO。
