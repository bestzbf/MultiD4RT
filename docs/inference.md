# 推理与 checkpoint

[返回 README](../README.md)

## 输入与命令

manifest 可以是 JSON 片段列表或 JSONL。图片路径相对于 manifest，使用同一相机 ID 和共享帧标签，例如：

```json
[{"id": "example", "observations": [
  {"image": "cam0_t0.png", "camera_id": 0, "time_idx": 0},
  {"image": "cam0_t5.png", "camera_id": 0, "time_idx": 5},
  {"image": "cam1_t0.png", "camera_id": 1, "time_idx": 0},
  {"image": "cam1_t5.png", "camera_id": 1, "time_idx": 5}
]}]
```

```bash
multicamera4rc infer --manifest observations.json --checkpoint model.pt --output outputs/prediction --source-obs 0 2 --query-times 0 5 --device cuda --amp --amp-dtype bfloat16 --no-query-chunking
```

`source_obs` 是展平后的观察下标；`query_times` 是帧标签，不是时间轴槽位。每个查询时刻需要至少一张有效输入图。观察 0 是参考角色，其他观察可以重排，只要元数据同步重排。

manifest 图片必须同样大小，且高宽为 14 的倍数、均至少 28。默认不 resize；`--image-size H W` 是显式 resize，调用方应选择保持比例的尺寸。原始数据更适合走 `--data-config config.yaml`：原生桥接层默认保留比例、补边，并同步变换 GT 内参、像素坐标和有效 mask。原生配置的尺寸变换应写在配置里，不与 CLI `--image-size` 混用。

原生适配器可能读取标定和 GT 以组织片段，但 forward 只接收 RGB、相机 ID、帧号及有效观察 mask。

## 权重兼容与输出

原生 checkpoint 保存模型 config 和权重。推理按保存的 `decoder_memory`、`time_origin`、宽度与层数加载；老版本缺少字段时使用兼容语义。`--decoder-memory`、`--time-origin` 是显式架构迁移选项，改变它们可能改变旧权重的输出。原 4RC 权重通过现有映射初始化，新增模块不等于已经训练好。

CLI 输出 `inference.json` 和按片段编号命名的 `clip_000000.pt` 等文件。张量保存到 CPU，结果中保留 `clip_id`；完整形状见 [模型结构](architecture.md)。输出坐标和尺度沿用模型约定，未知标定的 RGB 输入不保证绝对米制尺度。

可用 `model.encode()` 缓存同一输入，再用 `model.decode()` 请求不同源图与目标时刻。`multicamera4rc.tasks.scene_flow` 返回同一物理点的世界位移；`sample_tracks` 从源像素查询稠密预测轨迹，保留源点身份。

`demo` 和 `adapter-demo` 仅生成随机 tiny 权重验证 IO，不执行训练。
