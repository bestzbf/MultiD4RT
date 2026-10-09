# MultiD4RT

**Multi-camera 4D Reconstruction and Tracking**

MultiD4RT 从同步多相机 RGB 序列重建随时间变化的三维场景，输出稠密几何、scene flow、三维轨迹和逐相机可见性。输入只需要图像、相机身份和共享帧号，无需已知内外参或 timestamp。相机数量与序列长度可以变化。

## 安装

```bash
git clone --branch v0.5.0 https://github.com/bestzbf/MultiD4RT.git
cd MultiD4RT
python -m pip install -e '.[weights,viz]'
multid4rt --help
```

## 模型输入与输出

同步图像为 `[B,C,T,3,H,W]`，按相机优先展平为 `[B,S,3,H,W]`，其中 `S=C*T`。RGB 使用 `[0,1]` 浮点值或 uint8。保持图像比例，将高宽准备为 14 的倍数且均至少 28；基础 `infer` 命令默认不缩放。

- `camera_id [B,S]`：同一物理相机共享 ID，片段内一致即可。
- `time_idx [B,S]`：同步相机共享帧号，保留抽帧间隔，例如 `0,5,10`。
- `observation_valid [B,S]`：可选，用于屏蔽缺失观察。
- `source_obs_idx [B,Q]`：选择哪些输入图作为轨迹的源图。
- `query_time_idx [B,Tq]`：请求输出的目标帧号。

例如 C2T3：

```text
观察下标     0   1   2   3   4   5
camera_id    0   0   0   1   1   1
time_idx     0   5  10   0   5  10
```

| 输出 | 维度 | 含义 |
|---|---|---|
| `depth` | `[B,S,H,W]` | 每张输入图的深度 |
| `K` / `c2w` | `[B,S,3,3]` / `[B,S,4,4]` | 预测内参与相机位姿 |
| `base_points` | `[B,S,H,W,3]` | 统一世界坐标中的基础点 |
| `trajectory` | `[B,Q,Tq,H,W,3]` | 源图每个像素的三维轨迹 |
| `displacement` | `[B,Q,Tq,H,W,3]` | 相对基础点的三维位移 |
| `visibility` | `[B,Q,Tq,C,H,W]` | 每个目标时刻、相机中的可见性 |

## 模型结构

共享 encoder 先提取每张图的局部特征，再融合全部相机与时刻。相机身份通过同相机 attention bias 编码；共享帧编码和 time token 提供时间信息。

几何分支预测深度、内参和位姿，反投影得到基础点 `P`。运动 decoder 使用全部输入观察的 patch 特征作为 KV，以目标时间和源到目标的时间差进行 AdaLN 调制，预测三维位移 `Δ`。最终轨迹为 `X(t)=P+Δ(t)`，源时刻严格满足 `Δ(t_source)=0`。可见性分支结合轨迹投影与目标图像特征输出逐相机可见性。

giant 配置使用 patch14、1536 维 encoder 和 40 层；完整模块与中间张量维度见 [模型结构](docs/architecture.md)。

## Python 接口

```python
import torch
from multid4rt import MultiD4RT, MultiD4RTConfig, pack_synchronized_images

model = MultiD4RT(MultiD4RTConfig.tiny()).eval()
rgb = torch.rand(1, 2, 3, 3, 28, 42)
inputs = pack_synchronized_images(rgb, frame_indices=[0, 5, 10])

with torch.inference_mode():
    pred = model(**inputs, source_obs_idx=[0, 3])

print(pred['trajectory'].shape)  # [1,2,3,28,42,3]
print(pred['visibility'].shape)  # [1,2,3,2,28,42]
```

tiny 示例使用随机权重检查接口。实际推理加载对应 checkpoint：

```python
model = MultiD4RT.from_checkpoint('model.pt').eval()
```

`model.encode()` 的结果可供同一输入下多个源图/目标时间请求复用。`multid4rt.tasks.scene_flow` 提取同一物理点的世界位移，`sample_tracks` 查询源图像素的三维轨迹。

## 命令行推理

准备图片与相机/时间标签组成的 JSON 或 JSONL manifest，然后运行：

```bash
multid4rt infer --manifest observations.json --checkpoint model.pt --output outputs/prediction --device cuda --amp --amp-dtype bfloat16
```

输出 `inference.json` 和各片段的 `.pt` 预测结果。manifest 格式、源图选择、目标帧与 checkpoint 配置见 [推理说明](docs/inference.md)。

```bash
multid4rt demo --output outputs/demo
```

CPU demo 用随机 tiny 权重验证输入输出流程。

## 推理可视化

同一个命令支持三种输入：

```bash
# 多相机 4D：每个相机一个子目录，使用共享帧号
multid4rt visualize --input camera_sequences/ --mode multicamera --checkpoint model.pt --output outputs/multicamera --device cuda --amp

# 多视角单时刻：一个目录中的图片代表不同相机
multid4rt visualize --input scene_views/ --mode multiview --checkpoint model.pt --output outputs/multiview --device cuda --amp

# 单相机动态重建 / 追踪：视频或连续帧目录
multid4rt visualize --input video.mp4 --mode monocular --checkpoint model.pt --output outputs/monocular --device cuda --amp
```

打开输出目录中的 `viewer.html`，可以旋转三维场景、播放时间、切换相机、查看 RGB/深度和点选三维轨迹。页面离线可用，另导出 PLY、轨迹与 scene flow。可视化默认最多取 16 帧、等比例缩小至长边 518，使用 `--max-frames 0 --long-side 0` 保留全部帧和原始尺寸；补边像素会从显示与导出中排除。

没有 checkpoint 时可先检查界面：

```bash
multid4rt visualize --demo --mode multicamera --output outputs/viewer_demo --device cpu
```

此示例使用随机 tiny 权重，仅验证流程。文件夹格式、视频输入、原图像素查询、预测复用和导出维度见 [可视化说明](docs/visualization.md)。

组件来源与许可见 [NOTICE](NOTICE)。
