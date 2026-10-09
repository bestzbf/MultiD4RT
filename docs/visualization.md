# 推理可视化

`multid4rt visualize` 将普通 RGB 输入送入模型，再生成离线交互页面和三维导出。需要安装可选依赖：

```bash
python -m pip install -e '.[weights,viz]'
multid4rt visualize --help
```

## 三种输入

| 模式 | 输入 | 相机 / 时间标签 |
|---|---|---|
| `multicamera` | 每相机一个图片子目录，或每相机一个视频 | 子目录 / 视频对应相机；共享帧号对应时刻 |
| `multiview` | 一个目录中的多视角照片 | 每张图为不同相机，全部 `time_idx=0` |
| `monocular` | 一个视频，或一个目录中的连续帧 | 同一相机，保留帧号与抽帧间隔 |

### 同步多相机 4D

```text
camera_sequences/
  camera_0/
    frame_00000.png
    frame_00005.png
    frame_00010.png
  camera_1/
    frame_00000.png
    frame_00005.png
    frame_00010.png
```

```bash
multid4rt visualize --input camera_sequences/ --mode multicamera \
  --checkpoint model.pt --output outputs/multicamera --device cuda --amp
```

目录名按自然顺序分配相机 ID，文件名末尾的整数作为共享帧标签。同相机始终使用相同 ID。只选择所有流共有的帧号，再按 `--stride` 抽帧，因此缺帧不会让左右相机错位；没有共同帧号会报错。一时刻的多相机输入自动显示为多视角重建。

也可以在一个目录内放 `camera_0.mp4`、`camera_1.mp4` 等同步视频。视频帧标签为原始解码下标：`--stride 5` 对应 `0,5,10,...`。这要求各视频第 0 帧对应同一时刻，帧率与帧采样方式相同；程序不会估计时钟偏移。图片序列不要求 timestamp。

### 多视角单时刻重建

```text
scene_views/
  view_0.jpg
  view_1.jpg
  view_2.jpg
```

```bash
multid4rt visualize --input scene_views/ --mode multiview \
  --checkpoint model.pt --output outputs/multiview --device cuda --amp
```

输入是同一静态场景的不同视角。时间滑块和播放按钮自动禁用，默认显示当前各视角的世界点云及预测相机。模型按 C×T 的普通输入路径执行，此处 `T=1`。

### 单相机动态重建与追踪

```bash
multid4rt visualize --input video.mp4 --mode monocular \
  --checkpoint model.pt --output outputs/monocular --stride 5 --device cuda --amp
```

连续帧目录也可使用同一命令。文件名全部以数字结尾时保留原始帧号；否则按文件名自然排序，以 `0,1,2,...` 作为抽帧前的帧号。一个普通图片目录同时可能代表多视角照片或连续帧，因此有多张图时须指定 `--mode multiview` 或 `--mode monocular`。单个视频、单张图片、相机子目录和有标签的 manifest 可以使用默认 `--mode auto`。

## 图像准备与模型查询

默认最多选择 16 个时刻，各图等比例缩小到长边不超过 518，不放大小图；最后在右侧和底部补黑，得到公共 `[H,W]`，高宽均为 14 的倍数且至少 28。原图尺寸、实际缩放比例、内容范围和相机 / 时间标签保存在 `input/preprocessing.json`。补边像素不会显示或导出成点云。

```bash
# 保留全部帧与原始图像尺寸，仍会补齐 patch 所需边界
multid4rt visualize --input camera_sequences/ --checkpoint model.pt \
  --max-frames 0 --long-side 0 --output outputs/native --device cuda --amp
```

实际网络输入为 `[1,S,3,H,W]`，同步矩形片段中 `S=C*T`。默认取每相机在输入顺序中的第一张图作为轨迹源，`Q=C`；目标帧为所有选中共享帧。这样三维追踪源保持固定，随时间展示同一批物理点。也可以显式指定 `--source-obs 0 3` 与 `--query-times 0 10`。源图下标表示准备后的观察列表位置，具体对应关系见 `input/observations.json`。

编码器仍接收全部选中观察，decoder 的记忆范围与时间编码沿用 checkpoint 配置。改变显示点数不改变网络的稠密预测。默认关闭查询分块；需要使用 checkpoint 原有分块参数时添加 `--query-chunking`。`--no-visibility` 可以省去逐相机可见性分支。

相机内外参由模型预测，不要求输入标定。三维坐标与深度的单位由 checkpoint 决定，不能仅凭未标定 RGB 保证是米。

## 显式指定追踪像素

`queries.json` 使用原始图像中的像素坐标 `[source_observation,x,y]`，例如：

```json
[[0, 320, 180], [3, 145.5, 220.5]]
```

```bash
multid4rt visualize --input camera_sequences/ --checkpoint model.pt \
  --source-obs 0 3 --queries queries.json --output outputs/selected --device cuda --amp
```

原图坐标转换为准备后的坐标，采用像素中心映射 `u'=(u+0.5)*scale_x-0.5`，`v'=(v+0.5)*scale_y-0.5`，边缘裁到图像内容范围，然后调用 `sample_tracks` 双线性采样。查询源必须已经被 `--source-obs` 解码。未传 `--queries` 时，在显示的源点中均匀选取最多 `--num-tracks 24` 条轨迹；也可在网页中点选源点增加轨迹。

## 页面与导出

直接打开 `viewer.html`，无需启动服务器：

- 旋转、缩放三维场景，播放或拖动共享帧标签。
- “跟踪源点”展示固定源像素随时间的三维位置；“当前视角重建”展示当前输入图像各自预测的几何。
- 使用 RGB、相机身份或三维位移着色；切换相机过滤点云与图像。
- 显示预测相机位置和视锥；查看当前 RGB 上的三维轨迹投影或预测深度。
- 点击“跟踪源点”中的点添加完整轨迹；重置视角恢复默认观察位置。

页面内嵌 Plotly.js 和图像，不依赖 CDN，采用 [Plotly 官方离线 HTML 方式](https://plotly.com/python/interactive-html-export/)。轨迹表示源像素对应的物理点；点被遮挡时仍可能存在三维位置，页面未以可见性阈值删除轨迹。不同相机的点云直接放入共同世界坐标，未额外做配准或融合。

```text
outputs/multicamera/
  viewer.html
  visualization.json
  input/observations.json
  input/preprocessing.json
  input/observation_*.png
  prediction/inference.json
  prediction/clip_000000.pt
  ply/frame_0000_t0_geometry.ply
  ply/frame_0000_t0_tracked.ply
  tracks.npz
  scene_flow.npz
```

| 文件 | 数据 |
|---|---|
| `prediction/clip_000000.pt` | 完整稠密预测，含 depth、K、c2w、base_points、trajectory、displacement 等 |
| `*_geometry.ply` | 当前输入视角的世界 XYZ 和 RGB，二进制 PLY |
| `*_tracked.ply` | 固定源点在该目标帧的世界 XYZ 和源 RGB |
| `tracks.npz` | `trajectory [N,Tq,3]`、`valid [N,Tq]`、`queries [N,3]`、`time_idx [Tq]`；queries 为准备后像素 |
| `scene_flow.npz` | `displacement [Tq-1,M,3]`、`valid [Tq-1,M]`、`source_obs_idx [M]`、`source_uv [M,2]`、起止帧标签 |

Scene flow 是相邻**输出帧标签**之间同一源点的世界位移，不是除以时间间隔后的速度。单时刻输入的 flow 第一维为 0。`valid` 标记有限三维点，包含遮挡点；无效 flow 置零并须结合 mask 使用。

`--max-points 20000` 限制每帧显示与 PLY / scene-flow 导出的点数；源点只采样一次，后续帧保持身份一致。`--max-points 0` 保留全部有效点，HTML 会更大。完整 `.pt` 预测不受此参数影响。`--max-depth` 可在 checkpoint 深度单位下过滤背景显示点，不改模型输出；显式像素轨迹查询仍按原请求采样。

## 复用预测

```bash
# 已有完整 demo 输出目录，无需再加载模型或推理
multid4rt visualize --prediction outputs/multicamera \
  --max-points 10000 --num-tracks 12 --output outputs/new_viewer

# 已有 infer 的 .pt，需提供与预测同顺序、同尺寸的 RGB manifest
multid4rt visualize --prediction outputs/prediction/clip_000000.pt \
  --input observations.json --output outputs/from_prediction
```

复用预测时默认不抽帧、不缩放；使用 demo 输出目录时自动恢复原来的内容范围与坐标变换。原始预测文件保持不变。若只有 `.pt`，输入图必须已经按同样预处理准备；相机 / 帧号 / 分辨率不一致会报错。改变轨迹源或目标时间需要重新推理。

## 无权重的界面检查

```bash
multid4rt visualize --demo --mode multicamera --output outputs/demo_multicamera --device cpu
multid4rt visualize --demo --mode multiview --output outputs/demo_multiview --device cpu
multid4rt visualize --demo --mode monocular --output outputs/demo_monocular --device cpu
```

这三条命令生成 RGB 示例并使用随机 tiny 权重，页面会明确标注。它们验证读取、模型执行、三维导出和交互，不代表重建质量。实际效果请使用对应的已训练 checkpoint。
