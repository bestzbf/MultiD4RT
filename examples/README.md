# 精选重建实例

打开 [index.html](index.html) 查看五组真实模型重建。可离线旋转三维点云、播放时间、切换相机、查看深度和添加三维轨迹，无需安装 Python 或下载 checkpoint。

实例使用同一份已训练的 MultiD4RT giant checkpoint（step 56,000），执行原始分辨率 CUDA / bfloat16 推理。网络输入为 RGB、相机 ID、共享帧号与查询索引；相机内外参、深度、几何和轨迹均为模型预测。原图仅在右侧、底部补齐 patch 边界。

| 实例 | 输入 | 原图高 × 宽 | 原始帧号 | 显示点数 |
|---|---|---|---|---|
| [Syn4D · Temple](syn4d_temple_c2t4/viewer.html) | C2 × T4 | 720 × 1280 | 101、106、111、116 | 30,000 |
| [Syn4D · Warehouse](syn4d_warehouse_c2t4/viewer.html) | C2 × T4 | 720 × 1280 | 15、20、25、30 | 30,000 |
| [BlendedMVS · 城市建筑](blended_city_c8t1/viewer.html) | C8 × T1 | 420 × 560 | 0 | 50,000 |
| [Hypersim · 室内](hypersim_c4t1/viewer.html) | C4 × T1 | 768 × 1024 | 0 | 50,000 |
| [PointOdyssey · 人物序列](pointodyssey_c1t16/viewer.html) | C1 × T16 | 540 × 960 | 30–75，间隔 3 | 25,000 |

这些案例从固定验证样例中按重建质量与实际画面筛选，供展示使用，不代表完整验证集的平均性能。各来源保留原有数据权利。

## 文件与显示

```text
syn4d_temple_c2t4/
  viewer.html             离线三维交互页面
  preview.png             实际浏览器预览
  preview-last.png        最后一个目标帧预览（动态案例）
  inputs/
    observations.json     相机 / 帧号与原图路径
    camera_*/frame_*.png   原始 RGB
  queries.json            从原图人物区域选取的追踪像素（动态案例）
  provenance.json         权重、输入尺寸、推理用时、输入哈希与显示设置
  visualization.json      帧号、相机、显示点数与默认视角
  ply/                    当前视角几何与固定源点的逐帧 PLY
  tracks.npz              三维轨迹及源像素查询
  scene_flow.npz          相邻输出时刻之间同一源点的世界位移
```

显示点由有限、正深度的图像内容像素均匀取样，使用固定种子且不重复；不使用 GT、置信度或重建误差过滤。源点贯穿全部目标帧，补边不进入点云。动态案例的初始查询从第一张原图中的人物区域手动选取；可在页面点选其他源点添加轨迹。默认观察视角只改变观看方向，世界坐标与模型预测保持原值。

单时刻案例默认显示当前视角重建，播放按钮禁用；动态案例默认显示固定源点的三维轨迹。页面中的深度和预测相机可单独查看。三维单位由 checkpoint 决定。

## 重新推理

在项目根目录运行：

```bash
pip install -e '.[weights,viz]'

multid4rt visualize \
  --input examples/syn4d_temple_c2t4/inputs/observations.json \
  --queries examples/syn4d_temple_c2t4/queries.json \
  --checkpoint model.pt --long-side 0 --max-frames 0 --max-points 30000 \
  --output outputs/temple --device cuda --amp --no-visibility
```

其他动态案例可同时替换输入清单和查询文件。静态案例只需要输入清单，可设置 `--max-points 50000`；PointOdyssey 使用 `--max-points 25000`。manifest 显式记录相机与时间关系，默认 `--mode auto` 可辨识任务类型。`--long-side 0 --max-frames 0` 保留本页选中的原始尺寸和全部时刻。

各组 `provenance.json` 记录具体序列、源观察、帧号和 RGB SHA256。显示点数不改变完整稠密预测；更多参数见[可视化说明](../docs/visualization.md)。完整 checkpoint 与服务端稠密预测文件可在自行推理时生成。
