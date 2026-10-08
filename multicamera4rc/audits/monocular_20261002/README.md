# 单目混合训练 GT 核验（2026-10-02）

更新：下文保留最初 GT 核验时的状态。后续已完成密集适配并启动全量有效数据混合训练，当前实现与运行证据见[全量混合训练报告](../full_mixture_20261002/README.md)。

可以加入单目数据，但必须按实际 GT 能力监督。优先动态候选是原始 VKITTI2 和
DynamicReplica；PointOdyssey、ScanNet、ScanNet++、WildRGBD 和 TartanAir V1 可以补充几何。
“存在原生 Adapter”“GT 抽样一致”“已接入当前稠密训练”是三件不同的事。
当前 `MixtureDataset(supervision="dense")` 仍只接受 `kubric_multicam`，本次没有修改训练
数据组合、损失、模型或正在 GPU 4、5 上运行的作业。

核验在 B300zbf 上只读执行，GPU 不参与。检查了两套原始轨迹数据、VKITTI2、
TartanAir/Hypersim ZIP 和 `/data7/mugen` 的 14 个 H5 包。原始数据核验包括独立的
2D/3D 轨迹重投影、深度与可见顶点的关系，以及 GT 光流与相机/深度/scene flow 的
交叉验证；打包几何检查使用相邻视图的深度一致性，同时诊断 w2c/c2w 两种约定。
这是分层抽样与格式核验，不是全库逐帧认证，数值自洽也不能证明估计标签具有传感器真值精度。

机器可读结论见 [summary.json](summary.json)，逐样本测量在 [measurements](measurements/)，
只读核验脚本在 [scripts](scripts/)。远程原始报告目录是
`/data9/zbf/my_dfrt_multicamera4rc/outputs/training_preflight/monocular_gt_audit_20261001_1536`。
时间目录使用开始核验时的日期，文档日期使用北京时间。

**可用范围与接入条件**

| 数据 | 本机数据与覆盖 | 抽样结论 | 当前模型可监督的任务 | 接入条件 |
|---|---|---|---|---|
| 原始 VKITTI2 | `/data8/mugen/vKITTI2`；每路 50 个场景变体；5 个物理场景 × 两路相机 × 3 对帧 | 11,835,149 个有效 scene flow 像素；与光流交叉投影的逐帧对 P95 为 0.031–0.333 px；静态背景坐标差残差 P95 为 0.375–0.603 mm | 稠密深度/基础点、相机、原生相邻帧三维位移 | 转换两个时刻的相机坐标，屏蔽无效/截断值；按物理场景划分；新增 dense recipe |
| DynamicReplica | `/data5/d4rt_dataset/Dynamic_Replica`；当前 Adapter 索引 483 条 left 流；6 条序列 × 5 个轨迹帧，另测 12 对稠密光流 | 502,450 个可见顶点观察；重投影 P95 约 0.00010–0.00022 px；过滤深度边界后，光流端点的 world 误差相对深度 P95 为 0.103%–0.606% | 稠密几何、两帧均可见区域的稠密 scene flow | IEEE float16 位模式解码；光流端点深度反投影；遮挡、边界和端点有效性掩码；新增 dense flow recipe |
| PointOdyssey | `/data5/d4rt_dataset/PointOdyssey`；109 条 train 序列；8 条序列 × 5 帧，266,190 个可见轨迹观察 | H5 相机参数与原始 NPZ 完全一致；轨迹重投影 P95 为 0.261–0.933 px，符合原始半精度量化/边界 UV 调整现象 | 稠密深度/基础点、相机；原生粒子轨迹可用于单独的轨迹任务 | 当前要求稠密监督，因此不把粒子轨迹插值成像素轨迹，也不作为本轮稠密运动 GT |
| TartanAir V1 | `/data8/mugen/TartanAir_V1`；18 个 Easy 环境有深度和光流 ZIP；抽 3 条序列相邻帧 | 检查 921,600 个像素；静止 world 点投影与原生光流 EPE P95 为 1.31e-5–4.19e-5 px | 稠密几何、已确认静态场景的 world 零位移 | ZIP 内含 pose；新增 ZIP reader；使用已核验的 V1 相机与轴变换；V2 需要另查 |
| ScanNet H5 | `/data7/mugen/scannet.h5`；train 1,201；抽 3 条序列 × 3 帧 | 有效深度 83.3%–91.3%；相邻帧静态深度相对残差中位数 0.175%–0.278% | 深度/基础点、相机等几何任务 | 深度孔洞/边界掩码；不从缺失的运动标签生成动态位移或可见性标签 |
| ScanNet++ H5 | `/data7/mugen/scannetpp.h5`；train 855；抽 3 条序列 × 3 帧 | 有效深度 68.7%–98.1%；静态深度残差中位数 0.197%–0.363%；部分场景边界/遮挡尾部较大 | 几何任务 | 保留深度有效性与质量过滤；无动态三维轨迹 GT |
| WildRGBD H5 | `/data7/mugen/wildrgbd.h5`；train 包内 19,676 组；抽 3 条序列 × 3 帧 | 有效深度 69.3%–89.5%；静态深度残差中位数 0.059%–0.639% | 几何任务 | 按真实场景划分，保留孔洞/边界掩码；无动态三维轨迹 GT |
| Hypersim | `/data8/mugen/Hypersim/raw`；抽 3 个场景的 7 个存在的 GT 帧 | 官方投影的重投影 P95 为 0.162–0.764 px；world position 与 ray depth 的相对差 P95 为 0.0359%–0.0790% | 静态稠密几何；确认静态条件下可构造固定 world 点 | ZIP/RGB 读取；官方逐场景投影、资产单位转米、radial depth 转 camera-Z；特殊相机处理见下文 |
| Blended 多视图包 | `/data7/mugen/blended_mvg.h5`；train 包内 501 组；抽 3 组 × 3 帧 | 抽样几何自洽，有效深度 41.6%–95.8% | 静态多视图几何 | 不解释为有物理时间的视频，不施加动态 velocity/trajectory 标签 |
| Taskonomy H5 | `/data7/mugen/taskonomy.h5`；train 包内 537 组；抽 3 组 × 3 帧 | 有效深度 92.6%–96.2%；深度残差中位数 0.074%–0.198%；跨视图遮挡尾部明显 | 静态多视图几何 | 使用 covisibility 选择视图、保留有效性掩码；不虚构时间顺序 |
| DL3DV H5 | `/data7/mugen/dl3dv.h5`；train 包内 6,378 组；抽 3 组 × 3 帧 | 深度残差 P95 约 0.632%–0.938%，抽样自洽；本包没有标签来源/质量元数据 | 有条件的几何辅助候选 | 先追溯本地 depth/pose 来源并过滤估计标签；无原生动态运动 GT |
| CO3D H5 | `/data7/mugen/co3d.h5`；train 27,218；抽 3 组 × 3 帧 | 一组抽到了整帧深度全 0；另外两组几何一致 | 有条件的几何辅助候选 | 过滤无深度帧/低质量序列，保留原始质量与 mask；不能用来监督每像素动态轨迹 |
| MegaDepth H5 | `/data7/mugen/megadepth.h5`；train 包内 213 组；抽 3 组 × 3 帧 | 有效深度 25.3%–91.1%；可重叠区域的中位残差较小，遮挡/重建尾部较大 | SfM/MVS 估计标签的几何辅助 | 场景尺度一致化、质量过滤；照片集合没有物理时间及动态 GT |
| VKITTI 打包文件 | `/data7/mugen/vkitti.h5`；train 包内 50 组；抽 3 组 × 3 帧 | 几何可读；没有 scene flow 或持久轨迹字段；名字不能证明其版本就是 VKITTI2 | 几何辅助 | 优先使用上面的完整原始 VKITTI2，勿混淆版本 |
| Waymo H5 | `/data7/mugen/waymo.h5`；train 798；抽 3 组 × 3 帧 | 5 路相机交错；正确选择 stream 0 后有效深度只有 7.86%–8.88% | 可用于未来稀疏传感深度辅助；不纳入本轮要求稠密 GT 的首批 mixture | `frame_interleave=5`，独立流 padding、相机身份和逻辑帧；当前包无动态轨迹 GT |
| Mapillary H5 | `/data7/mugen/mapillary.h5`；train 74；抽 3 组 × 3 帧 | 首个邻帧对在两种外参约定下都没有可核验重叠，尚未验证相机与时间组织 | 暂缓 | 追溯 pack 的相机/视图/时间映射和标签来源；无重叠本身不证明 GT 损坏 |

表里的包内组数与实际 Adapter 的 train 组数不同：只有 train split 的包默认另留出 5%
验证组。例如 WildRGBD 实际 train 为 18,692，DL3DV 为 6,059，Taskonomy 为 510。
这些默认分组不能代替物理场景划分，特别是 VKITTI2 的天气/视角变体必须留在同一 split。

**确认不能直接使用的文件**

- `/data7/mugen/hm3d.h5` 无法正常打开：实际 EOF 为 7,418,710,441,960，头部期望
  7,421,354,688,488，文件缺少 2,644,246,528 字节。本次未修改或尝试修复原数据。
- `/data8/vKITTI2` 的 clone 目录缺 RGB 与内外参文件；不要使用这份不完整副本。
  `/data8/mugen/vKITTI2` 是已核验的另一份完整数据。
- 当前 `dtu.h5` 只有 test 119 组，`tnt.h5` 只有 test 21 组且缺深度，不混入训练。
- ScanNet++、Taskonomy、WildRGBD 最初的 20 秒 H5 头部读取超时，独立重试后正常；
  超时不是文件损坏证据。
- TartanAir V2 及 weidian/VGGT 生成数据本次未完成 GT 核验，不纳入首批真值监督。

**动态运动 GT 的关键区别**

VKITTI2 原生 scene flow 是两个时刻各自相机坐标中的三维位置差：

```text
SF_cam(t) = P_cam(t+1) - P_cam(t)
P_world(t+1) = c2w(t+1) · [P_cam(t) + SF_cam(t), 1]
delta_world = P_world(t+1) - c2w(t) · [P_cam(t), 1]
```

不能把它直接当 world 位移或只乘一次旋转矩阵。静态背景的原始 scene flow 可以很大，
正确消去相机运动后 world 位移接近 0。已通过光流、深度、静态背景和车辆 world/camera
中心的独立字段交叉核验。原生分量限制在 ±10 m，饱和值、天空/无效 depth 必须过滤。
原生 scene flow 不依赖目标图的可见深度，因此目标被遮挡时，不能仅因其目标深度不一致
就删除本来有效的三维位移标签。

没有把 `pose.txt` 当成已验证的整车刚体稠密配方。部分车辆像素的刚体重建与原生
scene flow 的逐帧对 P95 差异可达约 0.52 m，且每个 object 的 pose 并不覆盖整个序列。
原因尚未确定，长期每像素轨迹监督保持未准入状态。当前使用原生相邻帧 scene flow；
连续光流组合也不能恢复所有长期遮挡点的三维 GT。

DynamicReplica 可以用 `u1=u0+GT_flow(u0)`，结合目标深度和两个时刻的相机参数得到
world 端点及位移。这里只对端点有可靠目标深度的区域监督；遮挡区域不填 0。
原生 PNG 的 uint16 数值是 float16 位模式；flow 的两个分量在图像宽度方向并排存放，
mask 用 `>0`。直接按普通整数缩放解码会得到错误标签。
加入源/目标深度边界掩码后，12 对帧仍保留约 90.2%–96.4% 的整图像素，原生顶点
交叉验证的 world 端点相对误差 P95 为 0.103%–0.606%。这是稠密有效像素监督，
并未采样或补造 4,096 条查询。边界过滤阈值在脚本中声明，不代表整个数据集没有误差。
原生长期 mesh vertex 轨迹已核验，但不能扩展为所有像素的长期轨迹。

PointOdyssey 首轮严格检查使用了 0.05 px 重投影与 1e-4 旋转正交阈值，不能据此认定
这个数据集损坏。核对原始 NPZ 和官方生成代码后，K、外参、UV 都有原始半精度量化，
而 UV 在分割边界附近可能被移到邻近像素。因此保留严格诊断原始结果，并由
`pointodyssey_precision_review.json` 给出来源精度解释。本轮只建议将其稠密 depth/camera
接入几何损失，原生粒子轨迹不进入当前要求的每像素运动监督。

**Hypersim 的相机处理**

一些场景使用 tilt-shift 投影。同一个原始相机 pose 配统一对角 K，会在这次样本中产生
约 17–25 px 的系统误差。使用官方 4×4 投影后，P95 降至亚像素；不是靠拟合 GT 的
焦距或“修正”深度获得的结果。可将官方中心投影进行 RQ 分解，得到虚拟相机的 K 与
相机轴旋转，同时调整 w2c；depth 是 ray distance，需用虚拟相机射线长度转换成 camera-Z。
world position 与相机中心还需乘 `meters_per_asset_unit`。

官方 CSV 包含 482 个场景。本机相机类型清查时有 457 个场景包，其中 426 个在该
矩阵分解下兼容当前零 skew 相机族，31 个需要 skew 支持或图像/GT 联合校正。
初次目录扫描的 458 与后续 457 是各次扫描快照，不解释为全部场景标签均已核验；
实际像素核验仍只有 3 个场景、7 帧。清单及矩阵误差在 `hypersim_camera_models.json`。

**实际接入模型的规则**

单目是现有模型的 `C=1` 情况，输入保持
`images=[B,T,3,H,W]`、`camera_id=[0,...,0]`、`time_idx=[实际抽取的帧索引]`。
内外参仍只构造 GT/loss，不作为未知相机任务的网络输入。读取测试保留真实长宽比，
然后补齐 14 的 patch 倍数：PointOdyssey `540×960 → 546×966`，DynamicReplica
`720×1280 → 728×1288`，VKITTI2 `375×1242 → 378×1246`。新 GPU 训练还需预检整段
token 数与显存；若配置缩放，应等比变换 RGB、深度、K 和像素坐标。

打包 H5 已经处理成 560×560。去掉打包 padding 可以保留内容比例，但不能恢复原始
RAW 分辨率。原生数据和现成 H5 的分辨率必须分别记录，不再统一拉伸为 256 方图。

新增 source-specific dense recipe 和任务掩码后再混合：

- Kubric 多相机继续提供长期稠密 world 轨迹及遮挡后的 GT。
- VKITTI2 提供原生相邻时刻的 world 位移；DynamicReplica 提供可见相邻时刻的 world 位移。
- PointOdyssey/真实 RGB-D 数据提供全部有效深度像素的几何 GT；缺失 motion/visibility
  标签的任务 mask 为 false，不能将未知运动当成静止。
- 只有权威确认静态的数据才可构造固定 world 点与 world 零位移。静态照片集合用于
  单帧或多视角几何任务，不伪造物理时间、速度或动态标签。
- 每项损失按自身有效标签数归一化；补齐的相机、时间、图像边界不参与损失。缺任务不能
  降低另一项损失的均值。分布上保留动态监督的主体地位，再用固定验证集选择混合权重。

只删掉当前 dense loader 的 source 限制不构成正确接入；它还需要相邻目标时间选择、
world/scale 一致化、GT 能力掩码、可见性未知态、静态/动态标签及 DDP 路径验证。
本次未启动单目混合训练。

**复核命令与来源**

在仓库根目录使用 B300 现有 `.venv`，限制 CPU 线程并禁用 CUDA。下例只写新报告和
报告目录内的 Adapter 索引缓存，不写原始数据：

```bash
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$PWD"
AUDIT_SCRIPTS=multicamera4rc/audits/monocular_20261002/scripts
AUDIT_REPORT=outputs/training_preflight/monocular_gt_audit_20261001_1536
.venv/bin/python "$AUDIT_SCRIPTS/audit_tracks.py" --dataset dynamic_replica \
  --root /data5/d4rt_dataset/Dynamic_Replica --samples 6 --output "$AUDIT_REPORT/dynamic_replica.json"
.venv/bin/python "$AUDIT_SCRIPTS/audit_replica_flow.py" \
  --root /data5/d4rt_dataset/Dynamic_Replica --report-dir "$AUDIT_REPORT"
.venv/bin/python "$AUDIT_SCRIPTS/audit_tracks.py" --dataset pointodyssey \
  --root /data5/d4rt_dataset/PointOdyssey --samples 8 --output "$AUDIT_REPORT/pointodyssey.json"
.venv/bin/python "$AUDIT_SCRIPTS/audit_point_precision.py" \
  --root /data5/d4rt_dataset/PointOdyssey --report-dir "$AUDIT_REPORT"
.venv/bin/python "$AUDIT_SCRIPTS/audit_vkitti2.py" \
  --root /data8/mugen/vKITTI2 --output "$AUDIT_REPORT/vkitti2_native.json"
.venv/bin/python "$AUDIT_SCRIPTS/audit_geometry.py" --kind packed \
  --root /data7/mugen/scannet.h5 --samples 3 --output "$AUDIT_REPORT/scannet.json"
.venv/bin/python "$AUDIT_SCRIPTS/audit_geometry.py" --kind packed \
  --root /data7/mugen/waymo.h5 --samples 3 --frame-interleave 5 --camera-stream-index 0 \
  --output "$AUDIT_REPORT/waymo_stream0.json"
.venv/bin/python "$AUDIT_SCRIPTS/audit_geometry.py" --kind tartan \
  --root /data8/mugen/TartanAir_V1 --samples 3 --output "$AUDIT_REPORT/tartanair_v1.json"
.venv/bin/python "$AUDIT_SCRIPTS/audit_geometry.py" --kind hypersim \
  --root /data8/mugen/Hypersim/raw --samples 3 \
  --camera-metadata "$AUDIT_REPORT/metadata_camera_parameters.csv" \
  --output "$AUDIT_REPORT/hypersim_calibrated.json"
.venv/bin/python "$AUDIT_SCRIPTS/audit_geometry.py" --kind hypersim_camera_models \
  --root /data8/mugen/Hypersim/raw --camera-metadata "$AUDIT_REPORT/metadata_camera_parameters.csv" \
  --output "$AUDIT_REPORT/hypersim_camera_models.json"
```

Hypersim 的 CSV 使用下列官方链接；本次下载的 SHA256 与来源保存在
`metadata_camera_parameters.provenance.json`，报告内保留所用矩阵。

- [VKITTI2 官方格式、scene flow、相机和物体标签说明](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)
- [DynamicReplica 官方 PNG/flow 读取示例](https://github.com/facebookresearch/dynamic_stereo/blob/main/notebooks/Dynamic_Replica_demo.ipynb)
- [PointOdyssey 官方量化与 UV 边界处理生成代码](https://github.com/y-zheng18/point_odyssey/blob/main/utils/gen_tracking.py)
- [Hypersim 官方 tilt-shift 与投影说明](https://github.com/apple-aiml-research/ml-hypersim/tree/main/contrib/mikeroberts3000)
- [Hypersim 官方相机 CSV](https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv)
- [ScanNet 官方 RGB-D、相机与场景格式](https://github.com/ScanNet/ScanNet)
- [MegaDepth 官方 SfM/MVS 标签来源](https://www.cs.cornell.edu/projects/megadepth/)
- [CO3D 官方质量 mask 与序列格式](https://github.com/facebookresearch/co3d)
