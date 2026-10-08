# 数据、GT 与采样

[返回 README](../README.md) · 实现：[datasets/](../multicamera4rc/datasets/)

## 数据流与坐标合同

```text
原生文件 / 已打包 H5
  → registry 选择 adapter
  → UnifiedClip（RGB、原相机/帧号、K、world-to-camera、原生标签）
  → bridge.prepare_clip（整理观察槽、图像变换、参考坐标、GT mask）
  → collate_adapter_samples
  → 选择 Q 张源图
  → materialize_dense_targets（在训练设备构造完整有效监督）
  → model(RGB, camera_id, time_idx) + criterion(prediction, targets)
```

adapter 内外参约定由 metadata 显式声明，通常是 world-to-camera；Syn4D 显式提供 OpenCV camera-to-world。bridge 统一转为共同参考坐标下的 `c2w`。不要将数据字段名 `extrinsics` 直接当作 model 的 `c2w`。

第一张有效观察定义 GT 参考坐标。稠密训练按整个片段有效基础点到原点的平均欧氏距离取一个共同尺度：depth、相机/物体平移、基础点、世界轨迹和位移统一除以该尺度，K 和单位方向不缩放。ray GT 的六维顺序是 `[world_direction3, world_origin3]`。归一化位置/位移 EPE 不是直接的米制误差。

相机内外参、depth、粒子轨迹和 flow 只构造 `targets`，不能拼进网络输入以代替未知相机估计。`observation_valid`、像素有效性、三维轨迹有效性、可见性标签及其有效性是不同概念。

## 原分辨率和长宽比

`image_mode: native` 不指定固定 `image_size`，保留所读文件的所有原像素，只在右侧/底部补齐 14 的倍数，且最小边至少 28。此时 K 不做尺寸缩放，padding 不参与监督。其他显式模式：`fit` 按比例缩放后补边，`pad` 放入固定画布，`resize` 允许拉伸到指定尺寸。

| 原始尺寸 H×W | native 网络输入 |
|---|---|
| Kubric 256×256 | 266×266 |
| DynamicReplica 720×1280 | 728×1288 |
| PointOdyssey 540×960 | 546×966 |
| vKITTI2 375×1242 | 378×1246 |
| TartanAir V1 480×640 | 490×644 |
| TartanAir V2 640×640 | 644×644 |
| Syn4D V1 720×1280 | 728×1288 |
| Syn4D Sim 1024×1024 | 1036×1036 |

Kubric 当前生成器本来就渲染 256×256；不是本 bridge 将大图压到 256。对于已打包 H5，native 指 H5 实际存储的像素；若上游预处理已经缩放，当前代码无法恢复缺失细节。

## 已准入训练数据

下表是 B300zbf 上已核验并加入全量混合池的数据。百分比对应 **25k→50k 的 balanced_coverage 课程**，不代表更早 coverage 训练的采样比例。注册表中还有其他 adapter，存在 adapter 不等于该数据已通过 GT 核验并进入训练。

| 数据集族 | 比例 | 服务器路径 | 输入类型 | 实际监督 |
|---|---:|---|---|---|
| Kubric | 35% | `/data8/kubric_multicam_tracking_v3` | C1–C10，联合采样 T | 稠密几何、刚体三维长轨迹、位移、可见性 |
| DynamicReplica | 12% | `/data5/d4rt_dataset/Dynamic_Replica` | 左/右分别 C1，T≤16 | 全像素几何；有原生 flow 的窗口监督邻帧三维运动 |
| PointOdyssey | 12% | `/data5/d4rt_dataset/PointOdyssey` | C1，T≤24 | 全像素几何 + 全部有效原生粒子长轨迹/可见性 |
| vKITTI2 | 11% | `/data8/mugen/vKITTI2` | C1 或同刻 C2；单目 T≤24，双目 T≤12 | 稠密几何、原生邻帧 scene flow |
| TartanAir V1 | 4% | `/data8/mugen/TartanAir_V1` | C1 或左右 C2 | 几何；不将相机运动造成的 flow 当成物体运动 |
| TartanAir V2 | 7% | `/data8/mugen/TartanAir_V2_subset` | 六个同朝向左右 C2 / 左前 C1 | 几何；pinhole 原生 depth/pose |
| Hypersim | 4% | `/data8/mugen/Hypersim/raw` | 静态 C2/C4,T1 | 稠密多视角几何 |
| ScanNet | 3% | `/data7/mugen/scannet.h5` | C1，逐步加长 T | 几何 |
| ScanNet++ | 3% | `/data7/mugen/scannetpp.h5` | C1，逐步加长 T | 几何 |
| WildRGBD | 3% | `/data7/mugen/wildrgbd.h5` | C1，逐步加长 T | 几何；不臆造动态世界轨迹 |
| BlendedMVS | 4% | `/data7/mugen/blended_mvg.h5` | 静态 C4/C8,T1 | 共视邻域多视角几何 |
| Taskonomy | 2% | `/data7/mugen/taskonomy.h5` | 静态 C4/C8,T1 | 共视邻域多视角几何 |

静态图像按不同视图赋予相机身份，统一 `time_idx=0`。候选视图不足时按实际有效数量降级，不重复图像凑 C。当前课程未启用 C48T1，扩展前需要原生共视、有效 GT 与显存验证。

vKITTI2/V1 族内比例为双目 60%、左单目 20%、右单目 20%；V2 为前/后/左/右/上/下六个同方向左右双目各 15%，左前单目 10%。V2 只有一部分环境含完整双目方向包，缺少右相机的合格前左包仍可单目使用。同一个环境在所有 V2 子源中保持一致的训练/验证划分。

动态数据集族合计 70% 是数据源概率，不保证每一批次有动态前景或完整运动标签。DynamicReplica 当前训练没有组成左右同刻 C2，其缺 flow 的右流保留几何监督。

## 各种运动 GT 的边界

### Syn4D：网格表面对应的稠密长轨迹

新增原生 `syn4d` adapter 支持同步多相机、可变 C/T 和 stride-1/stride-5 记录。根据像素对应的 face ID、重心坐标和逐帧顶点动画恢复轨迹，覆盖非刚性形变；未用 Kubric 的刚体假设替代。源/目标之间的原始帧间隔保留，RGB 维持原分辨率。缺少人体/衣服元数据时屏蔽对应区域的未知运动，保留深度。新配置尚未改变上表正式训练配比；目录、GT 合同、官方对照和混合命令见 [Syn4D 接入](syn4d.md)。

### Kubric：选中源图的全像素长轨迹

深度反投影得到 `P_src`，instance ID 对应原生物体源/目标姿态：

```text
local = R_src^T * (P_src - t_src)
P_t   = R_t * local + t_t
Δ_t   = P_t - P_src
```

可信静态地面保留固定世界点；可见性由目标分割、深度和投影一致性决定。遮挡/离开目标图像不会使已知的三维轨迹失效。数据准入同时使用 `/data8/kubric_multicam_tracking_v3_state/validated` 的校验 stamp。训练池 4,752 场景，验证池 248 场景；每场景 10 台相机，24 或 48 个原生时刻。

`dense_target_version=2` 排除 HDRI 穹顶的代理几何。此生成器的背景标签 0 同时包含地面和约 40 米半径的贴图穹顶，不能把全部背景删除，也不能把穹顶深度当作照片内建筑的真实形状。数据配置顶层 `kubric_background_policy: auto` 为默认：仅对同时声明 `movi_tracking_v3` 和 `opencv_camera__blender_world` 的已审计 schema 启用过滤，保留全部 ID>0 物体，以及 ID=0 且距离世界地面 z=0 不超过 0.05 米的像素。地面方程先变换到参考相机坐标，再在米制、归一化之前判断；远处/飞行物体不会因距离而被删除。RGB 保持完整。

同一过滤掩码控制深度、ray、三维轨迹、位移、速度、静态约束、可见性监督、深度梯度边以及场景尺度。日志记录 `excluded_background_pixels/fraction`。未知 schema 在 `auto` 下保留原掩码；显式 `objects_ground` 遇到未知 schema 会报错。`all` 仅用于重现历史结果。checkpoint 和验证报告记录版本与策略，旧/新权重必须用同一新掩码和尺度重评，不能直接比较旧报告里的 loss/EPE。忽略 GT 区域不等于强制清除网络输出，实际推理不能用 GT 分割替代预测有效性。

### vKITTI2 / DynamicReplica：部分稠密时间对

vKITTI2 的原生 scene flow 表达涉及相机坐标端点。代码先恢复目标相机三维点，再用目标相机外参转换到共同世界坐标；不能只将 scene flow 向量乘源相机旋转。匹配同时检查相机 ID 和真实帧号，避免左右相机相同帧号冲突。

DynamicReplica 使用原生邻帧 optical flow、深度和相机位姿构造可验证的三维端点。没有原生 flow 的流/窗口只用几何。两者增加上下文 T 不会自动产生长期运动标签：当前运动配置固定 stride1，非邻接未知时间对由 mask 排除，不插值伪造长轨迹。

部分 DynamicReplica 文件在官方有效性 mask 全零时保存为 8 位全零占位 PNG。加载器先验证正反向 mask；任何一向完全无效，该时间对都不能通过双向一致性检查，因此保留几何、跳过这对运动标签。两向 mask 均非空时，仍严格要求原生 16 位光流及正确尺寸，不将损坏文件解释为静止 GT。

### PointOdyssey：原生三维粒子与可见深度恢复

PointOdyssey 是动态追踪数据集。已准入 131 段序列，其中 109 段有可用原生三维轨迹；另 22 段当前文件中的 `trajs_3d` 是标量 NaN。后者只在原生 2D 轨迹可见、深度有效且局部可靠的位置，通过深度、K、位姿恢复 XYZ；不为遮挡点生成未知三维 GT。

原生 XYZ、像素投影、原始深度编码、坐标轴和相机矩阵已联合校验。具有真正 XYZ 的遮挡轨迹可以保留三维监督；可见 XYZ 与原生 UV/depth 不一致时过滤源点和目标点。三维有效性与可见性标签独立。

每步选定 Q 个源观察后，使用这些源中的全部有效原生粒子，无 4096 上限，也没有稀疏粒子插值成伪稠密运动图。日志字段 `sparse_*` 表示粒子标签形式，不代表随机限额下采样。`native_tracks` 记录 GT 来源和筛选数量。

### TartanAir / 几何数据

V1/V2 位姿执行 NED→OpenCV 转换；V2 从原始四字节 PNG 还原 float32 深度，不能使用 RGB 转换后的字节重解释。采用有效深度上限 200 和 p98 尾部筛选，筛选只影响监督 mask，不裁剪真实 RGB。GT 证据见[双目/动态核验](../multicamera4rc/audits/native_motion_stereo_20261002/README.md)。

缺少可靠物体世界运动 GT 时不监督“零运动”。几何快速路径只计算所有观察的几何分支，避免执行没有标签约束的 motion head。

## C/T 课程与覆盖

每 epoch 100,000 个采样位置，通过最大余数法给各源分配比例配额，再由 DDP 打乱。池内采用常数空间仿射置换轮换原生窗口；可访问全部准入池，但有限步数不保证每个窗口已看过一遍。旧 `coverage` 会受原生窗口池大小影响；`balanced_coverage` 将池大小和源比例分开，避免 Taskonomy 因静态图池很大而占据约三分之一训练。

| 总步数阶段 | 通用单目 T 的概率 | 双目 T 的概率 |
|---|---|---|
| 25k→32k | 4/8/12/16：10/45/30/15% | 4/6/8：35/50/15% |
| 32k→40k | 8/12/16/24：25/25/30/20% | 4/6/8/12：20/35/30/15% |
| 40k→50k | 8/12/16/24/32：10/15/25/30/20% | 4/6/8/12：15/20/35/30% |

DynamicReplica 上限 T16、PointOdyssey/vKITTI2 单目上限 T24；超出概率合并到上限。短序列仍留在池中，按实际合法长度采样，不重复最后一帧。

Kubric C1–C10 使用联合 C/T 布局，首阶段最大 40 张图，后两阶段最大 48 张图。大 C 适当降低 T；C5–C10 保留 T2/T4，后期 C5–C8 可 T6。T32 只对足够长的原生场景生效。完整分布来自 [build_variable_curriculum.py](../scripts/build_variable_curriculum.py) 和[实测阶段配置](../multicamera4rc/audits/variable_ct_20261003/measurements/)。

每卡 BS1 支持不同时刻数、相机数与分辨率；不同 DDP rank 可采用不同布局。同一个 local batch 中仍需要共同 H/W 和 Tq，collator 只补观察、相机和粒子等支持的轴，不自动解决任意时间/图像大小的混批。BS>1 需要分桶。

## 配置、manifest 和扩展

原生数据 YAML、模型 config 和训练运行参数各自独立：YAML 控制数据布局/GT，`MultiCamera4RCConfig` 控制架构，CLI 控制训练。数据 YAML 的相对路径相对于 YAML 所在目录解析。

现成示例：[固定布局 Kubric](../multicamera4rc/configs/kubric_multicam_native.yaml)、[可变布局 Kubric](../multicamera4rc/configs/kubric_variable_native.yaml)。`supervision: dense` 配合 `num_queries: 0`，不会启用旧 point-query 配额。

自备推理图像也可使用 JSON manifest：

```json
{
  "clips": [{
    "id": "clip_001",
    "observations": [
      {"image": "cam0/000.png", "camera_id": 0, "time_idx": 0},
      {"image": "cam0/005.png", "camera_id": 0, "time_idx": 5},
      {"image": "cam1/000.png", "camera_id": 1, "time_idx": 0},
      {"image": "cam1/005.png", "camera_id": 1, "time_idx": 5}
    ],
    "query_time_idx": [0, 5]
  }]
}
```

manifest 路径相对于该 JSON，图片需提前准备成共同 patch14 画布。manifest loader 不自动补边；`infer --image-size` 是显式矩形 resize 兼容入口，保长宽比输入应先自行按原像素补边或使用 native adapter。用于训练时可增加 `targets` 指向 `.pt/.npz` 字典，必须准备一致坐标和像素变换；不能单独 resize RGB 后沿用旧标签。

接入新数据时，应实现 `BaseAdapter` 的索引、序列长度、`load_clip` 接口，返回 `UnifiedClip` 并注册懒加载名称。先以独立 GT 检查确定深度语义（Z/range）、内外参方向、单位/坐标轴、时间标签、同步相机身份、原始像素重投影以及动态端点；之后再选正确的 `dense_kind`。掩码必须区分未知轨迹与已知不可见轨迹。真实文件代表窗口审计和小模型反传通过后，再纳入已准入 manifest；单有 adapter 单测不能证明整个数据集 GT 正确。
