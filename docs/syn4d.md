# Syn4D 数据接入

[返回数据文档](data.md) · [训练配置](../multicamera4rc/configs/syn4d_dense.yaml) · [固定验证配置](../multicamera4rc/configs/syn4d_eval.yaml)

本实现直接读取 Syn4D 发布的 EXR、MP4/PNG、safetensors 和网格元数据，接入当前 `multicamera4rc` 混合训练框架。模型结构无需改变。它不依赖旧 D4RT 数据管线，也不把粒子轨迹插值为稠密标签。

## 官方依据

截至 2026-10-06，官方 GitHub README 仍将训练 dataloader 列为待发布内容；已经公开的可运行参考是可视化数据读取器。本适配器依据下列文件重新实现，实测逐像素与官方读取器比较，而非假定存在另一份官方训练 loader：

- [项目介绍](https://jzr99.github.io/Syn4D/)、[官方仓库](https://github.com/jzr99/Syn4D)。
- [数据目录规范](https://huggingface.co/datasets/Syn4D/Syn4D/blob/main/data/README.md)。
- [官方 syn4d_track.py](https://huggingface.co/datasets/Syn4D/Syn4D/blob/main/code/visualizer/syn4d_track.py)：相机欧拉角、厘米单位、网格动画、compact face ID、重心坐标及法向偏移恢复。
- [BEDLAM2 元数据准备](https://huggingface.co/datasets/Syn4D/Syn4D/blob/main/code/bedlam2/README.md)。人体资源需要按官方流程自行准备；数据卡明确没有随 Syn4D 重分发处理后的 BEDLAM2 元数据。

官方文件的 SHA256 和实测结果记录在 [Syn4D 审计](../multicamera4rc/audits/syn4d_20261006/README.md)。数据和上游资产的许可、归属以下载根目录的 `LICENSE`、`THIRD_PARTY_NOTICES.md` 和 `LICENSES/` 为准。

## 准备下载文件

服务器根目录是 `/data6/Syn4D`。支持完整 Hugging Face 根目录、`data` 目录或某个 subset 目录：

```text
/data6/Syn4D/
  data/
    metadata/new_weight_bone/<group>/<object>/vertices_sequence.npz
    metadata/bedlam2_smpl_npz/<subject>/<subject>_<animation>.npz  # 可选人体资源
    metadata/b2_assetdata_download/clothing/npz/                 # 可选衣服资源
    syn4d_v1_stride_5/
      <scene>.tar.zst
      mp4/<scene>/seq_000000_0.mp4
      <scene>/
        be_seq.csv
        be_seq_transform_channels.json
        ground_truth/meta_exr_csv/seq_000000_0_camera.csv
        exr_layers/depth/seq_000000_0/*_depth.exr
        tracking_safetensors/seq_000000_0{.safetensors,_index.json}
        env_mask_safetensors/seq_000000_0{.safetensors,_index.json}
        mp4/seq_000000_0.mp4                                   # 也支持这个视频位置
    syn4d_sim/<scene>/...
```

先安装项目数据依赖。解压命令额外需要系统可用的 `zstd`：

```bash
pip install -e '.[data]'
python -m scripts.prepare_syn4d \
  /data6/Syn4D/data/syn4d_v1_stride_5/bigoffice_v1.tar.zst \
  /data6/Syn4D/data/syn4d_sim/cave_sim.tar.zst
```

解压器检查路径、文件长度、压缩流完整性及归档是否在解压期间变化；完成后原子发布场景目录。已有场景保持原样。不要把 Hugging Face `.incomplete` 文件当作归档传入。DataLoader 不执行解压，也不在每个 batch 中重新扫描归档。

索引仅纳入已解压且 RGB、深度、相机 CSV 完整的相机流；视频帧数必须与 CSV 一致。未解压归档和只有 MP4 预览的场景不进入训练池，拒绝原因保留在 `adapter.rejected`。补下更多场景后重新解压并重建 dataset；现有 worker 不热更新采样池。

## C、T、坐标和输入维度

同一 `<scene>/seq_XXXXXX` 的相机流组成同步片段，不把不同序列拼成多相机。同一次采样选定 C 台相机，随后在每台相机取同一组 T 个真实时刻；不重复帧补足时长。还会核对各相机的物体、动画和世界变换元数据一致。

| 阶段 | 张量与语义 |
|---|---|
| 原始片段 | C×T 张原生 RGB、camera-Z depth、K 和 OpenCV C2W |
| 网络输入 | `images[B,S,3,Hpad,Wpad]`，S=C×T；`camera_id[B,S]`、`time_idx[B,S]`、`query_time_idx[B,T]` |
| 紧凑 GT | 每观察的 `face_map[S,Hpad,Wpad]`、环境 mask；`vertices[T,V,3]` 和 `faces[F,3]` |
| 选 Q 张源图后 | `trajectory/displacement[B,Q,T,Hpad,Wpad,3]`，对应逐像素有效 mask |
| 可见性 | `[B,Q,T,C,Hpad,Wpad]`，可见与标签已知是两个不同 mask |

V1 实测原图 H×W 为 720×1280，网络补边至 728×1288；Sim 为 1024×1024，补边至 1036×1036。`image_mode: native` 不缩小图像、不拉伸、不裁掉像素，K 保留原图标定。

默认 `frame_policy: reference` 依据实际 tracking index 选择所有完整相机共同具备记录的参考时刻。例如 `stride_5` 的采样位置 `[0,1,2]` 映射成真实 `time_idx=[0,5,10]`，不会误写为 `[0,1,2]`。配置 `strides` 的单位是这个可采样帧列表的位置；传入模型的时间始终保留原始间隔。T16 的 stride-5 片段通常覆盖 75 个原始帧间隔。

也支持 `frame_policy: all`，将所有 RGB 时刻用作输入/目标时间。没有 compact 记录的源帧不会生成前景运动标签；`--prefer-motion-sources` 优先选择存在网格对应记录的源观察。一个有记录的源可以恢复到任意已选目标时刻，不要求目标也是 stride-5 参考帧。

官方相机处在 Unreal 左手世界中。本实现对相机和所有世界顶点同时施加 `diag(1,-1,1)`，得到行列式为 +1 的旋转；图像和投影保持一致。EXR 的 `Depth` 通道和 CSV 相机平移从厘米除以 100 转成米，K 使用官方 `cx=W/2, cy=H/2` 的整数像素网格约定。bridge 再转到第一张观察的共同参考坐标。K、pose、网格、深度只用于 GT，不输入网络。

## 稠密运动及无效区域

对每个已知源像素，使用官方 face ID 定位三角形，从源深度恢复重心坐标和有符号法向距离：

```text
P_source = sum(barycentric[k] * triangle_source[k]) + distance * normal_source
P_target = sum(barycentric[k] * triangle_target[k]) + distance * normal_target
displacement = P_target - P_source
```

顶点来自真实动画和 `PerFrameMotion`，支持非刚性形变、刚体平移/旋转及尺度变化。Objaverse 动画遵循官方从顶点帧 1 开始循环的规则；BEDLAM 遵循 CSV `start_frame`。Sim 中 `static_mesh`/`is_dynamic=false` 不等于世界中静止，仍使用其逐帧变换。

监督规则：

- 深度使用全部有效像素，按官方过滤非有限、非正及超过 300 米的深度。
- 运动使用全部有效 compact 像素及明确标记的静态环境；没有 4,096 点上限。网格退化、无效 face ID、源点距网格表面超过官方 0.1 米阈值时，运动未知。
- `env` mask 中明确属于静态环境的点保持固定世界位置。不会将“没有网格对应”自动解释成环境或零位移。
- 人体/物体网格缺失时，`missing_mesh: mask` 保留其深度、自时刻基础点监督，屏蔽未知跨时刻轨迹、位移、速度和静态约束。日志 `rank_batches[].syn4d.missing_meshes` 给出缺失路径；`missing_mesh: error` 可以改为严格拒绝。
- 可见性由目标深度和投影一致性推导，标注为 `depth_consistency`，不是冒充官方独立可见性 GT。已知遮挡和图像外轨迹继续监督 XYZ；目标深度缺失或几何冲突时不强行监督可见/不可见。
- 场景尺度仍按全部有效基础点平均距离计算；depth、相机平移、轨迹和位移共享同一尺度。单独的 Kubric 穹顶规则不应用到 Syn4D。
- 可见性按每张图实际内容范围及半像素边界判断，补边区域属于出画面，图内深度空洞保持未知。物理投影、远景深度与加载器复查见 [2026-10-08 GT 独立审计](../multicamera4rc/audits/gt_integrity_20261008/README.md)。

2026-10-06 首次接入时缺少 BEDLAM 人体及衣服网格，只启用已有物体/环境的运动。2026-10-08 的完整数据配置使用补齐的 560 份人体网格和 560 份衣服网格，V1 设置 `require_human_meshes: true`：如果人体或衣服资源缺失，加载器明确报错，避免完整训练悄悄退回部分运动监督。衣服资源可以放在 `archive_map.json` 指向的本地 tar 容器中，无需全部展开为独立 NPZ。8 个上游未发布的物体网格仍单独记录并屏蔽未知运动。

## 接入现有混合训练

独立配置 `multicamera4rc/configs/syn4d_dense.yaml` 提供 V1 的 C1T16/C2T8/C4T4/C8T2 和 Sim 的 C1T16/C2T8/C4T4。大 C 降低 T，当前示例每个片段最多 16 张图。实际可选布局由该序列已下载的相机数和帧数决定。

生成包含全部旧数据源的新配置：

```bash
python -m scripts.build_syn4d_mixture \
  --base /data8/zbf/multicamera4rc_runs/extended_100k_20261005_165948UTC/motion_replay.yaml \
  --root /data6/Syn4D \
  --v1-subset syn4d_v1_stride_1 --v1-strides 1 2 5 \
  --fraction 0.10 --v1-fraction 0.50 \
  --output /path/to/new_full_mixture_syn4d.yaml --check
```

这是待采用的比例示例，不是已验证的最优训练比例：旧数据总占 90%，旧源之间相对比例保留，V1 和 Sim 各 5%。脚本检查所有源可索引后输出新 YAML，不覆盖旧训练配置、不启动训练。更多资源下载完成后，重新生成配置和准入审计。

`--v1-subset` 必须选择一个 V1 发布版本。stride-1 与 stride-5 中的同一场景/序列不能当成两份不同数据重复加权。stride-1 配合 `[1,2,5]` 采样间隔，在保留密集参考时刻的同时覆盖较长运动；参考时刻连续时，T16 可覆盖 75 个原始帧间隔，若有缺口则仍按真实帧号编码。Sim 保留逐帧采样。

2026-10-06 的预备配置位于下方；它基于当时的旧采样比例，后续恢复训练应从**当前正在使用的配置**生成新混合配置：

```text
/data8/zbf/multicamera4rc_runs/syn4d_integration_20261006/full_mixture_syn4d.yaml
```

混合训练继续使用每卡 BS1，不同 rank 可采不同数据集、C/T 和分辨率；本地 BS>1 要对任务字段、T 和 H/W 分桶。同一批内的 Syn4D 样本可以补齐相机/顶点/面数量，不能将不同 GT recipe 或不同 H/W 任意堆叠。

从旧 checkpoint 切换到新的混合池时，应显式使用 `--reset-data-cursor` 重新开始新配额的采样，同时保留模型和优化器；使用 `--prefer-motion-sources --no-query-chunking`。正式大模型的显存和速度需要针对选定 C/T 单独预检；此次真实混合反向测试使用 tiny 配置验证完整训练链路。

### 原分辨率巨型模型准入与后续扩池

`scripts/preflight_syn4d.py` 从正式 checkpoint 读取模型及完整 loss 配置，逐一运行新源的所有 C/T 布局，检查有限损失、梯度、动态监督及峰值显存。`--layouts 2x2` 仅是可以与旧训练共用 GPU 的初步测试，不代替全部生产布局的验证。

`scripts/audit_syn4d_pool.py` 检查全部准入序列的跨相机元数据、首尾动画顶点和资源路径，并为每个场景选一个原生多目片段，与发布的读取器独立比较 RGB、深度、标定及轨迹。它记录每个 split 的数量和缺失人体资源；这不是对所有帧逐像素的穷举审计。

默认情况下，任何 `adapter.rejected` 都会让全池审计失败。若逐项核实后决定保留这些排除，可传入 `--reviewed-exclusions <manifest.json>`：清单必须逐一匹配源名、根目录相对路径和原因，并写明核实结果。新增、原因变化或已经消失的排除都会再次阻止准入。该选项不修补标签、不把被拒相机加入训练，也不放宽官方逐像素对照的阈值。2026-10-07 的[明确排除清单](../multicamera4rc/audits/syn4d_20261007/reviewed_exclusions.json)记录了 hallway 的 6 路缺少 tracking 文件的相机，以及 3 条没有全相机共同参考时刻的序列；后者属于当前采样策略限制，不能据此认定标签错误。

2026-10-08 的完整配置将 V1 改为 `frame_policy: all`，恢复这 3 条序列以及其他序列被全相机参考帧交集排除的时刻。源像素仍必须有有效的 compact 对应才能获得前景运动监督；目标时刻只需有对应动画帧。`--prefer-motion-sources` 优先选择有运动对应的源图。原始帧号、同步相机关系和 train/val/test 划分保持各自明确的语义，禁止重复帧填补时长。

`scripts/audit_syn4d_complete.py` 使用独立 CPU 进程检查全部场景：逐序列核对跨相机几何及首尾动画范围，逐场景与官方读取器比较原像素 GT，额外检查缺物体网格和恢复采样的序列，并统计实际人体/衣服运动监督。相机平移比较允许厘米转换到米时最多两个 float32 ULP 的舍入差异；旋转与轨迹误差检查保持各自原有阈值。几百米坐标中的单个 ULP 不应被当成标定错误。

同一清单的 `missing_objects` 单独记录已核实缺失的物体网格，不能代替相机排除项。全池检查发现 8 个物体网格在本地和固定版本的官方仓库中均缺失，影响 8 条 V1 序列；[逐文件发布状态](../multicamera4rc/audits/syn4d_20261007/upstream_missing_objects.json)保存了版本号和 HTTP `404 EntryNotFound`。这些区域使用现有 `missing_mesh: mask` 规则：深度和自时刻几何保留，未知跨时刻运动及静态约束不产生梯度。其他物体和环境继续参与监督；未列明的新缺失物体仍阻止全池准入。

`scripts/continue_syn4d_training.py` 接管明确记录的 GPU4/5 训练进程，从完整 checkpoint 恢复模型、Adam、学习率计划和双卡 RNG。正式接入前还会运行一个双卡训练测试周期，覆盖 V1 四种和 Sim 三种布局。如果准入失败，保留错误记录并恢复旧混合数据训练。正式训练先运行 500 步，再恢复每 2k 步的固定验证和有界质量回退；目标维持 100k。Syn4D 增加独立固定验证，旧域质量指标继续采用原来的相同验证计划。

下载后的压缩包需要解包。2026-10-07 初始配置冻结了已校验场景；当时的后台扩池仅允许扩展场景列表，权重、C/T、GT 规则和其他源保持同一训练契约。2026-10-08 的完整配置已经包括全部 29 个 V1 和 8 个 Sim 场景，以及补齐的人体/衣服资源，并通过新的完整数据审计。该配置直接从 55k 检查点进行全池准入，不依赖旧扩池任务；恢复模型、Adam 和学习率计划，仅重置新池采样游标。运行中的 DataLoader 不热修改索引。

首次接入的历史配置和审计输出目录：

```text
/data8/zbf/multicamera4rc_runs/syn4d_activation_20261007/
  initial_mixture.json          # 首批已校验场景；所有旧源合计 90%，V1/Sim 各 5%
  initial_pool_audit.json       # 首批序列元数据及逐场景官方比较
  full_mixture.json             # 29+8 个场景的明确列表
  full_pool_admission.json      # passed=true 后才允许扩入全量场景
  giant_layouts.json            # 正式布局的巨型模型反向传播验证
 ddp_proof.json                # 双卡模型、优化器恢复及布局覆盖验证
```

完整人体/衣服 GT 配置和审计输出目录：

```text
/data8/zbf/multicamera4rc_runs/syn4d_complete_20261008/
  full_mixture.json            # 全部场景；旧源 90%，V1/Sim 各 5%
  human_assets_proof.json       # 560 份人体和 560 份衣服的完成凭据
  complete_gt_audit.json        # 821 条序列元数据、47 个片段官方逐像素比较
  mixture_index.json            # 全部 25 个混合数据源及 train/val 隔离
  giant_layouts.json            # V1 四种、Sim 三种布局的原分辨率反向传播
  ddp_proof.json                # 双卡实际参数更新与全部布局覆盖
```

可审阅的数据证据和恢复记录见 [2026-10-08 完整池审计](../multicamera4rc/audits/syn4d_20261008/README.md)。

这些文件的状态以服务器实际输出为准；准备好的配置不等于已启用，正式训练日志中的 `rank_batches[].dataset` 与控制器的 `syn4d_admitted` / `full_pool_admitted` 分别确认初始接入和全量扩入。

## 验证与实现位置

```bash
# 原像素 RGB、标定、米制轨迹与官方逐像素比较；包含单目、多目和长序列。
python -m scripts.audit_syn4d --root /data6/Syn4D \
  --output outputs/syn4d_audit/official_comparison.json

# 固定验证配置；eval 会独立选择 val，写出并复用同一验证计划。
python -m multicamera4rc.evaluate \
  --data-config multicamera4rc/configs/syn4d_eval.yaml \
  --checkpoints candidate=/path/to/checkpoint.pt \
  --plan outputs/syn4d_val/plan.json --output outputs/syn4d_val
```

划分是本项目确定性的 sequence 级 80/10/10 hash 划分，**不是官方 benchmark split**；相机后缀和 stride subset 不参与 split hash，相同物理序列不会因相机或 stride 版本不同跨 train/val/test。它不保证不同序列使用互不重叠的场景或物体资产；资产外泛化评估需要另建场景/资产隔离划分。`split: all` 仅用于显式审计。少量下载可能暂缺某 subset 的验证序列，加载器会明确报错而非拿训练数据充当验证。

| 代码 | 职责 |
|---|---|
| [adapters/syn4d.py](../multicamera4rc/datasets/adapters/syn4d.py) | 增量下载准入、同步 C/T、媒体读取、资源缺失策略 |
| [adapters/syn4d_geometry.py](../multicamera4rc/datasets/adapters/syn4d_geometry.py) | 官方坐标、深度单位、物体/人体顶点动画 |
| [syn4d_dense.py](../multicamera4rc/datasets/syn4d_dense.py) | 紧凑表面对应、逐像素轨迹和可见性 |
| [test_syn4d.py](../tests/multicamera4rc/test_syn4d.py) | 解析几何、缺失 GT、梯度屏蔽、混合采样与训练测试 |

测试依赖包含 `pytest` 以及 `.[data]`；独立官方对照脚本还需官方代码使用的 `pandas` 和 `Imath`。训练 adapter 本身不依赖 pandas、torchvision、Viser 或官方可视化进程。
