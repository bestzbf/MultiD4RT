# 静态多视角混合训练核验（2026-10-02）

现有全量混合池中的BlendedMVS、Taskonomy和Hypersim从单视角T=1改为多视角T=1。
这里的C=4/8表示随机选择4或8个逻辑视点，不是48相机。原有15个source的顺序、
权重、准入物理场景、有效帧和coverage索引全部保持一致。

| 来源 | 路径 | 输入布局 | 准入序列/帧 |
|---|---|---|---|
| BlendedMVS pack | `/data7/mugen/blended_mvg.h5` | C=4/8，T=1 | 475 / 109,223 |
| Taskonomy pack | `/data7/mugen/taskonomy.h5` | C=4/8，T=1 | 510 / 4,456,268 |
| Hypersim | `/data8/mugen/Hypersim/raw` | C=2/4，T=1 | 614 / 59,492 |

Hypersim保持原有较大图像尺寸，采用2/4视角降低每个样本的计算和显存开销。
其它数据布局保持原配置，包括Kubric C2/T12、C3/T8、C4/T6和动态单相机数据。
这次增加多视角几何训练，不补造动态scene flow或三维追踪GT。

**输入和GT规则**

网络输入为`images=[B,C,3,H,W]`、`camera_id=[0,...,C-1]`、`time_idx=[0,...,0]`、
`query_time_idx=[0]`。camera ID表示样本内的逻辑视点；静态扫描的移动相机照片
不被宣称为不同物理传感器的同步视频。原始图像序号保留在`view_indices`日志中，
不作为这些图片的时间输入。K/E用于离线选择视角及GT/loss，未作为网络输入。

保留coverage anchor，从同一场景的packed `pairs`中选邻居，再以原始K/E和camera-Z depth
检查共视：stride=16的诊断网格至少有3% anchor有效点在目标图上满足5%相对深度一致性。
这只是选视角的诊断，不限制训练像素数量。优先有平移基线的视角，再按匹配支持排序。
Hypersim根据朝向、距离和anchor有效表面投影的视锥重叠选视角；不跨物理场景配对。
所有选择都尊重原manifest有效帧。邻居不足时保持较小真实C，孤立视角为C=1，不复制图片凑数。

RGB/depth/K/E原值保持不变；原始H5的上游缩放不能由加载器恢复。按既有规则去打包padding，
再右侧/底部补到14的倍数。Hypersim既有投影/单位/skew联合处理保持不变。
GT仍在第一观察的共同坐标系中，按整个clip统一尺度归一化。depth/camera监督覆盖全部
输入观察的有效像素，基础点监督覆盖选定decoder source的全部有效像素，source-count仍为4。
T=1只有本时刻的基础点标签；未知运动、静态位移和可见性均不填造标签。

**完整池与真实窗口检查**

- [配置及索引一致性](measurements/static_data_plan.json)：coverage仍为5,961,438个窗口，
  epoch仍为11,922,876；比较原/新source的pool、有效帧、权重和coverage映射，允许保留断点游标。
- [全序列邻居检查](measurements/all_static_sequence_neighborhoods.json)：覆盖1,599个序列的
  一个有效中间帧。BlendedMVS有474个支持C4、473个支持C8；Taskonomy为509/507。
  Hypersim有610个支持C4，其中两个检查anchor孤立。不是每个anchor都能达到名义C。
- [36个真实窗口的GT检查](measurements/native_static_gt_audit.json)：三个family分别覆盖
  3个序列、首/末anchor和两个布局；确认distinct view、相机/时间标签、原RGB/depth/K/E
  一致、完整clip的平均基础点尺度为1、任务掩码正确，并独立检查跨视角深度一致性。
- 全序列邻居检查不是对几百万帧所有RGB/depth像素的离线认证。像素掩码和Hypersim
  原始world-position/range交叉检查仍在每次读取时执行；无有效depth的窗口按既有日志机制排除。

**整段相机attention与giant预检**

原来的学习偏置`alpha[h]*same_camera`可以写成
`Q'=[Q,sqrt(d)*alpha[h]*one_hot(cq)]`、`K'=[K,one_hot(ck)]`，SDPA的scale仍为`1/sqrt(d)`。
V额外维度补0，输出裁回原宽度。数学上保留原logits、参数和梯度，不引入可训练参数。
未补观察槽的CUDA fp16/bf16路径因此使用融合SDPA，不显式构造每头N×N学习偏置矩阵，
也不拆分query。兼容Q/K LayerNorm在AMP下产生fp32的情况。其它条件保留通用实现。

float64前向与所有Q/K/V/关系权重梯度通过严格对照；
[B300 bf16强制FlashAttention对照](measurements/camera_relation_cuda_equivalence.json)
覆盖head_dim=64/96及fp32 Q/K在AMP下的情况。前向相对L2差约0.22%，
关系权重梯度的最大相对L2差约0.98%，属于不同bf16计算路径的数值差异；零初始化关系权重仍有梯度。

[18个真实giant前向/反向检查](measurements/giant_static_summary.json)使用step12500权重、
native输入、dense损失、source-count=4、关闭全部query分块，并与原GPU4/5训练共存。
所有损失/梯度为有限值，运动GT均保持未知掩码。

| 布局 | 最大实测allocated GiB（不含AdamW/DDP额外状态） |
|---|---:|
| Blended C4 / C8 | 14.87 / 22.40 |
| Taskonomy C4 / C8 | 16.91 / 27.20 |
| Hypersim C2 / C4 | 19.46 / 33.05 |

[原24个Kubric固定验证窗口的step12500复测](measurements/fused_baseline12500_validation.json)
确认计算实现修改后指标接近原基线：depth AbsRel=0.008651，动态位移EPE=0.022840。
该复测不证明新数据带来收益。

新增`scripts/evaluate_static_multiview.py`使用Blended/Taskonomy原manifest中排除的物理场景，
以及Hypersim官方val场景，建立固定视角/anchor计划；保持与训练物理场景不重叠。
后续在同一计划上比较加入静态多视角前后的几何指标，并继续观察原Kubric动态验证指标。

训练切换从最新完整step15000断点恢复原AdamW、各rank RNG、data epoch/batch游标和
5000→25000的原cosine学习率计划；不重新warmup，不改变TartanAir≤200m再p98的标签掩码。
具体启动状态及后续观测以服务器`outputs/training_preflight/active_training.json`和本目录的
实际rollout记录为准。

**已部署状态**

2026-10-02 13:55（北京时间）从step15000恢复，在GPU4/5继续训练至原计划25000步。
运行目录为`/data9/zbf/my_dfrt_multicamera4rc/outputs/full_dense_mixture_static_multiview_gpu45_20261002_135514`。
每GPU batch=1，全局batch=2；optimizer/RNG恢复，data epoch=0、原batch游标=10000，原LR plan保留。
[启动记录](measurements/training_job.json)和[实际更新检查](measurements/training_health.json)
记录新增阶段的222次更新（15001→15222），loss与梯度范数全部有限，无跳步。
Blended和Taskonomy均实际出现C4/C8，Hypersim出现C2/C4，均为T=1；另有一个Taskonomy
短共视邻域返回C3，未复制图像。全量准入池和原动态数据未减少。

本地199项测试通过；B300服务器199项通过。服务器DDP测试使用显式loopback rendezvous，
避免hostname解析等待；日志解析忽略非dict JSON行。新增相对路径集成用例覆盖验证manifest解析。
[切换前step15000的12个静态固定验证窗口](measurements/static_heldout_pre_static15000.json)
也已跑完：Hypersim训练/验证为365/46个物理场景，Blended为475/26，Taskonomy为510/27，
重叠均为0。监督仍为几何，尚不能据启动检查认定泛化指标改善。

监督进程将在17500步进行固定Kubric和静态多视角对照，并在25000步结束训练后再次评测。
