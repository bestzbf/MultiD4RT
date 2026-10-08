# 原生双目、TartanAir V2 与 PointOdyssey 动态监督

服务器：B300zbf；代码：`/data9/zbf/my_dfrt_multicamera4rc`；分支：`codex/multicamera-4rc`。
该目录的 measurements 保存真实数据核验、实际混合配置与运行证据。图像只补齐 patch14，K/E只构造GT，不进入模型。

## 数据与监督

| 数据源 | 实际路径 | 新输入 | 使用的标签 |
|---|---|---|---|
| vKITTI2 | `/data8/mugen/vKITTI2` | 保留左右单目C1T8，增加同步C2T6 | 全像素depth、camera、ray、基础点；原生前后向邻帧scene flow与可见性 |
| TartanAir V1 | `/data8/mugen/TartanAir_V1` | 保留左右单目C1T8，增加同步C2T6 | 全像素几何；不冒充动态物体scene flow |
| TartanAir V2 | `/data8/mugen/TartanAir_V2_subset` | 六个同朝向左右双目C2T6，左前单目C1T8 | 原生pinhole RGB、无损float32 depth、NED camera pose；全像素几何 |
| PointOdyssey | `/data5/d4rt_dataset/PointOdyssey` | C1T8 | 全像素几何，加全部有效原生粒子的trajectory、displacement、velocity、visibility |

原始图像分别为vKITTI2 375×1242、V1 480×640、V2 640×640、PointOdyssey 540×960。
网络补齐为378×1246、490×644、644×644、546×966，没有额外resize。
同步双目采用camera-major排列：

```text
images        [B,12,3,H,W]
camera_id     [0,0,0,0,0,0,1,1,1,1,1,1]
time_idx      [t0,t1,t2,t3,t4,t5,t0,t1,t2,t3,t4,t5]
query_time_idx[t0,t1,t2,t3,t4,t5]
```

整数原生帧号保留时间间隔；不要求timestamp。左右相机共享同一个世界坐标，整个样本再规范到首个有效相机与共同scene scale。
V2有74个环境目录，其中19个环境具备完整的六朝向双目包；其余55个有完整左前RGB/depth/pose包，作为单目加入。
V2双目每朝向278个训练序列、12个验证序列；左前单目1050个训练序列、72个验证序列，训练70个环境、验证4个环境。
所有V2数据源共享按物理环境划分的留出集：AbandonedFactory2、Ruins、ModUrbanCity、ModernCityDowntown。
双目包只在AbandonedFactory2有对应验证序列。不同方向的物理轨迹会重复，不把六个方向的序列数当作独立场景数。

## 修复与 GT 原则

vKITTI2原生scene flow是相机坐标的端点差，包含相机运动。使用
`Xw1=inv(E1)*(Xc0+SF)`、`delta=Xw1-inv(E0)*Xc0`，不能只旋转SF当作世界位移。
邻帧索引从单独frame号改为`(camera_name, frame_number)`，防止双目重复帧号匹配另一台相机。
保留GT定义有效的遮挡端点，不强制目标深度可见；过滤原生编码饱和、非有限和光流投影不一致的标签。
之前的[单目核验](../monocular_20261002/README.md)已经通过原生物体pose独立核对clone序列的运动；
本次增加背景静止世界点、左右同instance的世界scene flow交叉检查。

TartanAir V1保持正确的NED→OpenCV位姿与Z-depth；V2使用90度pinhole的`fx=fy=320,cx=cy=320`。
V2深度按OpenCV返回的四字节直接`view('<f4')`，不得交换RGB通道后解码。
V1/V2先过滤非有限、非正、超过200米的深度，再对合格正深度计算p98掩码，不改变RGB或把远深度夹成错误的有效GT。
官方TartanAir flow基于静态环境的depth/pose重投影，不提供可靠的动态物体world scene flow。
因此不使用虚构零位移或相机运动产生的像素flow训练动态物体头。

PointOdyssey旧几何适配器没有接入粒子运动标签。新的`pointodyssey_tracks`在同一完整dense几何样本上添加原生动态标签：

- 131个准入序列、236883帧；109个有原生三维轨迹，22个只有有效2D轨迹而三维字段为空。
- 原生三维轨迹保留粒子ID和原始XYZ；若原世界坐标左手，则同时变换E与XYZ，不只改相机矩阵。
- 只从可见、valid、落在图内、与原生XYZ重投影/深度一致的位置发起查询。可见XYZ不一致时，source和target的三维监督都屏蔽；visibility标注独立保留。
- 原生三维标注可监督遮挡时刻的XYZ；22个2D-only序列仅用可见、有效且平滑表面的原生深度恢复XYZ，不生成遮挡三维位置。
- 原始半精度XYZ/UV及边界UV取整无法恢复成无限精度。源码采用2像素重投影阈值及3% Z-depth加深度量化容差；2D-only恢复还要求所有双线性插值邻点在平滑有效表面。
- decoder所选source使用其全部合格原生粒子，没有4096或其他查询配额，也没有把粒子插值成整张图的虚构运动GT。depth/ray/基础点仍监督全部有效像素。
- 增加真实帧间隔归一化的trajectory/displacement/velocity/visibility损失；velocity单位为每帧。训练日志保留标签来源、遮挡XYZ策略、实际参与损失的粒子数与各项动态损失。

## 共视率与核验范围

采用双向stride8有效原生深度网格，以目标图内且Z-depth误差不超过5%作为共视条件；
分母包含有效源点中投出视野或被遮挡的点。以下是每类8个首尾窗口、16个投影方向的统计，不是全数据集保证值。

| 数据 | 有效深度共视率中位数 | 范围 | 原生基线 |
|---|---:|---:|---:|
| vKITTI2 | 92.1% | 83.4%–95.0% | 0.532725 m |
| TartanAir V1 | 90.2% | 35.7%–95.6% | 0.25 m |
| V2 front/back/left/right/top/bottom | 95.3/92.6/93.4/91.9/91.9/93.1% | 各方向详见JSON | 0.25 m |

V1近物遮挡时重合度会下降；不同相反朝向可能没有共同视野，因此按同朝向左右配对。
vKITTI2额外以同instance、同物理时间、双方原生SF有效掩码检验左右世界位移，结果记录于GT报告。
有物体对应的窗口中，左右world displacement差异P95最大约1.44毫米。
nearest像素与5%深度筛选仍有边界误差，交叉比较统计不取代原生scene flow定义。

验证分为全池schema/zip成员/标定检查、每次加载时的标签掩码、真实窗口核验及giant反传。
10个新适配数据源的80个代表首尾窗口已通过；PointOdyssey另核验全部131序列首尾共262个窗口，
原生3D可见候选约99.35%通过source一致性检查。
并未逐像素离线重算全部数据集的每一帧；原始有限精度和缺失遮挡3D不能凭空消除。
macOS与B300zbf均通过211项测试，包含同帧不同相机误配回归、2D-only表面边界、可见XYZ目标筛选、
无visibility配置仍训练motion、全部原生查询与geometry/motion/visibility激活检查点的输出和全部参数梯度对照。

## 实际混合与训练接续

升级后24个采样源，6496033个coverage窗口，epoch size 12992066；旧准入数据池保留。
保留Kubric C2T12/C3T8/C4T6、DynamicReplica C1T8、几何单目，以及BlendedMVS/Taskonomy C4/8T1、Hypersim C2/4T1。
将原vKITTI2与V1采样权重的一半用于双目，保留各自总权重；PointOdyssey权重0.3→0.75；V2双目六方向各0.1、左前单目0.2。

配置改变coverage索引，使用`--reset-data-cursor`重启数据游标；仍恢复model、AdamW、各rank RNG和原5000→25000的cosine计划。
GPU4、5维持giant、bf16、每卡BS1、全局BS2、每个样本4个decoder source，无attention/visibility查询分块。
geometry DPT与完整motion/visibility head增加等价激活检查点；训练参数与输出定义未改变。
对照测试覆盖depth/depth_conf、ray/ray_conf、camera encoding、trajectory、displacement、motion confidence，以及开启/关闭visibility的全部参数梯度。
共享GPU剩余约42GiB无法完成新的PointOdyssey giant动态反传，32–37GiB限制下的预检OOM没有算作通过。
`scripts/cutover_native_motion_stereo.py`等待本项目原生17500步原子检查点，仅停止记录的同UID训练进程树，
以释放的GPU4完成15个真实native样本giant前向/反传，通过后接续新的混合训练；若核验失败，自动从17500恢复旧数据配置。
固定Kubric与静态多视角留出验证在25000步继续执行。实际运行阶段见`training_job.json`与`training_runtime.json`，giant证据见`giant_preflight.json`。

截至2026-10-02 08:40 UTC（北京时间16:40）的实际核验，15个giant native前向/反向样本全部通过，
PointOdyssey/vKITTI2 motion decoder收到非零梯度，TartanAir仅几何头收到任务监督，全部梯度有限。
单卡giant反传最大PyTorch allocated显存约53.39GiB；该预检不创建AdamW，不代表完整训练显存峰值。
GPU4/5已从17500恢复并运行至17645，共145次新配置optimizer更新，未发现非有限loss/grad、跳步或运行时无效clip。
训练参数仍为giant、bf16、每卡BS1、全局BS2、4个source、原生尺寸、整段查询。模型AdamW已恢复，原cosine计划保留，数据游标因新索引重置。
7个PointOdyssey批次已产生非零sparse trajectory/displacement/velocity/visibility损失；其中`kitchen_gfloor`批次使用36723条有效原生查询、293568个三维点时刻，仍监督2048704个有效几何source像素。
vKITTI2双目、V1双目、V2六方向双目及左前单目均已进入实际训练日志；配置仍可采样全部24个已准入源，并非声称145步已遍历全量数据。
训练输出：`/data9/zbf/my_dfrt_multicamera4rc/outputs/full_dense_native_motion_stereo_gpu45_20261002_082322UTC`，继续运行到25000步。

## 官方格式依据

- [vKITTI2原生RGB、depth、camera和scene flow定义](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)
- [TartanAir V2 pinhole、无损depth、NED pose、静态重投影flow](https://tartanair.org/modalities.html)
- [PointOdyssey原生tracking生成](https://github.com/y-zheng18/point_odyssey/blob/main/utils/gen_tracking.py)
