# 原生数据适配与 GT

[返回 README](../README.md)

发布版保留 `multicamera4rc/datasets`，用于原生数据推理和几何、运动评估。数据路径由 JSON/YAML 配置指定；适配器使用到相应格式时才加载额外依赖。混合配置可以包含多个数据源，`create_dataset` 与 `collate_adapter_samples` 负责片段读取与可变布局补齐，没有优化器或训练步骤。

适配器注册、来源与格式见 [registry.py](../multicamera4rc/datasets/registry.py)、[UPSTREAM.json](../multicamera4rc/datasets/UPSTREAM.json)。当前原生格式覆盖 Kubric、PointOdyssey、vKITTI2、TartanAir、BlendedMVS、Syn4D 等数据源；能评估什么取决于该数据源真实提供的 GT，不能从静态几何构造虚假动态轨迹。

## 几何和时间约定

- 图像默认保留原始比例，按 patch14 补边；resize 必须同步变换内参与像素坐标。
- 同一个片段所有相机使用共享帧标签，保留抽帧间隔。
- 世界坐标与场景尺度在整个片段中一致，GT 不进入模型 forward。
- padding、坏深度、缺失轨迹及未知可见性使用独立有效 mask。
- 遮挡不等于未知三维位置，出视野与未知标签也分别处理。

`bridge.py` 将原生记录转换为 RGB 与模型元数据；`dense.py`、`partial_dense.py` 和动态数据 recipe 将真实 GT 投到输出布局。原生粒子轨迹只监督/评估已有粒子，不凭空补成逐像素运动 GT。静态多视角统一为同一时刻；没有可靠运动标签的数据只评估几何。

配置中的 `train`、`val`、`test` 是官方数据分区或冻结的互斥场景集合。评估会保留这些标签并检查场景重合，它们不代表发布版包含训练循环。

## 评估

`evaluate` 使用原生 renderer 数据及固定验证计划，`native-dynamic` 读取冻结的动态验证布局，`static-multiview` 检查静态留出场景。它们直接计算物理误差，完整保留 GT 有效 mask，不计算训练损失。

ScanNet 和 Sintel 使用独立单目协议，GT 单位、深度上限、评估网格和对齐方式见 [单目评估](monocular-evaluation.md)。Syn4D 的网格轨迹要求完整的物体/人体/衣服资产，见 [Syn4D](syn4d.md)。
