# Syn4D 原生适配

[返回 README](../README.md) · [官方数据介绍](https://jzr99.github.io/Syn4D/)

[Syn4DAdapter](../multicamera4rc/datasets/adapters/syn4d.py) 保留相机约定、深度单位、参考帧与同步帧标签。人体、衣服和物体网格用于恢复同一物理表面在其他时刻的位置，静态环境使用独立的静态身份；缺少网格或无可靠匹配的表面不能伪造动态 GT。

`frame_policy: reference` 使用官方参考帧关系，保留原始帧号间隔。`frame_policy: all` 支持参考帧不完全重叠的相机序列。`missing_mesh: mask` 屏蔽缺失网格的未知运动；严格完整性检查可使用 `missing_mesh: error`。图像内容边界、目标深度空洞、可见性未知和已有三维轨迹各有独立 mask。

原生固定验证配置见 [syn4d_eval.yaml](../multicamera4rc/configs/syn4d_eval.yaml)。将其中的 `/path/to/Syn4D` 改为自己的数据目录后，按 `evaluate --help` 指定数据配置、权重、固定计划和输出路径。数据源 split 用于构造互斥的验证场景，发布版只执行评估。

资产目录和 metadata 保持官方布局，补充人体/衣服网格放在数据包对应的 metadata 位置。全局缩放应同时作用于相机平移、深度、静态几何与动态位移。适配器与测试保留这些检查；正式数据和网格不包含在代码发布包中。
