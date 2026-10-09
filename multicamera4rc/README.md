# MultiCamera4RC Python package

[项目入口](../README.md) · [模型与维度](../docs/architecture.md) · [推理](../docs/inference.md) · [数据与 GT](../docs/data.md)

安装包提供模型、权重兼容、原生数据读取和独立评估。CLI 命令为 `infer`、`evaluate`、`monocular`、`native-dynamic`、`static-multiview`、`demo`、`adapter-demo`。

输入 RGB `[B,S,3,H,W]`、相机 ID `[B,S]` 和共享帧标签 `[B,S]`，支持可变相机数/时刻数。输出深度、相机参数、世界基础点、基础点加三维位移的轨迹和逐相机可见性。原生 checkpoint 保存的时间与 decoder KV 语义在加载时保留。

发布版不包含训练入口、训练损失、优化器恢复或学习率控制。tiny demo 使用随机权重检验推理接口；真实效果需要适配该模型的已训练 checkpoint。
