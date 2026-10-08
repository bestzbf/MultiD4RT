# DynamicReplica 加载异常与 50k 续训恢复

2026-10-03，GPU4/5 可变 C/T 课程在完成第 29,022 步后，rank0 的 DataLoader 读取样本 `43498` 时失败；rank1 随后因等待同步超时退出。最近完整 checkpoint 为 27,500 步，后面的 1,522 步未保存。停止前最后一步 loss=-2.3019、裁剪前梯度范数 527.35，未跳过优化器更新；日志没有 OOM 或非有限 loss 导致本次退出的证据。

失败流为 `dynamic_replica/2dc7d2-3_obj_source_right`，T16 覆盖原生帧 284–299。`flow_backward/..._0286.png` 是形状 `[720,2560]` 的 8 位全零 PNG，对应官方 mask `[720,1280]` 也全零。旧代码先解码 float16 位图再读 mask，因此对已无有效标签的占位文件直接抛错。

## 修复语义

加载器先检查正反向 mask 的存在和尺寸。任一方向 mask 全零，时间对不可能通过现有双向一致性检查：保留几何，令该对运动 GT 未知。两向 mask 均非空时，继续严格检查两个 flow 的 16 位编码及尺寸。

没有按光流数值为零删除静止标签，没有将 8 位文件重解释为半精度，没有整体删除右相机或 DynamicReplica 数据源。未知运动没有转成零位移 GT。

## 原生文件检查

- [flow_headers.json](flow_headers.json)：扫描本地 train 的 966 个相机流目录、577,668 个光流 PNG 文件头。577,650 个为 16 位灰度；另外 18 个为 8 位灰度，逐一解码检查后均为全零占位且官方 mask 全零。未发现带非空 mask 的非 16 位文件或读文件错误。
- [failed_sample_replay.json](failed_sample_replay.json)：从原课程完整混合数据加载器重新读取同一 `sample_index=43498`，保持帧号 284–299，模型输入 `[16,3,728,1288]`。保留 14,745,600 个有效深度像素及通过过滤的 1 个邻帧运动像素，读取成功。

文件头扫描覆盖本地 train 文件的编码结构；不代表对全部 16 位光流内容完成逐像素物理正确性验证。原有深度、位姿、可见性和双向一致性过滤继续生效。

## 回归与恢复

回归覆盖：正向或反向 mask 全零的占位文件、几何保留、合法零运动邻居、有效 mask 下损坏编码继续报错，以及恢复控制器拒绝不匹配 checkpoint、保留后续训练/验证阶段、避免重新启动 LR warmup 和采样游标。

使用 `scripts/recover_variable_continuation.py` 从 27,500 步恢复，保持每卡 BS1 / global BS2、GPU4/5 共享、全有效 GT、原生长宽比、原课程和 50k 目标。checkpoint 中 1,058 个 AdamW 参数状态及两 rank 的 RNG 存在；原 `data_epoch=0,batches_consumed=2500`、25k–50k LR 计划继续沿用。

恢复显式设置 `decoder_memory=target_time`，保留该 checkpoint 的历史模型语义；不会随新版模型默认值切换为 `all_observations`。保存间隔改为 1,000 步，恢复脚本按实际 checkpoint 大小检查剩余磁盘。原失败控制器和日志保留，新建恢复控制目录。实时是否运行、最新步数及最终测试输出以服务器 `active_training.json`、恢复目录 `progress.json/supervisor.log` 和 `outputs/training_preflight/replica_recovery_20261003/` 为准。
