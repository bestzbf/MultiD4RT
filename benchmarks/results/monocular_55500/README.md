# 冻结的 55,500 步单目评估结果

[返回项目](../../../README.md) · [评估协议](../../../docs/monocular-evaluation.md)

此处保留已有 checkpoint 的完整评估结果，不是本次代码清理重新测得的精度。ScanNet test100：100/100 场景；Sintel Final14：14/14 场景；合计 114 场景、零失败。按场景等权汇总，场景内部使用全部有效评估网格像素。

| 指标 | ScanNet | Sintel |
|---|---:|---:|
| Depth S AbsRel | 0.020179 | 0.255322 |
| Depth SS AbsRel | 0.017680 | 0.202269 |
| Depth S δ1 | 99.2457% | 69.4671% |
| 点云 Sim3 L1 / m | 0.031174 | 1.481982 |
| ATE mean / m | 0.029777 | 0.055105 |
| RPE-T / m | 0.012745 | 0.022647 |
| RPE-R / degree | 0.378413 | 0.294246 |
| Pose AUC@30 | 43.6590% | 20.0952% |

输入为 C1、最多 48 个实际帧，保持比例、长边 518、patch14 补边；评估网格为 256。没有置信度过滤，也不采样 4,096 查询点。

- [汇总 JSON](summary.json) 与 [逐场景 CSV](per_scene.csv)
- ScanNet：[固定计划](scannet/fixed_plan.json)、[GT 审计](scannet/gt_audit.json)、[完整结果](scannet/evaluation.json)
- Sintel：[固定计划](sintel/fixed_plan.json)、[GT 审计](sintel/gt_audit.json)、[完整结果](sintel/evaluation.json)、[GT 位姿运动审计](sintel/gt_pose_motion_audit.json)

这是几何评估，不含 scene flow、动态轨迹或多相机评分。当前相机头和保持比例的输入与旧 D4RT 入口不同；点云 Sim3 是尺度对齐误差。Sintel 的零/近零 GT 平移会影响旧 translation-direction AUC 的解释，详见协议与 GT 审计。
