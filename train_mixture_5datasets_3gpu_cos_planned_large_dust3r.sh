#!/usr/bin/env bash
# large + DUSt3R-style two-branch 3D loss normalization.
#● cd /data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration && mkdir -p logs && LOG=logs/dust3r_large_$(date +%Y%m%d_%H%M%S).log && bash train_mixture_5datasets_3gpu_cos_planned_large_dust3r.sh 2>&1 | tee "$LOG"
# 行为：与 large 主脚本一致，但开启 --dust3r-metric-branch。
#   - metric 数据集（METRIC_DATASETS 内）：z̄=z 共享归一化（保留 metric scale 信号）
#   - 非 metric 数据集（如 co3dv2）：各自独立归一化（论文默认 scale-invariant）
#
# 冷启动：没有可用的 large checkpoint（giant ckpt 维度不兼容），
# 从 videomae-v2-large 初始化。输出目录用单独的 *_large_dust3r。
# 想暖启动就显式传 RESUME=<large_ckpt> 调用本脚本。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- DUSt3R two-branch normalization 开关 ---
export DUST3R_METRIC_BRANCH=1
# 默认 metric 列表：mixture 里所有带真实/伪真实 metric 单位的数据集。
# co3dv2 是 SfM 伪 GT 尺度不可信，不在内（保持 scale-invariant）。
export METRIC_DATASETS="${METRIC_DATASETS:-scannet,scannetpp,dynamic_replica,kubric,blendedmvs,pointodyssey,tartanair,vkitti2,mvssynth,waymo}"

# 冷启动：默认不 resume。large 没有兼容 ckpt，从 videomae-v2-large 初始化。
export RESUME="${RESUME:-}"

# 冷启动不需要 resume-warmup（lr warmup 由主 launcher 的 LR_WARMUP_STEPS 负责）。
export RESUME_WARMUP_STEPS="${RESUME_WARMUP_STEPS:-0}"
export EPOCHS="${EPOCHS:-300}"

# --- spool 放到 /dev/shm（内存 tmpfs），避免 ext4 HDD 随机 IO 成为瓶颈 ---
# spool 默认在 /tmp（ext4 磁盘），7 rank 并发 listdir 会打满 HDD；
# /dev/shm 是 1.5T tmpfs，listdir/stat 是纯内存操作，消除 IO 等待。
export SPOOL_DIR="${SPOOL_DIR:-/dev/shm/d4rt_spool}"

# --- 性能调优覆盖（可被环境变量继续覆盖）---
# 数据加载：从 4 worker 加到 8，减少 spool_ready=0 的概率
export NUM_WORKERS="${NUM_WORKERS:-8}"
# 慢样本（waymo tfrecord 等）等超 8s 就跳过，不堵 planned-order 队列
export D4RT_PLANNED_FORCE_SKIP_S="${D4RT_PLANNED_FORCE_SKIP_S:-15}"
# 扩大 relaxed window，让已 ready 样本先返回
export D4RT_PLANNED_RELAXED_LOOKAHEAD="${D4RT_PLANNED_RELAXED_LOOKAHEAD:-2400}"
# per-rank batch size：默认 14（比 large 主脚本的 28 保守，首跑先确认显存）
export BATCH_SIZE="${BATCH_SIZE:-14}"
# builder worker 数：默认 18 在 7卡×14bs 下不够，epoch 中期 spool 会耗尽
# waymo/pointodyssey 等慢数据集会把 worker 全占满，提到 36 保留补充余量
export BUILDER_WORKERS="${BUILDER_WORKERS:-32}"
export MAX_SPOOL_BYTES_GB="${MAX_SPOOL_BYTES_GB:-100}"

# --- stager 只对 NFS 数据集生效 ---
# pointodyssey/kubric/mvssynth/waymo/vkitti2 的 root 都在本地 NVMe（/data2,/data3），
# 源盘已是 ~1.6GB/s。stager 对它们做的是 NVMe->NVMe 整文件 copy + per-file flock +
# eviction 扫描，纯开销，还和同 scene 的其它 clip 抢锁（"线程冲突"），白占 builder。
# 摘掉后这些数据集 supports()=False，stage_sample 直接 yield adapter（不 copy），
# adapter 从源盘直读；spool 预生成流程不变。
# 只保留真正在 NFS（/data4,/data5，单机 10.129.32.193，~210MB/s）上的数据集走 stager，
# 因为它们靠 planned 顺序的 sequence_locality（同 scene ~96 个 clip 连续消费）把
# 一次 NFS 读摊销到 ~96 次本地 hardlink 命中。
export SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS:-co3dv2,scannetpp}"

# --- 隔离输出目录 ---
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/10dataset_large_dust3r}"

echo "[large_dust3r] DUST3R_METRIC_BRANCH=$DUST3R_METRIC_BRANCH"
echo "[large_dust3r] METRIC_DATASETS=$METRIC_DATASETS"
echo "[large_dust3r] RESUME_WARMUP_STEPS=$RESUME_WARMUP_STEPS"
echo "[large_dust3r] OUTPUT_DIR=$OUTPUT_DIR"
echo "[large_dust3r] RESUME=${RESUME:-<unset, cold start from videomae-v2-large>}"
echo "[large_dust3r] ⚠ 冷启动：metric 数据集 loss_3d 初期偏高是 dust3r 分支抓 scale 的预期行为；"
echo "[large_dust3r]   监控 loss_log.jsonl 的 pred_abs_depth_mean 是否朝 GT 收敛。"

exec bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned_large.sh" "$@"
