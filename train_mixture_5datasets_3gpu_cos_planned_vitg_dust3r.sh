#!/usr/bin/env bash
# vitg + DUSt3R-style two-branch 3D loss normalization.
#● cd /data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration && mkdir -p logs && LOG=logs/dust3r_$(date +%Y%m%d_%H%M%S).log && bash train_mixture_5datasets_3gpu_cos_planned_vitg_dust3r.sh 2>&1 | tee "$LOG"
# 行为：与 vitg 主脚本一致，但开启 --dust3r-metric-branch。
#   - metric 数据集（METRIC_DATASETS 内）：z̄=z 共享归一化（保留 metric scale 信号）
#   - 非 metric 数据集（如 co3dv2）：各自独立归一化（论文默认 scale-invariant）
#
# 默认从原 vitg 输出目录的 checkpoint_latest_144.pth 暖启动；
# 输出目录改为 *_dust3r 以避免覆盖原 run 的 ckpt。
# 想冷启动就 RESUME="" 调用本脚本。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- DUSt3R two-branch normalization 开关 ---
export DUST3R_METRIC_BRANCH=1
# 默认 metric 列表：你 mixture 里所有带真实/伪真实 metric 单位的数据集。
# co3dv2 是 SfM 伪 GT 尺度不可信，不在内（保持 scale-invariant）。
export METRIC_DATASETS="${METRIC_DATASETS:-scannet,scannetpp,dynamic_replica,kubric,blendedmvs,pointodyssey,tartanair,vkitti2,mvssynth,waymo}"

# export RESUME="${RESUME:-/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_vitg_videomaev2_dust3r/checkpoint_latest_158.pth}"
export RESUME="${RESUME:-/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/10dataset_vitg_dust3r_from0/checkpoint_latest_132.pth}"
# export PRETRAIN="${PRETRAIN:-outputs/mixture_6datasets_cos_planned_vitg_videomaev2_dust3r/checkpoint_latest_235.pth}"

# --- 暖启动 lr warmup ---
# 从 ckpt 暖启动后做一次 0 -> peak_lr 的线性 warmup（与冷启动同形状），
# 保护权重渡过新 loss 配置带来的梯度跃升。300 步 ≈ 6 epoch 的 batch 数（已经够缓）。
# 设为 0 可关闭，让 lr 直接接回 cosine 当前位置。
export RESUME_WARMUP_STEPS="${RESUME_WARMUP_STEPS:-400}"
export EPOCHS="${EPOCHS:-300}"

# --- staging 只对 NFS 数据集生效 ---
# co3dv2(/data4) 和 scannetpp(/data5) 在 NFS 上，staging 有收益。
# 其余数据集（pointodyssey/kubric/mvssynth/tartanair/waymo/vkitti2）都在本地 NVMe
# (/data2,/data3)，staging 是纯 copy 开销，不走。
export SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS:-co3dv2,scannetpp}"

# --- 隔离输出目录 ---
# 用单独目录，原 run 的 ckpt 不会被新 run 覆盖；
# RESUME 仍指向原目录的 ckpt_144 实现暖启动。
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/10dataset_vitg_dust3r_from0}"

echo "[vitg_dust3r] DUST3R_METRIC_BRANCH=$DUST3R_METRIC_BRANCH"
echo "[vitg_dust3r] METRIC_DATASETS=$METRIC_DATASETS"
echo "[vitg_dust3r] RESUME_WARMUP_STEPS=$RESUME_WARMUP_STEPS"
echo "[vitg_dust3r] OUTPUT_DIR=$OUTPUT_DIR"
echo "[vitg_dust3r] RESUME=${RESUME:-<unset, will use vitg default>}"
echo "[vitg_dust3r] ⚠ 暖启动后 metric 数据集 loss_3d 会从 ~0.02 跳到 ~1+，是 dust3r 分支抓到 scale 漂移的预期行为。"
echo "[vitg_dust3r]   前 $RESUME_WARMUP_STEPS optimizer steps 走 0 -> peak_lr 线性 warmup 缓冲；"
echo "[vitg_dust3r]   监控 loss_log.jsonl 的 pred_abs_depth_mean 是否朝 GT 收敛。"

exec bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned_vitg.sh" "$@"
