#!/usr/bin/env bash
# 连续过拟合测试：ScanNet / TartanAir / VirtualKitti 各 5 epoch，GPU 0，本地数据
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/d4rt/bin/torchrun}"
PRETRAIN="/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from0/checkpoint_latest_200.pth"
MASTER_PORT=28597

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONWARNINGS="ignore:mkl-service package failed to import:UserWarning"
export D4RT_SERIALIZE_ADAPTER_INIT=1
mkdir -p /data1/zbf/d4rt_tmp_overfit_seq

run_overfit() {
  local name="$1"
  local config="$2"
  local outdir="outputs/overfit_seq_${name}"
  echo ""
  echo "========================================"
  echo "[overfit-seq] dataset=${name}  config=${config}"
  echo "========================================"
  CUDA_VISIBLE_DEVICES=0 "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=1 --master_port=$MASTER_PORT \
    train_mixture.py \
    --config "$config" \
    --batch-size 5 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --grad-accum 1 \
    --log-interval 5 \
    --epochs 2 \
    --lr 1e-4 \
    --lr-warmup-steps 10 \
    --num-frames 48 \
    --output-dir "$outdir" \
    --patch-provider sampled_highres \
    --val-interval 999 \
    --val-samples 5 \
    --save-interval 1 \
    --loss-w-3d 1.0 \
    --loss-w-conf 0.1 \
    --loss-w-normal 0.0 \
    --loss-w-static-reprojection 0.9 \
    --loss-3d-mode scale_invariant \
    --dist-timeout-minutes 30 \
    --variant large \
    --use-videomae-v2-init \
    --videomae-model /data1/zbf/pretrained/videomae-v2-large \
    --pretrain "$PRETRAIN"
  echo "[overfit-seq] ${name} PASSED"
  MASTER_PORT=$((MASTER_PORT + 1))
}

run_overfit "scannet_scene0000_00"  "configs/overfit_scannet_scene.yaml"
run_overfit "tartanair_P000"        "configs/overfit_tartanair_scene.yaml"
run_overfit "vkitti2_clone"         "configs/overfit_vkitti2_scene.yaml"

echo ""
echo "All 3 datasets passed overfit test."
