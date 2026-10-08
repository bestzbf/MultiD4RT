#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Default points at the current overall-best vitg checkpoint (ckpt144, 10dataset_vitg_dust3r).
# Override CHECKPOINT or CHECKPOINT_DIR to evaluate a different one.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/10dataset_vitg_dust3r}"
DEFAULT_CHECKPOINT="$CHECKPOINT_DIR/checkpoint_latest_144.pth"
find_latest_checkpoint() {
  local latest
  latest="$(find "$CHECKPOINT_DIR" -maxdepth 1 -name 'checkpoint_latest_*.pth' -printf '%f\n' | sort -V | tail -n 1)"
  if [[ -z "$latest" ]]; then
    echo "No checkpoint_latest_*.pth found in CHECKPOINT_DIR=$CHECKPOINT_DIR" >&2
    exit 1
  fi
  printf '%s/%s\n' "$CHECKPOINT_DIR" "$latest"
}
# Prefer the pinned ckpt144 when present; otherwise fall back to newest in CHECKPOINT_DIR.
if [[ -z "${CHECKPOINT:-}" && -f "$DEFAULT_CHECKPOINT" ]]; then
  CHECKPOINT="$DEFAULT_CHECKPOINT"
fi
CHECKPOINT="${CHECKPOINT:-$(find_latest_checkpoint)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CHECKPOINT")/eval_$(basename "$CHECKPOINT" .pth)_sintel_paper14_strict_latest_best}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$SCRIPT_DIR:$ROOT_DIR" "$PYTHON_BIN" \
  "$SCRIPT_DIR/eval_sintel_checkpoint.py" \
  --root "${SINTEL_ROOT:-/data3/dataset/sintel}" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --model-variant "${MODEL_VARIANT:-giant}" \
  --patch-provider "${PATCH_PROVIDER:-sampled_resized}" \
  --resolution 256 \
  --num-frames 48 \
  --use-14-subset \
  --num-scenes "${NUM_SCENES:-0}" \
  --scene-names "${SCENE_NAMES:-}" \
  --sintel-pass "${SINTEL_PASS:-final}" \
  --depth-stride "${DEPTH_STRIDE:-1}" \
  --pointcloud-stride "${POINTCLOUD_STRIDE:-1}" \
  --max-pointcloud-points "${MAX_POINTCLOUD_POINTS:-0}" \
  --depth-confidence-quantile "${DEPTH_CONFIDENCE_QUANTILE:-0.0}" \
  --pointcloud-confidence-quantile "${POINTCLOUD_CONFIDENCE_QUANTILE:-0.0}" \
  --pose-grid-h "${POSE_GRID_H:-8}" \
  --pose-grid-w "${POSE_GRID_W:-8}" \
  --pose-confidence-threshold "${POSE_CONFIDENCE_THRESHOLD:-0.0}" \
  --pose-solver "${POSE_SOLVER:-umeyama}" \
  --pose-ransac-iters "${POSE_RANSAC_ITERS:-256}" \
  --pose-inlier-thresh "${POSE_INLIER_THRESH:-0.1}" \
  --reference-frame "${REFERENCE_FRAME:-0}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-4096}" \
  --device "${DEVICE:-cuda}" \
  "$@"
