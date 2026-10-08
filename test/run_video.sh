#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

INPUT_PATH="${1:-${INPUT_PATH:-}}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: bash visualization_latest_best_20260518/run_video.sh /path/to/video_or_images [extra args...]" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Default points at the current overall-best vitg checkpoint (ckpt144, 10dataset_vitg_dust3r).
# Override CHECKPOINT or CHECKPOINT_DIR to run a different one.
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
INPUT_BASENAME="$(basename "$INPUT_PATH")"
INPUT_STEM="${INPUT_BASENAME%.*}"
if [[ -d "$INPUT_PATH" && "$INPUT_BASENAME" == "images" ]]; then
  INPUT_STEM="$(basename "$(dirname "$INPUT_PATH")")"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CHECKPOINT")/vis_$(basename "$CHECKPOINT" .pth)_video_${INPUT_STEM}_densepc_s2}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$SCRIPT_DIR:$ROOT_DIR" "$PYTHON_BIN" \
  "$SCRIPT_DIR/visualize_video_checkpoint.py" \
  --input-path "$INPUT_PATH" \
  --checkpoint "$CHECKPOINT" \
  --model-variant "${MODEL_VARIANT:-giant}" \
  --output-dir "$OUTPUT_DIR" \
  --patch-provider "${PATCH_PROVIDER:-sampled_resized}" \
  --resolution "${RESOLUTION:-256}" \
  --num-frames "${NUM_FRAMES:-48}" \
  --sampling-mode "${SAMPLING_MODE:-contiguous}" \
  --start-index "${START_INDEX:--1}" \
  --query-grid-size "${QUERY_GRID_SIZE:-32}" \
  --query-margin "${QUERY_MARGIN:-0.06}" \
  --num-display-points "${NUM_DISPLAY_POINTS:-180}" \
  --track-tail "${TRACK_TAIL:-12}" \
  --dense-point-cloud-stride "${DENSE_POINT_CLOUD_STRIDE:-2}" \
  --dense-vis-threshold "${DENSE_VIS_THRESHOLD:-0.3}" \
  --dense-confidence-percentile "${DENSE_CONFIDENCE_PERCENTILE:-10}" \
  --dense-query-batch-size "${DENSE_QUERY_BATCH_SIZE:-8192}" \
  --dense-motion-percentile "${DENSE_MOTION_PERCENTILE:-85}" \
  --gif-fps "${GIF_FPS:-8}" \
  --seed "${SEED:-42}" \
  --device "${DEVICE:-cuda}" \
  --normalize-3d \
  "$@"
