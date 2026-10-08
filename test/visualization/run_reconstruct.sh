#!/usr/bin/env bash
# Multi-seed full-scene reconstruction wrapper.
#
# Seeds the dense query grid from several keyframes (not just the reference
# frame), decodes every seed into one unified camera frame, voxel-dedups, and
# writes a single canonical point cloud (mode A: modeA_full_scene.ply) plus a
# camera fly-through GIF, and a per-frame variant (mode B). This is the route
# that produces modeA_full_scene.ply — NOT run_video.sh (which is per-reference
# -frame and writes pred_dense_reference_frame_NNN.ply instead).
#
# Usage:
#   CHECKPOINT=./weights/giant.pth bash test/visualization/run_reconstruct.sh test/examples/sample_video.mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$TEST_DIR/.." && pwd)"
cd "$ROOT_DIR"

INPUT_PATH="${1:-${INPUT_PATH:-}}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: bash test/visualization/run_reconstruct.sh /path/to/video_or_images [extra args...]" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Default points at the current overall-best giant checkpoint. Override CHECKPOINT
# or CHECKPOINT_DIR to run a different one.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/outputs/10dataset_vitg_dust3r}"
DEFAULT_CHECKPOINT="$CHECKPOINT_DIR/checkpoint_latest_144.pth"
if [[ -z "${CHECKPOINT:-}" && -f "$ROOT_DIR/outputs/giant.pth" ]]; then
  CHECKPOINT="$ROOT_DIR/outputs/giant.pth"
elif [[ -z "${CHECKPOINT:-}" && -f "$DEFAULT_CHECKPOINT" ]]; then
  CHECKPOINT="$DEFAULT_CHECKPOINT"
fi
if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "Set CHECKPOINT=/path/to/weights.pth (no default giant checkpoint found)." >&2
  exit 1
fi

INPUT_BASENAME="$(basename "$INPUT_PATH")"
INPUT_STEM="${INPUT_BASENAME%.*}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/reconstruct_$(basename "$CHECKPOINT" .pth)_${INPUT_STEM}}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$SCRIPT_DIR:$TEST_DIR:$ROOT_DIR" "$PYTHON_BIN" \
  "$SCRIPT_DIR/visualize_multiseed_reconstruction.py" \
  --input-path "$INPUT_PATH" \
  --checkpoint "$CHECKPOINT" \
  --model-variant "${MODEL_VARIANT:-giant}" \
  --output-dir "$OUTPUT_DIR" \
  --patch-provider "${PATCH_PROVIDER:-sampled_resized}" \
  --resolution "${RESOLUTION:-256}" \
  --num-frames "${NUM_FRAMES:-48}" \
  --sampling-mode "${SAMPLING_MODE:-uniform}" \
  --start-index "${START_INDEX:--1}" \
  --num-seed-frames "${NUM_SEED_FRAMES:-8}" \
  --voxel-size "${VOXEL_SIZE:-0.01}" \
  --recon-mode "${RECON_MODE:-both}" \
  --dense-point-cloud-stride "${DENSE_POINT_CLOUD_STRIDE:-2}" \
  --dense-vis-threshold "${DENSE_VIS_THRESHOLD:-0.3}" \
  --dense-confidence-percentile "${DENSE_CONFIDENCE_PERCENTILE:-10}" \
  --dense-query-batch-size "${DENSE_QUERY_BATCH_SIZE:-16384}" \
  --gif-fps "${GIF_FPS:-8}" \
  --device "${DEVICE:-cuda}" \
  --normalize-3d \
  "$@"
