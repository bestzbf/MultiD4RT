#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

INPUT_PATH="${INPUT_PATH:-${VIDEO:-/data1/d4rt/datasets/Dynamic_Replica/test/01f258-3_obj_source_left/images}}"
if [[ $# -gt 0 ]]; then
  INPUT_PATH="$1"
  shift
fi

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: bash tools/run_visualize_video_checkpoint.sh /path/to/video.mp4_or_image_dir [extra args...]" >&2
  echo "   or: INPUT_PATH=/path/to/video.mp4_or_image_dir bash tools/run_visualize_video_checkpoint.sh" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT="${CHECKPOINT:-/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200/checkpoint_latest_466.pth}"
MODEL_VARIANT="${MODEL_VARIANT:-auto}"
INPUT_BASENAME="$(basename "$INPUT_PATH")"
INPUT_STEM="${INPUT_BASENAME%.*}"
if [[ -d "$INPUT_PATH" && "$INPUT_BASENAME" == "images" ]]; then
  INPUT_STEM="$(basename "$(dirname "$INPUT_PATH")")"
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/video_checkpoint_vis/${INPUT_STEM}}"

PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_resized}"
RESOLUTION="${RESOLUTION:-256}"
NUM_FRAMES="${NUM_FRAMES:-48}"
SAMPLING_MODE="${SAMPLING_MODE:-contiguous}"
START_INDEX="${START_INDEX:--1}"
QUERY_GRID_SIZE="${QUERY_GRID_SIZE:-32}"
QUERY_MARGIN="${QUERY_MARGIN:-0.06}"
NUM_DISPLAY_POINTS="${NUM_DISPLAY_POINTS:-180}"
TRACK_TAIL="${TRACK_TAIL:-12}"
DENSE_POINT_CLOUD_STRIDE="${DENSE_POINT_CLOUD_STRIDE:-2}"
DENSE_VIS_THRESHOLD="${DENSE_VIS_THRESHOLD:-0.3}"
DENSE_CONFIDENCE_PERCENTILE="${DENSE_CONFIDENCE_PERCENTILE:-10}"
DENSE_QUERY_BATCH_SIZE="${DENSE_QUERY_BATCH_SIZE:-4096}"
DENSE_MOTION_PERCENTILE="${DENSE_MOTION_PERCENTILE:-85}"
GIF_FPS="${GIF_FPS:-8}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
FLIP_Y_AXIS="${FLIP_Y_AXIS:-0}"
NORMALIZE_3D="${NORMALIZE_3D:-1}"

echo "[vis_video] PYTHON_BIN=$PYTHON_BIN"
echo "[vis_video] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[vis_video] INPUT_PATH=$INPUT_PATH"
echo "[vis_video] CHECKPOINT=$CHECKPOINT"
echo "[vis_video] MODEL_VARIANT=$MODEL_VARIANT"
echo "[vis_video] OUTPUT_DIR=$OUTPUT_DIR"
echo "[vis_video] PATCH_PROVIDER=$PATCH_PROVIDER"
echo "[vis_video] RESOLUTION=$RESOLUTION"
echo "[vis_video] NUM_FRAMES=$NUM_FRAMES"
echo "[vis_video] SAMPLING_MODE=$SAMPLING_MODE"
echo "[vis_video] START_INDEX=$START_INDEX"
echo "[vis_video] QUERY_GRID_SIZE=$QUERY_GRID_SIZE"
echo "[vis_video] QUERY_MARGIN=$QUERY_MARGIN"
echo "[vis_video] NUM_DISPLAY_POINTS=$NUM_DISPLAY_POINTS"
echo "[vis_video] TRACK_TAIL=$TRACK_TAIL"
echo "[vis_video] DENSE_POINT_CLOUD_STRIDE=$DENSE_POINT_CLOUD_STRIDE"
echo "[vis_video] DENSE_VIS_THRESHOLD=$DENSE_VIS_THRESHOLD"
echo "[vis_video] DENSE_CONFIDENCE_PERCENTILE=$DENSE_CONFIDENCE_PERCENTILE"
echo "[vis_video] DENSE_QUERY_BATCH_SIZE=$DENSE_QUERY_BATCH_SIZE"
echo "[vis_video] DENSE_MOTION_PERCENTILE=$DENSE_MOTION_PERCENTILE"
echo "[vis_video] GIF_FPS=$GIF_FPS"
echo "[vis_video] DEVICE=$DEVICE"
echo "[vis_video] FLIP_Y_AXIS=$FLIP_Y_AXIS"
echo "[vis_video] NORMALIZE_3D=$NORMALIZE_3D"

if [[ ! -x "$PYTHON_BIN" && ! -f "$PYTHON_BIN" ]]; then
  echo "[vis_video] Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -e "$INPUT_PATH" ]]; then
  echo "[vis_video] Input path not found: $INPUT_PATH" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[vis_video] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

ARGS=(
  md/visualize_video_checkpoint.py
  --input-path "$INPUT_PATH"
  --checkpoint "$CHECKPOINT"
  --model-variant "$MODEL_VARIANT"
  --output-dir "$OUTPUT_DIR"
  --patch-provider "$PATCH_PROVIDER"
  --resolution "$RESOLUTION"
  --num-frames "$NUM_FRAMES"
  --sampling-mode "$SAMPLING_MODE"
  --start-index "$START_INDEX"
  --query-grid-size "$QUERY_GRID_SIZE"
  --query-margin "$QUERY_MARGIN"
  --num-display-points "$NUM_DISPLAY_POINTS"
  --track-tail "$TRACK_TAIL"
  --dense-point-cloud-stride "$DENSE_POINT_CLOUD_STRIDE"
  --dense-vis-threshold "$DENSE_VIS_THRESHOLD"
  --dense-confidence-percentile "$DENSE_CONFIDENCE_PERCENTILE"
  --dense-query-batch-size "$DENSE_QUERY_BATCH_SIZE"
  --dense-motion-percentile "$DENSE_MOTION_PERCENTILE"
  --gif-fps "$GIF_FPS"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ "$FLIP_Y_AXIS" == "1" ]]; then
  ARGS+=(--flip-y-axis)
fi

if [[ "$NORMALIZE_3D" == "1" ]]; then
  ARGS+=(--normalize-3d)
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "${ARGS[@]}" "$@"
