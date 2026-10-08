#!/usr/bin/env bash
# Run the GPU7 acceleration microbenchmark without disturbing the live training
# on GPUs 0-6. Single device, single process, random tensors only.
#
#   bash tools/bench_accel_gpu7.sh                # run all 7 configs, batch=2
#   ONLY=A_baseline,B_nosync bash tools/...        # run only some
#   BATCH=1 STEPS=20 WARMUP=3 bash tools/...      # custom shape
#
# Output: stdout summary + bench_accel_gpu7.json in repo root.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
GPU_ID="${GPU_ID:-7}"
VARIANT="${VARIANT:-giant}"
BATCH="${BATCH:-2}"
STEPS="${STEPS:-30}"
WARMUP="${WARMUP:-5}"
NUM_FRAMES="${NUM_FRAMES:-48}"
RESOLUTION="${RESOLUTION:-256}"
NUM_QUERIES="${NUM_QUERIES:-2048}"
ONLY="${ONLY:-}"
OUT="${OUT:-bench_accel_gpu7.json}"

if [[ "$VARIANT" == "giant" ]]; then
  ENCODER_PATCH_SIZE=14
elif [[ "$VARIANT" == "huge" ]]; then
  ENCODER_PATCH_SIZE=14
else
  ENCODER_PATCH_SIZE=16
fi

# Avoid colliding with the running training: use only GPU7, pin threading low,
# and write logs to a separate file.
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Block VideoMAE pretrained download — we only need the architecture.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cmd=(
  "$PYTHON_BIN" tools/bench_accel_gpu7.py
  --variant "$VARIANT"
  --batch-size "$BATCH"
  --num-frames "$NUM_FRAMES"
  --resolution "$RESOLUTION"
  --num-queries "$NUM_QUERIES"
  --encoder-patch-size "$ENCODER_PATCH_SIZE"
  --steps "$STEPS"
  --warmup "$WARMUP"
  --out "$OUT"
)
if [[ -n "$ONLY" ]]; then
  cmd+=(--only "$ONLY")
fi

echo "[bench_accel_gpu7] GPU_ID=$GPU_ID VARIANT=$VARIANT BATCH=$BATCH STEPS=$STEPS WARMUP=$WARMUP"
echo "[bench_accel_gpu7] cmd: ${cmd[*]}"
"${cmd[@]}"
