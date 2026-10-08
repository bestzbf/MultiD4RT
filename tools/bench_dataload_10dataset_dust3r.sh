#!/usr/bin/env bash
# CPU-only data-loading benchmark for the 10-dataset dust3r/vitg run.
#
# Unlike tools/test_mixture_5datasets_3gpu_cos_planned_dataload.sh (stale: only
# injects 7 dataset roots), this reuses the *effective config that the live
# training already generated* as ground truth, so every root / sample_stage /
# builder param matches the running job exactly. No model, no CUDA, no
# forward/backward -- per-batch data_wait is pure data-loading cost.
#
# Defaults: 2 ranks, reuse the warm stage cache, simulate the real ~1900ms GPU
# step so the spool drains at the realistic rate. Spool is isolated from the
# live training's spool dir.
#
# Usage:
#   bash tools/bench_dataload_10dataset_dust3r.sh
# Override anything via env, e.g. NPROC_PER_NODE=6 TEST_BATCHES=100 ...

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/d4rt/bin/torchrun}"

# Ground-truth config: the snapshot copied from the live training's effective
# config. Override CONFIG to point at a freshly generated one if training was
# restarted with different roots.
CONFIG="${CONFIG:-/data1/zbf/d4rt_tmp/dataload_bench_effective.yaml}"

# Data params -- mirror the dust3r/vitg run exactly.
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-28599}"          # != 28580 (live training's port)
BATCH_SIZE="${BATCH_SIZE:-14}"
BUILDER_WORKERS="${BUILDER_WORKERS:-18}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-960}"
BATCH_PREFETCH_DEPTH="${BATCH_PREFETCH_DEPTH:-24}"
START_EPOCH="${START_EPOCH:-115}"            # matches resume ckpt_115 -> realistic planned order

# Benchmark loop knobs.
TEST_BATCHES="${TEST_BATCHES:-80}"
WARMUP_BATCHES="${WARMUP_BATCHES:-5}"
EPOCHS="${EPOCHS:-1}"
SIMULATE_COMPUTE_MS="${SIMULATE_COMPUTE_MS:-1900}"   # ~live GPU step total
STARTUP_SLEEP_S="${STARTUP_SLEEP_S:-20}"             # let builders prefill spool
LOG_INTERVAL="${LOG_INTERVAL:-10}"
DATA_WAIT_THRESHOLD_S="${DATA_WAIT_THRESHOLD_S:-2.0}"
DATA_WAIT_DETAIL_MAX_SAMPLES="${DATA_WAIT_DETAIL_MAX_SAMPLES:-8}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TMPDIR="${TMPDIR:-/data1/zbf/d4rt_tmp}"
REPORT_DIR="${REPORT_DIR:-${TMPDIR}/d4rt_dataload_bench.${RUN_ID}}"
# Isolated spool in shm; does NOT collide with the live training spool.
SPOOL_ROOT="${SPOOL_ROOT:-/dev/shm/d4rt_dataload_bench_${RUN_ID}}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$TMPDIR" "$REPORT_DIR" "$SPOOL_ROOT"

# --- env: no GPU, builder profiling on, mirror training's stage env ---
export CUDA_VISIBLE_DEVICES=""
export TMPDIR
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export D4RT_SERIALIZE_ADAPTER_INIT="${D4RT_SERIALIZE_ADAPTER_INIT:-1}"
export D4RT_SCANNETPP_TRUST_ALLOWLIST="${D4RT_SCANNETPP_TRUST_ALLOWLIST:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Builder + stage profiling so we can see per-dataset slow samples.
export D4RT_PROFILE_BUILDER="${D4RT_PROFILE_BUILDER:-1}"
export D4RT_PROFILE_BUILDER_ALL="${D4RT_PROFILE_BUILDER_ALL:-0}"
export D4RT_PROFILE_BUILDER_THRESHOLD_S="${D4RT_PROFILE_BUILDER_THRESHOLD_S:-3.0}"
export D4RT_SAMPLE_STAGE_PROFILE="${D4RT_SAMPLE_STAGE_PROFILE:-0}"
export D4RT_SAMPLE_STAGE_SLOW_THRESHOLD_S="${D4RT_SAMPLE_STAGE_SLOW_THRESHOLD_S:-3.0}"
export D4RT_SAMPLE_STAGE_SLOW_DOWNLOAD_S="${D4RT_SAMPLE_STAGE_SLOW_DOWNLOAD_S:-2.0}"
export D4RT_PLANNED_WAIT_LOG="${D4RT_PLANNED_WAIT_LOG:-0}"
export D4RT_MAX_TRACK_POINTS="${D4RT_MAX_TRACK_POINTS:-4096}"
# Match training's planned relaxed-order knobs so consumption order is realistic.
export D4RT_PLANNED_RELAXED_ORDER="${D4RT_PLANNED_RELAXED_ORDER:-1}"
export D4RT_PLANNED_RELAXED_LOOKAHEAD="${D4RT_PLANNED_RELAXED_LOOKAHEAD:-1600}"
export D4RT_PLANNED_RELAXED_GRACE_S="${D4RT_PLANNED_RELAXED_GRACE_S:-0.25}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[bench_dataload] CONFIG not found: $CONFIG" >&2
  echo "[bench_dataload] regenerate it by copying the live training effective config:" >&2
  echo "[bench_dataload]   ls -t /data1/zbf/d4rt_tmp/mixture_5datasets_cos_planned.*.yaml | head -1" >&2
  exit 1
fi
[[ -f "$PYTHON_BIN" ]]   || { echo "[bench_dataload] PYTHON_BIN not found: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$TORCHRUN_BIN" ]] || { echo "[bench_dataload] TORCHRUN_BIN not found: $TORCHRUN_BIN" >&2; exit 1; }

echo "[bench_dataload] CONFIG=$CONFIG"
echo "[bench_dataload] NPROC_PER_NODE=$NPROC_PER_NODE  BATCH_SIZE=$BATCH_SIZE"
echo "[bench_dataload] BUILDER_WORKERS=$BUILDER_WORKERS  PREFETCH_DEPTH=$PREFETCH_DEPTH  BATCH_PREFETCH_DEPTH=$BATCH_PREFETCH_DEPTH"
echo "[bench_dataload] START_EPOCH=$START_EPOCH  TEST_BATCHES=$TEST_BATCHES  WARMUP=$WARMUP_BATCHES"
echo "[bench_dataload] SIMULATE_COMPUTE_MS=$SIMULATE_COMPUTE_MS  STARTUP_SLEEP_S=$STARTUP_SLEEP_S"
echo "[bench_dataload] SPOOL_ROOT=$SPOOL_ROOT  REPORT_DIR=$REPORT_DIR"
echo "[bench_dataload] stage cache (reused warm): $(grep -E '^sample_stage_root:' "$CONFIG" | awk '{print $2}')"
echo "[bench_dataload] CUDA_VISIBLE_DEVICES=<empty>  (no GPU, no model)"

cmd=(
  "$TORCHRUN_BIN"
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  scripts/benchmark_planned_dataloader.py
  --config "$CONFIG"
  --batch-size "$BATCH_SIZE"
  --planned-mode
  --builder-workers "$BUILDER_WORKERS"
  --prefetch-depth "$PREFETCH_DEPTH"
  --batch-prefetch-depth "$BATCH_PREFETCH_DEPTH"
  --batches "$TEST_BATCHES"
  --warmup-batches "$WARMUP_BATCHES"
  --start-epoch "$START_EPOCH"
  --epochs "$EPOCHS"
  --simulate-compute-ms "$SIMULATE_COMPUTE_MS"
  --startup-sleep-s "$STARTUP_SLEEP_S"
  --log-interval "$LOG_INTERVAL"
  --data-wait-threshold-s "$DATA_WAIT_THRESHOLD_S"
  --detail-max-samples "$DATA_WAIT_DETAIL_MAX_SAMPLES"
  --report-dir "$REPORT_DIR"
  --spool-root "$SPOOL_ROOT"
  --profile-collate
  --no-pin-memory
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[bench_dataload] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
  exit 0
fi

cleanup() { rm -rf "$SPOOL_ROOT" 2>/dev/null || true; }
trap cleanup EXIT

"${cmd[@]}"
