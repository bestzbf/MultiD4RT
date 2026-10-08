#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/d4rt/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
MASTER_PORT="${MASTER_PORT:-28580}"
CONFIG="${CONFIG:-configs/mixture_5datasets_cos_planned.yaml}"

POINTODYSSEY_LOCAL_ROOT="${POINTODYSSEY_LOCAL_ROOT:-/data2/d4rt/datasets/PointOdyssey}"
if [[ -z "${POINTODYSSEY_ROOT:-}" ]]; then
  if [[ -d "$POINTODYSSEY_LOCAL_ROOT" ]]; then
    POINTODYSSEY_ROOT="$POINTODYSSEY_LOCAL_ROOT"
  else
    POINTODYSSEY_ROOT="/data_cos/hdu_datasets/PointOdyssey"
  fi
fi
POINTODYSSEY_FAST_ROOT="${POINTODYSSEY_FAST_ROOT:-}"
POINTODYSSEY_LOCAL_CACHE_DIR="${POINTODYSSEY_LOCAL_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_5datasets_local}"
POINTODYSSEY_REQUIRE_TRACKS="${POINTODYSSEY_REQUIRE_TRACKS:-1}"
POINTODYSSEY_ASSUME_TRACKS="${POINTODYSSEY_ASSUME_TRACKS:-0}"
POINTODYSSEY_TRACK_WORKERS="${POINTODYSSEY_TRACK_WORKERS:-16}"
KUBRIC_ROOT="${KUBRIC_ROOT:-/data_cos/hdu_datasets/Kubric}"
DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/data_cos/hdu_datasets/Dynamic_Replica}"
CO3DV2_ROOT="${CO3DV2_ROOT:-/data_cos/hdu_datasets/Co3Dv2}"
BLENDEDMVS_ROOT="${BLENDEDMVS_ROOT:-/data2/d4rt/datasets/BlendedMVS}"
MVSSYNTH_ROOT="${MVSSYNTH_ROOT:-/data2/d4rt/datasets/MVS-Synth/GTAV_1080}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/data_cos/hdu_datasets/scannetpp/data}"
SCANNETPP_SPLITS_DIR="${SCANNETPP_SPLITS_DIR:-/data_cos/hdu_datasets/scannetpp/splits}"
SCANNETPP_SCENES_RECORD="${SCANNETPP_SCENES_RECORD:-/data_cos/hdu_datasets/scannetpp/scenes_record.json}"
CO3DV2_DENYLIST="${CO3DV2_DENYLIST:-/data/zbf/openclaw/d4rt/configs/co3dv2_denylist_degenerate_clips_20260422.txt}"

INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_5datasets_local}"
TMPDIR="${TMPDIR:-/data1/zbf/d4rt_tmp}"
MAX_SPOOL_BYTES_GB="${MAX_SPOOL_BYTES_GB:-100}"
BUILDER_WORKERS="${BUILDER_WORKERS:-8}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-256}"
BATCH_PREFETCH_DEPTH="${BATCH_PREFETCH_DEPTH:-4}"
SAMPLE_STAGE_BACKEND="${SAMPLE_STAGE_BACKEND:-cos_sdk}"
SAMPLE_STAGE_SDK_WORKERS="${SAMPLE_STAGE_SDK_WORKERS:-32}"
SAMPLE_STAGE_CACHE_MAX_GB="${SAMPLE_STAGE_CACHE_MAX_GB:-100}"
SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO="${SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO:-0.9}"
SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S="${SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S:-30}"
SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S="${SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S:-30}"
SAMPLE_STAGE_WINDOW_RADIUS="${SAMPLE_STAGE_WINDOW_RADIUS:-0}"
SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS:-pointodyssey,kubric,dynamic_replica,co3dv2,scannetpp}"
SAMPLE_STAGE_SCENE_PREFETCH_DATASETS="${SAMPLE_STAGE_SCENE_PREFETCH_DATASETS:-}"
SAMPLE_STAGE_MOUNT_ROOT="${SAMPLE_STAGE_MOUNT_ROOT:-/data_cos}"
SAMPLE_STAGE_BUCKET="${SAMPLE_STAGE_BUCKET:-hd-ai-data-1251882982}"
SAMPLE_STAGE_REGION="${SAMPLE_STAGE_REGION:-ap-beijing}"
SAMPLE_STAGE_PASSWD_FILE="${SAMPLE_STAGE_PASSWD_FILE:-/etc/passwd-s3fs-data_cos}"

BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
DATA_WAIT_THRESHOLD_S="${DATA_WAIT_THRESHOLD_S:-2.0}"
DATA_WAIT_DETAIL_MAX_SAMPLES="${DATA_WAIT_DETAIL_MAX_SAMPLES:-8}"
START_EPOCH="${START_EPOCH:-398}"
TEST_BATCHES="${TEST_BATCHES:-50}"
WARMUP_BATCHES="${WARMUP_BATCHES:-3}"
EPOCHS="${EPOCHS:-1}"
SIMULATE_COMPUTE_MS="${SIMULATE_COMPUTE_MS:-1500}"
STARTUP_SLEEP_S="${STARTUP_SLEEP_S:-15}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-${TMPDIR}/d4rt_dataload_probe.${RUN_ID}}"
SAMPLE_STAGE_ROOT="${SAMPLE_STAGE_ROOT:-${TMPDIR}/d4rt_dataload_probe_stage}"
SPOOL_ROOT="${SPOOL_ROOT:-${REPORT_DIR}/spool}"
DRY_RUN="${DRY_RUN:-0}"

AUTO_WARM_INDEX_CACHE="${AUTO_WARM_INDEX_CACHE:-0}"
WARM_CACHE_ONLY="${WARM_CACHE_ONLY:-0}"
WARM_ONLY_DATASETS="${WARM_ONLY_DATASETS:-}"
WARM_INDEX_WORKERS="${WARM_INDEX_WORKERS:-16}"
WARM_VAL="${WARM_VAL:-0}"

D4RT_SUPPRESS_MKL_WARNING="${D4RT_SUPPRESS_MKL_WARNING:-1}"
D4RT_BUILDER_FAULTHANDLER="${D4RT_BUILDER_FAULTHANDLER:-0}"
D4RT_BUILD_TIMEOUT="${D4RT_BUILD_TIMEOUT:-100}"
D4RT_PROFILE_BUILDER="${D4RT_PROFILE_BUILDER:-1}"
D4RT_PROFILE_BUILDER_ALL="${D4RT_PROFILE_BUILDER_ALL:-0}"
D4RT_VERBOSE_BUILDER="${D4RT_VERBOSE_BUILDER:-0}"
D4RT_PLANNED_WAIT_LOG="${D4RT_PLANNED_WAIT_LOG:-1}"
D4RT_SLOW_SAMPLE_THRESHOLD_S="${D4RT_SLOW_SAMPLE_THRESHOLD_S:-20.0}"

mkdir -p "$TMPDIR" "$INDEX_CACHE_DIR" "$REPORT_DIR"
if [[ -n "$SAMPLE_STAGE_BACKEND" ]]; then
  mkdir -p "$SAMPLE_STAGE_ROOT"
fi
mkdir -p "$SPOOL_ROOT"

export CUDA_VISIBLE_DEVICES=""
export TMPDIR
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export D4RT_SERIALIZE_ADAPTER_INIT="${D4RT_SERIALIZE_ADAPTER_INIT:-1}"
export D4RT_SCANNETPP_TRUST_ALLOWLIST="${D4RT_SCANNETPP_TRUST_ALLOWLIST:-1}"
export D4RT_BUILDER_FAULTHANDLER
export D4RT_VERBOSE_BUILDER
export D4RT_BUILD_TIMEOUT
export D4RT_PROFILE_BUILDER
export D4RT_PROFILE_BUILDER_ALL
export D4RT_PLANNED_WAIT_LOG
export D4RT_SLOW_SAMPLE_THRESHOLD_S
export D4RT_POINTODYSSEY_STAGE_ANNO_H5="${D4RT_POINTODYSSEY_STAGE_ANNO_H5:-0}"
export D4RT_POINTODYSSEY_ANNO_FRAME_CACHE_DIR="${POINTODYSSEY_ANNO_FRAME_CACHE_DIR:-}"
export D4RT_BLENDEDMVS_DEPTH_CACHE_DIR="${BLENDEDMVS_DEPTH_CACHE_DIR:-}"
export MAX_SPOOL_BYTES_GB

if [[ "$D4RT_SUPPRESS_MKL_WARNING" == "1" ]]; then
  mkl_warning_filter="ignore:mkl-service package failed to import:UserWarning"
  if [[ -n "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="${mkl_warning_filter},${PYTHONWARNINGS}"
  else
    export PYTHONWARNINGS="$mkl_warning_filter"
  fi
fi

echo "[dataload_probe] NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[dataload_probe] MASTER_PORT=$MASTER_PORT"
echo "[dataload_probe] CONFIG=$CONFIG"
echo "[dataload_probe] START_EPOCH=$START_EPOCH"
echo "[dataload_probe] TEST_BATCHES=$TEST_BATCHES"
echo "[dataload_probe] WARMUP_BATCHES=$WARMUP_BATCHES"
echo "[dataload_probe] SIMULATE_COMPUTE_MS=$SIMULATE_COMPUTE_MS"
echo "[dataload_probe] STARTUP_SLEEP_S=$STARTUP_SLEEP_S"
echo "[dataload_probe] REPORT_DIR=$REPORT_DIR"
echo "[dataload_probe] SPOOL_ROOT=$SPOOL_ROOT"
echo "[dataload_probe] DRY_RUN=$DRY_RUN"
echo "[dataload_probe] CUDA_VISIBLE_DEVICES=<empty>"
echo "[dataload_probe] SAMPLE_STAGE_BACKEND=$SAMPLE_STAGE_BACKEND"
echo "[dataload_probe] SAMPLE_STAGE_ROOT=$SAMPLE_STAGE_ROOT"
echo "[dataload_probe] SAMPLE_STAGE_SDK_WORKERS=$SAMPLE_STAGE_SDK_WORKERS"
echo "[dataload_probe] SAMPLE_STAGE_WINDOW_RADIUS=$SAMPLE_STAGE_WINDOW_RADIUS"
echo "[dataload_probe] SAMPLE_STAGE_DATASETS=$SAMPLE_STAGE_DATASETS"
echo "[dataload_probe] BUILDER_WORKERS=$BUILDER_WORKERS"
echo "[dataload_probe] PREFETCH_DEPTH=$PREFETCH_DEPTH"
echo "[dataload_probe] BATCH_PREFETCH_DEPTH=$BATCH_PREFETCH_DEPTH"
echo "[dataload_probe] MAX_SPOOL_BYTES_GB=$MAX_SPOOL_BYTES_GB"

if [[ ! -f "$CONFIG" ]]; then
  echo "[dataload_probe] CONFIG not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CO3DV2_DENYLIST" ]]; then
  echo "[dataload_probe] CO3DV2_DENYLIST not found: $CO3DV2_DENYLIST" >&2
  exit 1
fi
if [[ ! -f "$PYTHON_BIN" ]]; then
  echo "[dataload_probe] PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TORCHRUN_BIN" ]]; then
  echo "[dataload_probe] TORCHRUN_BIN not found: $TORCHRUN_BIN" >&2
  exit 1
fi

TEMP_CONFIG=""
cleanup() {
  if [[ -n "$TEMP_CONFIG" && -f "$TEMP_CONFIG" ]]; then
    rm -f "$TEMP_CONFIG"
  fi
}
trap cleanup EXIT

if [[ "$AUTO_WARM_INDEX_CACHE" == "1" || "$WARM_CACHE_ONLY" == "1" ]]; then
  if [[ ! -f "$ROOT_DIR/tools/warm_index_cache_5datasets_cos.sh" ]]; then
    echo "[dataload_probe] warm script not found: $ROOT_DIR/tools/warm_index_cache_5datasets_cos.sh" >&2
    exit 1
  fi
  echo "[dataload_probe] warming index cache before benchmark"
  CONFIG="$CONFIG" \
  INDEX_CACHE_DIR="$INDEX_CACHE_DIR" \
  POINTODYSSEY_ROOT="$POINTODYSSEY_ROOT" \
  POINTODYSSEY_LOCAL_ROOT="$POINTODYSSEY_LOCAL_ROOT" \
  POINTODYSSEY_LOCAL_CACHE_DIR="$POINTODYSSEY_LOCAL_CACHE_DIR" \
  POINTODYSSEY_REQUIRE_TRACKS="$POINTODYSSEY_REQUIRE_TRACKS" \
  POINTODYSSEY_ASSUME_TRACKS="$POINTODYSSEY_ASSUME_TRACKS" \
  POINTODYSSEY_TRACK_WORKERS="$POINTODYSSEY_TRACK_WORKERS" \
  KUBRIC_ROOT="$KUBRIC_ROOT" \
  DYNAMIC_REPLICA_ROOT="$DYNAMIC_REPLICA_ROOT" \
  CO3DV2_ROOT="$CO3DV2_ROOT" \
  BLENDEDMVS_ROOT="$BLENDEDMVS_ROOT" \
  MVSSYNTH_ROOT="$MVSSYNTH_ROOT" \
  SCANNETPP_ROOT="$SCANNETPP_ROOT" \
  SCANNETPP_SPLITS_DIR="$SCANNETPP_SPLITS_DIR" \
  SCANNETPP_SCENES_RECORD="$SCANNETPP_SCENES_RECORD" \
  CO3DV2_DENYLIST="$CO3DV2_DENYLIST" \
  INDEX_WORKERS="$WARM_INDEX_WORKERS" \
  WARM_VAL="$WARM_VAL" \
  ONLY_DATASETS="$WARM_ONLY_DATASETS" \
  bash "$ROOT_DIR/tools/warm_index_cache_5datasets_cos.sh"
fi

if [[ "$WARM_CACHE_ONLY" == "1" ]]; then
  echo "[dataload_probe] WARM_CACHE_ONLY=1, stop after warmup"
  exit 0
fi

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/d4rt_dataload_probe.XXXXXX.yaml")"
"$PYTHON_BIN" - "$CONFIG" "$TEMP_CONFIG" "$POINTODYSSEY_ROOT" "$POINTODYSSEY_FAST_ROOT" "$KUBRIC_ROOT" "$DYNAMIC_REPLICA_ROOT" "$CO3DV2_ROOT" "$BLENDEDMVS_ROOT" "$MVSSYNTH_ROOT" "$SCANNETPP_ROOT" "$SCANNETPP_SPLITS_DIR" "$SCANNETPP_SCENES_RECORD" "$INDEX_CACHE_DIR" "$CO3DV2_DENYLIST" "$BUILDER_WORKERS" "$PREFETCH_DEPTH" "$SAMPLE_STAGE_BACKEND" "$SAMPLE_STAGE_ROOT" "$SAMPLE_STAGE_SDK_WORKERS" "$SAMPLE_STAGE_DATASETS" "$SAMPLE_STAGE_SCENE_PREFETCH_DATASETS" "$SAMPLE_STAGE_MOUNT_ROOT" "$SAMPLE_STAGE_BUCKET" "$SAMPLE_STAGE_REGION" "$SAMPLE_STAGE_PASSWD_FILE" "$SAMPLE_STAGE_CACHE_MAX_GB" "$SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO" "$SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S" "$SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S" "$SAMPLE_STAGE_WINDOW_RADIUS" <<'PY'
import os
from pathlib import Path
import sys

import yaml

src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])
pointodyssey_root = sys.argv[3]
pointodyssey_fast_root = sys.argv[4]
kubric_root = sys.argv[5]
dynamic_replica_root = sys.argv[6]
co3dv2_root = sys.argv[7]
blendedmvs_root = sys.argv[8]
mvssynth_root = sys.argv[9]
scannetpp_root = sys.argv[10]
scannetpp_splits_dir = sys.argv[11]
scannetpp_scenes_record = sys.argv[12]
index_cache_dir = sys.argv[13]
co3dv2_denylist = sys.argv[14]
builder_workers = int(sys.argv[15])
prefetch_depth = int(sys.argv[16])
sample_stage_backend = sys.argv[17].strip()
sample_stage_root = sys.argv[18].strip()
sample_stage_sdk_workers = int(sys.argv[19])
sample_stage_datasets = sys.argv[20].strip()
sample_stage_scene_prefetch_datasets = sys.argv[21].strip()
sample_stage_mount_root = sys.argv[22].strip()
sample_stage_bucket = sys.argv[23].strip()
sample_stage_region = sys.argv[24].strip()
sample_stage_passwd_file = sys.argv[25].strip()
sample_stage_cache_max_bytes = int(float(sys.argv[26]) * 1024**3)
sample_stage_cache_low_watermark_ratio = float(sys.argv[27])
sample_stage_cache_touch_interval_s = float(sys.argv[28])
sample_stage_cache_scan_interval_s = float(sys.argv[29])
sample_stage_window_radius = int(sys.argv[30])

required_markers = {
    "pointodyssey": ("train",),
    "kubric": (),
    "dynamic_replica": ("train",),
    "co3dv2": ("apple",),
    "blendedmvs": ("BlendedMVS_training.txt",),
    "mvssynth": (),
    "scannetpp": (),
}


def validate_root(dataset_name: str, root: str) -> None:
    path = Path(root)
    if not path.is_dir():
        raise SystemExit(
            f"[dataload_probe] {dataset_name}.root not found or not a directory: {root}"
        )
    for marker in required_markers.get(dataset_name, ()):
        if not (path / marker).exists():
            raise SystemExit(
                f"[dataload_probe] {dataset_name}.root is incomplete: missing {path / marker}"
            )


config = yaml.safe_load(src_path.read_text())
datasets = {item["name"]: item for item in config["datasets"]}

root_map = {
    "pointodyssey": pointodyssey_root,
    "kubric": kubric_root,
    "dynamic_replica": dynamic_replica_root,
    "co3dv2": co3dv2_root,
    "blendedmvs": blendedmvs_root,
    "mvssynth": mvssynth_root,
    "scannetpp": scannetpp_root,
}

for dataset_name, root in root_map.items():
    if dataset_name not in datasets:
        continue
    validate_root(dataset_name, root)
    datasets[dataset_name]["root"] = root

if "scannetpp" in datasets:
    datasets["scannetpp"].setdefault("adapter_kwargs", {})["splits_dir"] = scannetpp_splits_dir
    datasets["scannetpp"].setdefault("adapter_kwargs", {})["scenes_record"] = scannetpp_scenes_record
    datasets["scannetpp"].setdefault("adapter_kwargs", {})["strict"] = False

if "mvssynth" in datasets:
    mvssynth_cfg = datasets["mvssynth"].setdefault("adapter_kwargs", {})
    mvssynth_cfg["precompute_root"] = mvssynth_root
    mvssynth_cfg["load_precomputed"] = True
    mvssynth_cfg["load_normals"] = False
    mvssynth_cfg["io_workers"] = int(os.environ.get("D4RT_MVSSYNTH_IO_WORKERS", "4"))
    mvssynth_cfg["io_read_retries"] = int(os.environ.get("D4RT_MVSSYNTH_IO_READ_RETRIES", "3"))
    mvssynth_cfg["io_retry_delay_s"] = float(os.environ.get("D4RT_MVSSYNTH_IO_RETRY_DELAY_S", "0.05"))

if pointodyssey_fast_root:
    fast_root = Path(pointodyssey_fast_root)
    if not fast_root.is_dir():
        raise SystemExit(
            f"[dataload_probe] pointodyssey.fast_root not found or not a directory: {pointodyssey_fast_root}"
        )
    datasets["pointodyssey"].setdefault("adapter_kwargs", {})["fast_root"] = pointodyssey_fast_root
else:
    datasets["pointodyssey"].setdefault("adapter_kwargs", {}).pop("fast_root", None)

try:
    point_root_path = Path(pointodyssey_root)
    stage_mount_path = Path(sample_stage_mount_root)
    point_on_stage_mount = point_root_path.is_relative_to(stage_mount_path)
except Exception:
    point_on_stage_mount = False
if point_on_stage_mount:
    datasets["pointodyssey"].setdefault("adapter_kwargs", {})["runtime_sanitize"] = False
else:
    datasets["pointodyssey"].setdefault("adapter_kwargs", {}).pop("runtime_sanitize", None)

datasets["co3dv2"].setdefault("adapter_kwargs", {})["sequence_denylist"] = co3dv2_denylist

config["index_cache_dir"] = index_cache_dir
config["planned_mode"] = True
config["builder_workers"] = builder_workers
config["prefetch_depth"] = prefetch_depth
config["max_spool_bytes"] = int(float(os.environ.get("MAX_SPOOL_BYTES_GB", "100")) * 1024**3)
config["sample_stage_backend"] = sample_stage_backend
config["sample_stage_root"] = sample_stage_root
config["sample_stage_sdk_workers"] = sample_stage_sdk_workers
config["sample_stage_datasets"] = [item.strip() for item in sample_stage_datasets.split(",") if item.strip()]
config["sample_stage_scene_prefetch_datasets"] = [item.strip() for item in sample_stage_scene_prefetch_datasets.split(",") if item.strip()]
config["sample_stage_mount_root"] = sample_stage_mount_root
config["sample_stage_bucket"] = sample_stage_bucket
config["sample_stage_region"] = sample_stage_region
config["sample_stage_passwd_file"] = sample_stage_passwd_file
config["sample_stage_cache_max_bytes"] = sample_stage_cache_max_bytes
config["sample_stage_cache_low_watermark_ratio"] = sample_stage_cache_low_watermark_ratio
config["sample_stage_cache_touch_interval_s"] = sample_stage_cache_touch_interval_s
config["sample_stage_cache_scan_interval_s"] = sample_stage_cache_scan_interval_s
config["sample_stage_window_radius"] = sample_stage_window_radius

dst_path.write_text(yaml.safe_dump(config, sort_keys=False))
PY

echo "[dataload_probe] EFFECTIVE_CONFIG=$TEMP_CONFIG"

cmd=(
  "$TORCHRUN_BIN" \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  scripts/benchmark_planned_dataloader.py \
  --config "$TEMP_CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --planned-mode \
  --builder-workers "$BUILDER_WORKERS" \
  --prefetch-depth "$PREFETCH_DEPTH" \
  --batch-prefetch-depth "$BATCH_PREFETCH_DEPTH" \
  --batches "$TEST_BATCHES" \
  --warmup-batches "$WARMUP_BATCHES" \
  --start-epoch "$START_EPOCH" \
  --epochs "$EPOCHS" \
  --simulate-compute-ms "$SIMULATE_COMPUTE_MS" \
  --startup-sleep-s "$STARTUP_SLEEP_S" \
  --log-interval "$LOG_INTERVAL" \
  --data-wait-threshold-s "$DATA_WAIT_THRESHOLD_S" \
  --detail-max-samples "$DATA_WAIT_DETAIL_MAX_SAMPLES" \
  --report-dir "$REPORT_DIR" \
  --spool-root "$SPOOL_ROOT" \
  --profile-collate
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[dataload_probe] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
