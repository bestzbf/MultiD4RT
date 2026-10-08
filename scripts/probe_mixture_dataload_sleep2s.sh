#!/usr/bin/env bash
set -euo pipefail

# CPU-only planned-mode dataloader probe.
# It uses the same planned dataloader path as training, then sleeps after each
# batch to simulate GPU compute. No model is created and CUDA is hidden.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/d4rt/bin/torchrun}"
TMPDIR="${TMPDIR:-/data1/zbf/d4rt_tmp}"
TEMP_CONFIG=""
SAMPLE_STAGE_JANITOR_PID=""
ROLLING_WARM_PID=""
cleanup() {
  if [[ -n "$ROLLING_WARM_PID" ]]; then
    kill "$ROLLING_WARM_PID" 2>/dev/null || true
    wait "$ROLLING_WARM_PID" 2>/dev/null || true
  fi
  if [[ -n "$SAMPLE_STAGE_JANITOR_PID" ]]; then
    kill "$SAMPLE_STAGE_JANITOR_PID" 2>/dev/null || true
    wait "$SAMPLE_STAGE_JANITOR_PID" 2>/dev/null || true
  fi
  if [[ -n "$TEMP_CONFIG" && -f "$TEMP_CONFIG" ]]; then
    rm -f "$TEMP_CONFIG"
  fi
}
trap cleanup EXIT

CONFIG="${CONFIG:-latest-effective}"
if [[ "$CONFIG" == "latest-effective" ]]; then
  CONFIG="$(
    "$PYTHON_BIN" - "$TMPDIR" <<'PY'
from pathlib import Path
import sys

tmpdir = Path(sys.argv[1])
candidates = sorted(
    tmpdir.glob("mixture_5datasets_cos_planned.*.yaml"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not candidates:
    raise SystemExit(f"no effective config under {tmpdir}")
print(candidates[0])
PY
  )"
fi

COMPACT_SAMPLES="${COMPACT_SAMPLES:-1}"
PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_resized}"
if [[ -z "${PRECOMPUTE_PATCHES+x}" ]]; then
  case "$PATCH_PROVIDER" in
    sampled_resized|sampled_highres) PRECOMPUTE_PATCHES=0 ;;
    *) PRECOMPUTE_PATCHES=1 ;;
  esac
fi
if [[ -z "${PRECOMPUTE_FROM_HIGHRES+x}" ]]; then
  case "$PATCH_PROVIDER" in
    precomputed_highres) PRECOMPUTE_FROM_HIGHRES=1 ;;
    *) PRECOMPUTE_FROM_HIGHRES=0 ;;
  esac
fi
RETURN_HIGHRES_VIDEO="${RETURN_HIGHRES_VIDEO:-}"
STORE_VIDEO_UINT8="${STORE_VIDEO_UINT8:-1}"
STORE_AUXILIARY_TENSORS="${STORE_AUXILIARY_TENSORS:-0}"
LOAD_NORMALS="${LOAD_NORMALS:-0}"
USE_MOTION_BOUNDARIES="${USE_MOTION_BOUNDARIES:-1}"
DATASET_LOCALITY_SIZE="${DATASET_LOCALITY_SIZE:-}"
SEQUENCE_LOCALITY_SIZE="${SEQUENCE_LOCALITY_SIZE:-}"
FRAME_LOCALITY_RADIUS="${FRAME_LOCALITY_RADIUS:-}"
if [[ -z "${COLOR_AUG_AFTER_RESIZE+x}" ]]; then
  case "$PATCH_PROVIDER" in
    precomputed_highres|sampled_highres) COLOR_AUG_AFTER_RESIZE=0 ;;
    *) COLOR_AUG_AFTER_RESIZE=1 ;;
  esac
fi
MOTION_BOUNDARY_ON_RESIZED="${MOTION_BOUNDARY_ON_RESIZED:-1}"
if [[ -z "${KEEP_CROPPED_IMAGES+x}" ]]; then
  case "$PATCH_PROVIDER" in
    precomputed_highres|sampled_highres) KEEP_CROPPED_IMAGES=1 ;;
    *) KEEP_CROPPED_IMAGES=0 ;;
  esac
fi
# Match training defaults. Leaving this empty makes ScanNet++ precomputed H5
# chunks bypass the warm/cache path and can reintroduce 10s+ cold reads.
SCANNETPP_H5_CHUNK_CACHE_DIR="${SCANNETPP_H5_CHUNK_CACHE_DIR:-/data1/zbf/d4rt_scannetpp_h5_chunk_cache}"
SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES="${SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES:-1}"
SCANNETPP_H5_CHUNK_CACHE_MAX_GB="${SCANNETPP_H5_CHUNK_CACHE_MAX_GB:-80}"
SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO="${SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO:-0.9}"
SCANNETPP_H5_CHUNK_CACHE_SCAN_INTERVAL_S="${SCANNETPP_H5_CHUNK_CACHE_SCAN_INTERVAL_S:-600}"
SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S="${SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S:-12}"
SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES="${SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES:-1}"
SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS="${SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS:-16}"
SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES="${SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES:-67108864}"
SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES="${SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES:-2097152}"
SCANNETPP_RGB_READ_MODE="${SCANNETPP_RGB_READ_MODE:-cache}"
SCANNETPP_RGB_LOAD_WORKERS="${SCANNETPP_RGB_LOAD_WORKERS:-1}"
D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT="${D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT:-1}"
CO3DV2_FRAME_CACHE_ITEMS="${CO3DV2_FRAME_CACHE_ITEMS:-384}"
CO3DV2_IO_WORKERS="${CO3DV2_IO_WORKERS:-4}"
D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE="${D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE:-192}"
D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE="${D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE:-192}"
D4RT_DYNAMIC_REPLICA_IO_WORKERS="${D4RT_DYNAMIC_REPLICA_IO_WORKERS:-4}"
SAMPLE_STAGE_ROOT="${SAMPLE_STAGE_ROOT:-}"
SAMPLE_STAGE_SDK_WORKERS="${SAMPLE_STAGE_SDK_WORKERS:-2}"
SAMPLE_STAGE_REQUEST_TIMEOUT_S="${SAMPLE_STAGE_REQUEST_TIMEOUT_S:-5}"
SAMPLE_STAGE_REQUEST_RETRIES="${SAMPLE_STAGE_REQUEST_RETRIES:-1}"
SAMPLE_STAGE_CACHE_MAX_GB="${SAMPLE_STAGE_CACHE_MAX_GB:-372}"
SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO="${SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO:-0.85}"
SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S="${SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S:-30}"
SAMPLE_STAGE_EVICTION_MODE="${SAMPLE_STAGE_EVICTION_MODE:-external}"
DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS="${DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS:-1}"
SAMPLE_STAGE_JANITOR_SLEEP_S="${SAMPLE_STAGE_JANITOR_SLEEP_S:-5}"
SAMPLE_STAGE_WORK_STALE_MIN="${SAMPLE_STAGE_WORK_STALE_MIN:-30}"
SAMPLE_STAGE_WORK_CLEAN_INTERVAL_S="${SAMPLE_STAGE_WORK_CLEAN_INTERVAL_S:-300}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
BATCH_SIZE="${BATCH_SIZE:-40}"
BATCHES="${BATCHES:-100}"
WARMUP_BATCHES="${WARMUP_BATCHES:-10}"
START_BATCH="${START_BATCH:-0}"
START_EPOCH="${START_EPOCH:-0}"
SIMULATE_COMPUTE_MS="${SIMULATE_COMPUTE_MS:-1500}"
STARTUP_SLEEP_S="${STARTUP_SLEEP_S:-10}"
BUILDER_WORKERS="${BUILDER_WORKERS:-18}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-960}"
BATCH_PREFETCH_DEPTH="${BATCH_PREFETCH_DEPTH:-24}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
DATA_WAIT_THRESHOLD_S="${DATA_WAIT_THRESHOLD_S:-2.0}"
DETAIL_MAX_SAMPLES="${DETAIL_MAX_SAMPLES:-3}"
TORCH_THREADS="${TORCH_THREADS:-1}"
MASTER_PORT="${MASTER_PORT:-29632}"
MAX_WALL_S="${MAX_WALL_S:-300}"
D4RT_MAX_TRACK_POINTS="${D4RT_MAX_TRACK_POINTS:-4096}"
D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS="${D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS:-0}"
D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS="${D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS:-0}"
D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE="${D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE:-0}"
D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S="${D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S:-3600}"
export D4RT_CO3D_FRAME_CACHE_ITEMS="$CO3DV2_FRAME_CACHE_ITEMS"
export CO3DV2_IO_WORKERS
export D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE
export D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE
export D4RT_DYNAMIC_REPLICA_IO_WORKERS
export D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROLLING_WARM_ENABLE="${ROLLING_WARM_ENABLE:-0}"
ROLLING_WARM_RUN_ID="${ROLLING_WARM_RUN_ID:-$RUN_ID}"
ROLLING_WARM_BLOCKS="${ROLLING_WARM_BLOCKS:-0}"
ROLLING_WARM_LOOKAHEAD_BLOCKS="${ROLLING_WARM_LOOKAHEAD_BLOCKS:-6}"
ROLLING_WARM_MAX_READY_BLOCKS="${ROLLING_WARM_MAX_READY_BLOCKS:-8}"
ROLLING_WARM_INITIAL_READY_BLOCKS="${ROLLING_WARM_INITIAL_READY_BLOCKS:-6}"
ROLLING_WARM_INITIAL_WAIT_TIMEOUT_S="${ROLLING_WARM_INITIAL_WAIT_TIMEOUT_S:-1200}"
ROLLING_WARM_DATASETS="${ROLLING_WARM_DATASETS:-dynamic_replica,co3dv2,scannetpp}"
ROLLING_WARM_EXACT_DATASETS="${ROLLING_WARM_EXACT_DATASETS:-}"
# Match training: DynamicReplica should use the planned-file coscli include
# path. SDK exact warms individual paths too slowly.
ROLLING_WARM_SDK_EXACT_DATASETS="${ROLLING_WARM_SDK_EXACT_DATASETS:-}"
ROLLING_WARM_COSCLI_ROUTINES="${ROLLING_WARM_COSCLI_ROUTINES:-64}"
ROLLING_WARM_COSCLI_THREAD_NUM="${ROLLING_WARM_COSCLI_THREAD_NUM:-8}"
ROLLING_WARM_PREFIX_WORKERS="${ROLLING_WARM_PREFIX_WORKERS:-8}"
ROLLING_WARM_SDK_WORKERS="${ROLLING_WARM_SDK_WORKERS:-32}"
ROLLING_WARM_TIMEOUT_S="${ROLLING_WARM_TIMEOUT_S:-240}"
ROLLING_WARM_COMPUTE_S_PER_BATCH="${ROLLING_WARM_COMPUTE_S_PER_BATCH:-1.5}"
ROLLING_WARM_SKIP_EXISTING="${ROLLING_WARM_SKIP_EXISTING:-1}"
ROLLING_WARM_SCANNETPP_RGB_MODE="${ROLLING_WARM_SCANNETPP_RGB_MODE:-frame_cache}"
ROLLING_WARM_SCANNETPP_RGB_DECODE="${ROLLING_WARM_SCANNETPP_RGB_DECODE:-1}"
ROLLING_WARM_SCANNETPP_DECODE_WORKERS="${ROLLING_WARM_SCANNETPP_DECODE_WORKERS:-8}"
ROLLING_WARM_SCANNETPP_FRAME_WORKERS="${ROLLING_WARM_SCANNETPP_FRAME_WORKERS:-4}"
ROLLING_WARM_SCANNETPP_DEPTH_WORKERS="${ROLLING_WARM_SCANNETPP_DEPTH_WORKERS:-8}"
ROLLING_WARM_SCANNETPP_H5_WORKERS="${ROLLING_WARM_SCANNETPP_H5_WORKERS:-8}"
ROLLING_WARM_BLOCK_WORKERS="${ROLLING_WARM_BLOCK_WORKERS:-1}"
ROLLING_WARM_LOG_FILE="${ROLLING_WARM_LOG_FILE:-}"
if [[ "$ROLLING_WARM_ENABLE" == "1" ]]; then
  D4RT_ROLLING_WARM_READY_DIR="${D4RT_ROLLING_WARM_READY_DIR:-/data1/zbf/d4rt_probe_rolling_warm/${ROLLING_WARM_RUN_ID}/ready}"
  D4RT_ROLLING_WARM_PROGRESS_DIR="${D4RT_ROLLING_WARM_PROGRESS_DIR:-/data1/zbf/d4rt_probe_rolling_warm/${ROLLING_WARM_RUN_ID}/progress}"
  D4RT_ROLLING_WARM_TIMEOUT_S="${D4RT_ROLLING_WARM_TIMEOUT_S:-900}"
else
  D4RT_ROLLING_WARM_READY_DIR="${D4RT_ROLLING_WARM_READY_DIR:-}"
  D4RT_ROLLING_WARM_PROGRESS_DIR="${D4RT_ROLLING_WARM_PROGRESS_DIR:-}"
  D4RT_ROLLING_WARM_TIMEOUT_S="${D4RT_ROLLING_WARM_TIMEOUT_S:-0}"
fi

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

is_false() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off|"") return 0 ;;
    *) return 1 ;;
  esac
}

# Fast default matches training: patches are precomputed from the augmented
# 256x256 video. High-res patch providers must retain high-res crops and keep
# colour augmentation before resize so patch RGB and video RGB stay consistent.
case "$PATCH_PROVIDER" in
  precomputed_resized)
    if is_true "$PRECOMPUTE_FROM_HIGHRES"; then
      echo "[dataload-sleep-probe] ERROR: PATCH_PROVIDER=precomputed_resized requires PRECOMPUTE_FROM_HIGHRES=0" >&2
      exit 2
    fi
    ;;
  precomputed_highres|sampled_highres)
    if is_true "$COLOR_AUG_AFTER_RESIZE"; then
      echo "[dataload-sleep-probe] ERROR: $PATCH_PROVIDER requires COLOR_AUG_AFTER_RESIZE=0" >&2
      exit 2
    fi
    if is_false "$KEEP_CROPPED_IMAGES"; then
      echo "[dataload-sleep-probe] ERROR: $PATCH_PROVIDER requires KEEP_CROPPED_IMAGES=1" >&2
      exit 2
    fi
    ;;
esac

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-/data1/zbf/d4rt_probe_reports/${RUN_ID}_sleep${SIMULATE_COMPUTE_MS}ms_bw${BUILDER_WORKERS}_bpd${BATCH_PREFETCH_DEPTH}_${NPROC_PER_NODE}rank}"
SPOOL_ROOT="${SPOOL_ROOT:-/data1/zbf/d4rt_probe_spool/${RUN_ID}_sleep${SIMULATE_COMPUTE_MS}ms_bw${BUILDER_WORKERS}_bpd${BATCH_PREFETCH_DEPTH}_${NPROC_PER_NODE}rank}"
mkdir -p "$REPORT_DIR" "$SPOOL_ROOT"

if [[ "$COMPACT_SAMPLES" == "1" ]]; then
  TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/dataload_probe_compact.XXXXXX.yaml")"
  "$PYTHON_BIN" - "$CONFIG" "$TEMP_CONFIG" "$PRECOMPUTE_PATCHES" "$PRECOMPUTE_FROM_HIGHRES" "$STORE_VIDEO_UINT8" "$STORE_AUXILIARY_TENSORS" "$SCANNETPP_H5_CHUNK_CACHE_DIR" "$SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES" "$SCANNETPP_H5_CHUNK_CACHE_MAX_GB" "$SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO" "$SCANNETPP_H5_CHUNK_CACHE_SCAN_INTERVAL_S" "$SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S" "$SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES" "$SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS" "$SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES" "$PATCH_PROVIDER" "$RETURN_HIGHRES_VIDEO" "$LOAD_NORMALS" "$USE_MOTION_BOUNDARIES" "$COLOR_AUG_AFTER_RESIZE" "$MOTION_BOUNDARY_ON_RESIZED" "$KEEP_CROPPED_IMAGES" "$SAMPLE_STAGE_ROOT" "$SAMPLE_STAGE_SDK_WORKERS" "$SAMPLE_STAGE_REQUEST_TIMEOUT_S" "$SAMPLE_STAGE_REQUEST_RETRIES" "$DATASET_LOCALITY_SIZE" "$SEQUENCE_LOCALITY_SIZE" "$FRAME_LOCALITY_RADIUS" "$SAMPLE_STAGE_CACHE_MAX_GB" "$SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO" "$SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S" "$SAMPLE_STAGE_EVICTION_MODE" "$DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS" "$SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES" <<'PY'
from pathlib import Path
import os
import sys

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


config = yaml.safe_load(src.read_text())
patch_provider = sys.argv[16].strip()
if patch_provider:
    if patch_provider == "precomputed_highres":
        config["precompute_patches"] = True
        config["precompute_from_highres"] = True
        config["return_highres_video"] = False
    elif patch_provider == "precomputed_resized":
        config["precompute_patches"] = True
        config["precompute_from_highres"] = False
        config["return_highres_video"] = False
    elif patch_provider == "sampled_resized":
        config["precompute_patches"] = False
        config["precompute_from_highres"] = False
        config["return_highres_video"] = False
    elif patch_provider == "sampled_highres":
        config["precompute_patches"] = False
        config["precompute_from_highres"] = False
        config["return_highres_video"] = True
    else:
        raise SystemExit(f"unsupported PATCH_PROVIDER={patch_provider!r}")
else:
    config["precompute_patches"] = parse_bool(sys.argv[3])
    config["precompute_from_highres"] = parse_bool(sys.argv[4])
    return_highres = sys.argv[17].strip()
    if return_highres:
        config["return_highres_video"] = parse_bool(return_highres)
config["store_video_uint8"] = parse_bool(sys.argv[5])
config["store_auxiliary_tensors"] = parse_bool(sys.argv[6])
load_normals = parse_bool(sys.argv[18])
use_motion_boundaries = parse_bool(sys.argv[19])
color_aug_after_resize = parse_bool(sys.argv[20])
motion_boundary_on_resized = parse_bool(sys.argv[21])
keep_cropped_images_arg = sys.argv[22].strip()
sample_stage_root = sys.argv[23].strip()
sample_stage_sdk_workers = int(sys.argv[24])
sample_stage_request_timeout_s = float(sys.argv[25])
sample_stage_request_retries = int(sys.argv[26])
dataset_locality_size = sys.argv[27].strip()
sequence_locality_size = sys.argv[28].strip()
frame_locality_radius = sys.argv[29].strip()
sample_stage_cache_max_gb = float(sys.argv[30])
sample_stage_cache_low_watermark_ratio = float(sys.argv[31])
sample_stage_cache_scan_interval_s = float(sys.argv[32])
sample_stage_eviction_mode = sys.argv[33].strip()
dynamic_replica_skip_depth_when_tracks = parse_bool(sys.argv[34])
config["use_motion_boundaries"] = use_motion_boundaries
config["color_aug_after_resize"] = color_aug_after_resize
config["motion_boundary_on_resized"] = motion_boundary_on_resized
if sample_stage_root:
    config["sample_stage_root"] = sample_stage_root
config["sample_stage_sdk_workers"] = sample_stage_sdk_workers
config["sample_stage_request_timeout_s"] = sample_stage_request_timeout_s
config["sample_stage_request_retries"] = sample_stage_request_retries
config["sample_stage_cache_max_bytes"] = int(sample_stage_cache_max_gb * 1024**3)
config["sample_stage_cache_low_watermark_ratio"] = sample_stage_cache_low_watermark_ratio
config["sample_stage_cache_scan_interval_s"] = sample_stage_cache_scan_interval_s
config["sample_stage_eviction_mode"] = sample_stage_eviction_mode
if dataset_locality_size:
    config["dataset_locality_size"] = int(dataset_locality_size)
if sequence_locality_size:
    config["sequence_locality_size"] = int(sequence_locality_size)
if frame_locality_radius:
    config["frame_locality_radius"] = int(frame_locality_radius)
if keep_cropped_images_arg:
    config["keep_cropped_images"] = parse_bool(keep_cropped_images_arg)
datasets = {item.get("name"): item for item in config.get("datasets", [])}
for name in ("pointodyssey", "co3dv2", "blendedmvs"):
    item = datasets.get(name)
    if item is not None:
        item.setdefault("adapter_kwargs", {})["load_normals"] = load_normals
if datasets.get("co3dv2") is not None:
    co3d_kwargs = datasets["co3dv2"].setdefault("adapter_kwargs", {})
    co3d_kwargs["frame_cache_items"] = int(os.environ.get("D4RT_CO3D_FRAME_CACHE_ITEMS", "384"))
    co3d_kwargs["io_workers"] = int(os.environ.get("CO3DV2_IO_WORKERS", "4"))
if datasets.get("dynamic_replica") is not None:
    dynamic_kwargs = datasets["dynamic_replica"].setdefault("adapter_kwargs", {})
    dynamic_kwargs["skip_depth_when_tracks"] = dynamic_replica_skip_depth_when_tracks
    dynamic_kwargs["io_workers"] = int(os.environ.get("D4RT_DYNAMIC_REPLICA_IO_WORKERS", "4"))
if config["precompute_from_highres"] and not config["precompute_patches"]:
    raise SystemExit("PRECOMPUTE_FROM_HIGHRES=1 requires PRECOMPUTE_PATCHES=1")
for item in config.get("datasets", []):
    if item.get("name") != "scannetpp":
        continue
    kwargs = item.setdefault("adapter_kwargs", {})
    if sys.argv[7].strip():
        kwargs["precomputed_h5_chunk_cache_dir"] = sys.argv[7].strip()
        kwargs["precomputed_h5_chunk_cache_min_bytes"] = int(sys.argv[8])
        kwargs["precomputed_h5_chunk_cache_max_bytes"] = int(float(sys.argv[9]) * 1024**3)
        kwargs["precomputed_h5_chunk_cache_low_watermark_ratio"] = float(sys.argv[10])
        kwargs["precomputed_h5_chunk_cache_scan_interval_s"] = float(sys.argv[11])
    else:
        kwargs.pop("precomputed_h5_chunk_cache_dir", None)
        kwargs.pop("precomputed_h5_chunk_cache_min_bytes", None)
        kwargs.pop("precomputed_h5_chunk_cache_max_bytes", None)
        kwargs.pop("precomputed_h5_chunk_cache_low_watermark_ratio", None)
        kwargs.pop("precomputed_h5_chunk_cache_scan_interval_s", None)
    kwargs["precomputed_cos_timeout_s"] = int(float(sys.argv[12]))
    kwargs["precomputed_cos_range_retries"] = int(sys.argv[13])
    kwargs["precomputed_cos_range_workers"] = int(sys.argv[14])
    kwargs["precomputed_cos_range_merge_gap_bytes"] = int(sys.argv[15])
    kwargs["precomputed_cos_range_max_span_bytes"] = int(sys.argv[35])
dst.write_text(yaml.safe_dump(config, sort_keys=False))
PY
  CONFIG="$TEMP_CONFIG"
fi

echo "[dataload-sleep-probe] config=$CONFIG"
echo "[dataload-sleep-probe] compact_samples=$COMPACT_SAMPLES patch_provider=${PATCH_PROVIDER:-manual} precompute=$PRECOMPUTE_PATCHES highres=$PRECOMPUTE_FROM_HIGHRES return_highres=${RETURN_HIGHRES_VIDEO:-auto} video_uint8=$STORE_VIDEO_UINT8 aux=$STORE_AUXILIARY_TENSORS load_normals=$LOAD_NORMALS motion_boundaries=$USE_MOTION_BOUNDARIES color_after_resize=$COLOR_AUG_AFTER_RESIZE motion_boundary_on_resized=$MOTION_BOUNDARY_ON_RESIZED keep_cropped=${KEEP_CROPPED_IMAGES:-auto}"
echo "[dataload-sleep-probe] max_track_points=$D4RT_MAX_TRACK_POINTS"
echo "[dataload-sleep-probe] adapter_load_max_track_points dynamic_replica=$D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS scannetpp=$D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS"
echo "[dataload-sleep-probe] direct_cache_rebase=$D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE"
echo "[dataload-sleep-probe] scannetpp_h5_cache=$SCANNETPP_H5_CHUNK_CACHE_DIR min_bytes=$SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES max_gb=$SCANNETPP_H5_CHUNK_CACHE_MAX_GB low_watermark=$SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO scan_s=$SCANNETPP_H5_CHUNK_CACHE_SCAN_INTERVAL_S timeout=$SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S retries=$SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES range_workers=$SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS merge_gap=$SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES max_span=$SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES rgb_mode=$SCANNETPP_RGB_READ_MODE rgb_workers=$SCANNETPP_RGB_LOAD_WORKERS"
echo "[dataload-sleep-probe] scannetpp_depth_chunks_direct=$D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT"
echo "[dataload-sleep-probe] co3dv2_frame_cache_items=$CO3DV2_FRAME_CACHE_ITEMS"
echo "[dataload-sleep-probe] co3dv2_io_workers=$CO3DV2_IO_WORKERS"
echo "[dataload-sleep-probe] dynamic_replica_traj_cache=$D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE rgb_cache=$D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE"
echo "[dataload-sleep-probe] dynamic_replica_io_workers=$D4RT_DYNAMIC_REPLICA_IO_WORKERS"
echo "[dataload-sleep-probe] sample_stage_root=${SAMPLE_STAGE_ROOT:-config}"
echo "[dataload-sleep-probe] sample_stage_sdk_workers=$SAMPLE_STAGE_SDK_WORKERS timeout=$SAMPLE_STAGE_REQUEST_TIMEOUT_S retries=$SAMPLE_STAGE_REQUEST_RETRIES eviction_mode=$SAMPLE_STAGE_EVICTION_MODE"
echo "[dataload-sleep-probe] sample_stage_cache_max_gb=$SAMPLE_STAGE_CACHE_MAX_GB low_watermark=$SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO scan_s=$SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S"
echo "[dataload-sleep-probe] dynamic_replica_skip_depth_when_tracks=$DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS"
echo "[dataload-sleep-probe] sample_stage_usage_full_rescan_s=${D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S:-3600}"
echo "[dataload-sleep-probe] nproc=$NPROC_PER_NODE batch_size=$BATCH_SIZE batches=$BATCHES warmup=$WARMUP_BATCHES"
echo "[dataload-sleep-probe] start_epoch=$START_EPOCH start_batch=$START_BATCH"
echo "[dataload-sleep-probe] builder_workers=$BUILDER_WORKERS prefetch_depth=$PREFETCH_DEPTH batch_prefetch_depth=$BATCH_PREFETCH_DEPTH"
echo "[dataload-sleep-probe] read_only_spool=${D4RT_PLANNED_READ_ONLY_SPOOL:-0} cleanup_on_init=${D4RT_SPOOL_CLEANUP_ON_INIT:-1} preserve_on_cleanup=${D4RT_PRESERVE_SPOOL_ON_CLEANUP:-0}"
echo "[dataload-sleep-probe] locality dataset=${DATASET_LOCALITY_SIZE:-config} sequence=${SEQUENCE_LOCALITY_SIZE:-config} frame=${FRAME_LOCALITY_RADIUS:-config}"
echo "[dataload-sleep-probe] defer_large_collate_stack=${D4RT_DEFER_LARGE_COLLATE_STACK:-1}"
echo "[dataload-sleep-probe] planned_batch_balance=${D4RT_PLANNED_BATCH_BALANCE:-1} planned_batch_size=${D4RT_PLANNED_BATCH_SIZE:-$BATCH_SIZE}"
echo "[dataload-sleep-probe] sequence_affinity=${D4RT_PLANNED_SEQUENCE_AFFINITY:-0}"
echo "[dataload-sleep-probe] block_enqueue=${D4RT_PLANNED_BLOCK_ENQUEUE:-0} block_max=${D4RT_PLANNED_BLOCK_MAX_SAMPLES:-8} block_lookahead=${D4RT_PLANNED_BLOCK_LOOKAHEAD:-240}"
echo "[dataload-sleep-probe] relaxed_order=${D4RT_PLANNED_RELAXED_ORDER:-1} lookahead=${D4RT_PLANNED_RELAXED_LOOKAHEAD:-400} grace=${D4RT_PLANNED_RELAXED_GRACE_S:-0.25}"
if [[ -n "${D4RT_ROLLING_WARM_READY_DIR:-}" ]]; then
  echo "[dataload-sleep-probe] rolling_warm_ready=$D4RT_ROLLING_WARM_READY_DIR progress=${D4RT_ROLLING_WARM_PROGRESS_DIR:-<same as ready>} block_batches=${D4RT_ROLLING_WARM_BLOCK_BATCHES:-10} batch_size=${D4RT_ROLLING_WARM_BATCH_SIZE:-$BATCH_SIZE}"
  echo "[dataload-sleep-probe] rolling_warm_enable=$ROLLING_WARM_ENABLE datasets=$ROLLING_WARM_DATASETS exact=$ROLLING_WARM_EXACT_DATASETS sdk_exact=$ROLLING_WARM_SDK_EXACT_DATASETS lookahead=$ROLLING_WARM_LOOKAHEAD_BLOCKS initial_ready=$ROLLING_WARM_INITIAL_READY_BLOCKS block_workers=$ROLLING_WARM_BLOCK_WORKERS sdk_workers=$ROLLING_WARM_SDK_WORKERS"
fi
echo "[dataload-sleep-probe] simulate_compute_ms=$SIMULATE_COMPUTE_MS max_wall_s=$MAX_WALL_S"
echo "[dataload-sleep-probe] report_dir=$REPORT_DIR"
echo "[dataload-sleep-probe] spool_root=$SPOOL_ROOT"

# Export the runtime limits before starting helper processes.  The rolling
# warmer imports torch/numpy too; without these limits it can spawn hundreds of
# BLAS/OpenMP threads before the actual probe starts.
export CUDA_VISIBLE_DEVICES=""
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:mkl-service package failed to import:UserWarning}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export D4RT_SERIALIZE_ADAPTER_INIT="${D4RT_SERIALIZE_ADAPTER_INIT:-1}"
export D4RT_BUILDER_START_METHOD="${D4RT_BUILDER_START_METHOD:-fork}"
export D4RT_DEFER_LARGE_COLLATE_STACK="${D4RT_DEFER_LARGE_COLLATE_STACK:-1}"
export D4RT_PLANNED_BATCH_BALANCE="${D4RT_PLANNED_BATCH_BALANCE:-1}"
export D4RT_PLANNED_BATCH_SIZE="${D4RT_PLANNED_BATCH_SIZE:-$BATCH_SIZE}"
export D4RT_PLANNED_SEQUENCE_AFFINITY="${D4RT_PLANNED_SEQUENCE_AFFINITY:-0}"
export D4RT_PLANNED_BLOCK_ENQUEUE="${D4RT_PLANNED_BLOCK_ENQUEUE:-0}"
export D4RT_PLANNED_BLOCK_MAX_SAMPLES="${D4RT_PLANNED_BLOCK_MAX_SAMPLES:-8}"
export D4RT_PLANNED_BLOCK_LOOKAHEAD="${D4RT_PLANNED_BLOCK_LOOKAHEAD:-240}"
export D4RT_PLANNED_RELAXED_ORDER="${D4RT_PLANNED_RELAXED_ORDER:-1}"
export D4RT_PLANNED_RELAXED_LOOKAHEAD="${D4RT_PLANNED_RELAXED_LOOKAHEAD:-400}"
export D4RT_PLANNED_RELAXED_GRACE_S="${D4RT_PLANNED_RELAXED_GRACE_S:-0.25}"
export D4RT_PLANNED_RELAXED_LOG="${D4RT_PLANNED_RELAXED_LOG:-0}"
export D4RT_ROLLING_WARM_READY_DIR
export D4RT_ROLLING_WARM_PROGRESS_DIR
export D4RT_ROLLING_WARM_BLOCK_BATCHES="${D4RT_ROLLING_WARM_BLOCK_BATCHES:-10}"
export D4RT_ROLLING_WARM_BATCH_SIZE="${D4RT_ROLLING_WARM_BATCH_SIZE:-$BATCH_SIZE}"
export D4RT_ROLLING_WARM_TIMEOUT_S
export D4RT_ROLLING_WARM_LOG="${D4RT_ROLLING_WARM_LOG:-0}"
if [[ "$D4RT_ROLLING_WARM_BATCH_SIZE" != "$BATCH_SIZE" ]]; then
  echo "[dataload-sleep-probe] ERROR: D4RT_ROLLING_WARM_BATCH_SIZE=$D4RT_ROLLING_WARM_BATCH_SIZE must equal BATCH_SIZE=$BATCH_SIZE" >&2
  exit 2
fi
export D4RT_MAX_TRACK_POINTS
export D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS
export D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS
export D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE
export D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S
export D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT
export SCANNETPP_RGB_READ_MODE
export SCANNETPP_RGB_LOAD_WORKERS

IFS=$'\t' read -r JANITOR_BACKEND JANITOR_STAGE_ROOT JANITOR_MODE JANITOR_MAX_GB JANITOR_LOW JANITOR_SCAN < <(
  "$PYTHON_BIN" - "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
max_bytes = int(config.get("sample_stage_cache_max_bytes", int(372 * 1024**3)))
print(
    "\t".join(
        [
            str(config.get("sample_stage_backend", "")),
            str(config.get("sample_stage_root", "")),
            str(config.get("sample_stage_eviction_mode", "external")),
            f"{max_bytes / 1024**3:.6f}",
            str(config.get("sample_stage_cache_low_watermark_ratio", 0.95)),
            str(config.get("sample_stage_cache_scan_interval_s", 60)),
        ]
    )
)
PY
)

if [[ "$JANITOR_BACKEND" == "cos_sdk" && "$JANITOR_MODE" == "external" && -n "$JANITOR_STAGE_ROOT" ]]; then
  janitor_cmd=(
    "$PYTHON_BIN"
    "$REPO_ROOT/scripts/sample_stage_cache_janitor.py"
    --stage-root "$JANITOR_STAGE_ROOT"
    --cache-max-gb "$JANITOR_MAX_GB"
    --low-watermark "$JANITOR_LOW"
    --scan-interval-s "$JANITOR_SCAN"
    --sleep-s "$SAMPLE_STAGE_JANITOR_SLEEP_S"
    --work-stale-min "$SAMPLE_STAGE_WORK_STALE_MIN"
    --work-clean-interval-s "$SAMPLE_STAGE_WORK_CLEAN_INTERVAL_S"
    --pinned-manifest-root "$D4RT_ROLLING_WARM_READY_DIR"
    --parent-pid "$$"
  )
  if command -v ionice >/dev/null 2>&1; then
    nice -n 10 ionice -c2 -n7 "${janitor_cmd[@]}" &
  else
    nice -n 10 "${janitor_cmd[@]}" &
  fi
  SAMPLE_STAGE_JANITOR_PID="$!"
  echo "[dataload-sleep-probe] sample_stage_janitor_pid=$SAMPLE_STAGE_JANITOR_PID"
fi

wait_rolling_warm_initial_ready() {
  local need="${ROLLING_WARM_INITIAL_READY_BLOCKS:-0}"
  if (( need <= 0 )); then
    return
  fi
  local block_batches="${D4RT_ROLLING_WARM_BLOCK_BATCHES:-10}"
  local start_block=$(( START_BATCH / block_batches ))
  local timeout_s="${ROLLING_WARM_INITIAL_WAIT_TIMEOUT_S:-1200}"
  local deadline=$(( SECONDS + timeout_s ))
  local ready_root="$D4RT_ROLLING_WARM_READY_DIR/g0000"
  local last_log=0
  echo "[dataload-sleep-probe] waiting rolling warm blocks start=$start_block count=$need timeout=${timeout_s}s"
  while true; do
    local ready_count=0
    local missing=()
    local b
    for (( b = start_block; b < start_block + need; b++ )); do
      local marker
      printf -v marker "%s/block_%08d.ready" "$ready_root" "$b"
      if [[ -f "$marker" ]]; then
        ready_count=$(( ready_count + 1 ))
      else
        missing+=("$b")
      fi
    done
    if (( ready_count >= need )); then
      echo "[dataload-sleep-probe] rolling warm ready blocks=${ready_count}/${need}"
      return
    fi
    if (( SECONDS >= deadline )); then
      echo "[dataload-sleep-probe] ERROR: rolling warm wait timed out ready=${ready_count}/${need} missing=${missing[*]} log=${ROLLING_WARM_LOG_FILE:-<none>}" >&2
      exit 1
    fi
    if (( SECONDS - last_log >= 15 )); then
      echo "[dataload-sleep-probe] rolling warm waiting ready=${ready_count}/${need} missing=${missing[*]}"
      last_log=$SECONDS
    fi
    sleep 2
  done
}

if [[ "$ROLLING_WARM_ENABLE" == "1" ]]; then
  if ! command -v coscli >/dev/null 2>&1; then
    echo "[dataload-sleep-probe] ERROR: ROLLING_WARM_ENABLE=1 but coscli was not found" >&2
    exit 1
  fi
  if [[ -z "$JANITOR_STAGE_ROOT" ]]; then
    echo "[dataload-sleep-probe] ERROR: rolling warm requires sample_stage_root in config" >&2
    exit 1
  fi
  mkdir -p "$D4RT_ROLLING_WARM_READY_DIR" "$D4RT_ROLLING_WARM_PROGRESS_DIR"
  ROLLING_WARM_LOG_FILE="${ROLLING_WARM_LOG_FILE:-$REPORT_DIR/rolling_warm.log}"
  rolling_warm_cmd=(
    "$PYTHON_BIN"
    "$REPO_ROOT/scripts/probe_coscli_sequence_warm.py"
    --config "$CONFIG"
    --stage-root "$JANITOR_STAGE_ROOT"
    --epoch "$START_EPOCH"
    --world-size "$NPROC_PER_NODE"
    --batch-size "$BATCH_SIZE"
    --start-batch "$START_BATCH"
    --block-batches "${D4RT_ROLLING_WARM_BLOCK_BATCHES:-10}"
    --blocks "$ROLLING_WARM_BLOCKS"
    --compute-s-per-batch "$ROLLING_WARM_COMPUTE_S_PER_BATCH"
    --datasets "$ROLLING_WARM_DATASETS"
    --exact-warm-datasets "$ROLLING_WARM_EXACT_DATASETS"
    --sdk-exact-datasets "$ROLLING_WARM_SDK_EXACT_DATASETS"
    --coscli-routines "$ROLLING_WARM_COSCLI_ROUTINES"
    --coscli-thread-num "$ROLLING_WARM_COSCLI_THREAD_NUM"
    --prefix-workers "$ROLLING_WARM_PREFIX_WORKERS"
    --stage-sdk-workers "$ROLLING_WARM_SDK_WORKERS"
    --timeout-s "$ROLLING_WARM_TIMEOUT_S"
    --manifest-mode prefix
    --scannetpp-rgb-mode "$ROLLING_WARM_SCANNETPP_RGB_MODE"
    --scannetpp-decode-workers "$ROLLING_WARM_SCANNETPP_DECODE_WORKERS"
    --scannetpp-frame-workers "$ROLLING_WARM_SCANNETPP_FRAME_WORKERS"
    --scannetpp-depth-workers "$ROLLING_WARM_SCANNETPP_DEPTH_WORKERS"
    --scannetpp-h5-workers "$ROLLING_WARM_SCANNETPP_H5_WORKERS"
    --scannetpp-h5-cache-dir "$SCANNETPP_H5_CHUNK_CACHE_DIR"
    --scannetpp-h5-cache-min-bytes "$SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES"
    --scannetpp-h5-cache-max-gb "$SCANNETPP_H5_CHUNK_CACHE_MAX_GB"
    --scannetpp-h5-cache-low-watermark "$SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO"
    --ready-dir "$D4RT_ROLLING_WARM_READY_DIR"
    --progress-dir "$D4RT_ROLLING_WARM_PROGRESS_DIR"
    --generation 0
    --daemon
    --lookahead-blocks "$ROLLING_WARM_LOOKAHEAD_BLOCKS"
    --max-ready-blocks "$ROLLING_WARM_MAX_READY_BLOCKS"
    --block-workers "$ROLLING_WARM_BLOCK_WORKERS"
    --parent-pid "$$"
    --min-progress-ranks "$NPROC_PER_NODE"
  )
  if [[ "$ROLLING_WARM_SCANNETPP_RGB_DECODE" != "1" ]]; then
    rolling_warm_cmd+=(--no-scannetpp-rgb-decode)
  fi
  if [[ -n "$SCANNETPP_H5_CHUNK_CACHE_DIR" ]]; then
    rolling_warm_cmd+=(--include-scannetpp-h5)
  fi
  if [[ "$ROLLING_WARM_SKIP_EXISTING" == "1" ]]; then
    rolling_warm_cmd+=(--skip-existing)
  fi
  echo "[dataload-sleep-probe] starting rolling warm daemon log=$ROLLING_WARM_LOG_FILE"
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "${rolling_warm_cmd[@]}" >"$ROLLING_WARM_LOG_FILE" 2>&1 &
  else
    "${rolling_warm_cmd[@]}" >"$ROLLING_WARM_LOG_FILE" 2>&1 &
  fi
  ROLLING_WARM_PID="$!"
  echo "[dataload-sleep-probe] rolling_warm_pid=$ROLLING_WARM_PID"
  wait_rolling_warm_initial_ready
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:mkl-service package failed to import:UserWarning}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export D4RT_SERIALIZE_ADAPTER_INIT="${D4RT_SERIALIZE_ADAPTER_INIT:-1}"
export D4RT_BUILDER_START_METHOD="${D4RT_BUILDER_START_METHOD:-fork}"
export D4RT_DEFER_LARGE_COLLATE_STACK="${D4RT_DEFER_LARGE_COLLATE_STACK:-1}"
export D4RT_PLANNED_BATCH_BALANCE="${D4RT_PLANNED_BATCH_BALANCE:-1}"
export D4RT_PLANNED_BATCH_SIZE="${D4RT_PLANNED_BATCH_SIZE:-$BATCH_SIZE}"
export D4RT_PLANNED_SEQUENCE_AFFINITY="${D4RT_PLANNED_SEQUENCE_AFFINITY:-0}"
export D4RT_PLANNED_BLOCK_ENQUEUE="${D4RT_PLANNED_BLOCK_ENQUEUE:-0}"
export D4RT_PLANNED_BLOCK_MAX_SAMPLES="${D4RT_PLANNED_BLOCK_MAX_SAMPLES:-8}"
export D4RT_PLANNED_BLOCK_LOOKAHEAD="${D4RT_PLANNED_BLOCK_LOOKAHEAD:-240}"
export D4RT_PROFILE_BUILDER="${D4RT_PROFILE_BUILDER:-0}"
export D4RT_PROFILE_BUILDER_THRESHOLD_S="${D4RT_PROFILE_BUILDER_THRESHOLD_S:-5}"
export D4RT_PROFILE_SPOOL="${D4RT_PROFILE_SPOOL:-0}"
export D4RT_PROFILE_SPOOL_INTERVAL="${D4RT_PROFILE_SPOOL_INTERVAL:-10}"
export D4RT_PLANNED_WAIT_LOG="${D4RT_PLANNED_WAIT_LOG:-0}"
export D4RT_PLANNED_RELAXED_ORDER="${D4RT_PLANNED_RELAXED_ORDER:-1}"
export D4RT_PLANNED_RELAXED_LOOKAHEAD="${D4RT_PLANNED_RELAXED_LOOKAHEAD:-400}"
export D4RT_PLANNED_RELAXED_GRACE_S="${D4RT_PLANNED_RELAXED_GRACE_S:-0.25}"
export D4RT_PLANNED_RELAXED_LOG="${D4RT_PLANNED_RELAXED_LOG:-0}"
export D4RT_PLANNED_READ_ONLY_SPOOL="${D4RT_PLANNED_READ_ONLY_SPOOL:-0}"
export D4RT_PLANNED_READ_ONLY_SPOOL_TIMEOUT_S="${D4RT_PLANNED_READ_ONLY_SPOOL_TIMEOUT_S:-600}"
export D4RT_SPOOL_CLEANUP_ON_INIT="${D4RT_SPOOL_CLEANUP_ON_INIT:-1}"
export D4RT_SKIP_READY_ENQUEUE="${D4RT_SKIP_READY_ENQUEUE:-1}"
export D4RT_PRESERVE_SPOOL_ON_CLEANUP="${D4RT_PRESERVE_SPOOL_ON_CLEANUP:-0}"
export D4RT_ROLLING_WARM_READY_DIR="${D4RT_ROLLING_WARM_READY_DIR:-}"
export D4RT_ROLLING_WARM_PROGRESS_DIR="${D4RT_ROLLING_WARM_PROGRESS_DIR:-}"
export D4RT_ROLLING_WARM_BLOCK_BATCHES="${D4RT_ROLLING_WARM_BLOCK_BATCHES:-10}"
export D4RT_ROLLING_WARM_BATCH_SIZE="${D4RT_ROLLING_WARM_BATCH_SIZE:-$BATCH_SIZE}"
export D4RT_ROLLING_WARM_TIMEOUT_S="${D4RT_ROLLING_WARM_TIMEOUT_S:-0}"
export D4RT_ROLLING_WARM_LOG="${D4RT_ROLLING_WARM_LOG:-0}"
export D4RT_MAX_TRACK_POINTS
export D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE
export D4RT_SLOW_SAMPLE_THRESHOLD_S="${D4RT_SLOW_SAMPLE_THRESHOLD_S:-5}"
export D4RT_SLOW_DATA_THRESHOLD_S="${D4RT_SLOW_DATA_THRESHOLD_S:-5}"
export D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S
export D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT
export SCANNETPP_RGB_READ_MODE
export SCANNETPP_RGB_LOAD_WORKERS

timeout --kill-after=20s "${MAX_WALL_S}s" \
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
    scripts/benchmark_planned_dataloader.py \
    --config "$CONFIG" \
    --planned-mode \
    --builder-workers "$BUILDER_WORKERS" \
    --prefetch-depth "$PREFETCH_DEPTH" \
    --batch-prefetch-depth "$BATCH_PREFETCH_DEPTH" \
    --batch-size "$BATCH_SIZE" \
    --batches "$BATCHES" \
    --warmup-batches "$WARMUP_BATCHES" \
    --start-batch "$START_BATCH" \
    --start-epoch "$START_EPOCH" \
    --simulate-compute-ms "$SIMULATE_COMPUTE_MS" \
    --startup-sleep-s "$STARTUP_SLEEP_S" \
    --log-interval "$LOG_INTERVAL" \
    --data-wait-threshold-s "$DATA_WAIT_THRESHOLD_S" \
    --detail-max-samples "$DETAIL_MAX_SAMPLES" \
    --profile-collate \
    --torch-threads "$TORCH_THREADS" \
    --report-dir "$REPORT_DIR" \
    --spool-root "$SPOOL_ROOT" \
    --no-pin-memory
