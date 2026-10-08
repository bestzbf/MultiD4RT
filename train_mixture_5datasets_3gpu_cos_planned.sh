#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/d4rt/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
MASTER_PORT="${MASTER_PORT:-28580}"
CONFIG="${CONFIG:-configs/mixture_5datasets_cos_planned.yaml}"
SCANNET_ROOT="${SCANNET_ROOT:-/data4/d4rt_dataset/scannet/scans}"
SCANNET_PRECOMPUTE_ROOT="${SCANNET_PRECOMPUTE_ROOT:-/data4/d4rt_dataset/scannet/scans}"
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
POINTODYSSEY_ANNO_FRAME_CACHE_DIR="${POINTODYSSEY_ANNO_FRAME_CACHE_DIR:-}"
POINTODYSSEY_STAGE_ANNO_H5="${POINTODYSSEY_STAGE_ANNO_H5:-0}"
POINTODYSSEY_REQUIRE_TRACKS="${POINTODYSSEY_REQUIRE_TRACKS:-1}"
POINTODYSSEY_ASSUME_TRACKS="${POINTODYSSEY_ASSUME_TRACKS:-0}"
POINTODYSSEY_TRACK_WORKERS="${POINTODYSSEY_TRACK_WORKERS:-16}"
KUBRIC_LOCAL_ROOT="${KUBRIC_LOCAL_ROOT:-/data3/Kubric}"
if [[ -z "${KUBRIC_ROOT:-}" ]]; then
  if [[ -d "$KUBRIC_LOCAL_ROOT" ]]; then
    KUBRIC_ROOT="$KUBRIC_LOCAL_ROOT"
  else
    KUBRIC_ROOT="/data_cos/hdu_datasets/Kubric"
  fi
fi
DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/data5/d4rt_dataset/Dynamic_Replica}"
# CO3DV2_ROOT="${CO3DV2_ROOT:-/data2/d4rt/datasets/Co3Dv2}"
CO3DV2_ROOT="${CO3DV2_ROOT:-/data4/d4rt_dataset/Co3Dv2}"
BLENDEDMVS_ROOT="${BLENDEDMVS_ROOT:-/data2/d4rt/datasets/BlendedMVS}"
MVSSYNTH_ROOT="${MVSSYNTH_ROOT:-/data2/d4rt/datasets/MVS-Synth/GTAV_1080}"
TARTANAIR_LOCAL_ROOT="${TARTANAIR_LOCAL_ROOT:-/data2/d4rt/datasets/tartanairv1}"
if [[ -z "${TARTANAIR_ROOT:-}" ]]; then
  if [[ -d "$TARTANAIR_LOCAL_ROOT" ]]; then
    TARTANAIR_ROOT="$TARTANAIR_LOCAL_ROOT"
  else
    TARTANAIR_ROOT="/data5/d4rt_dataset/tartanairv1"
  fi
fi
WAYMO_ROOT="${WAYMO_ROOT:-/data2/d4rt/datasets/waymo_pre}"
VKITTI_ROOT="${VKITTI_ROOT:-/data3/dataset/VirtualKitti}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/data5/d4rt_dataset/scannetpp/data}"
SCANNETPP_SPLITS_DIR="${SCANNETPP_SPLITS_DIR:-/data5/d4rt_dataset/scannetpp/splits}"
SCANNETPP_SCENES_RECORD="${SCANNETPP_SCENES_RECORD:-/data5/d4rt_dataset/scannetpp/scenes_record.json}"
CO3DV2_DENYLIST="${CO3DV2_DENYLIST:-/data/zbf/openclaw/d4rt/configs/co3dv2_denylist_degenerate_clips_20260422.txt}"
CO3DV2_FRAME_CACHE_ITEMS="${CO3DV2_FRAME_CACHE_ITEMS:-384}"
CO3DV2_IO_WORKERS="${CO3DV2_IO_WORKERS:-4}"

INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_5datasets_local}"
TMPDIR="${TMPDIR:-/data1/zbf/d4rt_tmp}"
MAX_SPOOL_BYTES_GB="${MAX_SPOOL_BYTES_GB:-100}"
SPOOL_DIR="${SPOOL_DIR:-/dev/shm/d4rt_spool}"
BUILDER_WORKERS="${BUILDER_WORKERS:-18}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-960}"
BATCH_PREFETCH_DEPTH="${BATCH_PREFETCH_DEPTH:-24}"
DATASET_LOCALITY_SIZE="${DATASET_LOCALITY_SIZE:-48}"
SEQUENCE_LOCALITY_SIZE="${SEQUENCE_LOCALITY_SIZE:-96}"
FRAME_LOCALITY_RADIUS="${FRAME_LOCALITY_RADIUS:-12}"
SAMPLE_STAGE_BACKEND="${SAMPLE_STAGE_BACKEND-cos_sdk}"
SAMPLE_STAGE_ROOT="${SAMPLE_STAGE_ROOT:-/data1/zbf/d4rt_sample_stage}"
POINTODYSSEY_ANNO_FRAME_CACHE_DIR="${POINTODYSSEY_ANNO_FRAME_CACHE_DIR:-}"
BLENDEDMVS_DEPTH_CACHE_DIR="${BLENDEDMVS_DEPTH_CACHE_DIR:-}"
SAMPLE_STAGE_SDK_WORKERS="${SAMPLE_STAGE_SDK_WORKERS:-2}"
# DynamicReplica/Co3D samples touch many small COS objects. Keep the per-object
# tail bounded; failed samples are cheaper than holding a builder for 20s+.
SAMPLE_STAGE_REQUEST_TIMEOUT_S="${SAMPLE_STAGE_REQUEST_TIMEOUT_S:-5}"
SAMPLE_STAGE_REQUEST_RETRIES="${SAMPLE_STAGE_REQUEST_RETRIES:-1}"
# 372GiB is about 399GB decimal, keeping the raw stage cache under the
# requested 400GB class while preserving enough hot data to avoid churn.
SAMPLE_STAGE_CACHE_MAX_GB="${SAMPLE_STAGE_CACHE_MAX_GB:-372}"
SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO="${SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO:-0.85}"
SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S="${SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S:-30}"
SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S="${SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S:-30}"
SAMPLE_STAGE_EVICTION_MODE="${SAMPLE_STAGE_EVICTION_MODE:-external}"
SAMPLE_STAGE_PRECLEAN="${SAMPLE_STAGE_PRECLEAN:-1}"
SAMPLE_STAGE_PRECLEAN_FORCE_LOW_WATERMARK="${SAMPLE_STAGE_PRECLEAN_FORCE_LOW_WATERMARK:-0}"
SAMPLE_STAGE_JANITOR_SLEEP_S="${SAMPLE_STAGE_JANITOR_SLEEP_S:-5}"
SAMPLE_STAGE_WORK_STALE_MIN="${SAMPLE_STAGE_WORK_STALE_MIN:-30}"
SAMPLE_STAGE_WORK_CLEAN_INTERVAL_S="${SAMPLE_STAGE_WORK_CLEAN_INTERVAL_S:-300}"
SAMPLE_STAGE_WINDOW_RADIUS="${SAMPLE_STAGE_WINDOW_RADIUS:-0}"
SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS-co3dv2,scannetpp}"
SAMPLE_STAGE_SCENE_PREFETCH_DATASETS="${SAMPLE_STAGE_SCENE_PREFETCH_DATASETS:-}"
SAMPLE_STAGE_MOUNT_ROOT="${SAMPLE_STAGE_MOUNT_ROOT:-/data_cos}"
SAMPLE_STAGE_EXTRA_MOUNT_ROOTS="${SAMPLE_STAGE_EXTRA_MOUNT_ROOTS:-/data2,/data3,/data4,/data5}"
SAMPLE_STAGE_BUCKET="${SAMPLE_STAGE_BUCKET:-hd-ai-data-1251882982}"
SAMPLE_STAGE_REGION="${SAMPLE_STAGE_REGION:-ap-beijing}"
SAMPLE_STAGE_PASSWD_FILE="${SAMPLE_STAGE_PASSWD_FILE:-/etc/passwd-s3fs-data_cos}"
DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS="${DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS:-1}"
D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE="${D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE:-192}"
D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE="${D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE:-192}"
D4RT_DYNAMIC_REPLICA_IO_WORKERS="${D4RT_DYNAMIC_REPLICA_IO_WORKERS:-4}"
KUBRIC_VISIBILITY_MODE="${KUBRIC_VISIBILITY_MODE:-union}"
KUBRIC_VISIBILITY_DEPTH_TOL="${KUBRIC_VISIBILITY_DEPTH_TOL:-}"
# Cache only the ScanNet++ H5 byte ranges that training actually touches.
# This is separate from SAMPLE_STAGE_CACHE_MAX_GB.  The rolling warmer fills
# these chunks ahead of training; direct COS range reads remain the fallback.
SCANNETPP_H5_CHUNK_CACHE_DIR="${SCANNETPP_H5_CHUNK_CACHE_DIR-/data1/zbf/d4rt_scannetpp_h5_chunk_cache}"
SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES="${SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES:-1}"
SCANNETPP_H5_CHUNK_CACHE_MAX_GB="${SCANNETPP_H5_CHUNK_CACHE_MAX_GB:-200}"
SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO="${SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO:-0.9}"
SCANNETPP_H5_CHUNK_CACHE_SCAN_INTERVAL_S="${SCANNETPP_H5_CHUNK_CACHE_SCAN_INTERVAL_S:-1800}"
SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S="${SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S:-12}"
SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES="${SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES:-1}"
SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS="${SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS:-16}"
SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES="${SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES:-67108864}"
SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES="${SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES:-2097152}"
SCANNETPP_RGB_READ_MODE="${SCANNETPP_RGB_READ_MODE:-cache}"
SCANNETPP_RGB_LOAD_WORKERS="${SCANNETPP_RGB_LOAD_WORKERS:-1}"
D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT="${D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT:-1}"
AUTO_WARM_INDEX_CACHE="${AUTO_WARM_INDEX_CACHE:-1}"
WARM_CACHE_ONLY="${WARM_CACHE_ONLY:-0}"
WARM_ONLY_DATASETS="${WARM_ONLY_DATASETS:-}"
WARM_INDEX_WORKERS="${WARM_INDEX_WORKERS:-16}"
WARM_VAL="${WARM_VAL:-0}"
CO3DV2_FRAME_ANNO_CACHE="${CO3DV2_FRAME_ANNO_CACHE:-1}"
CO3DV2_FRAME_ANNO_CACHE_WORKERS="${CO3DV2_FRAME_ANNO_CACHE_WORKERS:-2}"
CO3DV2_FRAME_ANNO_CACHE_FORCE="${CO3DV2_FRAME_ANNO_CACHE_FORCE:-0}"

VAL_CONFIG="${VAL_CONFIG:-configs/mixture_3datasets_val_local.yaml}"
BATCH_SIZE="${BATCH_SIZE:-40}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
USE_COMPILE="${USE_COMPILE:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
PROFILE_DATA_LOADING="${PROFILE_DATA_LOADING:-0}"
DATA_PROFILE_INTERVAL="${DATA_PROFILE_INTERVAL:-10}"
BUILDER_PROFILE_THRESHOLD_S="${BUILDER_PROFILE_THRESHOLD_S:-5}"
DATA_WAIT_THRESHOLD_S="${DATA_WAIT_THRESHOLD_S:-10.0}"
DATA_WAIT_DETAIL="${DATA_WAIT_DETAIL:-1}"
DATA_WAIT_COMPARE_FWD="${DATA_WAIT_COMPARE_FWD:-0}"
DATA_WAIT_DETAIL_MAX_SAMPLES="${DATA_WAIT_DETAIL_MAX_SAMPLES:-0}"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-5e-5}"
ENCODER_LR_MULT="${ENCODER_LR_MULT:-1.0}"
DECODER_LR_MULT="${DECODER_LR_MULT:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_6datasets_cos_planned_from200}"
# DEFAULT_PRETRAIN="/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from0/checkpoint_latest_200.pth"
DEFAULT_PRETRAIN=""
PRETRAIN="${PRETRAIN-$DEFAULT_PRETRAIN}"
# DEFAULT_RESUME=""
DEFAULT_RESUME="/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200/checkpoint_latest_493.pth"
RESUME="${RESUME-$DEFAULT_RESUME}"
VAL_INTERVAL="${VAL_INTERVAL:-4}"
VAL_SAMPLES="${VAL_SAMPLES:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
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
STORE_VIDEO_UINT8="${STORE_VIDEO_UINT8:-1}"
STORE_AUXILIARY_TENSORS="${STORE_AUXILIARY_TENSORS:-0}"
LOAD_NORMALS="${LOAD_NORMALS:-0}"
USE_MOTION_BOUNDARIES="${USE_MOTION_BOUNDARIES:-1}"
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
LOSS_W_3D="${LOSS_W_3D:-1.0}"
LOSS_W_2D="${LOSS_W_2D:-0.1}"
LOSS_W_VIS="${LOSS_W_VIS:-0.1}"
LOSS_W_DISP="${LOSS_W_DISP:-0.1}"
LOSS_W_CONF="${LOSS_W_CONF:-0.1}"
LOSS_W_CONF_WARMUP_STEPS="${LOSS_W_CONF_WARMUP_STEPS:-0}"
LOSS_W_NORMAL="${LOSS_W_NORMAL:-0.0}"
LOSS_W_STATIC_REPROJ="${LOSS_W_STATIC_REPROJ:-1.0}"
LOSS_3D_MODE="${LOSS_3D_MODE:-scale_invariant}"
# DUSt3R-style two-branch normalization (off by default; behaviour identical
# to paper-default independent normalization when DUST3R_METRIC_BRANCH=0).
# When enabled, samples whose dataset_name is in METRIC_DATASETS use shared
# target-side normalizer (z̄=z, scale-anchoring); other samples keep
# scale-invariant independent normalization.
DUST3R_METRIC_BRANCH="${DUST3R_METRIC_BRANCH:-0}"
METRIC_DATASETS="${METRIC_DATASETS:-scannet,scannetpp,dynamic_replica,kubric,blendedmvs,pointodyssey,tartanair,vkitti2,mvssynth,waymo}"
# Resume-warmup window. When >0 and a checkpoint is being resumed, do a
# fresh-start-style linear warmup 0 -> peak_lr over this many optimizer steps
# before handing control back to the cosine schedule. Useful when the loss
# configuration changes (e.g. enabling DUSt3R metric branch) and the new
# loss surface produces much larger gradients on existing weights.
RESUME_WARMUP_STEPS="${RESUME_WARMUP_STEPS:-0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"
RESET_CONF_HEAD_ON_PRETRAIN="${RESET_CONF_HEAD_ON_PRETRAIN:-1}"
VARIANT="${VARIANT:-large}"
ENCODER_PATCH_SIZE="${ENCODER_PATCH_SIZE:-}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}"
BROADCAST_BUFFERS="${BROADCAST_BUFFERS:-0}"
D4RT_SUPPRESS_MKL_WARNING="${D4RT_SUPPRESS_MKL_WARNING:-1}"
D4RT_BUILDER_FAULTHANDLER="${D4RT_BUILDER_FAULTHANDLER:-0}"
D4RT_BUILD_TIMEOUT="${D4RT_BUILD_TIMEOUT:-100}"
D4RT_MAX_TRACK_POINTS="${D4RT_MAX_TRACK_POINTS:-4096}"
D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS="${D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS:-0}"
D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS="${D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS:-0}"
D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE="${D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE:-0}"
ROLLING_WARM_ENABLE="${ROLLING_WARM_ENABLE:-0}"
ROLLING_WARM_RUN_ID="${ROLLING_WARM_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
D4RT_ROLLING_WARM_READY_DIR="${D4RT_ROLLING_WARM_READY_DIR:-}"
D4RT_ROLLING_WARM_PROGRESS_DIR="${D4RT_ROLLING_WARM_PROGRESS_DIR:-}"
D4RT_ROLLING_WARM_BLOCK_BATCHES="${D4RT_ROLLING_WARM_BLOCK_BATCHES:-10}"
D4RT_ROLLING_WARM_TIMEOUT_S="${D4RT_ROLLING_WARM_TIMEOUT_S:-}"
D4RT_ROLLING_WARM_LOG="${D4RT_ROLLING_WARM_LOG:-0}"
ROLLING_WARM_EPOCH="${ROLLING_WARM_EPOCH:-}"
ROLLING_WARM_BLOCKS="${ROLLING_WARM_BLOCKS:-0}"
ROLLING_WARM_LOOKAHEAD_BLOCKS="${ROLLING_WARM_LOOKAHEAD_BLOCKS:-12}"
ROLLING_WARM_MAX_READY_BLOCKS="${ROLLING_WARM_MAX_READY_BLOCKS:-16}"
ROLLING_WARM_INITIAL_READY_BLOCKS="${ROLLING_WARM_INITIAL_READY_BLOCKS:-8}"
ROLLING_WARM_INITIAL_WAIT_TIMEOUT_S="${ROLLING_WARM_INITIAL_WAIT_TIMEOUT_S:-1200}"
ROLLING_WARM_DATASETS="${ROLLING_WARM_DATASETS:-co3dv2,scannetpp}"
ROLLING_WARM_EXACT_DATASETS="${ROLLING_WARM_EXACT_DATASETS:-}"
# DynamicReplica should use the planned-file coscli include path. Do not set it
# in SDK exact mode by default; SDK exact warms individual paths too slowly.
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
ROLLING_WARM_BLOCK_WORKERS="${ROLLING_WARM_BLOCK_WORKERS:-2}"
ROLLING_WARM_LOG_DIR="${ROLLING_WARM_LOG_DIR:-$ROOT_DIR/logs}"
D4RT_ROLLING_WARM_BATCH_SIZE="${D4RT_ROLLING_WARM_BATCH_SIZE:-$BATCH_SIZE}"
if [[ "$ROLLING_WARM_ENABLE" == "1" ]]; then
  D4RT_ROLLING_WARM_READY_DIR="${D4RT_ROLLING_WARM_READY_DIR:-/data1/zbf/d4rt_rolling_warm/${ROLLING_WARM_RUN_ID}/ready}"
  D4RT_ROLLING_WARM_PROGRESS_DIR="${D4RT_ROLLING_WARM_PROGRESS_DIR:-/data1/zbf/d4rt_rolling_warm/${ROLLING_WARM_RUN_ID}/progress}"
  D4RT_ROLLING_WARM_TIMEOUT_S="${D4RT_ROLLING_WARM_TIMEOUT_S:-900}"
else
  D4RT_ROLLING_WARM_READY_DIR=""
  D4RT_ROLLING_WARM_PROGRESS_DIR=""
  D4RT_ROLLING_WARM_TIMEOUT_S=0
fi
if [[ "$D4RT_ROLLING_WARM_BATCH_SIZE" != "$BATCH_SIZE" ]]; then
  echo "[mixture_5datasets_cos_planned] ERROR: D4RT_ROLLING_WARM_BATCH_SIZE=$D4RT_ROLLING_WARM_BATCH_SIZE must equal BATCH_SIZE=$BATCH_SIZE" >&2
  exit 2
fi
D4RT_PROFILE_BUILDER="${D4RT_PROFILE_BUILDER:-0}"
D4RT_PROFILE_BUILDER_ALL="${D4RT_PROFILE_BUILDER_ALL:-0}"
D4RT_VERBOSE_BUILDER="${D4RT_VERBOSE_BUILDER:-0}"
D4RT_PLANNED_WAIT_LOG="${D4RT_PLANNED_WAIT_LOG:-0}"
D4RT_PLANNED_SEQUENCE_AFFINITY="${D4RT_PLANNED_SEQUENCE_AFFINITY:-0}"
D4RT_PLANNED_BLOCK_ENQUEUE="${D4RT_PLANNED_BLOCK_ENQUEUE:-0}"
D4RT_DEFER_LARGE_COLLATE_STACK="${D4RT_DEFER_LARGE_COLLATE_STACK:-1}"
D4RT_PLANNED_BATCH_BALANCE="${D4RT_PLANNED_BATCH_BALANCE:-1}"

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

# Default is the fast, internally consistent resized-patch path:
#   video:          augmented 256x256 frames
#   local_patches: extracted from those same 256x256 frames
#
# If you intentionally switch PATCH_PROVIDER back to a high-res mode, keep the
# old high-res semantics: colour augmentation must run before resize and the
# high-res crop must be retained for patch extraction. Otherwise video RGB and
# local patch RGB come from different photometric pipelines.
case "$PATCH_PROVIDER" in
  precomputed_resized)
    if is_true "$PRECOMPUTE_FROM_HIGHRES"; then
      echo "[mixture_5datasets_cos_planned] ERROR: PATCH_PROVIDER=precomputed_resized requires PRECOMPUTE_FROM_HIGHRES=0" >&2
      exit 2
    fi
    ;;
  precomputed_highres|sampled_highres)
    if is_true "$COLOR_AUG_AFTER_RESIZE"; then
      echo "[mixture_5datasets_cos_planned] ERROR: $PATCH_PROVIDER requires COLOR_AUG_AFTER_RESIZE=0 to keep high-res patch/video augmentation consistent" >&2
      exit 2
    fi
    if is_false "$KEEP_CROPPED_IMAGES"; then
      echo "[mixture_5datasets_cos_planned] ERROR: $PATCH_PROVIDER requires KEEP_CROPPED_IMAGES=1 because high-res patches need the pre-resize crop" >&2
      exit 2
    fi
    ;;
esac

case "$VARIANT" in
  large)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-1}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-/data1/zbf/pretrained/videomae-v2-large}"
    ENCODER_PATCH_SIZE="${ENCODER_PATCH_SIZE:-16}"
    ;;
  base)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-1}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-/data1/zbf/pretrained/videomae-v2-base}"
    ENCODER_PATCH_SIZE="${ENCODER_PATCH_SIZE:-16}"
    ;;
  huge)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-1}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-OpenGVLab/VideoMAEv2-Huge}"
    ENCODER_PATCH_SIZE=14
    ;;
  giant|vitg)
    VARIANT="giant"
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-1}"
    if [[ -z "${VIDEOMAE_MODEL:-}" ]]; then
      if [[ -d /data1/zbf/pretrained/videomae-v2-giant ]]; then
        VIDEOMAE_MODEL="/data1/zbf/pretrained/videomae-v2-giant"
      else
        VIDEOMAE_MODEL="OpenGVLab/VideoMAEv2-giant"
      fi
    fi
    ENCODER_PATCH_SIZE=14
    ;;
  *)
    USE_VIDEOMAE_V2_INIT="${USE_VIDEOMAE_V2_INIT:-0}"
    VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-}"
    ENCODER_PATCH_SIZE="${ENCODER_PATCH_SIZE:-16}"
    ;;
esac

mkdir -p "$TMPDIR"
mkdir -p "$INDEX_CACHE_DIR"
if [[ -n "$SAMPLE_STAGE_BACKEND" ]]; then
  mkdir -p "$SAMPLE_STAGE_ROOT"
fi
if [[ -n "$BLENDEDMVS_DEPTH_CACHE_DIR" ]]; then
  mkdir -p "$BLENDEDMVS_DEPTH_CACHE_DIR"
fi
if [[ -n "$POINTODYSSEY_ANNO_FRAME_CACHE_DIR" ]]; then
  mkdir -p "$POINTODYSSEY_ANNO_FRAME_CACHE_DIR"
fi

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
export D4RT_PLANNED_SEQUENCE_AFFINITY
export D4RT_PLANNED_BLOCK_ENQUEUE
export D4RT_MAX_TRACK_POINTS
export D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS
export D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS
export D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE
export D4RT_ROLLING_WARM_READY_DIR
export D4RT_ROLLING_WARM_PROGRESS_DIR
export D4RT_ROLLING_WARM_BLOCK_BATCHES
export D4RT_ROLLING_WARM_BATCH_SIZE="${D4RT_ROLLING_WARM_BATCH_SIZE:-$BATCH_SIZE}"
export D4RT_ROLLING_WARM_TIMEOUT_S
export D4RT_ROLLING_WARM_LOG
# forkserver (code default) snapshots a clean state BEFORE the model + CUDA load,
# so per-epoch builder re-forks (set_epoch) don't inherit the ~8.5GiB rank heap.
# 'fork' here (leftover from experiment commit a6719bc) made each builder COW-diverge
# the full model/CUDA heap -> ~108 builders x 8.5GiB -> one-epoch RAM OOM.
export D4RT_BUILDER_START_METHOD="${D4RT_BUILDER_START_METHOD:-forkserver}"
export DATASET_LOCALITY_SIZE
export SEQUENCE_LOCALITY_SIZE
export FRAME_LOCALITY_RADIUS
export D4RT_PLANNED_RELAXED_ORDER="${D4RT_PLANNED_RELAXED_ORDER:-1}"
export D4RT_PLANNED_RELAXED_LOOKAHEAD="${D4RT_PLANNED_RELAXED_LOOKAHEAD:-400}"
export D4RT_PLANNED_RELAXED_GRACE_S="${D4RT_PLANNED_RELAXED_GRACE_S:-0.25}"
export D4RT_PLANNED_RELAXED_LOG="${D4RT_PLANNED_RELAXED_LOG:-0}"
export D4RT_DEFER_LARGE_COLLATE_STACK
export D4RT_PLANNED_BATCH_BALANCE
export D4RT_PLANNED_BATCH_SIZE="${D4RT_PLANNED_BATCH_SIZE:-$BATCH_SIZE}"
export D4RT_SLOW_SAMPLE_THRESHOLD_S="${D4RT_SLOW_SAMPLE_THRESHOLD_S:-99999}"
export D4RT_SLOW_DATA_THRESHOLD_S="${D4RT_SLOW_DATA_THRESHOLD_S:-5.0}"
export D4RT_SAMPLE_STAGE_SLOW_THRESHOLD_S="${D4RT_SAMPLE_STAGE_SLOW_THRESHOLD_S:-99999}"
export D4RT_SAMPLE_STAGE_SLOW_DOWNLOAD_S="${D4RT_SAMPLE_STAGE_SLOW_DOWNLOAD_S:-10.0}"
export D4RT_TRAIN_WATCHDOG_S="${D4RT_TRAIN_WATCHDOG_S:-120}"
export D4RT_TRAIN_WATCHDOG_INTERVAL_S="${D4RT_TRAIN_WATCHDOG_INTERVAL_S:-60}"
# The janitor maintains cache_usage_v1.txt incrementally.  A full os.walk over
# the raw cache is still useful as a safety calibration, but doing it every few
# minutes competes with COS staging once the cache contains hundreds of thousands
# of files.
export D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S="${D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S:-3600}"
export D4RT_POINTODYSSEY_STAGE_ANNO_H5="$POINTODYSSEY_STAGE_ANNO_H5"
export D4RT_POINTODYSSEY_ANNO_FRAME_CACHE_DIR="$POINTODYSSEY_ANNO_FRAME_CACHE_DIR"
export D4RT_BLENDEDMVS_DEPTH_CACHE_DIR="$BLENDEDMVS_DEPTH_CACHE_DIR"
export CO3DV2_FRAME_CACHE_ITEMS
export D4RT_CO3D_FRAME_CACHE_ITEMS="$CO3DV2_FRAME_CACHE_ITEMS"
export CO3DV2_IO_WORKERS
export D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE
export D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE
export D4RT_DYNAMIC_REPLICA_IO_WORKERS
export SCANNETPP_RGB_READ_MODE
export SCANNETPP_RGB_LOAD_WORKERS
export D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
if [[ "$D4RT_SUPPRESS_MKL_WARNING" == "1" ]]; then
  mkl_warning_filter="ignore:mkl-service package failed to import:UserWarning"
  if [[ -n "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="${mkl_warning_filter},${PYTHONWARNINGS}"
  else
    export PYTHONWARNINGS="$mkl_warning_filter"
  fi
fi

echo "[mixture_5datasets_cos_planned] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[mixture_5datasets_cos_planned] NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[mixture_5datasets_cos_planned] MASTER_PORT=$MASTER_PORT"
echo "[mixture_5datasets_cos_planned] CONFIG=$CONFIG"
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_ROOT=$POINTODYSSEY_ROOT"
if [[ -n "$POINTODYSSEY_FAST_ROOT" ]]; then
  echo "[mixture_5datasets_cos_planned] POINTODYSSEY_FAST_ROOT=$POINTODYSSEY_FAST_ROOT"
else
  echo "[mixture_5datasets_cos_planned] POINTODYSSEY_FAST_ROOT=<none>"
fi
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_LOCAL_ROOT=$POINTODYSSEY_LOCAL_ROOT"
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_LOCAL_CACHE_DIR=$POINTODYSSEY_LOCAL_CACHE_DIR"
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_ANNO_FRAME_CACHE_DIR=$POINTODYSSEY_ANNO_FRAME_CACHE_DIR"
echo "[mixture_5datasets_cos_planned] POINTODYSSEY_STAGE_ANNO_H5=$POINTODYSSEY_STAGE_ANNO_H5"
echo "[mixture_5datasets_cos_planned] KUBRIC_ROOT=$KUBRIC_ROOT"
echo "[mixture_5datasets_cos_planned] DYNAMIC_REPLICA_ROOT=$DYNAMIC_REPLICA_ROOT"
echo "[mixture_5datasets_cos_planned] CO3DV2_ROOT=$CO3DV2_ROOT"
echo "[mixture_5datasets_cos_planned] CO3DV2_FRAME_CACHE_ITEMS=$CO3DV2_FRAME_CACHE_ITEMS"
echo "[mixture_5datasets_cos_planned] CO3DV2_IO_WORKERS=$CO3DV2_IO_WORKERS"
echo "[mixture_5datasets_cos_planned] D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE=$D4RT_DYNAMIC_REPLICA_TRAJ_CACHE_SIZE"
echo "[mixture_5datasets_cos_planned] D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE=$D4RT_DYNAMIC_REPLICA_RGB_CACHE_SIZE"
echo "[mixture_5datasets_cos_planned] D4RT_DYNAMIC_REPLICA_IO_WORKERS=$D4RT_DYNAMIC_REPLICA_IO_WORKERS"
echo "[mixture_5datasets_cos_planned] BLENDEDMVS_ROOT=$BLENDEDMVS_ROOT"
echo "[mixture_5datasets_cos_planned] SCANNET_ROOT=$SCANNET_ROOT"
echo "[mixture_5datasets_cos_planned] SCANNET_PRECOMPUTE_ROOT=$SCANNET_PRECOMPUTE_ROOT"
echo "[mixture_5datasets_cos_planned] MVSSYNTH_ROOT=$MVSSYNTH_ROOT"
echo "[mixture_5datasets_cos_planned] TARTANAIR_ROOT=$TARTANAIR_ROOT"
echo "[mixture_5datasets_cos_planned] WAYMO_ROOT=$WAYMO_ROOT"
echo "[mixture_5datasets_cos_planned] VKITTI_ROOT=$VKITTI_ROOT"
echo "[mixture_5datasets_cos_planned] SCANNETPP_ROOT=$SCANNETPP_ROOT"
echo "[mixture_5datasets_cos_planned] SCANNETPP_SPLITS_DIR=$SCANNETPP_SPLITS_DIR"
echo "[mixture_5datasets_cos_planned] SCANNETPP_SCENES_RECORD=$SCANNETPP_SCENES_RECORD"
if [[ "$CO3DV2_ROOT" == /data_cos/* && ",$SAMPLE_STAGE_DATASETS," != *",co3dv2,"* ]]; then
  echo "[mixture_5datasets_cos_planned] WARNING: Co3Dv2 is on /data_cos but is not enabled in SAMPLE_STAGE_DATASETS; it will read through the mounted filesystem."
fi
if [[ "$CO3DV2_ROOT" != /data_cos/* && "$CO3DV2_ROOT" != /data2/* && ",$SAMPLE_STAGE_DATASETS," == *",co3dv2,"* ]]; then
  echo "[mixture_5datasets_cos_planned] Co3Dv2 on NFS ($CO3DV2_ROOT) with staging enabled via SAMPLE_STAGE_EXTRA_MOUNT_ROOTS=$SAMPLE_STAGE_EXTRA_MOUNT_ROOTS"
fi
echo "[mixture_5datasets_cos_planned] BLENDEDMVS_DEPTH_CACHE_DIR=$BLENDEDMVS_DEPTH_CACHE_DIR"
echo "[mixture_5datasets_cos_planned] INDEX_CACHE_DIR=$INDEX_CACHE_DIR"
echo "[mixture_5datasets_cos_planned] TMPDIR=$TMPDIR"
echo "[mixture_5datasets_cos_planned] MAX_SPOOL_BYTES_GB=$MAX_SPOOL_BYTES_GB"
echo "[mixture_5datasets_cos_planned] SPOOL_DIR=$SPOOL_DIR"
echo "[mixture_5datasets_cos_planned] BUILDER_WORKERS=$BUILDER_WORKERS"
echo "[mixture_5datasets_cos_planned] PREFETCH_DEPTH=$PREFETCH_DEPTH"
echo "[mixture_5datasets_cos_planned] BATCH_PREFETCH_DEPTH=$BATCH_PREFETCH_DEPTH"
echo "[mixture_5datasets_cos_planned] DATASET_LOCALITY_SIZE=$DATASET_LOCALITY_SIZE"
echo "[mixture_5datasets_cos_planned] SEQUENCE_LOCALITY_SIZE=$SEQUENCE_LOCALITY_SIZE"
echo "[mixture_5datasets_cos_planned] FRAME_LOCALITY_RADIUS=$FRAME_LOCALITY_RADIUS"
echo "[mixture_5datasets_cos_planned] D4RT_PLANNED_WAIT_LOG=$D4RT_PLANNED_WAIT_LOG"
echo "[mixture_5datasets_cos_planned] D4RT_PLANNED_SEQUENCE_AFFINITY=$D4RT_PLANNED_SEQUENCE_AFFINITY"
echo "[mixture_5datasets_cos_planned] D4RT_PLANNED_BLOCK_ENQUEUE=$D4RT_PLANNED_BLOCK_ENQUEUE"
echo "[mixture_5datasets_cos_planned] D4RT_PLANNED_RELAXED_ORDER=$D4RT_PLANNED_RELAXED_ORDER"
echo "[mixture_5datasets_cos_planned] D4RT_PLANNED_RELAXED_LOOKAHEAD=$D4RT_PLANNED_RELAXED_LOOKAHEAD"
echo "[mixture_5datasets_cos_planned] D4RT_PLANNED_RELAXED_GRACE_S=$D4RT_PLANNED_RELAXED_GRACE_S"
echo "[mixture_5datasets_cos_planned] D4RT_DEFER_LARGE_COLLATE_STACK=$D4RT_DEFER_LARGE_COLLATE_STACK"
echo "[mixture_5datasets_cos_planned] D4RT_PLANNED_BATCH_BALANCE=$D4RT_PLANNED_BATCH_BALANCE"
echo "[mixture_5datasets_cos_planned] D4RT_PLANNED_BATCH_SIZE=$D4RT_PLANNED_BATCH_SIZE"
echo "[mixture_5datasets_cos_planned] ROLLING_WARM_ENABLE=$ROLLING_WARM_ENABLE"
if [[ -n "$D4RT_ROLLING_WARM_READY_DIR" ]]; then
  echo "[mixture_5datasets_cos_planned] D4RT_ROLLING_WARM_READY_DIR=$D4RT_ROLLING_WARM_READY_DIR"
  echo "[mixture_5datasets_cos_planned] D4RT_ROLLING_WARM_PROGRESS_DIR=${D4RT_ROLLING_WARM_PROGRESS_DIR:-<same as ready dir>}"
  echo "[mixture_5datasets_cos_planned] D4RT_ROLLING_WARM_BLOCK_BATCHES=$D4RT_ROLLING_WARM_BLOCK_BATCHES"
  echo "[mixture_5datasets_cos_planned] D4RT_ROLLING_WARM_BATCH_SIZE=$D4RT_ROLLING_WARM_BATCH_SIZE"
  echo "[mixture_5datasets_cos_planned] D4RT_ROLLING_WARM_TIMEOUT_S=$D4RT_ROLLING_WARM_TIMEOUT_S"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_DATASETS=$ROLLING_WARM_DATASETS"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_EXACT_DATASETS=$ROLLING_WARM_EXACT_DATASETS"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_SDK_EXACT_DATASETS=$ROLLING_WARM_SDK_EXACT_DATASETS"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_LOOKAHEAD_BLOCKS=$ROLLING_WARM_LOOKAHEAD_BLOCKS"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_MAX_READY_BLOCKS=$ROLLING_WARM_MAX_READY_BLOCKS"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_INITIAL_READY_BLOCKS=$ROLLING_WARM_INITIAL_READY_BLOCKS"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_INITIAL_WAIT_TIMEOUT_S=$ROLLING_WARM_INITIAL_WAIT_TIMEOUT_S"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_BLOCK_WORKERS=$ROLLING_WARM_BLOCK_WORKERS"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_SDK_WORKERS=$ROLLING_WARM_SDK_WORKERS"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_SKIP_EXISTING=$ROLLING_WARM_SKIP_EXISTING"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_SCANNETPP_RGB_MODE=$ROLLING_WARM_SCANNETPP_RGB_MODE"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_SCANNETPP_RGB_DECODE=$ROLLING_WARM_SCANNETPP_RGB_DECODE"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_SCANNETPP_FRAME_WORKERS=$ROLLING_WARM_SCANNETPP_FRAME_WORKERS"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_SCANNETPP_H5_WORKERS=$ROLLING_WARM_SCANNETPP_H5_WORKERS"
fi
echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_BACKEND=$SAMPLE_STAGE_BACKEND"
if [[ -n "$SAMPLE_STAGE_BACKEND" ]]; then
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_ROOT=$SAMPLE_STAGE_ROOT"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_SDK_WORKERS=$SAMPLE_STAGE_SDK_WORKERS"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_REQUEST_TIMEOUT_S=$SAMPLE_STAGE_REQUEST_TIMEOUT_S"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_REQUEST_RETRIES=$SAMPLE_STAGE_REQUEST_RETRIES"
echo "[mixture_5datasets_cos_planned] DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS=$DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS"
echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_CACHE_MAX_GB=$SAMPLE_STAGE_CACHE_MAX_GB"
echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO=$SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO"
echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S=$SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S"
echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S=$SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S"
echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_EVICTION_MODE=$SAMPLE_STAGE_EVICTION_MODE"
echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_PRECLEAN=$SAMPLE_STAGE_PRECLEAN"
echo "[mixture_5datasets_cos_planned] D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S=${D4RT_SAMPLE_STAGE_USAGE_FULL_RESCAN_INTERVAL_S:-3600}"
echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_JANITOR_SLEEP_S=$SAMPLE_STAGE_JANITOR_SLEEP_S"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_WORK_STALE_MIN=$SAMPLE_STAGE_WORK_STALE_MIN"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_WORK_CLEAN_INTERVAL_S=$SAMPLE_STAGE_WORK_CLEAN_INTERVAL_S"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_WINDOW_RADIUS=$SAMPLE_STAGE_WINDOW_RADIUS"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_DATASETS=$SAMPLE_STAGE_DATASETS"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_SCENE_PREFETCH_DATASETS=$SAMPLE_STAGE_SCENE_PREFETCH_DATASETS"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_BUCKET=$SAMPLE_STAGE_BUCKET"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_REGION=$SAMPLE_STAGE_REGION"
fi
echo "[mixture_5datasets_cos_planned] AUTO_WARM_INDEX_CACHE=$AUTO_WARM_INDEX_CACHE"
echo "[mixture_5datasets_cos_planned] CO3DV2_FRAME_ANNO_CACHE=$CO3DV2_FRAME_ANNO_CACHE"
echo "[mixture_5datasets_cos_planned] CO3DV2_FRAME_ANNO_CACHE_WORKERS=$CO3DV2_FRAME_ANNO_CACHE_WORKERS"
echo "[mixture_5datasets_cos_planned] WARM_CACHE_ONLY=$WARM_CACHE_ONLY"
echo "[mixture_5datasets_cos_planned] WARM_INDEX_WORKERS=$WARM_INDEX_WORKERS"
echo "[mixture_5datasets_cos_planned] WARM_VAL=$WARM_VAL"
if [[ -n "$WARM_ONLY_DATASETS" ]]; then
  echo "[mixture_5datasets_cos_planned] WARM_ONLY_DATASETS=$WARM_ONLY_DATASETS"
fi
if [[ -n "$VAL_CONFIG" ]]; then
  echo "[mixture_5datasets_cos_planned] VAL_CONFIG=$VAL_CONFIG"
fi
echo "[mixture_5datasets_cos_planned] OUTPUT_DIR=$OUTPUT_DIR"
echo "[mixture_5datasets_cos_planned] BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  (effective=$(( BATCH_SIZE * GRAD_ACCUM * NPROC_PER_NODE )))"
echo "[mixture_5datasets_cos_planned] NUM_WORKERS=$NUM_WORKERS"
echo "[mixture_5datasets_cos_planned] LOG_INTERVAL=$LOG_INTERVAL"
echo "[mixture_5datasets_cos_planned] PROFILE_DATA_LOADING=$PROFILE_DATA_LOADING"
echo "[mixture_5datasets_cos_planned] DATA_PROFILE_INTERVAL=$DATA_PROFILE_INTERVAL"
echo "[mixture_5datasets_cos_planned] BUILDER_PROFILE_THRESHOLD_S=$BUILDER_PROFILE_THRESHOLD_S"
echo "[mixture_5datasets_cos_planned] DATA_WAIT_THRESHOLD_S=$DATA_WAIT_THRESHOLD_S"
echo "[mixture_5datasets_cos_planned] DATA_WAIT_DETAIL=$DATA_WAIT_DETAIL"
echo "[mixture_5datasets_cos_planned] DATA_WAIT_COMPARE_FWD=$DATA_WAIT_COMPARE_FWD"
echo "[mixture_5datasets_cos_planned] D4RT_TRAIN_WATCHDOG_S=$D4RT_TRAIN_WATCHDOG_S"
echo "[mixture_5datasets_cos_planned] D4RT_TRAIN_WATCHDOG_INTERVAL_S=$D4RT_TRAIN_WATCHDOG_INTERVAL_S"
echo "[mixture_5datasets_cos_planned] DIST_TIMEOUT_MINUTES=$DIST_TIMEOUT_MINUTES"
echo "[mixture_5datasets_cos_planned] BROADCAST_BUFFERS=$BROADCAST_BUFFERS"
echo "[mixture_5datasets_cos_planned] PATCH_PROVIDER=$PATCH_PROVIDER"
echo "[mixture_5datasets_cos_planned] PRECOMPUTE_PATCHES=$PRECOMPUTE_PATCHES"
echo "[mixture_5datasets_cos_planned] PRECOMPUTE_FROM_HIGHRES=$PRECOMPUTE_FROM_HIGHRES"
echo "[mixture_5datasets_cos_planned] STORE_VIDEO_UINT8=$STORE_VIDEO_UINT8"
echo "[mixture_5datasets_cos_planned] STORE_AUXILIARY_TENSORS=$STORE_AUXILIARY_TENSORS"
echo "[mixture_5datasets_cos_planned] COLOR_AUG_AFTER_RESIZE=$COLOR_AUG_AFTER_RESIZE"
echo "[mixture_5datasets_cos_planned] MOTION_BOUNDARY_ON_RESIZED=$MOTION_BOUNDARY_ON_RESIZED"
echo "[mixture_5datasets_cos_planned] KEEP_CROPPED_IMAGES=${KEEP_CROPPED_IMAGES:-auto}"
echo "[mixture_5datasets_cos_planned] D4RT_MAX_TRACK_POINTS=$D4RT_MAX_TRACK_POINTS"
echo "[mixture_5datasets_cos_planned] D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS=$D4RT_DYNAMIC_REPLICA_LOAD_MAX_TRACK_POINTS"
echo "[mixture_5datasets_cos_planned] D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS=$D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS"
echo "[mixture_5datasets_cos_planned] D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE=$D4RT_SAMPLE_STAGE_DIRECT_CACHE_REBASE"
echo "[mixture_5datasets_cos_planned] SCANNETPP_H5_CHUNK_CACHE_DIR=$SCANNETPP_H5_CHUNK_CACHE_DIR"
echo "[mixture_5datasets_cos_planned] SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES=$SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES"
echo "[mixture_5datasets_cos_planned] SCANNETPP_H5_CHUNK_CACHE_MAX_GB=$SCANNETPP_H5_CHUNK_CACHE_MAX_GB"
echo "[mixture_5datasets_cos_planned] SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO=$SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO"
echo "[mixture_5datasets_cos_planned] SCANNETPP_H5_CHUNK_CACHE_SCAN_INTERVAL_S=$SCANNETPP_H5_CHUNK_CACHE_SCAN_INTERVAL_S"
echo "[mixture_5datasets_cos_planned] SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S=$SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S"
echo "[mixture_5datasets_cos_planned] SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES=$SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES"
echo "[mixture_5datasets_cos_planned] SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS=$SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS"
echo "[mixture_5datasets_cos_planned] SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES=$SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES"
echo "[mixture_5datasets_cos_planned] SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES=$SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES"
echo "[mixture_5datasets_cos_planned] SCANNETPP_RGB_READ_MODE=$SCANNETPP_RGB_READ_MODE"
echo "[mixture_5datasets_cos_planned] SCANNETPP_RGB_LOAD_WORKERS=$SCANNETPP_RGB_LOAD_WORKERS"
echo "[mixture_5datasets_cos_planned] D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT=$D4RT_SCANNETPP_STAGE_DEPTH_CHUNKS_DIRECT"
echo "[mixture_5datasets_cos_planned] ENCODER_LR_MULT=$ENCODER_LR_MULT"
echo "[mixture_5datasets_cos_planned] DECODER_LR_MULT=$DECODER_LR_MULT"
echo "[mixture_5datasets_cos_planned] LOSS_W_3D=$LOSS_W_3D"
echo "[mixture_5datasets_cos_planned] LOSS_W_2D=$LOSS_W_2D"
echo "[mixture_5datasets_cos_planned] LOSS_W_VIS=$LOSS_W_VIS"
echo "[mixture_5datasets_cos_planned] LOSS_W_DISP=$LOSS_W_DISP"
echo "[mixture_5datasets_cos_planned] LOSS_W_CONF=$LOSS_W_CONF"
echo "[mixture_5datasets_cos_planned] LOSS_W_CONF_WARMUP_STEPS=$LOSS_W_CONF_WARMUP_STEPS"
echo "[mixture_5datasets_cos_planned] LOSS_W_NORMAL=$LOSS_W_NORMAL"
echo "[mixture_5datasets_cos_planned] RESET_CONF_HEAD_ON_PRETRAIN=$RESET_CONF_HEAD_ON_PRETRAIN"
echo "[mixture_5datasets_cos_planned] VARIANT=$VARIANT"
echo "[mixture_5datasets_cos_planned] ENCODER_PATCH_SIZE=$ENCODER_PATCH_SIZE"
echo "[mixture_5datasets_cos_planned] USE_VIDEOMAE_V2_INIT=$USE_VIDEOMAE_V2_INIT"
if [[ -n "$VIDEOMAE_MODEL" ]]; then
  echo "[mixture_5datasets_cos_planned] VIDEOMAE_MODEL=$VIDEOMAE_MODEL"
fi
if [[ -n "$PRETRAIN" ]]; then
  if [[ ! -f "$PRETRAIN" ]]; then
    echo "[mixture_5datasets_cos_planned] PRETRAIN not found: $PRETRAIN" >&2
    exit 1
  fi
  echo "[mixture_5datasets_cos_planned] PRETRAIN=$PRETRAIN"
else
  echo "[mixture_5datasets_cos_planned] PRETRAIN=<none>"
fi
if [[ -n "$RESUME" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "[mixture_5datasets_cos_planned] RESUME not found: $RESUME" >&2
    exit 1
  fi
  echo "[mixture_5datasets_cos_planned] RESUME=$RESUME"
else
  echo "[mixture_5datasets_cos_planned] RESUME=<none>"
fi
if [[ -n "$VAL_CONFIG" && ! -f "$VAL_CONFIG" ]]; then
  echo "[mixture_5datasets_cos_planned] VAL_CONFIG not found: $VAL_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[mixture_5datasets_cos_planned] CONFIG not found: $CONFIG" >&2
  exit 1
fi
if grep -q "name: co3dv2" "$CONFIG" && [[ ! -f "$CO3DV2_DENYLIST" ]]; then
  echo "[mixture_5datasets_cos_planned] CO3DV2_DENYLIST not found: $CO3DV2_DENYLIST" >&2
  exit 1
fi

TEMP_CONFIG=""
CONFIG_TO_USE="$CONFIG"
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

if [[ ! -f "$PYTHON_BIN" ]]; then
  echo "[mixture_5datasets_cos_planned] PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TORCHRUN_BIN" ]]; then
  echo "[mixture_5datasets_cos_planned] TORCHRUN_BIN not found: $TORCHRUN_BIN" >&2
  exit 1
fi

detect_rolling_warm_epoch() {
  if [[ -n "$ROLLING_WARM_EPOCH" ]]; then
    printf '%s\n' "$ROLLING_WARM_EPOCH"
    return
  fi
  local resume_base=""
  if [[ -n "$RESUME" ]]; then
    resume_base="$(basename "$RESUME")"
    if [[ "$resume_base" =~ checkpoint_latest_([0-9]+)\.pth$ ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}"
      return
    fi
    if [[ "$resume_base" =~ checkpoint_epoch_([0-9]+)\.pth$ ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}"
      return
    fi
  fi
  printf '%s\n' "0"
}

wait_rolling_warm_initial_ready() {
  local need="${ROLLING_WARM_INITIAL_READY_BLOCKS:-0}"
  if (( need <= 0 )); then
    return
  fi
  local start_batch="${ROLLING_WARM_START_BATCH:-0}"
  local block_batches="$D4RT_ROLLING_WARM_BLOCK_BATCHES"
  local start_block=$(( start_batch / block_batches ))
  local timeout_s="${ROLLING_WARM_INITIAL_WAIT_TIMEOUT_S:-1200}"
  local deadline=$(( SECONDS + timeout_s ))
  local ready_root="$D4RT_ROLLING_WARM_READY_DIR/g0000"
  local last_log=0
  echo "[mixture_5datasets_cos_planned] waiting for initial rolling warm blocks start=$start_block count=$need timeout=${timeout_s}s"
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
      echo "[mixture_5datasets_cos_planned] initial rolling warm ready blocks=${ready_count}/${need}"
      return
    fi
    if (( SECONDS >= deadline )); then
      echo "[mixture_5datasets_cos_planned] initial rolling warm wait timed out: ready=${ready_count}/${need} missing=${missing[*]} log=${ROLLING_WARM_LOG_FILE:-<none>}" >&2
      exit 1
    fi
    if (( SECONDS - last_log >= 15 )); then
      echo "[mixture_5datasets_cos_planned] initial rolling warm waiting: ready=${ready_count}/${need} missing=${missing[*]}"
      last_log=$SECONDS
    fi
    sleep 2
  done
}

if [[ "$AUTO_WARM_INDEX_CACHE" == "1" || "$WARM_CACHE_ONLY" == "1" ]]; then
  if [[ ! -f "$ROOT_DIR/tools/warm_index_cache_5datasets_cos.sh" ]]; then
    echo "[mixture_5datasets_cos_planned] warm script not found: $ROOT_DIR/tools/warm_index_cache_5datasets_cos.sh" >&2
    exit 1
  fi
  echo "[mixture_5datasets_cos_planned] warming index cache before torchrun"
  CONFIG="$CONFIG" \
  INDEX_CACHE_DIR="$INDEX_CACHE_DIR" \
  SCANNET_ROOT="$SCANNET_ROOT" \
  SCANNET_PRECOMPUTE_ROOT="$SCANNET_PRECOMPUTE_ROOT" \
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
  TARTANAIR_ROOT="$TARTANAIR_ROOT" \
  WAYMO_ROOT="$WAYMO_ROOT" \
  VKITTI_ROOT="$VKITTI_ROOT" \
  SCANNETPP_ROOT="$SCANNETPP_ROOT" \
  SCANNETPP_SPLITS_DIR="$SCANNETPP_SPLITS_DIR" \
  SCANNETPP_SCENES_RECORD="$SCANNETPP_SCENES_RECORD" \
  CO3DV2_DENYLIST="$CO3DV2_DENYLIST" \
  INDEX_WORKERS="$WARM_INDEX_WORKERS" \
  WARM_VAL="$WARM_VAL" \
  ONLY_DATASETS="$WARM_ONLY_DATASETS" \
  CO3DV2_FRAME_ANNO_CACHE="$CO3DV2_FRAME_ANNO_CACHE" \
  CO3DV2_FRAME_ANNO_CACHE_WORKERS="$CO3DV2_FRAME_ANNO_CACHE_WORKERS" \
  CO3DV2_FRAME_ANNO_CACHE_FORCE="$CO3DV2_FRAME_ANNO_CACHE_FORCE" \
  bash "$ROOT_DIR/tools/warm_index_cache_5datasets_cos.sh"
fi

if [[ "$WARM_CACHE_ONLY" == "1" ]]; then
  echo "[mixture_5datasets_cos_planned] WARM_CACHE_ONLY=1, stop after warmup"
  exit 0
fi

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/mixture_5datasets_cos_planned.XXXXXX.yaml")"
"$PYTHON_BIN" - "$CONFIG" "$TEMP_CONFIG" "$POINTODYSSEY_ROOT" "$POINTODYSSEY_FAST_ROOT" "$KUBRIC_ROOT" "$DYNAMIC_REPLICA_ROOT" "$CO3DV2_ROOT" "$BLENDEDMVS_ROOT" "$MVSSYNTH_ROOT" "$TARTANAIR_ROOT" "$WAYMO_ROOT" "$VKITTI_ROOT" "$SCANNETPP_ROOT" "$SCANNETPP_SPLITS_DIR" "$SCANNETPP_SCENES_RECORD" "$INDEX_CACHE_DIR" "$CO3DV2_DENYLIST" "$BUILDER_WORKERS" "$PREFETCH_DEPTH" "$MAX_SPOOL_BYTES_GB" "$SAMPLE_STAGE_BACKEND" "$SAMPLE_STAGE_ROOT" "$SAMPLE_STAGE_SDK_WORKERS" "$SAMPLE_STAGE_REQUEST_TIMEOUT_S" "$SAMPLE_STAGE_REQUEST_RETRIES" "$SAMPLE_STAGE_DATASETS" "$SAMPLE_STAGE_SCENE_PREFETCH_DATASETS" "$SAMPLE_STAGE_MOUNT_ROOT" "$SAMPLE_STAGE_EXTRA_MOUNT_ROOTS" "$SAMPLE_STAGE_BUCKET" "$SAMPLE_STAGE_REGION" "$SAMPLE_STAGE_PASSWD_FILE" "$SAMPLE_STAGE_CACHE_MAX_GB" "$SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO" "$SAMPLE_STAGE_CACHE_TOUCH_INTERVAL_S" "$SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S" "$SAMPLE_STAGE_EVICTION_MODE" "$SAMPLE_STAGE_WINDOW_RADIUS" "$PRECOMPUTE_PATCHES" "$PRECOMPUTE_FROM_HIGHRES" "$STORE_VIDEO_UINT8" "$STORE_AUXILIARY_TENSORS" "$SCANNETPP_H5_CHUNK_CACHE_DIR" "$SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES" "$SCANNETPP_H5_CHUNK_CACHE_MAX_GB" "$SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO" "$SCANNETPP_H5_CHUNK_CACHE_SCAN_INTERVAL_S" "$SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S" "$SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES" "$SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS" "$SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES" "$SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES" "$LOAD_NORMALS" "$USE_MOTION_BOUNDARIES" "$COLOR_AUG_AFTER_RESIZE" "$MOTION_BOUNDARY_ON_RESIZED" "$KEEP_CROPPED_IMAGES" "$DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS" "$SCANNET_ROOT" "$SCANNET_PRECOMPUTE_ROOT" <<'PY'
from pathlib import Path
import os
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
tartanair_root = sys.argv[10]
waymo_root = sys.argv[11]
vkitti_root = sys.argv[12]
scannetpp_root = sys.argv[13]
scannetpp_splits_dir = sys.argv[14]
scannetpp_scenes_record = sys.argv[15]
index_cache_dir = sys.argv[16]
co3dv2_denylist = sys.argv[17]
builder_workers = int(sys.argv[18])
prefetch_depth = int(sys.argv[19])
max_spool_bytes = int(float(sys.argv[20]) * 1024**3)
sample_stage_backend = sys.argv[21].strip()
sample_stage_root = sys.argv[22].strip()
sample_stage_sdk_workers = int(sys.argv[23])
sample_stage_request_timeout_s = float(sys.argv[24])
sample_stage_request_retries = int(sys.argv[25])
sample_stage_datasets = sys.argv[26].strip()
sample_stage_scene_prefetch_datasets = sys.argv[27].strip()
sample_stage_mount_root = sys.argv[28].strip()
sample_stage_extra_mount_roots = sys.argv[29].strip()
sample_stage_bucket = sys.argv[30].strip()
sample_stage_region = sys.argv[31].strip()
sample_stage_passwd_file = sys.argv[32].strip()
sample_stage_cache_max_bytes = int(float(sys.argv[33]) * 1024**3)
sample_stage_cache_low_watermark_ratio = float(sys.argv[34])
sample_stage_cache_touch_interval_s = float(sys.argv[35])
sample_stage_cache_scan_interval_s = float(sys.argv[36])
sample_stage_eviction_mode = sys.argv[37].strip()
sample_stage_window_radius = int(sys.argv[38])
precompute_patches_arg = sys.argv[39]
precompute_from_highres_arg = sys.argv[40]
store_video_uint8_arg = sys.argv[41]
store_auxiliary_tensors_arg = sys.argv[42]
scannetpp_h5_chunk_cache_dir = sys.argv[43].strip()
scannetpp_h5_chunk_cache_min_bytes = int(sys.argv[44])
scannetpp_h5_chunk_cache_max_bytes = int(float(sys.argv[45]) * 1024**3)
scannetpp_h5_chunk_cache_low_watermark_ratio = float(sys.argv[46])
scannetpp_h5_chunk_cache_scan_interval_s = float(sys.argv[47])
scannetpp_precomputed_cos_timeout_s = int(float(sys.argv[48]))
scannetpp_precomputed_cos_range_retries = int(sys.argv[49])
scannetpp_precomputed_cos_range_workers = int(sys.argv[50])
scannetpp_precomputed_cos_range_merge_gap_bytes = int(sys.argv[51])
scannetpp_precomputed_cos_range_max_span_bytes = int(sys.argv[52])
load_normals_arg = sys.argv[53]
use_motion_boundaries_arg = sys.argv[54]
color_aug_after_resize_arg = sys.argv[55]
motion_boundary_on_resized_arg = sys.argv[56]
keep_cropped_images_arg = sys.argv[57].strip()
dynamic_replica_skip_depth_when_tracks_arg = sys.argv[58]
scannet_root = sys.argv[59].strip()
scannet_precompute_root = sys.argv[60].strip()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


precompute_patches = parse_bool(precompute_patches_arg)
precompute_from_highres = parse_bool(precompute_from_highres_arg)
store_video_uint8 = parse_bool(store_video_uint8_arg)
store_auxiliary_tensors = parse_bool(store_auxiliary_tensors_arg)
load_normals = parse_bool(load_normals_arg)
use_motion_boundaries = parse_bool(use_motion_boundaries_arg)
color_aug_after_resize = parse_bool(color_aug_after_resize_arg)
motion_boundary_on_resized = parse_bool(motion_boundary_on_resized_arg)
dynamic_replica_skip_depth_when_tracks = parse_bool(dynamic_replica_skip_depth_when_tracks_arg)

if precompute_from_highres and not precompute_patches:
    raise SystemExit(
        "[mixture_5datasets_cos_planned] PRECOMPUTE_FROM_HIGHRES=1 requires PRECOMPUTE_PATCHES=1"
    )

required_markers = {
    "pointodyssey": ("train",),
    "kubric": (),
    "dynamic_replica": ("train",),
    "co3dv2": ("apple",),
    "blendedmvs": ("BlendedMVS_training.txt",),
    "mvssynth": (),
    "tartanair": (),
    "waymo": ("training",),
    "vkitti2": (),
    "scannetpp": (),
}


def validate_root(dataset_name: str, root: str) -> None:
    path = Path(root)
    if not path.is_dir():
        raise SystemExit(
            f"[mixture_5datasets_cos_planned] {dataset_name}.root not found or not a directory: {root}"
        )
    for marker in required_markers.get(dataset_name, ()):
        if not (path / marker).exists():
            raise SystemExit(
                f"[mixture_5datasets_cos_planned] {dataset_name}.root is incomplete: missing {path / marker}"
            )
    if dataset_name == "kubric":
        pass  # skip iterdir check — index cache already validates scene existence


def validate_optional_dir(label: str, root: str) -> None:
    if not root:
        return
    path = Path(root)
    if not path.is_dir():
        raise SystemExit(
            f"[mixture_5datasets_cos_planned] {label} not found or not a directory: {root}"
        )


config = yaml.safe_load(src_path.read_text())
datasets = {item["name"]: item for item in config["datasets"]}

root_map = {
    "scannet": scannet_root,
    "pointodyssey": pointodyssey_root,
    "kubric": kubric_root,
    "dynamic_replica": dynamic_replica_root,
    "co3dv2": co3dv2_root,
    "blendedmvs": blendedmvs_root,
    "mvssynth": mvssynth_root,
    "tartanair": tartanair_root,
    "waymo": waymo_root,
    "vkitti2": vkitti_root,
    "scannetpp": scannetpp_root,
}

for dataset_name, root in root_map.items():
    if dataset_name not in datasets:
        continue
    validate_root(dataset_name, root)
    datasets[dataset_name]["root"] = root

if "scannet" in datasets:
    scannet_cfg = datasets["scannet"].setdefault("adapter_kwargs", {})
    scannet_cfg["precompute_root"] = scannet_precompute_root
    # loss_normal disabled by default (LOSS_W_NORMAL=0); skip the per-pixel
    # normals read from precomputed h5/npz to cut IO. Honours LOAD_NORMALS.
    scannet_cfg["load_normals"] = load_normals

if "dynamic_replica" in datasets:
    dynamic_replica_cfg = datasets["dynamic_replica"].setdefault("adapter_kwargs", {})
    dynamic_replica_cfg["skip_depth_when_tracks"] = dynamic_replica_skip_depth_when_tracks
    dynamic_replica_cfg["io_workers"] = int(os.environ.get("D4RT_DYNAMIC_REPLICA_IO_WORKERS", "4"))

if "mvssynth" in datasets:
    mvssynth_cfg = datasets["mvssynth"].setdefault("adapter_kwargs", {})
    mvssynth_cfg["precompute_root"] = mvssynth_root
    mvssynth_cfg["load_precomputed"] = True
    mvssynth_cfg["load_normals"] = False
    mvssynth_cfg["io_workers"] = int(os.environ.get("D4RT_MVSSYNTH_IO_WORKERS", "4"))
    mvssynth_cfg["io_read_retries"] = int(os.environ.get("D4RT_MVSSYNTH_IO_READ_RETRIES", "3"))
    mvssynth_cfg["io_retry_delay_s"] = float(os.environ.get("D4RT_MVSSYNTH_IO_RETRY_DELAY_S", "0.05"))

if "tartanair" in datasets:
    tartanair_cfg = datasets["tartanair"].setdefault("adapter_kwargs", {})
    tartanair_cfg["precompute_root"] = tartanair_root
    tartanair_cfg["load_precomputed"] = True
    tartanair_cfg["load_normals"] = False
    tartanair_cfg["camera"] = tartanair_cfg.get("camera", "left")
    tartanair_cfg["io_workers"] = int(os.environ.get("D4RT_TARTANAIR_IO_WORKERS", "4"))
    tartanair_cfg["io_read_retries"] = int(os.environ.get("D4RT_TARTANAIR_IO_READ_RETRIES", "3"))
    tartanair_cfg["io_retry_delay_s"] = float(os.environ.get("D4RT_TARTANAIR_IO_RETRY_DELAY_S", "0.05"))
    tartanair_cfg["pose_cache_items"] = int(os.environ.get("D4RT_TARTANAIR_POSE_CACHE_ITEMS", "128"))

if "waymo" in datasets:
    waymo_cfg = datasets["waymo"].setdefault("adapter_kwargs", {})
    waymo_cfg["precompute_root"] = waymo_root
    waymo_cfg["camera_name"] = waymo_cfg.get("camera_name", "FRONT")
    waymo_cfg["extract_flow"] = False
    waymo_cfg["depth_dilation_radius"] = int(os.environ.get("D4RT_WAYMO_DEPTH_DILATION_RADIUS", "6"))
    waymo_cfg["io_workers"] = int(os.environ.get("D4RT_WAYMO_IO_WORKERS", "4"))
    waymo_cfg["io_read_retries"] = int(os.environ.get("D4RT_WAYMO_IO_READ_RETRIES", "3"))
    waymo_cfg["io_retry_delay_s"] = float(os.environ.get("D4RT_WAYMO_IO_RETRY_DELAY_S", "0.05"))
    waymo_cfg["calib_cache_items"] = int(os.environ.get("D4RT_WAYMO_CALIB_CACHE_ITEMS", "128"))

if "vkitti2" in datasets:
    vkitti_cfg = datasets["vkitti2"].setdefault("adapter_kwargs", {})
    vkitti_cfg["camera"] = vkitti_cfg.get("camera", "Camera_0")
    vkitti_cfg["precompute_root"] = vkitti_root
    vkitti_cfg["load_precomputed"] = True
    vkitti_cfg["load_normals"] = False
    vkitti_cfg["load_flow"] = False

if "kubric" in datasets:
    kubric_cfg = datasets["kubric"].setdefault("adapter_kwargs", {})
    kubric_visibility_mode = os.environ.get("KUBRIC_VISIBILITY_MODE", "union").strip()
    if kubric_visibility_mode:
        kubric_cfg["visibility_mode"] = kubric_visibility_mode
    kubric_depth_tol = os.environ.get("KUBRIC_VISIBILITY_DEPTH_TOL", "").strip()
    if kubric_depth_tol:
        kubric_cfg["visibility_depth_rel_tol"] = float(kubric_depth_tol)

# scannetpp splits_dir and strict=False for COS mount
if "scannetpp" in datasets:
    scannetpp_cfg = datasets["scannetpp"].setdefault("adapter_kwargs", {})
    scannetpp_cfg["splits_dir"] = scannetpp_splits_dir
    scannetpp_cfg["scenes_record"] = scannetpp_scenes_record
    scannetpp_cfg["strict"] = False
    if scannetpp_h5_chunk_cache_dir:
        scannetpp_cfg["precomputed_h5_chunk_cache_dir"] = scannetpp_h5_chunk_cache_dir
        scannetpp_cfg["precomputed_h5_chunk_cache_min_bytes"] = scannetpp_h5_chunk_cache_min_bytes
        scannetpp_cfg["precomputed_h5_chunk_cache_max_bytes"] = scannetpp_h5_chunk_cache_max_bytes
        scannetpp_cfg["precomputed_h5_chunk_cache_low_watermark_ratio"] = scannetpp_h5_chunk_cache_low_watermark_ratio
        scannetpp_cfg["precomputed_h5_chunk_cache_scan_interval_s"] = scannetpp_h5_chunk_cache_scan_interval_s
    else:
        scannetpp_cfg.pop("precomputed_h5_chunk_cache_dir", None)
        scannetpp_cfg.pop("precomputed_h5_chunk_cache_min_bytes", None)
        scannetpp_cfg.pop("precomputed_h5_chunk_cache_max_bytes", None)
        scannetpp_cfg.pop("precomputed_h5_chunk_cache_low_watermark_ratio", None)
        scannetpp_cfg.pop("precomputed_h5_chunk_cache_scan_interval_s", None)
    scannetpp_cfg["precomputed_cos_range_workers"] = scannetpp_precomputed_cos_range_workers
    scannetpp_cfg["precomputed_cos_timeout_s"] = scannetpp_precomputed_cos_timeout_s
    scannetpp_cfg["precomputed_cos_range_retries"] = scannetpp_precomputed_cos_range_retries
    scannetpp_cfg["precomputed_cos_range_merge_gap_bytes"] = (
        scannetpp_precomputed_cos_range_merge_gap_bytes
    )
    scannetpp_cfg["precomputed_cos_range_max_span_bytes"] = (
        scannetpp_precomputed_cos_range_max_span_bytes
    )

if "pointodyssey" in datasets:
    validate_optional_dir("pointodyssey.fast_root", pointodyssey_fast_root)

    pointodyssey_cfg = datasets["pointodyssey"].setdefault("adapter_kwargs", {})
    if pointodyssey_fast_root:
        pointodyssey_cfg["fast_root"] = pointodyssey_fast_root
    else:
        pointodyssey_cfg.pop("fast_root", None)
    try:
        point_root_path = Path(pointodyssey_root)
        stage_mount_path = Path(sample_stage_mount_root)
        point_on_stage_mount = point_root_path.is_relative_to(stage_mount_path)
    except Exception:
        point_on_stage_mount = False
    if point_on_stage_mount:
        pointodyssey_cfg["runtime_sanitize"] = False
    else:
        pointodyssey_cfg.pop("runtime_sanitize", None)
    pointodyssey_cfg["load_normals"] = load_normals

if "co3dv2" in datasets:
    co3dv2_cfg = datasets["co3dv2"].setdefault("adapter_kwargs", {})
    co3dv2_cfg["sequence_denylist"] = co3dv2_denylist
    co3dv2_cfg["load_normals"] = load_normals
    co3dv2_cfg["frame_cache_items"] = int(os.environ.get("CO3DV2_FRAME_CACHE_ITEMS", "384"))
    co3dv2_cfg["io_workers"] = int(os.environ.get("CO3DV2_IO_WORKERS", "4"))

if "blendedmvs" in datasets:
    blendedmvs_cfg = datasets["blendedmvs"].setdefault("adapter_kwargs", {})
    blendedmvs_cfg["load_normals"] = load_normals

config["index_cache_dir"] = index_cache_dir
config["planned_mode"] = True
config["builder_workers"] = builder_workers
config["prefetch_depth"] = prefetch_depth
config["max_spool_bytes"] = max_spool_bytes
_spool_dir = os.environ.get("SPOOL_DIR", "").strip()
if _spool_dir:
    config["spool_dir"] = _spool_dir + "/rank{rank}"
config["dataset_locality_size"] = int(os.environ["DATASET_LOCALITY_SIZE"])
config["sequence_locality_size"] = int(os.environ["SEQUENCE_LOCALITY_SIZE"])
config["frame_locality_radius"] = int(os.environ["FRAME_LOCALITY_RADIUS"])
config["precompute_patches"] = precompute_patches
config["precompute_from_highres"] = precompute_from_highres
config["store_video_uint8"] = store_video_uint8
config["store_auxiliary_tensors"] = store_auxiliary_tensors
config["use_motion_boundaries"] = use_motion_boundaries
config["color_aug_after_resize"] = color_aug_after_resize
config["motion_boundary_on_resized"] = motion_boundary_on_resized
if keep_cropped_images_arg:
    config["keep_cropped_images"] = parse_bool(keep_cropped_images_arg)
if sample_stage_backend:
    config["sample_stage_backend"] = sample_stage_backend
    config["sample_stage_root"] = sample_stage_root
    config["sample_stage_sdk_workers"] = sample_stage_sdk_workers
    config["sample_stage_request_timeout_s"] = sample_stage_request_timeout_s
    config["sample_stage_request_retries"] = sample_stage_request_retries
    config["sample_stage_datasets"] = [
        item.strip()
        for item in sample_stage_datasets.split(",")
        if item.strip()
    ]
    config["sample_stage_scene_prefetch_datasets"] = [
        item.strip()
        for item in sample_stage_scene_prefetch_datasets.split(",")
        if item.strip()
    ]
    config["sample_stage_mount_root"] = sample_stage_mount_root
    config["sample_stage_extra_mount_roots"] = [
        item.strip()
        for item in sample_stage_extra_mount_roots.split(",")
        if item.strip()
    ]
    config["sample_stage_bucket"] = sample_stage_bucket
    config["sample_stage_region"] = sample_stage_region
    config["sample_stage_passwd_file"] = sample_stage_passwd_file
    config["sample_stage_cache_max_bytes"] = sample_stage_cache_max_bytes
    config["sample_stage_cache_low_watermark_ratio"] = sample_stage_cache_low_watermark_ratio
    config["sample_stage_cache_touch_interval_s"] = sample_stage_cache_touch_interval_s
    config["sample_stage_cache_scan_interval_s"] = sample_stage_cache_scan_interval_s
    config["sample_stage_eviction_mode"] = sample_stage_eviction_mode
    config["sample_stage_window_radius"] = sample_stage_window_radius
    rolling_ready = os.environ.get("D4RT_ROLLING_WARM_READY_DIR", "").strip()
    if rolling_ready:
        # Block manifests are written under the ready dir.  The progress dir
        # only contains rank progress files, so using it here lets the janitor
        # evict freshly warmed block inputs.
        config["sample_stage_pinned_manifest_root"] = rolling_ready

dst_path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
CONFIG_TO_USE="$TEMP_CONFIG"
echo "[mixture_5datasets_cos_planned] EFFECTIVE_CONFIG=$CONFIG_TO_USE"

if [[ "$SAMPLE_STAGE_BACKEND" == "cos_sdk" && "$SAMPLE_STAGE_EVICTION_MODE" == "external" ]]; then
  janitor_cmd=(
    "$PYTHON_BIN"
    "$ROOT_DIR/scripts/sample_stage_cache_janitor.py"
    --stage-root "$SAMPLE_STAGE_ROOT"
    --cache-max-gb "$SAMPLE_STAGE_CACHE_MAX_GB"
    --low-watermark "$SAMPLE_STAGE_CACHE_LOW_WATERMARK_RATIO"
    --scan-interval-s "$SAMPLE_STAGE_CACHE_SCAN_INTERVAL_S"
    --sleep-s "$SAMPLE_STAGE_JANITOR_SLEEP_S"
    --work-stale-min "$SAMPLE_STAGE_WORK_STALE_MIN"
    --work-clean-interval-s "$SAMPLE_STAGE_WORK_CLEAN_INTERVAL_S"
    --pinned-manifest-root "$D4RT_ROLLING_WARM_READY_DIR"
    --parent-pid "$$"
  )
  if [[ "$SAMPLE_STAGE_PRECLEAN" == "1" ]]; then
    echo "[mixture_5datasets_cos_planned] pre-cleaning sample stage cache before torchrun"
    preclean_cmd=("${janitor_cmd[@]}" --once)
    if [[ "$SAMPLE_STAGE_PRECLEAN_FORCE_LOW_WATERMARK" == "1" ]]; then
      preclean_cmd+=(--force-low-watermark)
    fi
    if command -v ionice >/dev/null 2>&1; then
      nice -n 10 ionice -c2 -n7 "${preclean_cmd[@]}"
    else
      nice -n 10 "${preclean_cmd[@]}"
    fi
  fi
  if command -v ionice >/dev/null 2>&1; then
    nice -n 10 ionice -c2 -n7 "${janitor_cmd[@]}" &
  else
    nice -n 10 "${janitor_cmd[@]}" &
  fi
  SAMPLE_STAGE_JANITOR_PID="$!"
  echo "[mixture_5datasets_cos_planned] SAMPLE_STAGE_JANITOR_PID=$SAMPLE_STAGE_JANITOR_PID"
fi

if [[ "$ROLLING_WARM_ENABLE" == "1" ]]; then
  if ! command -v coscli >/dev/null 2>&1; then
    echo "[mixture_5datasets_cos_planned] ROLLING_WARM_ENABLE=1 but coscli was not found" >&2
    exit 1
  fi
  mkdir -p "$D4RT_ROLLING_WARM_READY_DIR" "$D4RT_ROLLING_WARM_PROGRESS_DIR" "$ROLLING_WARM_LOG_DIR"
  ROLLING_WARM_EPOCH_EFFECTIVE="$(detect_rolling_warm_epoch)"
  ROLLING_WARM_LOG_FILE="$ROLLING_WARM_LOG_DIR/rolling_warm_${ROLLING_WARM_RUN_ID}.log"
  rolling_warm_cmd=(
    "$PYTHON_BIN"
    "$ROOT_DIR/scripts/probe_coscli_sequence_warm.py"
    --config "$CONFIG_TO_USE"
    --stage-root "$SAMPLE_STAGE_ROOT"
    --epoch "$ROLLING_WARM_EPOCH_EFFECTIVE"
    --world-size "$NPROC_PER_NODE"
    --batch-size "$BATCH_SIZE"
    --start-batch "${ROLLING_WARM_START_BATCH:-0}"
    --block-batches "$D4RT_ROLLING_WARM_BLOCK_BATCHES"
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
  echo "[mixture_5datasets_cos_planned] starting rolling warm daemon epoch=$ROLLING_WARM_EPOCH_EFFECTIVE log=$ROLLING_WARM_LOG_FILE"
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "${rolling_warm_cmd[@]}" >"$ROLLING_WARM_LOG_FILE" 2>&1 &
  else
    "${rolling_warm_cmd[@]}" >"$ROLLING_WARM_LOG_FILE" 2>&1 &
  fi
  ROLLING_WARM_PID="$!"
  echo "[mixture_5datasets_cos_planned] ROLLING_WARM_PID=$ROLLING_WARM_PID"
  wait_rolling_warm_initial_ready
fi

cmd=(
  "$TORCHRUN_BIN"
  --rdzv-backend=c10d
  --rdzv-endpoint="localhost:$MASTER_PORT"
  --nproc_per_node="$NPROC_PER_NODE"
  train_mixture.py
  --config "$CONFIG_TO_USE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --grad-accum "$GRAD_ACCUM"
  --log-interval "$LOG_INTERVAL"
  --epochs "$EPOCHS"
  --lr "$LR"
  --encoder-lr-mult "$ENCODER_LR_MULT"
  --decoder-lr-mult "$DECODER_LR_MULT"
  --num-frames 48
  --output-dir "$OUTPUT_DIR"
  --encoder-patch-size "$ENCODER_PATCH_SIZE"
  --patch-provider "$PATCH_PROVIDER"
  --val-interval "$VAL_INTERVAL"
  --val-samples "$VAL_SAMPLES"
  --save-interval "$SAVE_INTERVAL"
  --loss-w-3d "$LOSS_W_3D"
  --loss-w-2d "$LOSS_W_2D"
  --loss-w-vis "$LOSS_W_VIS"
  --loss-w-disp "$LOSS_W_DISP"
  --loss-w-conf "$LOSS_W_CONF"
  --loss-conf-warmup-steps "$LOSS_W_CONF_WARMUP_STEPS"
  --loss-w-normal "$LOSS_W_NORMAL"
  --loss-w-static-reprojection "$LOSS_W_STATIC_REPROJ"
  --loss-3d-mode "$LOSS_3D_MODE"
  --lr-warmup-steps "$LR_WARMUP_STEPS"
  --dist-timeout-minutes "$DIST_TIMEOUT_MINUTES"
  --variant "$VARIANT"
  --planned-mode
  --builder-workers "$BUILDER_WORKERS"
  --prefetch-depth "$PREFETCH_DEPTH"
  --batch-prefetch-depth "$BATCH_PREFETCH_DEPTH"
)

if [[ "$USE_COMPILE" == "1" ]]; then
  cmd+=(--compile)
fi
if [[ "$USE_VIDEOMAE_V2_INIT" == "1" ]]; then
  cmd+=(--use-videomae-v2-init)
fi
if [[ -n "$VIDEOMAE_MODEL" ]]; then
  cmd+=(--videomae-model "$VIDEOMAE_MODEL")
fi
if [[ "$RESET_CONF_HEAD_ON_PRETRAIN" == "1" ]]; then
  cmd+=(--reset-confidence-head-on-pretrain)
fi
if [[ "$BROADCAST_BUFFERS" == "1" ]]; then
  cmd+=(--broadcast-buffers)
fi
if [[ "$PROFILE_DATA_LOADING" == "1" ]]; then
  export D4RT_PROFILE_SPOOL="${D4RT_PROFILE_SPOOL:-0}"
  export D4RT_PROFILE_SPOOL_INTERVAL="${D4RT_PROFILE_SPOOL_INTERVAL:-$DATA_PROFILE_INTERVAL}"
  export D4RT_PROFILE_BUILDER="${D4RT_PROFILE_BUILDER:-0}"
  export D4RT_PROFILE_BUILDER_THRESHOLD_S="${D4RT_PROFILE_BUILDER_THRESHOLD_S:-$BUILDER_PROFILE_THRESHOLD_S}"
  export D4RT_PROFILE_DATA_WAIT="${D4RT_PROFILE_DATA_WAIT:-1}"
  export D4RT_DATA_WAIT_THRESHOLD_S="${D4RT_DATA_WAIT_THRESHOLD_S:-$DATA_WAIT_THRESHOLD_S}"
  export D4RT_DATA_WAIT_DETAIL="${D4RT_DATA_WAIT_DETAIL:-$DATA_WAIT_DETAIL}"
  export D4RT_DATA_WAIT_COMPARE_FWD="${D4RT_DATA_WAIT_COMPARE_FWD:-$DATA_WAIT_COMPARE_FWD}"
  export D4RT_DATA_WAIT_DETAIL_MAX_SAMPLES="${D4RT_DATA_WAIT_DETAIL_MAX_SAMPLES:-$DATA_WAIT_DETAIL_MAX_SAMPLES}"
  cmd+=(--profile-data-loading --data-profile-interval "$DATA_PROFILE_INTERVAL")
fi
if [[ -n "$VAL_CONFIG" ]]; then
  cmd+=(--val-config "$VAL_CONFIG")
fi
if [[ -n "$PRETRAIN" ]]; then
  cmd+=(--pretrain "$PRETRAIN")
fi
if [[ -n "$RESUME" ]]; then
  cmd+=(--resume "$RESUME")
fi
if [[ "$DUST3R_METRIC_BRANCH" == "1" ]]; then
  cmd+=(--dust3r-metric-branch --metric-datasets "$METRIC_DATASETS")
fi
if [[ "${RESUME_WARMUP_STEPS:-0}" -gt 0 ]]; then
  cmd+=(--resume-warmup-steps "$RESUME_WARMUP_STEPS")
fi
cmd+=("$@")

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
