#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Disable HDF5 file locking to avoid ENOLCK errors (errno 37) on NFS/shared filesystems
# when multiple builder processes access H5 precomputed files concurrently.
export HDF5_USE_FILE_LOCKING=FALSE

case "${VARIANT:-giant}" in
  giant|vitg|vit-g)
    ;;
  *)
    echo "[mixture_5datasets_cos_planned_vitg] ERROR: expected VARIANT=giant, got ${VARIANT}" >&2
    exit 2
    ;;
esac

export VARIANT="giant"
export USE_VIDEOMAE_V2_INIT=1
export ENCODER_PATCH_SIZE=14
export BATCH_SIZE="${BATCH_SIZE:-15}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
export USE_COMPILE="${USE_COMPILE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_6datasets_cos_planned_vitg_videomaev2}"
# export RESUME="${RESUME:-/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_vitg_videomaev2/checkpoint_latest_146.pth}"
export EPOCHS="${EPOCHS:-300}"
# 当 rank0 data_wait 超过 threshold(120s)+kill_after(300s) 时自动 SIGKILL,
# 触发 NCCL 错误传播让其他 rank 退出,避免整个训练永久 hang。
export D4RT_TRAIN_WATCHDOG_KILL_AFTER_S="${D4RT_TRAIN_WATCHDOG_KILL_AFTER_S:-300}"
# vitg 的 7-rank mixture 数据路径很容易被少数慢样本堵住 planned-order
# 前窗口；并跑 base 时更明显。扩大 relaxed window 并缩短 fallback 等待，
# 让已经 ready 的样本先返回，避免明明 spool 里有样本还空等 60s。
export D4RT_PLANNED_RELAXED_LOOKAHEAD="${D4RT_PLANNED_RELAXED_LOOKAHEAD:-1600}"
export D4RT_PLANNED_FORCE_SKIP_S="${D4RT_PLANNED_FORCE_SKIP_S:-25}"
export VITG_PREFER_COS_ROOTS="${VITG_PREFER_COS_ROOTS:-0}"
export VITG_ALLOW_COS_FALLBACK="${VITG_ALLOW_COS_FALLBACK:-0}"

_d4rt_dir_usable() {
  [[ -d "$1" ]] && stat "$1" >/dev/null 2>&1
}

_d4rt_fallback_root() {
  local var_name="$1"
  local default_path="$2"
  local fallback_path="$3"
  if [[ -n "${!var_name+x}" ]]; then
    return
  fi
  if [[ "$VITG_PREFER_COS_ROOTS" == "1" ]] && _d4rt_dir_usable "$fallback_path"; then
    export "$var_name=$fallback_path"
    echo "[mixture_5datasets_cos_planned_vitg] $var_name using COS root: $fallback_path" >&2
    return
  fi
  if _d4rt_dir_usable "$default_path"; then
    return
  fi
  if [[ "$VITG_ALLOW_COS_FALLBACK" == "1" ]] && _d4rt_dir_usable "$fallback_path"; then
    export "$var_name=$fallback_path"
    echo "[mixture_5datasets_cos_planned_vitg] $var_name fallback: $default_path -> $fallback_path" >&2
    return
  fi
  echo "[mixture_5datasets_cos_planned_vitg] ERROR: $var_name is not usable: $default_path" >&2
  echo "[mixture_5datasets_cos_planned_vitg]        Fix the mount or set $var_name explicitly. Set VITG_ALLOW_COS_FALLBACK=1 only if you want /data_cos fallback." >&2
  exit 2
}

_d4rt_fallback_file() {
  local var_name="$1"
  local default_path="$2"
  local fallback_path="$3"
  if [[ -n "${!var_name+x}" ]]; then
    return
  fi
  if [[ "$VITG_PREFER_COS_ROOTS" == "1" ]] && [[ -f "$fallback_path" ]] && stat "$fallback_path" >/dev/null 2>&1; then
    export "$var_name=$fallback_path"
    echo "[mixture_5datasets_cos_planned_vitg] $var_name using COS root: $fallback_path" >&2
    return
  fi
  if [[ -f "$default_path" ]] && stat "$default_path" >/dev/null 2>&1; then
    return
  fi
  if [[ "$VITG_ALLOW_COS_FALLBACK" == "1" ]] && [[ -f "$fallback_path" ]] && stat "$fallback_path" >/dev/null 2>&1; then
    export "$var_name=$fallback_path"
    echo "[mixture_5datasets_cos_planned_vitg] $var_name fallback: $default_path -> $fallback_path" >&2
    return
  fi
  echo "[mixture_5datasets_cos_planned_vitg] ERROR: $var_name is not usable: $default_path" >&2
  echo "[mixture_5datasets_cos_planned_vitg]        Fix the mount or set $var_name explicitly. Set VITG_ALLOW_COS_FALLBACK=1 only if you want /data_cos fallback." >&2
  exit 2
}

_d4rt_fallback_root SCANNET_ROOT \
  /data4/d4rt_dataset/scannet/scans \
  /data_cos/hdu_datasets/scannet/scans
_d4rt_fallback_root SCANNET_PRECOMPUTE_ROOT \
  /data4/d4rt_dataset/scannet/scans \
  /data_cos/hdu_datasets/scannet/scans
_d4rt_fallback_root DYNAMIC_REPLICA_ROOT \
  /data5/d4rt_dataset/Dynamic_Replica \
  /data_cos/hdu_datasets/Dynamic_Replica
_d4rt_fallback_root CO3DV2_ROOT \
  /data4/d4rt_dataset/Co3Dv2 \
  /data_cos/hdu_datasets/Co3Dv2
_d4rt_fallback_root MVSSYNTH_ROOT \
  /data2/d4rt/datasets/MVS-Synth/GTAV_1080 \
  /data5/d4rt_dataset/MVS-Synth/GTAV_1080
_d4rt_fallback_root TARTANAIR_ROOT \
  /data2/d4rt/datasets/tartanairv1 \
  /data5/d4rt_dataset/tartanairv1
if [[ -z "${WAYMO_ROOT:-}" ]]; then
  export WAYMO_ROOT="/data2/d4rt/datasets/waymo_pre"
fi
if ! _d4rt_dir_usable "$WAYMO_ROOT"; then
  echo "[mixture_5datasets_cos_planned_vitg] ERROR: WAYMO_ROOT is not usable: $WAYMO_ROOT" >&2
  echo "[mixture_5datasets_cos_planned_vitg]        Expected the extracted Waymo precompute layout, e.g. root/training/<sequence>.tfrecord/calib.json." >&2
  exit 2
fi
_d4rt_fallback_root VKITTI_ROOT \
  /data3/dataset/VirtualKitti \
  /data5/d4rt_dataset/VirtualKitti
_d4rt_fallback_root SCANNETPP_ROOT \
  /data5/d4rt_dataset/scannetpp/data \
  /data_cos/hdu_datasets/scannetpp/data
_d4rt_fallback_root SCANNETPP_SPLITS_DIR \
  /data5/d4rt_dataset/scannetpp/splits \
  /data_cos/hdu_datasets/scannetpp/splits
_d4rt_fallback_file SCANNETPP_SCENES_RECORD \
  /data5/d4rt_dataset/scannetpp/scenes_record.json \
  /data_cos/hdu_datasets/scannetpp/scenes_record.json

# Keep the common launcher default unless explicitly overridden.  Do not force
# scannet/dynamic_replica through /data_cos by default.

if [[ -z "${VIDEOMAE_MODEL:-}" ]]; then
  if [[ -d /data1/zbf/pretrained/videomae-v2-giant ]]; then
    export VIDEOMAE_MODEL="/data1/zbf/pretrained/videomae-v2-giant"
  else
    export VIDEOMAE_MODEL="OpenGVLab/VideoMAEv2-giant"
  fi
fi

exec bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned.sh" "$@"
