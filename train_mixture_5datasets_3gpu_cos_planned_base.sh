#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

case "${VARIANT:-base}" in
  base|vitb|vit-b)
    ;;
  *)
    echo "[mixture_5datasets_cos_planned_base] ERROR: expected VARIANT=base, got ${VARIANT}" >&2
    exit 2
    ;;
esac

export VARIANT="base"
export USE_VIDEOMAE_V2_INIT=1
export ENCODER_PATCH_SIZE=16
export BATCH_SIZE="${BATCH_SIZE:-100}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_6datasets_cos_planned_base_videomaev2}"
export RESUME="${RESUME:-/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_base_videomaev2/checkpoint_latest_10.pth}"
export EPOCHS="${EPOCHS:-50}"

export MASTER_PORT="${MASTER_PORT:-28585}"
# 跟 vitg 并跑时的关键安全设置:
# - tartanair 数据走 /data2/d4rt/datasets/tartanairv1 (本地 NVMe),不读 /data5 NFS。
# - planned spool/TMPDIR 也走 /data2,避免 base rank0 清理或竞争 vitg 的
#   /data1/zbf/d4rt_tmp/d4rt_spool_rank0。
# - SAMPLE_STAGE_BACKEND="" 显式关闭 staging；共同脚本必须用
#   ${SAMPLE_STAGE_BACKEND-cos_sdk} 才会保留这个空字符串。
# - 默认关闭 index warmup,并把 planned builder/prefetch 保持在较低但可用的
#   balanced 档，避免 base 饿 GPU，同时不抢 vitg 的 ScanNet++/COS 路径。
export TMPDIR="${TMPDIR:-/data2/zbf/d4rt_tmp_base}"
export SAMPLE_STAGE_BACKEND="${SAMPLE_STAGE_BACKEND:-}"
export SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS:-}"
export SAMPLE_STAGE_ROOT="${SAMPLE_STAGE_ROOT:-/data2/zbf/d4rt_sample_stage_base}"
export SAMPLE_STAGE_CACHE_MAX_GB="${SAMPLE_STAGE_CACHE_MAX_GB:-200}"
export SAMPLE_STAGE_EVICTION_MODE="${SAMPLE_STAGE_EVICTION_MODE:-external}"
export SAMPLE_STAGE_PRECLEAN="${SAMPLE_STAGE_PRECLEAN:-1}"
export SAMPLE_STAGE_SCENE_PREFETCH_DATASETS="${SAMPLE_STAGE_SCENE_PREFETCH_DATASETS:-}"
export CONFIG="${CONFIG:-configs/tartanair_only_cos_planned_base.yaml}"
export INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-$ROOT_DIR/.index_cache_tartanair_only_base}"
export SCANNETPP_H5_CHUNK_CACHE_DIR="${SCANNETPP_H5_CHUNK_CACHE_DIR:-}"
export AUTO_WARM_INDEX_CACHE="${AUTO_WARM_INDEX_CACHE:-0}"
export WARM_INDEX_WORKERS="${WARM_INDEX_WORKERS:-2}"
export ROLLING_WARM_ENABLE="${ROLLING_WARM_ENABLE:-0}"
# 7 卡数据加载档 (CPU 受限优化版):
# 实测 256 核机器,154 builders × io_workers=4 → CPU 308 核需求,sys=54% 锁争抢,
# 每 sample build 时间被拉长,spool 永远空 (.building 而非 .pt)。
# 改用 16 builders × 7 ranks = 112 个,每 builder 分 ~2.3 核,io_workers=4 可吃 4 核,
# 给 PNG decode 留够 CPU,IPC 锁争抢也降一档。
# - prefetch_depth=1500/rank 缓 ~25 步数据;batch_prefetch_depth=32 吸收 collate 抖动。
# - max_spool=96GB/rank → 7 ranks 共 ~672GB on /data2 (1.8TB 余量),不会刷爆盘。
export BUILDER_WORKERS="${BUILDER_WORKERS:-16}"
export PREFETCH_DEPTH="${PREFETCH_DEPTH:-1500}"
export BATCH_PREFETCH_DEPTH="${BATCH_PREFETCH_DEPTH:-32}"
export MAX_SPOOL_BYTES_GB="${MAX_SPOOL_BYTES_GB:-96}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

export LOSS_W_VIS="${LOSS_W_VIS:-0.1}"

if [[ -z "${VIDEOMAE_MODEL:-}" ]]; then
  if [[ -d /data1/zbf/pretrained/videomae-v2-base ]]; then
    export VIDEOMAE_MODEL="/data1/zbf/pretrained/videomae-v2-base"
  else
    export VIDEOMAE_MODEL="OpenGVLab/VideoMAEv2-Base"
  fi
fi

if [[ "${BASE_LOW_PRIORITY:-0}" == "1" ]]; then
  BASE_NICE="${BASE_NICE:-10}"
  BASE_IONICE_CLASS="${BASE_IONICE_CLASS:-3}"
  BASE_IONICE_LEVEL="${BASE_IONICE_LEVEL:-7}"
  if command -v ionice >/dev/null 2>&1; then
    if [[ "$BASE_IONICE_CLASS" == "3" ]]; then
      exec nice -n "$BASE_NICE" ionice -c3 bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned.sh" "$@"
    fi
    exec nice -n "$BASE_NICE" ionice -c"$BASE_IONICE_CLASS" -n"$BASE_IONICE_LEVEL" bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned.sh" "$@"
  fi
  exec nice -n "$BASE_NICE" bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned.sh" "$@"
fi

exec bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned.sh" "$@"
