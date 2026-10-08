#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# CHECKPOINT=/data/zbf/openclaw/d4rt/outputs/mixture_5datasets_blendedmvs_large_3gpu_bs5/checkpoint_latest_41.pth \
# bash tools/run_visualize_checkpoint.sh

# 统一 checkpoint 可视化包装脚本。
#
# 用法：
#   bash tools/run_visualize_checkpoint.sh pointodyssey
#   bash tools/run_visualize_checkpoint.sh dynamic_replica
#
# 也支持环境变量：
#   DATASET=pointodyssey bash tools/run_visualize_checkpoint.sh
#
# 额外命令行参数会继续透传给 Python 脚本：
#   bash tools/run_visualize_checkpoint.sh pointodyssey --num-samples 1




# 直接打开 .sh 改默认值。统一脚本里主要改这些行：
# run_visualize_checkpoint.sh (line 67)

# CUDA_VISIBLE_DEVICES
# CONFIG
# CHECKPOINT
# OUTPUT_DIR
# PATCH_PROVIDER
# NUM_SAMPLES
# DENSE_GT_POINT_SOURCE
# DENSE_PRED_POINT_CLOUD_STRIDE
# 改完直接跑：

# bash tools/run_visualize_checkpoint.sh pointodyssey
# 如果你是想“以后默认就用某个 checkpoint / 某个输出目录”，就改 .sh 里的默认值。
# 如果只是这一次想试试，建议用第一种环境变量覆盖，不容易把默认配置改乱。


DATASET_ARG="${1:-}"
case "$DATASET_ARG" in
  pointodyssey|dynamic_replica|kubric|co3dv2|mixture|mixture_5datasets|mixture_5datasets_blendedmvs)
    DATASET="${DATASET:-$DATASET_ARG}"
    shift
    ;;
  "")
    : ;;
  *)
    DATASET="${DATASET:-$DATASET_ARG}"
    ;;
esac

# 如果用户指定了 CHECKPOINT 但没有指定 DATASET，从路径自动推断
if [[ -z "${DATASET:-}" && -n "${CHECKPOINT:-}" ]]; then
  if [[ "$CHECKPOINT" == *mixture_5datasets_blendedmvs* ]]; then
    DATASET="mixture_5datasets_blendedmvs"
  elif [[ "$CHECKPOINT" == *mixture* ]]; then
    DATASET="mixture"
  elif [[ "$CHECKPOINT" == *kubric* ]]; then
    DATASET="kubric"
  elif [[ "$CHECKPOINT" == *dynamic_replica* ]]; then
    DATASET="dynamic_replica"
  elif [[ "$CHECKPOINT" == *pointodyssey* ]]; then
    DATASET="pointodyssey"
  elif [[ "$CHECKPOINT" == *co3dv2* ]]; then
    DATASET="co3dv2"
  fi
fi

DATASET="${DATASET:-pointodyssey}"

if [[ "$DATASET" != "pointodyssey" && "$DATASET" != "dynamic_replica" && "$DATASET" != "kubric" && "$DATASET" != "co3dv2" && "$DATASET" != "mixture" && "$DATASET" != "mixture_5datasets" && "$DATASET" != "mixture_5datasets_blendedmvs" ]]; then
  echo "Usage: bash tools/run_visualize_checkpoint.sh [pointodyssey|dynamic_replica|kubric|co3dv2|mixture] [extra args...]" >&2
  echo "   or: CHECKPOINT=/path/to/co3dv2_xxx/ckpt.pth bash tools/run_visualize_checkpoint.sh" >&2
  echo "Unknown DATASET: $DATASET" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
DEVICE="${DEVICE:-cuda}"
MODEL_VARIANT="${MODEL_VARIANT:-auto}"

# 样本选择参数。
NUM_SAMPLES="${NUM_SAMPLES:-3}"
START_INDEX="${START_INDEX:-0}"
MAX_SEARCH="${MAX_SEARCH:-200}"
SAMPLE_INDICES="${SAMPLE_INDICES:-}"
REFERENCE_FRAME="${REFERENCE_FRAME:--1}"
SEED="${SEED:-42}"
SPLIT="${SPLIT:-val}"
ALLOW_NO_TRACKS="${ALLOW_NO_TRACKS:-}"

auto_select_co3dv2_config() {
  local split="$1"
  local checkpoint_path="$2"
  local suffix=""
  if [[ "$split" == "val" ]]; then
    suffix="_val"
  fi

  case "$checkpoint_path" in
    *co3dv2_5cat*)
      echo "configs/single_co3dv2_5cat${suffix}.yaml"
      ;;
    *)
      echo "configs/single_co3dv2${suffix}.yaml"
      ;;
  esac
}

# 稀疏 tracking 可视化参数。
NUM_POINTS="${NUM_POINTS:-1024}"
NUM_DISPLAY_POINTS="${NUM_DISPLAY_POINTS:-180}"

# 稠密 GT 动态点云可视化参数。
# 0 = 保留每一帧按 DENSE_GT_POINT_SOURCE 选出的全部点。
DENSE_GT_MAX_POINTS="${DENSE_GT_MAX_POINTS:-0}"

# 稠密预测点云可视化参数。
DENSE_PRED_POINT_CLOUD_STRIDE="${DENSE_PRED_POINT_CLOUD_STRIDE:-2}"
DENSE_PRED_VIS_THRESHOLD="${DENSE_PRED_VIS_THRESHOLD:-0.3}"
DENSE_PRED_CONFIDENCE_PERCENTILE="${DENSE_PRED_CONFIDENCE_PERCENTILE:-10}"
DENSE_PRED_QUERY_BATCH_SIZE="${DENSE_PRED_QUERY_BATCH_SIZE:-4096}"
# 100 = 不过滤；80 = 保留最近 80%，mask 掉最远 20%
DENSE_PRED_DEPTH_PERCENTILE="${DENSE_PRED_DEPTH_PERCENTILE:-100}"
# reference frame 上按预测深度过滤 query 格点：只保留最近 N% 的格点。
# 50 = 只用前景/近处一半；100 = 不过滤（全部格点）
DENSE_PRED_QUERY_DEPTH_PERCENTILE="${DENSE_PRED_QUERY_DEPTH_PERCENTILE:-100}"

# GIF 参数。
GIF_FPS="${GIF_FPS:-8}"
DEPTH_STRIDE="${DEPTH_STRIDE:-2}"

# 3D 坐标轴约定。
# 0 = 保持原始 Y
# 1 = 对 3D 图和 PLY 导出翻转 Y
FLIP_Y_AXIS="${FLIP_Y_AXIS:-0}"
# 0 = 不过滤；>0 = mask 掉相机坐标系 Z > MAX_DEPTH 的远距离点
MAX_DEPTH="${MAX_DEPTH:-0}"

case "$DATASET" in
  pointodyssey)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
    CONFIG="${CONFIG:-configs/single_pointodyssey_highres_noaug.yaml}"
    CHECKPOINT="${CHECKPOINT:-/data/zbf/openclaw/d4rt/outputs/pointodyssey_single_gpu_highres_noaug1/checkpoint_latest_55.pth}"
    PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
    RESOLUTION="${RESOLUTION:-256}"
    NUM_FRAMES="${NUM_FRAMES:-48}"
    DENSE_GT_POINT_SOURCE="${DENSE_GT_POINT_SOURCE:-all_finite}"
    ;;
  dynamic_replica)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
    CONFIG="${CONFIG:-configs/single_dynamic_replica.yaml}"
    CHECKPOINT="${CHECKPOINT:-/data/zbf/openclaw/d4rt/outputs/dynamic_replica_single_gpu_0/checkpoint_latest_30.pth}"
    PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_resized}"
    RESOLUTION="${RESOLUTION:-256}"
    NUM_FRAMES="${NUM_FRAMES:-48}"
    DENSE_GT_POINT_SOURCE="${DENSE_GT_POINT_SOURCE:-visible}"
    ;;
  kubric)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
    CONFIG="${CONFIG:-configs/single_kubric.yaml}"
    CHECKPOINT="${CHECKPOINT:-/data/zbf/openclaw/d4rt/outputs/kubric_single_gpu_5/checkpoint_latest_XX.pth}"
    PATCH_PROVIDER="${PATCH_PROVIDER:-auto}"
    RESOLUTION="${RESOLUTION:-256}"
    NUM_FRAMES="${NUM_FRAMES:-48}"
    DENSE_GT_POINT_SOURCE="${DENSE_GT_POINT_SOURCE:-all_finite}"
    ;;
  co3dv2)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
    CHECKPOINT="${CHECKPOINT:-/data/zbf/openclaw/d4rt/outputs/single_co3dv2/checkpoint_latest_20.pth}"
    if [[ -z "${CONFIG:-}" ]]; then
      CONFIG="$(auto_select_co3dv2_config "$SPLIT" "$CHECKPOINT")"
      AUTO_SELECTED_CONFIG=1
    fi
    PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_resized}"
    RESOLUTION="${RESOLUTION:-256}"
    NUM_FRAMES="${NUM_FRAMES:-48}"
    DENSE_GT_POINT_SOURCE="${DENSE_GT_POINT_SOURCE:-all_finite}"
    ALLOW_NO_TRACKS="${ALLOW_NO_TRACKS:-1}"
    ;;
  mixture|mixture_5datasets|mixture_5datasets_blendedmvs)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    CONFIG="${CONFIG:-configs/mixture_5datasets_blendedmvs.yaml}"
    CHECKPOINT="${CHECKPOINT:-/data/zbf/openclaw/d4rt/outputs/mixture_5datasets_blendedmvs_large_3gpu_bs5/checkpoint_latest_41.pth}"
    PATCH_PROVIDER="${PATCH_PROVIDER:-sampled_highres}"
    RESOLUTION="${RESOLUTION:-256}"
    NUM_FRAMES="${NUM_FRAMES:-48}"
    DENSE_GT_POINT_SOURCE="${DENSE_GT_POINT_SOURCE:-all_finite}"
    ;;
esac

# 在 case 之后设最终默认值，确保 co3dv2 的 ALLOW_NO_TRACKS=1 不被覆盖
ALLOW_NO_TRACKS="${ALLOW_NO_TRACKS:-0}"

# 从 CHECKPOINT 路径自动派生 OUTPUT_DIR：<checkpoint目录>/vis_<checkpoint名去掉.pth>
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  _ckpt_dir="$(dirname "$CHECKPOINT")"
  _ckpt_name="$(basename "$CHECKPOINT" .pth)"
  OUTPUT_DIR="${_ckpt_dir}/vis_${_ckpt_name}"
fi

LOG_PREFIX="vis_${DATASET}"

echo "[$LOG_PREFIX] DATASET=$DATASET"
echo "[$LOG_PREFIX] SPLIT=$SPLIT"
echo "[$LOG_PREFIX] PYTHON_BIN=$PYTHON_BIN"
echo "[$LOG_PREFIX] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[$LOG_PREFIX] CONFIG=$CONFIG"
echo "[$LOG_PREFIX] CHECKPOINT=$CHECKPOINT"
echo "[$LOG_PREFIX] MODEL_VARIANT=$MODEL_VARIANT"
echo "[$LOG_PREFIX] OUTPUT_DIR=$OUTPUT_DIR"
echo "[$LOG_PREFIX] PATCH_PROVIDER=$PATCH_PROVIDER"
echo "[$LOG_PREFIX] RESOLUTION=$RESOLUTION"
echo "[$LOG_PREFIX] NUM_FRAMES=$NUM_FRAMES"
echo "[$LOG_PREFIX] NUM_SAMPLES=$NUM_SAMPLES"
echo "[$LOG_PREFIX] START_INDEX=$START_INDEX"
echo "[$LOG_PREFIX] MAX_SEARCH=$MAX_SEARCH"
echo "[$LOG_PREFIX] SAMPLE_INDICES=$SAMPLE_INDICES"
echo "[$LOG_PREFIX] REFERENCE_FRAME=$REFERENCE_FRAME"
echo "[$LOG_PREFIX] NUM_POINTS=$NUM_POINTS"
echo "[$LOG_PREFIX] NUM_DISPLAY_POINTS=$NUM_DISPLAY_POINTS"
echo "[$LOG_PREFIX] DENSE_GT_MAX_POINTS=$DENSE_GT_MAX_POINTS"
echo "[$LOG_PREFIX] DENSE_GT_POINT_SOURCE=$DENSE_GT_POINT_SOURCE"
echo "[$LOG_PREFIX] DENSE_PRED_POINT_CLOUD_STRIDE=$DENSE_PRED_POINT_CLOUD_STRIDE"
echo "[$LOG_PREFIX] DENSE_PRED_VIS_THRESHOLD=$DENSE_PRED_VIS_THRESHOLD"
echo "[$LOG_PREFIX] DENSE_PRED_CONFIDENCE_PERCENTILE=$DENSE_PRED_CONFIDENCE_PERCENTILE"
echo "[$LOG_PREFIX] DENSE_PRED_QUERY_BATCH_SIZE=$DENSE_PRED_QUERY_BATCH_SIZE"
echo "[$LOG_PREFIX] DENSE_PRED_DEPTH_PERCENTILE=$DENSE_PRED_DEPTH_PERCENTILE"
echo "[$LOG_PREFIX] DENSE_PRED_QUERY_DEPTH_PERCENTILE=$DENSE_PRED_QUERY_DEPTH_PERCENTILE"
echo "[$LOG_PREFIX] GIF_FPS=$GIF_FPS"
echo "[$LOG_PREFIX] DEPTH_STRIDE=$DEPTH_STRIDE"
echo "[$LOG_PREFIX] FLIP_Y_AXIS=$FLIP_Y_AXIS"
if [[ "${AUTO_SELECTED_CONFIG:-0}" == "1" ]]; then
  echo "[$LOG_PREFIX] CONFIG auto-selected from CHECKPOINT + SPLIT"
fi

if [[ ! -x "$PYTHON_BIN" && ! -f "$PYTHON_BIN" ]]; then
  echo "[$LOG_PREFIX] Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[$LOG_PREFIX] Config not found: $CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[$LOG_PREFIX] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

ARGS=(
  md/visualize_dynamic_replica_checkpoint.py
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --model-variant "$MODEL_VARIANT"
  --output-dir "$OUTPUT_DIR"
  --patch-provider "$PATCH_PROVIDER"
  --resolution "$RESOLUTION"
  --num-frames "$NUM_FRAMES"
  --num-samples "$NUM_SAMPLES"
  --start-index "$START_INDEX"
  --max-search "$MAX_SEARCH"
  --reference-frame "$REFERENCE_FRAME"
  --num-points "$NUM_POINTS"
  --split "$SPLIT"
  --num-display-points "$NUM_DISPLAY_POINTS"
  --gif-fps "$GIF_FPS"
  --dense-gt-max-points "$DENSE_GT_MAX_POINTS"
  --dense-gt-point-source "$DENSE_GT_POINT_SOURCE"
  --dense-pred-point-cloud-stride "$DENSE_PRED_POINT_CLOUD_STRIDE"
  --dense-pred-vis-threshold "$DENSE_PRED_VIS_THRESHOLD"
  --dense-pred-confidence-percentile "$DENSE_PRED_CONFIDENCE_PERCENTILE"
  --dense-pred-query-batch-size "$DENSE_PRED_QUERY_BATCH_SIZE"
  --dense-pred-depth-percentile "$DENSE_PRED_DEPTH_PERCENTILE"
  --dense-pred-query-depth-percentile "$DENSE_PRED_QUERY_DEPTH_PERCENTILE"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ "$FLIP_Y_AXIS" == "1" ]]; then
  ARGS+=(--flip-y-axis)
fi

if [[ "$ALLOW_NO_TRACKS" == "1" ]]; then
  ARGS+=(--allow-no-tracks)
fi

if [[ "$MAX_DEPTH" != "0" ]]; then
  ARGS+=(--max-depth "$MAX_DEPTH")
fi

if [[ -n "$SAMPLE_INDICES" ]]; then
  ARGS+=(--sample-indices "$SAMPLE_INDICES")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "${ARGS[@]}" "$@"

# 所有数据集额外跑深度图可视化
DEPTH_OUTPUT_DIR="${OUTPUT_DIR}_depth"
echo "[$LOG_PREFIX] Running depth visualization -> $DEPTH_OUTPUT_DIR"
DEPTH_ARGS=(
  md/visualize_co3dv2_depth.py
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --model-variant "$MODEL_VARIANT"
  --output-dir "$DEPTH_OUTPUT_DIR"
  --num-samples "$NUM_SAMPLES"
  --num-frames "$NUM_FRAMES"
  --resolution "$RESOLUTION"
  --patch-provider "$PATCH_PROVIDER"
  --stride "$DEPTH_STRIDE"
  --gif-fps "$GIF_FPS"
  --split "$SPLIT"
  --seed "$SEED"
  --start-index "$START_INDEX"
  --max-search "$MAX_SEARCH"
  --device "$DEVICE"
)
if [[ -n "$SAMPLE_INDICES" ]]; then
  DEPTH_ARGS+=(--sample-indices "$SAMPLE_INDICES")
fi
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "${DEPTH_ARGS[@]}"
