#!/usr/bin/env bash
# 对多个 checkpoint 跑4个数据集的 val 可视化（点云 + 深度图）
# 用法：bash tools/run_visualize_all_datasets.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
SPLIT="${SPLIT:-val}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"

CHECKPOINTS=(
  "outputs/mixture_4datasets/checkpoint_latest_35.pth"
  "outputs/mixture_3datasets_finetune_v2_gpu5/checkpoint_latest_50.pth"
  "outputs/mixture_3datasets_conf_static_form0_gpu6/checkpoint_latest_35.pth"
  "outputs/single_co3dv2_5cat/checkpoint_latest_50.pth"
  "outputs/dynamic_replica_single_gpu_0/checkpoint_latest_30.pth"
)

DATASETS=(pointodyssey dynamic_replica kubric co3dv2)

for CKPT in "${CHECKPOINTS[@]}"; do
  _ckpt_name="$(basename "$(dirname "$CKPT")")__$(basename "$CKPT" .pth)"
  echo ""
  echo "====== Checkpoint: $CKPT ======"

  for DS in "${DATASETS[@]}"; do
    OUT_BASE="outputs/vis_all/${_ckpt_name}/${DS}"
    echo "--- [$DS] -> $OUT_BASE ---"

    _start_index=0
    [[ "$DS" == "co3dv2" ]] && _start_index=13

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      CHECKPOINT="$CKPT" \
      NUM_SAMPLES="$NUM_SAMPLES" \
      SPLIT="$SPLIT" \
      PYTHON_BIN="$PYTHON_BIN" \
      OUTPUT_DIR="$OUT_BASE" \
      START_INDEX="$_start_index" \
      bash "$SCRIPT_DIR/run_visualize_checkpoint.sh" "$DS" || echo "  [WARN] $DS failed, continuing"

    echo "--- [$DS] done ---"
  done
done

METRICS_MD="outputs/vis_all/metrics_table.md"
METRICS_CSV="outputs/vis_all/metrics_table.csv"
echo ""
echo "====== Aggregating metrics table ======"
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_visualization_metrics.py" \
  --root outputs/vis_all \
  --output-md "$METRICS_MD" \
  --output-csv "$METRICS_CSV" || echo "  [WARN] metrics aggregation failed"

echo ""
echo "====== All done. Results in outputs/vis_all/ ======"
echo "====== Metrics markdown: $METRICS_MD ======"
echo "====== Metrics csv: $METRICS_CSV ======"
