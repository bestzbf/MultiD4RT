#!/usr/bin/env bash
# 统一 DUSt3R 启动器：用 VARIANT={base|large|giant} 一键切换模型规模，
# bs / builder 容量 / 数据加载档按 variant 自动配置，dust3r two-branch loss 默认开。
#
#● 用法（直接运行即可，GPU 在下方"用户可编辑区"改，日志自动记到 logs/）：
#   VARIANT=base  bash train_mixture_dust3r.sh                        # base
#   VARIANT=large bash train_mixture_dust3r.sh                        # large
#   VARIANT=giant bash train_mixture_dust3r.sh                        # giant(vitg)
#   bash train_mixture_dust3r.sh                                      # 默认 giant
#   GPUS=0,1,2,3 VARIANT=large bash train_mixture_dust3r.sh           # 临时换 GPU
#   BATCH_SIZE=28 VARIANT=large bash train_mixture_dust3r.sh          # 临时覆盖 bs
#   RESUME=<ckpt> RESUME_WARMUP_STEPS=400 VARIANT=giant bash ...      # 暖启动(恢复optimizer/epoch)
#   PRETRAIN=<ckpt> VARIANT=large bash ...                            # 只加载权重，从 epoch 0 重训
#   （日志会自动写到 logs/dust3r_<variant>_<时间>.log，无需手动 tee）
#
# 设计说明：
#   - 底层 train_mixture_5datasets_3gpu_cos_planned.sh 已原生支持 VARIANT case
#     （自动选 VIDEOMAE_MODEL / ENCODER_PATCH_SIZE / USE_VIDEOMAE_V2_INIT），
#     也自带所有数据集 root 的本地默认值。本启动器只补底层缺失或需按 variant
#     区分的项，不再重复 model/patch/root（数据全本地，COS fallback 已无意义）。
#   - dust3r 行为：METRIC_DATASETS 内的数据集走 z̄=z 共享归一化（保留 metric scale），
#     其余（如 co3dv2，SfM 伪 GT 尺度不可信）各自独立归一化（论文默认 scale-invariant）。
#   - 三个 variant 跑的是同一套 10-dataset dust3r mixture，只有 bs / builder 容量
#     / 加载档不同。base 在此跑 mixture（不是 tartanair-only 陪跑实验）。
#   - 注意：另有 train_mixture_5datasets_3gpu_cos_planned_base.sh 是独立的
#     tartanair-only 陪跑实验（和 vitg 同机并跑、关 staging、低优先级），
#     与本脚本的 base mixture 无关，两者别混。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VARIANT="${VARIANT:-giant}"
export VARIANT

# ============================================================================
#  用户可编辑区：GPU 选择
# ----------------------------------------------------------------------------
#  GPUS  = 用哪几张卡（逗号分隔的物理卡号），进程数(NPROC_PER_NODE)自动=卡数。
#  改这一行即可，无需命令行传参；也可临时用 GPUS=0,1 bash ... 覆盖。
GPUS="${GPUS:-1,2,3,4,5,6}"
# ============================================================================

# --- 冷启动默认：不 resume（large/giant 想暖启动就显式传 RESUME=<ckpt>）---
# 暖启动建议同时设 RESUME_WARMUP_STEPS=400 做一次 0->peak_lr 线性 warmup，
# 保护权重渡过 dust3r 新 loss 配置带来的梯度跃升。
# PRETRAIN：只加载权重、从 epoch 0 重训（optimizer/epoch 不接续，区别于 RESUME 的完整恢复）；
#           默认空 = 不加载，模型按 VARIANT 从 videomae-v2-* 初始化。
export PRETRAIN="${PRETRAIN:-}"
export RESUME="${RESUME:-}"
export RESUME_WARMUP_STEPS="${RESUME_WARMUP_STEPS:-100}"
export EPOCHS="${EPOCHS:-300}"



export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPUS}"
# 进程数 = 选用的卡数（按 GPUS 里的逗号分隔计数）。已显式设 NPROC_PER_NODE 则尊重之。
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(printf '%s' "$GPUS" | awk -F, '{print NF}')"
fi
export NPROC_PER_NODE

# ============================================================================
#  自动日志：所有输出同时写到 logs/dust3r_<variant>_<时间>.log 并保留终端显示。
#  设 NO_AUTO_LOG=1 可关闭（例如自己外层已经 tee）。
if [[ "${NO_AUTO_LOG:-0}" != "1" && -z "${_DUST3R_LOG_ACTIVE:-}" ]]; then
  mkdir -p "$ROOT_DIR/logs"
  _LOG_FILE="$ROOT_DIR/logs/dust3r_${VARIANT}_$(date +%Y%m%d_%H%M%S).log"
  export _DUST3R_LOG_ACTIVE=1
  echo "[train_mixture_dust3r] 日志记录到: $_LOG_FILE"
  exec > >(tee -a "$_LOG_FILE") 2>&1
fi
# ============================================================================

# --- DUSt3R two-branch normalization：默认开 ---
export DUST3R_METRIC_BRANCH="${DUST3R_METRIC_BRANCH:-1}"
# metric 列表：mixture 里所有带真实/伪真实 metric 单位的数据集。
# co3dv2 是 SfM 伪 GT 尺度不可信，不在内（保持 scale-invariant）。
export METRIC_DATASETS="${METRIC_DATASETS:-scannet,scannetpp,dynamic_replica,kubric,blendedmvs,pointodyssey,tartanair,vkitti2,mvssynth,waymo}"

# --- 底层 cos_planned.sh 没有、但所有 variant 都需要的项 ---
# 1) NFS/共享盘上多 builder 并发读 H5 precompute 会触发 ENOLCK(errno 37)，关掉 HDF5 文件锁。
export HDF5_USE_FILE_LOCKING=FALSE
# 2) rank0 data_wait 超过 threshold(120s)+kill_after(600s) 时自动 SIGKILL，
#    触发 NCCL 错误传播让其它 rank 退出，避免整个训练永久 hang。
#    600s 兜底：validation(~180s)+存 14GB ckpt(~215s) 贴近旧的 300s 阈值，曾误杀；
#    train_mixture.py 的 watchdog 已修(验证逐 batch + 存盘所有 rank 刷新)，这里再留余量。
export D4RT_TRAIN_WATCHDOG_KILL_AFTER_S="${D4RT_TRAIN_WATCHDOG_KILL_AFTER_S:-600}"
# 3) torch.compile：底层默认 0，large/giant 都要 1（否则丢编译加速、训练明显变慢）。
export USE_COMPILE="${USE_COMPILE:-1}"
# 4) 慢样本（waymo tfrecord 等）build 超过此秒数就跳过，不堵 planned-order 前窗口。
#    三个 variant 统一 15s（giant step 虽慢，多等也只是个别慢样本，影响很小）。
export D4RT_PLANNED_FORCE_SKIP_S="${D4RT_PLANNED_FORCE_SKIP_S:-15}"
# 5) planned-order 乱序前瞻窗口：正好要的样本没 ready 时，允许返回 [N, N+此值] 内
#    已 ready 的样本，不死等。窗口大只是"可选范围大"、不强制乱序，对训练无影响。
#    统一 2400（大窗口对 giant 无坏处，对消费快的 base/large 更不易饿 GPU）。
export D4RT_PLANNED_RELAXED_LOOKAHEAD="${D4RT_PLANNED_RELAXED_LOOKAHEAD:-2400}"
# 6) index cache 预热：底层默认 1，每次启动都在 torchrun 前重跑 warm_index_cache 脚本
#    (内部再调 build_co3dv2_frame_cache.py)。但 index cache(.index_cache_5datasets_local,
#    覆盖全 10 数据集) + co3dv2 frame anno cache(.../co3dv2_frame_anno/*.npz) 建一次就够，
#    重跑既慢又依赖这两个 git 未跟踪脚本(缺失会 exit 1 直接终止启动)。默认关，
#    缓存已建好时 builder 直接读；首次建缓存 / 新增数据集时显式 AUTO_WARM_INDEX_CACHE=1。
export AUTO_WARM_INDEX_CACHE="${AUTO_WARM_INDEX_CACHE:-0}"

# --- spool 放 /dev/shm（内存 tmpfs），避免 ext4 HDD 随机 IO 成瓶颈 ---
# 底层默认已是 /dev/shm/d4rt_spool，这里显式保留以便一眼可见 / 易覆盖。
export SPOOL_DIR="${SPOOL_DIR:-/dev/shm/d4rt_spool}"

# --- staging 只对真正在 NFS 上的数据集生效 ---
# co3dv2(/data4) 和 scannetpp(/data5) 在 NFS，staging 靠 sequence_locality 把一次
# NFS 读摊销到 ~96 次本地 hardlink 命中，有收益。其余数据集都在本地 NVMe
# (/data2,/data3)，staging 是纯 copy 开销 + 抢 builder，不走。
export SAMPLE_STAGE_DATASETS="${SAMPLE_STAGE_DATASETS:-co3dv2,scannetpp}"

# --- 唯一真正按 variant 区分的：bs / builder 容量 / 加载档 / 输出目录 ---
# model / patch_size / videomae-v2-init 全部交给底层 VARIANT case 自动设，不在此重复。
case "$VARIANT" in
  base|vitb|vit-b)
    export VARIANT=base
    # base 是最小/最快的模型，step 时间最短 → 数据消费速率最高 → 最容易饿 GPU，
    # 所以给到接近安全内存上限的 builder 数（~36/rank × 6 ranks ≈ 1TB，3TB 机器可容）。
    # bs 先给保守值 40，首跑确认显存后可上调（base 显存占用远小于 large/giant）。
    export BATCH_SIZE="${BATCH_SIZE:-100}"
    export BUILDER_WORKERS="${BUILDER_WORKERS:-36}"
    export NUM_WORKERS="${NUM_WORKERS:-8}"
    export MAX_SPOOL_BYTES_GB="${MAX_SPOOL_BYTES_GB:-100}"
    export OUTPUT_DIR="${OUTPUT_DIR:-outputs/10dataset_base_dust3r}"
    ;;
  large)
    export BATCH_SIZE="${BATCH_SIZE:-25}"
    # 7卡×14bs 下 18 builder 不够，epoch 中期 spool 会耗尽；
    # waymo/pointodyssey 等慢数据集会占满 worker，提到 32 留补充余量。
    export BUILDER_WORKERS="${BUILDER_WORKERS:-32}"
    export NUM_WORKERS="${NUM_WORKERS:-8}"
    export MAX_SPOOL_BYTES_GB="${MAX_SPOOL_BYTES_GB:-100}"
    export OUTPUT_DIR="${OUTPUT_DIR:-outputs/10dataset_large_dust3r}"
    ;;
  giant|vitg)
    export VARIANT=giant
    export BATCH_SIZE="${BATCH_SIZE:-15}"
    export BUILDER_WORKERS="${BUILDER_WORKERS:-16}"
    export NUM_WORKERS="${NUM_WORKERS:-8}"
    export OUTPUT_DIR="${OUTPUT_DIR:-outputs/10dataset_vitg_dust3r}"
    ;;
  *)
    echo "[train_mixture_dust3r] ERROR: 不支持的 VARIANT=$VARIANT（可选 base|large|giant）" >&2
    exit 2
    ;;
esac

# --- 帧顺序：SHUFFLE=1 切到"打乱时间顺序"的随机配置 ---
# 默认(SHUFFLE 未设/为 0)用常规 config（48 帧时间升序）。
# SHUFFLE=1 时改用 *_random.yaml：每个 clip 的 48 帧随机 permute（无序多视图），
#   factory.py 会随之自动关掉 motion boundary。
# 已显式传入 CONFIG= 时一律尊重用户，不覆盖。
SHUFFLE="${SHUFFLE:-0}"
if [[ -z "${CONFIG:-}" ]]; then
  case "$SHUFFLE" in
    1|true|yes|on)
      export CONFIG="configs/mixture_5datasets_cos_planned_random.yaml"
      echo "[train_mixture_dust3r] SHUFFLE=$SHUFFLE -> CONFIG=$CONFIG (帧时间顺序打乱, motion boundary 自动关)"
      ;;
    *)
      export CONFIG="configs/mixture_5datasets_cos_planned.yaml"
      ;;
  esac
else
  echo "[train_mixture_dust3r] CONFIG 已显式指定=$CONFIG (忽略 SHUFFLE=$SHUFFLE)"
fi

echo "[train_mixture_dust3r] VARIANT=$VARIANT  BATCH_SIZE=$BATCH_SIZE  BUILDER_WORKERS=$BUILDER_WORKERS"
echo "[train_mixture_dust3r] GPUS=$GPUS  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[train_mixture_dust3r] DUST3R_METRIC_BRANCH=$DUST3R_METRIC_BRANCH  USE_COMPILE=$USE_COMPILE"
echo "[train_mixture_dust3r] FORCE_SKIP_S=$D4RT_PLANNED_FORCE_SKIP_S  RELAXED_LOOKAHEAD=$D4RT_PLANNED_RELAXED_LOOKAHEAD"
echo "[train_mixture_dust3r] AUTO_WARM_INDEX_CACHE=$AUTO_WARM_INDEX_CACHE (0=直接读现成缓存; 首次建缓存/新增数据集设 1)"
echo "[train_mixture_dust3r] OUTPUT_DIR=$OUTPUT_DIR"
echo "[train_mixture_dust3r] RESUME=${RESUME:-<unset, 冷启动>}  RESUME_WARMUP_STEPS=$RESUME_WARMUP_STEPS"
echo "[train_mixture_dust3r] PRETRAIN=${PRETRAIN:-<none>}"
echo "[train_mixture_dust3r] ⚠ 冷启动：metric 数据集 loss_3d 初期偏高是 dust3r 分支抓 scale 的预期行为；"
echo "[train_mixture_dust3r]   监控 loss_log.jsonl 的 pred_abs_depth_mean 是否朝 GT 收敛。"

exec bash "$ROOT_DIR/train_mixture_5datasets_3gpu_cos_planned.sh" "$@"
