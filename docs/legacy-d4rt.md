> 历史 D4RT 文档，保留原入口供旧模型使用。当前多相机项目请从[仓库 README](../README.md)进入。下文权重、分辨率和变体仅适用于旧 D4RT；命令均从仓库根目录运行。

# D4RT — Dense 4D Reconstruction & Tracking

D4RT is a video transformer that, from a short clip, jointly predicts per-frame
**dense depth / point clouds**, **camera pose & intrinsics**, and **2D/3D point tracks**.
This repo holds the model, the training code, and a small self-contained inference package
under [`test/`](../test/) for running the model on your own video.

The new **[MultiCamera4RC package](../multicamera4rc/README.md)** implements the separate
4RC-based synchronous multi-camera architecture: unknown K/pose, shared frame indices,
same-camera attention relations, target-time multi-view memory, differentiable world
geometry, anchored dense/sparse 3D tracks, and per-camera visibility. It includes its own
native H5/dataset adapters, training/inference entry points and CPU tests; existing D4RT checkpoints and pipelines
retain their current interface.

---

## Repository layout

| Path | What it is |
|---|---|
| `models/` | Model definition. `models/d4rt.py` + `create_d4rt(...)` is the entry point; `encoder.py` (VideoMAEv2 backbone), `decoder.py`, `dense_tracking.py`. |
| **`test/`** | **Inference package** — run the model on a video / image folder. **See [Run on your video](#run-on-your-video) below.** |
| `train_mixture.py` + `train_mixture_*.sh` | Training entry point and per-variant launchers. |
| `models/`, `losses/`, `utils/` | Model, training losses, shared geometry/camera helpers. |

Model variants (by encoder embed dim): `base` (768) · `large` (1024) · `huge` (1280) · `giant` (1408).
The current best checkpoint is a **`giant`** model.

---

## Setup

```bash
conda activate d4rt          # Python 3.10, PyTorch + CUDA
```

Checkpoints are plain `torch.save` dicts with a top-level `"model"` key. Checkpoints saved from a
`torch.compile()`-wrapped model carry an `_orig_mod.` prefix on every key; the loader strips it
automatically, so both compiled and eager checkpoints load with `strict=True`.

---

## Pretrained weights

Released on ModelScope: **https://www.modelscope.cn/models/binfengzou/d4rt**

| File | Size | `MODEL_VARIANT` | Notes |
|---|---|---|---|
| `giant.pth` | ~14.5 GB | `giant` | Best checkpoint (recommended). |
| `large.pth` | ~4.6 GB | `large` | Mid-size variant; near-`giant` pose accuracy at a fraction of the size. |
| `base.pth` | ~1.6 GB | `base` | Smaller / faster variant. |

Download (CLI or Python):

```bash
# CLI
pip install modelscope
modelscope download --model binfengzou/d4rt giant.pth --local_dir ./weights
modelscope download --model binfengzou/d4rt large.pth --local_dir ./weights
modelscope download --model binfengzou/d4rt base.pth  --local_dir ./weights
```

```python
# Python
from modelscope.hub.snapshot_download import snapshot_download
path = snapshot_download("binfengzou/d4rt")   # downloads all files; returns the local dir
```

> The model variant is **not** stored in the checkpoint. You **must** set `MODEL_VARIANT` to match
> the file you downloaded (`giant` for `giant.pth`, `large` for `large.pth`, `base` for `base.pth`),
> otherwise `load_state_dict` fails with a shape mismatch.

---

## Run on your video

A ready-to-run sample video ships with the repo at
[`test/examples/sample_video.mp4`](../test/examples/sample_video.mp4) (324 frames, 832×464).

### Quick start — full-scene reconstruction (`modeA_full_scene.ply`)

[`test/visualization/run_reconstruct.sh`](../test/visualization/run_reconstruct.sh) is the
canonical example. It seeds the dense query grid from **several keyframes** (so regions
not visible in the reference frame are still reconstructed), decodes every seed into one
unified camera frame, voxel-dedups, and writes a single canonical point cloud
**`modeA_full_scene.ply`** plus a camera fly-through GIF (and a per-frame `modeB` variant).
It is a thin env-var wrapper around `test/visualization/visualize_multiseed_reconstruction.py`.

```bash
# Bundled sample, giant weights (default MODEL_VARIANT=giant):
CHECKPOINT=./weights/giant.pth \
  bash test/visualization/run_reconstruct.sh test/examples/sample_video.mp4
```

Outputs land in `OUTPUT_DIR` (default `outputs/reconstruct_<checkpoint>_sample_video/`):

| File | What it shows |
|---|---|
| `modeA_full_scene.ply` | the full static scene as **one** canonical point cloud (reference camera frame) |
| `modeA_camera_flythrough.gif` / `modeA_static.png` | that single cloud, camera flown through it |
| `modeB_refframe_scene.ply` | per-frame reconstruction (each frame deduped on its own visibility) |
| `modeB_per_frame.gif` / `modeB_static.png` | the per-frame variant |
| `summary.json` | seed frames, voxel size, per-seed point counts, dedup stats |

Key knobs: `SAMPLING_MODE` (`uniform` default — spread seeds across the whole clip),
`NUM_SEED_FRAMES` (`8`), `VOXEL_SIZE` (`0.01`, dedup edge), `RECON_MODE` (`A` / `B` / `both`),
`CUDA_VISIBLE_DEVICES`, `OUTPUT_DIR`.

### Per-frame visualization (`run_video.sh`)

[`test/run_video.sh`](../test/run_video.sh) is the simpler alternative: it runs the model on a
video **or** a folder of image frames and writes per-reference-frame dense point clouds
(`pred_dense_reference_frame_NNN.ply`), 2D/3D point tracks, and GIFs — it does **not**
produce `modeA_full_scene.ply` (that is the reconstruction route above).

```bash
# Bundled sample:
CHECKPOINT=./weights/giant.pth \
  bash test/run_video.sh test/examples/sample_video.mp4

# Your own video file, a folder of frames, or the base weights:
CHECKPOINT=./weights/giant.pth MODEL_VARIANT=giant bash test/run_video.sh /path/to/your_video.mp4
CHECKPOINT=./weights/giant.pth MODEL_VARIANT=giant bash test/run_video.sh /path/to/frames_dir
CHECKPOINT=./weights/base.pth  MODEL_VARIANT=base  bash test/run_video.sh /path/to/your_video.mp4
```

The model consumes a window of `NUM_FRAMES` (default 48) frames. If your input has fewer
frames, the last frame is repeated to fill the window; if it has more, a 48-frame window starting at
`START_INDEX` is used.

### Options (env vars)

These apply to `run_video.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `CHECKPOINT` | best giant ckpt | path to the `.pth` weights |
| `MODEL_VARIANT` | `giant` | **must match the checkpoint** (`giant` / `base` / `large` / `huge`) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU to use |
| `OUTPUT_DIR` | derived from checkpoint + input name | where results are written |
| `NUM_FRAMES` | `48` | frames fed to the model |
| `START_INDEX` | `-1` (auto) | first frame index of the window |
| `SAMPLING_MODE` | `contiguous` | how frames are picked from the input |
| `RESOLUTION` | `256` | model input resolution |
| `QUERY_GRID_SIZE` | `32` | density of the query-point grid for tracking |
| `DENSE_VIS_THRESHOLD` | `0.3` | visibility cutoff for dense point cloud |
| `DENSE_CONFIDENCE_PERCENTILE` | `10` | drop bottom-N% confidence points |
| `DENSE_MOTION_PERCENTILE` | `85` | threshold for selecting dynamic (moving) points |
| `GIF_FPS` | `8` | output GIF frame rate |

Example with overrides:

```bash
CHECKPOINT=./weights/giant.pth MODEL_VARIANT=giant \
CUDA_VISIBLE_DEVICES=1 NUM_FRAMES=48 GIF_FPS=12 \
OUTPUT_DIR=outputs/my_clip_vis \
  bash test/run_video.sh /path/to/your_video.mp4
```

### Outputs

Written to `OUTPUT_DIR`:

| File | What it shows |
|---|---|
| `pred_dense_reference.gif` / `.ply` | dense predicted point cloud (animated + exportable) |
| `pred_dense_motion_reference.*` | dense cloud with **dynamic / moving** points highlighted |
| `pred_3d_reference.gif` / `.png` / `.ply` | sparse 3D query-point reconstruction |
| `tracks_2d_overlay.gif` / `.png` | 2D point tracks overlaid on the video |
| `reference_queries.png` | the query points on the reference frame |
| `summary.json` | quantitative stats: visible-point ratio, mean confidence, track displacement, reference-frame depth, run config |

---

## Benchmark evaluation (default test case)

[`test/run_eval_sintel.sh`](../test/run_eval_sintel.sh) is the default way to score a
checkpoint on the **Sintel** benchmark — it reports depth (AbsRel, scale-only `S` and
scale+shift `SS`), point-cloud L1, and camera pose (ATE / RPE-T / RPE-R) over the
paper's 14-sequence `final`-pass subset. It is a thin env-var wrapper around
`test/eval_sintel_checkpoint.py` (sets `PYTHONPATH` for `test/utils/`, then calls the
Python script).

```bash
# Score the giant weights on Sintel (GPU 7). Point SINTEL_ROOT at your Sintel copy.
CUDA_VISIBLE_DEVICES=7 MODEL_VARIANT=giant \
CHECKPOINT=./weights/giant.pth \
SINTEL_ROOT=/path/to/sintel \
  bash test/run_eval_sintel.sh
```

Per-scene JSON lines stream to stdout; the aggregate metrics and a `summary.json` land
in `OUTPUT_DIR` (derived from the checkpoint path by default). The eval prints a timing
line at the end: `model loaded in <s>` (one-time checkpoint load) and
`compute (all scenes) <s>`.

**Decoding is frame-batched for speed.** The query interface is per-query independent
given `(coords, t_src, t_tgt, t_cam)` and the (B=1) encoder features, so
`decode_frames_batched` flattens *all* frames' query grids into one fused query stream,
accumulates on-GPU, and copies to CPU once — instead of looping the decoder frame by
frame with a GPU→CPU sync per chunk. This collapses ~1.5k tiny decode launches into a
few dozen large ones. On a `giant` checkpoint over the 14 Sintel scenes this cut the
compute phase from **231 s → 169 s (≈1.37×)** with the GPU reaching 100% utilization;
metrics are numerically identical (only AbsRel-SS moved by 1e-4, float32 reduction-order
noise). Raise `QUERY_BATCH_SIZE` (default 4096) to pack more queries per launch —
`QUERY_BATCH_SIZE=65536` runs a 256×256 grid in a single launch and was used for the
numbers above.

| Variable | Default | Meaning |
|---|---|---|
| `CHECKPOINT` | best giant ckpt | path to the `.pth` weights |
| `MODEL_VARIANT` | `giant` | **must match the checkpoint** |
| `SINTEL_ROOT` | `/data3/dataset/sintel` | Sintel dataset root |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU to use |
| `QUERY_BATCH_SIZE` | `4096` | queries per decode launch (raise to pack the grid) |
| `OUTPUT_DIR` | derived from checkpoint | where per-scene + `summary.json` are written |

---

## Training (optional)

```bash
# Multi-GPU mixture training (giant + DUSt3R metric branch). Edit the env vars at the top
# of the launcher (RESUME, EPOCHS, LR, CONFIG, ...) then:
bash train_mixture_5datasets_3gpu_cos_planned_vitg_dust3r.sh

# Or call the entry point directly:
python train_mixture.py --config configs/<mixture>.yaml \
  --batch-size 8 --num-frames 48 --resolution 256 --num-queries 2048 \
  --lr 1e-4 --epochs 300 [--compile] [--dust3r-metric-branch]
```

Notes:
- Set `RESUME=""` for a cold start; otherwise the launcher warm-starts with a short LR warmup
  (`RESUME_WARMUP_STEPS`).
- `D4RT_BUILDER_START_METHOD` must be `forkserver` (not `fork`).

See `python train_mixture.py --help` for the full set of flags.
