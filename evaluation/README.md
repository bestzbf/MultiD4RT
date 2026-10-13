# Published 60k checkpoint comparison

This folder evaluates the byte-identical published MultiD4RT checkpoint against
the complete released OmniX network. It contains inference and evaluation
utilities, with no training entry point. The main package and release README
are unchanged.

## Completed shared rerun

| Test | Coverage | Metric | MultiD4RT 60k | OmniX release |
|---|---|---|---:|---:|
| TAPVid-3D PStudio | full_eval 106 videos, 954 complete T16 clips | APD3D ↑ | 6.6445% | 21.3096% |
| TAPVid-3D PStudio | same clips | camera-coordinate EPE ↓ | 0.22713 m | 0.12794 m |
| Panoptic multiview | all 6 released sequences, C4T150 | Δavg ↑ | 46.965% | 21.135% |
| Panoptic multiview | same fixed queries | MTE ↓ | 22.515 cm | 39.020 cm |
| DexYCB multiview | all 10 sequences, C4T24/25 | Δavg ↑ | 64.872% | 68.125% |
| DexYCB multiview | same fixed queries | MTE ↓ | 6.208 cm | 5.592 cm |
| MV-Kubric | all 30 official test sequences, C4T24 | Δavg ↑ | 54.0353% | 54.1430% |
| MV-Kubric | same fixed queries | MTE ↓ | 56.1277 | 78.9550 |
| Sintel Final | 14 sequences, 37 complete T16 clips | AbsRel S ↓ | 0.35509 | 0.38286 |
| Sintel Final | same clips | AbsRel SS ↓ | 0.27407 | 0.28740 |

MV-Kubric error units are **original scene units ×100**, following its official
scorer; do not relabel them as real-world centimeters. Multiview Panoptic uses
the official `any` group; DexYCB and MV-Kubric use `dynamic-static-mean`. Averages
use the upstream rounded per-sequence results.

These are paired reruns with documented preprocessing, rather than direct
copies of paper tables. OmniX's paper-specific complete evaluation wrapper is
not available in its release. The original 4RC/CUT3R depth scorer, TAPVid-3D
scorer, MVTracker scorer and dataset readers are preserved and fingerprinted.
No test sequences were removed based on either model's results.

## Input conditions

**PStudio:** RGB and supplied 2D queries. Each released 150-frame video produces
nine complete 16-frame clips; the six-frame tail is recorded, with no repeated
padding frames. Queries are every GT-visible point at the first frame of each
clip, projected into that frame using annotation intrinsics. Future labels are
only scoring inputs. Each method converts its world trajectories into its own
predicted target-camera frame. The untouched TAPVid-3D scorer applies the
official depth-adaptive thresholds after scaling annotation intrinsics to an
image short side of 256, exactly as upstream `evaluate_model.py`, and one global
median scale per clip. The RGB network still uses its original shared
preprocessing, and camera-coordinate XYZ stays in meters. EPE
uses exactly the same float64 scale. The headline aggregate gives each of the
106 videos equal weight. No GT pose registration or 3D source anchoring is used.

**Multiview:** RGB networks plus provided calibration for camera-center Sim(3)
registration and source-pixel projection, and the official **supplied 3D query
XYZ**. A constant offset anchors each sampled trajectory to that query at its
query time. No future track is fitted. This input condition is
`rgb_calibrated_gauge_and_supplied_3d_queries`; it should be separated from RGB-D
baselines and from uncalibrated 2D-query reconstruction. All 21,504 fixed queries
and all original sequence timesteps are retained. Model predictions are dense;
the sparse query counts belong to the official evaluation annotations.

**Sintel:** shared RGB-only inputs, original corrected 2015 depth/camera files,
the 4RC/CUT3R depth function with GT depth <70m, and a single scale or scale/shift
fit per 16-frame clip. The headline is a clip macro average; equal-scene depth
averages are also saved. GT camera centers make Sim(3) undefined for 21 clips,
so ATE/RPE translation are disclosed on the same 16 identifiable clips for both
methods. Rotation RPE is additionally computed on all 37 clips after bounded
SO(3) cleanup of rounded SDK rotations and float32 predictions. The maximum SDK
orthogonality deviation is approximately 2.5e-6.

PStudio/Sintel use long side 504; multiview uses long side 266. RGB retains the
entire field of view and its aspect ratio, then receives bottom/right padding
to patch14. Both models receive byte-identical prepared RGB. Exact native,
content and padded dimensions are recorded per case. Query/intrinsic mapping
uses the same interpolation convention as RGB. These resolutions are an
explicit shared-rerun choice, not a claim that paper default preprocessing is
identical.

## Weights and execution semantics

The required MultiD4RT file is `multid4rt_giant_step60000.pt`, SHA256:

```
1161d6bfaa0d277b22108c5b9e803b9e1c36e289f98799e35bd59a9b821197e0
```

The checkpoint's execution scope is restored: **decoder_memory=target_time,
time_origin=clip_start**. All cameras/times are encoded; the released decoder
uses the target-time camera group. New-model defaults do not override this
historical execution graph. Only gradient checkpointing is disabled for
inference. All 1,106 state tensors load strictly.

OmniX `eccv_release.ckpt`, SHA256:

```
871638d02bf8591ccb24de9da28bf862a0c1fa10a8640442e76cb58face343ed
```

All 1,119 `net.*` tensors load strictly into pristine upstream commit
`5c6c8928f85552d78bb80cd8b9bbc36f7b4987db`. Camera/video slots and local/global
time indices preserve its synchronized-video sampler semantics; its full
original ST target axis is retained. Canonical first-camera slots correspond
to the shared physical target times. No time-token repair or learned adapter
is inserted into the baseline.

For B300 compatibility, FA3 uses PyTorch SDPA and deformable attention uses the
upstream grid-sample reference. The reference is numerically audited. Speed
claims against the paper's native compiled kernels are deliberately omitted.

OmniX originally exceeds B300 memory at C4T150. The query memory adapter omits
unused auxiliary copies, projects independently captured ST outputs immediately,
bounds tokenwise MLP temporaries, streams the **exact maximum over every target
slot** in the offset head, and calls the unchanged final trajectory function for
requested source images. Every input, attention operation, learned head and
physical reference time remains present. On actual C2T4, C4T8 and C4T24 inputs,
the dense trajectories for every source and canonical physical target time,
base points, intrinsics and poses match the original complete forward
**bit for bit**. Native proofs are required before
running or reusing any cached predictions. The 40 earlier completed predictions
retain their original archive hashes, and array/official-score equality is
verified during reuse. The upstream source checkout is never edited.

OmniX has no released occlusion predictor. Its all-visible storage placeholder
is explicitly excluded from the AJ3D/OA comparison. Δavg/APD and visible-point
errors remain comparable.

## Camera count diagnostic

Three sequence IDs, **1, 15, 30**, are fixed independently of result quality.
T8 uses cached first-frame queries with a future visible observation; point IDs
and GT visibility from reference cameras 0–3 remain identical at every input
camera count. The original reader's expanded RGB and calibration are checked
against its four-view output. This diagnostic is separate from the complete
30-sequence, T24 benchmark. Its headline below uses the `any` point group.

| Cameras | 60k Δavg ↑ | OmniX Δavg ↑ | 60k MTE ↓ | OmniX MTE ↓ |
|---:|---:|---:|---:|---:|
| 4 | 49.0367% | 38.7833% | 31.3367 | 61.3767 |
| 6 | 50.9833% | 38.8567% | 27.6600 | 53.9200 |
| 8 | 52.4167% | 39.2567% | 25.6000 | 52.8167 |
| 10 | 53.2267% | 38.3567% | 24.8067 | 52.0400 |

Both models execute ten-camera input. This tests variable camera counts and
their effect on accuracy; it does not establish a hard maximum for either
model or full-dataset superiority at ten cameras.

## Reproduction

Use the release package dependencies, plus NumPy, Pillow, OpenCV, SciPy, evo,
OmegaConf/Hydra and the upstream dataset-reader dependencies. Evaluation runs
in an isolated environment. Set `OMNIX_SOURCE`, `DA3_SOURCE`, and `OMNIX_DEPS`
to pristine source/dependency directories when evaluating OmniX.

```bash
python -m evaluation.run_multiview \
  --method multid4rt --checkpoint /path/multid4rt_giant_step60000.pt \
  --root /data6/benchmarks \
  --plan /data6/benchmarks/evaluation/protocols/tracking-main.json \
  --output /path/results/multid4rt --device cuda:0 --resume

python -m evaluation.check_omnix_queries \
  --checkpoint /path/eccv_release.ckpt \
  --omnix-runner evaluation/native_omnix.py \
  --output /path/omnix-native-parity.json --device cuda:1

python -m evaluation.run_multiview \
  --method omnix --checkpoint /path/eccv_release.ckpt \
  --omnix-runner evaluation/native_omnix.py \
  --parity-report /path/omnix-native-parity.json \
  --plan /data6/benchmarks/evaluation/protocols/tracking-main.json \
  --output /path/results/omnix --device cuda:1 --resume

python -m evaluation.compare_multiview --directory /path/results
```

`prepare_tapvid` downloads the official RGB-inclusive NPZ archives and verifies
the complete split; `launch_tapvid` runs paired T16 evaluation and refuses a
full aggregate if any case or RGB hash differs. `run_sintel` and
`summarize_sintel` retain exact plans and saved geometry. `run_camera_sweep`
performs the separately labeled fixed-query diagnostic. `download_reports` and
`build_report` create the offline HTML/CSV/JSON report. All failure traces and
preflight snapshots are preserved in the remote campaign.

Remote evidence: `/data6/benchmarks/evaluation/multid4rt60k_comparison_20261013`.
Local report: `outputs/official_comparison_60k_20261013/comparison.html` under the
paper workspace, outside the release checkout.

The original PStudio `queries_xyt` field contains some timestamp/projection
inconsistencies. Its audit is preserved separately. The evaluated T16 protocol
does not consume that field; it derives each clip's starting XY from the
visible XYZ at that clip's first frame. Official 3D tracks and visibility are
unchanged, and no annotation-based case filtering is applied.

## Baselines that cannot be reported as reruns

No runnable **official D4RT** model code/weights were found on its project page
or the publisher's repositories as checked on 2026-10-13. OpenD4RT and other
community reimplementations are distinct models; none is labeled official
D4RT here. Its paper numbers can be cited as literature references, but not
inserted into this paired-rerun table. N3DV requires a shared novel-view renderer
and held-out RGB protocol; it is not a 3D tracking benchmark and was not assigned
invented PSNR/SSIM/LPIPS results.

Sources: [OmniX](https://github.com/yanqinJiang/OmniX),
[TAPVid-3D](https://github.com/google-deepmind/tapnet),
[MVTracker](https://github.com/ethz-vlg/mvtracker),
[4RC](https://github.com/Luo-Yihang/4RC),
[evo](https://github.com/MichaelGrupp/evo),
[D4RT](https://d4rt-paper.github.io/).


## PStudio scoring correction and geometry diagnosis

The initial APD3D values **3.2823% / 13.9657% are withdrawn**. The adapter called
upstream `metrics.py` with original-resolution intrinsics, omitting the
short-side-256 normalization in upstream `evaluate_model.py`. At 360×640, the
omitted factor is `256/360`; the old distance thresholds were 1.40625 times too
strict. Re-scoring all 954 clips of both releases gives **6.6445% / 21.3096%**.
All predicted XYZ arrays, queries, GT, weights, EPE and other benchmarks remain
unchanged. Historical reports are preserved under
[legacy_raw_intrinsics](results/60k_20261013/tapvid16_pstudio_full_eval/legacy_raw_intrinsics/).

`rescore_tapvid_resolution` validates unchanged inference ASTs, GT archive
hashes, prediction hashes and query identities before recording the correction.
`check_tapvid_resolution` executes the unchanged official caller functions on
an 8 mm nonzero-error regression and eight real cached clips. An identity-only
scorer check could not detect this bug. Future inference requires original
image dimensions and records the scoring resolution in its protocol contract.

The remaining model gap is primarily initial geometry. Across all 954 clips,
60k's effective focal ratios are 0.805 (x) and 0.790 (y), with source-position
EPE 22.5 cm versus OmniX's 13.1 cm. The intrinsic regression agrees with the
actual camera head on four fixed clips chosen by plan indices 0,318,636,953.
On those clips, changing only the source rays to GT intrinsics, while keeping
predicted depth, motion and target poses, reduces 60k mean EPE from 26.7 to
8.0 cm. **This GT intervention is a diagnostic, not a benchmark result.**
Native-resolution and patch-grid-resolution controls do not remove the bias.
The 67k checkpoint retains the issue on the same four clips; these are not a
full 67k benchmark. No model architecture or ongoing training was changed.
See [full diagnosis and evidence](results/60k_20261013/pstudio_diagnosis_20261013/diagnosis.md).
