# Syn4D integration audit — 2026-10-06

Implementation and usage: [Syn4D integration](../../../docs/syn4d.md).

The adapter is integrated with native dense supervision, variable camera/time
layouts, mixed sampling, training, and fixed validation. These checks establish
data and pipeline compatibility. The GPU run used the **tiny** preset and does
not establish convergence, final model quality, or memory feasibility for the
production giant model. The prepared mixture has **not** been activated for
production training.

## Official reference

- [Syn4D project](https://jzr99.github.io/Syn4D/)
- [Official repository](https://github.com/jzr99/Syn4D)
- [Published data format](https://huggingface.co/datasets/Syn4D/Syn4D/blob/main/data/README.md)
- [Released visualizer reader](https://huggingface.co/datasets/Syn4D/Syn4D/blob/main/code/visualizer/syn4d_track.py)

The official repository still lists its training dataloader as pending. The
comparison therefore executes the released visualizer's `_get_views` and mesh
reconstruction functions, disables its crop/resize and UI initialization, and
compares native pixels. The helper that lifts depth to world points is supplied
by the audit script; the official camera and mesh animation/recovery code is
executed independently of the new adapter.

Reference file on B300zbf: `/data6/Syn4D/code/visualizer/syn4d_track.py`.
SHA256: `6c4ffd29625545ce4b1fca06e6e2c2dd401ed72c3e371432f0530dbdabd493b2`.

## Available data and supervision

Two downloaded archives were extracted with `scripts.prepare_syn4d`:

- `/data6/Syn4D/data/syn4d_v1_stride_5/bigoffice_v1.tar.zst`
- `/data6/Syn4D/data/syn4d_sim/cave_sim.tar.zst`

[Ready inventory](ready_inventory.json) contains 37 sequences: V1 has 4 train
and 1 validation; Sim has 28 train, 3 validation and 1 test. This inventory
checks files and metadata; it is not a pixel audit of all 37 sequences. Other
unextracted archives remain outside the current sampling pool.

The split is the adapter's deterministic sequence hash, not an official Syn4D
benchmark partition. Camera/stride variants of the same physical sequence share
the split. Different sequences may reuse scene or object assets.

The checked object mesh files are present. BEDLAM body/clothing meshes are
missing from this download. V1 retains valid depth and source geometry on human
pixels, but unknown human motion is masked from trajectory, displacement,
velocity and static-motion losses. Existing object/environment motion remains
usable. Real human motion must be re-audited after adding those external assets;
the present body-transform coverage is an analytic fixture test. Visibility
labels are conservative target-depth consistency estimates, not independent
official visibility annotations.

## Official reader comparison

[Full measurements](official_comparison.json): **12 cases on 4 physical
sequences**, covering C1T3, C4T3, C8T2 and C1T16, late reference times, distinct
query times/cameras and animation wrapping.

| Measurement | Result |
|---|---:|
| Compared valid pixel–time 3D points | 73,393,765 |
| Largest per-query mean error | 2.4007812e-6 m |
| Largest per-query P99 error | 4.8799072e-6 m |
| Largest individual point error | 2.2851629e-4 m |
| Additional valid pixel–times beyond the official mask | 0 |

RGB is compared byte for byte; depth, intrinsics and camera poses are checked
numerically. Track comparison applies the same world reflection and shared
reference-camera frame. Normalized depth, tracks and displacements are checked
against a common scene scale. All cases passed. Comparisons cover labels valid
in both readers, and therefore do not validate the missing human meshes.

## Tests and real training pipeline

[Test results](test_results.json): the complete project suite passed locally
and remotely with **287 tests** each. After the final MP4 frame-number guard and
geometry-only shard isolation change, all **16 adapter tests** passed again on
both machines. Coverage includes analytic camera/mesh geometry, native padding,
time gaps, unknown GT gradient masking, visibility, variable C/T collation,
fixed validation, mixed configuration generation, archive extraction and a
model forward/backward/checkpoint cycle.

[DDP health](mixed_ddp_health.json): real GPU 4/5 run, 2 ranks, BS1 per GPU,
6 optimizer steps, native-resolution C2T2 dense supervision. Kubric, Syn4D V1
and Syn4D Sim each supplied 4 rank batches. Losses and gradients were finite,
no update was skipped, and the step-6 checkpoint contains both rank RNG states.
This is a bounded compatibility smoke test, not a convergence experiment.

[Fixed validation health](evaluation_health.json): C2T4 on one held-out
sequence from each Syn4D subset passed. There are 32 training and 4 validation
sequences with zero sequence overlap; both validation clips loaded and ran the
tiny checkpoint. The mean loss of 7.4300 is recorded only to make the run
traceable, not as evidence of trained-model quality.

Server outputs are preserved at
`/data8/zbf/multicamera4rc_runs/syn4d_integration_20261006/`.
The detailed comparison is also at
`/data9/zbf/my_dfrt_syn4d_20261005/outputs/syn4d_audit/official_comparison.json`.

## Prepared mixture

[Full mixed configuration](full_mixture_syn4d.yaml) was generated from
`/data8/zbf/multicamera4rc_runs/dome_recovery_20261005/data.yaml` and all 25
source entries were successfully indexed. The 23 existing entries retain their
relative weights and together receive 90%; Syn4D V1 and Sim receive 5% each.
These are initial configurable proportions, not experimentally optimized ones.

The usable server copy is
`/data8/zbf/multicamera4rc_runs/syn4d_integration_20261006/full_mixture_syn4d.yaml`.
Changing to this sampling pool on resume requires `--reset-data-cursor`; model
and optimizer state can still be restored. No production training controller
or active configuration was changed by this integration.
