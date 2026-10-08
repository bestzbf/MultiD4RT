# 100k continuation after the corrected-GT migration

Prepared on 2026-10-06 (Asia/Shanghai); server timestamps use UTC.
See [training documentation](../../../docs/training.md#gt-修复后的-100k-续训).

## Why training stopped

The prior job completed its 45,000–45,200 segment, then exited at a validation
guard. It did **not** finish 50k and did not fail a training batch. Both GPU4/5
were idle when inspected.

[Recorded diagnosis](stop_diagnosis.json) compares the same fixed plans with
the corrected Kubric masks and scene normalization:

| Unweighted normalized error | 45,000 baseline | 45,200 | Change |
|---|---:|---:|---:|
| Kubric depth AbsRel | 0.27372 | 0.06531 | −76.1% |
| Kubric C4T6 temporal trajectory | 0.29481 | 0.10560 | −64.2% |
| PointOdyssey temporal displacement | 0.001226 | 0.002297 | +87.4% |
| vKITTI2 dynamic displacement | 0.006857 | 0.010558 | +54.0% |

The old controller treated any unchanged-dataset metric increase over 25% as
a terminal failure, preserving the checkpoints but providing no recovery phase.
The migration had also raised peak LR to backbone/head `5e-7/5e-6`; this is a
plausible contributor to interference, not a separately established cause.

## Resumed run

The complete corrected checkpoint at step 45,200 is restored on physical
GPU4/5, using the giant model, BF16, BS1 per GPU (global BS2), native image
sizes and full valid-pixel/particle supervision. The model configuration stays
`decoder_memory=target_time`, `time_origin=clip_start`; optimizer moments,
rank RNG and sampler cursor are retained for the first recovery segment.

The new LR phase uses peaks `1e-7/1e-6`, a 50-step warmup from 50%, cosine
decay ending at **100,000**, and a 20% minimum. The explicit LR end is extended,
not merely the training step argument. The first phase runs to 45,300 and
executes all four existing fixed validation suites.

[Live training health snapshot](resumed_health.json) records actual resumed
optimizer updates, finite loss/gradients and zero skipped updates. It does not
claim that the recovery has fixed the validation regressions. Continued
convergence must be assessed from the upcoming fixed results.

Server output root:
`/data8/zbf/multicamera4rc_runs/extended_100k_20261005_165948UTC`.

## Automatic continuation

[`continue_extended_training.py`](../../../scripts/continue_extended_training.py)
waits for the recorded probe controller to exit, verifies code/proofs, and
adopts its full checkpoint. It trains another 500 steps, then validates at
2,000-step boundaries through 100k, saving every 1,000 steps and at each segment
end. It preserves the LR schedule instead of restarting warmup each segment.

If the old motion regressions remain over 25%, the prepared
[motion replay configuration](motion_replay.yaml) is used: Kubric 28%,
PointOdyssey 16%, vKITTI2 family 14%; all other sources and every C/T layout
remain intact. All 23 source entries passed
[index verification](replay_index_check.json). A change of sampling quotas
resets only the data cursor; optimizer and rank RNG remain restored. Syn4D
is not activated by this continuation.

The previous, best corrected and historical pre-repair validation panels are
recorded separately. New per-metric regressions above 25%, or geometric mean
regressions above 10%, relative to the previous/best corrected checkpoint cause
a rollback to the accepted segment start and a lower-LR retry. At most two
retries are allowed; failed candidates and evidence are retained. Historical
regressions drive replay and remain visible instead of repeatedly causing an
unconditional stop. Numeric/data/runtime failures still fail explicitly.

[Controller proof](controller_proof.json): **22 tests passed locally and on
B300zbf**, covering adoption, protection against another active job, passing
50k automatically, exact target termination, data-cursor handling, rollback,
LR backoff, and bounded retry exhaustion. A launcher dry run passed for a
52k segment with LR ending at 100k. These are controller checks; the live giant
health snapshot supplies the actual training evidence.

Live state should be read from the server, not inferred from this historical
snapshot:

```text
/data9/zbf/my_dfrt_multicamera4rc/outputs/training_preflight/active_training.json
/data9/zbf/my_dfrt_multicamera4rc/outputs/training_preflight/queued_extended_training.json
<control_dir>/progress.json
<output_root>/latest_accepted_checkpoint.json
<output_root>/latest_evaluated_checkpoint.json
<output_root>/best_checkpoint.json
```
