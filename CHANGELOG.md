# Changelog

## v0.3.0 — 2026-10-09

Inference and evaluation release, curated from v0.2.0.

- Keep the model, native checkpoint compatibility, aspect-preserving data adapters and GT recipes.
- Remove training entry points, losses, optimizer/recovery/LR controllers, training configurations and runtime audit history.
- Remove the duplicated legacy D4RT model/data/tool tree.
- Install monocular, native dynamic and static multiview benchmarks inside the Python package; evaluate physical errors directly.
- Replace training demos with random-weight tiny inference demos and document inference/evaluation usage.
- Retain the complete frozen ScanNet test100 and Sintel Final14 monocular results and third-party notices.

## v0.2.0 — 2026-10-08

Initial standalone full source snapshot, exported from `bestzbf/my_dfrt` commit `404d4fdfe547574bb2cbfa6a283a1dae917c7a15`. The original snapshot and release assets remain available at the v0.2.0 tag.
