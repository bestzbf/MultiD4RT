# Changelog

## v0.5.0 — 2026-10-09

- Add one inference/visualization command for synchronized multi-camera clips, static multiview images and monocular videos/sequences.
- Preserve image aspect ratio, shared frame labels, manifest reference order and raw-pixel query transforms; exclude padding from point exports.
- Export an offline interactive HTML viewer, binary PLY, 3D tracks and scene-flow arrays, with prediction reuse.
- Add optional visualization dependencies and packaged viewer assets; retain the existing network and native checkpoint behavior.

## v0.4.0 — 2026-10-09

- Rename the project, Python package, model and config to MultiD4RT, multid4rt, MultiD4RT and MultiD4RTConfig.
- Focus the release on the model, RGB manifest inference and downstream point/flow APIs.
- Keep checkpoint parameter keys and accept earlier native architecture markers.
- Simplify README and retain detailed model/inference documentation.

Earlier tagged releases remain available in repository history.
