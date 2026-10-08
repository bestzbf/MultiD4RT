# Changelog

## v0.2.0 — 2026-10-08

First standalone MultiCamera4RC source release, exported from `bestzbf/my_dfrt` commit `404d4fdfe547574bb2cbfa6a283a1dae917c7a15`.

- Synchronized uncalibrated multi-camera geometry, scene flow and 3D tracking based on 4RC base points plus displacement.
- Explicit camera identity and shared integer frame indices; variable camera/time layouts, aspect-preserving image handling, geometry and visibility heads.
- Native mixed-dataset adapters, dense supervision, dynamic GT validation and Syn4D mesh-based motion supervision.
- DDP training, complete checkpoint continuation, learning-rate trial and fixed validation controllers.
- Sintel Final 14 and ScanNet test100 monocular geometry evaluation with all valid evaluation-grid pixels, frozen plans and complete 55.5k results.
- Architecture, data, training and evaluation documentation; existing third-party license/provenance notices retained.
- Standalone repository URLs and Python distribution name `multicamera4rc` at version `0.2.0`. The import name and CLI remain `multicamera4rc`.

See [release validation and scope](releases/v0.2.0/README.md) for the source manifest, installation checks and evaluation limits.
