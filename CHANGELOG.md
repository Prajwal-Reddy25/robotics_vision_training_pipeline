# Changelog

## 1.1.0 — release candidate, 2026-10-04

- Added deterministic paired lighting/texture/blur/noise/viewpoint-proxy/occlusion
  evaluations at two fixed severities, retaining class support, confusion matrices,
  class IoU and nominal-relative degradation with source/transformed hashes.
- Added optional blur/texture/viewpoint/occlusion training augmentation with zero
  defaults and backward-compatible comparison of legacy resume configurations.
- Added predeclared seven-arm, two-model, three-seed studies; partial-progress
  ledgers are atomic, failed attempts/tracebacks persist, and failed studies exit
  nonzero. No test-driven arm selection or successful-only filtering.
- Added finite training-loss/evaluation-logit diagnostics and raw benchmark timings.
- Measured 42 training runs / 546 condition evaluations; retained accuracy losses
  alongside gains, CPU repeat-training equality, dynamic deployment parity and
  fresh CUDA/AMP execution/benchmarks. Preserved a real failed GPU bitwise-resume
  attempt as evidence of best-effort CUDA determinism.
- Expanded tests/CPU CI and documented technical principles, owner competencies,
  measured evidence, versions, limits, release review and next work.

No release tag or remote publication was created. Package version is aligned to
1.1.0 for the candidate.

## Provisional stable baseline — `54f3a2a`

The owner requested continuation from a released/tagged v1.0.0. This checkout has
no tags or remotes and its original package version was 0.1.0. The owner approved
HEAD as the provisional baseline. This record does not assert that any of the
three existing commits is the missing release tag.

Existing features include synthetic segmentation data, TinyUNet/LiteFCN, CPU/CUDA
training, AMP, full-state checkpoints/resume, TorchScript/ONNX/ORT, optional
TensorRT integration, automated tests and CPU/GPU benchmark evidence.
