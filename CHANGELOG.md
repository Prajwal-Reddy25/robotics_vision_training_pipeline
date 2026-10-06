# Changelog

## 1.2.0 — release candidate, 2026-10-06

- Added official real COCO panoptic subset preparation, license/hash/area/geometry checks,
  native-resolution evaluation, explicit binary floor and restricted storage label mappings.
- Froze all 42 v1.1 checkpoints; measured 84 real dataset evaluations at 64²/128²,
  retaining all seven arms/three seeds and substantial failed-transfer results.
- Added paired seed statistics, comparable ID−real gaps, descriptive synthetic-shift
  associations, per-class/source-category and boundary diagnostics, deterministic failure gallery.
- Preserved failed/neutral arms and prior CUDA bitwise-resume limitation; added offline
  data/metric/geometry/statistics tests and checked interrupted-ledger continuation.
- Verified native real metrics from two independently repeated CPU configurations;
  real-input TorchScript/ORT parity, CPU/CUDA suites, fresh FP32/AMP/resume/benchmarks
  and wheel packaging. Added technical notes, owner checklists and bounded evidence.
- No new architecture/runtime dependency, real-data tuning, tag or publication.

The scope is partial-ontology real-photo evaluation, not full warehouse deployment.
Local baseline is exact commit `d9a74a0`; the v1.1 tag is absent. The v1.0 tag now
points to `54f3a2a`; earlier absence records below describe historical checkout state.

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
