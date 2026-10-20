# Changelog

## Unreleased — maintenance review, 2026-10-20

- Refresh release status, README results/gallery, reproduction guidance and prepared
  GitHub Release copy; preserve original experiment dates and historical audits.
- Add a standard-library offline verifier of frozen v1.2 arithmetic, hashes,
  seed statistics, CSV and attribution consistency, with corruption tests and CI.
- Add local documentation-link validation and portable historical CPU constraints.
- Include the required configuration/script/documentation/evidence context in Python
  source distributions, while excluding raw data, generated runs and checkpoints.
- Keep package version 1.2.0 and the existing v1.2.0 tag unchanged. These maintenance
  additions are not present in that original tag. A v1.2.1 tooling patch is proposed,
  subject to maintainer approval; no datasets, results or checkpoint files change.

## 1.2.0 — 2026-10-06 (GitHub Release published 2026-10-20)

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
- No new architecture/runtime dependency or real-data tuning. The original work
  stopped before publication; the owner subsequently tagged and published it.

The scope is partial-ontology real-photo evaluation, not full warehouse deployment.
The released tag is `v1.2.0` at `856e057`; baseline tag `v1.1.0` points to `d9a74a0`.
Tag `v1.0.0` points to `54f3a2a`. Experiment dates above are preserved; publication
was later. Earlier absence records below describe their historical checkout state.

## 1.1.0 — 2026-10-04 (subsequently tagged v1.1.0)

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

At the 2026-10-04 review, no release tag or remote publication had been created;
package version was aligned to 1.1.0. The tag now identifies `d9a74a0`.

## Provisional stable baseline — `54f3a2a`

Historical checkout observation: the owner requested continuation from a released/
tagged v1.0.0, but that checkout had no tags or remotes and package version 0.1.0. The owner approved
HEAD as the provisional baseline. This record does not assert that any of the
three existing commits is the missing release tag.

Existing features include synthetic segmentation data, TinyUNet/LiteFCN, CPU/CUDA
training, AMP, full-state checkpoints/resume, TorchScript/ONNX/ORT, optional
TensorRT integration, automated tests and CPU/GPU benchmark evidence.
