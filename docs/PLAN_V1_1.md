# v1.1.0 milestone: generalization and domain shift

Plan fixed before implementation, 2026-10-04. Provisional baseline: `54f3a2a`.

Scope: six paired test conditions (lighting, texture, Gaussian blur, Gaussian
noise, planar viewpoint proxy, foreground occlusion), two fixed severities plus
nominal; deterministic per-scene transforms independent of model/training seed;
optional training transforms; a controlled two-architecture, three-seed study.

Non-goals: new architectures, external data, real-camera calibration, claims
of sim-to-real transfer, true 3D simulation, TensorRT expansion, production ROS
integration, safety claims, or dependency-heavy experiment infrastructure.

## Acceptance criteria

1. Preserve baseline generator bytes, model definitions, CLIs, and old default
   augmentation/resume behavior. All baseline tests continue passing.
2. Every shift has deterministic content hashes, unchanged IDs and dimensions,
   bounded RGB, valid labels, and documented geometry/occlusion label semantics.
3. Same held-out scenes and corruption seeds across models/arms/training seeds;
   no training/test leakage, no test-based checkpoint selection.
4. Complete 2 × 7 × 3 training matrix using 15 epochs, data seed 17, training
   seeds 42/43/44, nominal validation checkpoint selection. Arms: no augmentation,
   baseline, baseline+blur, +texture, +viewpoint, +occlusion, all additions.
5. Report nominal and every condition's aggregate confusion matrix, mIoU,
   class IoU, absolute and relative degradation; per-run and aggregate values,
   sample SD and paired baseline deltas. No positive-result filter or required gain.
6. Measure latency and parameter/checkpoint size for every trained checkpoint;
   export/benchmark both baseline seed-42 models with TorchScript and ORT.
   Fresh CUDA/AMP smoke and synchronized GPU benchmarks when device access exists.
7. Verify data/transform determinism and exact repeat training on CPU. Record
   environment, commands, resolved configs, seeds, hashes and failures. CI runs
   a small complete study rather than the expensive release matrix.
8. Documentation distinguishes implemented/tested/simulated/measured/unvalidated;
   engineering notes include principles, tradeoffs, failure modes and interview
   competency. Final review checks source/evidence hygiene and repository state.

## Metrics and experiments

IoU_c = TP_c/(TP_c+FP_c+FN_c). Use existing present-union macro IoU policy,
and report supports to reveal class disappearance under geometric shifts.
Degradation = nominal mIoU − shifted mIoU (negative values retained).
Relative degradation = degradation / nominal (null when nominal=0).
Equal-weight mean across 12 shifts is a declared synthetic stress index, not a
real-world risk estimate. Training-seed sample SD measures optimization variation,
not uncertainty over real scenes. Compare each augmentation arm to baseline with
paired seed deltas; all training settings and compute budgets fixed.

Latency: batch 1, one CPU thread, 20 warmups, 100 timed model calls, FP32,
mean/p50/p95 and raw samples; CUDA synchronization around GPU timing.
Report parameter bytes separately from full training-checkpoint and deployment
file sizes. Dense fixed-shape inference timing is per checkpoint, not a claim of
condition-dependent computational speed.

## Required changes

- Add a small transformation module and evaluation dataset wrapper; retain the
  nominal data generator unchanged. Extend augmentation with zero-valued defaults.
- Add domain evaluation and a strictly validated study runner/CLI, partial-progress
  ledger and explicit failed-run records. Never silently reuse or overwrite runs.
- Add transform geometry/label/determinism tests, configuration failures,
  degradation arithmetic, study integration/failure preservation and repeat checks.
- Extend CPU CI; retain current export/resume tests. Record the executed dependency
  versions without imposing a broad new dependency-management architecture.
- Add README links, CHANGELOG, ROADMAP, validation/benchmarks/domain/ablation/
  limitations/release-readiness documents, compact tracked JSON evidence and
  ENGINEERING_NOTES with an Owner competency checklist.

No release tag or publication is authorized by this plan. Recommend v1.1.0 only
after the acceptance checks pass, with the historical baseline discrepancy explicit.
