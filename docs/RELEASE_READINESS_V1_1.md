# v1.1.0 release-readiness review

Review date: 2026-10-04. Recommendation: **v1.1.0**, with the evidence boundaries
below. Development branch: `feature/v1.1.0`. No tag or remote publication created.
The owner-approved provisional stable baseline is `54f3a2a`; the claimed historical
v1.0.0 tag does not exist in this checkout and was not manufactured.

## Acceptance review

| Plan gate | Result |
| --- | --- |
| Preserve existing models/generator/default augmentation and tests | passed; architectures and generator unchanged, additions zero by default, legacy resume configs normalized |
| Six paired shifts/two severities, valid masks, deterministic IDs/content | passed; transform/homography/occlusion tests plus fresh content hash review |
| Shared held-out scenes, no shifted-test checkpoint selection | passed; all 42 runs share 27 test IDs and identical 13 condition hashes; checkpoint selection remains nominal validation |
| Full predeclared 2×7×3 matrix | passed; 42 successful runs, 15 epochs each, seeds 42/43/44, no filtered observations |
| Quality/class/degradation/seed metrics including losses | passed; 546 measured cases, confusion-derived metrics reviewed, per-seed and paired deltas retained |
| Model size/latency/deployment/GPU execution evidence | passed; 42 CPU eager measurements, two-model TS/ONNX/ORT parity, synchronized GPU benchmarks and finite FP32/AMP execution |
| CPU reproducibility/failure handling/CI smoke | passed; bitwise full-training repeats and exact resume; injected failure/interruption tests; four-run CPU study smoke |
| Documentation/engineering ownership/evidence/hygiene | passed; audit/plan, README/changelog/roadmap, technical notes, measured reports, owner checklists, failure artifacts and limitations |

Final local checks: CPU suite **56 passed / 1 CUDA-only skip**, CUDA-environment
suite **57 passed**, Ruff passed, dependency consistency passed, wheel build
passed, `git diff --check` passed. The [validation record](VALIDATION_V1_1.md)
links commands/logs, seeds, environment and success/failure criteria.

Review scripts independently verify the planned run matrix, resolved settings,
all transformed-input identities, local checkpoint byte hashes, timing sample
counts, model file sizes and mIoU from confusion matrices. Tracked evidence hashes
are inventoried in [SHA256.json](evidence/v1_1/SHA256.json). Reports were generated
from raw results and their references checked for consistency.

## Important non-gating failure and limits

One real CUDA LiteFCN FP32 exact-resume attempt failed bitwise equality by
1.79e-7 while metrics and recovered state matched. The warning-mode CUDA baseline
never guaranteed strict determinism. The strict check remains; collection retains
failures and exits nonzero rather than changing its equality tolerance. Both the
failed attempt and later passing collection are preserved. This rules out a general
GPU-bitwise claim but does not invalidate CPU reproducibility or domain metrics.

The ablation's negative effects are results, not failed acceptance gates: LiteFCN
combined augmentation loses 4.56 pp mean shifted mIoU. Strong TinyUNet seed
variation and severe-light failures are prominent findings. No test-selected arm
is promoted as a universally improved default.

Unvalidated: TensorRT execution, remote CI, calibrated 3D/sensor simulation,
real-camera/physical-robot generalization, scene-level confidence intervals,
long-horizon AMP convergence and target end-to-end latency. The optional TensorRT
absence path was exercised and fails explicitly. See [limitations](LIMITATIONS.md).

## Repository review

- Changes remain within the existing package/config/test/CI/documentation layout;
  no new runtime dependencies or model architectures were added.
- Inspected source diff, retained baseline tests, verified installed CLIs and
  reproducibility commands, built the wheel, checked documentation links/metrics.
- Quantitative JSON/log/version evidence is deliberately tracked (about 2.3 MiB
  in total); the approximately 1.9 MiB study ledger contains all 42 runs and 546
  confusion/metric records. These are intentional evidence, not model binaries.
- Generated data, model/checkpoint/export binaries, build products, virtual
  environments, caches and original session junk remain ignored and untracked.
  No credentials or private files were added; an added-file credential-pattern
  review and large-file inventory were performed.
- Local commits preserve recoverable implementation and evidence checkpoints.
  After the final evidence/documentation commit, the worktree is clean. No tag,
  push, history rewrite or privileged/system modification was performed.

Professional proposed notes are in [RELEASE_NOTES_V1_1.md](RELEASE_NOTES_V1_1.md).
The next proposed milestone is independent scene/camera/session generalization
with licensed real or higher-fidelity data and scene-level uncertainty, as detailed
in [ROADMAP.md](../ROADMAP.md).

## Owner competency checklist

- [ ] Explain the release problem, design and raw measured results without relying
  on a single successful seed or omitting negative results.
- [ ] Reproduce a study and verify a claim using config/ID/hash/confusion evidence.
- [ ] Explain inverse homography and visible occlusion label handling in source.
- [ ] Explain deterministic CPU resume and the preserved GPU failure's implications.
- [ ] Separate implemented, tested, simulated, measured and unvalidated capabilities.
- [ ] Design an independent next experiment rather than tune to this test report.
