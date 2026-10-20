# Roadmap

## v1.1.0 — released synthetic study

Controlled synthetic domain shifts; reproducible augmentation ablations across
existing models and three training seeds; per-class robustness, performance and
failure evidence; engineering explanations and owner competency checklists.
See [release readiness](docs/RELEASE_READINESS_V1_1.md) for actual gates and limits.

## v1.2.0 — released real-photo evaluation

All 42 frozen models, 80 licensed COCO real photographs, 84 native-ground-truth
real dataset evaluations, all arms/seeds retained. Primary floor and restricted
cabinet/storage metrics reveal large transfer gaps and a reversal of mean synthetic
augmentation signs. No real-data tuning or additional architecture. See
[real-data scope](docs/REAL_DATA_GENERALIZATION_V1_2.md) and
[release readiness](docs/RELEASE_READINESS_V1_2.md).

## Proposed v1.2.1 — release-maintenance tooling

The maintenance branch prepares offline evidence verification, CI gates, portable
CPU constraints, documentation consistency and clearer release presentation.
No experiment, model, dataset or measured result changes. Package version and tag
remain 1.2.0 during review. See [maintenance review](docs/MAINTENANCE_REVIEW_2026_10_20.md).

## Proposed v1.3.0 — representative warehouse adaptation experiment

Obtain versioned human labels for all supported classes, with scene/camera/session
grouped training/tuning/final-test splits and traceable licenses. Freeze a new test
before adaptation; keep the v1.2 final test out of recipe tuning. Compare small,
controlled interventions addressing palette/scene shortcuts or normalization,
without expanding architecture count. Report per-class/boundary metrics, paired
seed results and scene-level uncertainty; retain negative findings.

Acceptance: compatible ontology with documented unknowns, group-leakage tests,
independent final test, reproducible intervention configs, confidence intervals
appropriate to scene sampling, fair nominal/OOD contrasts and all failures.
Physical camera/robot claims require recorded physical observations; latency
requires a stated full preprocessing/inference/postprocessing protocol.

## Subsequent engineering work — not implemented

- Isolate the CUDA loss-reduction nondeterminism observed in v1.1, add a strict
  determinism option with a genuinely deterministic loss path and resume tests.
- Migrate deprecated TorchScript consumers toward `torch.export` with parity gates.
- Evaluate TensorRT only on a compatible deployment stack; keep optional failures clear.
- Add transactional dataset creation and dependency/action pinning as justified
  reproducibility work, not a broad architectural rewrite.
- Embed and enforce data-content identity in checkpoints while defining explicit
  migration/resume behavior for legacy checkpoints without manifest digests.
- Measure end-to-end preprocessing/transfers/postprocessing, cold start, sustained
  load, power/thermal variation and failure/recovery behavior on a stated target.

## Owner competency checklist

- [ ] Explain the v1.1 findings and limitations from raw seed/class measurements.
- [ ] Design group splits that prevent scene/session leakage.
- [ ] Explain the difference between optimization-seed SD and scene bootstrap uncertainty.
- [ ] Describe an experiment that tests the dim-light or seed-variance hypotheses.
- [ ] Define which real data and hardware measurements would support a deployment claim.
