# Roadmap

## v1.1.0 candidate — implemented and measured

Controlled synthetic domain shifts; reproducible augmentation ablations across
existing models and three training seeds; per-class robustness, performance and
failure evidence; engineering explanations and owner competency checklists.
See [release readiness](docs/RELEASE_READINESS_V1_1.md) for actual gates and limits.

## Proposed v1.2.0 — independent scene generalization

Prioritize a versioned, license-reviewed real or higher-fidelity warehouse dataset
with scene/camera/session group splits. Freeze an independent evaluation split
before selecting augmentation settings. Measure class/boundary performance and
scene-level uncertainty with paired bootstrap or repeated group splits. Investigate
why severe dimming collapses minority-class IoU and why TinyUNet seed variation is
large before increasing model count.

Proposed acceptance: traceable data license/hash/group provenance, leakage tests,
independent nominal/shifted scene metrics with uncertainty, retained failed
experiments, target-input preprocessing measurements, and documented mechanisms
supported by experiments. Hardware claims require actual recorded hardware data.

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
