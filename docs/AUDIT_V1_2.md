# Audit of the frozen v1.1 implementation

2026-10-05, clean HEAD `d9a74a0` on `feature/v1.1.0` before v1.2 work.
Package version 1.1.0. No `v1.1.0` tag exists locally, so the exact commit is
the frozen baseline; no release tag is inferred or created. A `v1.0.0` tag
has since been added and points to `54f3a2a`. Older documents describing its
absence record the historical checkout state, not the current tag inventory.

Inspected source, configurations, tests, CI, documentation, git history, tracked
evidence inventory and local generated artifacts. Fresh CPU baseline checks:
56 passed / 1 CUDA-only skip, Ruff passed. `review_v1_1.py` verifies all 42
checkpoint hashes, identical 13 transformed inputs, resolved study configuration,
timing counts, file sizes and 546 confusion-derived metric records. All 21 tracked
evidence file hashes match their inventory.

Confirmed shifted mIoU: TinyUNet baseline 0.652816, occlusion 0.710097
(+0.057281); LiteFCN baseline 0.593255, occlusion 0.572102 (−0.021153).
Combined augmentation gains 0.047082 for TinyUNet and loses 0.045650 for LiteFCN.
These are three-seed observations on one procedural split, not universal rankings.
Raw negative arms are present. The original CUDA resume failure is present and
its bitwise comparison is false; the later successful collection does not erase it.

Architecture remains a small PyTorch/Pillow package: synthetic manifest/hash
validation, two fixed segmentation models, full-state training/resume, domain
views and predeclared studies, metric/export/backend benchmark CLIs. There is
no real-image loader, ignore-label/shared-ontology evaluator, real data, or
real-image qualitative evidence in the frozen baseline.

Relevant limitations: 27 ID test scenes, class/color shortcuts, large TinyUNet
seed variance, synthetic camera/occlusion proxies, no real generalization,
best-effort CUDA determinism, deprecated TorchScript, TensorRT unavailable,
model-call-only latency, broad dependencies, trusted pickle checkpoints and
resume assuming unchanged dataset contents. Keep these limits visible.

The next change should be evaluation infrastructure and independently sourced
ground truth, not another model or another synthetic training recipe. Existing
frozen checkpoint files are locally available and must be hash-verified before use.
