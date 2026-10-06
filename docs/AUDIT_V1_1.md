# Forensic audit before v1.1 work

2026-10-04; inspected clean `main` at `54f3a2a`, all tracked source/tests/configs,
CI, manifests, documentation, local result JSON, and three-commit history.
No files were changed during inspection. No local tags or remotes exist.
The requested v1.0.0 tag cannot be inspected; package metadata says 0.1.0.
The owner authorized HEAD as the **provisional baseline**. No historical tag
will be manufactured or baseline results relabeled.

| Area | Finding |
| --- | --- |
| Architecture | Python package and seven CLIs; Pillow scene generator → manifest/hash gates → PyTorch loaders → AdamW/cosine training → metrics/export/benchmark. No ROS, launch files, or external dataset. |
| Features | Four-class synthetic warehouse segmentation, TinyUNet (29,948 parameters), LiteFCN (2,804), paired flip/brightness/contrast/noise, CUDA AMP, atomic format-3 full-state checkpoints, exact resume, TorchScript, checked dynamic ONNX, CPU ORT, optional Torch-TensorRT. |
| Validation | Fresh baseline: 27 passed, 1 CUDA-only skip; Ruff passed. Existing tests include bitwise CPU resume, leakage/hash/path failures, metric edge cases, dynamic export and end-to-end lifecycle. CPU CI includes both architectures. |
| Evidence | Local baseline artifacts reproduce documented seed-42 test mIoU 0.833952/0.598296. Earlier GPU report contains executed CUDA/AMP/ORT probes and numerical-parity failures. Fresh read-only GPU probe detects RTX 5070 Laptop outside sandbox. TensorRT is absent. |
| Assumptions | Fixed RGB color/class association, 64² top-down shapes, four classes, trusted pickle checkpoints, generator-level random splits; validation chooses best epoch. |
| Limits | 180 images, only 27 held-out scenes, single seed, no real-camera data; approximate pixel AP; latency excludes preprocessing/transfers; CUDA determinism warns rather than fails. |
| Weak coverage | No controlled domain shift, no multi-seed or augmentation study, no long-horizon AMP or TensorRT parity test, no transactional dataset creation. |
| Debt | Broad dependency ranges, deprecated TorchScript/export compatibility APIs, unpinned CI action revisions, generated evidence previously ignored, no release changelog/roadmap, version/tag discrepancy. Exact resume compares manifest path rather than an embedded content digest and assumes immutable data. |
| Best improvement | Paired severity-controlled domain-shift analysis and predeclared augmentation ablations, retaining all outcomes with hashes/configs/seed dispersion. |

Baseline command: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q` (4.89 s).
Existing generator and model architectures should be preserved.
