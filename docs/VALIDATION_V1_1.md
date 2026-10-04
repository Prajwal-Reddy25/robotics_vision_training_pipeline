# Validation v1.1

Executed 2026-10-04. This document distinguishes execution checks, measured
experiments and outstanding validation. The [audit](AUDIT_V1_1.md) and
[preimplementation plan](PLAN_V1_1.md) define the starting point and gates.

## Environment and provenance

- Ubuntu/Linux 7.0.0-31, x86_64, Python 3.12.3; AMD Ryzen 9 8940HX, 32 logical CPUs.
- CPU: PyTorch 2.14.0+cpu, NumPy 2.5.3, Pillow 12.3.0, PyYAML 6.0.3.
- Deployment: ONNX 1.23.0, ONNX Runtime 1.30.0, ONNXScript 0.7.2.
- CUDA: RTX 5070 Laptop GPU, driver 595.84, PyTorch 2.14.0+cu130,
  runtime CUDA 13.0. Device access requires execution outside the sandbox.
- Pytest 9.1.1, Ruff 0.16.8. Both isolated project environments pass `pip check`
  with the host ROS `PYTHONPATH` unset. No system package installation was needed.
- Exact installed version records: [CPU](evidence/v1_1/requirements-cpu.txt),
  [GPU](evidence/v1_1/requirements-gpu.txt). They record the executed environments,
  not a portable lock or a claim of clean-install verification on another host.
- Numerical study ran at clean commit `4e7a9cc`, based on owner-approved
  `54f3a2a`. Atomic ledger hardening and validation/report scripts were finalized
  afterward; the trained algorithm/configuration did not change. Later reports
  correctly retain their observed Git dirty state. Source/evidence are committed
  together for review; no released/tagged historical baseline is asserted.

## Executed checks

| Check | Outcome | Evidence |
| --- | --- | --- |
| Baseline CPU suite | 27 passed, 1 CUDA-only skip | audit |
| Final CPU unit/integration suite | 56 passed, 1 CUDA-only skip | [log](evidence/v1_1/cpu_tests.log) |
| Full suite in CUDA environment | 57 passed | [log](evidence/v1_1/gpu_tests.log) |
| Ruff/static analysis | passed | [log](evidence/v1_1/lint.log) |
| Wheel build, editable installs, dependency consistency | passed | [build log](evidence/v1_1/build.log) |
| CI-equivalent two-model/two-arm study | 4 successful runs, all 13 cases | [record](evidence/v1_1/ci_smoke.json) |
| Predeclared release study | 42/42 successful, 546 condition evaluations | [ledger](evidence/v1_1/study.json) |
| Matrix/config/input/artifact/metric review | all passed; IoU recomputed from confusion | [review](evidence/v1_1/review.json) |
| Independent full CPU training repeat | both seed-42 baseline models bitwise identical | [deployment record](evidence/v1_1/deployment.json) |
| TorchScript/ONNX/ORT | both models pass finite/allclose/argmax gates at two shapes and full-test mIoU comparison | deployment record |
| Fresh CUDA FP32/AMP | both models, two epochs each, finite model/optimizer; AMP scale 65,536, six updates | [GPU record](evidence/v1_1/gpu.json) |
| CUDA strict resume | first LiteFCN FP32 check failed; later four-mode collection passed | both attempt records below |
| Synchronized GPU eager/TorchScript benchmark | both trained baseline models measured; CPU/GPU mIoU difference <0.001 | GPU record |
| Optional TensorRT absence path | documented explicit RuntimeError observed | [probe](evidence/v1_1/tensorrt_probe.json) |

All tests from the baseline are retained. New tests check transform identity,
bounds/determinism, homography direction, image/mask interior agreement, visible
occlusion labels/rounded area, legacy resume defaults, paired hashes, complete
repeat-study equality, invalid configs, nonfinite loss/logit rejection, failed-run
preservation, and interrupted atomic ledger writes. Test-only injected failures
are distinguished from the real GPU failure.

## Commands, seeds and criteria

Commands were run from repository root; `.venv` and `.venv-gpu` already existed.
No unrecorded training retries or post hoc hyperparameter changes were used.

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/pytest -q
.venv/bin/ruff check .
env -u PYTHONPATH .venv/bin/python -m pip check
env -u PYTHONPATH .venv-gpu/bin/python -m pip check
env -u PYTHONPATH .venv/bin/python -m pip wheel --no-deps --no-build-isolation . \
  -w /tmp/rv_v1_1_wheel

env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/rv-study \
  --study configs/ablation_v1_1.yaml --output-dir runs/v1_1_ablation
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/rv-study \
  --study configs/ablation_smoke.yaml --output-dir runs/v1_1_ci_smoke
env -u PYTHONPATH .venv/bin/python scripts/review_v1_1.py \
  --study runs/v1_1_ablation/study.json
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/validate_v1_1.py --study runs/v1_1_ablation/study.json \
  --output-dir runs/v1_1_deployment

env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-gpu/bin/python \
  scripts/validate_v1_1.py --study runs/v1_1_ablation/study.json \
  --output-dir runs/v1_1_gpu --gpu --deployment-dir runs/v1_1_deployment
# Initial command exited 1 at the strict LiteFCN FP32 bitwise check.
# Collection was extended to retain strict failures, finish other modes, and
# still exit nonzero if any strict check fails. The original failed assertion
# remains a check; no numerical tolerance replaces bitwise equality.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-gpu/bin/python \
  scripts/validate_v1_1.py --study runs/v1_1_ablation/study.json \
  --output-dir runs/v1_1_gpu_complete --gpu --deployment-dir runs/v1_1_deployment
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-gpu/bin/python -m pytest -q
```

For reproduction choose **fresh output paths**; existing/partial directories are
preserved. Use an isolated Python 3.12 environment, an appropriate PyTorch build,
and the README installation commands. Installed version records can constrain
resolution on a compatible host; the CPU-specific torch wheel needs the CPU
PyTorch index. The captured GPU environment contains both ORT distributions from
the historical setup; this milestone uses CPU ORT and does not validate that
dual-distribution setup as a fresh-install recipe.

Protocol: data seed 17 (180 scenes), optimization seeds 42/43/44, shift seed 2026,
64² RGB, 15 epochs, batch 16, nominal validation checkpoint selection. CUDA smoke
uses data seed 7, training seed 9, 24 scenes at 32², batch 4, two planned epochs.
These short GPU scores are functionality checks, not quality claims.

Criteria: complete matrix without filtered observations; reproducible input hashes;
finite outputs; correct label semantics; exact CPU repeat/resume; export allclose
rtol=1e-4/atol=1e-5 plus ≥99.9% pixel argmax agreement; CPU/deployment and CPU/GPU
mIoU difference <0.001 in smoke comparisons. No positive augmentation gain is
required. CUDA bitwise determinism was never a v1.1 acceptance criterion because
the baseline already documents best-effort nondeterminism.

## Real failed/neutral observations

The first CUDA LiteFCN FP32 resume check failed: max state delta
`1.7881393432617188e-7`, train-loss delta about `7.95e-8`, unchanged validation
mIoU. Scheduler/scaler, loader RNG and Torch RNG match. PyTorch emits
`nll_loss2d_forward_out_cuda_template does not have a deterministic implementation`.
This supports kernel nondeterminism as an explanation, without proving attribution
to one operation. See [failure log](evidence/v1_1/gpu_initial_failure.log),
[computed comparison](evidence/v1_1/gpu_initial_failure.json), and
[original partial report](evidence/v1_1/gpu_initial_partial.json).

The new collection directory retained this attempt and subsequently obtained
bitwise equality in four modes. That outcome does not erase or negate the failure.
Accuracy losses in the ablation are also retained, including LiteFCN combined
augmentation's −4.56 pp mean shifted change. There were no crashed real CPU
ablation attempts and no fabricated failed examples represented as real runs.

## Evidence and outstanding checks

[SHA256.json](evidence/v1_1/SHA256.json) inventories tracked JSON, logs and version
records. Raw generated datasets/checkpoints/exports remain ignored locally;
compact quantitative evidence is committed, without binary model/data payloads.
`scripts/document_v1_1.py` regenerates measurement tables from the same records.
`scripts/review_v1_1.py` requires regenerated local data/checkpoints to verify
artifact bytes and fresh transformed-input hashes, rather than trusting summary text.

Remote GitHub Actions was not run/published; its local smoke and integration paths
passed. TensorRT is not installed and no engine was built. No physical-robot or
real-camera data exists. Calibrated sensor/3D simulation, target end-to-end latency,
long-horizon AMP convergence, clean install on another host, and scene-level
confidence intervals remain outstanding. See [limitations](LIMITATIONS.md).

## Owner competency checklist

- [ ] Reproduce the smoke and full matrix with fresh paths and frozen seeds.
- [ ] Verify metrics from confusion matrices and transformed-input hashes.
- [ ] Explain every success/failure gate and the GPU exception to bitwise guarantees.
- [ ] Distinguish actual execution, simulated data and missing hardware validation.
