# NVIDIA GPU validation

Validation date: 2026-09-24

This report records execution on the available NVIDIA GPU. It supplements the CPU-focused
technical audit in `AUDIT_REPORT.md`; it does not replace that report. Generated datasets,
checkpoints, exports, and raw benchmark JSON remain Git-ignored. No driver, system CUDA package,
tag, release, or remote repository was changed.

## Outcome

- TinyUNet and LiteFCN eager inference, training, CUDA AMP, evaluation, and synchronized
  benchmarking executed successfully.
- Both TorchScript artifacts executed on CUDA and were bitwise equal to eager CUDA at the two
  tested shapes.
- Exact CUDA resume initially failed because checkpoint `map_location="cuda"` also relocated CPU
  RNG byte states. The fix normalizes global RNG, CUDA RNG, and DataLoader generator states to CPU
  before passing them to PyTorch's state APIs. The resumed TinyUNet state then matched the
  uninterrupted state bit for bit.
- ONNX Runtime 1.30 selected its CUDA provider and ran convolutional compute on CUDA, but the
  exported graphs require CPU fallback for some shape/concatenation operations. Predictions were
  highly consistent, but CUDA ORT logits did not satisfy the exporter's strict CPU equivalence
  tolerance. This path is therefore **validated with numerical and hybrid-execution caveats**, not
  certified as fully GPU-only or strictly numerically equivalent.
- PyTorch warned that CUDA `nll_loss2d` has no deterministic implementation in this stack. The
  repository deliberately requests deterministic algorithms in warning-only mode. Same-stack
  resume was bitwise reproducible in the tested run, but strict general CUDA determinism is not
  claimed.

## Environment and dependency verification

| Item | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| VRAM | 8,151 MiB |
| Compute capability | 12.0 (`sm_120`) |
| Driver | 595.84, open kernel module |
| Kernel | Linux 7.0.0-31-generic x86_64 |
| Python | 3.12.3 |
| PyTorch | 2.14.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| cuDNN | 9.24.0 (`92400`) |
| PyTorch compiled architectures | `sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, `sm_120` |
| ONNX Runtime | 1.30.0 |
| ORT providers | TensorRT, CUDA, CPU (availability only; TensorRT was not exercised) |

The normal `.venv` remains the audited CPU environment. GPU wheels were installed in the ignored
`.venv-gpu` directory. `pip check`, with the host ROS `PYTHONPATH` removed, reported no broken
requirements. A synchronized CUDA matrix multiplication completed with finite output before the
project tests began.

`nvidia-smi` and CUDA were not visible inside the restricted execution sandbox because its
`/dev/nvidia*` devices were hidden. Read-only diagnostics and GPU commands were consequently run
with direct device access. Outside that sandbox, the GPU, driver, and CUDA runtime operated
normally.

Environment creation used only virtual-environment packages:

```bash
python3 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --upgrade pip
.venv-gpu/bin/python -m pip install torch==2.14.0
.venv-gpu/bin/python -m pip install -e ".[all]"
.venv-gpu/bin/python -m pip install onnxruntime-gpu==1.30.0
env -u PYTHONPATH .venv-gpu/bin/python -m pip check
```

## CUDA training and AMP

The existing 24-sample, 32×32 smoke dataset and CLI/config override mechanism were used. These
short runs validate execution and checkpoint state; their validation scores are not model-quality
claims.

| Model | Precision | Epochs | Last train loss | Last validation loss | Finite model/optimizer | AMP scaler |
| --- | --- | ---: | ---: | ---: | --- | --- |
| TinyUNet | FP32 | 2 | 1.314397 | 1.499229 | yes | disabled |
| LiteFCN | FP32 | 2 | 1.456476 | 1.374401 | yes | disabled |
| TinyUNet | CUDA AMP | 1 | 1.472199 | 1.497866 | yes | 65,536; healthy |
| LiteFCN | CUDA AMP | 1 | 1.510669 | 1.363458 | yes | 65,536; healthy |

Both AMP checkpoints contained finite model and AdamW state tensors. Each scaler completed three
updates without backoff (`growth_tracker=3`, scale 65,536). One epoch is enough to detect basic
overflow/NaN and scaler integration defects, but not long-horizon AMP convergence drift.

Representative commands:

```bash
env -u PYTHONPATH .venv-gpu/bin/rv-train \
  --config configs/smoke.yaml \
  --set experiment.name=gpu_tiny_fp32 \
  --set training.device=cuda \
  --set training.epochs=2 \
  --set model.architecture=tiny_unet

env -u PYTHONPATH .venv-gpu/bin/rv-train \
  --config configs/smoke.yaml \
  --set experiment.name=gpu_tiny_amp \
  --set training.device=cuda \
  --set training.amp=true \
  --set model.architecture=tiny_unet
```

Equivalent runs were made with `model.architecture=lite_fcn`.

### Checkpoint recovery

TinyUNet resumed from `epoch_000.pt` into a separate run and completed epoch 1. Compared with the
uninterrupted two-epoch run:

- model state: bitwise equal;
- DataLoader generator state: equal;
- scheduler state: equal;
- scaler state: equal;
- train loss and all recorded validation values: equal.

The original failure and correction are described under Changes made below.

### Determinism limitation

PyTorch emitted this warning during both FP32 and AMP training:

```text
nll_loss2d_forward_out_cuda_template does not have a deterministic implementation
```

`training.deterministic=true` seeds all documented RNGs, disables cuDNN benchmarking, and requests
deterministic algorithms with `warn_only=True`. It is therefore best-effort on this CUDA stack.
The bitwise resume result is strong evidence for this exact workload/environment, not a promise
across runs, GPUs, driver versions, PyTorch versions, or other tensor shapes.

## CPU/GPU prediction and metric comparison

The audited 15-epoch `best.pt` checkpoints and complete 27-image test split were used. Logits were
compared batch by batch at `rtol=1e-4`, `atol=1e-5`; metrics were independently accumulated on CPU
and CUDA.

| Model | Max logit abs error | Minimum pixel argmax agreement | Confusion matrix | Largest reported metric delta |
| --- | ---: | ---: | --- | ---: |
| TinyUNet | 0.001152 | 100.0000% | equal | 0.00000762 (pixel mAP) |
| LiteFCN | 0.001188 | 99.9878% | not equal | 0.00002957 (mean precision) |

Strict logit `allclose` failed for both models, as expected from different CPU/CUDA convolution
kernels. Outputs were finite. TinyUNet's class predictions and confusion-derived metrics were
identical. LiteFCN changed a very small number of near-boundary pixel decisions; its pixel accuracy
differed by 0.00000904 and mIoU by 0.00002009.

Measured CUDA test-set accuracy was:

| Model | Pixel accuracy | mIoU | Mean precision | Mean recall | Foreground pixel mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyUNet | 0.971644 | 0.833952 | 0.853958 | 0.972862 | 0.979147 |
| LiteFCN | 0.889251 | 0.598316 | 0.638799 | 0.920099 | 0.812501 |

These scores reproduce the already trained held-out artifacts; they are not results from the
one- or two-epoch smoke runs.

## Synchronized GPU benchmarks

The project benchmark runner performed 20 unmeasured warmups followed by 100 measurements. Each
timed eager/TorchScript call is bracketed by `torch.cuda.synchronize()`. Inputs are uniform-random
NCHW float32 tensors; data loading and preprocessing are excluded. Runs were separate processes,
not simultaneous workloads.

| Model/backend | Batch | Mean ms/batch | p50 ms | p95 ms | Throughput images/s | Peak allocated MiB | Artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyUNet eager | 1 | 1.4731 | 1.4160 | 1.7576 | 678.9 | 2.76 | 425,855 |
| TinyUNet eager | 16 | 1.5059 | 1.4486 | 1.8312 | 10,624.9 | 35.81 | 425,855 |
| TinyUNet TorchScript | 1 | 0.3463 | 0.3321 | 0.4150 | 2,888.0 | 1.19 | 190,127 |
| LiteFCN eager | 1 | 0.7773 | 0.7270 | 0.9038 | 1,286.6 | 0.61 | 87,641 |
| LiteFCN eager | 16 | 0.7526 | 0.7252 | 0.8586 | 21,260.2 | 7.93 | 87,641 |
| LiteFCN TorchScript | 1 | 0.2261 | 0.2214 | 0.2672 | 4,423.7 | 0.33 | 72,661 |

Checkpoint byte sizes in eager rows are training artifacts containing optimizer/RNG state and must
not be interpreted as deployment-model size. Peak memory is the PyTorch allocator's
`max_memory_allocated` after the model and benchmark input were resident; it includes inference
intermediates but excludes CUDA context/driver allocations. Reserved allocator memory was 46 MiB
for TinyUNet eager/TorchScript, 30 MiB for LiteFCN eager, and 26 MiB for LiteFCN TorchScript at
batch 1. No power mode, application clocks, laptop thermal state, or concurrent load was fixed, so
these are local microbenchmarks rather than deployment guarantees.

## TorchScript CUDA

Both existing TorchScript files loaded directly onto CUDA and executed at:

- `1×3×64×64`;
- `2×3×65×67` (dynamic odd spatial dimensions).

For both models and both shapes, TorchScript and eager CUDA logits were bitwise equal and had 100%
pixel argmax agreement. Test-set metrics also matched eager CUDA. PyTorch 2.14 emits deprecation
warnings for TorchScript; the artifact remains functional but should eventually migrate to the
recommended `torch.export` deployment path.

## ONNX Runtime CUDA

ORT sessions were explicitly constructed with providers
`[CUDAExecutionProvider, CPUExecutionProvider]`. A session with CPU fallback disabled failed to
initialize. Profiling showed that convolutional compute ran on CUDA, while small dynamic-shape
operations used CPU fallback:

| Model | Profiled CUDA node events | Profiled CPU node events | CPU fallback operations observed |
| --- | ---: | ---: | --- |
| TinyUNet | 279 | 90 | Concat, Div, Reshape, Squeeze |
| LiteFCN | 171 | 18 | Concat |

Counts are execution events across the profiled inference sequence, not unique graph-node counts.
CUDA ORT repeated the same input bitwise identically. Compared with CPU eager outputs:

| Model/shape | Max abs error | Pixel argmax agreement | Strict allclose |
| --- | ---: | ---: | --- |
| TinyUNet `1×3×64×64` | 0.001395 | 99.9268% | no |
| TinyUNet `2×3×65×67` | 0.001333 | 99.9770% | no |
| LiteFCN `1×3×64×64` | 0.004534 | 100.0000% | no |
| LiteFCN `2×3×65×67` | 0.00000739 | 100.0000% | yes |

The project's minimum export argmax agreement is 99.9%, which all four cases met, but the
`rtol=1e-4`, `atol=1e-5` logit criterion did not. Full-test ORT CPU/CUDA mIoU differed by 0.0001034
for TinyUNet and 0.0000149 for LiteFCN. Their confusion matrices were not identical.

A separate ORT timing probe used 20 warmups and 100 blocking `session.run` calls with CPU inputs
and outputs, so host/device transfers are included. TinyUNet averaged 0.3217 ms (3,108.8 images/s)
and LiteFCN 0.1546 ms (6,468.9 images/s) at batch 1. The main benchmark wrapper remains explicitly
CPU-only for ONNX Runtime; these numbers are reported separately to avoid changing or obscuring its
documented timing semantics. ORT CUDA memory was not measured with PyTorch allocator APIs because
ORT owns a separate allocator.

## Changes made

1. Fixed CUDA resume by moving restored CPU RNG, CUDA RNG, and DataLoader generator byte states to
   CPU before calling their state restoration APIs.
2. Added a CUDA-gated regression test that restores checkpoint RNG tensors after CUDA relocation.
3. Extended benchmark JSON with CUDA allocator baseline, current allocated/reserved, and peak
   allocated/reserved bytes plus an explicit scope note.
4. Added a CPU assertion for the new nullable benchmark memory fields and documented their scope.
5. Ignored the isolated `.venv-gpu` environment.

No model, optimizer, metric definition, dataset, checkpoint contents, export graph, or reported
trained-model score was rewritten.

## Test and check results

The final commands and their results are recorded here; generated raw measurement JSON is under
the ignored `runs/gpu_validation/` directory in this workspace.

```text
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-gpu/bin/python -m pytest -q
  28 passed

env -u PYTHONPATH .venv-gpu/bin/ruff check .
  All checks passed

env -u PYTHONPATH .venv-gpu/bin/python -m pip check
  No broken requirements found

env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
  27 passed, 1 CUDA-only test skipped
```

GPU execution additionally covered both FP32 training smokes, both AMP smokes, exact resume,
complete-split CPU/CUDA evaluation, six synchronized eager/TorchScript benchmarks, four dynamic
TorchScript comparisons, four dynamic ONNX CUDA comparisons, full-test ORT CPU/CUDA metrics, and
ORT execution-provider profiling.

## Unsupported or unvalidated paths

- TensorRT was **not tested**. ORT listed `TensorrtExecutionProvider`, but provider availability is
  not proof that a graph builds or executes with TensorRT. `torch_tensorrt` was not installed.
- Multi-GPU, distributed training, CUDA graphs, `torch.compile`, FP16-exported ONNX, and INT8 were
  not tested.
- ORT CUDA peak GPU memory was not measured.
- Long-duration AMP convergence, thermal throttling, energy consumption, cold-start latency,
  preprocessing/postprocessing, concurrent streams, and end-to-end camera latency were not tested.
- Strict CUDA determinism is blocked by the current `nll_loss2d` implementation; warning-only
  deterministic mode must not be described as a universal bitwise guarantee.

## Recommendations

1. Add a dedicated GPU CI runner that executes the CUDA-gated RNG restore test and one smoke epoch
   for each architecture.
2. Decide whether `training.deterministic=true` should become strict-fail on unsupported kernels or
   whether a separately named best-effort mode is preferable.
3. Add an explicit ONNX provider setting and I/O-binding benchmark before presenting ORT CUDA as a
   supported deployment backend; retain CPU fallback disclosure and per-provider profiling.
4. Define backend-specific numerical acceptance based on downstream prediction/metric impact,
   while retaining strict CPU export checks.
5. Repeat benchmarks under controlled power/thermal conditions with longer runs, multiple trials,
   batch/concurrency sweeps, cold starts, and end-to-end preprocessing/postprocessing.
6. Migrate TorchScript coverage toward `torch.export` before the deprecated APIs are removed.
