# v1.1.0 proposed release notes

The existing pipeline's nominal synthetic scores and one training seed could not
show sensitivity to changed input conditions or distinguish augmentation gains
from optimization variation. v1.1 adds controlled generalization analysis while
preserving TinyUNet, LiteFCN and the existing training/deployment lifecycle.

Six paired stressors cover lighting, texture, blur, noise, planar viewpoint and
visible occlusion at two severities. Deterministic scene seeds and content hashes
keep model/arm comparisons paired. Configurable training additions default to
zero. A predeclared seven-arm, three-seed study retains all scores, class supports,
confusion matrices, degradation, raw latency observations and failed attempts.
Nonfinite diagnostics and atomic study ledgers improve failure visibility.

Validation completed 42 CPU training runs and 546 condition evaluations, 56 CPU
tests (one CUDA-only skip), 57 tests in the CUDA environment, Ruff, package build,
dependency checks, local CI study smoke, input/artifact/metric consistency review,
and independent bitwise-equal CPU repeat training for both baseline models.
TorchScript/ONNX/ORT exports passed two-shape numerical checks and full-test
quality comparison. Fresh two-epoch CUDA FP32/AMP training and synchronized
eager/TorchScript benchmarks were executed on an RTX 5070 Laptop GPU.

Measured baseline nominal mIoU across three seeds is 0.6874 ± 0.1272 for TinyUNet
and 0.6070 ± 0.0166 for LiteFCN. Severe dimming reduces mean mIoU by 25.04 and
17.44 percentage points respectively, exposing minority-class failures. Adding
occlusion augmentation changes mean shifted mIoU by +5.73 pp for TinyUNet and
−2.12 pp for LiteFCN. Combined augmentation changes it by +4.71/−4.56 pp. These
model-dependent results are measured; proposed causal explanations remain hypotheses.

At batch 1/64², measured CPU ORT mean model-call latency is 1.5414 ms for TinyUNet
and 0.1889 ms for LiteFCN, with ONNX sizes 174,218/47,890 bytes. GPU TorchScript
mean latency is 0.6746/0.5325 ms under the documented local protocol. Timing is
not end-to-end robot latency and does not control power/thermals.

Limits: one small procedural split, three optimization seeds, palette shortcuts,
planar camera proxy and synthetic occluders; no sim-to-real or safety claim.
A real LiteFCN CUDA FP32 bitwise-resume failure (max state difference 1.79e-7)
is preserved despite a successful later collection. TensorRT remains unavailable
and unvalidated. Remote CI and physical-robot validation were not performed.

Next proposed milestone: independent scene/camera/session splits with licensed
real or higher-fidelity data, scene-level uncertainty, and experiments investigating
dim-light failure and TinyUNet seed variability before adding model count.

Recommend **v1.1.0** after the documented review. No tag or remote publication
was created. Historical v1.0.0 is absent from this checkout; the owner-approved
provisional baseline is `54f3a2a`, rather than a manufactured release tag.

## Owner competency checklist

- [ ] Present these results with their seed, scene, timing and hardware limits.
- [ ] Explain the augmentation losses and CUDA failure as well as the gains.
- [ ] Walk a reviewer from a release claim to its raw config/metric/hash artifact.
