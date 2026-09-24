# Local verification record

This file records observed results from 2026-09-24, not projected performance or a hardware-independent claim. Raw JSON, images, checkpoints, and exports were generated locally under ignored `runs/`, `data/processed/`, and `artifacts/` paths.

## Environment and checks

- Linux 7.0.0 x86-64, Python 3.12.3
- PyTorch 2.14.0+cpu, NumPy 2.5.3
- ONNX 1.23.0, ONNX Runtime 1.30.0 CPU execution provider
- CUDA unavailable; CUDA/AMP acceleration was not exercised
- Torch-TensorRT unavailable; optional TensorRT export was not exercised
- Ruff: passed
- Pytest at initial implementation: 12 passed; the later audit suite and results are recorded in `AUDIT_REPORT.md`
- Dataset validation: 180 samples, 126/27/27 train/validation/test, all SHA-256 checks passed

The original measured training artifacts were created before the repository was initialized, so their saved `environment.json` records `git_commit: null`. The later audit ran from initialized branch `main`; see the audit report for current checks.

## Architecture comparison

Both models used the same generated data (data seed 17), training seed 42, augmentation, class-weighted loss, 15-epoch budget, and selected their best validation-IoU checkpoint. Metrics below are from the 27-image held-out test split. Systems measurements use TorchScript, batch 1, one CPU thread, five warmups, and 30 measured iterations.

| Model | Best epoch | Parameters | Test pixel accuracy | Test mean IoU | Foreground pixel mAP | TorchScript size | Mean latency | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyUNet | 6 | 29,948 | 0.971644 | 0.833952 | 0.979140 | 190,127 B | 1.52641 ms/image | 655.13 images/s |
| LiteFCN | 13 | 2,804 | 0.889242 | 0.598296 | 0.812523 | 72,661 B | 0.287047 ms/image | 3,483.74 images/s |

This single-seed result shows the expected local trade-off: TinyUNet was more accurate, while LiteFCN was smaller and faster. It is insufficient for uncertainty estimates or a general architecture ranking.

## Backend comparison

TinyUNet’s test metrics were identical at the reported precision across eager PyTorch, TorchScript, and ONNX Runtime. Fixed-condition batch-1 CPU measurements were:

| Backend | Artifact size | Mean latency | Throughput |
| --- | ---: | ---: | ---: |
| Eager PyTorch checkpoint | 425,855 B | 2.03366 ms/image | 491.72 images/s |
| TorchScript | 190,127 B | 1.52641 ms/image | 655.13 images/s |
| ONNX Runtime | 174,218 B | 0.827924 ms/image | 1,207.84 images/s |

Export verification observed 100% pixel-argmax agreement for both TorchScript models and LiteFCN ONNX. TinyUNet ONNX agreed on 99.9756% of pixels (one near-tie pixel in the 64×64 verification input). TinyUNet’s maximum absolute error was 0 for TorchScript and `2.08616e-06` for ONNX Runtime; LiteFCN’s was 0 and `3.33786e-06`, respectively. Both ONNX files passed `onnx.checker`, used opset 18, and embedded their weights without sidecars.

Latency figures are short microbenchmarks and will vary with CPU model, power state, runtime version, thread affinity, and competing load. Training and test data come from the same simple procedural generator; these scores do not demonstrate real-world robotics performance.
