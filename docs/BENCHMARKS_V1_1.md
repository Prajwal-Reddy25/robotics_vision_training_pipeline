# Benchmarks v1.1

Measured 2026-10-04; batch 1, 64×64 RGB FP32, one CPU thread, 20 warmups,
100 timed model calls. Times exclude image loading, corruption, preprocessing,
postprocessing and host/device input transfer. CUDA calls synchronize before and
after each measurement. No CPU affinity, clocks, thermals or power limits were
fixed. These are local microbenchmarks, not end-to-end robot latency.

CPU: AMD Ryzen 9 8940HX. GPU: NVIDIA GeForce RTX 5070 Laptop GPU, driver 595.84.
CPU PyTorch 2.14.0+cpu; GPU PyTorch 2.14.0+cu130. ORT uses CPUExecutionProvider.
Quality/size/latency results are linked by checkpoint and artifact SHA-256.

## Baseline seed-42 deployments

| Model | Device/backend | mIoU | Mean ms | p50 ms | p95 ms | File bytes | Artifact kind |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| tiny_unet | cpu/pytorch | 0.834201 | 1.7019 | 1.6803 | 1.8846 | 564803 | training_checkpoint |
| lite_fcn | cpu/pytorch | 0.597966 | 0.3452 | 0.3436 | 0.3698 | 112897 | training_checkpoint |
| tiny_unet | cpu/torchscript | 0.834201 | 3.2482 | 2.9926 | 4.6822 | 190127 | deployment_artifact |
| tiny_unet | cpu/onnxruntime | 0.834201 | 1.5414 | 1.4398 | 2.1944 | 174218 | deployment_artifact |
| lite_fcn | cpu/torchscript | 0.597966 | 0.7613 | 0.6422 | 1.4358 | 72677 | deployment_artifact |
| lite_fcn | cpu/onnxruntime | 0.597966 | 0.1889 | 0.1634 | 0.4201 | 47890 | deployment_artifact |
| tiny_unet | cuda/pytorch | 0.834274 | 2.8973 | 2.7437 | 4.1334 | 564803 | training_checkpoint |
| tiny_unet | cuda/torchscript | 0.834274 | 0.6746 | 0.6634 | 0.8455 | 190127 | deployment_artifact |
| lite_fcn | cuda/pytorch | 0.597966 | 1.6476 | 1.5389 | 2.3001 | 112897 | training_checkpoint |
| lite_fcn | cuda/torchscript | 0.597966 | 0.5325 | 0.4822 | 0.8857 | 72677 | deployment_artifact |

All 42 trained checkpoints have their own nominal-quality and eager latency
measurements, raw duration samples, checkpoint file size and parameter bytes in
[study.json](evidence/v1_1/study.json). Deployment/parity details are in
[deployment.json](evidence/v1_1/deployment.json), synchronized GPU measurements
in [gpu.json](evidence/v1_1/gpu.json).

TinyUNet has 29,948 parameters / 119,792 FP32 parameter bytes; LiteFCN has 2,804 /
11,216 bytes. Full checkpoint sizes include optimizer, best model and RNG state;
they are not deployment-model footprints. TorchScript/ONNX sizes include graph
and buffer overhead. Shapes/operators stay fixed across corruptions; timing is
per checkpoint. No condition-specific latency was measured.

## Numerical checks and limitations

Both baseline exports passed finite logits, allclose rtol=1e-4/atol=1e-5, and
≥99.9% argmax agreement at 1×3×64×64 and 2×3×65×67. Full test mIoU is identical
across fresh CPU eager, TorchScript and ORT. GPU eager/TorchScript agree on test
mIoU; GPU minus CPU mIoU is +0.00007217 for TinyUNet and 0 for LiteFCN, below the
declared 0.001 smoke-comparison threshold. Cross-device bitwise parity is not claimed.

CUDA AMP training is exercised, but the accuracy ablation and timing table are
FP32. This milestone does not add an ORT CUDA benchmark or validate TensorRT.
Neither speed nor export parity establishes perception accuracy on real sensors.

## Commands

```bash
python scripts/validate_v1_1.py --study runs/v1_1_ablation/study.json \
  --output-dir runs/my_v1_1_deployment
python scripts/validate_v1_1.py --study runs/v1_1_ablation/study.json \
  --output-dir runs/my_v1_1_gpu --gpu --deployment-dir runs/my_v1_1_deployment
```

Use the CPU environment for the first command and a compatible CUDA PyTorch
environment for the second. Output paths must be fresh. Strict GPU resume
failures are retained and cause nonzero exit even when remaining measurements finish.

## Owner competency checklist

- [ ] Explain CUDA synchronization and the excluded latency stages.
- [ ] Distinguish parameter bytes, training artifacts and deployment file sizes.
- [ ] Locate raw durations and calculate p50/p95 without dropping outliers.
- [ ] Explain numerical export parity separately from held-out segmentation quality.
