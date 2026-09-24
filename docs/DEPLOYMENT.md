# Export and deployment

## Artifact contract

The export command loads a selected checkpoint on CPU and writes:

- `model.torchscript.pt`, compiled with `torch.jit.script`;
- `model.onnx`, opset 18 with dynamic batch/height/width axes and embedded weights for these small models;
- `metadata.json`, including source checkpoint/epoch, model settings, tensor contract, file sizes, maximum absolute conversion error, and pixel-argmax agreement.

Input is float32 RGB in NCHW order scaled to `[0, 1]`. There is no mean/std standardization. Output is NCHW logits; apply `argmax(axis=1)` for class IDs.

```bash
rv-export --config configs/tiny_unet.yaml \
  --checkpoint runs/tiny_unet/checkpoints/best.pt \
  --output-dir artifacts/tiny_unet
```

ONNX is structurally checked with `onnx.checker` and, by default, executed with ONNX Runtime CPU for numerical comparison. Current PyTorch exporters also use `onnxscript`; it is included in the `export`/`all` extras. Use `--no-onnx-verify` only when intentionally exporting on a build host without ONNX Runtime.

## Backend benchmarks

```bash
rv-benchmark --config configs/tiny_unet.yaml \
  --model runs/tiny_unet/checkpoints/best.pt --backend pytorch \
  --output runs/tiny_unet/benchmark_pytorch.json

rv-benchmark --config configs/tiny_unet.yaml \
  --model artifacts/tiny_unet/model.torchscript.pt --backend torchscript \
  --output runs/tiny_unet/benchmark_torchscript.json

rv-benchmark --config configs/tiny_unet.yaml \
  --model artifacts/tiny_unet/model.onnx --backend onnxruntime \
  --output runs/tiny_unet/benchmark_onnxruntime.json
```

Every benchmark first evaluates the configured held-out split, then runs warmup and measured synthetic batches. The report includes artifact size, parameter count when introspectable, accuracy metrics, mean/p50/p95 batch latency, per-image latency, throughput, device, thread count, batch size, and iteration counts. ONNX Runtime currently uses its CPU execution provider for comparable, dependency-light deployment tests.

## Optional TensorRT

TensorRT is intentionally optional because it requires NVIDIA hardware plus compatible CUDA, TensorRT, and either Torch-TensorRT or `trtexec` versions. On a configured Torch-TensorRT host:

```bash
rv-export --config configs/tiny_unet.yaml \
  --checkpoint runs/tiny_unet/checkpoints/best.pt \
  --output-dir artifacts/tiny_unet --tensorrt
```

Alternatively, build from ONNX on the deployment host:

```bash
trtexec \
  --onnx=artifacts/tiny_unet/model.onnx \
  --saveEngine=artifacts/tiny_unet/model.engine \
  --minShapes=image:1x3x64x64 \
  --optShapes=image:1x3x64x64 \
  --maxShapes=image:8x3x64x64 \
  --fp16
```

Do not use FP16 blindly: compare predictions and held-out metrics against the source checkpoint and benchmark on the actual target. TensorRT engine files are hardware/software-specific and are Git-ignored.

## Production checklist

- Pin and scan the runtime image and model artifact.
- Verify SHA-256/signature in the artifact registry.
- Enforce tensor shape/type/range before inference.
- Record preprocessing and class-map versions with predictions.
- Benchmark cold start, steady state, concurrency, and memory on target hardware.
- Define confidence/health checks, fallback behavior, alerting, rollback, and drift monitoring.
- Re-run representative safety and regression suites after every engine/runtime change.
