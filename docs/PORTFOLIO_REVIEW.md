# Portfolio requirement map

| Requirement | Evidence |
| --- | --- |
| Reproducible preparation | Seeded generator, versioned manifest, SHA-256 per file, `rv-prepare` |
| Train/validation/test | Deterministic disjoint assignment and leakage validator |
| Augmentation | Deterministic paired flip plus photometric/noise transforms |
| Configurable PyTorch training | YAML/default merge, validated `--set`, CPU/CUDA/AMP controls |
| Checkpoint and resume | Atomic full-state checkpoints including optimizer/scheduler/scaler/RNG |
| Evaluation | Held-out split, pixel accuracy, precision, recall, IoU, pixel AP/mAP |
| Confusion/error analysis | Full confusion matrix and ranked visual worst cases |
| Experiment logging | JSONL, CSV, optional TensorBoard, resolved config, environment/data provenance |
| Reproducibility options | Separate data/training seeds, deterministic algorithms and workers |
| Export/deployment | TorchScript, ONNX checker, ONNX Runtime verification, optional TensorRT |
| Benchmarking | Accuracy, mean/p50/p95 latency, throughput, artifact size, parameter count |
| Model comparison | TinyUNet and LiteFCN configs plus `rv-compare` for measured JSON |
| Tests/CI | Unit and lifecycle integration tests; CPU GitHub Actions workflow |
| Documentation | Professional README, dataset card, training guide, model card, deployment guide |
| Artifact hygiene | Data/runs/checkpoints/ONNX/TensorRT ignored; no external dataset committed |

The project is deliberately small enough to run end to end during review. Its strongest portfolio signal is the explicit contract between each lifecycle stage: data hashes feed validated loaders; resolved configs and environments accompany checkpoints; the selected checkpoint feeds evaluated exports; and benchmarks retain accuracy alongside systems measurements.

Before presenting results, run both model configurations under the same conditions and keep the generated comparison files with the portfolio release. Do not copy metrics between machines or claim TensorRT support unless it was exercised on the stated target.

