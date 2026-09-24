# Technical audit report

Audit date: 2026-09-24

Scope: dataset lifecycle, TinyUNet/LiteFCN, deterministic training and resume, metrics, TorchScript/ONNX/ONNX Runtime, benchmarking, tests/CI/dependencies/security/hygiene, and project documentation.

## Executive assessment

The repository implements a coherent, runnable CPU semantic-segmentation lifecycle rather than an inference-only demo. Both models are small but structurally valid, the generated dataset is reproducible, training/evaluation/export paths execute end to end, and generated data/model artifacts are excluded from Git. The audit found no evidence that documented accuracy or latency results were fabricated.

The original implementation nevertheless had several correctness and hardening gaps: dataset leakage checks relied only on IDs, a stale manifest could be reused under a changed config, unknown/invalid configuration values were accepted, mIoU penalized classes absent from both truth and predictions, resume could silently change the cosine schedule and could omit `best.pt` in a new run directory, export equivalence was recorded but not enforced or tested dynamically, and benchmark provenance was incomplete. These defects were reproduced and corrected without replacing the functional architecture.

## Findings

### Dataset generation and validation

What was sound:

- Scene generation uses independent seeded Python and NumPy RNGs and stable per-index seeds.
- Split assignment is a seeded shuffle with one manifest record per generated ID.
- Images/masks carry SHA-256 digests; modes, dimensions, label range, non-empty splits, and aggregate class presence are checked.
- Training class weights use training-split counts only, avoiding test-label prevalence leakage.

Confirmed defects and disposition:

- **Fixed — content leakage:** globally unique IDs did not prevent the same image bytes or paths appearing under different IDs/splits. A constructed duplicate passed validation. The validator now rejects reused resolved image/mask paths and duplicate image content when SHA checks are enabled.
- **Fixed — path traversal:** manifest paths were joined without containment checks. Relative `..`, absolute paths, or symlink resolution could escape the dataset root. All record paths are now relative, resolved, and root-confined.
- **Fixed — stale provenance:** changing seed/sample count while keeping a manifest path silently reused old data. Reuse now requires generator/version, seed, dimensions, sample count, and split ratios to match the active config.
- **Fixed — destructive force guard:** `--force` could recursively remove an arbitrary configured directory. Root/home/repository-containing paths, symlink roots, and unrecognized non-empty external directories are refused.
- **Fixed — schema robustness:** IDs, record mappings, image-size metadata, generator metadata, paths, and duplicate content now receive explicit validation errors rather than incidental exceptions.

Remaining methodology limits:

- `--skip-hashes` cannot detect byte-identical images stored at distinct paths.
- Generation writes directly into the target tree rather than staging and atomically renaming a complete dataset.
- Only global class presence is required by dataset validation; training separately refuses a class with zero training pixels.
- The procedural generator is deliberately simple and does not establish real-world robotics validity.

### TinyUNet and LiteFCN

- TinyUNet has two downsampling stages, a bottleneck, skip-connected bilinear decoding, and a 1×1 classifier. Explicit interpolation to encoder shapes correctly handles odd spatial dimensions.
- LiteFCN uses depthwise spatial convolution followed by pointwise projection, two stride-2 reductions, and bilinear logit upsampling. Group counts and channel transitions are correct.
- Both return `N×C×H×W` logits and preserve arbitrary tested spatial sizes. Parameter counts are derived from model parameters, not hard-coded.
- Tests exercised both eager and TorchScript execution at odd sizes. ONNX Runtime was additionally exercised at default and dynamic odd shapes.

Remaining model limits are architectural rather than implementation bugs: no pretrained backbone, limited receptive field/capacity, BatchNorm sensitivity for small-domain training, and no uncertainty or temporal modeling.

### Configuration, determinism, checkpointing, and resume

What was sound:

- Python, NumPy, PyTorch CPU/CUDA, DataLoader generator, and worker RNGs are seeded.
- Augmentations derive randomness from training seed, epoch, and sample index; workers are recreated per epoch.
- Checkpoints are atomically replaced and contain optimizer, scheduler, scaler, RNG, and DataLoader generator state.
- A focused interrupted-vs-uninterrupted CPU test produced bitwise-identical model weights and identical RNG/DataLoader states under the same planned schedule.

Confirmed defects and disposition:

- **Fixed — configuration typos/invalid values:** unknown YAML keys and invalid values such as zero benchmark iterations, negative workers/LR, zero channels, or flip probability above one were accepted. Nested keys, types, ranges, paths, and non-empty split allocations are now validated.
- **Fixed — resume trajectory drift:** architecture/data identity was checked, but augmentation, optimizer inputs, seed, workers, batch size, AMP, and total epochs could change. Exact resume now rejects trajectory-changing settings.
- **Fixed — cosine extension:** increasing total epochs after restoring `CosineAnnealingLR` could continue beyond the original `T_max` with unintended behavior. Exact resume requires the original planned schedule; extension is documented as a separate future fine-tuning feature.
- **Fixed — missing best checkpoint:** a resume into a new run directory could produce no `best.pt` when later epochs did not improve. Format-3 checkpoints carry best model state/epoch and materialize a valid best artifact. A materialized best whose optimizer state is not aligned is explicitly evaluation-only.
- **Fixed — learning-rate logging:** the logged LR was the post-scheduler value for the next epoch. It now records the LR actually used for that epoch.
- **Fixed — overwrite safety:** run deletion requires safe paths plus recognized run markers and rejects symlinked/broad targets.

Security note: complete training checkpoints require `torch.load(..., weights_only=False)` and therefore must be treated as trusted pickle artifacts. Format-1/2 files remain usable for evaluation/export, but exact resume requires format 3.

### Evaluation metrics and error analysis

- Confusion matrix orientation is correctly documented as rows=ground truth and columns=prediction.
- Pixel accuracy, per-class precision/recall/IoU, macro precision/recall, and foreground pixel mAP are computed from streaming accumulators.
- Pixel AP is explicitly a bounded-memory, histogram-approximated, one-vs-rest semantic metric—not COCO/instance AP.
- Invalid prediction/target ranges and incompatible shapes now fail explicitly.
- **Fixed — absent-class mIoU:** a perfect two-class sample evaluated with an unused third class previously reported mIoU `0.6667`. Mean IoU now averages only classes with non-zero truth/prediction union and reports the number of evaluated classes. Undefined per-class ratios remain `0.0` under a documented zero-division policy.
- Worst-case panels correctly combine input, truth, and prediction and rank by per-image mean IoU.

Remaining limits: AP accuracy depends on `ap_bins`; macro precision/recall use a zero-division policy that should be considered when comparing against tools with different defaults; no confidence intervals or multi-seed aggregation are produced.

### Export and ONNX Runtime equivalence

- TorchScript scripting and ONNX opset-18 export both succeed for TinyUNet and LiteFCN.
- ONNX models pass `onnx.checker`, embed weights, expose dynamic batch/height/width, and execute with ONNX Runtime CPU.
- **Fixed — unenforced equivalence:** export now fails unless outputs are finite, logits satisfy `rtol=1e-4`/`atol=1e-5`, and pixel argmax agreement is at least 99.9%.
- **Fixed — dynamic claims:** verification now covers `1×3×64×64` and `2×3×65×67` for the default models, rather than only the tracing/export shape.
- **Fixed — artifact provenance:** metadata now includes source-checkpoint SHA-256 plus artifact hashes, sizes, dynamic verification cases, tolerances, actual opset, and external-data status.

Observed audit results:

| Model | Shape | ONNX max absolute error | Argmax agreement | Allclose |
| --- | --- | ---: | ---: | --- |
| TinyUNet | `1×3×64×64` | `2.16067e-6` | 1.0 | yes |
| TinyUNet | `2×3×65×67` | `2.27615e-6` | 1.0 | yes |
| LiteFCN | `1×3×64×64` | `2.62260e-6` | 1.0 | yes |
| LiteFCN | `2×3×65×67` | `7.86781e-6` | 1.0 | yes |

TorchScript was bitwise equal to eager output for all four cases. PyTorch 2.14 warns that TorchScript and `dynamic_axes` compatibility APIs are deprecated; they remain functional requirements here but need a migration plan.

### Benchmark methodology

What was sound:

- Warmup is excluded from measurements.
- CUDA paths synchronize before and after each timed model call.
- Throughput uses measured batch latency and batch size correctly.
- Mean, p50, and p95 are based on individual iterations, and artifact size includes ONNX sidecars.
- ONNX Runtime is deliberately CPU-only in this tool and configures one inter-op thread plus the requested intra-op threads.

Improvements made:

- Dataset hashes are validated before accuracy measurement.
- Reports now include backend/runtime/provider, model SHA-256, artifact kind, parameter bytes, environment/Git commit/dirty state, input shape, timer, timing scope, and mean/stddev/min/max/p50/p95.
- Training checkpoints and deployment artifacts are labeled separately so checkpoint size is not mistaken for deployable model size.

The audited TinyUNet ONNX Runtime run reproduced test mIoU `0.833952` and pixel accuracy `0.971644`; batch-1, one-thread model-call latency over 30 iterations was mean `0.799812 ms`, p95 `0.805551 ms`, throughput `1250.29 images/s`. These are short local microbenchmark results, not target-hardware guarantees.

### Tests, CI, dependencies, security, and hygiene

- The expanded local suite passes 27 tests and covers configuration rejection, reproducible generation, leakage/path/hash failures, augmentation, metric edge cases, both models, bitwise resume, best-checkpoint behavior, dynamic TorchScript/ONNX equivalence, evaluation, benchmark metadata, and comparison output.
- CI installs CPU PyTorch and all extras, runs dependency consistency, Ruff, pytest, complete smoke prepare/train/evaluate/export, TorchScript and ONNX Runtime benchmarks, and two-architecture comparison.
- The wheel builds successfully. Generated data, runs, checkpoints, ONNX, TensorRT, caches, and virtual environments are ignored; no generated/model artifacts are tracked.
- YAML uses `safe_load`, subprocess use is fixed-argument/read-only, and manifest/run deletion paths are now guarded.

Dependency/reproducibility findings:

- Project dependency ranges are intentionally broad and there is no lock file; a future upstream release can change behavior.
- GitHub Actions are version-tagged rather than pinned to immutable commit SHAs.
- No dependency vulnerability scanner, SBOM, artifact signature, or secret scanner is configured.
- Normal local `pip check` sees an unrelated ROS package through the host `PYTHONPATH` and reports its missing `typeguard`; with `PYTHONPATH` unset, the isolated project environment reports no broken requirements. CI runs in a clean environment.

### Documentation

The README, dataset card, training guide, model card, deployment guide, portfolio map, and verification record are substantive and generally honest about synthetic-domain limits. This audit aligned them with the corrected behavior: content/path leakage checks, schedule-preserving resume, trusted-checkpoint warning, present-class mIoU, enforced dynamic export equivalence, and benchmark scope/provenance.

## Changes made

1. Added strict recursive configuration schema/type/range/path validation.
2. Added config-to-manifest identity checks before data reuse.
3. Added manifest path containment, unique resolved path, and duplicate-content checks.
4. Added guarded dataset/run deletion.
5. Added training-class presence enforcement.
6. Upgraded checkpoints to format 3 with best model/epoch and resume capability metadata.
7. Enforced trajectory-preserving exact resume and corrected LR logging.
8. Corrected mIoU handling for absent classes and added metric input validation.
9. Added checkpoint/config compatibility checks to evaluation, export, and eager benchmark paths.
10. Enforced deterministic two-shape TorchScript/ONNX equivalence and added artifact hashes.
11. Added benchmark dataset validation, provenance, runtime/provider, parameter bytes, dispersion, and timing-scope fields.
12. Expanded tests and CI, including ONNX Runtime benchmarking and dependency consistency.
13. Updated all affected user documentation.

## Test results

Commands executed in the CPU audit environment:

```text
ruff check .                                      PASS
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q        27 passed
rv-validate --config configs/tiny_unet.yaml       PASS; 180 hashes checked, 126/27/27 split
rv-export ... TinyUNet                            PASS; TorchScript + ONNX, 2 shapes
rv-export ... LiteFCN                             PASS; TorchScript + ONNX, 2 shapes
rv-benchmark ... --backend onnxruntime            PASS; held-out metrics + 30 timings
python -m pip wheel --no-deps .                   PASS
env -u PYTHONPATH python -m pip check             PASS
git diff --check                                  PASS
```

The remote GitHub Actions workflow was inspected and improved but was not executed from this local audit. CUDA was unavailable (`torch.cuda.is_available() == false`), and Torch-TensorRT was not installed. No CUDA, AMP-on-CUDA, TensorRT build, or TensorRT runtime claim is made.

## Remaining risks

- Synthetic same-generator test performance does not predict real-robot performance or safety.
- Only one training seed is recorded; architecture rankings lack uncertainty estimates.
- Deterministic algorithms use `warn_only=True`; a future nondeterministic CUDA operator could warn rather than stop.
- CUDA, CUDA AMP, TensorRT, other ONNX providers, and target-device thermals/memory/concurrency remain untested.
- ONNX dynamic behavior is sampled at two shapes, not exhaustively verified for every valid dimension.
- TorchScript and the compatibility `dynamic_axes` API are deprecated in the installed PyTorch version.
- Pixel AP is approximate and can differ slightly across backends when probabilities cross histogram-bin boundaries.
- Benchmarks exclude preprocessing, data transfer, postprocessing, cold start, memory, concurrency, and long-run variance.
- Full checkpoints are trusted pickle files; hashes detect accidental/tampered bytes only when compared to trusted metadata and are not signatures.
- Dependencies and CI actions are not immutably locked; no vulnerability/SBOM/signing pipeline exists.
- Dataset generation is not transactionally staged, so interruption can leave a partial target directory.

## Recommendations

1. Validate on a versioned, license-reviewed real or high-fidelity robotics dataset with scene/group-level splits and domain-shift tests.
2. Run at least three to five seeds and report mean/dispersion for quality and latency before selecting an architecture.
3. Add dedicated GPU CI or target-hardware jobs for CUDA determinism, AMP, ONNX CUDA provider, TensorRT conversion, numerical parity, memory, and sustained-load latency.
4. Add an explicit fine-tuning/extension command with a documented scheduler restart instead of weakening exact-resume checks.
5. Migrate deployment experiments toward `torch.export` while retaining TorchScript only for consumers that require it.
6. Add a constraints/lock strategy, dependency update automation, vulnerability scanning, SBOM generation, immutable Action SHAs, and signed artifact manifests.
7. Add atomic temporary-directory dataset generation and rename-on-success for interruption safety.
8. Extend benchmarks to preprocessing/postprocessing, cold start, memory, batch/concurrency sweeps, and target power/thermal conditions.
