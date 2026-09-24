# Training and reproducibility

## Reproducibility controls

The pipeline seeds Python, NumPy, CPU PyTorch, all CUDA devices, the DataLoader generator, and workers. With `training.deterministic: true`, deterministic PyTorch algorithms are requested with warnings and cuDNN benchmarking is disabled. Augmentation randomness is derived from `(training seed, epoch, sample index)`, so it does not depend on sample access order.

The data generator has its own `data.seed`. Keeping data and training seeds separate allows repeated optimization runs on the exact same samples and splits.

Exact bitwise equality across different PyTorch, CUDA, cuDNN, driver, CPU, or GPU combinations is not promised. The saved `environment.json` makes those differences auditable.

## Optimization

Both models produce unnormalized `N×4×H×W` logits. Training uses:

- cross-entropy with inverse-square-root class-frequency weights from the training split only;
- AdamW with configurable learning rate and weight decay;
- cosine annealing over the configured total epochs;
- optional CUDA automatic mixed precision (`training.amp: true`);
- best-checkpoint selection by validation mean IoU.

AMP is deliberately disabled on CPU even if requested. `training.device: auto` selects CUDA only when PyTorch reports it available. Explicit `cuda` fails instead of silently falling back.

## Checkpoint policy

`last.pt` is atomically replaced after every completed epoch. `best.pt` is copied whenever validation mean IoU improves. `epoch_NNN.pt` follows `save_every`. Format-3 checkpoints contain the current and best model states, best epoch/score, optimizer, scheduler, AMP scaler, global RNG, and DataLoader shuffle-generator state. A resumed run can therefore materialize a valid `best.pt` even in a new output directory. Worker processes are recreated each epoch so they receive the current deterministic augmentation epoch.

Exact resume requires the original planned epoch count and rejects trajectory-changing configuration edits. Extending a completed cosine schedule is intentionally rejected because silently continuing beyond `T_max` changes learning-rate semantics; start an explicitly named fine-tuning experiment instead. Format-1/2 checkpoints remain usable for evaluation/export but do not meet the format-3 exact-resume contract.

Full-state checkpoints are loaded with `torch.load(..., weights_only=False)` and can execute pickle payloads. Load only checkpoints produced by a trusted run or artifact store.

## Logging

- `events.jsonl`: lifecycle events and complete epoch records.
- `metrics.csv`: compact epoch-by-epoch optimization metrics.
- `tensorboard/`: emitted only when TensorBoard is installed.
- `resolved_config.yaml`: merged defaults plus overrides.
- `environment.json`: software/hardware provenance and Git commit when available.
- `data_validation.json`: sample counts and pixel distributions.

## Suggested experiment protocol

1. Freeze the generated manifest and validate hashes.
2. Choose architecture/hyperparameters using only training and validation splits.
3. Evaluate the selected checkpoint once on the test split.
4. Export that exact checkpoint and verify numerical agreement.
5. Benchmark every backend under the same threads, batch size, input shape, warmup, and iteration count.
6. Preserve the config, environment, metrics, export metadata, and benchmark JSON with the release.

The default 15 epochs remain deliberately modest. Inspect learning curves and repeat with multiple seeds before declaring convergence or an architecture winner.
