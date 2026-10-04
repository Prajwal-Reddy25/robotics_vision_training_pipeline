# Engineering notes and technical ownership

## Baseline data, models and training (`data.py`, `models.py`, `training.py`)

**Problem:** provide a small executable segmentation lifecycle with independent
data and optimization randomness. **Principle:** supervised categorical pixel
classification; logits z produce p_c = exp(z_c)/Σ_k exp(z_k). The weighted loss
is −Σ_i w_yi log p_yi / Σ_i w_yi, where class weights are normalized
sqrt(total training pixels / class training pixels). Counts use training only.
TinyUNet preserves detail through skip concatenation; LiteFCN decomposes spatial
and channel mixing: k²CinCout becomes k²Cin + CinCout multiply-adds per output
location in a depthwise/pointwise layer. Those counts alone do not predict latency.

**Implementation:** leave generator version 1.0, RGB/mask bytes, architectures,
AdamW/cosine schedule and nominal validation checkpoint selection unchanged.
**Decisions/alternatives:** measure the existing models rather than add a backbone;
do not choose epochs/arms with shifted test scores. External datasets and pretrained
models would change the problem and require separate licensing/split provenance.
**Assumptions/failures:** palette is a powerful shortcut, floors dominate pixels,
BatchNorm depends on training statistics, small test sets can reverse rankings.
**Validation:** baseline suite plus seed-42 fresh training agrees with historical
metrics; multi-seed results are reported in the ablation document.
**Interview questions:** Why compute class weights on train only? Why can an
imbalanced model have high accuracy but poor cone IoU? What detail can LiteFCN's
stride-4 representation lose? Where is the best epoch selected?

## Controlled evaluation (`shifts.py`, `domain.py`)

**Problem:** nominal same-generator accuracy does not measure distribution-shift
sensitivity. **Principle:** paired interventions X'_s = T_s(X; ξ), Y'_s = G_s(Y; ξ)
on identical scene IDs. ξ derives from SHA-256(version, corruption seed, ID, kind),
independent of training seed/access order/process hashing. Low/high strengths use
common randomness, reducing nuisance variation. Source and transformed tensor
hashes identify the actual virtual evaluation sets.

**Implementation:** lighting scales RGB; sinusoidal texture multiplies RGB;
Gaussian blur and additive noise change input only. Perspective uses a homography:
u = (ax+by+c)/(gx+hy+1), v = (dx+ey+f)/(gx+hy+1).
Eight corner constraints form an 8×8 linear system. Pillow needs the inverse
(destination → source) map. RGB is bilinear; class IDs are nearest neighbor.
New out-of-view regions are filled as floor. Foreground occlusion replaces a
rectangular RGB region and its labels with obstacle, covering the visible surface.

**Design decisions:** labels are never interpolated; no hidden-object prediction
task is introduced. Viewpoint is explicitly a planar top-inset trapezoid proxy.
Occlusion adds a visible obstacle, so class support changes; supports/confusion
matrices are saved. The virtual dataset avoids duplicated PNG trees, and retains
IDs, recipe/version, manifest hash and content hash. Hashes encode little-endian
float32 RGB and int64 masks in dataset order.
**Alternatives:** a renderer with camera intrinsics/extrinsics would be more
realistic but require a new scene model. Random erasing with unchanged labels
would measure hidden-object inference; ignore regions would remove the challenge
and require a new ignore-label metric contract. Neither matches visible semantic
segmentation here.
**Assumptions/failure modes:** stressor strengths are synthetic, not calibrated;
RGB saturation/blur can destroy minority-class evidence; viewpoint changes class
areas; occluders resemble palette obstacles. Relative degradation is undefined
when nominal IoU is zero. Negative degradation is possible and must be retained.
**Validation:** zero-strength identities, source nonmutation, label bounds,
corner equations, interior image/mask alignment, occluder color/area/labels,
paired dataset hashes, repeat study equality; all 12 shifts evaluated in every run.
**Interview questions:** Why inverse warp? Why not bilinear masks? What is held
constant across models? Why does occlusion change label support? How would you
extend this to a calibrated camera and scene-level splits?

## Configurable training augmentation (`config.py`, `data.py`)

**Problem:** test robustness may benefit from training distribution coverage, but
augmentation can destroy detail or harm nominal performance. **Principle:**
minimize E_(X,Y),ξ [L(fθ(Tξ(X)), Gξ(Y))] rather than assume any augmentation helps.
Sample added strengths uniformly in [0, configured maximum] per scene/epoch.

**Implementation:** retain the original flip → brightness → contrast → noise
path; optional blur → texture → viewpoint → occlusion follows it. New maxima are
zero by default. Only active additions cause re-quantization to RGB bytes. Each
uses deterministic local RNG state; epoch changes draw new transforms.
**Decisions:** keep nominal validation and test unaugmented; forbid augmented
sources in shifted evaluation. Merge zero defaults when comparing old checkpoint
configs, while rejecting any active new transform during exact resume.
**Alternatives:** transform all samples on disk (wastes storage), add torchvision
dependency (unnecessary), reuse global RNG (worker order would matter).
**Assumptions/failures:** composition order matters, re-quantization introduces
≤1/255 differences in active arms, perspective border fills and occlusion alter
training class frequency while class weights remain fixed to source training
counts. These are declared interventions, not isolated sensor physics.
**Validation:** config limits, deterministic epochs, unchanged nominal evaluation,
legacy resume comparison, baseline bitwise resume, individually added transforms
and combined arm measured at identical training budgets.
**Interview questions:** Which RNG determines strength? How does epoch reach
workers? Why leave validation nominal? Why might blur augmentation reduce cone IoU?

## Experiment ownership (`study.py`, `configs/ablation_v1_1.yaml`)

**Problem:** single-run improvements can be noise; failed runs can disappear in
ad hoc notebooks. **Principle:** fixed paired design with seeds 42/43/44,
same initialization/shuffle seed per arm/model, fixed data seed 17 and corruption
seed 2026. Report each observation before aggregation. Sample SD is
sqrt(Σ_s (q_s−q̄)²/(n−1)); it is null for n=1. Paired delta d_s = q_arm,s−q_base,s.

**Implementation:** strict study schema, validated augmentation-only arm changes,
fresh output directory, resolved spec/hash, run ledger written before/after each
attempt. Ordinary exceptions retain tracebacks and continue the matrix. CLI exits
nonzero if any attempt fails; no automatic overwrite/retry or successful-only
selection. Interruptions leave started entries; exact training resume remains a
separate CLI operation. A pretraining dataset failure aborts before the run matrix.
**Decisions/alternatives:** simple JSON/YAML infrastructure rather than another
experiment platform; no post hoc tuning. Seven arms isolate each new transform;
the combined arm measures interaction but cannot attribute its effect to one
component. Lighting/noise are already in the baseline; no_aug measures removal
of the entire original augmentation bundle, not each original transform alone.
**Assumptions/failures:** fixed scene split limits conclusions; seed SD is not a
scene-level confidence interval, n=3 is small, repeated tests risk study overfit
if a subsequent milestone tunes to these scores. Equal mean of 12 shifts assigns
artificial equal prevalence and cannot estimate deployment risk.
**Validation:** tiny full repeated study has identical condition metrics/hashes;
injected first-run failure is persisted while the next run succeeds; real matrix
counts/status/hashes are checked by evidence review. No acceptance gate requires
positive accuracy improvement.
**Interview questions:** Why paired seeds? What does SD quantify? Why is the test
set unsuitable for selecting an arm? What can the combined arm tell you about
causality? What do you do with a crash or negative result?

## Diagnostics and performance (`engine.py`, `benchmarking.py`, `exporting.py`)

**Problem:** NaNs can look like predictions; short latency summaries hide outliers.
**Principle:** finite-value invariants before optimizer steps and metric updates;
timing must measure completed work. CUDA calls are bracketed by synchronization.
**Implementation:** fail on nonfinite loss/logits, save raw timing samples alongside
mean/p50/p95; retain existing numerical export gates and separate parameter,
checkpoint and deployment bytes. Dense fixed-shape latency is measured per model,
not per corruption. Export verifies logits and argmax at two shapes.
**Decisions/alternatives:** existing backend runner rather than new performance
framework. FP32 CPU study prioritizes repeatability; GPU/AMP is a separate execution
check. Host transfers/preprocessing/end-to-end latency remain outside timing.
**Assumptions/failures:** finite logits do not guarantee quality; training guard
does not by itself catch every finite-loss/nonfinite-gradient scenario (AMP scaler
handles overflow); GPU warmup, thermals, clock policy and competing load matter.
TorchScript is deprecated and TensorRT remains unvalidated.
**Validation:** nonfinite evaluator test, preserved failure test, raw timing counts,
CPU lifecycle/parity, fresh CUDA/AMP/resume execution and synchronized benchmarks
where supported; see validation and benchmark records for actual outcomes.
**Interview questions:** Why synchronize CUDA? Why checkpoint size is not model
size? What does export parity prove? Why is a synthetic speed test not robot latency?

## Owner competency checklist

- [ ] Trace one scene through `prepare_dataset`, `SegmentationDataset`, model,
  weighted cross-entropy, best validation checkpoint, export and benchmark.
- [ ] Derive class IoU from `metrics.py` and explain absent-class policies.
- [ ] Derive the homography equations and locate RGB/mask interpolation in `shifts.py`.
- [ ] Explain the visible-surface occlusion contract and changing class supports.
- [ ] Explain local augmentation seeds, epoch state, loader shuffle/RNG restoration.
- [ ] Read `study.json`: find a seed/config, actual confusion matrix, hashes,
  negative paired delta, latency sample, failed record and exact checkpoint.
- [ ] Explain the difference between training-seed dispersion and uncertainty over scenes.
- [ ] Reproduce a small study without silently overwriting a run.
- [ ] Explain nominal versus shifted results without claiming sim-to-real validation.
- [ ] Explain CPU/GPU numerical tolerances, AMP scaler state and CUDA timing scope.
- [ ] Identify what these experiments cannot establish and design the next real-data test.

This checklist describes personal understanding required to present the feature as
a portfolio skill; execution of the code does not certify the owner's competency.
