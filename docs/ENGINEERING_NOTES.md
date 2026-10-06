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
Exact resume additionally presumes immutable data: legacy checkpoints compare
manifest paths and do not embed a content-identity gate. Study provenance verifies
the actual data for these experiments; retrofitting checkpoint identity is separate work.
**Validation:** baseline suite and fresh seed-42 training near historical metrics;
small historical numerical differences are recorded rather than called gains.
Exact repeats under the fresh CPU protocol match; multi-seed results are in the
ablation document.
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
attempt; temporary write plus atomic replace preserves the last ledger if a write
is interrupted. Ordinary exceptions retain tracebacks and continue the matrix. CLI exits
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
injected first-run failure is persisted while the next run succeeds; interrupted
ledger-write test preserves the previous JSON; real matrix
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

## Real-data adapter and metric projections (v1.2)

**Problem:** the synthetic four-class label space has no exact public real-data
counterpart. Treating furniture, people or unknown background as arbitrary model
classes would give a misleading sim-to-real score.

**Principle:** evaluate only defined semantic correspondences. For confusion C
(rows truth, columns prediction), binary projection is B = P C Pᵀ, with
P = [[1,0,0,0],[0,1,1,1]]. The real-data truth projection assigns source floor
categories 118/190 to 0 and other annotated categories to 1. Void/crowd are excluded.
For the restricted endpoint, retain truth rows floor/cabinet and all four prediction
columns: predicting cone/robot on a labelled floor/cabinet pixel still creates an
error. IoU_c = C_cc / (row_c + column_c − C_cc); use only the two supported classes
in this endpoint's macro average. Δ_sim2real = metric_projected_ID − metric_real;
a negative gap is possible and is never clipped.

**Implementation:** `real_data.py` reads official ZIP members over verified HTTPS
byte ranges; checks CRC, pinned metadata SHA, segment IDs/areas, licensing, dimensions
and input hashes. `real_evaluation.py` letterboxes RGB at 64/128, removes padding
from argmax predictions and restores them to native ground-truth dimensions.
Ground truth stays native. Frozen checkpoints and nominal confusion matrices come
from the immutable v1.1 study. No training/validation split is invented for this
external test set: no real-data tuning is allowed.

**Decisions/alternatives:** retain existing models/dependencies/training pipeline.
A full four-class map, pseudo-labels or Hypersim would respectively distort semantics,
replace ground truth, or remain synthetic. Binary floor transfer is the primary
endpoint; cabinet-as-storage-obstacle is explicitly a narrow secondary subset.
Selection is metadata-only with a fixed hash order, before any model prediction.
Input size sensitivity is predeclared; it is not test-set hyperparameter selection.

**Assumptions:** COCO panoptic human annotations define visible floor; cabinet
furniture belongs to the synthetic static-obstacle concept. Selected domestic/public
interiors are genuinely real, but not representative industrial warehouse scenes.
**Failure modes:** semantic mismatch, floor sampling bias, native boundary errors
from low-resolution logits, invalid/corrupt annotations, silent void inclusion,
unsupported-class false predictions and distribution-dependent BatchNorm behavior.
**Validation:** see offline native-geometry/projection/ignore/corruption tests in
`tests/test_real_data.py` and `tests/test_real_evaluation.py`. Actual photo results
will be recorded in the v1.2 evidence and analysis documents; tests alone establish
no generalization improvement.
**Interview questions:** Why not compare four-class ID mIoU to binary real mIoU?
Why can ignored cabinet-surrounding pixels inflate apparent secondary accuracy?
What does this class mapping fail to establish? Why restore predictions instead of
shrinking labels? How could final-test tuning create experiment leakage?

## Owner competency checklist — v1.2 experiment design

- [ ] Derive the binary confusion projection and the restricted rectangular label-space metric.
- [ ] Trace `prepare_real_dataset` → manifest validation → `letterbox` → native predictions.
- [ ] Explain domain shift, distribution mismatch, augmentation and synthetic-to-real transfer.
- [ ] Explain overfitting, nominal validation checkpoint selection and final-test leakage.
- [ ] Explain why there is no real validation split when there is no real-data tuning.
- [ ] Compute per-class IoU/precision/recall and identify undefined/ignored classes.
- [ ] Distinguish training-seed variance from uncertainty across images or datasets.
- [ ] Locate opposite architecture-dependent v1.1 augmentation results and check their real-data signs.
- [ ] Explain why synthetic corruption correlations are descriptive and not causal evidence.
- [ ] Identify limitations of synthetic benchmarks, partial labels and selected COCO interiors.

### Interrupted real-study continuation

A server restart detached the worker while its ledger showed 63 completed
measurements and one `started` entry. The original worker subsequently completed
all 84; attempted continuation correctly refused an already-complete study.
No actual evaluation was retried or marked interrupted. `continue_report` checks source and dataset
hashes and the predeclared evaluation keys before continuing. Successful and failed
terminal results stay immutable; only incomplete attempts are retried, and their
`interrupted` records remain. Duplicate terminal keys and changed input hashes
fail. This is a ledger continuation, independent of optimizer checkpoint/resume.
The injected interruption/changed-input test verifies that prior evidence survives.

## Paired statistics and failure selection (v1.2)

**Problem:** a favorable seed or a synthetic-corruption ranking can be mistaken
for reliable real-data improvement. **Principle:** within each fixed architecture,
Δ_s = metric_augmented,s − metric_baseline,s; report all three values and
sample SD sqrt(Σ_s(Δ_s − mean Δ)²/(n−1)). These three training seeds do not estimate
uncertainty over new datasets. Spearman ρ is Pearson correlation of average ranks;
constant ranks return null, not an invented zero. Correlating drops shares the ID
term, so also inspect raw score correlations. No p-values or causal mechanism claims.

**Implementation:** `real_analysis.py` refuses incomplete matrices, retains negative
values, groups all seven arms/seeds and computes paired differences and within-model
21-run associations. `review_real_v1_2.py` independently checks native per-image
confusion sums, metric arithmetic, source/data hashes and the metadata-only subset.
`gallery_real_v1_2.py` selects baseline seed42 worst/lower-median/best floor IoU
and the lower-scoring member of each input-tag pair, then shows both architectures'
baseline and occlusion predictions. Selection is deterministic and recorded.

**Decisions/alternatives:** use existing NumPy/standard library, average tied ranks,
no new statistics dependency. Do not pool architectures into a potentially
confounded correlation or choose arms from real scores. A scene/bootstrap confidence
interval would answer a different question and is left to a representative dataset
milestone, not substituted for measured optimization SD.

**Assumptions:** seed pairing improves comparability on fixed training/evaluation
splits; image-level pixel aggregation deliberately weights large images more.
**Failure modes:** shared-term/spurious associations, non-independent arms/seeds,
large seed dispersion, scene-selection bias, missing author attribution and subjective
tags. A thumbnail ID typo (343468 → 343466) was corrected against source metadata;
the original pre-prediction tags are retained and the correction is explicit.
**Validation:** known tied/negative/constant correlation and sample-SD tests;
complete-matrix gate; all 84 native metric records recomputed; exact repeat real
metrics from both reproduced baseline checkpoints; gallery predictions checked
against stored per-image counts. Flickr lookup failures are provenance limits,
not exclusions from numerical evaluation.
**Interview questions:** Why report paired SD rather than subtract group SDs?
Why is a positive drop correlation insufficient to select a corruption surrogate?
How could pixel pooling hide minority-scene failure? What does a retained bad example
or a neutral seed contribute to the engineering claim?

## Owner competency checklist — v1.2 measured evidence

- [ ] Recompute baseline/occlusion/combined real deltas from each seed in `real_runs.csv`.
- [ ] Explain the reversal of the synthetic mean augmentation signs and the mixed seed signs.
- [ ] Explain why real binary mIoU gains can coexist with nearly unchanged floor IoU.
- [ ] Locate floor/cabinet gaps and state why cone/robot gaps are null.
- [ ] Contrast drop-gap versus score-score correlations and explain the common-ID confound.
- [ ] Read a failure panel and distinguish false floor, missed floor, void and unsupported predictions.
- [ ] Explain which data-selection, attribution, CUDA and hardware limitations remain.

### Frozen input identity on fresh downloads

A release-reproduction review identified that locally recorded hashes alone accept
self-consistent changed downloads. Preparation/evaluation/review scripts now compare
image/mask hashes, source annotation records, selection and semantic categories with
the tracked original manifest. Renaming the data root is permitted; silently changing
the final test contents is not. The offline test replaces an image and updates its
local hash: ordinary integrity succeeds, frozen-reference identity correctly fails.
The original 80-image manifest passes this new gate unchanged; measured metrics are
not recomputed or altered by this validation-only change. Captured Pillow versions
matter because a lossless mask can serialize to different PNG bytes across versions.
The native boundary band uses a radius 3 square (Chebyshev) dilation of valid
floor/nonfloor adjacency edges, excluding void pixels; it is not a calibrated
physical boundary-distance measure.

### Class-gap comparison requires identical evaluation masks

Primary floor and restricted storage IoUs have different false-positive eligibility.
Ranking their gaps against each other would be invalid. The v1.2 analysis records
restricted floor IoU/gap alongside storage on the identical shared mask. Storage
has the larger shared-mask gap in both baseline models; broad binary floor failure
remains severe separately. Source: `real_analysis.observed_values` and the class table
in `SIM_TO_REAL_ANALYSIS_V1_2.md`. Explain why ignoring other categories changes
a floor union even when all predictions stay identical.
