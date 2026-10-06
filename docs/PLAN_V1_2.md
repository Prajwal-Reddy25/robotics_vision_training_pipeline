# v1.2 experiment plan — frozen before prediction evaluation

2026-10-05. Exact v1.1 baseline: `d9a74a0`. No v1.1 tag exists in this checkout.
No model, optimizer, augmentation setting, seed or epoch will be tuned on real data.

## Dataset and scientific scope

Use **COCO 2017 validation images with official COCO panoptic ground truth**.
Real photographs, human annotations, native-resolution non-overlapping segment
masks. Official source: [COCO](https://cocodataset.org/),
[panoptic API/category definitions](https://github.com/cocodataset/panopticapi),
and the public `images.cocodataset.org` S3 bucket. COCO/COCO-Stuff annotations
are CC BY 4.0; photos have individual Flickr licenses. Restrict this study to
metadata license ID 4 (CC BY 2.0). Retain source/Flickr links and attribution
metadata; raw photos remain ignored locally. No annotation model creates labels.

Investigated COCO-Stuff, NYUv2, warehouse pallet datasets and Hypersim.
COCO-Stuff's original annotation endpoint returned HTTP 406; original COCO HTTPS
hostname had a certificate mismatch. Verified HTTPS path-style S3 access works,
including byte ranges. Official panoptic data avoid a third-party mirror and any
TLS verification bypass. NYUv2 is large with less explicit usage terms on its
primary page; warehouse instance datasets do not supply matching four-class
dense labels; Hypersim is photorealistic but would not be real-data validation.

COCO has **no cone or mobile-robot semantic category** compatible with this
project. Do not map people/cars to robots or arbitrary background to obstacles.
Primary endpoint: floor-vs-other-known-category segmentation, projecting existing
prediction class 0 to floor and classes 1/2/3 to nonfloor. Ground-truth floor
categories are 118 `floor-wood` and 190 `floor-other-merged`. Other annotated
categories are nonfloor; panoptic void and crowd regions are ignored.

Secondary restricted endpoint: floor and category 188 `cabinet-merged` (storage
furniture, a rectangular static-obstacle subset). Map cabinet pixels to model
class 1; **ignore all remaining source categories**, report pixel coverage and
all four prediction columns, and average only target classes 0/1. Unknown
predictions 2/3 remain errors on labelled pixels. This evaluates a narrow obstacle
subset, not warehouse obstacle completeness. Cones/robots remain unvalidated.
ID comparisons use the **same projections** of frozen confusion matrices;
never subtract four-class ID mIoU from binary real-data mIoU.

Select before predictions: images with metadata CC BY license 4 and ≥2% native
floor area. Source metadata has 131 eligible images, 34 with ≥1% cabinet area.
Take all 34 cabinet cases and 46 other cases by SHA-256(`2026:<image_id>`) order,
for 80 images. Record the entire inclusion/exclusion ledger. This stratification
ensures secondary label support but is not a representative warehouse distribution.
No real data is used for training/tuning; all 80 are final evaluation data.
There is no real validation split because no tuning is permitted.

## Frozen comparisons and questions

Use all **42 v1.1 checkpoints** (2 architectures × 7 augmentation arms × seeds
42/43/44). Primary contrasts: baseline vs occlusion, then combined. Remaining
arms stay in the analysis; no successful-only selection. Zero new experimental
training runs: source training/configs and validation-selected checkpoints stay frozen.

1. Compare paired real-data changes per architecture/seed, including losses.
2. Test whether the opposite occlusion/combined responses seen in v1.1 persist
   for the supported floor/storage endpoints, without extrapolating to all classes.
3. Correlate each synthetic shift's projected degradation with real degradation
   within each architecture (21 frozen runs). Spearman associations are descriptive;
   repeated arms/seeds are not independent samples, and both drops share ID terms.
   Also report score associations to expose that confound; do not claim causation.
4. Report floor and restricted storage-class gaps; cone/robot gaps are null,
   since compatible ground truth does not exist here.
5. Pair real gains/losses with original four-class nominal changes and comparable
   projected ID changes; do not claim an accuracy/robustness tradeoff if unmeasured.

## Preprocessing, metrics and failure analysis

RGB float32 [0,1], no color standardization. Preserve aspect ratio with bilinear
resize into a centered **64×64 letterbox**; constant RGB 127 padding. Crop padding
from class predictions, then nearest-neighbor upsample to **native ground-truth
resolution**. Ground truth is never resized for the primary metrics. Evaluate
128×128 as a predeclared resolution sensitivity check, not a tuned input choice.
Both existing networks accept these inputs without retraining.

Report binary mIoU, floor/nonfloor IoU, pixel accuracy, precision/recall; secondary
restricted class IoU/mIoU and coverage; full source confusion/false-floor counts;
delta_sim2real = comparable projected ID metric − real metric. Save native
per-image confusion matrices and all individual seeds, mean/sample SD, paired
augmentation differences. Latency at both fixed tensor sizes, one CPU thread,
20 warmups/100 model calls; checkpoint/parameter bytes separately. Export existing
frozen models, verify dynamic/parity paths; preprocessing time is reported separately
where timed, not silently included in model-call latency.

Boundary errors: error rates within a 3-native-pixel band around floor boundaries
and away from boundaries, with ignored pixels excluded. Record input luminance,
gradient/blur proxies and prediction-vs-source-category confusion. Inspect real
images and annotate descriptive lighting/shadow/texture/occlusion/blur/viewpoint
tags before inspecting model results; tags are qualitative, not measured physics.
Panels include deterministic worst/median/best cases and tag-specific failures.
If even the best fails a meaningful threshold, call it relatively best, not success.

## Engineering and release gates

- Keep nominal data/models/training and all original tests. Add dedicated
  external-data validation, ignore-label/projected metrics, letterbox/native
  geometry, boundary/correlation and reproducibility tests, plus a tiny offline
  real-evaluator integration fixture clearly marked as a synthetic test fixture.
- Hash-check every source/checkpoint/input; validate panoptic segment IDs/areas
  against metadata, paths/modes/dimensions and license selection; duplicate images
  rejected; missing/corrupt data fails rather than being skipped silently.
- Preserve CPU/CUDA/resume/export checks. GPU bitwise limitations remain visible;
  no new strict CUDA determinism claim is a release gate.
- Run all 84 real dataset evaluations (42 frozen models × 2 sizes), retain all
  outputs/failures, update source/data/metric/qualitative evidence and requested docs.
- Recommend v1.2 only for **real-photo floor transfer plus a restricted storage
  obstacle endpoint**, with unsupported classes, sampling bias and domain mismatch
  explicit. No physical-robot or full four-class sim-to-real claim. No tag/push.
