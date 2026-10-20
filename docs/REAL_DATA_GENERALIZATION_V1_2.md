# Real-data generalization v1.2

Measured 2026-10-05/06. This is a **partial-ontology real-photo evaluation** of
all 42 frozen v1.1 checkpoints, not a full four-class warehouse deployment claim.
The [predeclared plan](PLAN_V1_2.md) predates predictions; the [audit](AUDIT_V1_2.md)
identifies exact baseline `d9a74a0bebc48b0facd049f24cb2889bf3bbda4e`. The v1.1 tag
was unavailable during collection; it now points to that same commit. No architecture
or training configuration was added/tuned. Publication and later validation are
tracked in the [dated maintenance review](MAINTENANCE_REVIEW_2026_10_20.md).

## Dataset and label contract

Official [COCO panoptic API/categories](https://github.com/cocodataset/panopticapi)
and [COCO terms](https://cocodataset.org/#termsofuse).
Annotations: COCO Consortium, CC BY 4.0. Each photograph has its own Flickr license;
only metadata license ID4 (CC BY 2.0) is eligible. The pinned metadata/image URLs,
full annotation records, hashes and all 5,000 inclusion/exclusion decisions are in
[manifest.json](evidence/v1_2/manifest.json). Raw data are ignored locally under
`data/external/coco_v1_2`; no pseudo-labels/manual masks replace public ground truth.

Of 5,000 validation images, 4,143 fail the license filter and 726 remaining images
lack ≥2% native floor. Of 131 eligible images, take 34 with ≥1% cabinet plus 46
others in SHA-256(`2026:<image_id>`) order; 51 eligible images hit the fixed cap.
All 80 are final evaluation data, never training/validation/tuning data. This is
floor- and cabinet-enriched selection, not representative warehouse sampling.
There is no real validation split because no real-data tuning is performed.

| Endpoint | Real ground truth | Model predictions | Ignored / incompatibilities |
| --- | --- | --- | --- |
| Binary | 118/190 floor; others nonfloor | 0 floor; 1/2/3 nonfloor | void/crowd |
| Restricted | floor0, cabinet188 → 1 | all four prediction columns | other GT ignored |

Source categories are 118 `floor-wood`, 190 `floor-other-merged`, 188 `cabinet-merged`.
Rugs and other categories keep their COCO semantics; this is not a traversability
map. Real walls/furniture are novel nonfloor categories absent from the synthetic
object vocabulary, so the primary score includes semantic-complexity mismatch.

Cabinets are a narrow static-obstacle subset. COCO has no compatible safety-cone
or mobile-robot ground truth. Those class gaps are null, not zero. Unsupported
categories are not relabelled to manufacture compatibility. Restricted metrics
ignore surrounding walls/people, so their apparent floor IoU cannot replace the
primary binary endpoint. Coverage is 4,125,539 / 20,123,139 valid pixels (20.50%);
cabinet support is 1,644,495 pixels. Native total is 21,994,306 pixels.

RGB [0,1], bilinear aspect-preserving centered letterbox, RGB127 padding, 64²
primary input. Crop padding from argmax predictions, nearest-neighbor restore to
original image size; **ground truth stays native**. 128² is a predeclared sensitivity
analysis, not test-set-selected resolution. ID references use projected original
64² confusion matrices. The 128² ID−real column retains that 64² reference and
mixes input-resolution effects with dataset shift; interpret matched gaps at 64².

## Entire measured matrix

Mean ± sample SD over training seeds 42/43/44. Metrics pool native pixels within
one dataset per seed; large images receive more weight. *Storage IoU uses only the
restricted floor/cabinet truth mask. All arms, including negative/neutral results,
are retained. No seed or arm is selected for deployment from this final test.

| Model | Arm | Input | Real binary mIoU | Floor IoU | Storage IoU* | ID−real binary |
| --- | --- | --- | --- | --- | --- | --- |
| lite_fcn | baseline | 64 | 0.2711 ± 0.0809 | 0.1314 ± 0.0082 | 0.0193 ± 0.0193 | 0.4546 ± 0.0770 |
| lite_fcn | baseline | 128 | 0.2670 ± 0.0803 | 0.1309 ± 0.0081 | 0.0261 ± 0.0294 | 0.4586 ± 0.0774 |
| lite_fcn | blur | 64 | 0.2886 ± 0.0973 | 0.1288 ± 0.0040 | 0.0593 ± 0.0888 | 0.4223 ± 0.0770 |
| lite_fcn | blur | 128 | 0.2845 ± 0.1090 | 0.1297 ± 0.0027 | 0.0658 ± 0.1025 | 0.4264 ± 0.0885 |
| lite_fcn | combined | 64 | 0.2862 ± 0.0867 | 0.1303 ± 0.0063 | 0.0908 ± 0.1268 | 0.4141 ± 0.0793 |
| lite_fcn | combined | 128 | 0.2882 ± 0.0932 | 0.1279 ± 0.0064 | 0.0927 ± 0.1275 | 0.4121 ± 0.0858 |
| lite_fcn | no_aug | 64 | 0.2998 ± 0.0698 | 0.1299 ± 0.0112 | 0.0421 ± 0.0584 | 0.4383 ± 0.0584 |
| lite_fcn | no_aug | 128 | 0.2968 ± 0.0815 | 0.1305 ± 0.0091 | 0.0562 ± 0.0775 | 0.4413 ± 0.0698 |
| lite_fcn | occlusion | 64 | 0.2799 ± 0.0820 | 0.1319 ± 0.0078 | 0.0411 ± 0.0491 | 0.4611 ± 0.0878 |
| lite_fcn | occlusion | 128 | 0.2829 ± 0.0848 | 0.1303 ± 0.0082 | 0.0483 ± 0.0637 | 0.4582 ± 0.0916 |
| lite_fcn | texture | 64 | 0.2534 ± 0.0937 | 0.1302 ± 0.0070 | 0.0292 ± 0.0332 | 0.4886 ± 0.0994 |
| lite_fcn | texture | 128 | 0.2582 ± 0.0964 | 0.1293 ± 0.0063 | 0.0346 ± 0.0451 | 0.4839 ± 0.1039 |
| lite_fcn | viewpoint | 64 | 0.2382 ± 0.1162 | 0.1300 ± 0.0079 | 0.0590 ± 0.0714 | 0.4970 ± 0.1033 |
| lite_fcn | viewpoint | 128 | 0.2374 ± 0.1196 | 0.1305 ± 0.0048 | 0.0603 ± 0.0850 | 0.4978 ± 0.1060 |
| tiny_unet | baseline | 64 | 0.1913 ± 0.0365 | 0.1271 ± 0.0087 | 0.0275 ± 0.0236 | 0.5583 ± 0.1619 |
| tiny_unet | baseline | 128 | 0.1949 ± 0.0217 | 0.1262 ± 0.0074 | 0.0327 ± 0.0269 | 0.5547 ± 0.1484 |
| tiny_unet | blur | 64 | 0.2507 ± 0.0342 | 0.1336 ± 0.0052 | 0.0081 ± 0.0071 | 0.5480 ± 0.0913 |
| tiny_unet | blur | 128 | 0.2474 ± 0.0318 | 0.1310 ± 0.0057 | 0.0108 ± 0.0094 | 0.5512 ± 0.0805 |
| tiny_unet | combined | 64 | 0.1668 ± 0.0529 | 0.1269 ± 0.0093 | 0.0235 ± 0.0229 | 0.6551 ± 0.0878 |
| tiny_unet | combined | 128 | 0.1707 ± 0.0481 | 0.1275 ± 0.0093 | 0.0243 ± 0.0193 | 0.6512 ± 0.0786 |
| tiny_unet | no_aug | 64 | 0.2735 ± 0.0655 | 0.1376 ± 0.0125 | 0.0683 ± 0.0091 | 0.4285 ± 0.2227 |
| tiny_unet | no_aug | 128 | 0.2672 ± 0.0452 | 0.1337 ± 0.0116 | 0.0759 ± 0.0259 | 0.4348 ± 0.2026 |
| tiny_unet | occlusion | 64 | 0.1822 ± 0.0179 | 0.1283 ± 0.0013 | 0.0157 ± 0.0121 | 0.6490 ± 0.0464 |
| tiny_unet | occlusion | 128 | 0.1877 ± 0.0213 | 0.1282 ± 0.0011 | 0.0187 ± 0.0154 | 0.6435 ± 0.0436 |
| tiny_unet | texture | 64 | 0.1596 ± 0.0305 | 0.1263 ± 0.0025 | 0.0163 ± 0.0158 | 0.6112 ± 0.1215 |
| tiny_unet | texture | 128 | 0.1685 ± 0.0398 | 0.1258 ± 0.0011 | 0.0205 ± 0.0178 | 0.6024 ± 0.1120 |
| tiny_unet | viewpoint | 64 | 0.1830 ± 0.0428 | 0.1265 ± 0.0055 | 0.0238 ± 0.0229 | 0.6107 ± 0.0823 |
| tiny_unet | viewpoint | 128 | 0.1847 ± 0.0443 | 0.1253 ± 0.0050 | 0.0270 ± 0.0274 | 0.6090 ± 0.0763 |

[Individual runs/classes/precision/recall/accuracy](evidence/v1_2/real_runs.csv),
[raw per-image counts and timings](evidence/v1_2/study.json),
[paired and correlation analysis](evidence/v1_2/analysis.json).
84 completed dataset evaluations = 42 frozen models × 2 sizes, 6,720 photo-model
predictions. Two additional engineering repeat evaluations reproduce metrics exactly.
Zero new experimental training runs; 42 old training runs reused. Separate release
checks run two full CPU repeats, four CUDA smoke trains and four CUDA resumes.

Seed SD is optimization variability on one fixed split. It is not a confidence
interval over independent scenes. No statistical significance or physical robot
claim follows. See [sim-to-real conclusions](SIM_TO_REAL_ANALYSIS_V1_2.md),
[failure analysis](FAILURE_ANALYSIS_V1_2.md) and [release gates](RELEASE_READINESS_V1_2.md).

## Reproduction

Use the executed CPU versions in `evidence/v1_2/cpu_versions.txt`; install the
appropriate PyTorch CPU build and this repository with its existing `[all]` extras.
The study needs 42 v1.1 checkpoints (model binaries are intentionally not in Git).
Keep existing `runs/v1_1_ablation` if available. Otherwise reproduce the original
predeclared CPU study under the captured environment:

```bash
rv-study --study configs/ablation_v1_1.yaml --output-dir runs/reproduced_v1_1
python scripts/review_v1_1.py --study runs/reproduced_v1_1/study.json
python scripts/prepare_real_v1_2.py
RV_V1_2_SOURCE=runs/reproduced_v1_1/study.json
python scripts/evaluate_real_v1_2.py --source "$RV_V1_2_SOURCE" --output-dir runs/real
python scripts/review_real_v1_2.py --source "$RV_V1_2_SOURCE" --study runs/real/study.json
```

Use a fresh output directory. Reproduced checkpoints/source JSON may have different
serialization/path/commit hashes; compare weights, configurations and confusion
counts, not arbitrary file-byte identity across renamed runs. Original frozen
measurements reject changed original checkpoint hashes. Two original configuration
repeats independently proved best-weight and native-metric equality in this environment;
all 42 repeat trainings were not rerun for v1.2. Endpoint availability/license metadata
can change; pinned annotations and expected hashes fail closed. Internet is required
only to prepare external data. Scripts compare fresh image/mask/source identities
with the tracked original manifest; updated local hashes cannot redefine this test.
Use the captured Pillow version to preserve deterministic PNG serialization.
HTTPS ZIP range requests avoid downloading the entire
860MB annotation bundle; TLS is never disabled. No new dependency is introduced.
`--resume` is only for a stopped ledger with identical source/data; failed terminal
attempts stay failed, completed studies cannot be resumed. Never attach a second
worker to an active output directory; continuation is not a concurrent-run lock.

## Owner competency checklist

- [ ] Explain the license/subset/ontology decision and unsupported semantic classes.
- [ ] Recompute a native floor IoU and comparable ID−real gap from saved counts.
- [ ] Explain seed SD, pixel pooling, test leakage and resolution sensitivity limits.
- [ ] Reproduce the frozen CPU configuration without tuning on this real test set.
