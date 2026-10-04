# Domain-shift analysis v1.1

Measured 2026-10-04, from the predeclared [plan](PLAN_V1_1.md) and
[configuration](../configs/ablation_v1_1.yaml). Source:
[full measured study](evidence/v1_1/study.json). These are synthetic image
interventions, not physical robot or calibrated sensor validation.

## Evaluation contract

Each checkpoint sees the same 27 test scene IDs from the 180-image, 64×64 dataset
(126 train / 27 val / 27 test, data seed 17). Corruption seed 2026 is independent
of training seeds 42/43/44. Each condition's content hash is identical across all
42 runs. Nominal validation selects checkpoints; shifted test scores select nothing.
Virtual evaluation sets are recreated from the manifest, recipe/version and
per-scene SHA-derived randomness in `src/robotics_vision/shifts.py`.

| Condition | Low | High | Label semantics |
| --- | --- | --- | --- |
| Lighting | RGB ×0.75 | RGB ×0.50 | unchanged |
| Texture | sinusoidal RGB multiplier amplitude 0.20 | amplitude 0.40 | unchanged |
| Blur | Gaussian radius 0.70 px | radius 1.40 px | unchanged |
| Noise | Gaussian std 0.04 in [0,1] | std 0.08 | unchanged; clipped RGB |
| Viewpoint | top inset 10% width | inset 20% | homography, nearest masks, floor fill |
| Occlusion | area 6% | area 15% | visible class-1 obstacle, rounded side lengths |

Texture frequencies are six horizontal and five vertical cycles with seeded phases;
it changes appearance across all surfaces, not material physics. Lighting tests
only dimming, blur is isotropic rather than motion blur, and viewpoint is a planar
top-inset trapezoid proxy, not 3D camera rotation. Occlusion changes class supports;
it evaluates visible-surface segmentation, not recovery of hidden labels.

## Baseline across training seeds

Mean ± sample SD across three optimization seeds. Class IoU columns are seed means.
Drop = nominal mIoU − condition mIoU in **percentage points**; negative drops
are retained. Relative drop is the mean per-seed ratio, not a ratio of means.
Raw JSON includes every confusion matrix, class support, class drop, and per-run
relative drop; no aggregation over the pixels of different trained models occurs.

### tiny_unet

| Condition | mIoU ± SD | Floor | Obstacle | Cone | Robot | Drop pp | Relative drop % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | 0.6874 ± 0.1272 | 0.8909 | 0.7460 | 0.4620 | 0.6507 | 0.00 | 0.00 |
| blur_low | 0.7120 ± 0.1118 | 0.9312 | 0.7729 | 0.5121 | 0.6316 | -2.46 | -4.01 |
| blur_high | 0.6578 ± 0.1103 | 0.9410 | 0.7607 | 0.3375 | 0.5922 | 2.96 | 4.09 |
| lighting_low | 0.6869 ± 0.1371 | 0.9283 | 0.7424 | 0.4578 | 0.6191 | 0.05 | 0.24 |
| lighting_high | 0.4370 ± 0.0112 | 0.9139 | 0.5766 | 0.0565 | 0.2010 | 25.04 | 35.00 |
| noise_low | 0.6679 ± 0.1308 | 0.8724 | 0.7216 | 0.4386 | 0.6390 | 1.95 | 2.95 |
| noise_high | 0.6258 ± 0.1327 | 0.8340 | 0.6638 | 0.4010 | 0.6045 | 6.16 | 9.22 |
| occlusion_low | 0.6684 ± 0.1270 | 0.8662 | 0.7608 | 0.4484 | 0.5981 | 1.90 | 2.81 |
| occlusion_high | 0.6732 ± 0.1314 | 0.8542 | 0.8144 | 0.4387 | 0.5857 | 1.42 | 2.15 |
| texture_low | 0.6803 ± 0.1327 | 0.8853 | 0.7399 | 0.4544 | 0.6416 | 0.71 | 1.14 |
| texture_high | 0.6641 ± 0.1406 | 0.8720 | 0.7186 | 0.4414 | 0.6243 | 2.33 | 3.66 |
| viewpoint_low | 0.6901 ± 0.1138 | 0.9190 | 0.7364 | 0.4906 | 0.6143 | -0.27 | -0.65 |
| viewpoint_high | 0.6703 ± 0.1180 | 0.9271 | 0.7204 | 0.4686 | 0.5651 | 1.71 | 2.34 |

### lite_fcn

| Condition | mIoU ± SD | Floor | Obstacle | Cone | Robot | Drop pp | Relative drop % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | 0.6070 ± 0.0166 | 0.8893 | 0.6387 | 0.3158 | 0.5842 | 0.00 | 0.00 |
| blur_low | 0.6388 ± 0.0356 | 0.9099 | 0.6608 | 0.3938 | 0.5906 | -3.18 | -5.20 |
| blur_high | 0.6669 ± 0.0707 | 0.9263 | 0.6773 | 0.4885 | 0.5755 | -5.99 | -9.76 |
| lighting_low | 0.6814 ± 0.0365 | 0.9355 | 0.6893 | 0.4611 | 0.6399 | -7.44 | -12.24 |
| lighting_high | 0.4326 ± 0.0141 | 0.9150 | 0.5545 | 0.1294 | 0.1315 | 17.44 | 28.72 |
| noise_low | 0.5942 ± 0.0131 | 0.8792 | 0.6335 | 0.2900 | 0.5740 | 1.28 | 2.10 |
| noise_high | 0.5419 ± 0.0275 | 0.8163 | 0.5843 | 0.2215 | 0.5455 | 6.51 | 10.73 |
| occlusion_low | 0.5894 ± 0.0144 | 0.8515 | 0.6370 | 0.3153 | 0.5537 | 1.76 | 2.90 |
| occlusion_high | 0.6013 ± 0.0180 | 0.8281 | 0.7016 | 0.3159 | 0.5597 | 0.57 | 0.94 |
| texture_low | 0.5991 ± 0.0163 | 0.8841 | 0.6356 | 0.3008 | 0.5760 | 0.79 | 1.29 |
| texture_high | 0.5580 ± 0.0418 | 0.8548 | 0.6116 | 0.2754 | 0.4904 | 4.90 | 8.10 |
| viewpoint_low | 0.6155 ± 0.0215 | 0.9095 | 0.6246 | 0.3477 | 0.5801 | -0.85 | -1.39 |
| viewpoint_high | 0.6000 ± 0.0188 | 0.9200 | 0.5970 | 0.3272 | 0.5557 | 0.70 | 1.16 |

## Interpretation and limits

TinyUNet baseline nominal seed results are 0.8342, 0.6165, 0.6115: large optimization
variation undermines a single-seed quality claim. LiteFCN's are 0.5980, 0.6262,
0.5968. Class-wise values reveal failures hidden by floor-heavy pixel accuracy.
Small or negative drops under some conditions are measured outcomes, not discarded
as mistakes; synthetic transforms can simplify boundaries or shift class prevalence.

The equal-weight mean over 12 shifts used in the ablation report is an artificial
stress index. Neither it nor seed SD estimates deployment risk or scene-population
uncertainty. All test scenes come from one procedural generator and split.
Unknown materials, bright/glare lighting, motion blur, lens distortion, true 3D
occlusion/viewpoint, temporal behavior and real robots remain unvalidated.

## Reproduction

```bash
rv-study --study configs/ablation_v1_1.yaml --output-dir runs/my_v1_1_study
python scripts/review_v1_1.py --study runs/my_v1_1_study/study.json
rv-domain --config configs/domain_shift.yaml \
  --checkpoint runs/my_v1_1_study/training/tiny_unet_baseline_s42/checkpoints/best.pt \
  --output runs/my_v1_1_domain.json --shift-seed 2026
```

Output paths must be fresh. Use `--set model.architecture=lite_fcn` with `rv-domain`
when evaluating its checkpoint. See [validation](VALIDATION_V1_1.md) for versions,
environment and execution commands, and [engineering notes](ENGINEERING_NOTES.md)
for equations and ownership questions.

## Owner competency checklist

- [ ] Explain paired corruption seeds and transformed tensor hashes in `shifts.py`.
- [ ] Derive homography sampling and identify nearest-neighbor label resampling.
- [ ] Calculate absolute/relative degradation and explain negative values.
- [ ] Explain why class supports change for viewpoint and occlusion.
- [ ] Distinguish these procedural interventions from physical sensor validation.
