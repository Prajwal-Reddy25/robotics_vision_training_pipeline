# Augmentation ablation v1.1

Measured 2026-10-04. The full predeclared 2 models × 7 arms × 3 seeds × 15 epochs
completed: **42 successful training runs, zero execution failures**, 546 complete
nominal/shift evaluations. See [raw records](evidence/v1_1/study.json), including
every configuration, checkpoint hash, selected epoch, timing sample and metric.
Negative/neutral experiments below are deliberately retained.

## Controlled design

Data seed 17; optimization seeds 42/43/44; corruption seed 2026. All arms share
the same scenes/splits, model width 8, AdamW learning rate 0.001/weight decay 0.0001,
batch 16, inverse-square-root class weights, cosine schedule, 15 epochs and
nominal-validation checkpoint selection. CPU FP32, one thread. New transform
strengths are sampled uniformly from zero to the named maximum each scene/epoch.
The original bundle is flip p=0.5, brightness/contrast ±0.15 and noise std 0.02.

| Arm | Change from baseline |
| --- | --- |
| no_aug | disable the entire original augmentation bundle |
| baseline | original bundle, all new transforms disabled |
| blur | add radius up to 1.4 px |
| texture | add sinusoidal amplitude up to 0.4 |
| viewpoint | add top inset up to 0.2 |
| occlusion | add obstacle area up to 0.15 |
| combined | all four additions, in blur → texture → viewpoint → occlusion order |

No architectures or hyperparameters were selected using test outcomes. Mean
shifted mIoU weights each of the 12 fixed conditions equally. Mean ± sample SD
measures training-seed dispersion. Deltas are paired by optimization seed against
baseline, in percentage points; raw paired deltas and their SD are also retained.
This is not a confidence interval or significance test over real scenes.

| Model | Arm | Nominal mIoU ± SD | Shifted mIoU ± SD | Nominal Δ pp | Shifted Δ pp |
| --- | --- | ---: | ---: | ---: | ---: |
| lite_fcn | baseline | 0.6070 ± 0.0166 | 0.5933 ± 0.0240 | +0.00 | +0.00 |
| lite_fcn | blur | 0.5960 ± 0.0100 | 0.5766 ± 0.0205 | -1.10 | -1.67 |
| lite_fcn | combined | 0.5499 ± 0.0627 | 0.5476 ± 0.0727 | -5.71 | -4.57 |
| lite_fcn | no_aug | 0.6224 ± 0.0191 | 0.5894 ± 0.0262 | +1.54 | -0.39 |
| lite_fcn | occlusion | 0.6047 ± 0.0448 | 0.5721 ± 0.0537 | -0.23 | -2.12 |
| lite_fcn | texture | 0.6268 ± 0.0242 | 0.6023 ± 0.0056 | +1.98 | +0.90 |
| lite_fcn | viewpoint | 0.6140 ± 0.0066 | 0.5831 ± 0.0200 | +0.70 | -1.02 |
| tiny_unet | baseline | 0.6874 ± 0.1272 | 0.6528 ± 0.1136 | +0.00 | +0.00 |
| tiny_unet | blur | 0.7067 ± 0.0939 | 0.6794 ± 0.1006 | +1.93 | +2.65 |
| tiny_unet | combined | 0.7365 ± 0.0748 | 0.6999 ± 0.0765 | +4.91 | +4.71 |
| tiny_unet | no_aug | 0.6424 ± 0.1493 | 0.5982 ± 0.1184 | -4.50 | -5.46 |
| tiny_unet | occlusion | 0.7486 ± 0.0762 | 0.7101 ± 0.0696 | +6.12 | +5.73 |
| tiny_unet | texture | 0.6931 ± 0.1138 | 0.6481 ± 0.0998 | +0.57 | -0.47 |
| tiny_unet | viewpoint | 0.7011 ± 0.0960 | 0.6624 ± 0.1079 | +1.37 | +0.95 |

## All nominal observations

| Model | Arm | Seed 42 | Seed 43 | Seed 44 | Best epochs (zero based) |
| --- | --- | ---: | ---: | ---: | --- |
| lite_fcn | baseline | 0.5980 | 0.6262 | 0.5968 | 13, 14, 14 |
| lite_fcn | blur | 0.5888 | 0.5917 | 0.6075 | 13, 14, 14 |
| lite_fcn | combined | 0.5840 | 0.5882 | 0.4776 | 13, 14, 14 |
| lite_fcn | no_aug | 0.6009 | 0.6376 | 0.6286 | 14, 14, 14 |
| lite_fcn | occlusion | 0.6186 | 0.6409 | 0.5547 | 13, 14, 14 |
| lite_fcn | texture | 0.6029 | 0.6513 | 0.6262 | 13, 5, 14 |
| lite_fcn | viewpoint | 0.6178 | 0.6063 | 0.6179 | 5, 14, 14 |
| tiny_unet | baseline | 0.8342 | 0.6165 | 0.6115 | 6, 5, 5 |
| tiny_unet | blur | 0.8054 | 0.6185 | 0.6962 | 5, 4, 14 |
| tiny_unet | combined | 0.8228 | 0.6918 | 0.6948 | 5, 5, 4 |
| tiny_unet | no_aug | 0.8122 | 0.5319 | 0.5830 | 7, 6, 5 |
| tiny_unet | occlusion | 0.8355 | 0.6935 | 0.7166 | 6, 5, 5 |
| tiny_unet | texture | 0.8245 | 0.6263 | 0.6287 | 5, 5, 5 |
| tiny_unet | viewpoint | 0.8030 | 0.6125 | 0.6877 | 6, 4, 6 |

## Measured findings, including negative results

TinyUNet occlusion augmentation has a mean shifted gain of **5.73 pp**, but LiteFCN
loses **2.12 pp** with the same addition. Combined augmentation gains **4.71 pp**
for TinyUNet and loses **4.56 pp** for LiteFCN. Blur gains **2.65 pp** for TinyUNet
and loses **1.67 pp** for LiteFCN. Texture gains **0.90 pp** for LiteFCN and loses
**0.47 pp** for TinyUNet. The baseline is not replaced with an arm chosen on this
test set. Three seeds and 27 held-out scenes do not justify a universal ranking.

Hypotheses: skip features may tolerate transformed boundaries better; LiteFCN's
lower-resolution bottleneck may discard small-class detail under composed shifts;
additional obstacle examples may change class balance. These mechanisms were
**not isolated or demonstrated** by this experiment. A further study would need
feature/boundary analysis, changed scene splits and independent validation data.

The earlier single-seed baseline mIoU values were 0.833952/0.598296; fresh seed-42
values are 0.834201/0.597966 under this explicitly one-thread protocol. These small
historical numerical differences are not augmentation improvements. Within the
fresh protocol, two independent full training repeats are bitwise identical.

## Failure preservation

No real ablation run crashed; several lost accuracy and all are retained. Tests
inject a nonfinite-loss failure and an interrupted ledger write: failed run/error
records survive, subsequent runs execute, and the CLI reports failure. A separate
real CUDA exact-resume probe failed once for LiteFCN FP32 (max state delta
1.79e-7), while its later collection passed. Both attempts are preserved in
[validation](VALIDATION_V1_1.md); successful GPU reruns do not erase the failure.

## Reproduction and ownership

Run `rv-study --study configs/ablation_v1_1.yaml --output-dir runs/my_v1_1_study`
with a fresh directory. A failed study exits nonzero, leaves partial training
artifacts and a ledger/traceback, and never silently overwrites results. Use the
existing exact-resume CLI for a separately named interrupted training experiment.

## Owner competency checklist

- [ ] Trace augmentation-only arm changes in `study.py` and the YAML plan.
- [ ] Compute a paired seed delta and sample SD from raw observations.
- [ ] Explain why the combined arm cannot isolate each component's effect.
- [ ] Describe negative results without post hoc exclusions or test-based tuning.
- [ ] Explain which additional experiments would test the stated hypotheses.
