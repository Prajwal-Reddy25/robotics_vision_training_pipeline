# v1.2.0 — real-data generalization and sim-to-real evaluation

The v1.1 synthetic augmentation gains could not establish real-camera generalization.
v1.2 freezes the existing42 models and evaluates them on80 independently sourced
COCO real photographs with native panoptic ground truth. It adds explicit floor and
restricted storage-obstacle mappings, hash/licensing/geometry checks, paired
three-seed analyses, failure visualizations and reproducible evidence. No additional
architecture, pretrained model, real-data tuning or runtime dependency is introduced.

## Measured results

84 complete real dataset evaluations (two predeclared input sizes), zero model
execution failures; two additional exact engineering repeat evaluations.
Primary 64² binary mIoU, mean ± sample SD:

| Frozen recipe | TinyUNet | LiteFCN |
| --- | --- | --- |
| Baseline | 0.1913 ± 0.0365 | 0.2711 ± 0.0809 |
| Occlusion augmentation | 0.1822 ± 0.0179 | 0.2799 ± 0.0820 |
| Combined augmentation | 0.1668 ± 0.0529 | 0.2862 ± 0.0867 |

Occlusion real mean changes are −0.0091/+0.0089, reversing the synthetic mean
signs even under a matched binary projection. Individual signs are mixed. Disabling
baseline augmentation improves all three seeds in both models; neutral/negative
arms remain recorded. Floor IoU is only 0.1271/0.1314 at baseline, with comparable
ID−real floor gaps 0.7638/0.7578. These are substantial transfer failures, not a
production-readiness claim or an augmentation-win narrative.

TinyUNet/LiteFCN CPU model-call p50 at 64²: **1.5152/0.3498ms** (three-seed mean).
At 128²:4.3482/0.6968ms. Parameter counts 29,948/2,804. Raw latency samples, class
metrics, seed deltas, sizes and correlation results are retained.

## Validation and limitations

CPU 76 passed with one CUDA skip; CUDA 77 passed. Ruff and dependency checks pass;
wheel build/import, frozen source/data reviews, exact CPU repeat/resume, two repeated
real-metric evaluations and real-input TS/ORT parity at 64²/128² pass. Fresh GPU
FP32/AMP smoke/resume and synchronized eager/TS benchmarks pass. The prior v1.1
LiteFCN CUDA bitwise-resume failure remains documented; no strict general CUDA
determinism guarantee is added. TensorRT and physical-camera/robot validation
remain unavailable. Hosted GitHub Actions was not executed from this local branch.

COCO interiors are floor/cabinet-enriched, not representative warehouses. Cones
and mobile robots lack compatible labels; cabinet metrics evaluate a restricted
subset. n=3 seed dispersion is not scene-level uncertainty. Correlations are
confounded/descriptive; mechanism explanations remain hypotheses. Nine gallery
photos have recovered source creator credits; two input photos stay local because
attribution is unresolved, with numerical results and diagnostic masks retained.

Next milestone: representative warehouse data with all supported classes,
scene/session-grouped tuning and final-test splits, scene-level uncertainty and
controlled adaptation/mechanism experiments. Keep this external final test out of
tuning. CUDA determinism/export migration remain separate engineering work.

Recommend **v1.2.0 for this documented partial-ontology evaluation scope**. No tag
or publication is created automatically. See [release gate](RELEASE_READINESS_V1_2.md)
and [full measured report](REAL_DATA_GENERALIZATION_V1_2.md).

## Owner competency checklist

- [ ] Explain the evidence supporting the recommendation without claiming robust robot perception.
- [ ] State the failed-transfer result, retained negative seeds and scope of the real labels.
- [ ] Reproduce results and explain dataset leakage, seed variance and the next missing experiment.
