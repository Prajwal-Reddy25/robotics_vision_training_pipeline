# Sim-to-real analysis v1.2

The controlled synthetic conclusions do **not** persist as a general augmentation
ranking on this independent real-photo subset. All 42 frozen configurations and
three seeds are retained; no real-data hyperparameter tuning occurred. This is a
partial-ontology finding, not a four-class warehouse validation. Contract/data:
[real-data report](REAL_DATA_GENERALIZATION_V1_2.md). Full contrasts and correlations:
[all tables](SIM_TO_REAL_TABLES_V1_2.md), [machine-readable analysis](evidence/v1_2/analysis.json).

## 1. Does synthetic augmentation improve real generalization?

Sometimes for these supported metrics, without a universal benefit. TinyUNet blur
adds binary mIoU **0.05939 ± 0.04293**, with paired changes +0.07412/+0.01104/+0.09301.
Its occlusion arm changes **−0.00905 ± 0.05168**, and combined **−0.02448 ± 0.04679**.
LiteFCN blur, occlusion and combined means improve by 0.01750/0.00888/0.01514,
but each contains at least one negative seed. This is not significant-evidence
language: n=3 optimization seeds on one selected dataset, no scene confidence interval.

Disabling all baseline augmentation (`no_aug`) improves real binary mIoU for **all
three seeds in both architectures**, by 0.08218 ± 0.03072 (TinyUNet) and
0.02877 ± 0.01230 (LiteFCN). This retained control prevents portraying augmentation
as inherently beneficial. The baseline already includes geometric/color/noise
augmentation; new arms add one or all v1.1 transforms. These are comparisons of
frozen recipes, not a search for a winning real-data recipe.

## 2. Does the architecture-dependent v1.1 result persist?

The **mean signs reverse**, while individual-seed signs are mixed. Original v1.1
four-class shifted averages are also reprojected to the same binary floor metric
so ontology changes do not manufacture the conclusion:

| Added arm | TinyUNet synthetic binary shifted change | TinyUNet real binary change | LiteFCN synthetic binary shifted change | LiteFCN real binary change |
| --- | --- | --- | --- | --- |
| Occlusion | +0.06829 | −0.00905 | −0.00626 | +0.00888 |
| Combined | +0.06252 | −0.02448 | −0.02105 | +0.01514 |

Synthetic values are algebraic projections of the frozen 12 condition matrices,
not newly executed inference. Original four-class changes (+0.05728/−0.02115 for
occlusion and +0.04708/−0.04565 for combined) remain in v1.1 records unchanged.
No claim is made that one architecture intrinsically responds better to augmentation
on real robotics scenes. The real dataset, label prevalence and visual distribution
are materially different; the finding is that the synthetic ordering fails to transfer.

## 3. Which synthetic shifts correlate with real degradation?

Within each architecture, use all 21 arm/seed runs at primary64². Correlate projected
synthetic drop D_k = binary_ID − binary_shift,k with G = binary_ID − binary_real.
TinyUNet's largest positive association is high blur (Spearman ρ=0.8532), followed
by low blur (0.8299). LiteFCN's largest positive value is low noise (0.2156), weak
and descriptive; its largest absolute association is low blur (−0.5026), opposite
in sign. See all 12 shifts at both sizes, including negative values, in the tables.

**Shared-ID confound:** both drops include the same ID score. TinyUNet high-blur
raw shifted-score vs real-score association is **−0.3117**, despite its positive
drop-gap association. LiteFCN low-blur score association is −0.2455. Correlated
augmentation arms and seeds are not21 independent experiments. There is no validated
universal synthetic surrogate for real robustness, no causality test, and no shift
is selected as a predictor from this final test. Luminance/gradient and qualitative
tags are descriptive input proxies, not calibrated causes.

## 4. Which classes have the largest gap?

Δ_sim2real,c = projected ID IoU_c − real IoU_c, at matched64² input.
Primary binary floor gaps are **0.7638 TinyUNet / 0.7578 LiteFCN**, with real
floor IoU 0.1271 / 0.1314. This endpoint penalizes false floor on all known categories.
For a fair **class ranking**, compare floor and storage on the same restricted mask:

| Model | Restricted floor IoU | Restricted floor gap | Storage IoU | Storage gap |
| --- | --- | --- | --- | --- |
| TinyUNet | 0.5166 | 0.3751 | 0.0275 | 0.7252 |
| LiteFCN | 0.4552 | 0.4383 | 0.0193 | 0.6280 |

Storage has the larger gap among the two supported classes on this shared mask.
Do not rank a primary floor gap against a restricted storage gap: evaluation coverage
and false-positive eligibility differ. The restricted endpoint ignores non-floor/
non-cabinet ground truth; its larger floor score is a mask effect, not an improvement
in the model. Both endpoint definitions and raw supports are retained.

Cones/robots have **no real IoU/gap**, because compatible labels do not exist here.
Cabinet labels cannot establish general warehouse obstacle recall. Large false-floor
rates on walls, furniture and people suggest broad semantic collapse rather than
only boundary artifacts; see [failure evidence](FAILURE_ANALYSIS_V1_2.md).

## 5. Does better OOD performance reduce nominal accuracy?

It can in these measured mean contrasts. LiteFCN combined improves real binary
mIoU by +0.01514 while its original nominal four-class mIoU falls −0.05706.
Its real paired changes are +0.05246/−0.02408/+0.01705: the gain is not uniform.
LiteFCN occlusion gains +0.00888 real with −0.00229 nominal on average, also mixed.
TinyUNet occlusion/combined improve nominal four-class scores by +0.06116/+0.04908
while real binary scores decline. This shows ID gains do not imply transfer gains;
it does not establish a general accuracy–robustness law or causal tradeoff.

At 128², occlusion mean real changes retain their signs (TinyUNet −0.0071,
LiteFCN +0.0158). Neither baseline is consistently rescued by higher resolution.
128² gap values retain the original64² ID reference; only the 64² gaps are a matched
input-resolution comparison. No better-performing input size is chosen for release.

## Demonstrated, hypothesized and missing

**Demonstrated:** native real-photo metrics for all frozen runs, substantial gaps,
mean augmentation-order reversal, retained negative/neutral seeds, resolution
sensitivity, exact two-model repeat metrics and real-input backend parity.
**Hypotheses:** synthetic palette shortcuts, BatchNorm statistics and simplified
object/viewpoint structure contribute to poor transfer. No feature attribution,
normalization intervention or new training experiment isolates these mechanisms.
**Missing:** representative warehouse scenes/session splits, all four classes,
scene-level confidence intervals, calibrated sensors, hardware perception and
real-data-trained/domain-adapted models. This real test set must not become a future
tuning set without explicitly retiring it and freezing a new final test set.

## Owner competency checklist

- [ ] Recompute matched-metric synthetic and real contrasts from saved confusion matrices.
- [ ] Explain why a mean-sign reversal with three seeds is limited evidence.
- [ ] Explain the shared-ID correlation confound and unsupported-class nulls.
- [ ] Describe leakage-safe validation/test separation for a future adaptation study.
