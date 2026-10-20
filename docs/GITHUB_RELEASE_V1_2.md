# v1.2.0 — Real-data generalization and sim-to-real evaluation

Small segmentation models can improve on synthetic stress tests while losing
accuracy on real photographs. This release freezes the existing TinyUNet/LiteFCN
study and evaluates that gap, retaining failed and neutral augmentation results.

## Experimental scope

- **42 frozen models:** two architectures × seven augmentation arms × seeds 42/43/44.
- **80 real COCO photographs**, native official panoptic ground truth; **84 dataset
  evaluations** at 64²/128² plus two exact engineering repeat evaluations.
- Primary binary floor/nonfloor scoring; secondary restricted floor/cabinet-storage
  scoring. No real-data tuning, new architecture or experimental retraining.

## Principal results

At 64², binary mIoU is mean ± sample SD over the three training seeds:

| Recipe | TinyUNet | LiteFCN |
| --- | --- | --- |
| Baseline | 0.1913 ± 0.0365 | 0.2711 ± 0.0809 |
| Occlusion augmentation | 0.1822 ± 0.0179 | 0.2799 ± 0.0820 |
| Combined augmentation | 0.1668 ± 0.0529 | 0.2862 ± 0.0867 |

Matched binary projections show the synthetic occlusion-effect signs reverse on
average on real data: TinyUNet −0.0091 and LiteFCN +0.0089. Individual signs are
mixed. Baseline real floor IoU is only 0.1271/0.1314. Disabling baseline augmentation
improves all three real-test seeds in both models. This is evidence of substantial
failed transfer, not a general augmentation or deployment success claim.

Synthetic ID four-class mIoU is 0.6874 ± 0.1272 / 0.6070 ± 0.0166 (TinyUNet/LiteFCN).
Those four-class scores must not be directly subtracted from binary real mIoU.
All seven arms, paired seed deltas, class scores, confusion matrices and timing
samples are retained in the linked evidence.

## Reproducibility and validation

The original experiment was recorded on 2026-10-05/06. The tag identifies commit
`856e0574098a38dfaa1374cde1fcc3e0c923397b`; the GitHub Release was published on 2026-10-20.
The source archive contains recorded counts, configurations, hashes, version captures
and attributed diagnostics. It does not include original photo caches or trained
checkpoints; regenerating inference requires the documented data/model procedure.

Original validation record: CPU 76 passed with one CUDA-only skip, CUDA 77 passed;
Ruff, packaging, CPU repeat/resume, TorchScript/ONNX/ORT parity and fresh GPU FP32/AMP
checks passed. Two independently repeated baseline configurations reproduced native
real metrics exactly. The [published-tag CI run](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/actions/runs/39390109521)
passed on 2026-10-20. These historical results are not new model experiments from the
later maintenance review.

## Limitations

COCO interiors are floor/cabinet-enriched and not representative warehouse scenes.
Safety cones and mobile robots lack compatible real ground truth. Restricted metrics
score 20.50% of valid pixels; scene-level confidence intervals and causal mechanisms
are not established. Three-seed dispersion measures optimization variability.

The prior CUDA bitwise-resume failure remains documented despite passing later
probes. TensorRT, long-horizon real-data training and physical camera/robot deployment
remain unvalidated. Latency is model-call-only. Two original-photo creators remain
unresolved; their photographs are omitted from tracked panels without excluding
numerical cases. Next planned research: compatible warehouse labels and separate
scene/session-grouped tuning and final-test splits.

## Evidence

- [Dataset, mappings and full reproduction](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/blob/v1.2.0/docs/REAL_DATA_GENERALIZATION_V1_2.md)
- [Sim-to-real conclusions](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/blob/v1.2.0/docs/SIM_TO_REAL_ANALYSIS_V1_2.md)
- [Raw recorded counts/timing](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/blob/v1.2.0/docs/evidence/v1_2/study.json)
- [All seeds/classes CSV](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/blob/v1.2.0/docs/evidence/v1_2/real_runs.csv)
- [Failure analysis](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/blob/v1.2.0/docs/FAILURE_ANALYSIS_V1_2.md)
- [Attribution](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/blob/v1.2.0/docs/evidence/v1_2/gallery/ATTRIBUTION.md)
- [Original validation record](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/blob/v1.2.0/docs/RELEASE_READINESS_V1_2.md)
- [Full changelog from v1.1.0](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/compare/v1.1.0...v1.2.0)
