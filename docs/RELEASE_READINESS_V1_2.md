# Release readiness v1.2

Recommend **v1.2.0 for partial-ontology real-photo evaluation and qualified sim-to-real
analysis**. The evidence demonstrates poor transfer and mixed augmentation effects;
it does not establish robust four-class warehouse perception. No tag or publication
is created automatically. Frozen baseline: `d9a74a0bebc48b0facd049f24cb2889bf3bbda4e`
(no local v1.1 tag). Primary experiment code: `e932b267c9c918ea6cd0c71c12fdefcb4a4fd351`,
clean at study start. Latest tested analysis/verifier/package code: `1b86015`,
including frozen-input identity guard `d3b7c20`; subsequent changes are
only documentation/evidence and prose rendering. Exact final commit and
clean worktree are reported after the final documentation commit; verify with
`git rev-parse HEAD` and `git status --porcelain`.

## Acceptance gates

| Gate | Actual outcome |
| --- | --- |
| Preserve frozen source/results/negative experiments | passed: all 21 historical evidence hashes unchanged, all 42 checkpoint hashes and 546 synthetic confusion records verified; no source-model/training/export/CI redesign |
| License/ground truth/compatible mapping | passed for scoped endpoints: official pinned COCO metadata and 80 original real photographs; license4 filter, RGB segment decoding/metadata area checks, floor and restricted cabinet mapping, void/crowd excluded; unsupported cone/robot gaps null |
| Selection before prediction / no final-test tuning | passed: committed plan and source before experiment, metadata-only ordered selection; all frozen recipes/seeds reused without tuning; no real validation split invented |
| Complete fair comparison matrix | passed: 2 models × 7 arms × 3 seeds × 2 input sizes = 84 successful dataset evaluations; every model sees identical 80 photographs; all losses/neutral seeds retained |
| Metric/provenance arithmetic | passed: native per-image/aggregate confusion recomputation, projected ID metrics, matched 64² gaps, full inclusion/exclusion ledger; all data/checkpoint hashes validated |
| Statistical robustness | passed within declared scope: individual seeds, paired deltas, mean/sample SD; descriptive correlations at 21 runs per architecture; no significance/scene-CI claim |
| Failure evidence | passed: class/precision/recall/false-floor/boundary analysis, 11 deterministically selected panels; no successful-only gallery; input tags and attribution/ID corrections explicit |
| CPU unit/integration tests | 76 passed, 1 CUDA-only skip |
| CUDA suite | 77 passed, actual CUDA access |
| Lint/dependencies/package | Ruff passed; both environments `pip check` pass; wheel 1.2.0 built, installed into an isolated target and imports there |
| Export/backend checks | fresh TS/ONNX/ORT conversion/synthetic checks; 8 real-input TS/ORT comparisons at 64²/128² satisfy max absolute error≤5e-4 and argmax agreement≥0.999 |
| CPU reproducibility/resume | preserved unit bitwise resume; 2 full 15-epoch repeats reproduce last and validation-selected best weights; 2 native real dataset repeats reproduce every metric exactly |
| GPU/AMP/resume/performance | 4 finite FP32/AMP smoke trains + 4 strict resume probes pass; 4 synchronized eager/TS benchmarks collected; historical bitwise failure remains a limitation |
| Reproducible evidence and hygiene | compact lossless raw/aggregate JSON, per-run CSV, hashes, versions and command logs retained; generated raw images/checkpoints/exports/caches ignored; intentional bounded gallery/evidence only |

[Machine-readable engineering record](evidence/v1_2/engineering_validation.json)
links commands, environment, seeds, criteria and actual logs. Dataset/benchmark
records identify their executed source commits. Some later engineering checks ran
while documentation was being edited (`git_dirty=true` is preserved); packaged
model/training code is unchanged from the frozen baseline and the primary study
started clean. Do not erase those provenance flags to imply an entirely clean run.

## Counts and quantitative outcome

Zero new experimental training runs; 42 released-study runs reused. Separate functional
validation: 2 complete CPU repeats, 4 CUDA smoke trainings and 4 CUDA resume invocations
(10 release-check invocations; pytest's temporary training fixtures are not research runs).

84 main real dataset evaluations / 6,720 native photo-model predictions;2 additional
real reproducibility dataset evaluations (86 completed real dataset evaluations total).
8 real-input export parity comparisons. Prior 546 synthetic conditions are reprojected
algebraically, not passed off as newly executed inference. Fresh backend benchmark
quality checks use original synthetic test data. No new physics simulation or physical
camera/robot trial was performed; the synthetic stressors are raster transformations.

Baseline 64² real binary mIoU: TinyUNet 0.1913 ± 0.0365, LiteFCN 0.2711 ± 0.0809.
Baseline floor IoU 0.1271/0.1314 and floor gaps 0.7638/0.7578 reveal large transfer failures.
Occlusion real paired means −0.00905/+0.00888; combined −0.02448/+0.01514, with mixed
seed signs. See [all quantitative results](REAL_DATA_GENERALIZATION_V1_2.md),
[research conclusions](SIM_TO_REAL_ANALYSIS_V1_2.md),
[performance](BENCHMARKS_V1_2.md) and [failure examples](FAILURE_ANALYSIS_V1_2.md).

## Failures and unresolved limits

- **Quality failures retained:** poor baseline transfer, TinyUNet occlusion/combined
  mean losses, texture/viewpoint losses and every neutral/negative individual seed.
  No model evaluation crashed in the 84-run collection.
- **Operational failures retained:** original worker continued across a server restart;
  attempted continuation rejected the already-complete ledger. One thumbnail ID typo
  and an initially unsuitable Flickr lookup URL were corrected. Two creator lookups
  remain unresolved; those photos stay local, with diagnostic maps/results retained.
  [Operational record](evidence/v1_2/operational_failures.json).
- **Historical CUDA limitation:** a v1.1 LiteFCN FP32 resume differed by up to 1.788e-7.
  Four passing fresh probes do not prove general bitwise determinism; nondeterministic
  reduction warnings and prior failed raw records remain visible.
- **Scope limits:** floor/cabinet-enriched COCO interiors, partial label compatibility,
  restricted 20.50% scoring coverage, pixel-pooling prevalence, no scene confidence
  interval, shared-ID correlation confound and no mechanism-isolating intervention.
  At 128², the ID reference remains 64², so matched gap interpretation uses 64² only.
- **Not performed:** all 42 full training repeats, representative industrial/full-class
  real validation, TensorRT execution, hosted GitHub Actions and physical robot/camera
  experiments. Internet/HTTPS-range availability remains a dataset preparation dependency.
  External generation is not fully transactional; ledger continuation has no worker lock.
  Latency excludes full preprocessing/transfers/native postprocessing and uncontrolled
  power/affinity/thermal/load effects.

## Environment and reproducibility review

Python 3.12.3; CPU PyTorch 2.14.0+cpu; GPU PyTorch 2.14.0+cu130, CUDA 13.0/cuDNN92400;
NumPy 2.5.3, Pillow 12.3.0, ONNX 1.23.0, ONNX Runtime 1.30.0. AMD Ryzen 9 8940HX,
32 logical CPUs (one thread for the experiment); NVIDIA RTX 5070 Laptop GPU.
Exact captured [CPU versions](evidence/v1_2/cpu_versions.txt) and
[CUDA versions](evidence/v1_2/cuda_versions.txt). Training seeds 42/43/44, procedural
data 17, synthetic shifts 2026, external selection hash prefix 2026. GPU functionality
uses smoke data seed 7/training seed 9, base_channels 4, two epochs; it is not a new
long-horizon real-data training experiment.

Original data/checkpoints/deployment files are local and ignored. Regeneration
commands and file-hash-vs-weight-identity limits are documented in the real-data
report. Independent repeat evidence covers both baseline configurations, not all 42.
Inference and metric reproduction are exact for those CPU repeats; cross-platform
or version equality is not promised. Reproduction requires the external dataset
endpoints and the documented environment, not a fabricated model download.

## Owner competency checklist

- [ ] Explain every gate, identify the exact source/data/seed and reproduce a saved count.
- [ ] State why a rigorous negative generalization finding is release-worthy within scope.
- [ ] Explain why this release cannot be sold as warehouse/robot deployment validation.
- [ ] Describe the v1.3 warehouse grouped-split experiment without leaking this final test.
