# Robotics Vision Training Pipeline

Reproducible semantic-segmentation training, deployment and domain-shift analysis
with two small PyTorch models: **TinyUNet** and **LiteFCN**. The project records where
synthetic robustness fails to transfer to real photographs, including negative results.

[v1.2.0 release](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/releases/tag/v1.2.0)
· [Passing tag CI, checked 2026-10-20](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/actions/runs/39390109521)
· [Reproducibility guide](docs/REPRODUCIBILITY.md)

- **Data and experiments:** deterministic four-class warehouse scenes, hash-checked
  splits, six controlled synthetic shifts, seven augmentation arms and three seeds.
- **Training and deployment:** CPU/CUDA, mixed precision, full-state checkpoints/resume,
  TorchScript/ONNX export, ONNX Runtime and measured model-call latency/size.
- **Real-data evaluation:** 42 frozen models × 80 COCO photos × two input sizes;
  native ground truth, explicit partial label mappings, per-class metrics and failure panels.

The synthetic lifecycle is offline. Real-photo preparation is an optional download;
no pretrained weights are downloaded. TensorRT integration remains unvalidated.

## Key measured results

Baseline recipes, mean ± sample SD across seeds 42/43/44; no real-data tuning:

| Model | Synthetic ID mIoU, 4 classes | Synthetic shifted mIoU, 4 classes | Real mIoU, binary floor/nonfloor at 64² |
| --- | --- | --- | --- |
| TinyUNet | 0.6874 ± 0.1272 | 0.6528 ± 0.1136 | 0.1913 ± 0.0365 |
| LiteFCN | 0.6070 ± 0.0166 | 0.5933 ± 0.0240 | 0.2711 ± 0.0809 |

**The columns have different label spaces; do not subtract their mIoUs.** The
[sim-to-real analysis](docs/SIM_TO_REAL_ANALYSIS_V1_2.md) uses matching binary
projections for gaps and augmentation contrasts. Occlusion augmentation changes
synthetic shifted binary mIoU by +0.0683/−0.0063 (TinyUNet/LiteFCN), versus
−0.0091/+0.0089 on real data. Mean signs reverse, individual seed signs are mixed,
and baseline real floor IoU remains only 0.1271/0.1314.

v1.1 measured 42 training runs and 546 synthetic-condition evaluations. v1.2 reused
those models for 84 real dataset evaluations plus two exact engineering repeats.
It validates floor recognition and a restricted cabinet/storage-obstacle endpoint;
cones, mobile robots and physical warehouse deployment remain unvalidated.

## Prediction gallery

Existing diagnostic figures, baseline/occlusion seed 42 at 64². Green = floor,
red error pixels = false floor, blue = missed floor. Click for full-size panels.

<a href="docs/evidence/v1_2/gallery/206218.png"><img src="docs/evidence/v1_2/gallery/206218.png" width="900" alt="Relatively best case, with substantial remaining false-floor errors"></a>

**Relatively best, errors remain:** COCO 206218. Photo: [Mechatronics Guy](https://www.flickr.com/photos/_/7161410855/),
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), resized into a diagnostic panel.

<a href="docs/evidence/v1_2/gallery/554291.png"><img src="docs/evidence/v1_2/gallery/554291.png" width="900" alt="Blur and clutter failure with widespread false floor and missed floor"></a>

**Blur/clutter failure:** COCO 554291. Photo: [Eirik Newth](https://www.flickr.com/photos/_/256248429/),
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), resized into a diagnostic panel.
Ground-truth annotations: COCO Consortium, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Full success/failure gallery](docs/FAILURE_ANALYSIS_V1_2.md) ·
[Original attribution](docs/evidence/v1_2/gallery/ATTRIBUTION.md) ·
[2026-10-20 link review and two unresolved creators](docs/ATTRIBUTION_REVIEW.md).

## Check the published evidence offline

Maintenance addition in this checkout; it is not part of the original v1.2.0 tag.
Python 3.10+ and the tracked evidence are sufficient:

```bash
python -I -S scripts/verify_published_v1_2.py
python -I -S scripts/check_docs.py
```

The verifier independently recomputes recorded metrics, all seed summaries/paired
contrasts and correlations, checks CSV/gallery consistency and 63 inventoried files.
It neither downloads data nor loads checkpoints. It checks recorded arithmetic and
integrity; original-pixel validity, inference, hardware timing and measurement
authenticity require additional evidence. See [scope and full reproduction](docs/REPRODUCIBILITY.md).

## v1.2: real-photo transfer

Released tag `v1.2.0` points to `856e0574098a38dfaa1374cde1fcc3e0c923397b`;
package version is 1.2.0. Experiments were recorded on 2026-10-05/06; the tag was
created on October 6 and the GitHub Release published on October 20. The frozen
v1.1 baseline `d9a74a0` now has tag `v1.1.0`. Original dated audits are preserved.

Full native-image inference needs the real data and 42 locally generated checkpoints:

```bash
python scripts/prepare_real_v1_2.py
python scripts/evaluate_real_v1_2.py --output-dir runs/my_real_study
python scripts/review_real_v1_2.py --study runs/my_real_study/study.json
```

Use fresh output paths. [Regeneration instructions](docs/REAL_DATA_GENERALIZATION_V1_2.md#reproduction)
explain how to rebuild ignored data/checkpoints. No new architecture, runtime dependency
or experimental training was added in v1.2.

- [Dataset, mapping, measured arms and seeds](docs/REAL_DATA_GENERALIZATION_V1_2.md)
- [Research conclusions](docs/SIM_TO_REAL_ANALYSIS_V1_2.md), [all contrasts](docs/SIM_TO_REAL_TABLES_V1_2.md)
- [Performance](docs/BENCHMARKS_V1_2.md), [historical release validation](docs/RELEASE_READINESS_V1_2.md)
- [Release notes](docs/RELEASE_NOTES_V1_2.md), [prepared GitHub Release body](docs/GITHUB_RELEASE_V1_2.md)
- [Current maintenance review](docs/MAINTENANCE_REVIEW_2026_10_20.md)

## v1.1: generalization and domain shift

The existing models now have a controlled study of lighting, texture, blur, noise,
planar viewpoint and foreground occlusion at two severities. All conditions use the
same held-out scenes and corruption seeds. Seven augmentation arms across two
models and three training seeds produced 42 measured runs and 546 evaluations.
New training transforms default to zero; the original generator/models/default
augmentation behavior and existing commands are preserved.

```bash
rv-study --study configs/ablation_smoke.yaml --output-dir runs/my_shift_smoke
rv-study --study configs/ablation_v1_1.yaml --output-dir runs/my_ablation
python scripts/review_v1_1.py --study runs/my_ablation/study.json
```

Use fresh output paths. Runs retain resolved configurations, source/transformed
hashes, all per-seed metrics, raw latency samples, and explicit failure records.
The full study uses CPU FP32, 15 epochs, data seed 17, training seeds 42/43/44 and
corruption seed 2026. Best checkpoints are selected on nominal validation only.

TinyUNet's baseline nominal mIoU is **0.6874 ± 0.1272** across seeds; LiteFCN's is
**0.6070 ± 0.0166**. Adding occlusion augmentation changes mean shifted mIoU by
**+5.73 percentage points** for TinyUNet and **−2.12 points** for LiteFCN.
These synthetic results include negative experiments and do not establish
real-world robustness. The original single-seed results remain dated historical
evidence. Tags `v1.0.0` and `v1.1.0` now identify `54f3a2a` and `d9a74a0`.
Historical audits retain their original observations about tags unavailable at collection.

- [Domain-shift measurements](docs/DOMAIN_SHIFT_V1_1.md)
- [Augmentation ablations, including negative results](docs/AUGMENTATION_ABLATION_V1_1.md)
- [Validation and reproduction](docs/VALIDATION_V1_1.md)
- [CPU/GPU benchmarks](docs/BENCHMARKS_V1_1.md)
- [Engineering notes and owner competency checklist](docs/ENGINEERING_NOTES.md)
- [Limitations](docs/LIMITATIONS.md), [release review](docs/RELEASE_READINESS_V1_1.md)
- [Changelog](CHANGELOG.md), [roadmap](ROADMAP.md)

## Lifecycle

```mermaid
flowchart LR
    A[Seeded scene generator] --> B[Manifest + SHA-256 validation]
    B --> C[Disjoint train / val / test]
    C --> D[Paired augmentation]
    D --> E[PyTorch train + checkpoints]
    E --> F[Metrics + error analysis]
    E --> G[TorchScript / ONNX]
    G --> H[PyTorch / TorchScript / ONNX Runtime benchmark]
    F --> I[Measured model comparison]
    H --> I
```

Two export-friendly architectures are included:

- `tiny_unet`: skip-connected encoder/decoder with stronger spatial detail capacity.
- `lite_fcn`: depthwise-separable encoder with fewer operations and parameters.

The repository never ships invented scores. Commands below generate metrics from the current environment and write machine-readable JSON/CSV alongside provenance.

## Quick start

Python 3.10+ is required. Install the PyTorch build appropriate for your CPU or CUDA system first if the default package index is not suitable.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"

rv-prepare --config configs/tiny_unet.yaml
rv-train --config configs/tiny_unet.yaml
rv-evaluate \
  --config configs/tiny_unet.yaml \
  --checkpoint runs/tiny_unet/checkpoints/best.pt
rv-export \
  --config configs/tiny_unet.yaml \
  --checkpoint runs/tiny_unet/checkpoints/best.pt \
  --output-dir artifacts/tiny_unet
rv-benchmark \
  --config configs/tiny_unet.yaml \
  --model artifacts/tiny_unet/model.onnx \
  --backend onnxruntime \
  --output runs/tiny_unet/benchmark_onnxruntime.json
```

For a fast integration check, replace the config with `configs/smoke.yaml`. The smoke run checks functionality; its one-epoch metrics are not a quality claim.

## Compare both architectures

```bash
rv-train --config configs/tiny_unet.yaml
rv-evaluate --config configs/tiny_unet.yaml \
  --checkpoint runs/tiny_unet/checkpoints/best.pt
rv-benchmark --config configs/tiny_unet.yaml \
  --model runs/tiny_unet/checkpoints/best.pt --backend pytorch \
  --output runs/tiny_unet/benchmark_pytorch.json

rv-train --config configs/lite_fcn.yaml
rv-evaluate --config configs/lite_fcn.yaml \
  --checkpoint runs/lite_fcn/checkpoints/best.pt
rv-benchmark --config configs/lite_fcn.yaml \
  --model runs/lite_fcn/checkpoints/best.pt --backend pytorch \
  --output runs/lite_fcn/benchmark_pytorch.json

rv-compare \
  --result tiny_unet=runs/tiny_unet/benchmark_pytorch.json \
  --result lite_fcn=runs/lite_fcn/benchmark_pytorch.json \
  --output-prefix runs/model_comparison
```

The comparison command refuses fewer than two measured files and copies values without inventing missing data.

## Resume training

Checkpoints include model, optimizer, scheduler, AMP scaler, epoch, best score, resolved configuration, DataLoader shuffle state, and Python/NumPy/PyTorch/CUDA RNG state.

```bash
rv-train --config configs/tiny_unet.yaml \
  --resume runs/tiny_unet/checkpoints/last.pt
```

Plan the final epoch count before the initial run. Exact resume rejects changes to the epoch schedule, optimizer, batch size, seeds, augmentation, workers, AMP mode, architecture, input size, class count, or manifest. CPU tests compare interrupted and uninterrupted model weights bit for bit. Cross-device or cross-version equality is not promised. Checkpoints use Python pickle for full optimizer/RNG state and must come from a trusted source.

## Configuration

YAML files inherit validated defaults from `robotics_vision.config.DEFAULT_CONFIG`. Known values can be overridden without editing a file:

```bash
rv-train --config configs/tiny_unet.yaml \
  --set training.device=cpu \
  --set training.epochs=10 \
  --set experiment.name=unet_cpu_10e
```

Unknown keys, invalid devices, empty splits, unsupported architectures, and inconsistent class counts fail early. Every run stores `resolved_config.yaml`, `environment.json`, `data_validation.json`, line-delimited events, epoch CSV, and checkpoint metadata.

## Evaluation and error analysis

Evaluation reports:

- pixel accuracy;
- per-class and macro precision/recall;
- per-class IoU and mean IoU over classes present in the evaluated ground-truth/prediction union;
- the full ground-truth-row/prediction-column confusion matrix;
- histogram-approximated one-vs-rest pixel average precision and foreground pixel mAP;
- the lowest-IoU examples as `input | ground truth | prediction` PNG panels.

Instance-detection mAP is not applicable to this semantic-segmentation task. Pixel mAP is clearly named and includes its bin count to avoid conflating the two.

## Outputs

```text
data/processed/<dataset>/
  images/ masks/ manifest.json
runs/<experiment>/
  resolved_config.yaml environment.json data_validation.json
  events.jsonl metrics.csv training_summary.json
  checkpoints/{last,best,epoch_*.pt}
  evaluation/metrics.json
  evaluation/error_analysis/{summary.json,*.png}
artifacts/<experiment>/
  model.torchscript.pt model.onnx metadata.json
```

These paths are excluded from Git because datasets and model binaries do not belong in source history. v1.2 also ignores `data/external/`; bounded JSON/CSV diagnostics and licensed gallery figures are tracked under `docs/evidence/v1_2/`.

## Quality gates

```bash
pytest
ruff check .
```

Tests cover deterministic generation, ID/path/content leakage, checksum/label validation, deterministic paired augmentation, exact metric edge cases, both model shapes, bitwise CPU resume, dynamic TorchScript/ONNX equivalence, evaluation, benchmarking, and measured comparison output. GitHub Actions runs those checks plus the complete tiny lifecycle and both architectures on CPU.

Additional tests cover paired shifts, homography/occlusion label semantics,
augmentation bounds, legacy resume defaults, repeat-study equality, nonfinite
loss/logit rejection, failure preservation and atomic study ledgers. CPU CI also
runs a four-run domain-study smoke and uploads its evidence even on failure.

## Documentation

- [Dataset card](docs/DATASET_CARD.md)
- [Training and reproducibility](docs/TRAINING.md)
- [Model card](docs/MODEL_CARD.md)
- [Export and deployment](docs/DEPLOYMENT.md)
- [Portfolio requirement map](docs/PORTFOLIO_REVIEW.md)
- [Local verification record](docs/VERIFICATION.md)
- [Technical audit report](docs/AUDIT_REPORT.md)
- [NVIDIA GPU validation](docs/GPU_VALIDATION.md)

## License

Code is MIT licensed. Bundled synthetic scenes are CC0-1.0 as recorded in manifests. External COCO annotations are CC BY 4.0 and selected photographs have source-metadata CC BY 2.0; they are not covered by the code license. See the [real-data contract](docs/REAL_DATA_GENERALIZATION_V1_2.md) and [gallery attribution](docs/evidence/v1_2/gallery/ATTRIBUTION.md).
