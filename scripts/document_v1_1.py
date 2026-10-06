"""Render measured tables and copy bounded, reviewable release evidence."""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path

from robotics_vision.benchmarking import _sha256
from robotics_vision.runlog import write_json


def distribution(value: dict) -> str:
    sd = value["sample_sd"]
    return f"{value['mean']:.4f} ± {sd:.4f}" if sd is not None else f"{value['mean']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--evidence-dir", default="docs/evidence/v1_1")
    args = parser.parse_args()
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    study = json.loads(Path(args.study).read_text())
    deployment = json.loads(Path(args.deployment).read_text())
    gpu = json.loads(Path(args.gpu).read_text())
    for source, name in ((args.study, "study.json"), (args.deployment, "deployment.json"),
                         (args.gpu, "gpu.json")):
        shutil.copyfile(source, evidence / name)

    text = """# Domain-shift analysis v1.1

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

"""
    for architecture in ("tiny_unet", "lite_fcn"):
        summary = next(s for s in study["summaries"]
                       if s["architecture"] == architecture and s["arm"] == "baseline")
        text += f"### {architecture}\n\n"
        text += ("| Condition | mIoU ± SD | Floor | Obstacle | Cone | Robot | "
                 "Drop pp | Relative drop % |\n")
        text += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        names = ["nominal"] + [f"{kind}_{level}"
                               for kind in study["resolved_study"]["conditions"]
                               for level in ("low", "high")]
        for name in names:
            values = summary["conditions"][name]
            classes = values["class_iou"]
            rel = statistics.fmean(
                r["domain"]["conditions"][name]["relative_drop"]
                for r in study["records"] if r["architecture"] == architecture
                and r["arm"] == "baseline"
            )
            cells = [f"{classes[c]['mean']:.4f}"
                     for c in ("floor", "obstacle", "safety_cone", "mobile_robot")]
            text += (f"| {name} | {distribution(values['mean_iou'])} | {' | '.join(cells)} | "
                     f"{100 * values['absolute_drop']['mean']:.2f} | {100 * rel:.2f} |\n")
        text += "\n"
    text += """## Interpretation and limits

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
rv-domain --config configs/domain_shift.yaml \\
  --checkpoint runs/my_v1_1_study/training/tiny_unet_baseline_s42/checkpoints/best.pt \\
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
"""
    Path("docs/DOMAIN_SHIFT_V1_1.md").write_text(text)

    text = """# Augmentation ablation v1.1

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
"""
    for summary in study["summaries"]:
        text += (f"| {summary['architecture']} | {summary['arm']} | "
                 f"{distribution(summary['conditions']['nominal']['mean_iou'])} | "
                 f"{distribution(summary['mean_shifted_iou'])} | "
                 f"{100 * summary['paired_nominal_delta']['mean']:+.2f} | "
                 f"{100 * summary['paired_shifted_delta']['mean']:+.2f} |\n")
    text += """
## All nominal observations

| Model | Arm | Seed 42 | Seed 43 | Seed 44 | Best epochs (zero based) |
| --- | --- | ---: | ---: | ---: | --- |
"""
    for summary in study["summaries"]:
        records = [r for r in study["records"]
                   if r["architecture"] == summary["architecture"] and r["arm"] == summary["arm"]]
        scores = " | ".join(
            f"{r['domain']['conditions']['nominal']['mean_iou']:.4f}" for r in records
        )
        epochs = ", ".join(str(r["training"]["best_epoch"]) for r in records)
        text += f"| {summary['architecture']} | {summary['arm']} | {scores} | {epochs} |\n"
    text += """
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
"""
    Path("docs/AUGMENTATION_ABLATION_V1_1.md").write_text(text)

    text = """# Benchmarks v1.1

Measured 2026-10-04; batch 1, 64×64 RGB FP32, one CPU thread, 20 warmups,
100 timed model calls. Times exclude image loading, corruption, preprocessing,
postprocessing and host/device input transfer. CUDA calls synchronize before and
after each measurement. No CPU affinity, clocks, thermals or power limits were
fixed. These are local microbenchmarks, not end-to-end robot latency.

CPU: AMD Ryzen 9 8940HX. GPU: NVIDIA GeForce RTX 5070 Laptop GPU, driver 595.84.
CPU PyTorch 2.14.0+cpu; GPU PyTorch 2.14.0+cu130. ORT uses CPUExecutionProvider.
Quality/size/latency results are linked by checkpoint and artifact SHA-256.

## Baseline seed-42 deployments

| Model | Device/backend | mIoU | Mean ms | p50 ms | p95 ms | File bytes | Artifact kind |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
"""
    for record in study["records"]:
        if record["arm"] == "baseline" and record["seed"] == 42:
            b = record["benchmark"]
            text += _benchmark_row(record["architecture"], b)
    for report in (deployment, gpu):
        for model in report["models"]:
            for benchmark in model["benchmarks"]:
                text += _benchmark_row(model["architecture"], benchmark)
    text += """
All 42 trained checkpoints have their own nominal-quality and eager latency
measurements, raw duration samples, checkpoint file size and parameter bytes in
[study.json](evidence/v1_1/study.json). Deployment/parity details are in
[deployment.json](evidence/v1_1/deployment.json), synchronized GPU measurements
in [gpu.json](evidence/v1_1/gpu.json).

TinyUNet has 29,948 parameters / 119,792 FP32 parameter bytes; LiteFCN has 2,804 /
11,216 bytes. Full checkpoint sizes include optimizer, best model and RNG state;
they are not deployment-model footprints. TorchScript/ONNX sizes include graph
and buffer overhead. Shapes/operators stay fixed across corruptions; timing is
per checkpoint. No condition-specific latency was measured.

## Numerical checks and limitations

Both baseline exports passed finite logits, allclose rtol=1e-4/atol=1e-5, and
≥99.9% argmax agreement at 1×3×64×64 and 2×3×65×67. Full test mIoU is identical
across fresh CPU eager, TorchScript and ORT. GPU eager/TorchScript agree on test
mIoU; GPU minus CPU mIoU is +0.00007217 for TinyUNet and 0 for LiteFCN, below the
declared 0.001 smoke-comparison threshold. Cross-device bitwise parity is not claimed.

CUDA AMP training is exercised, but the accuracy ablation and timing table are
FP32. This milestone does not add an ORT CUDA benchmark or validate TensorRT.
Neither speed nor export parity establishes perception accuracy on real sensors.

## Commands

```bash
python scripts/validate_v1_1.py --study runs/v1_1_ablation/study.json \\
  --output-dir runs/my_v1_1_deployment
python scripts/validate_v1_1.py --study runs/v1_1_ablation/study.json \\
  --output-dir runs/my_v1_1_gpu --gpu --deployment-dir runs/my_v1_1_deployment
```

Use the CPU environment for the first command and a compatible CUDA PyTorch
environment for the second. Output paths must be fresh. Strict GPU resume
failures are retained and cause nonzero exit even when remaining measurements finish.

## Owner competency checklist

- [ ] Explain CUDA synchronization and the excluded latency stages.
- [ ] Distinguish parameter bytes, training artifacts and deployment file sizes.
- [ ] Locate raw durations and calculate p50/p95 without dropping outliers.
- [ ] Explain numerical export parity separately from held-out segmentation quality.
"""
    Path("docs/BENCHMARKS_V1_1.md").write_text(text)
    hashes = {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
              for path in evidence.iterdir() if path.is_file() and path.name != "SHA256.json"}
    write_json(evidence / "SHA256.json", hashes)


def _benchmark_row(architecture: str, b: dict) -> str:
    return (f"| {architecture} | {b['device']}/{b['backend']} | {b['mean_iou']:.6f} | "
            f"{b['latency_ms_per_batch_mean']:.4f} | {b['latency_ms_per_batch_p50']:.4f} | "
            f"{b['latency_ms_per_batch_p95']:.4f} | {b['model_size_bytes']} | "
            f"{b['artifact_kind']} |\n")


if __name__ == "__main__":
    main()
