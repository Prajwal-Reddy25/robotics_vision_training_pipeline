"""Render all measured arms/seeds and bounded release evidence without inventing scores."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from robotics_vision.benchmarking import _sha256
from robotics_vision.real_analysis import observed_values, summarize_real
from robotics_vision.runlog import write_json


def fmt(distribution):
    return f"{distribution['mean']:.4f} ± {distribution['sample_sd']:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/v1_2_real/study.json")
    parser.add_argument("--manifest", default="data/external/coco_v1_2/manifest.json")
    parser.add_argument("--deployment", default="runs/v1_2_deployment/validation.json")
    parser.add_argument("--gpu", default="runs/v1_2_gpu/validation.json")
    parser.add_argument("--reproducibility", default="runs/v1_2_real/reproducibility.json")
    parser.add_argument("--gallery", default="runs/v1_2_real/gallery_release")
    parser.add_argument("--evidence", default="docs/evidence/v1_2")
    args = parser.parse_args()
    study = json.loads(Path(args.study).read_text())
    analysis = summarize_real(study)
    evidence = Path(args.evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    copied = {}
    for path, name in [
        (args.study, "study.json"),
        (args.manifest, "manifest.json"),
        (args.deployment, "deployment.json"),
        (args.gpu, "gpu.json"),
        (args.reproducibility, "reproducibility.json"),
    ]:
        value = json.loads(Path(path).read_text())
        (evidence / name).write_text(
            json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"
        )
        copied[name] = {
            "original_path": path,
            "original_sha256": _sha256(Path(path)),
            "note": "Lossless compact JSON; whitespace differs from original.",
        }
    write_json(evidence / "analysis.json", analysis)
    with (evidence / "real_runs.csv").open("w", newline="") as stream:
        writer = None
        for r in study["records"]:
            if r["status"] != "success":
                continue
            row = {k: r[k] for k in ("architecture", "arm", "seed", "size")}
            row.update(observed_values(r))
            for endpoint in ("binary", "shared"):
                for name, metrics in r["measured"][endpoint]["per_class"].items():
                    for key in ("iou", "precision", "recall", "support_pixels"):
                        row[f"{endpoint}_{name}_{key}"] = metrics[key]
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
                writer.writeheader()
            writer.writerow(row)
    for source in Path(args.gallery).iterdir():
        target = evidence / "gallery" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    write_json(evidence / "provenance.json", copied)
    table = (
        "\n| Model | Arm | Input | Real binary mIoU | Floor IoU | Storage IoU* | ID−real binary |\n"
    )
    table += "| --- | --- | --- | --- | --- | --- | --- |\n"
    for group in analysis["summaries"]:
        m = group["metrics"]
        table += (
            f"| {group['architecture']} | {group['arm']} | {group['size']} | "
            f"{fmt(m['real_binary_miou'])} | {fmt(m['real_floor_iou'])} | "
            f"{fmt(m['real_storage_iou'])} | {fmt(m['gap_binary_miou'])} |\n"
        )
    text = """# Real-data generalization v1.2

Measured 2026-10-05/06. This is a **partial-ontology real-photo evaluation** of
all 42 frozen v1.1 checkpoints, not a full four-class warehouse deployment claim.
The [predeclared plan](PLAN_V1_2.md) predates predictions; the [audit](AUDIT_V1_2.md)
identifies exact baseline `d9a74a0bebc48b0facd049f24cb2889bf3bbda4e` despite the
missing local v1.1 tag. No architecture or training configuration was added/tuned.

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
"""
    text += table
    text += """
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
"""
    Path("docs/REAL_DATA_GENERALIZATION_V1_2.md").write_text(text)
    paired = "# Sim-to-real measured contrasts — complete tables\n\n"
    paired += "Augmented − baseline, mean ± sample SD; deltas ordered seeds 42/43/44.\n\n"
    paired += "| Model | Arm | Input | Real binary delta | Seed deltas | Four-class ID delta |\n"
    paired += "| --- | --- | --- | --- | --- | --- |\n"
    for group in analysis["paired_deltas"]:
        delta = group["deltas"]["real_binary_miou"]
        seeds = ", ".join(f"{v:+.4f}" for v in delta["individual_values"])
        paired += (
            f"| {group['architecture']} | {group['arm']} | {group['size']} | "
            f"{fmt(delta)} | {seeds} | {fmt(group['deltas']['id_four_class_miou'])} |\n"
        )
    paired += "\n## Descriptive correlations (all shifts)\n\n"
    paired += "| Model | Input | Shift | Drop ↔ gap ρ | Shift score ↔ real score ρ |\n"
    paired += "| --- | --- | --- | --- | --- |\n"
    for group in analysis["correlations"]:
        paired += (
            f"| {group['architecture']} | {group['size']} | {group['shift']} | "
            f"{group['drop_gap_spearman']:.4f} | "
            f"{group['shift_score_real_score_spearman']:.4f} |\n"
        )
    Path("docs/SIM_TO_REAL_TABLES_V1_2.md").write_text(paired)
    benchmarks = "# Performance measurements v1.2\n\n"
    benchmarks += "CPU FP32, batch1, one thread, 20 warmups/100 completed model calls per run.\n"
    benchmarks += (
        "Baseline three-seed mean ± sample SD of each run's p50/p95, not pooled quantiles.\n\n"
    )
    benchmarks += (
        "| Model | Input | p50 ms | p95 ms | Parameters | Parameter bytes | Checkpoint bytes |\n"
    )
    benchmarks += "| --- | --- | --- | --- | --- | --- | --- |\n"
    for group in analysis["summaries"]:
        if group["arm"] != "baseline":
            continue
        m = group["metrics"]
        benchmarks += (
            f"| {group['architecture']} | {group['size']} | "
            f"{fmt(m['latency_p50_ms'])} | {fmt(m['latency_p95_ms'])} | "
            f"{group['parameters']} | {group['parameter_bytes']} | "
            f"{min(group['checkpoint_bytes'])}–{max(group['checkpoint_bytes'])} |\n"
        )
    benchmarks += """
Raw samples for all 84 model/size measurements are in `evidence/v1_2/study.json`.
Model timing excludes preprocessing, transfers and native postprocessing.
RGB resizing/letterbox/tensor conversion mean times are recorded separately in the
CSV/raw records; they are not end-to-end robot latency. CPU affinity, power/clocks
and competing load are uncontrolled. The detached worker completed across a server
restart; timing remains an observed environment result, not an isolated speed guarantee.

Fresh synchronized GPU eager/TorchScript benchmark samples and synthetic accuracy
are in `evidence/v1_2/gpu.json`; fresh CPU TS/ORT results and deployment artifact
sizes are in `deployment.json`. These are the original64² synthetic benchmark
inputs, not native real-photo pipeline/GPU OOD quality benchmarks. TensorRT remains
optional and unavailable, and no physical camera/robot data were collected.
"""
    Path("docs/BENCHMARKS_V1_2.md").write_text(benchmarks)
    credits = json.loads((Path(args.gallery) / "credits.json").read_text())
    attribution = "# Diagnostic gallery attribution\n\n"
    attribution += (
        "COCO Consortium ground-truth annotations: [CC BY 4.0]"
        "(https://creativecommons.org/licenses/by/4.0/). "
        "[COCO terms](https://cocodataset.org/#termsofuse).\n\n"
    )
    attribution += "Photographs resized into diagnostic panels; source metadata license4 "
    attribution += "= [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).\n\n"
    for credit in credits:
        author = credit.get("author_name")
        attribution += f"- COCO {credit['coco_id']}: "
        if author:
            attribution += f"[{author}]({credit['author_url']}), "
            attribution += f"[{credit.get('title') or 'source photograph'}]({credit['source']}).\n"
        else:
            attribution += (
                f"[Original source]({credit['source']}); creator lookup unavailable. "
                "Photograph omitted from tracked panel; all ground truth/predictions "
                "and numerical results retained. Original remains local.\n"
            )
    (evidence / "gallery" / "ATTRIBUTION.md").write_text(attribution)
    print(
        {
            "evidence": str(evidence),
            "groups": len(analysis["summaries"]),
            "individual_rows": 84,
            "gallery_panels": len(credits),
        }
    )


if __name__ == "__main__":
    main()
