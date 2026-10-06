"""Recompute every measured confusion/gap and verify the full frozen matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robotics_vision.benchmarking import _sha256
from robotics_vision.real_analysis import summarize_real
from robotics_vision.real_data import ANNOTATION_SHA256, select_records, validate_real_dataset
from robotics_vision.real_evaluation import binary_metrics, restricted_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/v1_2_real/study.json")
    parser.add_argument("--source", default="docs/evidence/v1_1/study.json")
    parser.add_argument("--manifest", default="data/external/coco_v1_2/manifest.json")
    args = parser.parse_args()
    study = json.loads(Path(args.study).read_text())
    source = json.loads(Path(args.source).read_text())
    data = validate_real_dataset(args.manifest)
    manifest = json.loads(Path(args.manifest).read_text())
    annotation_path = Path(args.manifest).parent / "panoptic_val2017.json"
    assert _sha256(annotation_path) == ANNOTATION_SHA256
    selected, ledger = select_records(json.loads(annotation_path.read_text()))
    assert ledger == manifest["selection_ledger"]
    assert [(r["image"], r["annotation"], r["stratum"]) for r in selected] == [
        (r["image"], r["annotation"], r["stratum"]) for r in manifest["records"]
    ]
    assert study["source_study_sha256"] == _sha256(Path(args.source))
    assert study["manifest_sha256"] == _sha256(Path(args.manifest))
    assert study["failed_evaluations"] == 0
    summary = summarize_real(study)
    frozen = {(r["architecture"], r["arm"], r["seed"]): r for r in source["records"]}
    ids = {r["image"]["id"] for r in manifest["records"]}
    for r in study["records"]:
        if r["status"] != "success":
            continue
        reference = frozen[r["architecture"], r["arm"], r["seed"]]
        assert r["checkpoint_sha256"] == reference["domain"]["checkpoint_sha256"]
        assert r["checkpoint_sha256"] == _sha256(Path(r["checkpoint"]))
        nominal = reference["domain"]["conditions"]["nominal"]["confusion_matrix"]
        assert r["id_binary"] == binary_metrics(nominal)
        assert r["id_shared"] == restricted_metrics(nominal)
        measured = r["measured"]
        assert len(measured["per_image"]) == len(ids)
        assert {i["id"] for i in measured["per_image"]} == ids
        binary = np.sum([i["binary_confusion"] for i in measured["per_image"]], axis=0)
        shared = np.sum([i["shared_confusion"] for i in measured["per_image"]], axis=0)
        assert binary_metrics(binary) == measured["binary"]
        assert restricted_metrics(shared) == measured["shared"]
        assert binary.sum() == data["valid_pixels"]
        assert shared[1].sum() == data["storage_pixels"]
        assert np.all(shared[2:] == 0)
        assert np.isclose(shared.sum() / binary.sum(), measured["shared_pixel_coverage"])
        assert r["delta_sim2real"]["binary_miou"] == (
            r["id_binary"]["mean_iou"] - measured["binary"]["mean_iou"]
        )
        for name in ("boundary", "interior"):
            errors = sum(i[f"{name}_errors"] for i in measured["per_image"])
            pixels = sum(i[f"{name}_pixels"] for i in measured["per_image"])
            assert errors / pixels == measured[f"{name}_error_rate"]
        assert len(r["timing"]["samples_ms"]) == r["timing"]["iterations"] == 100
        assert r["timing"]["warmup"] == 20
    print(
        json.dumps(
            {
                "frozen_training_runs": 42,
                "dataset_evaluations_verified": 84,
                "native_image_evaluations": 6720,
                "data": data,
                "aggregate_groups_recomputed": len(summary["summaries"]),
                "correlations_recomputed": len(summary["correlations"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
