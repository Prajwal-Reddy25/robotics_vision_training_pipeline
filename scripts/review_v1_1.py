"""Check the actual release matrix, paired inputs, metrics, and artifact identity."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from robotics_vision.benchmarking import _sha256
from robotics_vision.data import SegmentationDataset, validate_dataset
from robotics_vision.shifts import ShiftedDataset
from robotics_vision.study import load_study, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--plan", default="configs/ablation_v1_1.yaml")
    args = parser.parse_args()
    study = json.loads(Path(args.study).read_text())
    spec, base = load_study(args.plan)
    assert study["resolved_study"]["spec"] == spec
    assert study["resolved_study"]["base"] == base
    expected = {(a, arm, seed) for a in spec["architectures"]
                for arm in spec["arms"] for seed in spec["seeds"]}
    actual = {(r["architecture"], r["arm"], r["seed"]) for r in study["records"]}
    assert actual == expected and len(study["records"]) == len(expected)
    assert study["complete"] and study["failed_runs"] == 0
    assert all(r["status"] == "success" for r in study["records"])
    assert study["summaries"] == summarize(study["records"])
    manifest = Path(base["data"]["manifest"])
    assert study["manifest_sha256"] == _sha256(manifest)
    validate_dataset(manifest, check_hashes=True)
    nominal = SegmentationDataset(manifest, "test")
    reference = study["records"][0]["domain"]
    assert len(reference["conditions"]) == 13
    for condition in reference["conditions"].values():
        dataset = ShiftedDataset(nominal, condition["kind"], condition["strength"],
                                 spec["shift_seed"])
        assert dataset.fingerprint() == condition["content_sha256"]
    for record in study["records"]:
        domain, benchmark = record["domain"], record["benchmark"]
        assert domain["source_ids"] == reference["source_ids"]
        assert domain["checkpoint_sha256"] == _sha256(Path(domain["checkpoint"]))
        assert benchmark["model_sha256"] == domain["checkpoint_sha256"]
        assert benchmark["measured_iterations"] == len(benchmark["latency_ms_samples"])
        assert benchmark["model_size_bytes"] == Path(benchmark["model_path"]).stat().st_size
        assert benchmark["mean_iou"] == domain["conditions"]["nominal"]["mean_iou"]
        nominal_iou = domain["conditions"]["nominal"]["mean_iou"]
        for name, condition in domain["conditions"].items():
            assert condition["content_sha256"] == reference["conditions"][name]["content_sha256"]
            assert condition["samples"] == len(nominal)
            matrix = condition["confusion_matrix"]
            assert sum(map(sum, matrix)) == len(nominal) * base["data"]["image_size"] ** 2
            ious = []
            for i in range(4):
                union = sum(matrix[i]) + sum(row[i] for row in matrix) - matrix[i][i]
                if union:
                    ious.append(matrix[i][i] / union)
            assert math.isclose(sum(ious) / len(ious), condition["mean_iou"], abs_tol=1e-12)
            assert math.isclose(nominal_iou - condition["mean_iou"],
                                condition["absolute_drop"], abs_tol=1e-12)
    print(json.dumps({"runs_verified": len(expected), "paired_conditions": 13,
                      "condition_evaluations": 13 * len(expected),
                      "summary_recomputed": True, "input_hashes_reproduced": True,
                      "metrics_recomputed_from_confusion": True,
                      "checkpoint_hashes_verified": True}, indent=2))


if __name__ == "__main__":
    main()
