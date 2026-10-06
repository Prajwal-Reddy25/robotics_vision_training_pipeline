"""Descriptive paired analyses; no test-set model selection or significance claims."""

from __future__ import annotations

import statistics
from collections import defaultdict

import numpy as np


def distribution(values: list[float]) -> dict:
    if not values or not np.isfinite(values).all():
        raise ValueError("Distribution requires nonempty finite observations")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "individual_values": values,
    }


def _ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or not np.isfinite(first + second).all():
        raise ValueError("Correlation requires paired finite observations")
    if len(first) < 3:
        return None
    a, b = _ranks(first), _ranks(second)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denominator) if denominator else None


def observed_values(record: dict) -> dict:
    real = record["measured"]
    binary, shared = real["binary"], real["shared"]
    return {
        "real_binary_miou": binary["mean_iou"],
        "real_floor_iou": binary["per_class"]["floor"]["iou"],
        "real_nonfloor_iou": binary["per_class"]["nonfloor"]["iou"],
        "real_pixel_accuracy": binary["pixel_accuracy"],
        "real_mean_precision": binary["mean_precision"],
        "real_mean_recall": binary["mean_recall"],
        "real_shared_miou": shared["mean_iou"],
        "real_shared_floor_iou": shared["per_class"]["floor"]["iou"],
        "real_storage_iou": shared["per_class"]["storage_obstacle"]["iou"],
        "id_binary_miou": record["id_binary"]["mean_iou"],
        "id_four_class_miou": record["id_four_class_miou"],
        "synthetic_shifted_binary_miou": statistics.fmean(
            c["mean_iou"] for name, c in record["projected_shifts"].items() if name != "nominal"
        ),
        "gap_binary_miou": record["delta_sim2real"]["binary_miou"],
        "gap_floor_iou": record["delta_sim2real"]["floor_iou"],
        "gap_shared_floor_iou": record["id_shared"]["per_class"]["floor"]["iou"]
        - shared["per_class"]["floor"]["iou"],
        "gap_storage_iou": record["delta_sim2real"]["storage_iou"],
        "boundary_error_rate": real["boundary_error_rate"],
        "interior_error_rate": real["interior_error_rate"],
        "latency_p50_ms": record["timing"]["p50_ms"],
        "latency_p95_ms": record["timing"]["p95_ms"],
        "preprocess_mean_ms": real["preprocess_ms_mean"],
    }


def summarize_real(study: dict) -> dict:
    records = [r for r in study["records"] if r["status"] == "success"]
    expected = {
        (a, arm, seed, size)
        for a in ("tiny_unet", "lite_fcn")
        for arm in ("no_aug", "baseline", "blur", "texture", "viewpoint", "occlusion", "combined")
        for seed in (42, 43, 44)
        for size in (64, 128)
    }
    actual = {(r["architecture"], r["arm"], r["seed"], r["size"]) for r in records}
    if actual != expected or len(records) != 84 or not study["complete"]:
        raise ValueError("Summary requires the entire successful predeclared matrix")
    groups = defaultdict(list)
    lookup = {}
    for record in records:
        groups[record["architecture"], record["arm"], record["size"]].append(record)
        lookup[record["architecture"], record["arm"], record["seed"], record["size"]] = record
    summaries, paired, correlations = [], [], []
    for (architecture, arm, size), group in sorted(groups.items()):
        group.sort(key=lambda r: r["seed"])
        measures = [observed_values(r) for r in group]
        summaries.append(
            {
                "architecture": architecture,
                "arm": arm,
                "size": size,
                "seeds": [r["seed"] for r in group],
                "metrics": {key: distribution([m[key] for m in measures]) for key in measures[0]},
                "parameters": group[0]["parameters"],
                "parameter_bytes": group[0]["parameter_bytes"],
                "checkpoint_bytes": [r["checkpoint_bytes"] for r in group],
                "shared_pixel_coverage": group[0]["measured"]["shared_pixel_coverage"],
            }
        )
        if arm != "baseline":
            differences = []
            for record in group:
                base = observed_values(lookup[architecture, "baseline", record["seed"], size])
                differences.append({k: v - base[k] for k, v in observed_values(record).items()})
            paired.append(
                {
                    "architecture": architecture,
                    "arm": arm,
                    "size": size,
                    "reference": "baseline",
                    "seeds": [r["seed"] for r in group],
                    "deltas": {
                        key: distribution([d[key] for d in differences]) for key in differences[0]
                    },
                }
            )
    for architecture in ("tiny_unet", "lite_fcn"):
        for size in (64, 128):
            group = [r for r in records if r["architecture"] == architecture and r["size"] == size]
            gaps = [r["delta_sim2real"]["binary_miou"] for r in group]
            real_scores = [r["measured"]["binary"]["mean_iou"] for r in group]
            for shift in sorted(group[0]["projected_shifts"]):
                if shift == "nominal":
                    continue
                scores = [r["projected_shifts"][shift]["mean_iou"] for r in group]
                drops = [r["id_binary"]["mean_iou"] - s for r, s in zip(group, scores, strict=True)]
                correlations.append(
                    {
                        "architecture": architecture,
                        "size": size,
                        "shift": shift,
                        "n": len(group),
                        "drop_gap_spearman": spearman(drops, gaps),
                        "shift_score_real_score_spearman": spearman(scores, real_scores),
                        "note": "Descriptive; shared ID term, correlated arms/seeds; not causal.",
                    }
                )
    return {
        "summaries": summaries,
        "paired_deltas": paired,
        "correlations": correlations,
        "failed_attempts": [r for r in study["records"] if r["status"] == "failed"],
        "interrupted_attempts": [r for r in study["records"] if r["status"] == "interrupted"],
        "aggregation": "Pixel-pooled dataset metrics per seed; mean and sample SD over 3 seeds.",
        "unsupported_class_gaps": {"safety_cone": None, "mobile_robot": None},
    }
