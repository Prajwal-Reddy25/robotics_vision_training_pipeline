"""Check the shipped v1.2 evidence offline using only Python's standard library.

This checks recorded evidence, not original pixels, checkpoint bytes or inference.
No package imports, network calls, training, pickle loading or artifact writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ARCHITECTURES = ("tiny_unet", "lite_fcn")
ARMS = ("no_aug", "baseline", "blur", "texture", "viewpoint", "occlusion", "combined")
SEEDS, SIZES = (42, 43, 44), (64, 128)
CLASSES = ("floor", "obstacle", "safety_cone", "mobile_robot")
MAX_FILE_BYTES, MAX_TOTAL_BYTES, MAX_FILES = 8 * 1024**2, 16 * 1024**2, 100


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def read_json(path):
    require(path.stat().st_size <= MAX_FILE_BYTES, f"{path.name}: input too large")

    def nonfinite(value):
        raise VerificationError(f"{path.name}: nonfinite JSON constant {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=nonfinite)


def compare(expected, actual, label):
    """Compare numeric schema fields; allow unrelated explanatory metadata."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{label}: expected object")
        for key, value in expected.items():
            require(key in actual, f"{label}: missing {key}")
            compare(value, actual[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), f"{label}: list length")
        for i, (a, b) in enumerate(zip(expected, actual, strict=True)):
            compare(a, b, f"{label}[{i}]")
    elif isinstance(expected, float):
        require(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12),
            f"{label}: {actual!r} != {expected!r}",
        )
    else:
        require(type(expected) is type(actual) and expected == actual, f"{label}: value mismatch")


def verify_inventory(root):
    inventory = read_json(root / "SHA256.json")
    require(0 < len(inventory) <= MAX_FILES, "inventory: file count exceeds bound")
    total = 0
    for name, item in inventory.items():
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "inventory: unsafe relative path",
        )
        path = root / relative
        require(
            not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            "inventory: symlink/path escape",
        )
        require(path.is_file(), f"inventory: missing {name}")
        size = path.stat().st_size
        total += size
        require(size <= MAX_FILE_BYTES and total <= MAX_TOTAL_BYTES, "inventory: byte bound")
        require(size == item["bytes"], f"inventory: size mismatch: {name}")
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
            f"inventory: SHA256 mismatch: {name}",
        )
    actual = set()
    for path in root.rglob("*"):
        if path.is_file() and path != root / "SHA256.json":
            actual.add(str(path.relative_to(root)))
            require(len(actual) <= MAX_FILES, "inventory: unexpected file count")
    require(actual == set(inventory), "inventory: unlisted files")
    return len(inventory)


def matrix(value, size):
    require(isinstance(value, list) and len(value) == size, "confusion: row count")
    require(
        all(isinstance(row, list) and len(row) == size for row in value), "confusion: column count"
    )
    require(
        all(type(n) is int and 0 <= n <= 10**9 for row in value for n in row),
        "confusion: counts must be bounded nonnegative integers",
    )
    return value


def metrics(value, names):
    n = len(value)
    matrix(value, n)
    rows = [sum(row) for row in value]
    columns = [sum(value[i][j] for i in range(n)) for j in range(n)]
    stats, active = {}, []
    for i, name in enumerate(names):
        tp, union = value[i][i], rows[i] + columns[i] - value[i][i]
        stats[name] = {
            "iou": tp / union if union else 0.0,
            "precision": tp / columns[i] if columns[i] else 0.0,
            "recall": tp / rows[i] if rows[i] else 0.0,
            "support_pixels": rows[i],
        }
        if union:
            active.append(stats[name]["iou"])
    require(sum(rows) > 0 and active, "confusion: empty scoring mask")
    return {
        "mean_iou": statistics.fmean(active),
        "per_class": stats,
        "pixel_accuracy": sum(value[i][i] for i in range(n)) / sum(rows),
        "mean_precision": statistics.fmean(s["precision"] for s in stats.values()),
        "mean_recall": statistics.fmean(s["recall"] for s in stats.values()),
        "evaluated_iou_classes": len(active),
        "confusion_matrix": value,
    }


def binary(value):
    matrix(value, 4)
    return metrics(
        [
            [value[0][0], sum(value[0][1:])],
            [sum(row[0] for row in value[1:]), sum(sum(row[1:]) for row in value[1:])],
        ],
        ("floor", "nonfloor"),
    )


def shared(value):
    matrix(value, 4)
    clipped = [value[0][:], value[1][:], [0] * 4, [0] * 4]
    return metrics(clipped, ("floor", "storage_obstacle"))


def summed(matrices, size):
    return [[sum(m[i][j] for m in matrices) for j in range(size)] for i in range(size)]


def distribution(values):
    return {
        "n": len(values),
        "individual_values": values,
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def spearman(a, b):
    def ranks(values):
        ordered = sorted(range(len(values)), key=values.__getitem__)
        result = [0.0] * len(values)
        for _, group in itertools.groupby(ordered, key=values.__getitem__):
            indices = list(group)
            rank = sum(ordered.index(i) for i in indices) / len(indices)
            for i in indices:
                result[i] = rank
        mean = statistics.fmean(result)
        return [v - mean for v in result]

    x, y = ranks(a), ranks(b)
    denominator = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    return sum(i * j for i, j in zip(x, y, strict=True)) / denominator if denominator else None


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def observed(record):
    m, b, s = record["measured"], record["measured"]["binary"], record["measured"]["shared"]
    values = {
        "real_binary_miou": b["mean_iou"],
        "real_floor_iou": b["per_class"]["floor"]["iou"],
        "real_nonfloor_iou": b["per_class"]["nonfloor"]["iou"],
        "real_pixel_accuracy": b["pixel_accuracy"],
        "real_mean_precision": b["mean_precision"],
        "real_mean_recall": b["mean_recall"],
        "real_shared_miou": s["mean_iou"],
        "real_shared_floor_iou": s["per_class"]["floor"]["iou"],
        "real_storage_iou": s["per_class"]["storage_obstacle"]["iou"],
        "id_binary_miou": record["id_binary"]["mean_iou"],
        "id_four_class_miou": record["id_four_class_miou"],
        "synthetic_shifted_binary_miou": statistics.fmean(
            v["mean_iou"] for k, v in record["projected_shifts"].items() if k != "nominal"
        ),
        "gap_shared_floor_iou": record["id_shared"]["per_class"]["floor"]["iou"]
        - s["per_class"]["floor"]["iou"],
        "boundary_error_rate": m["boundary_error_rate"],
        "interior_error_rate": m["interior_error_rate"],
        "latency_p50_ms": record["timing"]["p50_ms"],
        "latency_p95_ms": record["timing"]["p95_ms"],
        "preprocess_mean_ms": m["preprocess_ms_mean"],
    }
    for key in ("binary_miou", "floor_iou", "storage_iou"):
        values[f"gap_{key}"] = record["delta_sim2real"][key]
    return values


def unique_lookup(records, fields, expected, label):
    result = {tuple(r[k] for k in fields): r for r in records}
    require(
        len(records) == len(result) and set(result) == expected,
        f"{label}: incomplete/duplicate matrix",
    )
    return result


def verify_records(study, source, manifest):
    fields = ("architecture", "arm", "seed", "size")
    runs = unique_lookup(
        study["records"],
        fields,
        set(itertools.product(ARCHITECTURES, ARMS, SEEDS, SIZES)),
        "real runs",
    )
    frozen = unique_lookup(
        source["records"],
        fields[:3],
        set(itertools.product(ARCHITECTURES, ARMS, SEEDS)),
        "frozen runs",
    )
    require(
        study["complete"]
        and source["complete"]
        and study["failed_evaluations"] == 0
        and source["failed_runs"] == 0,
        "study: incomplete or failed",
    )
    compare(
        {
            "expected_evaluations": 84,
            "evaluation_sizes": list(SIZES),
            "new_experimental_training_runs": 0,
            "reused_training_runs": 42,
        },
        study,
        "study",
    )
    images = {r["image"]["id"]: r for r in manifest["records"]}
    require(len(images) == len(manifest["records"]) == 80, "manifest: duplicate/missing images")
    support, totals = [0] * 256, {}
    for id, r in images.items():
        require(r["image"]["license"] == 4, "manifest: unexpected image license")
        counts = Counter()
        for segment in r["annotation"]["segments_info"]:
            if not segment.get("iscrowd", 0):
                counts[segment["category_id"]] += segment["area"]
        totals[id] = (counts[118] + counts[190], counts[188], sum(counts.values()))
        for category, count in counts.items():
            support[category] += count
    for key, r in runs.items():
        require(r["status"] == "success", f"{key}: non-success record")
        f = frozen[key[:3]]
        require(f["status"] == "success", "frozen run: non-success record")
        compare(f["domain"]["checkpoint_sha256"], r["checkpoint_sha256"], "checkpoint identity")
        for field, source_field in [
            ("parameters", "parameters"),
            ("parameter_bytes", "parameter_bytes"),
            ("checkpoint_bytes", "model_size_bytes"),
        ]:
            compare(f["benchmark"][source_field], r[field], field)
        for name, condition in f["domain"]["conditions"].items():
            original = matrix(condition["confusion_matrix"], 4)
            computed = metrics(original, CLASSES)
            compare(
                {k: computed[k] for k in ("mean_iou", "per_class", "pixel_accuracy")},
                condition,
                f"synthetic.{name}",
            )
            compare(binary(original), r["projected_shifts"][name], f"projected.{name}")
        nominal = f["domain"]["conditions"]["nominal"]
        compare(nominal["mean_iou"], r["id_four_class_miou"], "nominal four-class")
        compare(binary(nominal["confusion_matrix"]), r["id_binary"], "ID binary")
        compare(shared(nominal["confusion_matrix"]), r["id_shared"], "ID shared")
        m = r["measured"]
        compare({"samples": 80, "size": r["size"]}, m, "measurement schema")
        examples = unique_lookup(m["per_image"], ("id",), {(id,) for id in images}, "native images")
        for (id,), example in examples.items():
            b, s = matrix(example["binary_confusion"], 2), matrix(example["shared_confusion"], 4)
            floor, storage, valid = totals[id]
            require([sum(row) for row in b] == [floor, valid - floor], "native binary supports")
            require([sum(row) for row in s] == [floor, storage, 0, 0], "native shared supports")
            require(
                b[0] == [s[0][0], sum(s[0][1:])] and s[1][0] <= b[1][0],
                "native endpoints disagree",
            )
            compare(
                metrics(b, ("floor", "nonfloor"))["per_class"]["floor"]["iou"],
                example["floor_iou"],
                "native floor IoU",
            )
            require(
                sum(example[f"{k}_pixels"] for k in ("boundary", "interior")) == valid,
                "native boundary/interior support",
            )
            require(
                sum(example[f"{k}_errors"] for k in ("boundary", "interior")) == b[0][1] + b[1][0],
                "native boundary/interior errors",
            )
            require(
                all(
                    0 <= example[f"{k}_errors"] <= example[f"{k}_pixels"]
                    for k in ("boundary", "interior")
                ),
                "native error bounds",
            )
        b = metrics(
            summed([e["binary_confusion"] for e in examples.values()], 2), ("floor", "nonfloor")
        )
        s = shared(summed([e["shared_confusion"] for e in examples.values()], 4))
        compare(b, m["binary"], "real.binary")
        compare(s, m["shared"], "real.shared")
        compare(
            sum(map(sum, s["confusion_matrix"])) / sum(map(sum, b["confusion_matrix"])),
            m["shared_pixel_coverage"],
            "shared coverage",
        )
        compare(
            {
                "binary_miou": r["id_binary"]["mean_iou"] - b["mean_iou"],
                "floor_iou": r["id_binary"]["per_class"]["floor"]["iou"]
                - b["per_class"]["floor"]["iou"],
                "shared_miou": r["id_shared"]["mean_iou"] - s["mean_iou"],
                "storage_iou": r["id_shared"]["per_class"]["storage_obstacle"]["iou"]
                - s["per_class"]["storage_obstacle"]["iou"],
            },
            r["delta_sim2real"],
            "gaps",
        )
        compare(support, m["source_category_support"], "source category supports")
        false = m["source_category_false_floor"]
        require(
            len(false) == 256
            and all(type(n) is int and 0 <= n <= support[i] for i, n in enumerate(false)),
            "false-floor bounds",
        )
        require(
            sum(false) == b["confusion_matrix"][1][0] and false[118] == false[190] == 0,
            "false-floor counts",
        )
        for kind in ("boundary", "interior"):
            errors = sum(e[f"{kind}_errors"] for e in examples.values())
            pixels = sum(e[f"{kind}_pixels"] for e in examples.values())
            compare(errors / pixels, m[f"{kind}_error_rate"], f"{kind} rate")
        times = r["timing"]["samples_ms"]
        compare([1, 3, r["size"], r["size"]], r["timing"]["input_shape"], "timing shape")
        require(
            len(times) == r["timing"]["iterations"] == 100
            and r["timing"]["warmup"] == 20
            and all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in times),
            "timing: sample protocol",
        )
        compare(
            {
                "mean_ms": statistics.fmean(times),
                "p50_ms": percentile(times, 0.5),
                "p95_ms": percentile(times, 0.95),
            },
            r["timing"],
            "timing",
        )
    return runs


def verify_analysis(runs, analysis, csv_path):
    summaries = unique_lookup(
        analysis["summaries"],
        ("architecture", "arm", "size"),
        set(itertools.product(ARCHITECTURES, ARMS, SIZES)),
        "summaries",
    )
    pairs = unique_lookup(
        analysis["paired_deltas"],
        ("architecture", "arm", "size"),
        set(itertools.product(ARCHITECTURES, set(ARMS) - {"baseline"}, SIZES)),
        "pairs",
    )
    for (arch, arm, size), group in summaries.items():
        values = [observed(runs[arch, arm, seed, size]) for seed in SEEDS]
        compare(list(SEEDS), group["seeds"], "summary seed order")
        compare(
            {k: distribution([v[k] for v in values]) for k in values[0]},
            group["metrics"],
            f"summary.{arch}.{arm}.{size}",
        )
        if arm != "baseline":
            differences = [
                {k: v[k] - observed(runs[arch, "baseline", seed, size])[k] for k in v}
                for seed, v in zip(SEEDS, values, strict=True)
            ]
            compare(
                {
                    "reference": "baseline",
                    "seeds": list(SEEDS),
                    "deltas": {k: distribution([v[k] for v in differences]) for k in values[0]},
                },
                pairs[arch, arm, size],
                "paired deltas",
            )
    correlations = unique_lookup(
        analysis["correlations"],
        ("architecture", "size", "shift"),
        {
            (a, size, shift)
            for a in ARCHITECTURES
            for size in SIZES
            for shift in next(iter(runs.values()))["projected_shifts"]
            if shift != "nominal"
        },
        "correlations",
    )
    for (arch, size, shift), correlation in correlations.items():
        group = [r for (a, _, _, s), r in runs.items() if a == arch and s == size]
        scores = [r["projected_shifts"][shift]["mean_iou"] for r in group]
        drops = [r["id_binary"]["mean_iou"] - v for r, v in zip(group, scores, strict=True)]
        compare(
            {
                "n": 21,
                "drop_gap_spearman": spearman(
                    drops, [r["delta_sim2real"]["binary_miou"] for r in group]
                ),
                "shift_score_real_score_spearman": spearman(
                    scores, [r["measured"]["binary"]["mean_iou"] for r in group]
                ),
            },
            correlation,
            "correlation",
        )
    compare(
        {
            "failed_attempts": [],
            "interrupted_attempts": [],
            "unsupported_class_gaps": {"safety_cone": None, "mobile_robot": None},
        },
        analysis,
        "analysis",
    )
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    csv_runs = unique_lookup(
        [{**r, "seed": int(r["seed"]), "size": int(r["size"])} for r in rows],
        ("architecture", "arm", "seed", "size"),
        set(runs),
        "CSV",
    )
    for key, row in csv_runs.items():
        expected = observed(runs[key])
        for endpoint in ("binary", "shared"):
            for name, fields in runs[key]["measured"][endpoint]["per_class"].items():
                expected.update({f"{endpoint}_{name}_{k}": v for k, v in fields.items()})
        for name, value in expected.items():
            compare(float(value), float(row[name]), f"CSV.{name}")


def verify_gallery(root, runs):
    selection, credits = (
        read_json(root / "gallery/selection.json"),
        read_json(root / "gallery/credits.json"),
    )
    ids = {p["id"] for p in selection["panels"]}
    require(len(ids) == len(selection["panels"]) == len(credits) == 11, "gallery: count")
    by_id = {c["coco_id"]: c for c in credits}
    require(set(by_id) == ids, "gallery: credit IDs")
    for panel in selection["panels"]:
        credit = by_id[panel["id"]]
        require(
            panel["photo_included"] == bool(credit.get("author_name")), "gallery: missing creator"
        )
        require(
            credit["license"] == "https://creativecommons.org/licenses/by/2.0/", "gallery: license"
        )
        for arch, arm in itertools.product(ARCHITECTURES, ("baseline", "occlusion")):
            example = next(
                e
                for e in runs[arch, arm, 42, 64]["measured"]["per_image"]
                if e["id"] == panel["id"]
            )
            compare(example["floor_iou"], panel["floor_iou"][f"{arch}:{arm}"], "gallery floor IoU")
    return sorted(id for id, credit in by_id.items() if not credit.get("author_name"))


def verify(repo_root):
    evidence = Path(repo_root) / "docs/evidence"
    counts = {v: verify_inventory(evidence / v) for v in ("v1_1", "v1_2")}
    root = evidence / "v1_2"
    study, source, manifest, analysis = [
        read_json(p)
        for p in (
            root / "study.json",
            evidence / "v1_1/study.json",
            root / "manifest.json",
            root / "analysis.json",
        )
    ]
    require(
        hashlib.sha256((evidence / "v1_1/study.json").read_bytes()).hexdigest()
        == study["source_study_sha256"],
        "frozen study hash",
    )
    provenance = read_json(root / "provenance.json")
    for name, metadata in provenance.items():
        require(
            name
            in {
                "study.json",
                "manifest.json",
                "deployment.json",
                "gpu.json",
                "reproducibility.json",
            },
            "provenance: unexpected file",
        )
        original = (
            json.dumps(read_json(root / name), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        require(
            hashlib.sha256(original).hexdigest() == metadata["original_sha256"],
            f"{name}: reconstructed original JSON hash",
        )
    compare(
        provenance["manifest.json"]["original_sha256"],
        study["manifest_sha256"],
        "manifest identity",
    )
    runs = verify_records(study, source, manifest)
    verify_analysis(runs, analysis, root / "real_runs.csv")
    unresolved = verify_gallery(root, runs)
    return {
        "status": "passed",
        "inventory_files_verified": counts,
        "frozen_models": 42,
        "dataset_evaluations": len(runs),
        "native_image_records": 6720,
        "seed_summary_groups": 28,
        "paired_contrasts": 24,
        "descriptive_correlations": 48,
        "gallery_panels": 11,
        "unresolved_creator_ids": unresolved,
        "scope": "Recorded arithmetic, cross-artifact consistency and hashes only; no inference.",
        "not_verified": [
            "original image/mask/checkpoint bytes",
            "measurement authenticity",
            "hardware latency",
            "current external links/licenses",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.repo_root), indent=2, sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError, StopIteration, RecursionError) as error:
        print(f"Published evidence verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
