"""Small predeclared ablation runner; retain every success and failure."""
from __future__ import annotations

import copy
import hashlib
import json
import statistics
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

from .benchmarking import _sha256, run_benchmark
from .config import DEFAULT_CONFIG, load_config, validate_config
from .data import prepare_dataset
from .domain import run_domain_evaluation
from .runlog import environment_metadata, write_json
from .shifts import CONDITIONS, SHIFT_VERSION
from .training import run_training


def load_study(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"version", "base_config", "seeds", "architectures", "arms", "shift_seed"}
    if not isinstance(spec, dict) or set(spec) != required or spec["version"] != 1:
        raise ValueError(f"Study must contain exactly {sorted(required)} with version=1")
    if not isinstance(spec["base_config"], str):
        raise ValueError("base_config must be a path string")
    base = load_config(path.parent / spec["base_config"])
    if base["training"]["resume"] is not None or base["evaluation"]["split"] != "test":
        raise ValueError("Study requires fresh training and evaluation.split=test")
    for key in ("seeds", "architectures"):
        values = spec[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"Study {key} must be a nonempty unique list")
    for seed in [*spec["seeds"], spec["shift_seed"]]:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Study seeds must be nonnegative integers")
    if set(spec["architectures"]) - {"tiny_unet", "lite_fcn"}:
        raise ValueError("Unknown study architecture")
    if not isinstance(spec["arms"], dict) or "baseline" not in spec["arms"]:
        raise ValueError("Study arms must be a mapping including baseline")
    for name, overrides in spec["arms"].items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Study arm names must be identifiers")
        if (not isinstance(overrides, dict)
                or set(overrides) - DEFAULT_CONFIG["augmentation"].keys()):
            raise ValueError(f"Unknown augmentation override in arm {name}")
        candidate = copy.deepcopy(base)
        candidate["augmentation"].update(overrides)
        validate_config(candidate)
    return spec, base


def _distribution(values: list[float]) -> dict[str, Any]:
    return {"mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values), "values": values}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [record for record in records if record["status"] == "success"]
    groups = sorted({(r["architecture"], r["arm"]) for r in records})
    summaries = []
    for architecture, arm in groups:
        group = [r for r in successful if (r["architecture"], r["arm"]) == (architecture, arm)]
        if not group:
            summaries.append({"architecture": architecture, "arm": arm, "n": 0})
            continue
        conditions = {}
        for name in group[0]["domain"]["conditions"]:
            conditions[name] = {
                "mean_iou": _distribution([r["domain"]["conditions"][name]["mean_iou"]
                                           for r in group]),
                "absolute_drop": _distribution([r["domain"]["conditions"][name]["absolute_drop"]
                                                for r in group]),
                "class_iou": {
                    cls: _distribution([r["domain"]["conditions"][name]["per_class"][cls]["iou"]
                                        for r in group])
                    for cls in group[0]["domain"]["conditions"][name]["per_class"]
                },
            }
        paired = []
        for record in group:
            match = next((r for r in successful if r["architecture"] == architecture
                          and r["arm"] == "baseline" and r["seed"] == record["seed"]), None)
            if match:
                paired.append({
                    "seed": record["seed"],
                    "nominal_delta": record["domain"]["conditions"]["nominal"]["mean_iou"]
                    - match["domain"]["conditions"]["nominal"]["mean_iou"],
                    "shifted_delta": record["domain"]["mean_shifted_iou"]
                    - match["domain"]["mean_shifted_iou"],
                })
        summaries.append({
            "architecture": architecture, "arm": arm, "n": len(group), "conditions": conditions,
            "mean_shifted_iou": _distribution([r["domain"]["mean_shifted_iou"] for r in group]),
            "paired_baseline_deltas": paired,
            "paired_nominal_delta": _distribution([p["nominal_delta"] for p in paired])
            if paired else None,
            "paired_shifted_delta": _distribution([p["shifted_delta"] for p in paired])
            if paired else None,
        })
    return summaries


def run_study(study_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    spec, base = load_study(study_path)
    output = Path(output_dir)
    # Never overwrite an experiment, including failed/partial runs.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(base["benchmark"]["threads"])
    manifest = prepare_dataset(base)
    resolved = {"spec": spec, "base": base, "conditions": CONDITIONS,
                "shift_version": SHIFT_VERSION}
    report = {
        "format_version": 1, "resolved_study": resolved,
        "study_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
        "manifest_sha256": _sha256(manifest), "environment": environment_metadata(),
        "expected_runs": len(spec["seeds"]) * len(spec["architectures"]) * len(spec["arms"]),
        "records": [], "complete": False,
    }
    write_json(output / "study.json", report)
    for architecture in spec["architectures"]:
        for arm, overrides in spec["arms"].items():
            for seed in spec["seeds"]:
                name = f"{architecture}_{arm}_s{seed}"
                config = copy.deepcopy(base)
                config["experiment"] = {"name": name, "output_dir": str(output / "training")}
                config["model"]["architecture"] = architecture
                config["training"]["seed"] = seed
                config["augmentation"].update(overrides)
                record = {"architecture": architecture, "arm": arm, "seed": seed,
                          "config": config, "status": "started"}
                report["records"].append(record)
                write_json(output / "study.json", report)
                print(f"STUDY started {name}", flush=True)
                try:
                    training = run_training(config)
                    checkpoint = training["best_checkpoint"]
                    record["training"] = training
                    record["domain"] = run_domain_evaluation(
                        config, checkpoint, output / f"{name}.domain.json", spec["shift_seed"],
                    )
                    record["benchmark"] = run_benchmark(
                        config, checkpoint, "pytorch", output / f"{name}.benchmark.json",
                    )
                    record["status"] = "success"
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    (output / f"{name}.failure.txt").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                write_json(output / "study.json", report)
                print(f"STUDY {record['status']} {name}", flush=True)
    report["failed_runs"] = sum(r["status"] != "success" for r in report["records"])
    report["complete"] = len(report["records"]) == report["expected_runs"]
    report["summaries"] = summarize(report["records"])
    write_json(output / "study.json", report)
    return report
