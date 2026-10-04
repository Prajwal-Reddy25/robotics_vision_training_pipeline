from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmarking import _sha256
from .config import config_fingerprint, validate_config
from .data import SegmentationDataset, make_dataloader, validate_dataset
from .engine import evaluate_model
from .models import build_model
from .reproducibility import resolve_device, seed_everything
from .runlog import environment_metadata, write_json
from .shifts import CONDITIONS, SHIFT_VERSION, ShiftedDataset
from .training import load_checkpoint, validate_checkpoint_model_config


def degradation(nominal: float, shifted: float) -> dict[str, float | None]:
    drop = nominal - shifted
    return {"absolute_drop": drop, "relative_drop": drop / nominal if nominal else None}


def run_domain_evaluation(
    config: dict[str, Any], checkpoint_path: str | Path,
    output_path: str | Path, shift_seed: int = 2026,
) -> dict[str, Any]:
    validate_config(config)
    if Path(output_path).exists():
        raise FileExistsError(f"Domain report already exists: {output_path}")
    if isinstance(shift_seed, bool) or not isinstance(shift_seed, int) or shift_seed < 0:
        raise ValueError("shift_seed must be a nonnegative integer")
    validation = validate_dataset(config["data"]["manifest"], check_hashes=True)
    seed_everything(config["training"]["seed"], config["training"]["deterministic"])
    device = resolve_device(config["training"]["device"])
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    validate_checkpoint_model_config(config, checkpoint)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    nominal = SegmentationDataset(config["data"]["manifest"], config["evaluation"]["split"])
    cases = [("nominal", "nominal", 0.0)]
    cases += [(f"{kind}_{level}", kind, strength)
              for kind, strengths in CONDITIONS.items()
              for level, strength in zip(("low", "high"), strengths, strict=True)]
    results = {}
    for name, kind, strength in cases:
        dataset = ShiftedDataset(nominal, kind, strength, shift_seed)
        loader = make_dataloader(
            dataset, config["training"]["batch_size"], False,
            config["data"]["num_workers"], shift_seed,
        )
        measured = evaluate_model(model, loader, device, ap_bins=config["evaluation"]["ap_bins"])
        results[name] = {
            "kind": kind, "strength": strength, "content_sha256": dataset.fingerprint(),
            **{key: measured.metrics[key] for key in (
                "mean_iou", "per_class", "confusion_matrix", "pixel_accuracy",
                "samples", "evaluated_iou_classes",
            )},
        }
    nominal_iou = results["nominal"]["mean_iou"]
    for result in results.values():
        result.update(degradation(nominal_iou, result["mean_iou"]))
        result["class_iou_drop"] = {
            name: results["nominal"]["per_class"][name]["iou"] - values["iou"]
            for name, values in result["per_class"].items()
        }
    report = {
        "format_version": 1, "shift_version": SHIFT_VERSION, "shift_seed": shift_seed,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(Path(checkpoint_path)),
        "checkpoint_epoch": checkpoint["epoch"], "config_fingerprint": config_fingerprint(config),
        "manifest_sha256": _sha256(Path(config["data"]["manifest"])),
        "source_ids": [record["id"] for record in nominal.records],
        "split": config["evaluation"]["split"], "data_validation": validation,
        "environment": environment_metadata(), "device": str(device), "conditions": results,
        "mean_shifted_iou": sum(r["mean_iou"] for n, r in results.items() if n != "nominal") / 12,
        "note": "Paired synthetic stressors; viewpoint is a planar proxy, not calibrated 3D.",
    }
    write_json(output_path, report)
    return report
