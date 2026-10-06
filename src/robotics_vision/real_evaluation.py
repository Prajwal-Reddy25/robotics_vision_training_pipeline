"""Frozen-model real-photo transfer with explicitly projected label spaces."""

from __future__ import annotations

import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .benchmarking import _sha256
from .metrics import SegmentationMetrics
from .models import build_model, count_parameters
from .real_data import FLOOR_CATEGORIES, STORAGE_CATEGORY, validate_real_dataset
from .reproducibility import seed_everything
from .runlog import environment_metadata, write_json
from .training import load_checkpoint, validate_checkpoint_model_config


def letterbox(image: Image.Image, size: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    if image.mode != "RGB" or size < 16:
        raise ValueError("Letterbox requires RGB and size >=16")
    width, height = image.size
    scale = size / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - w) // 2, (size - h) // 2
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(image.resize((w, h), Image.Resampling.BILINEAR), (left, top))
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy() / 255
    return torch.from_numpy(array).unsqueeze(0), (left, top, w, h)


def restore_prediction(
    logits: torch.Tensor, box: tuple[int, int, int, int], original_size: tuple[int, int]
) -> np.ndarray:
    if logits.ndim != 4 or logits.shape[:2] != (1, 4) or not torch.isfinite(logits).all():
        raise ValueError("Real evaluation requires finite [1,4,H,W] logits")
    left, top, width, height = box
    if min(left, top) < 0 or min(width, height) <= 0:
        raise ValueError("Invalid letterbox geometry")
    if left + width > logits.shape[3] or top + height > logits.shape[2]:
        raise ValueError("Letterbox crop exceeds logits")
    classes = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    cropped = classes[top : top + height, left : left + width]
    return np.asarray(Image.fromarray(cropped).resize(original_size, Image.Resampling.NEAREST))


def binary_metrics(confusion: np.ndarray | list) -> dict[str, Any]:
    matrix = np.asarray(confusion)
    if not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
        raise ValueError("Confusion counts must be nonnegative integers")
    if matrix.shape == (4, 4):
        matrix = np.array(
            [[matrix[0, 0], matrix[0, 1:].sum()], [matrix[1:, 0].sum(), matrix[1:, 1:].sum()]]
        )
    if matrix.shape != (2, 2) or np.any(matrix < 0):
        raise ValueError("Expected nonnegative 2×2 or 4×4 confusion matrix")
    meter = SegmentationMetrics(2, ["floor", "nonfloor"])
    meter.confusion = matrix.copy()
    return meter.compute()


def restricted_metrics(confusion: np.ndarray | list) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (4, 4) or np.any(matrix < 0):
        raise ValueError("Expected a nonnegative four-class confusion matrix")
    matrix = matrix.copy()
    matrix[2:] = 0  # Ignore unsupported target classes, keep all prediction errors.
    names = ["floor", "storage_obstacle", "unsupported_cone", "unsupported_robot"]
    meter = SegmentationMetrics(4, names)
    meter.confusion = matrix
    metrics = meter.compute()
    union = matrix.sum(0) + matrix.sum(1) - np.diag(matrix)
    eligible = [i for i in (0, 1) if union[i] > 0]
    metrics["mean_iou"] = (
        statistics.fmean(metrics["per_class"][names[i]]["iou"] for i in eligible)
        if eligible
        else None
    )
    metrics["mean_precision"] = statistics.fmean(
        metrics["per_class"][names[i]]["precision"] for i in (0, 1)
    )
    metrics["mean_recall"] = statistics.fmean(
        metrics["per_class"][names[i]]["recall"] for i in (0, 1)
    )
    metrics["per_class"] = {name: metrics["per_class"][name] for name in names[:2]}
    metrics["evaluated_iou_classes"] = len(eligible)
    metrics["unsupported_prediction_pixels"] = int(matrix[:, 2:].sum())
    metrics["confusion_columns"] = names
    return metrics


def boundary_band(floor: np.ndarray, valid: np.ndarray, radius: int = 3) -> np.ndarray:
    if floor.shape != valid.shape or radius < 0:
        raise ValueError("Invalid boundary inputs")
    edge = np.zeros(floor.shape, dtype=bool)
    vertical = (floor[1:] != floor[:-1]) & valid[1:] & valid[:-1]
    horizontal = (floor[:, 1:] != floor[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[1:] |= vertical
    edge[:-1] |= vertical
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    padded = np.pad(edge, radius)
    band = np.zeros_like(edge)
    h, w = edge.shape
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            band |= padded[y : y + h, x : x + w]
    return band & valid


def native_confusions(
    prediction: np.ndarray, categories: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if (
        prediction.shape != categories.shape
        or not np.issubdtype(prediction.dtype, np.integer)
        or np.any(prediction < 0)
        or np.any(prediction > 3)
    ):
        raise ValueError("Invalid native predictions or dimensions")
    valid = categories != 255
    floor = np.isin(categories, list(FLOOR_CATEGORIES))
    truth_binary = (~floor).astype(np.int64)
    pred_binary = (prediction != 0).astype(np.int64)
    binary = np.bincount((2 * truth_binary + pred_binary)[valid], minlength=4).reshape(2, 2)
    eligible = floor | (categories == STORAGE_CATEGORY)
    truth_shared = (categories == STORAGE_CATEGORY).astype(np.int64)
    shared = np.bincount((4 * truth_shared + prediction)[eligible], minlength=16).reshape(4, 4)
    return binary, shared


def load_cases(manifest_path: Path) -> tuple[dict, list[dict]]:
    validate_real_dataset(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cases = []
    for record in manifest["records"]:
        with Image.open(manifest_path.parent / record["image_path"]) as source:
            image = source.copy()
        with Image.open(manifest_path.parent / record["category_path"]) as source:
            categories = np.asarray(source).copy()
        floor = np.isin(categories, list(FLOOR_CATEGORIES))
        valid = categories != 255
        intensity = np.asarray(image, dtype=np.float32).mean(2) / 255
        cases.append(
            {
                "id": record["image"]["id"],
                "image": image,
                "categories": categories,
                "floor": floor,
                "valid": valid,
                "band": boundary_band(floor, valid),
                "input_proxies": {
                    "mean_luminance": float(intensity.mean()),
                    "luminance_sd": float(intensity.std()),
                    "gradient_energy": float(
                        np.mean(np.diff(intensity, axis=0) ** 2)
                        + np.mean(np.diff(intensity, axis=1) ** 2)
                    ),
                },
            }
        )
    return manifest, cases


@torch.inference_mode()
def evaluate_real_cases(model: torch.nn.Module, cases: list[dict], size: int) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    total_binary, total_shared = np.zeros((2, 2), np.int64), np.zeros((4, 4), np.int64)
    examples, preprocess_ms = [], []
    source_support, false_floor = np.zeros(256, np.int64), np.zeros(256, np.int64)
    band_errors = band_pixels = interior_errors = interior_pixels = 0
    for case in cases:
        started = time.perf_counter()
        image, box = letterbox(case["image"], size)
        preprocess_ms.append((time.perf_counter() - started) * 1000)
        prediction = restore_prediction(model(image.to(device)), box, case["image"].size)
        binary, shared = native_confusions(prediction, case["categories"])
        total_binary += binary
        total_shared += shared
        errors = (prediction == 0) != case["floor"]
        band = case["band"]
        interior = case["valid"] & ~band
        be, bp = int((errors & band).sum()), int(band.sum())
        ie, ip = int((errors & interior).sum()), int(interior.sum())
        band_errors += be
        band_pixels += bp
        interior_errors += ie
        interior_pixels += ip
        source_support += np.bincount(case["categories"][case["valid"]], minlength=256)
        false = case["valid"] & ~case["floor"] & (prediction == 0)
        false_floor += np.bincount(case["categories"][false], minlength=256)
        examples.append(
            {
                "id": case["id"],
                "binary_confusion": binary.tolist(),
                "shared_confusion": shared.tolist(),
                "floor_iou": binary_metrics(binary)["per_class"]["floor"]["iou"],
                "boundary_errors": be,
                "boundary_pixels": bp,
                "interior_errors": ie,
                "interior_pixels": ip,
                "input_proxies": case["input_proxies"],
            }
        )
    return {
        "size": size,
        "samples": len(cases),
        "binary": binary_metrics(total_binary),
        "shared": restricted_metrics(total_shared),
        "per_image": examples,
        "shared_pixel_coverage": float(total_shared.sum() / total_binary.sum()),
        "boundary_error_rate": band_errors / band_pixels if band_pixels else None,
        "interior_error_rate": interior_errors / interior_pixels if interior_pixels else None,
        "source_category_support": source_support.tolist(),
        "source_category_false_floor": false_floor.tolist(),
        "preprocess_ms_mean": statistics.fmean(preprocess_ms),
        "preprocess_note": "RGB resize/letterbox/tensor conversion; excludes transfer/inference.",
    }


@torch.inference_mode()
def time_model(model: torch.nn.Module, size: int) -> dict:
    device = next(model.parameters()).device
    image = torch.zeros(1, 3, size, size, device=device)
    for _ in range(20):
        model(image)
    durations = []
    for _ in range(100):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - started) * 1000)
    return {
        "warmup": 20,
        "iterations": 100,
        "input_shape": [1, 3, size, size],
        "mean_ms": statistics.fmean(durations),
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "samples_ms": durations,
        "scope": "Resident FP32 model call only; excludes preprocessing/transfers/postprocessing",
    }


def run_real_study(source_path: str | Path, manifest_path: str | Path, output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_path, manifest_path = Path(source_path), Path(manifest_path)
    source = json.loads(source_path.read_text())
    if not source["complete"] or source["failed_runs"] or len(source["records"]) != 42:
        raise ValueError("Requires the complete frozen v1.1 study")
    _, cases = load_cases(manifest_path)
    torch.set_num_threads(1)
    report = {
        "format_version": 1,
        "frozen_baseline_commit": "d9a74a0bebc48b0facd049f24cb2889bf3bbda4e",
        "source_study_sha256": _sha256(source_path),
        "manifest_sha256": _sha256(manifest_path),
        "environment": environment_metadata(),
        "new_experimental_training_runs": 0,
        "reused_training_runs": 42,
        "evaluation_sizes": [64, 128],
        "records": [],
        "expected_evaluations": 84,
        "complete": False,
    }

    def save():
        temporary = output / "study.json.tmp"
        write_json(temporary, report)
        temporary.replace(output / "study.json")

    save()
    for frozen in source["records"]:
        config = frozen["config"]
        checkpoint_path = Path(frozen["training"]["best_checkpoint"])
        if _sha256(checkpoint_path) != frozen["domain"]["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint content changed")
        seed_everything(frozen["seed"], True)
        checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint_model_config(config, checkpoint)
        model = build_model(config["model"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        nominal = frozen["domain"]["conditions"]["nominal"]["confusion_matrix"]
        identity = {
            "architecture": frozen["architecture"],
            "arm": frozen["arm"],
            "seed": frozen["seed"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "parameters": count_parameters(model),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "id_four_class_miou": frozen["domain"]["conditions"]["nominal"]["mean_iou"],
            "id_binary": binary_metrics(nominal),
            "id_shared": restricted_metrics(nominal),
            "projected_shifts": {
                name: binary_metrics(value["confusion_matrix"])
                for name, value in frozen["domain"]["conditions"].items()
            },
        }
        for size in (64, 128):
            entry = {**identity, "size": size, "status": "started"}
            report["records"].append(entry)
            save()
            print(
                f"REAL started {entry['architecture']} {entry['arm']} {entry['seed']} {size}",
                flush=True,
            )
            try:
                measured = evaluate_real_cases(model, cases, size)
                entry.update(measured=measured, timing=time_model(model, size), status="success")
                entry["delta_sim2real"] = {
                    "binary_miou": identity["id_binary"]["mean_iou"]
                    - measured["binary"]["mean_iou"],
                    "floor_iou": identity["id_binary"]["per_class"]["floor"]["iou"]
                    - measured["binary"]["per_class"]["floor"]["iou"],
                    "shared_miou": identity["id_shared"]["mean_iou"]
                    - measured["shared"]["mean_iou"],
                    "storage_iou": identity["id_shared"]["per_class"]["storage_obstacle"]["iou"]
                    - measured["shared"]["per_class"]["storage_obstacle"]["iou"],
                }
            except Exception as exc:
                entry.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            save()
            print(f"REAL {entry['status']}", flush=True)
    report["failed_evaluations"] = sum(r["status"] != "success" for r in report["records"])
    report["complete"] = len(report["records"]) == report["expected_evaluations"]
    save()
    return report
