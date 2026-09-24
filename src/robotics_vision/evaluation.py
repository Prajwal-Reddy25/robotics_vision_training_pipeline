from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import validate_config
from .data import SegmentationDataset, colorize_mask, make_dataloader, validate_dataset
from .engine import evaluate_model
from .models import build_model
from .reproducibility import resolve_device, seed_everything
from .runlog import write_json
from .training import load_checkpoint, validate_checkpoint_model_config


def _save_error_analysis(
    examples: list[dict[str, Any]], output_dir: Path, limit: int
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(examples, key=lambda item: item["mean_iou"])
    summary: list[dict[str, Any]] = []
    for rank, example in enumerate(ranked[:limit], start=1):
        image = np.clip(example["image"].numpy().transpose(1, 2, 0) * 255.0, 0, 255).astype(
            np.uint8
        )
        truth = colorize_mask(example["target"].numpy())
        prediction = colorize_mask(example["prediction"].numpy())
        separator = np.full((image.shape[0], 2, 3), 255, dtype=np.uint8)
        panel = np.concatenate((image, separator, truth, separator, prediction), axis=1)
        filename = f"{rank:02d}_{example['id']}.png"
        Image.fromarray(panel).save(output_dir / filename)
        summary.append(
            {"rank": rank, "id": example["id"], "mean_iou": example["mean_iou"], "panel": filename}
        )
    write_json(
        output_dir / "summary.json",
        {
            "columns": ["input", "ground_truth", "prediction"],
            "ranking": "ascending per-image mean IoU",
            "examples": summary,
        },
    )
    return summary


def run_evaluation(
    config: dict[str, Any], checkpoint_path: str | Path, output_dir: str | Path | None = None
) -> dict[str, Any]:
    validate_config(config)
    seed_everything(int(config["training"]["seed"]), bool(config["training"]["deterministic"]))
    device = resolve_device(config["training"]["device"])
    validate_dataset(
        config["data"]["manifest"], check_hashes=bool(config["data"]["validate_hashes"])
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    validate_checkpoint_model_config(config, checkpoint)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    split = config["evaluation"]["split"]
    dataset = SegmentationDataset(
        config["data"]["manifest"], split, seed=config["training"]["seed"]
    )
    loader = make_dataloader(
        dataset,
        int(config["training"]["batch_size"]),
        False,
        int(config["data"]["num_workers"]),
        int(config["training"]["seed"]),
    )
    output = evaluate_model(
        model,
        loader,
        device,
        ap_bins=int(config["evaluation"]["ap_bins"]),
        collect_examples=True,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": split,
        "device": str(device),
        **output.metrics,
    }
    destination = (
        Path(output_dir)
        if output_dir
        else Path(config["experiment"]["output_dir"]) / config["experiment"]["name"] / "evaluation"
    )
    destination.mkdir(parents=True, exist_ok=True)
    result["worst_examples"] = _save_error_analysis(
        output.examples, destination / "error_analysis", int(config["evaluation"]["save_worst"])
    )
    write_json(destination / "metrics.json", result)
    return result
