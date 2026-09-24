from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import config_fingerprint, save_config
from .data import (
    CLASS_NAMES,
    SegmentationDataset,
    make_dataloader,
    prepare_dataset,
    validate_dataset,
)
from .engine import evaluate_model, train_one_epoch
from .models import build_model, count_parameters
from .reproducibility import capture_rng_state, resolve_device, restore_rng_state, seed_everything
from .runlog import (
    JsonlLogger,
    environment_metadata,
    optional_tensorboard_writer,
    write_epoch_csv,
    write_json,
)


def _class_weights(validation: dict[str, Any], device: torch.device) -> torch.Tensor:
    training_pixels = validation["split_class_pixels"]["train"]
    pixels = np.asarray([training_pixels[name] for name in CLASS_NAMES], dtype=np.float64)
    weights = np.sqrt(pixels.sum() / np.maximum(pixels, 1.0))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    best_mean_iou: float,
    config: dict[str, Any],
    data_loader_generator: torch.Generator | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 2,
            "epoch": epoch,
            "best_mean_iou": best_mean_iou,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "data_loader_generator_state": (
                data_loader_generator.get_state() if data_loader_generator is not None else None
            ),
            "config": config,
            "config_fingerprint": config_fingerprint(config),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    required = {"format_version", "epoch", "model_state", "config"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    return checkpoint


def _validate_resume_config(current: dict[str, Any], previous: dict[str, Any]) -> None:
    immutable_paths = [
        ("data", "manifest"),
        ("data", "image_size"),
        ("model", "architecture"),
        ("model", "num_classes"),
        ("model", "base_channels"),
    ]
    mismatches = []
    for section, key in immutable_paths:
        if current[section][key] != previous[section][key]:
            mismatches.append(f"{section}.{key}")
    if mismatches:
        raise ValueError(f"Resume configuration changes model/data identity: {mismatches}")


def run_training(config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    training = config["training"]
    seed = int(training["seed"])
    deterministic = bool(training["deterministic"])
    seed_everything(seed, deterministic)
    device = resolve_device(training["device"])
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    run_dir = Path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
    last_checkpoint = run_dir / "checkpoints" / "last.pt"
    resume_path = Path(training["resume"]) if training.get("resume") else None
    if last_checkpoint.exists() and resume_path is None:
        if not overwrite:
            raise FileExistsError(
                f"Run already contains a checkpoint: {last_checkpoint}. Use --overwrite or resume."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_dataset(config)
    validation = validate_dataset(
        config["data"]["manifest"], check_hashes=bool(config["data"]["validate_hashes"])
    )
    save_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", environment_metadata())
    write_json(run_dir / "data_validation.json", validation)

    train_dataset = SegmentationDataset(
        config["data"]["manifest"], "train", config["augmentation"], seed=seed
    )
    val_dataset = SegmentationDataset(config["data"]["manifest"], "val", seed=seed)
    train_loader = make_dataloader(
        train_dataset,
        int(training["batch_size"]),
        True,
        int(config["data"]["num_workers"]),
        seed,
    )
    val_loader = make_dataloader(
        val_dataset,
        int(training["batch_size"]),
        False,
        int(config["data"]["num_workers"]),
        seed,
    )
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(validation, device))
    start_epoch = 0
    best_mean_iou = -1.0
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, map_location=device)
        _validate_resume_config(config, checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        loader_generator_state = checkpoint.get("data_loader_generator_state")
        if loader_generator_state is not None and train_loader.generator is not None:
            train_loader.generator.set_state(loader_generator_state)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mean_iou = float(checkpoint["best_mean_iou"])
    if start_epoch >= int(training["epochs"]):
        raise ValueError(
            f"Checkpoint already reached epoch {start_epoch}; set training.epochs above that value"
        )

    event_log = JsonlLogger(run_dir / "events.jsonl")
    event_log.log(
        {
            "event": "run_started",
            "time_unix": time.time(),
            "device": str(device),
            "amp_enabled": amp_enabled,
            "parameters": count_parameters(model),
            "start_epoch": start_epoch,
        }
    )
    writer = optional_tensorboard_writer(run_dir / "tensorboard")
    final_record: dict[str, Any] = {}
    for epoch in range(start_epoch, int(training["epochs"])):
        train_dataset.set_epoch(epoch)
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, amp_enabled
        )
        evaluation = evaluate_model(
            model,
            val_loader,
            device,
            criterion=criterion,
            ap_bins=int(config["evaluation"]["ap_bins"]),
        )
        scheduler.step()
        final_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": evaluation.metrics["loss"],
            "val_pixel_accuracy": evaluation.metrics["pixel_accuracy"],
            "val_mean_iou": evaluation.metrics["mean_iou"],
            "val_mean_precision": evaluation.metrics["mean_precision"],
            "val_mean_recall": evaluation.metrics["mean_recall"],
            "val_pixel_map_foreground": evaluation.metrics["pixel_map_foreground"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        event_log.log({"event": "epoch_completed", **final_record})
        write_epoch_csv(run_dir / "metrics.csv", final_record)
        if writer is not None:
            for key, value in final_record.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        improved = evaluation.metrics["mean_iou"] > best_mean_iou
        if improved:
            best_mean_iou = evaluation.metrics["mean_iou"]
        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_mean_iou,
            config,
            train_loader.generator,
        )
        if improved:
            shutil.copy2(last_checkpoint, run_dir / "checkpoints" / "best.pt")
        if (epoch + 1) % int(training["save_every"]) == 0:
            shutil.copy2(last_checkpoint, run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
    if writer is not None:
        writer.close()
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(last_checkpoint),
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "best_val_mean_iou": best_mean_iou,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "parameters": count_parameters(model),
        "last_epoch": final_record,
    }
    write_json(run_dir / "training_summary.json", summary)
    event_log.log({"event": "run_completed", "time_unix": time.time(), **summary})
    return summary
