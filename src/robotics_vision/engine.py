from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CLASS_NAMES
from .metrics import SegmentationMetrics, StreamingPixelAP


@dataclass
class EvaluationOutput:
    metrics: dict[str, Any]
    examples: list[dict[str, Any]]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    criterion: nn.Module | None = None,
    ap_bins: int = 256,
    collect_examples: bool = False,
) -> EvaluationOutput:
    model.eval()
    confusion = SegmentationMetrics(len(CLASS_NAMES), CLASS_NAMES)
    ap = StreamingPixelAP(len(CLASS_NAMES), CLASS_NAMES, bins=ap_bins)
    examples: list[dict[str, Any]] = []
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        confusion.update(predictions, targets)
        ap.update(logits, targets)
        if criterion is not None:
            loss = criterion(logits, targets)
            total_loss += float(loss) * images.shape[0]
        total_samples += images.shape[0]
        if collect_examples:
            for index, identifier in enumerate(batch["id"]):
                pred = predictions[index].detach().cpu()
                truth = targets[index].detach().cpu()
                sample_metrics = SegmentationMetrics(len(CLASS_NAMES), CLASS_NAMES)
                sample_metrics.update(pred, truth)
                examples.append(
                    {
                        "id": identifier,
                        "mean_iou": sample_metrics.compute()["mean_iou"],
                        "image": images[index].detach().cpu(),
                        "target": truth,
                        "prediction": pred,
                    }
                )
    metrics = confusion.compute()
    metrics.update(ap.compute())
    if criterion is not None:
        metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["samples"] = total_samples
    return EvaluationOutput(metrics=metrics, examples=examples)
