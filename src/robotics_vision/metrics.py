from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    # Undefined class metrics use the explicit and JSON-safe zero-division policy.
    output = np.zeros(numerator.shape, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator != 0)
    return output


class SegmentationMetrics:
    def __init__(self, num_classes: int, class_names: list[str]) -> None:
        self.num_classes = num_classes
        self.class_names = class_names
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        pred = prediction.detach().to("cpu", dtype=torch.int64).reshape(-1).numpy()
        truth = target.detach().to("cpu", dtype=torch.int64).reshape(-1).numpy()
        valid = (truth >= 0) & (truth < self.num_classes)
        encoded = self.num_classes * truth[valid] + pred[valid]
        self.confusion += np.bincount(encoded, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, Any]:
        matrix = self.confusion
        true_positive = np.diag(matrix).astype(np.float64)
        actual = matrix.sum(axis=1).astype(np.float64)
        predicted = matrix.sum(axis=0).astype(np.float64)
        precision = _safe_divide(true_positive, predicted)
        recall = _safe_divide(true_positive, actual)
        iou = _safe_divide(true_positive, actual + predicted - true_positive)
        total = matrix.sum()
        return {
            "pixel_accuracy": float(true_positive.sum() / total) if total else float("nan"),
            "mean_precision": float(np.mean(precision)),
            "mean_recall": float(np.mean(recall)),
            "mean_iou": float(np.mean(iou)),
            "per_class": {
                name: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "iou": float(iou[index]),
                    "support_pixels": int(actual[index]),
                }
                for index, name in enumerate(self.class_names)
            },
            "confusion_matrix": matrix.tolist(),
            "confusion_matrix_rows": "ground_truth",
            "confusion_matrix_columns": "prediction",
        }


class StreamingPixelAP:
    """Histogram-based one-vs-rest pixel AP with bounded memory usage."""

    def __init__(self, num_classes: int, class_names: list[str], bins: int = 256) -> None:
        if bins < 2:
            raise ValueError("AP bins must be at least 2")
        self.num_classes = num_classes
        self.class_names = class_names
        self.bins = bins
        self.positive = np.zeros((num_classes, bins), dtype=np.int64)
        self.negative = np.zeros((num_classes, bins), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probabilities = torch.softmax(logits.detach(), dim=1).to("cpu")
        target_cpu = target.detach().to("cpu")
        for class_index in range(self.num_classes):
            scores = probabilities[:, class_index].reshape(-1).numpy()
            labels = target_cpu.reshape(-1).numpy() == class_index
            indices = np.minimum((scores * self.bins).astype(np.int64), self.bins - 1)
            self.positive[class_index] += np.bincount(indices[labels], minlength=self.bins)
            self.negative[class_index] += np.bincount(indices[~labels], minlength=self.bins)

    def compute(self) -> dict[str, Any]:
        values: dict[str, float] = {}
        for class_index, name in enumerate(self.class_names):
            positives = self.positive[class_index, ::-1]
            negatives = self.negative[class_index, ::-1]
            total_positive = positives.sum()
            if total_positive == 0:
                values[name] = float("nan")
                continue
            true_positive = np.cumsum(positives)
            false_positive = np.cumsum(negatives)
            precision = true_positive / np.maximum(true_positive + false_positive, 1)
            recall = true_positive / total_positive
            recall_step = np.diff(np.concatenate(([0.0], recall)))
            values[name] = float(np.sum(precision * recall_step))
        foreground = [values[name] for name in self.class_names[1:] if not np.isnan(values[name])]
        return {
            "pixel_average_precision": values,
            "pixel_map_foreground": float(np.mean(foreground)) if foreground else float("nan"),
            "pixel_ap_bins": self.bins,
            "pixel_ap_note": (
                "Histogram-approximated one-vs-rest pixel AP; not instance-detection mAP."
            ),
        }
