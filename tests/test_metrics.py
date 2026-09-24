from __future__ import annotations

import pytest
import torch

from robotics_vision.metrics import SegmentationMetrics, StreamingPixelAP


def test_confusion_metrics_known_example() -> None:
    metric = SegmentationMetrics(2, ["background", "object"])
    truth = torch.tensor([[0, 0], [1, 1]])
    prediction = torch.tensor([[0, 1], [1, 1]])
    metric.update(prediction, truth)
    result = metric.compute()
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["pixel_accuracy"] == pytest.approx(0.75)
    assert result["per_class"]["object"]["precision"] == pytest.approx(2 / 3)
    assert result["per_class"]["object"]["recall"] == pytest.approx(1.0)
    assert result["per_class"]["object"]["iou"] == pytest.approx(2 / 3)


def test_pixel_ap_is_one_for_separable_predictions() -> None:
    meter = StreamingPixelAP(2, ["background", "object"], bins=16)
    logits = torch.tensor([[[[8.0, -8.0]], [[-8.0, 8.0]]]])
    target = torch.tensor([[[0, 1]]])
    meter.update(logits, target)
    result = meter.compute()
    assert result["pixel_average_precision"]["object"] == pytest.approx(1.0)
    assert result["pixel_map_foreground"] == pytest.approx(1.0)
