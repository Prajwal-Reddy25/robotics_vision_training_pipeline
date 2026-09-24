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


def test_mean_iou_ignores_classes_absent_from_truth_and_prediction() -> None:
    metric = SegmentationMetrics(3, ["first", "second", "absent"])
    target = torch.tensor([[0, 1]])
    metric.update(target, target)
    result = metric.compute()
    assert result["mean_iou"] == pytest.approx(1.0)
    assert result["evaluated_iou_classes"] == 2
    assert result["per_class"]["absent"]["iou"] == 0.0


def test_metrics_reject_out_of_range_prediction() -> None:
    metric = SegmentationMetrics(2, ["background", "object"])
    with pytest.raises(ValueError, match="out-of-range"):
        metric.update(torch.tensor([[0, 2]]), torch.tensor([[0, 1]]))
