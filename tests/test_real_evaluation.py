"""Numerically checked projections and offline end-to-end evaluator invariants."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from PIL import Image
from test_real_data import fixture_manifest

from robotics_vision.real_evaluation import (
    binary_metrics,
    boundary_band,
    evaluate_real_cases,
    letterbox,
    load_cases,
    native_confusions,
    restore_prediction,
    restricted_metrics,
)


def test_projected_metrics_keep_unsupported_prediction_errors():
    matrix = np.array([[5, 1, 2, 0], [1, 3, 0, 1], [8, 4, 2, 0], [1, 0, 0, 1]])
    binary = binary_metrics(matrix)
    assert binary["confusion_matrix"] == [[5, 3], [10, 11]]
    assert binary["mean_iou"] == pytest.approx((5 / 18 + 11 / 24) / 2)
    shared = restricted_metrics(matrix)
    assert shared["mean_iou"] == pytest.approx((5 / 9 + 3 / 6) / 2)
    assert shared["per_class"]["floor"]["recall"] == pytest.approx(5 / 8)
    assert shared["pixel_accuracy"] == pytest.approx(8 / 13)
    assert shared["unsupported_prediction_pixels"] == 3
    assert shared["confusion_matrix"][2:] == [[0] * 4, [0] * 4]


def test_native_mapping_ignores_void_and_does_not_call_everything_obstacles():
    labels = np.array([[118, 190, 188, 1, 255]], np.uint8)
    prediction = np.array([[0, 1, 3, 0, 2]], np.uint8)
    binary, shared = native_confusions(prediction, labels)
    assert binary.tolist() == [[1, 1], [1, 1]]
    assert shared.tolist() == [[1, 1, 0, 0], [0, 0, 0, 1], [0] * 4, [0] * 4]
    assert restricted_metrics(shared)["per_class"]["storage_obstacle"]["iou"] == 0
    for broken in [np.array([[-1, 0, 1, 2, 3]]), prediction.astype(float)]:
        with pytest.raises(ValueError):
            native_confusions(broken, labels)


def test_letterbox_crops_padding_and_restores_native_predictions():
    image = Image.new("RGB", (40, 20), (255, 0, 0))
    tensor, box = letterbox(image, 16)
    assert tensor.shape == (1, 3, 16, 16) and box == (0, 4, 16, 8)
    assert tensor[0, 0, 0, 0] == pytest.approx(127 / 255)
    logits = torch.zeros(1, 4, 16, 16)
    logits[:, 3] = 1
    logits[:, 0, 4:12, :8] = 2
    logits[:, 1, 4:12, 8:] = 2
    restored = restore_prediction(logits, box, image.size)
    assert restored.shape == (20, 40)
    assert np.all(restored[:, :20] == 0) and np.all(restored[:, 20:] == 1)
    with pytest.raises(ValueError, match="finite"):
        restore_prediction(logits * float("nan"), box, image.size)
    with pytest.raises(ValueError, match="exceeds"):
        restore_prediction(logits, (0, 15, 16, 8), image.size)


def test_boundary_band_excludes_void_and_has_no_false_image_border():
    floor = np.zeros((9, 12), bool)
    floor[:, :6] = True
    valid = np.ones_like(floor)
    valid[0] = False
    band = boundary_band(floor, valid, radius=1)
    assert np.all(band[1:, 4:8]) and not band[0].any()
    assert not band[:, :4].any() and not band[:, 8:].any()
    assert not boundary_band(np.ones_like(floor), valid).any()


def test_offline_evaluation_is_native_repeatable_and_does_not_mutate_model(tmp_path):
    manifest_path, _ = fixture_manifest(tmp_path)
    _, cases = load_cases(manifest_path)
    model = torch.nn.Conv2d(3, 4, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    state = copy.deepcopy(model.state_dict())
    first = evaluate_real_cases(model, cases, 16)
    repeat = evaluate_real_cases(model, cases, 32)
    assert first["binary"] == repeat["binary"]
    assert first["per_image"] == repeat["per_image"]
    assert first["binary"]["confusion_matrix"] == [[288, 0], [95, 0]]
    assert first["shared"]["per_class"]["storage_obstacle"]["recall"] == 0
    assert first["samples"] == 1 and first["shared_pixel_coverage"] == 1
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())


def test_continuation_preserves_results_interruptions_and_rejects_changed_inputs():
    from robotics_vision.real_evaluation import continue_report

    identity = {"architecture": "lite_fcn", "arm": "baseline", "seed": 42}
    success = {**identity, "size": 64, "status": "success", "measured": {"sentinel": 3}}
    started = {**identity, "size": 128, "status": "started"}
    report = {
        "complete": False,
        "source_study_sha256": "a",
        "manifest_sha256": "b",
        "records": [copy.deepcopy(success), started],
    }
    expected = {("lite_fcn", "baseline", 42, size) for size in (64, 128)}
    with pytest.raises(ValueError, match="identical"):
        continue_report(copy.deepcopy(report), "changed", "b", expected)
    finished = continue_report(report, "a", "b", expected)
    assert finished == {("lite_fcn", "baseline", 42, 64)}
    assert report["records"][0] == success
    assert report["records"][1]["status"] == "interrupted"
    report["records"].append(copy.deepcopy(success))
    with pytest.raises(ValueError, match="Duplicate"):
        continue_report(report, "a", "b", expected)
