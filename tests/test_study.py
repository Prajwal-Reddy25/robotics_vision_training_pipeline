from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from robotics_vision.config import ConfigError, load_config
from robotics_vision.domain import degradation
from robotics_vision.engine import evaluate_model
from robotics_vision.study import load_study, run_study


def _study_file(tmp_path: Path) -> Path:
    config = load_config("configs/smoke.yaml")
    root = tmp_path / "data"
    config["data"].update(root=str(root), manifest=str(root / "manifest.json"))
    config["benchmark"].update(warmup=0, iterations=2)
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(config))
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump({
        "version": 1, "base_config": "base.yaml", "seeds": [9],
        "architectures": ["lite_fcn"], "shift_seed": 2026,
        "arms": {"baseline": {}, "blur": {"blur_radius": 1.4}},
    }))
    return path


def test_degradation_retains_improvements_and_handles_zero_nominal() -> None:
    assert degradation(0.8, 0.6)["absolute_drop"] == pytest.approx(0.2)
    assert degradation(0.8, 0.6)["relative_drop"] == pytest.approx(0.25)
    assert degradation(0.6, 0.8)["absolute_drop"] == pytest.approx(-0.2)
    assert degradation(0, 0.1)["relative_drop"] is None


def test_study_integration_is_paired_and_reproducible(tmp_path: Path) -> None:
    study = _study_file(tmp_path)
    first = run_study(study, tmp_path / "first")
    repeat = run_study(study, tmp_path / "repeat")
    assert first["complete"] and first["failed_runs"] == 0
    assert len(first["records"]) == 2
    for original, repeated in zip(first["records"], repeat["records"], strict=True):
        assert len(original["domain"]["conditions"]) == 13
        assert original["domain"]["conditions"] == repeated["domain"]["conditions"]
        assert original["domain"]["mean_shifted_iou"] == repeated["domain"]["mean_shifted_iou"]
        assert len(original["benchmark"]["latency_ms_samples"]) == 2
    for name in first["records"][0]["domain"]["conditions"]:
        assert (first["records"][0]["domain"]["conditions"][name]["content_sha256"]
                == first["records"][1]["domain"]["conditions"][name]["content_sha256"])
    assert first["summaries"][0]["conditions"]["nominal"]["mean_iou"]["sample_sd"] is None
    with pytest.raises(FileExistsError):
        run_study(study, tmp_path / "first")


def test_failed_attempt_is_preserved_and_later_runs_execute(tmp_path: Path, monkeypatch) -> None:
    import robotics_vision.study as module

    original = module.run_training
    attempts = []

    def failing_once(config):
        attempts.append(config["experiment"]["name"])
        if len(attempts) == 1:
            raise FloatingPointError("injected nonfinite loss")
        return original(config)

    monkeypatch.setattr(module, "run_training", failing_once)
    report = run_study(_study_file(tmp_path), tmp_path / "failed")
    assert report["complete"] and report["failed_runs"] == 1
    assert [r["status"] for r in report["records"]] == ["failed", "success"]
    persisted = json.loads((tmp_path / "failed" / "study.json").read_text())
    assert "injected" in persisted["records"][0]["error"]
    assert len(list((tmp_path / "failed").glob("*.failure.txt"))) == 1
    assert len(report["summaries"]) == 2


@pytest.mark.parametrize("change", [{"seeds": [1, 1]}, {"shift_seed": -1},
                                   {"architectures": ["unknown"]}, {"typo": True},
                                   {"arms": {"baseline": {"blur_radius": -0.1}}}])
def test_study_rejects_invalid_configuration(tmp_path: Path, change: dict) -> None:
    path = _study_file(tmp_path)
    spec = yaml.safe_load(path.read_text())
    spec.update(copy.deepcopy(change))
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises((ValueError, ConfigError)):
        load_study(path)


def test_nonfinite_evaluation_fails_before_metric_accumulation() -> None:
    class Nonfinite(torch.nn.Module):
        def forward(self, images):
            return torch.full((1, 4, 16, 16), float("nan"))

    batch = {"image": torch.zeros(1, 3, 16, 16), "mask": torch.zeros(1, 16, 16, dtype=torch.long)}
    with pytest.raises(FloatingPointError, match="logits"):
        evaluate_model(Nonfinite(), [batch], torch.device("cpu"))
