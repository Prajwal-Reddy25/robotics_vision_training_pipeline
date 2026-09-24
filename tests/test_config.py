from __future__ import annotations

from pathlib import Path

import pytest

from robotics_vision.config import ConfigError, config_fingerprint, load_config


def test_load_config_merges_defaults_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("model:\n  architecture: lite_fcn\n", encoding="utf-8")
    config = load_config(path, ["training.epochs=7", "training.deterministic=false"])
    assert config["model"]["architecture"] == "lite_fcn"
    assert config["training"]["epochs"] == 7
    assert config["training"]["deterministic"] is False
    assert len(config_fingerprint(config)) == 12


def test_unknown_override_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown configuration"):
        load_config(path, ["training.not_a_setting=1"])


def test_invalid_split_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "data:\n  split:\n    train: 0.8\n    val: 0.1\n    test: 0.2\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)
