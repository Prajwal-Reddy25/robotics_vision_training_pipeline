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


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("training:\n  lernng_rate: 1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="training.lernng_rate"):
        load_config(path)


def test_invalid_split_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "data:\n  split:\n    train: 0.8\n    val: 0.1\n    test: 0.2\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("benchmark:\n  iterations: 0\n", "benchmark.iterations"),
        ("augmentation:\n  horizontal_flip_probability: 2\n", "horizontal_flip_probability"),
        ("model:\n  base_channels: 0\n", "model.base_channels"),
        ("training:\n  learning_rate: -1\n", "training.learning_rate"),
        ("data:\n  num_workers: -2\n", "data.num_workers"),
        ("experiment:\n  name: ../escape\n", "experiment.name"),
        ("training: invalid\n", "must be a mapping"),
        ("augmentation:\n  blur_radius: -0.1\n", "blur_radius"),
        ("augmentation:\n  viewpoint_inset: 0.5\n", "viewpoint_inset"),
        ("augmentation:\n  occlusion_fraction: .nan\n", "occlusion_fraction"),
    ],
)
def test_invalid_operational_values_are_rejected(
    tmp_path: Path, document: str, message: str
) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)
