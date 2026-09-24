from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": "tiny_unet", "output_dir": "runs"},
    "data": {
        "root": "data/processed/warehouse_shapes",
        "manifest": "data/processed/warehouse_shapes/manifest.json",
        "image_size": 64,
        "num_samples": 180,
        "seed": 17,
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "num_workers": 0,
        "validate_hashes": True,
    },
    "augmentation": {
        "enabled": True,
        "horizontal_flip_probability": 0.5,
        "brightness": 0.15,
        "contrast": 0.15,
        "gaussian_noise_std": 0.02,
    },
    "model": {"architecture": "tiny_unet", "num_classes": 4, "base_channels": 8},
    "training": {
        "epochs": 15,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "seed": 42,
        "device": "auto",
        "deterministic": True,
        "amp": False,
        "resume": None,
        "save_every": 1,
    },
    "evaluation": {"split": "test", "ap_bins": 256, "save_worst": 5},
    "benchmark": {"warmup": 5, "iterations": 30, "batch_size": 1, "threads": 1},
}


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise ConfigError(f"Unknown configuration path: {dotted_key}")
        cursor = cursor[part]
    if parts[-1] not in cursor:
        raise ConfigError(f"Unknown configuration key: {dotted_key}")
    cursor[parts[-1]] = value


def validate_config(config: dict[str, Any]) -> None:
    required = {"experiment", "data", "augmentation", "model", "training", "evaluation"}
    missing = required - config.keys()
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")
    split = config["data"]["split"]
    if set(split) != {"train", "val", "test"}:
        raise ConfigError("data.split must contain exactly train, val, and test")
    if abs(sum(float(value) for value in split.values()) - 1.0) > 1e-8:
        raise ConfigError("data.split ratios must sum to 1.0")
    if any(float(value) <= 0 for value in split.values()):
        raise ConfigError("all data.split ratios must be positive")
    if int(config["data"]["num_samples"]) < 3:
        raise ConfigError("data.num_samples must be at least 3")
    if int(config["data"]["image_size"]) < 16:
        raise ConfigError("data.image_size must be at least 16")
    if config["model"]["architecture"] not in {"tiny_unet", "lite_fcn"}:
        raise ConfigError("model.architecture must be tiny_unet or lite_fcn")
    if int(config["model"]["num_classes"]) != 4:
        raise ConfigError("the bundled dataset has exactly 4 classes")
    if int(config["training"]["epochs"]) < 1:
        raise ConfigError("training.epochs must be positive")
    if int(config["training"]["batch_size"]) < 1:
        raise ConfigError("training.batch_size must be positive")
    if config["training"]["device"] not in {"auto", "cpu", "cuda"}:
        raise ConfigError("training.device must be auto, cpu, or cuda")


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ConfigError("Configuration root must be a mapping")
    config = _deep_merge(DEFAULT_CONFIG, document)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must use key=value syntax: {item}")
        key, raw_value = item.split("=", 1)
        _set_nested(config, key, yaml.safe_load(raw_value))
    validate_config(config)
    return config


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
