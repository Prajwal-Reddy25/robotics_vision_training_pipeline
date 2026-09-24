from __future__ import annotations

import copy
import hashlib
import json
import math
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


def _reject_unknown_keys(
    document: dict[str, Any], template: dict[str, Any], prefix: str = ""
) -> None:
    for key, value in document.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in template:
            raise ConfigError(f"Unknown configuration key: {path}")
        if isinstance(template[key], dict) and not isinstance(value, dict):
            raise ConfigError(f"Configuration section must be a mapping: {path}")
        if isinstance(value, dict):
            if not isinstance(template[key], dict):
                raise ConfigError(f"Configuration value must not be a mapping: {path}")
            _reject_unknown_keys(value, template[key], path)


def _require_int(value: Any, path: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer >= {minimum}")
    return value


def _require_float(value: Any, path: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise ConfigError(f"{path} must be finite and >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ConfigError(f"{path} must be <= {maximum}")
    return numeric


def _require_bool(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false")


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
    required = {
        "experiment",
        "data",
        "augmentation",
        "model",
        "training",
        "evaluation",
        "benchmark",
    }
    missing = required - config.keys()
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")
    experiment_name = config["experiment"]["name"]
    if (
        not isinstance(experiment_name, str)
        or not experiment_name
        or Path(experiment_name).name != experiment_name
        or experiment_name in {".", ".."}
    ):
        raise ConfigError("experiment.name must be a non-empty single path component")
    if (
        not isinstance(config["experiment"]["output_dir"], str)
        or not config["experiment"]["output_dir"]
    ):
        raise ConfigError("experiment.output_dir must not be empty")

    data = config["data"]
    if not isinstance(data["root"], str) or not isinstance(data["manifest"], str):
        raise ConfigError("data.root and data.manifest must be path strings")
    if Path(data["root"]).resolve() != Path(data["manifest"]).parent.resolve():
        raise ConfigError("data.manifest must be directly inside data.root")
    split = data["split"]
    if set(split) != {"train", "val", "test"}:
        raise ConfigError("data.split must contain exactly train, val, and test")
    split_values = {
        name: _require_float(value, f"data.split.{name}", 0.0, 1.0) for name, value in split.items()
    }
    if any(value == 0 for value in split_values.values()):
        raise ConfigError("all data.split ratios must be positive")
    if abs(sum(split_values.values()) - 1.0) > 1e-8:
        raise ConfigError("data.split ratios must sum to 1.0")
    num_samples = _require_int(data["num_samples"], "data.num_samples", 3)
    raw_counts = {name: num_samples * ratio for name, ratio in split_values.items()}
    counts = {name: int(value) for name, value in raw_counts.items()}
    for name in sorted(split_values, key=lambda key: raw_counts[key] - counts[key], reverse=True):
        if sum(counts.values()) == num_samples:
            break
        counts[name] += 1
    if any(count == 0 for count in counts.values()):
        raise ConfigError("data split ratios and num_samples must produce non-empty splits")
    _require_int(data["image_size"], "data.image_size", 16)
    _require_int(data["seed"], "data.seed", 0)
    _require_int(data["num_workers"], "data.num_workers", 0)
    _require_bool(data["validate_hashes"], "data.validate_hashes")

    augmentation = config["augmentation"]
    _require_bool(augmentation["enabled"], "augmentation.enabled")
    _require_float(
        augmentation["horizontal_flip_probability"],
        "augmentation.horizontal_flip_probability",
        0.0,
        1.0,
    )
    _require_float(augmentation["brightness"], "augmentation.brightness", 0.0, 1.0)
    _require_float(augmentation["contrast"], "augmentation.contrast", 0.0, 1.0)
    _require_float(augmentation["gaussian_noise_std"], "augmentation.gaussian_noise_std", 0.0, 1.0)

    if config["model"]["architecture"] not in {"tiny_unet", "lite_fcn"}:
        raise ConfigError("model.architecture must be tiny_unet or lite_fcn")
    if _require_int(config["model"]["num_classes"], "model.num_classes", 1) != 4:
        raise ConfigError("the bundled dataset has exactly 4 classes")
    _require_int(config["model"]["base_channels"], "model.base_channels", 1)

    training = config["training"]
    _require_int(training["epochs"], "training.epochs", 1)
    _require_int(training["batch_size"], "training.batch_size", 1)
    _require_float(training["learning_rate"], "training.learning_rate", 0.0)
    if float(training["learning_rate"]) == 0:
        raise ConfigError("training.learning_rate must be > 0")
    _require_float(training["weight_decay"], "training.weight_decay", 0.0)
    _require_int(training["seed"], "training.seed", 0)
    _require_bool(training["deterministic"], "training.deterministic")
    _require_bool(training["amp"], "training.amp")
    _require_int(training["save_every"], "training.save_every", 1)
    if training["resume"] is not None and not isinstance(training["resume"], str):
        raise ConfigError("training.resume must be null or a path string")
    if training["device"] not in {"auto", "cpu", "cuda"}:
        raise ConfigError("training.device must be auto, cpu, or cuda")

    evaluation = config["evaluation"]
    if evaluation["split"] not in {"train", "val", "test"}:
        raise ConfigError("evaluation.split must be train, val, or test")
    _require_int(evaluation["ap_bins"], "evaluation.ap_bins", 2)
    _require_int(evaluation["save_worst"], "evaluation.save_worst", 0)

    benchmark = config["benchmark"]
    _require_int(benchmark["warmup"], "benchmark.warmup", 0)
    _require_int(benchmark["iterations"], "benchmark.iterations", 1)
    _require_int(benchmark["batch_size"], "benchmark.batch_size", 1)
    _require_int(benchmark["threads"], "benchmark.threads", 1)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ConfigError("Configuration root must be a mapping")
    _reject_unknown_keys(document, DEFAULT_CONFIG)
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
