from __future__ import annotations

import copy
from pathlib import Path

import torch

from robotics_vision.config import DEFAULT_CONFIG
from robotics_vision.evaluation import run_evaluation
from robotics_vision.exporting import run_export
from robotics_vision.training import load_checkpoint, run_training


def _smoke_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    root = tmp_path / "data"
    config["experiment"] = {"name": "smoke", "output_dir": str(tmp_path / "runs")}
    config["data"].update(
        {
            "root": str(root),
            "manifest": str(root / "manifest.json"),
            "image_size": 16,
            "num_samples": 12,
            "split": {"train": 0.5, "val": 0.25, "test": 0.25},
            "num_workers": 0,
        }
    )
    config["model"].update({"architecture": "lite_fcn", "base_channels": 2})
    config["training"].update(
        {"epochs": 1, "batch_size": 3, "device": "cpu", "amp": False, "save_every": 1}
    )
    config["evaluation"].update({"ap_bins": 8, "save_worst": 1})
    return config


def test_train_resume_evaluate_and_torchscript_export(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    config = _smoke_config(tmp_path)
    first = run_training(config)
    checkpoint = load_checkpoint(first["checkpoint"])
    assert checkpoint["epoch"] == 0
    assert checkpoint["data_loader_generator_state"] is not None

    resumed_config = copy.deepcopy(config)
    resumed_config["training"]["epochs"] = 2
    resumed_config["training"]["resume"] = first["checkpoint"]
    resumed = run_training(resumed_config)
    assert load_checkpoint(resumed["checkpoint"])["epoch"] == 1

    evaluation = run_evaluation(resumed_config, resumed["best_checkpoint"])
    assert evaluation["samples"] == 3
    assert len(evaluation["confusion_matrix"]) == 4
    export = run_export(
        resumed_config,
        resumed["best_checkpoint"],
        tmp_path / "artifacts",
        export_onnx=False,
    )
    assert Path(export["artifacts"]["torchscript"]["path"]).is_file()
    assert export["artifacts"]["torchscript"]["argmax_agreement"] == 1.0
