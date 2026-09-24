from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from robotics_vision.benchmarking import run_benchmark
from robotics_vision.config import DEFAULT_CONFIG
from robotics_vision.evaluation import run_evaluation
from robotics_vision.exporting import run_export
from robotics_vision.reproducibility import capture_rng_state, restore_rng_state
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
        {"epochs": 2, "batch_size": 3, "device": "cpu", "amp": False, "save_every": 1}
    )
    config["evaluation"].update({"ap_bins": 8, "save_worst": 1})
    config["benchmark"].update({"warmup": 1, "iterations": 3, "threads": 1})
    return config


def test_exact_resume_evaluate_export_and_benchmark(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    config = _smoke_config(tmp_path)
    uninterrupted = run_training(config)
    checkpoint = load_checkpoint(uninterrupted["checkpoint"])
    assert checkpoint["epoch"] == 1
    interrupted_checkpoint = Path(uninterrupted["run_dir"]) / "checkpoints" / "epoch_000.pt"

    resumed_config = copy.deepcopy(config)
    resumed_config["experiment"]["name"] = "resumed"
    resumed_config["training"]["resume"] = str(interrupted_checkpoint)
    resumed = run_training(resumed_config)
    resumed_checkpoint = load_checkpoint(resumed["checkpoint"])
    assert resumed_checkpoint["epoch"] == 1
    assert all(
        torch.equal(checkpoint["model_state"][key], resumed_checkpoint["model_state"][key])
        for key in checkpoint["model_state"]
    )
    assert torch.equal(
        checkpoint["data_loader_generator_state"],
        resumed_checkpoint["data_loader_generator_state"],
    )

    evaluation = run_evaluation(resumed_config, resumed["best_checkpoint"])
    assert evaluation["samples"] == 3
    assert len(evaluation["confusion_matrix"]) == 4
    onnx_available = (
        importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("onnxruntime") is not None
    )
    export = run_export(
        resumed_config,
        resumed["best_checkpoint"],
        tmp_path / "artifacts",
        export_onnx=onnx_available,
    )
    assert Path(export["artifacts"]["torchscript"]["path"]).is_file()
    assert len(export["artifacts"]["torchscript"]["verification_cases"]) == 2
    assert all(
        case["allclose"] for case in export["artifacts"]["torchscript"]["verification_cases"]
    )
    if onnx_available:
        assert len(export["artifacts"]["onnx"]["verification_cases"]) == 2
        assert all(case["allclose"] for case in export["artifacts"]["onnx"]["verification_cases"])

    benchmark = run_benchmark(
        resumed_config,
        export["artifacts"]["torchscript"]["path"],
        "torchscript",
        tmp_path / "benchmark.json",
    )
    assert benchmark["input_shape"] == [1, 3, 16, 16]
    assert benchmark["latency_ms_per_batch_min"] > 0
    assert benchmark["model_sha256"] == export["artifacts"]["torchscript"]["sha256"]
    assert benchmark["cuda_max_memory_allocated_bytes"] is None


def test_resume_rejects_changed_schedule(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    config = _smoke_config(tmp_path)
    result = run_training(config)
    changed = copy.deepcopy(config)
    changed["experiment"]["name"] = "changed"
    changed["training"]["epochs"] = 3
    changed["training"]["resume"] = result["checkpoint"]
    try:
        run_training(changed)
    except ValueError as exc:
        assert "training.epochs" in str(exc)
    else:
        raise AssertionError("Changed resume schedule was accepted")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rng_restore_accepts_checkpoint_state_mapped_to_cuda() -> None:
    state = capture_rng_state()
    mapped_state = dict(state)
    mapped_state["torch"] = state["torch"].cuda()
    mapped_state["cuda"] = [item.cuda() for item in state["cuda"]]

    restore_rng_state(mapped_state)

    assert torch.equal(torch.get_rng_state(), state["torch"])
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), state["cuda"], strict=True)
    )
