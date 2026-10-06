"""Execute repeat training, deployment parity, and optional real CUDA checks.

Run with the repository installed; commands and outputs are release evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import traceback
from pathlib import Path

import torch

from robotics_vision.benchmarking import _sha256, run_benchmark
from robotics_vision.config import load_config
from robotics_vision.exporting import run_export
from robotics_vision.runlog import environment_metadata, write_json
from robotics_vision.training import load_checkpoint, run_training


def finite_tree(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return True


def equal_state(first, second) -> bool:
    return all(torch.equal(first[key].cpu(), second[key].cpu()) for key in first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--deployment-dir", help="CPU export directory (required with --gpu)")
    args = parser.parse_args()
    study = json.loads(Path(args.study).read_text())
    if not study["complete"] or study["failed_runs"] or len(study["records"]) != 42:
        raise ValueError("Release validation requires a complete successful 42-run study")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = {"environment": environment_metadata(), "study_sha256": _sha256(Path(args.study)),
              "mode": "cuda" if args.gpu else "cpu", "models": [], "gpu_training": [],
              "strict_check_failures": []}
    if args.gpu:
        if not args.deployment_dir:
            raise ValueError("GPU validation requires --deployment-dir with checked CPU exports")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU validation requires actual CUDA access")
        report["gpu_name"] = torch.cuda.get_device_name(0)
        for architecture in ("tiny_unet", "lite_fcn"):
            for amp in (False, True):
                config = load_config("configs/smoke.yaml")
                name = f"{architecture}_{'amp' if amp else 'fp32'}"
                config["experiment"] = {"name": name, "output_dir": str(out / "training")}
                config["model"]["architecture"] = architecture
                config["training"].update(device="cuda", amp=amp, epochs=2)
                trained = run_training(config)
                final = load_checkpoint(trained["checkpoint"])
                assert finite_tree(final["model_state"]) and finite_tree(final["optimizer_state"])
                if amp:
                    assert final["scaler_state"]["scale"] > 0
                resumed_config = copy.deepcopy(config)
                resumed_config["experiment"]["name"] += "_resumed"
                resumed_config["training"]["resume"] = str(
                    Path(trained["run_dir"]) / "checkpoints" / "epoch_000.pt"
                )
                resumed = run_training(resumed_config)
                recovered = load_checkpoint(resumed["checkpoint"])
                equal = equal_state(final["model_state"], recovered["model_state"])
                max_delta = max((final["model_state"][key] - recovered["model_state"][key])
                                .abs().max().item() for key in final["model_state"])
                try:
                    assert equal, f"{name}: CUDA resume model state is not bitwise identical"
                    assert final["scheduler_state"] == recovered["scheduler_state"]
                    assert final["scaler_state"] == recovered["scaler_state"]
                    assert torch.equal(final["data_loader_generator_state"],
                                       recovered["data_loader_generator_state"])
                except AssertionError as exc:
                    # Preserve the strict failure, continue collecting, exit nonzero below.
                    report["strict_check_failures"].append(str(exc))
                    (out / f"{name}.strict_failure.txt").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                report["gpu_training"].append({
                    "architecture": architecture, "amp": amp, "config": config,
                    "training": trained, "resumed": resumed, "finite_model_optimizer": True,
                    "scaler_state": final["scaler_state"], "bitwise_resume": equal,
                    "max_resume_state_delta": max_delta,
                })
                write_json(out / "validation.json", report)
    for record in study["records"]:
        if record["arm"] != "baseline" or record["seed"] != 42:
            continue
        config = copy.deepcopy(record["config"])
        architecture = record["architecture"]
        checkpoint = record["training"]["best_checkpoint"]
        item = {"architecture": architecture, "checkpoint_sha256": _sha256(Path(checkpoint))}
        if args.gpu:
            config["training"]["device"] = "cuda"
            cpu_export = Path(args.deployment_dir) / architecture
            torchscript = cpu_export / "model.torchscript.pt"
            item["benchmarks"] = [
                run_benchmark(config, checkpoint, "pytorch", out / f"{architecture}.eager.json"),
                run_benchmark(config, torchscript, "torchscript", out / f"{architecture}.ts.json"),
            ]
            eager_iou = item["benchmarks"][0]["mean_iou"]
            item["cpu_gpu_miou_delta"] = eager_iou - record["benchmark"]["mean_iou"]
            assert abs(item["cpu_gpu_miou_delta"]) < 0.001
        else:
            export = run_export(config, checkpoint, out / architecture)
            item["export"] = export
            item["benchmarks"] = [
                run_benchmark(config, export["artifacts"]["torchscript"]["path"],
                              "torchscript", out / f"{architecture}.ts.json"),
                run_benchmark(config, export["artifacts"]["onnx"]["path"],
                              "onnxruntime", out / f"{architecture}.ort.json"),
            ]
            for benchmark in item["benchmarks"]:
                assert abs(benchmark["mean_iou"] - record["benchmark"]["mean_iou"]) < 0.001
            config["experiment"] = {"name": architecture, "output_dir": str(out / "repeat")}
            repeated = run_training(config)
            first = load_checkpoint(record["training"]["checkpoint"])
            second = load_checkpoint(repeated["checkpoint"])
            item["repeat_training_bitwise_equal"] = equal_state(
                first["model_state"], second["model_state"]
            )
            assert item["repeat_training_bitwise_equal"]
            item["repeat_training"] = repeated
        report["models"].append(item)
        write_json(out / "validation.json", report)
    print(json.dumps({"output": str(out / "validation.json"), "mode": report["mode"],
                      "models": len(report["models"]), "gpu_runs": len(report["gpu_training"])}))
    if report["strict_check_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
