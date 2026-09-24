from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from .data import SegmentationDataset, make_dataloader
from .engine import evaluate_model
from .models import build_model, count_parameters
from .reproducibility import resolve_device, seed_everything
from .runlog import write_json
from .training import load_checkpoint


class InferenceModel(Protocol):
    def eval(self) -> Any: ...

    def __call__(self, image: torch.Tensor) -> torch.Tensor: ...


class OnnxRuntimeModel:
    def __init__(self, path: str | Path, threads: int = 1) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime benchmark requires onnxruntime") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def eval(self) -> OnnxRuntimeModel:
        return self

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        output = self.session.run(["logits"], {"image": image.detach().cpu().numpy()})[0]
        return torch.from_numpy(output)


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def _artifact_size(path: Path) -> int:
    related = path.parent.glob(f"{path.name}.data*")
    return path.stat().st_size + sum(item.stat().st_size for item in related)


def run_benchmark(
    config: dict[str, Any],
    model_path: str | Path,
    backend: str,
    output_path: str | Path,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    seed_everything(int(config["training"]["seed"]), bool(config["training"]["deterministic"]))
    threads = int(benchmark["threads"])
    torch.set_num_threads(threads)
    requested_device = config["training"]["device"]
    device = torch.device("cpu") if backend == "onnxruntime" else resolve_device(requested_device)
    model_path = Path(model_path)
    parameter_count: int | None = None
    if backend == "pytorch":
        checkpoint = load_checkpoint(model_path, map_location=device)
        torch_model = build_model(config["model"]).to(device)
        torch_model.load_state_dict(checkpoint["model_state"])
        model: InferenceModel = torch_model
        parameter_count = count_parameters(torch_model)
    elif backend == "torchscript":
        model = torch.jit.load(str(model_path), map_location=device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    elif backend == "onnxruntime":
        model = OnnxRuntimeModel(model_path, threads=threads)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    model.eval()

    split = config["evaluation"]["split"]
    dataset = SegmentationDataset(config["data"]["manifest"], split)
    accuracy_loader = make_dataloader(
        dataset,
        int(config["training"]["batch_size"]),
        False,
        int(config["data"]["num_workers"]),
        int(config["training"]["seed"]),
    )
    accuracy = evaluate_model(
        model,
        accuracy_loader,
        device,
        ap_bins=int(config["evaluation"]["ap_bins"]),
    ).metrics

    image_size = int(config["data"]["image_size"])
    batch_size = int(benchmark["batch_size"])
    example = torch.rand(batch_size, 3, image_size, image_size, device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(int(benchmark["warmup"])):
            model(example)
        synchronize()
        durations: list[float] = []
        for _ in range(int(benchmark["iterations"])):
            synchronize()
            started = time.perf_counter()
            model(example)
            synchronize()
            durations.append((time.perf_counter() - started) * 1000.0)

    mean_batch_ms = statistics.fmean(durations)
    result = {
        "backend": backend,
        "model_path": str(model_path),
        "model_size_bytes": _artifact_size(model_path),
        "model_size_note": "On-disk size including ONNX external-data sidecars, if present.",
        "parameters": parameter_count,
        "device": str(device),
        "torch_threads": threads,
        "batch_size": batch_size,
        "warmup_iterations": int(benchmark["warmup"]),
        "measured_iterations": int(benchmark["iterations"]),
        "latency_ms_per_batch_mean": mean_batch_ms,
        "latency_ms_per_batch_p50": _percentile(durations, 50),
        "latency_ms_per_batch_p95": _percentile(durations, 95),
        "latency_ms_per_image_mean": mean_batch_ms / batch_size,
        "throughput_images_per_second": batch_size * 1000.0 / mean_batch_ms,
        "accuracy_split": split,
        "pixel_accuracy": accuracy["pixel_accuracy"],
        "mean_iou": accuracy["mean_iou"],
        "mean_precision": accuracy["mean_precision"],
        "mean_recall": accuracy["mean_recall"],
        "pixel_map_foreground": accuracy["pixel_map_foreground"],
    }
    write_json(output_path, result)
    return result
