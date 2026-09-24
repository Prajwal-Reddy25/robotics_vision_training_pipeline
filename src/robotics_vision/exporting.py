from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models import build_model
from .runlog import write_json
from .training import load_checkpoint


def _file_metadata(path: Path) -> dict[str, Any]:
    related = sorted(path.parent.glob(f"{path.name}.data*"))
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size + sum(item.stat().st_size for item in related),
        "related_files": [str(item) for item in related],
    }


def run_export(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    export_onnx: bool = True,
    verify_onnx: bool = True,
    export_tensorrt: bool = False,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(config["data"]["image_size"])
    example = torch.rand(1, 3, image_size, image_size)
    with torch.inference_mode():
        reference = model(example)

    torchscript_path = destination / "model.torchscript.pt"
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, torchscript_path)
    with torch.inference_mode():
        script_output = scripted(example)
    artifacts: dict[str, Any] = {
        "torchscript": {
            **_file_metadata(torchscript_path),
            "max_abs_error": float((reference - script_output).abs().max()),
            "argmax_agreement": float(
                (reference.argmax(1) == script_output.argmax(1)).float().mean()
            ),
        }
    }

    onnx_path = destination / "model.onnx"
    if export_onnx:
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError("ONNX export requires the 'export' optional dependencies") from exc
        export_options: dict[str, Any] = {
            "export_params": True,
            "opset_version": 18,
            "do_constant_folding": True,
            "input_names": ["image"],
            "output_names": ["logits"],
            "dynamic_axes": {
                "image": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch", 2: "height", 3: "width"},
            },
        }
        # New exporters default to external weights even for tiny models. Keep these
        # portfolio artifacts self-contained while retaining older PyTorch support.
        if "external_data" in inspect.signature(torch.onnx.export).parameters:
            export_options["external_data"] = False
        torch.onnx.export(model, example, onnx_path, **export_options)
        loaded = onnx.load(str(onnx_path))
        onnx.checker.check_model(loaded)
        uses_external_data = any(
            onnx.external_data_helper.uses_external_data(initializer)
            for initializer in loaded.graph.initializer
        )
        stale_sidecar = onnx_path.with_name(f"{onnx_path.name}.data")
        if not uses_external_data and stale_sidecar.is_file():
            stale_sidecar.unlink()
        default_domain = next(
            (entry.version for entry in loaded.opset_import if entry.domain in {"", "ai.onnx"}),
            None,
        )
        onnx_record: dict[str, Any] = {
            **_file_metadata(onnx_path),
            "opset": default_domain,
            "external_data": uses_external_data,
        }
        if verify_onnx:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("ONNX verification requires onnxruntime") from exc
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            output = session.run(["logits"], {"image": example.numpy()})[0]
            reference_numpy = reference.detach().numpy()
            onnx_record.update(
                {
                    "verified_with": "onnxruntime/CPUExecutionProvider",
                    "max_abs_error": float(np.max(np.abs(reference_numpy - output))),
                    "argmax_agreement": float(
                        np.mean(np.argmax(reference_numpy, axis=1) == np.argmax(output, axis=1))
                    ),
                }
            )
        artifacts["onnx"] = onnx_record

    if export_tensorrt:
        try:
            import torch_tensorrt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT export requested, but torch_tensorrt is unavailable. "
                "Use the documented trtexec ONNX path on a TensorRT host."
            ) from exc
        trt_module = torch_tensorrt.compile(
            scripted,
            inputs=[torch_tensorrt.Input(example.shape)],
            enabled_precisions={torch.float32},
        )
        tensorrt_path = destination / "model.tensorrt.ts"
        torch.jit.save(trt_module, tensorrt_path)
        artifacts["tensorrt"] = _file_metadata(tensorrt_path)

    metadata = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint_path),
        "source_epoch": int(checkpoint["epoch"]),
        "model": config["model"],
        "input": {"name": "image", "dtype": "float32", "layout": "NCHW", "range": [0, 1]},
        "output": {"name": "logits", "layout": "NCHW"},
        "artifacts": artifacts,
    }
    write_json(destination / "metadata.json", metadata)
    return metadata


def load_export_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
