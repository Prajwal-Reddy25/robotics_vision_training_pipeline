from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import validate_config
from .models import build_model
from .runlog import write_json
from .training import load_checkpoint, validate_checkpoint_model_config

EQUIVALENCE_RTOL = 1e-4
EQUIVALENCE_ATOL = 1e-5
MIN_ARGMAX_AGREEMENT = 0.999


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path) -> dict[str, Any]:
    related = sorted(path.parent.glob(f"{path.name}.data*"))
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size + sum(item.stat().st_size for item in related),
        "sha256": _sha256(path),
        "related_files": [
            {"path": str(item), "size_bytes": item.stat().st_size, "sha256": _sha256(item)}
            for item in related
        ],
    }


def _comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Export output shape mismatch: {reference.shape} != {candidate.shape}")
    finite = bool(np.isfinite(candidate).all())
    record = {
        "output_shape": list(candidate.shape),
        "finite": finite,
        "max_abs_error": float(np.max(np.abs(reference - candidate))),
        "mean_abs_error": float(np.mean(np.abs(reference - candidate))),
        "argmax_agreement": float(
            np.mean(np.argmax(reference, axis=1) == np.argmax(candidate, axis=1))
        ),
        "allclose": bool(
            finite
            and np.allclose(reference, candidate, rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL)
        ),
    }
    if not record["allclose"] or record["argmax_agreement"] < MIN_ARGMAX_AGREEMENT:
        raise RuntimeError(f"Export equivalence check failed: {record}")
    return record


def _verification_inputs(image_size: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(20260924)
    return [
        torch.rand(1, 3, image_size, image_size, generator=generator),
        torch.rand(2, 3, image_size + 1, image_size + 3, generator=generator),
    ]


def run_export(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    export_onnx: bool = True,
    verify_onnx: bool = True,
    export_tensorrt: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    validate_checkpoint_model_config(config, checkpoint)
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(config["data"]["image_size"])
    verification_inputs = _verification_inputs(image_size)
    example = verification_inputs[0]

    torchscript_path = destination / "model.torchscript.pt"
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, torchscript_path)
    script_cases = []
    with torch.inference_mode():
        for case in verification_inputs:
            reference_output = model(case).numpy()
            script_output = scripted(case).numpy()
            script_cases.append(
                {"input_shape": list(case.shape), **_comparison(reference_output, script_output)}
            )
    artifacts: dict[str, Any] = {
        "torchscript": {
            **_file_metadata(torchscript_path),
            "verification_cases": script_cases,
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
            onnx_cases = []
            with torch.inference_mode():
                for case in verification_inputs:
                    reference_numpy = model(case).numpy()
                    output = session.run(["logits"], {"image": case.numpy()})[0]
                    onnx_cases.append(
                        {
                            "input_shape": list(case.shape),
                            **_comparison(reference_numpy, output),
                        }
                    )
            onnx_record.update(
                {
                    "verified_with": "onnxruntime/CPUExecutionProvider",
                    "verification_cases": onnx_cases,
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
        "format_version": 2,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256(Path(checkpoint_path)),
        "source_epoch": int(checkpoint["epoch"]),
        "model": config["model"],
        "input": {"name": "image", "dtype": "float32", "layout": "NCHW", "range": [0, 1]},
        "output": {"name": "logits", "layout": "NCHW"},
        "equivalence_tolerances": {
            "rtol": EQUIVALENCE_RTOL,
            "atol": EQUIVALENCE_ATOL,
            "minimum_argmax_agreement": MIN_ARGMAX_AGREEMENT,
        },
        "artifacts": artifacts,
    }
    write_json(destination / "metadata.json", metadata)
    return metadata


def load_export_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
