"""Repeat real metrics with reproduced checkpoints; check real-input export parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from robotics_vision.models import build_model
from robotics_vision.real_evaluation import evaluate_real_cases, letterbox, load_cases
from robotics_vision.runlog import environment_metadata, write_json
from robotics_vision.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/v1_2_real/study.json")
    parser.add_argument("--deployment", default="runs/v1_2_deployment/validation.json")
    parser.add_argument("--source", default="docs/evidence/v1_1/study.json")
    parser.add_argument("--manifest", default="data/external/coco_v1_2/manifest.json")
    parser.add_argument("--output", default="runs/v1_2_real/reproducibility.json")
    args = parser.parse_args()
    study, deployment, source = [
        json.loads(Path(p).read_text()) for p in (args.study, args.deployment, args.source)
    ]
    _, cases = load_cases(Path(args.manifest))
    torch.set_num_threads(1)
    report = {"environment": environment_metadata(), "models": []}
    for exported in deployment["models"]:
        architecture = exported["architecture"]
        frozen = next(
            r
            for r in source["records"]
            if r["architecture"] == architecture and r["arm"] == "baseline" and r["seed"] == 42
        )
        original = load_checkpoint(frozen["training"]["best_checkpoint"])
        repeated = load_checkpoint(exported["repeat_training"]["best_checkpoint"])
        equal = all(
            torch.equal(original["model_state"][key], repeated["model_state"][key])
            for key in original["model_state"]
        )
        if not equal:
            raise ValueError("Repeated validation-selected model state changed")
        model = build_model(frozen["config"]["model"]).eval()
        model.load_state_dict(repeated["model_state"])
        measured = evaluate_real_cases(model, cases, 64)
        reference = next(
            r
            for r in study["records"]
            if r["architecture"] == architecture
            and r["arm"] == "baseline"
            and r["seed"] == 42
            and r["size"] == 64
        )
        equal_metrics = all(
            measured[k] == reference["measured"][k] for k in measured if k != "preprocess_ms_mean"
        )
        if not equal_metrics:
            raise ValueError("Native real-data metrics failed exact reproduction")
        item = {
            "architecture": architecture,
            "best_weights_bitwise_equal": equal,
            "native_real_metrics_exactly_equal": equal_metrics,
            "repeated_dataset_evaluations": 1,
            "samples": len(cases),
            "exports": [],
        }
        artifacts = exported["export"]["artifacts"]
        script = torch.jit.load(artifacts["torchscript"]["path"]).eval()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            artifacts["onnx"]["path"], options, providers=["CPUExecutionProvider"]
        )
        for size in (64, 128):
            image, _ = letterbox(cases[0]["image"], size)
            with torch.inference_mode():
                expected, scripted = model(image).numpy(), script(image).numpy()
            onnx = session.run(None, {session.get_inputs()[0].name: image.numpy()})[0]
            for backend, output in [("torchscript", scripted), ("onnxruntime", onnx)]:
                error = float(np.max(np.abs(output - expected)))
                agreement = float(np.mean(output.argmax(1) == expected.argmax(1)))
                if error > 5e-4 or agreement < 0.999:
                    raise ValueError(f"{backend} real-input parity failed")
                item["exports"].append(
                    {
                        "backend": backend,
                        "size": size,
                        "max_abs_logit_error": error,
                        "argmax_agreement": agreement,
                        "image_id": cases[0]["id"],
                    }
                )
        report["models"].append(item)
        write_json(args.output, report)
    print({"repeated_dataset_evaluations": 2, "export_parity_comparisons": 8})


if __name__ == "__main__":
    main()
