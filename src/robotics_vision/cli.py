from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmarking import run_benchmark
from .comparison import compare_results
from .config import load_config
from .data import prepare_dataset, validate_dataset
from .domain import run_domain_evaluation
from .evaluation import run_evaluation
from .exporting import run_export
from .runlog import json_safe
from .study import run_study
from .training import run_training


def _config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a known configuration key; repeatable",
    )
    return parser


def _print(value: Any) -> None:
    print(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False))


def prepare_main() -> None:
    parser = _config_parser("Generate the deterministic warehouse segmentation dataset")
    parser.add_argument("--force", action="store_true", help="Replace generated data at data.root")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    path = prepare_dataset(config, force=args.force)
    _print({"manifest": str(path), **validate_dataset(path, check_hashes=True)})


def validate_main() -> None:
    parser = _config_parser("Validate files, masks, labels, splits, and optional checksums")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    _print(validate_dataset(config["data"]["manifest"], check_hashes=not args.skip_hashes))


def train_main() -> None:
    parser = _config_parser("Train or resume a semantic-segmentation experiment")
    parser.add_argument("--resume", help="Checkpoint path (overrides training.resume)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing named run")
    args = parser.parse_args()
    overrides = list(args.set)
    if args.resume:
        overrides.append(f"training.resume={json.dumps(args.resume)}")
    _print(run_training(load_config(args.config, overrides), overwrite=args.overwrite))


def evaluate_main() -> None:
    parser = _config_parser("Evaluate a checkpoint and create confusion/error analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    _print(run_evaluation(config, args.checkpoint, args.output_dir))


def export_main() -> None:
    parser = _config_parser("Export TorchScript, ONNX, and optionally TensorRT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--no-onnx-verify", action="store_true")
    parser.add_argument("--tensorrt", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    _print(
        run_export(
            config,
            args.checkpoint,
            args.output_dir,
            export_onnx=not args.no_onnx,
            verify_onnx=not args.no_onnx_verify,
            export_tensorrt=args.tensorrt,
        )
    )


def benchmark_main() -> None:
    parser = _config_parser("Benchmark accuracy, latency, throughput, and artifact size")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend", required=True, choices=["pytorch", "torchscript", "onnxruntime"]
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    _print(run_benchmark(config, args.model, args.backend, args.output))


def compare_main() -> None:
    parser = argparse.ArgumentParser(description="Compare two or more measured benchmark files")
    parser.add_argument("--result", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    _print(compare_results(args.result, Path(args.output_prefix)))


def domain_main() -> None:
    parser = _config_parser("Evaluate paired synthetic domain shifts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shift-seed", type=int, default=2026)
    args = parser.parse_args()
    report = run_domain_evaluation(
        load_config(args.config, args.set), args.checkpoint, args.output, args.shift_seed,
    )
    _print({"output": args.output, "conditions": len(report["conditions"]),
            "mean_shifted_iou": report["mean_shifted_iou"]})


def study_main() -> None:
    parser = argparse.ArgumentParser(description="Run a predeclared augmentation study")
    parser.add_argument("--study", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run_study(args.study, args.output_dir)
    _print({"output": args.output_dir, "runs": len(report["records"]),
            "failed_runs": report["failed_runs"]})
    if report["failed_runs"]:
        raise SystemExit(1)
