.PHONY: install test lint prepare train evaluate export benchmark clean-generated

install:
	python3 -m pip install -e ".[all]"

test:
	pytest

lint:
	ruff check .

prepare:
	rv-prepare --config configs/tiny_unet.yaml

train:
	rv-train --config configs/tiny_unet.yaml

evaluate:
	rv-evaluate --config configs/tiny_unet.yaml --checkpoint runs/tiny_unet/checkpoints/best.pt

export:
	rv-export --config configs/tiny_unet.yaml --checkpoint runs/tiny_unet/checkpoints/best.pt --output-dir artifacts/tiny_unet

benchmark:
	rv-benchmark --config configs/tiny_unet.yaml --model artifacts/tiny_unet/model.onnx --backend onnxruntime --output runs/tiny_unet/benchmark_onnxruntime.json

clean-generated:
	@echo "Generated data/runs/artifacts are ignored. Remove them manually after verifying paths."
