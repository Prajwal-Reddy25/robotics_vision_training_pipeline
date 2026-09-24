from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runlog import write_json

COMPARISON_FIELDS = [
    "backend",
    "device",
    "parameters",
    "model_size_bytes",
    "pixel_accuracy",
    "mean_iou",
    "mean_precision",
    "mean_recall",
    "pixel_map_foreground",
    "latency_ms_per_image_mean",
    "throughput_images_per_second",
]


def compare_results(entries: list[str], output_prefix: str | Path) -> dict[str, Any]:
    if len(entries) < 2:
        raise ValueError("At least two label=benchmark.json entries are required")
    rows = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Comparison entry must be label=path: {entry}")
        label, path_text = entry.split("=", 1)
        with Path(path_text).open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        rows.append({"label": label, **{field: result.get(field) for field in COMPARISON_FIELDS}})
    output_prefix = Path(output_prefix)
    payload = {"results": rows, "source_entries": entries}
    write_json(output_prefix.with_suffix(".json"), payload)
    headers = ["label", *COMPARISON_FIELDS]
    lines = [
        "# Model benchmark comparison",
        "",
        "All values below are copied from measured benchmark JSON files; no values are inferred.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    markdown_path = output_prefix.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": str(output_prefix.with_suffix(".json")),
        "markdown": str(markdown_path),
        **payload,
    }
