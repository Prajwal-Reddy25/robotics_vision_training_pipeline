from __future__ import annotations

import json
from pathlib import Path

from robotics_vision.comparison import compare_results


def test_compare_requires_and_writes_two_measured_results(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"result_{index}.json"
        path.write_text(
            json.dumps({"backend": "pytorch", "device": "cpu", "mean_iou": 0.1 + index}),
            encoding="utf-8",
        )
        paths.append(path)
    output = compare_results([f"first={paths[0]}", f"second={paths[1]}"], tmp_path / "comparison")
    assert Path(output["json"]).is_file()
    assert Path(output["markdown"]).is_file()
    assert len(output["results"]) == 2
