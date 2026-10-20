from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_docs", ROOT / "scripts/check_docs.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_repository_local_links_resolve():
    report = checker.check(ROOT)
    assert not report["failures"], report["failures"]


def test_checker_handles_images_anchors_and_ignores_code_examples(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        '[Guide](docs/guide.md#same-title-1)\n<img src="docs/figure.svg">\n'
        "```md\n[example](missing.md)\n```\n[external](https://example.test)\n"
    )
    (tmp_path / "docs/guide.md").write_text("# Same title\n\n# Same title\n")
    (tmp_path / "docs/figure.svg").write_text("<svg/>")
    for name in ("CHANGELOG.md", "ROADMAP.md"):
        (tmp_path / name).write_text("")
    assert checker.check(tmp_path)["failures"] == []
    (tmp_path / "README.md").write_text("[bad anchor](docs/guide.md#absent)\n![bad](lost.png)")
    assert len(checker.check(tmp_path)["failures"]) == 2


def test_portable_constraints_and_declared_versions_match_release():
    import re

    original = (ROOT / "docs/evidence/v1_2/cpu_versions.txt").read_text().splitlines()
    expected = [line for line in original if "==" in line and not line.startswith("#")]
    constraints = (ROOT / "constraints/v1_2_cpu.txt").read_text().splitlines()
    assert [line for line in constraints if line and not line.startswith("#")] == expected
    assert not any(line.startswith("-e") for line in constraints)
    pyproject = (ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE).group(1)
    import robotics_vision

    assert robotics_vision.__version__ == declared


def test_readme_result_rows_match_frozen_three_seed_evidence():
    import json
    import statistics

    readme = (ROOT / "README.md").read_text()
    source = json.loads((ROOT / "docs/evidence/v1_1/study.json").read_text())
    real = json.loads((ROOT / "docs/evidence/v1_2/analysis.json").read_text())
    for architecture, label in [("tiny_unet", "TinyUNet"), ("lite_fcn", "LiteFCN")]:
        row = next(line for line in readme.splitlines() if line.startswith(f"| {label} |"))
        records = [
            r
            for r in source["records"]
            if r["architecture"] == architecture and r["arm"] == "baseline"
        ]
        nominal = [r["domain"]["conditions"]["nominal"]["mean_iou"] for r in records]
        shifted = [r["domain"]["mean_shifted_iou"] for r in records]
        for values in (nominal, shifted):
            assert f"{statistics.fmean(values):.4f} ± {statistics.stdev(values):.4f}" in row
        group = next(
            g
            for g in real["summaries"]
            if g["architecture"] == architecture and g["arm"] == "baseline" and g["size"] == 64
        )
        score = group["metrics"]["real_binary_miou"]
        assert f"{score['mean']:.4f} ± {score['sample_sd']:.4f}" in row
