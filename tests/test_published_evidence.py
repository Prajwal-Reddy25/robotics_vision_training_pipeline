"""Independent arithmetic and deliberately corrupted copies; never edit frozen evidence."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_published_v1_2.py"
spec = importlib.util.spec_from_file_location("published_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_standard_library_cli_from_unrelated_working_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["dataset_evaluations"] == 84
    assert report["native_image_records"] == 6720
    assert report["unresolved_creator_ids"] == [287347, 464358]
    assert report["inventory_files_verified"] == {"v1_1": 21, "v1_2": 42}


def test_independent_projections_keep_unknown_prediction_errors():
    counts = [[5, 1, 2, 0], [1, 3, 0, 1], [8, 4, 2, 0], [1, 0, 0, 1]]
    binary, shared = verifier.binary(counts), verifier.shared(counts)
    assert binary["confusion_matrix"] == [[5, 3], [10, 11]]
    assert binary["mean_iou"] == pytest.approx((5 / 18 + 11 / 24) / 2)
    assert shared["mean_iou"] == pytest.approx((5 / 9 + 3 / 6) / 2)
    assert shared["pixel_accuracy"] == pytest.approx(8 / 13)
    assert shared["per_class"]["floor"]["recall"] == pytest.approx(5 / 8)
    assert counts[2][0] == 8  # Projection did not mutate the input.


def test_rank_and_quantile_checks_have_known_answers():
    assert verifier.spearman([1, 1, 3], [9, 9, 0]) == pytest.approx(-1)
    assert verifier.spearman([1, 1, 1], [1, 2, 3]) is None
    assert verifier.percentile([0, 10], 0.95) == pytest.approx(9.5)
    assert verifier.distribution([-1.0, 0.0, 2.0])["individual_values"] == [-1.0, 0.0, 2.0]
    with pytest.raises(verifier.VerificationError, match="nonnegative integers"):
        verifier.matrix([[1, 0], [-1, 1]], 2)


@pytest.fixture
def copied_release(tmp_path):
    for version in ("v1_1", "v1_2"):
        shutil.copytree(ROOT / "docs/evidence" / version, tmp_path / "docs/evidence" / version)
    return tmp_path


def rewrite_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


def rehash(root, name):
    path = root / name
    inventory_path = root / "SHA256.json"
    inventory = json.loads(inventory_path.read_text())
    inventory[name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    rewrite_json(inventory_path, inventory)


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("metric", "real.binary.mean_iou"),
        ("missing_run", "incomplete/duplicate"),
        ("paired_delta", "paired deltas"),
        ("csv", "CSV.real_binary_miou"),
        ("creator", "missing creator"),
    ],
)
def test_consistency_rejects_rehashed_corruptions(copied_release, damage, reason):
    root = copied_release / "docs/evidence/v1_2"
    if damage in ("metric", "missing_run"):
        path = root / "study.json"
        data = json.loads(path.read_text())
        if damage == "metric":
            data["records"][0]["measured"]["binary"]["mean_iou"] += 0.01
        else:
            data["records"].pop()
        rewrite_json(path, data)
        provenance = json.loads((root / "provenance.json").read_text())
        original = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
        provenance["study.json"]["original_sha256"] = hashlib.sha256(original).hexdigest()
        rewrite_json(root / "provenance.json", provenance)
        rehash(root, "provenance.json")
    elif damage == "paired_delta":
        path = root / "analysis.json"
        data = json.loads(path.read_text())
        data["paired_deltas"][0]["deltas"]["real_binary_miou"]["mean"] += 0.1
        rewrite_json(path, data)
    elif damage == "csv":
        path = root / "real_runs.csv"
        rows = list(csv.DictReader(path.open()))
        rows[0]["real_binary_miou"] = "0.99"
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        path.write_text(stream.getvalue())
    else:
        path = root / "gallery/selection.json"
        data = json.loads(path.read_text())
        next(p for p in data["panels"] if p["id"] == 287347)["photo_included"] = True
        rewrite_json(path, data)
    rehash(root, str(path.relative_to(root)))
    with pytest.raises(verifier.VerificationError, match=reason):
        verifier.verify(copied_release)


def test_inventory_rejects_corruption_without_refreshing_hashes(copied_release):
    path = copied_release / "docs/evidence/v1_2/real_runs.csv"
    path.write_text(path.read_text().replace("0.211009", "0.911009", 1))
    with pytest.raises(verifier.VerificationError, match="SHA256 mismatch"):
        verifier.verify(copied_release)


def test_inventory_rejects_parent_path_before_reading_it(tmp_path):
    rewrite_json(tmp_path / "SHA256.json", {"../outside": {"bytes": 1, "sha256": "0" * 64}})
    with pytest.raises(verifier.VerificationError, match="unsafe relative path"):
        verifier.verify_inventory(tmp_path)


def test_optimized_interpreter_still_rejects_invalid_evidence(copied_release):
    path = copied_release / "docs/evidence/v1_2/real_runs.csv"
    path.write_bytes(path.read_bytes() + b"corruption\n")
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-O", str(SCRIPT), "--repo-root", str(copied_release)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "Published evidence verification failed" in result.stderr
