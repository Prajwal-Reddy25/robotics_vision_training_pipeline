"""Offline fixtures are synthetic; these tests do not constitute real-data evidence."""

from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
from PIL import Image

from robotics_vision.benchmarking import _sha256
from robotics_vision.real_data import (
    ANNOTATION_SHA256,
    RangeReader,
    decode_panoptic,
    select_records,
    validate_real_dataset,
)


def fixture_manifest(tmp_path):
    rgb, mask = tmp_path / "image.png", tmp_path / "mask.png"
    Image.new("RGB", (24, 16), (40, 80, 120)).save(rgb)
    labels = np.full((16, 24), 118, np.uint8)
    labels[:4] = 188
    labels[0, 0] = 255
    Image.fromarray(labels).save(mask)
    manifest = {
        "format_version": 1,
        "annotation_sha256": ANNOTATION_SHA256,
        "categories": [{"id": 118}, {"id": 188}],
        "records": [
            {
                "image": {"id": 9, "license": 4, "width": 24, "height": 16},
                "image_path": rgb.name,
                "category_path": mask.name,
                "image_sha256": _sha256(rgb),
                "category_sha256": _sha256(mask),
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_manifest_integrity_and_native_void_counts(tmp_path):
    path, _ = fixture_manifest(tmp_path)
    assert validate_real_dataset(path) == {
        "samples": 1,
        "native_pixels": 384,
        "valid_pixels": 383,
        "storage_pixels": 95,
        "hashes_checked": True,
    }
    with (tmp_path / "image.png").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        validate_real_dataset(path)


@pytest.mark.parametrize("change", ["path", "license", "duplicate", "unknown", "mode"])
def test_manifest_rejects_unsafe_or_incompatible_data(tmp_path, change):
    path, manifest = fixture_manifest(tmp_path)
    record = manifest["records"][0]
    if change == "path":
        record["category_path"] = "../outside.png"
    elif change == "license":
        record["image"]["license"] = 1
    elif change == "duplicate":
        manifest["records"].append(copy.deepcopy(record))
    elif change == "unknown":
        Image.new("L", (24, 16), 119).save(tmp_path / "mask.png")
        record["category_sha256"] = _sha256(tmp_path / "mask.png")
    else:
        Image.new("L", (24, 16), 40).save(tmp_path / "image.png")
        record["image_sha256"] = _sha256(tmp_path / "image.png")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_real_dataset(path)


def test_panoptic_decode_checks_full_24bit_ids_void_crowd_and_area():
    rgb = np.array([[[1, 2, 3], [7, 0, 0], [0, 0, 0]]], np.uint8)
    segments = [
        {"id": 197121, "category_id": 118, "area": 1},
        {"id": 7, "category_id": 188, "area": 1, "iscrowd": 1},
    ]
    assert decode_panoptic(rgb, segments).tolist() == [[118, 255, 255]]
    for broken in [
        segments[:1],
        segments + [segments[0]],
        [dict(segments[0], area=2), segments[1]],
    ]:
        with pytest.raises(ValueError):
            decode_panoptic(rgb, broken)


def test_selection_is_metadata_only_order_invariant_and_records_exclusions():
    images = [{"id": i, "license": 4, "width": 10, "height": 10} for i in range(140)]
    images[0]["license"] = 1
    annotations = [
        {
            "image_id": i,
            "segments_info": [
                {"category_id": 118, "area": 0 if i == 1 else 20},
                {"category_id": 188, "area": 5 if i < 30 else 0},
            ],
        }
        for i in range(140)
    ]
    selected, ledger = select_records({"images": images, "annotations": annotations})
    repeated, _ = select_records({"images": images[::-1], "annotations": annotations[::-1]})
    assert selected == repeated
    assert len(selected) == 80 and sum(r["stratum"] == "storage" for r in selected) == 28
    assert ledger[0]["reason"] == "license" and ledger[1]["reason"] == "floor_area"
    assert sum(r["reason"] == "included" for r in ledger) == 80
    assert sum(r["reason"] == "deterministic_cap" for r in ledger) == 58


def test_range_reader_verifies_exact_ranges_and_truncation(monkeypatch):
    import robotics_vision.real_data as module

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-0/6"}

    calls = []

    def urlopen(request, timeout):
        calls.append(request.get_header("Range"))
        response = Response(b"abcdef"[0:1] if len(calls) == 1 else b"cd")
        if len(calls) > 1:
            requested = request.get_header("Range").removeprefix("bytes=")
            response.headers = {"Content-Range": f"bytes {requested}/6"}
        return response

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    reader = RangeReader("https://example.test/archive")
    assert reader.seek(-4, 2) == 2
    assert reader.read(2) == b"cd" and reader.tell() == 4
    reader.seek(2)
    with pytest.raises(ValueError, match="Truncated"):
        reader.read(3)


def test_range_reader_rejects_http_and_range_ignoring_server(monkeypatch):
    import robotics_vision.real_data as module

    with pytest.raises(ValueError, match="HTTPS"):
        RangeReader("http://example.test/a")
    response = io.BytesIO(b"full archive")
    response.status = 200
    response.headers = {}
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="byte ranges"):
        RangeReader("https://example.test/a")


def test_frozen_reference_rejects_self_consistent_changed_download(tmp_path):
    path, manifest = fixture_manifest(tmp_path)
    reference = tmp_path / "frozen_manifest.json"
    reference.write_text(json.dumps(manifest))
    assert validate_real_dataset(path, reference)["samples"] == 1
    # A new, valid image plus an updated local checksum must not redefine frozen data.
    Image.new("RGB", (24, 16), (120, 80, 40)).save(tmp_path / "image.png")
    manifest["records"][0]["image_sha256"] = _sha256(tmp_path / "image.png")
    path.write_text(json.dumps(manifest))
    assert validate_real_dataset(path)["samples"] == 1
    with pytest.raises(ValueError, match="frozen reference"):
        validate_real_dataset(path, reference)
