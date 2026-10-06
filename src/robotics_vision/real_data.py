"""Official real-image evaluation data; never used by the training loader."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .benchmarking import _sha256
from .runlog import write_json

ARCHIVE_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/annotations/"
    "panoptic_annotations_trainval2017.zip"
)
ANNOTATION_SHA256 = "454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a"
FLOOR_CATEGORIES = {118, 190}
STORAGE_CATEGORY = 188


class RangeReader(io.RawIOBase):
    """Seekable HTTPS archive view; request only the selected ZIP members.

    TLS verification stays enabled, Content-Range and exact response length are
    enforced. ZipFile additionally verifies each decompressed member's CRC.
    """

    def __init__(self, url: str) -> None:
        if not url.startswith("https://"):
            raise ValueError("Archive URL must use HTTPS")
        self.url, self.position = url, 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            header = response.headers.get("Content-Range", "")
            if not header.startswith("bytes 0-0/") or response.status != 206:
                raise ValueError("Archive server must support exact byte ranges")
            self.size = int(header.split("/")[-1])

    def seek(self, offset: int, whence: int = 0) -> int:
        bases = {0: 0, 1: self.position, 2: self.size}
        if whence not in bases or bases[whence] + offset < 0:
            raise ValueError("Invalid archive seek")
        self.position = bases[whence] + offset
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self.size - self.position)
        size = min(size if size >= 0 else remaining, remaining)
        if not size:
            return b""
        end = self.position + size - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            expected = f"bytes {self.position}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Unexpected archive range response")
            content = response.read()
        if len(content) != size:
            raise ValueError("Truncated archive response")
        self.position += size
        return content


def select_records(annotation: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    images = {image["id"]: image for image in annotation["images"]}
    candidates, ledger = [], []
    for record in annotation["annotations"]:
        image = images[record["image_id"]]
        area = image["width"] * image["height"]
        floor = sum(
            s["area"]
            for s in record["segments_info"]
            if s["category_id"] in FLOOR_CATEGORIES and not s.get("iscrowd", 0)
        )
        storage = sum(
            s["area"]
            for s in record["segments_info"]
            if s["category_id"] == STORAGE_CATEGORY and not s.get("iscrowd", 0)
        )
        reason = (
            "license" if image["license"] != 4 else "floor_area" if floor / area < 0.02 else None
        )
        entry = {
            "id": image["id"],
            "reason": reason,
            "floor_fraction": floor / area,
            "storage_fraction": storage / area,
        }
        ledger.append(entry)
        if reason is None:
            candidates.append(
                {
                    "image": image,
                    "annotation": record,
                    "stratum": "storage" if storage / area >= 0.01 else "other",
                }
            )

    def key(record):
        return hashlib.sha256(f"2026:{record['image']['id']}".encode()).hexdigest()

    storage = sorted((r for r in candidates if r["stratum"] == "storage"), key=key)[:40]
    other = sorted((r for r in candidates if r["stratum"] == "other"), key=key)
    selected = storage + other[: 80 - len(storage)]
    ids = {r["image"]["id"] for r in selected}
    for entry in ledger:
        if entry["reason"] is None:
            entry["reason"] = "included" if entry["id"] in ids else "deterministic_cap"
    return selected, ledger


def decode_panoptic(rgb: np.ndarray, segments: list[dict]) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Panoptic IDs must be an RGB uint8 array")
    ids = rgb.astype(np.int32)
    ids = ids[:, :, 0] + 256 * ids[:, :, 1] + 65536 * ids[:, :, 2]
    unique, counts = np.unique(ids, return_counts=True)
    areas = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    declared = {segment["id"] for segment in segments}
    if len(declared) != len(segments) or set(areas) - declared - {0}:
        raise ValueError("Undeclared or duplicate panoptic segment ID")
    categories = np.full(ids.shape, 255, dtype=np.uint8)
    for segment in segments:
        if areas.get(segment["id"], 0) != segment["area"]:
            raise ValueError("Panoptic segment area disagrees with metadata")
        if not 0 <= segment["category_id"] < 255:
            raise ValueError("Category ID must fit uint8 and differ from void")
        if not segment.get("iscrowd", 0):
            categories[ids == segment["id"]] = segment["category_id"]
    return categories


def prepare_real_dataset(root: str | Path) -> Path:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        validate_real_dataset(manifest_path)
        return manifest_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    (root / "categories").mkdir(exist_ok=True)
    with zipfile.ZipFile(RangeReader(ARCHIVE_URL)) as archive:
        raw = archive.read("annotations/panoptic_val2017.json")
        if hashlib.sha256(raw).hexdigest() != ANNOTATION_SHA256:
            raise ValueError("Official annotation metadata changed")
        annotation = json.loads(raw)
        records, ledger = select_records(annotation)
        masks = archive.read("annotations/panoptic_val2017.zip")
    (root / "panoptic_val2017.json").write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(masks)) as archive:
        for index, record in enumerate(records):
            image, ann = record["image"], record["annotation"]
            filename = image["file_name"]
            if Path(filename).name != filename or not filename.endswith(".jpg"):
                raise ValueError("Unsafe source filename")
            url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{filename}"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            image_path = root / "images" / filename
            image_path.write_bytes(content)
            png_name = ann["file_name"]
            raw_mask = archive.read(f"panoptic_val2017/{png_name}")
            with Image.open(io.BytesIO(raw_mask)) as source:
                categories = decode_panoptic(
                    np.asarray(source.convert("RGB")), ann["segments_info"]
                )
            mask_path = root / "categories" / png_name
            Image.fromarray(categories).save(mask_path)
            record.update(
                image_path=f"images/{filename}",
                category_path=f"categories/{png_name}",
                image_sha256=_sha256(image_path),
                category_sha256=_sha256(mask_path),
                source_panoptic_sha256=hashlib.sha256(raw_mask).hexdigest(),
                source_url=url,
            )
            print(f"REAL prepare {index + 1}/{len(records)} {image['id']}", flush=True)
    manifest = {
        "format_version": 1,
        "dataset": "COCO2017_panoptic_real_floor_storage_v1",
        "annotation_url": ARCHIVE_URL,
        "annotation_sha256": ANNOTATION_SHA256,
        "mask_archive_sha256": hashlib.sha256(masks).hexdigest(),
        "annotation_license": "CC-BY-4.0",
        "image_license": "CC-BY-2.0",
        "categories": annotation["categories"],
        "records": records,
        "selection_ledger": ledger,
        "note": "Evaluation only; no real training or tuning. Floor/storage partial compatibility.",
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    write_json(temporary, manifest)
    temporary.replace(manifest_path)
    validate_real_dataset(manifest_path)
    return manifest_path


def validate_real_dataset(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    root = manifest_path.parent.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1 or not manifest.get("records"):
        raise ValueError("Invalid external evaluation manifest")
    if manifest.get("annotation_sha256") != ANNOTATION_SHA256:
        raise ValueError("Unrecognized annotation source")
    seen_ids, seen_hashes, seen_paths = set(), set(), set()
    pixels, valid, storage = 0, 0, 0
    for record in manifest["records"]:
        image = record["image"]
        if image["id"] in seen_ids or image["license"] != 4:
            raise ValueError("Duplicate image ID or unsupported image license")
        seen_ids.add(image["id"])
        paths = []
        for field, sha_field in (
            ("image_path", "image_sha256"),
            ("category_path", "category_sha256"),
        ):
            relative = Path(record[field])
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root) or path in seen_paths:
                raise ValueError("Unsafe or reused external data path")
            seen_paths.add(path)
            if _sha256(path) != record[sha_field]:
                raise ValueError("External data checksum mismatch")
            paths.append(path)
        if record["image_sha256"] in seen_hashes:
            raise ValueError("Duplicate image content")
        seen_hashes.add(record["image_sha256"])
        with Image.open(paths[0]) as rgb, Image.open(paths[1]) as mask:
            expected = (image["width"], image["height"])
            if (
                rgb.mode != "RGB"
                or mask.mode != "L"
                or rgb.size != expected
                or mask.size != expected
            ):
                raise ValueError("External image/mask modes or dimensions disagree")
            labels = np.asarray(mask)
            allowed = {c["id"] for c in manifest["categories"]} | {255}
            if set(np.unique(labels).tolist()) - allowed:
                raise ValueError("Unknown source semantic category")
            pixels += labels.size
            valid += int((labels != 255).sum())
            storage += int((labels == STORAGE_CATEGORY).sum())
    return {
        "samples": len(seen_ids),
        "native_pixels": pixels,
        "valid_pixels": valid,
        "storage_pixels": storage,
        "hashes_checked": True,
    }
