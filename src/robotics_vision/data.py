from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from .reproducibility import seed_worker

CLASS_NAMES = ["floor", "obstacle", "safety_cone", "mobile_robot"]
PALETTE = np.asarray(
    [
        [75, 75, 78],
        [45, 100, 190],
        [245, 125, 20],
        [55, 180, 110],
    ],
    dtype=np.uint8,
)
GENERATOR_VERSION = "1.0"
GENERATOR_NAME = "robotics_vision.synthetic_warehouse"


class DataValidationError(ValueError):
    """Raised when the dataset manifest or a sample is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_scene(index: int, image_size: int, seed: int) -> tuple[Image.Image, Image.Image]:
    rng = random.Random(seed + index * 104729)
    array_rng = np.random.default_rng(seed + index * 104729)
    base = array_rng.normal(0.0, 4.0, (image_size, image_size, 3))
    gradient = np.linspace(48, 94, image_size, dtype=np.float32)[:, None, None]
    rgb = np.clip(base + gradient + np.asarray([16, 14, 12]), 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    mask = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    # Warehouse lane markings are visual distractors and remain floor in the mask.
    lane_x = rng.randint(image_size // 5, image_size * 4 // 5)
    draw.line((lane_x, 0, lane_x, image_size), fill=(210, 190, 70), width=max(1, image_size // 40))

    for _ in range(rng.randint(1, 3)):
        width = rng.randint(max(5, image_size // 9), max(7, image_size // 4))
        height = rng.randint(max(5, image_size // 10), max(7, image_size // 3))
        x0 = rng.randint(1, max(1, image_size - width - 2))
        y0 = rng.randint(1, max(1, image_size - height - 2))
        box = (x0, y0, x0 + width, y0 + height)
        draw.rectangle(box, fill=(40, 95 + rng.randint(0, 35), 185), outline=(20, 40, 85), width=1)
        mask_draw.rectangle(box, fill=1)

    for _ in range(rng.randint(1, 2)):
        width = rng.randint(max(5, image_size // 12), max(7, image_size // 7))
        height = rng.randint(max(8, image_size // 8), max(10, image_size // 4))
        x0 = rng.randint(1, max(1, image_size - width - 2))
        y0 = rng.randint(1, max(1, image_size - height - 2))
        polygon = [(x0 + width // 2, y0), (x0, y0 + height), (x0 + width, y0 + height)]
        draw.polygon(polygon, fill=(245, 120, 15), outline=(255, 210, 100))
        mask_draw.polygon(polygon, fill=2)

    if rng.random() < 0.9:
        radius = rng.randint(max(4, image_size // 12), max(6, image_size // 8))
        cx = rng.randint(radius + 1, image_size - radius - 2)
        cy = rng.randint(radius + 1, image_size - radius - 2)
        body = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(body, fill=(45, 175, 105), outline=(15, 70, 45), width=1)
        mask_draw.ellipse(body, fill=3)
        heading = (cx, cy - radius, cx + max(2, radius // 3), cy)
        draw.rectangle(heading, fill=(225, 240, 235))
        mask_draw.rectangle(heading, fill=3)
    return image, mask


def _split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * float(ratio) for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    for name in sorted(ratios, key=lambda key: raw[key] - counts[key], reverse=True):
        if sum(counts.values()) == total:
            break
        counts[name] += 1
    return counts


def _assert_safe_generated_root(root: Path, manifest_path: Path) -> None:
    resolved = root.resolve()
    dangerous = {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}
    if resolved in dangerous or Path.cwd().resolve().is_relative_to(resolved):
        raise DataValidationError(f"Refusing to replace unsafe data root: {root}")
    if root.is_symlink():
        raise DataValidationError(f"Refusing to replace symlinked data root: {root}")
    if not any(root.iterdir()):
        return
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        if manifest.get("generator") == GENERATOR_NAME:
            return
    managed_base = (Path.cwd() / "data" / "processed").resolve()
    if not resolved.is_relative_to(managed_base) or resolved == managed_base:
        raise DataValidationError(
            "Refusing to replace a non-empty directory without a matching generated manifest"
        )


def _validate_manifest_matches_config(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    data_config = config["data"]
    actual_samples = sum(len(records) for records in manifest.get("splits", {}).values())
    expected = {
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "seed": int(data_config["seed"]),
        "image_size": [int(data_config["image_size"]), int(data_config["image_size"])],
        "split_ratios": data_config["split"],
        "samples": int(data_config["num_samples"]),
    }
    actual = {
        "generator": manifest.get("generator"),
        "generator_version": manifest.get("generator_version"),
        "seed": manifest.get("seed"),
        "image_size": manifest.get("image_size"),
        "split_ratios": manifest.get("split_ratios"),
        "samples": actual_samples,
    }
    mismatches = [key for key in expected if actual[key] != expected[key]]
    if mismatches:
        raise DataValidationError(
            "Existing manifest does not match config fields "
            f"{mismatches}; use --force to regenerate"
        )


def _resolve_record_path(root: Path, value: Any, field: str, identifier: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"Invalid {field} path for sample {identifier}")
    relative = Path(value)
    if relative.is_absolute():
        raise DataValidationError(f"Absolute {field} path is not allowed for sample {identifier}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise DataValidationError(f"{field} path escapes dataset root for sample {identifier}")
    return resolved


def prepare_dataset(config: dict[str, Any], force: bool = False) -> Path:
    data_config = config["data"]
    root = Path(data_config["root"])
    manifest_path = Path(data_config["manifest"])
    if root.resolve() != manifest_path.parent.resolve():
        raise DataValidationError("data.manifest must be directly inside data.root")
    if manifest_path.exists() and not force:
        validate_dataset(manifest_path, check_hashes=bool(data_config["validate_hashes"]))
        _validate_manifest_matches_config(load_manifest(manifest_path), config)
        return manifest_path
    if root.exists() and force:
        _assert_safe_generated_root(root, manifest_path)
        shutil.rmtree(root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    total = int(data_config["num_samples"])
    seed = int(data_config["seed"])
    image_size = int(data_config["image_size"])
    identifiers = [f"scene_{index:05d}" for index in range(total)]
    shuffled = identifiers.copy()
    random.Random(seed).shuffle(shuffled)
    counts = _split_counts(total, data_config["split"])
    split_by_id: dict[str, str] = {}
    cursor = 0
    for split_name in ("train", "val", "test"):
        for identifier in shuffled[cursor : cursor + counts[split_name]]:
            split_by_id[identifier] = split_name
        cursor += counts[split_name]

    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for index, identifier in enumerate(identifiers):
        image, mask = _draw_scene(index, image_size, seed)
        image_relative = Path("images") / f"{identifier}.png"
        mask_relative = Path("masks") / f"{identifier}.png"
        image_path = root / image_relative
        mask_path = root / mask_relative
        image.save(image_path, format="PNG", optimize=False)
        mask.save(mask_path, format="PNG", optimize=False)
        record = {
            "id": identifier,
            "image": image_relative.as_posix(),
            "mask": mask_relative.as_posix(),
            "image_sha256": _sha256(image_path),
            "mask_sha256": _sha256(mask_path),
        }
        splits[split_by_id[identifier]].append(record)

    manifest = {
        "schema_version": 1,
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "license": "CC0-1.0",
        "seed": seed,
        "image_size": [image_size, image_size],
        "class_names": CLASS_NAMES,
        "split_ratios": data_config["split"],
        "splits": splits,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    validate_dataset(manifest_path, check_hashes=True)
    return manifest_path


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"Manifest does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Could not read manifest {path}: {exc}") from exc
    return manifest


def validate_dataset(manifest_path: str | Path, check_hashes: bool = True) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    if manifest.get("schema_version") != 1:
        raise DataValidationError("Unsupported or missing manifest schema_version")
    if not isinstance(manifest.get("generator"), str) or not isinstance(
        manifest.get("generator_version"), str
    ):
        raise DataValidationError("Manifest generator metadata is missing or invalid")
    if manifest.get("class_names") != CLASS_NAMES:
        raise DataValidationError(
            f"Expected classes {CLASS_NAMES}, got {manifest.get('class_names')}"
        )
    if set(manifest.get("splits", {})) != {"train", "val", "test"}:
        raise DataValidationError("Manifest must contain train, val, and test splits")

    image_size = manifest.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in image_size
        )
    ):
        raise DataValidationError("Manifest image_size must contain two positive integers")

    seen: set[str] = set()
    seen_image_paths: dict[Path, str] = {}
    seen_mask_paths: dict[Path, str] = {}
    seen_image_hashes: dict[str, str] = {}
    class_pixels = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    split_class_pixels: dict[str, dict[str, int]] = {}
    split_counts: dict[str, int] = {}
    expected_size = tuple(image_size)
    for split_name, records in manifest["splits"].items():
        if not isinstance(records, list) or not records:
            raise DataValidationError(f"Split {split_name} is empty")
        split_counts[split_name] = len(records)
        current_split_pixels = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        for record in records:
            if not isinstance(record, dict):
                raise DataValidationError(f"Invalid record in split {split_name}")
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise DataValidationError(f"Invalid sample id in split {split_name}")
            if identifier in seen:
                raise DataValidationError(f"Duplicate or leaked sample id: {identifier}")
            seen.add(identifier)
            image_path = _resolve_record_path(root, record.get("image"), "image", identifier)
            mask_path = _resolve_record_path(root, record.get("mask"), "mask", identifier)
            if image_path in seen_image_paths:
                raise DataValidationError(
                    f"Image path reused by {seen_image_paths[image_path]} and {identifier}"
                )
            if mask_path in seen_mask_paths:
                raise DataValidationError(
                    f"Mask path reused by {seen_mask_paths[mask_path]} and {identifier}"
                )
            seen_image_paths[image_path] = identifier
            seen_mask_paths[mask_path] = identifier
            if not image_path.is_file() or not mask_path.is_file():
                raise DataValidationError(f"Missing image or mask for sample {identifier}")
            if check_hashes:
                image_hash = _sha256(image_path)
                mask_hash = _sha256(mask_path)
                if image_hash != record.get("image_sha256"):
                    raise DataValidationError(f"Image checksum mismatch: {identifier}")
                if mask_hash != record.get("mask_sha256"):
                    raise DataValidationError(f"Mask checksum mismatch: {identifier}")
                if image_hash in seen_image_hashes:
                    raise DataValidationError(
                        "Duplicate image content in "
                        f"{seen_image_hashes[image_hash]} and {identifier}"
                    )
                seen_image_hashes[image_hash] = identifier
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.mode != "RGB" or mask.mode != "L":
                    raise DataValidationError(
                        f"Invalid modes for {identifier}: {image.mode}, {mask.mode}"
                    )
                if image.size != expected_size or mask.size != expected_size:
                    raise DataValidationError(f"Invalid dimensions for {identifier}")
                labels, counts = np.unique(np.asarray(mask), return_counts=True)
                if labels.size and (labels.min() < 0 or labels.max() >= len(CLASS_NAMES)):
                    raise DataValidationError(
                        f"Out-of-range label in {identifier}: {labels.tolist()}"
                    )
                class_pixels[labels] += counts
                current_split_pixels[labels] += counts
        split_class_pixels[split_name] = dict(
            zip(CLASS_NAMES, current_split_pixels.tolist(), strict=True)
        )
    if np.any(class_pixels == 0):
        missing = [CLASS_NAMES[i] for i in np.flatnonzero(class_pixels == 0)]
        raise DataValidationError(f"Classes with no pixels: {missing}")
    return {
        "valid": True,
        "samples": len(seen),
        "split_counts": split_counts,
        "class_pixels": dict(zip(CLASS_NAMES, class_pixels.tolist(), strict=True)),
        "split_class_pixels": split_class_pixels,
        "hashes_checked": check_hashes,
    }


@dataclass
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip_probability: float = 0.5
    brightness: float = 0.15
    contrast: float = 0.15
    gaussian_noise_std: float = 0.02


class SegmentationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        augmentation: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        manifest = load_manifest(self.manifest_path)
        if split not in manifest["splits"]:
            raise ValueError(f"Unknown split: {split}")
        self.records = manifest["splits"][split]
        self.split = split
        self.augmentation = AugmentationConfig(**(augmentation or {"enabled": False}))
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(self.root / record["image"]) as source:
            image = source.convert("RGB")
        with Image.open(self.root / record["mask"]) as source:
            mask = source.convert("L")
        local_seed = self.seed + self.epoch * 1_000_003 + index * 97
        rng = random.Random(local_seed)
        aug = self.augmentation
        if aug.enabled:
            if rng.random() < aug.horizontal_flip_probability:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)
            if aug.brightness:
                factor = 1.0 + rng.uniform(-aug.brightness, aug.brightness)
                image = ImageEnhance.Brightness(image).enhance(factor)
            if aug.contrast:
                factor = 1.0 + rng.uniform(-aug.contrast, aug.contrast)
                image = ImageEnhance.Contrast(image).enhance(factor)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if aug.enabled and aug.gaussian_noise_std:
            noise_rng = np.random.default_rng(local_seed + 1)
            noise = noise_rng.normal(0.0, aug.gaussian_noise_std, image_array.shape)
            image_array = np.clip(image_array + noise, 0.0, 1.0).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return {"image": image_tensor, "mask": mask_tensor, "id": record["id"]}


def make_dataloader(
    dataset: SegmentationDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        # Fresh workers inherit dataset.epoch after set_epoch() on every epoch.
        persistent_workers=False,
    )


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.min() < 0 or mask.max() >= len(PALETTE):
        raise ValueError("Mask contains an out-of-range class")
    return PALETTE[mask]
