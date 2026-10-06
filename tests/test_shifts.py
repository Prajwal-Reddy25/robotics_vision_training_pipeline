from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from robotics_vision.config import DEFAULT_CONFIG
from robotics_vision.data import PALETTE, SegmentationDataset, make_dataloader, prepare_dataset
from robotics_vision.shifts import (
    CONDITIONS,
    ShiftedDataset,
    _perspective_coefficients,
    apply_shift,
    scene_seed,
)
from robotics_vision.training import _validate_resume_config


def _pair() -> tuple[Image.Image, Image.Image]:
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[6:26, 6:26] = 2
    return Image.fromarray(PALETTE[labels]), Image.fromarray(labels)


@pytest.mark.parametrize("kind", CONDITIONS)
def test_shift_determinism_identity_bounds_and_labels(kind: str) -> None:
    image, mask = _pair()
    original_image, original_mask = np.asarray(image).copy(), np.asarray(mask).copy()
    unchanged = apply_shift(image, mask, kind, 0, 71)
    assert np.array_equal(np.asarray(unchanged[0]), original_image)
    assert np.array_equal(np.asarray(unchanged[1]), original_mask)
    for strength in CONDITIONS[kind]:
        first = apply_shift(image, mask, kind, strength, 71)
        second = apply_shift(image, mask, kind, strength, 71)
        assert first[0].size == image.size and first[1].size == mask.size
        assert np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
        assert np.array_equal(np.asarray(first[1]), np.asarray(second[1]))
        assert set(np.unique(first[1])) <= {0, 1, 2, 3}
        assert not np.array_equal(np.asarray(first[0]), original_image)
        if kind not in {"viewpoint", "occlusion"}:
            assert np.array_equal(np.asarray(first[1]), original_mask)
    assert np.array_equal(np.asarray(image), original_image)  # No source mutation.
    assert np.array_equal(np.asarray(mask), original_mask)


def test_occlusion_has_visible_obstacle_labels_and_actual_area() -> None:
    image, mask = _pair()
    image, mask = apply_shift(image, mask, "occlusion", 0.15, 31)
    occluder = np.asarray(mask) == 1
    assert occluder.sum() == round(32 * np.sqrt(0.15)) ** 2
    assert np.all(np.asarray(image)[occluder] == PALETTE[1])


def test_homography_maps_destination_corners_to_source_and_preserves_labels() -> None:
    a, b, c, d, e, f, g, h = _perspective_coefficients(32, 32, 0.2)
    for (x, y), expected in zip(
        [(6.2, 0), (24.8, 0), (31, 31), (0, 31)],
        [(0, 0), (31, 0), (31, 31), (0, 31)], strict=True,
    ):
        mapped = ((a * x + b * y + c) / (g * x + h * y + 1),
                  (d * x + e * y + f) / (g * x + h * y + 1))
        assert mapped == pytest.approx(expected)
    image, mask = apply_shift(*_pair(), "viewpoint", 0.2, 0)
    assert set(np.unique(mask)) == {0, 2}
    # All cone-colored interior pixels must agree with nearest-neighbor labels.
    interior = np.all(np.asarray(image) == PALETTE[2], axis=2)
    assert interior.sum() > 0
    assert np.all(np.asarray(mask)[interior] == 2)


def test_stable_scene_seed_and_common_severity_noise() -> None:
    assert scene_seed(1, "a", "noise") == scene_seed(1, "a", "noise")
    assert scene_seed(1, "a", "noise") != scene_seed(1, "b", "noise")
    image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    mask = Image.new("L", (32, 32))
    low, _ = apply_shift(image, mask, "noise", 0.04, 71)
    high, _ = apply_shift(image, mask, "noise", 0.08, 71)
    assert np.allclose((np.asarray(low).astype(float) - 128) * 2,
                       np.asarray(high).astype(float) - 128, atol=1)


def test_new_augmentation_is_repeatable_and_eval_stays_nominal(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    root = tmp_path / "data"
    config["data"].update(root=str(root), manifest=str(root / "manifest.json"),
                          image_size=32, num_samples=24)
    manifest = prepare_dataset(config)
    aug = {**config["augmentation"], "blur_radius": 1.4, "texture_strength": 0.4,
           "viewpoint_inset": 0.2, "occlusion_fraction": 0.15}
    dataset = SegmentationDataset(manifest, "train", aug, seed=91)
    assert torch.equal(dataset[0]["image"], dataset[0]["image"])
    first = dataset[0]
    dataset.set_epoch(1)
    assert not torch.equal(first["image"], dataset[0]["image"])
    assert set(dataset[0]["mask"].unique().tolist()) <= {0, 1, 2, 3}
    loader = make_dataloader(dataset, 3, False, 0, 7)
    assert torch.equal(next(iter(loader))["image"][0], dataset[0]["image"])
    nominal = SegmentationDataset(manifest, "test")
    clean = ShiftedDataset(nominal, "nominal", 0, 2026)
    assert torch.equal(clean[0]["image"], nominal[0]["image"])
    noisy = ShiftedDataset(nominal, "noise", 0.08, 2026)
    assert noisy.fingerprint() == ShiftedDataset(nominal, "noise", 0.08, 2026).fingerprint()
    assert noisy.fingerprint() != clean.fingerprint()
    assert [noisy[i]["id"] for i in range(len(noisy))] == [r["id"] for r in nominal.records]
    with pytest.raises(ValueError, match="unaugmented"):
        ShiftedDataset(dataset, "noise", 0.04, 2026)


def test_resume_accepts_legacy_zero_defaults_but_rejects_new_aug() -> None:
    current = copy.deepcopy(DEFAULT_CONFIG)
    previous = copy.deepcopy(current)
    for key in ("blur_radius", "texture_strength", "viewpoint_inset", "occlusion_fraction"):
        previous["augmentation"].pop(key)
    _validate_resume_config(current, previous)
    current["augmentation"]["blur_radius"] = 1.4
    with pytest.raises(ValueError, match="augmentation"):
        _validate_resume_config(current, previous)


@pytest.mark.parametrize("kind,strength", [("unknown", 0.1), ("noise", float("nan")),
                                           ("viewpoint", 0.5), ("occlusion", -0.1)])
def test_invalid_shifts_fail(kind: str, strength: float) -> None:
    with pytest.raises(ValueError):
        apply_shift(*_pair(), kind, strength, 0)
