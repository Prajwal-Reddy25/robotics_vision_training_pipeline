"""Paired, deterministic synthetic stressors; these are not sensor simulations."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .data import PALETTE, SegmentationDataset

SHIFT_VERSION = 1
# Strength units are documented in DOMAIN_SHIFT_V1_1.md. Fixed before the study.
CONDITIONS = {
    "lighting": (0.25, 0.50),
    "texture": (0.20, 0.40),
    "blur": (0.70, 1.40),
    "noise": (0.04, 0.08),
    "viewpoint": (0.10, 0.20),
    "occlusion": (0.06, 0.15),
}


def scene_seed(seed: int, identifier: str, kind: str) -> int:
    # Do not use Python hash(): process hash randomization would change experiments.
    encoded = f"{SHIFT_VERSION}:{seed}:{identifier}:{kind}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _perspective_coefficients(width: int, height: int, inset: float) -> tuple[float, ...]:
    """Solve destination-to-source homography for a top-inset trapezoid."""
    w, h = width - 1, height - 1
    source = [(0, 0), (w, 0), (w, h), (0, h)]
    destination = [(inset * w, 0), ((1 - inset) * w, 0), (w, h), (0, h)]
    rows, values = [], []
    for (x, y), (u, v) in zip(destination, source, strict=True):
        rows.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y],
                     [0, 0, 0, x, y, 1, -v * x, -v * y]))
        values.extend((u, v))
    return tuple(np.linalg.solve(np.asarray(rows), np.asarray(values)).tolist())


def apply_shift(
    image: Image.Image, mask: Image.Image, kind: str, strength: float, seed: int
) -> tuple[Image.Image, Image.Image]:
    if kind not in CONDITIONS:
        raise ValueError(f"Unknown shift: {kind}")
    if not np.isfinite(strength) or strength < 0 or strength > max(CONDITIONS[kind]):
        raise ValueError(f"Invalid {kind} strength: {strength}")
    if image.mode != "RGB" or mask.mode != "L" or image.size != mask.size:
        raise ValueError("Shifts require matching RGB image and L mask")
    if strength == 0:
        return image.copy(), mask.copy()
    rng = np.random.default_rng(seed)
    width, height = image.size
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(strength)), mask.copy()
    if kind == "viewpoint":
        coefficients = _perspective_coefficients(width, height, strength)
        warped_image = image.transform(
            image.size, Image.Transform.PERSPECTIVE, coefficients,
            resample=Image.Resampling.BILINEAR, fillcolor=tuple(PALETTE[0].tolist()),
        )
        warped_mask = mask.transform(
            mask.size, Image.Transform.PERSPECTIVE, coefficients,
            resample=Image.Resampling.NEAREST, fillcolor=0,
        )
        return warped_image, warped_mask
    array = np.asarray(image, dtype=np.float32) / 255.0
    labels = np.asarray(mask).copy()
    if kind == "lighting":
        array *= 1 - strength
    elif kind == "noise":
        array += rng.normal(0, strength, array.shape).astype(np.float32)
    elif kind == "texture":
        y, x = np.mgrid[:height, :width]
        phase = rng.uniform(0, 2 * np.pi, 2)
        field = np.sin(2 * np.pi * x / max(width / 6, 1) + phase[0])
        field *= np.sin(2 * np.pi * y / max(height / 5, 1) + phase[1])
        array *= (1 + strength * field[:, :, None]).astype(np.float32)
    elif kind == "occlusion":
        # Foreground obstacle: visible-surface semantics, not hidden-object inference.
        side = np.sqrt(strength)
        box_w, box_h = max(1, round(width * side)), max(1, round(height * side))
        x = int(rng.uniform() * (width - box_w + 1))
        y = int(rng.uniform() * (height - box_h + 1))
        array[y:y + box_h, x:x + box_w] = PALETTE[1].astype(np.float32) / 255
        labels[y:y + box_h, x:x + box_w] = 1
    rgb = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgb), Image.fromarray(labels)


class ShiftedDataset(Dataset[dict[str, Any]]):
    """A virtual evaluation set with content hashes; no augmentation or shuffle."""

    def __init__(
        self, nominal: SegmentationDataset, kind: str, strength: float, seed: int
    ) -> None:
        if nominal.augmentation.enabled:
            raise ValueError("Domain evaluation requires an unaugmented source dataset")
        if kind != "nominal" and kind not in CONDITIONS:
            raise ValueError(f"Unknown shift: {kind}")
        self.nominal, self.kind, self.strength, self.seed = nominal, kind, strength, seed

    def __len__(self) -> int:
        return len(self.nominal)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.nominal[index]
        if self.kind != "nominal":
            image = Image.fromarray(
                np.rint(sample["image"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            mask = Image.fromarray(sample["mask"].numpy().astype(np.uint8))
            image, mask = apply_shift(
                image, mask, self.kind, self.strength,
                scene_seed(self.seed, sample["id"], self.kind),
            )
            sample["image"] = torch.from_numpy(
                (np.asarray(image, dtype=np.float32) / 255).transpose(2, 0, 1).copy()
            )
            sample["mask"] = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return sample

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for sample in self:
            digest.update(sample["id"].encode())
            digest.update(sample["image"].numpy().astype("<f4").tobytes())
            digest.update(sample["mask"].numpy().astype("<i8").tobytes())
        return digest.hexdigest()
