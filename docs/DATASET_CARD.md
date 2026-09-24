# Dataset card: Synthetic Warehouse Shapes

## Summary

Synthetic Warehouse Shapes is a deterministic, procedurally generated semantic-segmentation dataset intended for pipeline development and robotics-perception demonstrations. Each RGB image resembles a top-down warehouse floor containing lane paint, blue rectangular obstacles/pallets, orange safety cones, and green mobile robots. Its mask has four integer classes:

| ID | Class | Meaning |
| ---: | --- | --- |
| 0 | `floor` | Traversable floor and lane-paint distractors |
| 1 | `obstacle` | Rectangular static obstacles or pallets |
| 2 | `safety_cone` | Triangular safety markers |
| 3 | `mobile_robot` | Circular mobile robot footprint and heading marker |

This is a synthetic learning problem, not a claim of sim-to-real transfer.

## Creation and license

`rv-prepare` calls the repository's seeded Pillow/NumPy generator. It uses no external images, annotations, personal data, or pretrained assets. Generated samples are dedicated to the public domain under CC0-1.0; generator source code is MIT licensed. The manifest records generator/version, seed, size, classes, split ratios, relative paths, and a SHA-256 digest for every image and mask.

## Default release

The default configuration generates 180 images at 64×64 pixels. A seeded shuffle assigns 70%/15%/15% train/validation/test splits by unique scene ID. The split allocator handles rounding deterministically and validation rejects empty or overlapping splits.

No generated release is committed. Recreate it with:

```bash
rv-prepare --config configs/tiny_unet.yaml
rv-validate --config configs/tiny_unet.yaml
```

Re-running without `--force` validates and reuses an existing manifest. `--force` replaces only the configured generated-data root; verify custom paths before using it.

## Validation

The validator checks:

- schema and exact class map;
- required non-empty splits and globally unique IDs;
- file existence, RGB/L modes, matching expected dimensions;
- labels constrained to `[0, 3]` and global class presence;
- image and mask SHA-256 values unless explicitly skipped.

## Biases and limitations

- Shapes, texture, illumination, and camera geometry are deliberately simple.
- Objects may overlap, and later-drawn objects own overlapping mask pixels.
- Class frequency is imbalanced because floor occupies most pixels; training uses inverse-square-root weights computed from the training split only.
- There are no sensor artifacts such as motion blur, lens distortion, occlusion realism, depth noise, or adverse weather.
- The test split measures in-generator generalization only and must not be used as evidence of real-robot safety or performance.

For a real deployment, replace the generator with a versioned, license-reviewed dataset adapter while retaining the manifest, hash, split, and validation contracts.
