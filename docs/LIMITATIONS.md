# Limitations and evidence boundaries

| Status | What is supported |
| --- | --- |
| Implemented | Existing data/training/checkpoint/export/backends; deterministic paired stressors; optional training augmentation; study/failure ledger and raw timing samples. |
| Tested | CPU unit/integration suite, deterministic transformations, label geometry, legacy resume, nonfinite failures, atomic ledger interruption, exact CPU repeat/resume, dynamic exports, local CPU CI-equivalent smoke. |
| Experimentally demonstrated | 42 real CPU training runs, 546 measured synthetic-condition evaluations, retained negative augmentation results, class IoU/degradation/seed SD, measured CPU/GPU latency/size, fresh CUDA/AMP finite execution. |
| Simulated/procedural | Warehouse images, lighting/texture/blur/noise stressors, planar homography viewpoint, visible foreground-rectangle occlusion. No robotics physics engine was used or needed for this raster-data project. |
| Not validated | Real camera/robot generalization, calibrated 3D viewpoint, physical sensor noise/motion blur, safety-critical navigation, long-horizon AMP convergence, TensorRT execution, remote GitHub Actions run for this branch. |

The dataset has only 180 scenes and 27 test images, with fixed semantic colors.
Training-seed SD (n=3) measures optimization variability within one fixed split;
it is not a confidence interval over unseen scenes. No multi-split or real-data
claim is justified. The 12-shift mean uses artificial equal prevalence; individual
class results and supports must accompany it. TinyUNet baseline nominal SD 0.1272
shows that a favorable single-seed result is particularly misleading.

Viewpoint is a trapezoid warp of a top-down image. RGB bilinear sampling and floor
fill can affect boundaries/prevalence; masks use nearest neighbor. Occluders are
palette-colored visible obstacles with class-1 labels, not unknowable hidden
labels. Texture affects all surfaces, lighting only dims, and blur is isotropic.
There are no real material, camera, temporal or illumination physics.

CUDA deterministic mode uses `warn_only=True`. A real LiteFCN FP32 resume probe
failed bitwise equality by up to 1.79e-7 while metrics and recovered RNG/scheduler
states matched. A later collection passed, demonstrating variability rather
than a general guarantee. Strict failures remain recorded. CPU exact repeat and
resume pass under the same software/thread protocol. Cross-device/version equality
is not promised. Finite-loss guards do not establish gradient health in every
scenario; AMP scaler overflow behavior is still relevant.

Latency is model-call-only, batch 1, 64² FP32, 100 iterations. It excludes loading,
augmentation, preprocessing, transfers and postprocessing; affinity, clocks,
thermals and power mode were not fixed. Checkpoint bytes include optimizer/RNG and
best state; parameter bytes and deployment bytes are different quantities.
Export parity validates conversion, not robot perception quality.

Full-state checkpoints use trusted Python pickle. Dependencies/actions retain
broad ranges; captured versions are an executed environment record, not a
cross-platform lock. TorchScript is deprecated. Dataset generation itself is
not transactional (the new study ledger is atomic). Interrupted studies retain
started records; the runner does not silently retry/reuse existing output trees.
Checkpoint resume checks the manifest path/configuration rather than a saved
manifest-content digest; its exact-trajectory claim assumes unchanged data bytes.
The new study records/verifies manifest and transformed-content hashes, but does
not retrofit data-identity guarantees into historical checkpoints.

The requested historical v1.0.0 tag is absent; `54f3a2a` is explicitly the
owner-approved provisional baseline. No tag, hardware observation, successful
score or failure attribution has been manufactured to fill missing evidence.

## Owner competency checklist

- [ ] State these boundaries when presenting the repository to employers.
- [ ] Explain which raw evidence supports a claim and which experiment remains missing.
- [ ] Distinguish bitwise determinism from stable metrics and restored RNG states.
- [ ] Describe a representative real-data evaluation without reusing this test set for tuning.
