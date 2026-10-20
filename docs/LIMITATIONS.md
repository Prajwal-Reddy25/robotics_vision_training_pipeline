# Limitations and evidence boundaries

| Status | What is supported |
| --- | --- |
| Implemented | Existing data/training/checkpoint/export/backends; deterministic paired stressors; optional training augmentation; study/failure ledger and raw timing samples. |
| Tested | CPU unit/integration suite, deterministic transformations, label geometry, legacy resume, nonfinite failures, atomic ledger interruption, exact CPU repeat/resume, dynamic exports, local CPU CI-equivalent smoke. |
| Experimentally demonstrated | 42 measured synthetic-data CPU training runs, 546 measured synthetic-condition evaluations, retained negative augmentation results, class IoU/degradation/seed SD, measured CPU/GPU latency/size, fresh CUDA/AMP finite execution. |
| Simulated/procedural | Warehouse images, lighting/texture/blur/noise stressors, planar homography viewpoint, visible foreground-rectangle occlusion. No robotics physics engine was used or needed for this raster-data project. |
| Not validated | Full four-class warehouse/physical camera/robot generalization, calibrated 3D viewpoint, physical sensor noise/motion blur, safety-critical navigation, long-horizon AMP convergence, TensorRT execution, hosted execution of the new maintenance branch. |

The dataset has only 180 scenes and 27 test images, with fixed semantic colors.
Training-seed SD (n=3) measures optimization variability within one fixed split;
it is not a confidence interval over unseen scenes. No multi-split or broad real-data
claim follows from these synthetic measurements. The 12-shift mean uses artificial equal prevalence; individual
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

v1.1 latency is model-call-only, batch 1, 64² FP32, 100 iterations. v1.2 adds64²/128² CPU model timing with the same exclusions. It excludes loading,
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

At the original audit the requested historical v1.0.0 tag was absent; `54f3a2a` is explicitly the
owner-approved provisional baseline at that audit. No tag, hardware observation, successful
score or failure attribution has been manufactured to fill missing evidence.

## Owner competency checklist

- [ ] State these boundaries when presenting the repository to employers.
- [ ] Explain which raw evidence supports a claim and which experiment remains missing.
- [ ] Distinguish bitwise determinism from stable metrics and restored RNG states.
- [ ] Describe a representative real-data evaluation without reusing this test set for tuning.

## v1.2 real-data boundaries

Real photographs and native human panoptic ground truth are now evaluated, for
binary floor and a restricted cabinet/storage obstacle subset. All 80 selected
images and42 frozen models are included; all 84 size/model evaluations completed.
This changes the prior absence of real-data evidence, but does not validate cones,
mobile robots, representative warehouses, calibrated sensors or physical robot use.

Floor/cabinet/licensing selection creates sampling bias. Only 20.50% of valid pixels
are scored in the restricted endpoint; surrounding false predictions can be ignored
there, so primary binary metrics are essential. Pixel pooling weights scene sizes;
three optimization seeds do not quantify scene uncertainty. Correlations share an
ID term and non-independent arms. The 128² sensitivity gap retains 64² ID metrics,
so it mixes input resolution and domain changes. Future tuning on these 80 test
images would retire their independent-test status.

Substantial transfer failures and negative/neutral augmentation outcomes are
measured, not omitted. Palette/BatchNorm/viewpoint mechanisms remain hypotheses.
Two original photos remain local because creator attribution is unavailable; GT
and predictions remain in tracked panels and all metrics. Input condition tags
are agent visual observations, not human-certified physical measurements.

External preparation requires verified HTTPS range support and endpoint availability;
source/checksum mismatch fails. External-data generation is not fully transactional.
Continuation validates source/data identity but provides no concurrent-worker lock.
The worker continued across a server restart; continuation refused the already
completed ledger. Frozen original checkpoint binaries remain local/ignored; rebuilding
needs the documented environment/configurations. Only two full CPU configurations
were independently retrained for this release, not all 42. CUDA bitwise-resume limits
remain despite four passing fresh probes. No TensorRT execution was added. Hosted CI
had not run at the original review; the published-tag CPU workflow subsequently
[passed on 2026-10-20](https://github.com/Prajwal-Reddy25/robotics_vision_training_pipeline/actions/runs/39390109521).

Tag inventory checked 2026-10-20: v1.0.0 points to `54f3a2a`, v1.1.0 to `d9a74a0`,
and v1.2.0 to `856e057`. Earlier missing-tag observations are retained in dated audits.
The maintenance offline verifier checks recorded arithmetic/integrity, not original
pixels, checkpoint bytes, measurement authenticity or physical deployment. Its new
hosted CI gate has not been published or run from this branch.

- [ ] Explain partial ontology, selection bias, metric projection and absence of scene confidence intervals.
- [ ] Describe how to create a new leakage-safe final test before warehouse adaptation.

Real-image letterboxing also changes padding/aspect context relative to square
synthetic training. Novel walls/rugs/furniture, class prevalence and preprocessing
co-occur with visual domain shift; their separate causal contributions are not isolated.
