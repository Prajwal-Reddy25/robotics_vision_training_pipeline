# Model card: Warehouse semantic segmentation

## Model details

This repository trains one of two small, fully convolutional PyTorch models from random initialization:

- **TinyUNet**: two encoder stages, a bottleneck, bilinear decoding, and skip connections.
- **LiteFCN**: a depthwise-separable encoder and bilinear logit upsampling.

Both accept float32 NCHW RGB tensors in `[0, 1]` and return four-channel NCHW logits at the input spatial resolution. The class map is defined in the dataset card.

## Intended use

- teaching and demonstrating a complete ML lifecycle;
- automated tests for training, export, and deployment infrastructure;
- architecture/backend trade-off experiments on constrained devices;
- a starting template for a properly reviewed robotics dataset.

## Out-of-scope use

- safety-critical obstacle avoidance or navigation;
- direct real-world deployment without representative data and validation;
- person detection, tracking, range estimation, or instance counting;
- claims of robustness outside the procedural generator distribution.

## Metrics

The evaluator emits pixel accuracy, precision, recall, IoU, mean IoU, confusion matrix, and foreground pixel mAP. Pixel mAP is a histogram approximation of one-vs-rest semantic pixel ranking; it is not COCO-style instance mAP. Worst per-image mean-IoU panels support qualitative error analysis.

## Results

The local verification record contains one explicitly dated, single-seed measured run with its environment and limitations. Run the README commands to produce fresh `evaluation/metrics.json` and benchmark JSON for the current code and hardware; do not treat the recorded run as a release guarantee.

## Limitations and risks

- Training and evaluation share a simple procedural domain.
- Tiny images omit fine boundaries and long-range context.
- Color strongly predicts classes, so color/lighting shift is a major risk.
- Pixel accuracy can hide minority-class failure; use per-class recall/IoU and the confusion matrix.
- Export agreement validates numerical conversion, not end-system correctness.
- Latency depends on device, thread settings, power state, provider version, and batch size.

## Deployment requirements

Use the exact preprocessing contract, inspect export agreement in `metadata.json`, run target-hardware benchmarks, define reject/fallback behavior, and conduct representative safety validation. Monitor input drift and per-class outcomes after deployment. Retain a rollback-capable, versioned artifact and configuration.
