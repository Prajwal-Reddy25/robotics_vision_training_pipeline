# Performance measurements v1.2

CPU FP32, batch1, one thread, 20 warmups/100 completed model calls per run.
Baseline three-seed mean ± sample SD of each run's p50/p95, not pooled quantiles.

| Model | Input | p50 ms | p95 ms | Parameters | Parameter bytes | Checkpoint bytes |
| --- | --- | --- | --- | --- | --- | --- |
| lite_fcn | 64 | 0.3498 ± 0.0051 | 0.3722 ± 0.0144 | 2804 | 11216 | 112833–112897 |
| lite_fcn | 128 | 0.6968 ± 0.0156 | 0.7843 ± 0.0346 | 2804 | 11216 | 112833–112897 |
| tiny_unet | 64 | 1.5152 ± 0.0011 | 1.6545 ± 0.0055 | 29948 | 119792 | 564803–564803 |
| tiny_unet | 128 | 4.3482 ± 0.0073 | 4.8626 ± 0.3041 | 29948 | 119792 | 564803–564803 |

Raw samples for all 84 model/size measurements are in `evidence/v1_2/study.json`.
Model timing excludes preprocessing, transfers and native postprocessing.
RGB resizing/letterbox/tensor conversion mean times are recorded separately in the
CSV/raw records; they are not end-to-end robot latency. CPU affinity, power/clocks
and competing load are uncontrolled. The detached worker completed across a server
restart; timing remains an observed environment result, not an isolated speed guarantee.

Fresh synchronized GPU eager/TorchScript benchmark samples and synthetic accuracy
are in `evidence/v1_2/gpu.json`; fresh CPU TS/ORT results and deployment artifact
sizes are in `deployment.json`. These are the original64² synthetic benchmark
inputs, not native real-photo pipeline/GPU OOD quality benchmarks. TensorRT remains
optional and unavailable, and no physical camera/robot data were collected.
