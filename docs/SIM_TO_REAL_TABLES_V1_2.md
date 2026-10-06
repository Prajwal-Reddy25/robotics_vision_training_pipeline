# Sim-to-real measured contrasts — complete tables

Augmented − baseline, mean ± sample SD; deltas ordered seeds 42/43/44.

| Model | Arm | Input | Real binary delta | Seed deltas | Four-class ID delta |
| --- | --- | --- | --- | --- | --- |
| lite_fcn | blur | 64 | 0.0175 ± 0.0563 | +0.0705, -0.0417, +0.0237 | -0.0110 ± 0.0226 |
| lite_fcn | blur | 128 | 0.0175 ± 0.0584 | +0.0645, -0.0479, +0.0359 | -0.0110 ± 0.0226 |
| lite_fcn | combined | 64 | 0.0151 ± 0.0383 | +0.0525, -0.0241, +0.0170 | -0.0571 ± 0.0552 |
| lite_fcn | combined | 128 | 0.0211 ± 0.0375 | +0.0536, -0.0198, +0.0296 | -0.0571 ± 0.0552 |
| lite_fcn | no_aug | 64 | 0.0288 ± 0.0123 | +0.0411, +0.0287, +0.0165 | 0.0154 ± 0.0148 |
| lite_fcn | no_aug | 128 | 0.0297 ± 0.0015 | +0.0299, +0.0281, +0.0312 | 0.0154 ± 0.0148 |
| lite_fcn | occlusion | 64 | 0.0089 ± 0.0094 | -0.0001, +0.0187, +0.0081 | -0.0023 ± 0.0347 |
| lite_fcn | occlusion | 128 | 0.0158 ± 0.0070 | +0.0078, +0.0192, +0.0205 | -0.0023 ± 0.0347 |
| lite_fcn | texture | 64 | -0.0177 ± 0.0199 | -0.0105, -0.0401, -0.0023 | 0.0198 ± 0.0130 |
| lite_fcn | texture | 128 | -0.0089 ± 0.0229 | -0.0008, -0.0348, +0.0089 | 0.0198 ± 0.0130 |
| lite_fcn | viewpoint | 64 | -0.0328 ± 0.0403 | -0.0773, -0.0228, +0.0015 | 0.0070 ± 0.0233 |
| lite_fcn | viewpoint | 128 | -0.0297 ± 0.0448 | -0.0778, -0.0219, +0.0107 | 0.0070 ± 0.0233 |
| tiny_unet | blur | 64 | 0.0594 ± 0.0429 | +0.0741, +0.0110, +0.0930 | 0.0193 ± 0.0587 |
| tiny_unet | blur | 128 | 0.0525 ± 0.0390 | +0.0643, +0.0090, +0.0843 | 0.0193 ± 0.0587 |
| tiny_unet | combined | 64 | -0.0245 ± 0.0468 | +0.0064, -0.0015, -0.0783 | 0.0491 ± 0.0525 |
| tiny_unet | combined | 128 | -0.0242 ± 0.0459 | +0.0006, +0.0039, -0.0771 | 0.0491 ± 0.0525 |
| tiny_unet | no_aug | 64 | 0.0822 ± 0.0307 | +0.0586, +0.1169, +0.0710 | -0.0450 ± 0.0344 |
| tiny_unet | no_aug | 128 | 0.0723 ± 0.0252 | +0.0540, +0.1011, +0.0619 | -0.0450 ± 0.0344 |
| tiny_unet | occlusion | 64 | -0.0091 ± 0.0517 | +0.0497, -0.0474, -0.0295 | 0.0612 ± 0.0537 |
| tiny_unet | occlusion | 128 | -0.0071 ± 0.0427 | +0.0407, -0.0413, -0.0208 | 0.0612 ± 0.0537 |
| tiny_unet | texture | 64 | -0.0317 ± 0.0510 | -0.0012, -0.0905, -0.0033 | 0.0057 ± 0.0139 |
| tiny_unet | texture | 128 | -0.0264 ± 0.0540 | +0.0046, -0.0887, +0.0049 | 0.0057 ± 0.0139 |
| tiny_unet | viewpoint | 64 | -0.0083 ± 0.0655 | +0.0295, -0.0840, +0.0295 | 0.0137 ± 0.0558 |
| tiny_unet | viewpoint | 128 | -0.0102 ± 0.0582 | +0.0217, -0.0774, +0.0250 | 0.0137 ± 0.0558 |

## Descriptive correlations (all shifts)

| Model | Input | Shift | Drop ↔ gap ρ | Shift score ↔ real score ρ |
| --- | --- | --- | --- | --- |
| tiny_unet | 64 | blur_high | 0.8532 | -0.3117 |
| tiny_unet | 64 | blur_low | 0.8299 | -0.4636 |
| tiny_unet | 64 | lighting_high | 0.6987 | 0.3519 |
| tiny_unet | 64 | lighting_low | 0.8130 | -0.2351 |
| tiny_unet | 64 | noise_high | -0.4039 | -0.3844 |
| tiny_unet | 64 | noise_low | -0.5052 | -0.3260 |
| tiny_unet | 64 | occlusion_high | 0.6896 | -0.3909 |
| tiny_unet | 64 | occlusion_low | 0.4597 | -0.3221 |
| tiny_unet | 64 | texture_high | -0.4792 | -0.3961 |
| tiny_unet | 64 | texture_low | -0.3338 | -0.3506 |
| tiny_unet | 64 | viewpoint_high | 0.7208 | -0.3766 |
| tiny_unet | 64 | viewpoint_low | 0.7519 | -0.3766 |
| tiny_unet | 128 | blur_high | 0.8195 | -0.2974 |
| tiny_unet | 128 | blur_low | 0.8065 | -0.3987 |
| tiny_unet | 128 | lighting_high | 0.7416 | 0.4312 |
| tiny_unet | 128 | lighting_low | 0.8065 | -0.1805 |
| tiny_unet | 128 | noise_high | -0.3883 | -0.2922 |
| tiny_unet | 128 | noise_low | -0.4779 | -0.2377 |
| tiny_unet | 128 | occlusion_high | 0.6455 | -0.3143 |
| tiny_unet | 128 | occlusion_low | 0.4000 | -0.2532 |
| tiny_unet | 128 | texture_high | -0.4169 | -0.3013 |
| tiny_unet | 128 | texture_low | -0.2792 | -0.2779 |
| tiny_unet | 128 | viewpoint_high | 0.7130 | -0.2987 |
| tiny_unet | 128 | viewpoint_low | 0.7208 | -0.2987 |
| lite_fcn | 64 | blur_high | -0.4662 | -0.3792 |
| lite_fcn | 64 | blur_low | -0.5026 | -0.2455 |
| lite_fcn | 64 | lighting_high | -0.2442 | -0.3961 |
| lite_fcn | 64 | lighting_low | -0.3416 | -0.5377 |
| lite_fcn | 64 | noise_high | 0.1169 | 0.1182 |
| lite_fcn | 64 | noise_low | 0.2156 | 0.2312 |
| lite_fcn | 64 | occlusion_high | -0.0390 | 0.1078 |
| lite_fcn | 64 | occlusion_low | 0.0896 | 0.2182 |
| lite_fcn | 64 | texture_high | -0.4091 | -0.4545 |
| lite_fcn | 64 | texture_low | -0.1649 | 0.1078 |
| lite_fcn | 64 | viewpoint_high | -0.3675 | -0.0143 |
| lite_fcn | 64 | viewpoint_low | -0.4805 | -0.1429 |
| lite_fcn | 128 | blur_high | -0.4974 | -0.3818 |
| lite_fcn | 128 | blur_low | -0.5130 | -0.2260 |
| lite_fcn | 128 | lighting_high | -0.2870 | -0.4052 |
| lite_fcn | 128 | lighting_low | -0.3883 | -0.5597 |
| lite_fcn | 128 | noise_high | 0.0506 | 0.1039 |
| lite_fcn | 128 | noise_low | 0.1545 | 0.2532 |
| lite_fcn | 128 | occlusion_high | 0.0506 | 0.1468 |
| lite_fcn | 128 | occlusion_low | 0.1429 | 0.2455 |
| lite_fcn | 128 | texture_high | -0.4182 | -0.4468 |
| lite_fcn | 128 | texture_low | -0.2065 | 0.1351 |
| lite_fcn | 128 | viewpoint_high | -0.3156 | 0.0221 |
| lite_fcn | 128 | viewpoint_low | -0.4234 | -0.1052 |
