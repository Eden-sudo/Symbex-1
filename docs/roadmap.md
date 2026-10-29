# SYMBEX-1 roadmap

The one roadmap for the team. Every agent reads it at the start of a session and updates it in the same pull request
as the work that changes it (`.claude/skills/symbex-sync/SKILL.md`). Day-to-day progress goes to the **Team sync**
issue, not here.

State: **done** (merged), **open** (pull request open), **next** (agreed, not started), **proposed** (an idea that
still needs the owner's agreement). Updated 2026-10-28.

## Shared target

All chip measurements are made on **Eden's ESP32-S3-DevKitC-1 N16R8** (16 MB flash, 8 MB octal PSRAM, 240 MHz),
connected to Eden's PC. Board runs are requested on the pull request (skill, section 2). Results from other boards are
kept, labelled with the board.

## LLM on the chip (Obed)

| Item | State | Where |
|---|---|---|
| Memory bandwidth per tier and 1-bit GEMV kernels on the S3 and the ESP32-CAM | open | PR #2, `benchmarks/edge_llm_probe` |
| Time `lib_symbex`'s S3 HAL on the same shapes (`symbex_dense_input_u8_mask`, `symbex_dense_input_u8_planes_bits` k = 4, 8), bit-exact | next, board run asked from Eden | PR #2 |
| Signed-int8 PIE kernel (`ee.vmulas.s8`) in `lib_symbex` for the S3, bit-exact against the reference | next | |
| Power: idle baseline and the 20 s energy phases with a USB meter, mJ per token | next, waiting for the meter | PR #2 |
| Train a 15-30M parameter 1-bit model (TinyStories, ~4k token vocabulary), run it from PSRAM or flash, measure tokens/s | proposed | `docs/llm_1bit_embedded.md` in PR #2 |
| microSD read speed on the ESP32-CAM with SDHC cards (secondary board) | proposed | PR #2 |

## Convolutional networks (Eden)

| Item | State | Where |
|---|---|---|
| ResNet-8 on CIFAR-10, ESP32 and S3, 3 seeds, sizing study | done | `benchmarks/conv2d/cnn_cifar_resnet8` |
| 1D convolutions, Pico 2, benchmarks grouped by network type | done | PR #4 |
| UniMiB-SHAR whole networks: SYMBEX vs ultracompactBNN (GVSoC), TFLite INT8 and lowbit on the boards | open | PR #11, `benchmarks/conv1d/har_ultracompact` |
| ResNet-8 sizing study: rank the variants on a validation split from the training data | next | `cnn_cifar_resnet8/README.md` |
| Rewrite the ternary kernel (12-20 ms in `mlp_fashion`, the slowest path) | proposed | `mlp/mlp_fashion/README.md` |
| Host the shared S3 and run the board requests | ongoing | |

## FOMO detection (Abraham)

| Item | State | Where |
|---|---|---|
| Binary person detector on Pascal VOC, ESP32 and S3, sizing study | done | `benchmarks/conv2d/fomo_voc_person` |
| Detection score tests, `.u32` labels, classic ESP32 run, sizing by validation F1, one fixed random validation split | done | PRs #5, #6, #8, #9, #10 |
| Re-measure the published results with the new validation split (they were calibrated on the previous one) | next | `fomo_voc_person/README.md` |

## Not assigned

| Item | State | Where |
|---|---|---|
| Re-measure `mlp_fashion` and `mlp_digits` with the current engine | next | `benchmarks/README.md` |
| Energy for every benchmark (no claims until a meter reading exists) | waiting for a meter | |

Take an unassigned item by saying so in the **Team sync** issue and moving it to your track here.
