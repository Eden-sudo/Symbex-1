# SYMBEX-1 roadmap

The one roadmap for the team. Every agent reads it at the start of a session and updates it in the same pull request
as the work that changes it (`.claude/skills/symbex-sync/SKILL.md`). Day-to-day progress goes to the **Team sync**
issue, not here.

State: **done** (merged), **open** (pull request open), **next** (agreed, not started), **proposed** (an idea that
still needs the owner's agreement). FOMO and hardware policy updated 2026-10-30; other track states retain their
2026-10-28 snapshot unless noted.

## Shared target and active classic ESP32 target

Both targets remain active, as confirmed by Obed on 2026-10-30:

- **Abrahan's classic ESP32 without PSRAM:** FOMO results measured on his board, including PR #15's ESP32-D0WD-V3
  runs, remain a first-class track. Re-measure the new validation split on that classic ESP32.
- **Shared ESP32-S3 with microSD on Eden's PC:** every S3 measurement goes through a board request (skill, section 2),
  the existing physical-access approval and an exclusive board turn. Eden reported ESP32-S3-DevKitC-1 N16R8
  (16 MB flash, 8 MB PSRAM) in PR #2; reconfirm the actual module/configuration before a run. The microSD module,
  pinout, SPI/SDMMC interface, card capacity and configuration remain unverified; confirm them before SD-specific
  work. No integrated slot or card capacity is assumed.

Label each result by board, owner, commit, environment and raw log. Existing classic results are not future S3
measurements; request S3 FOMO numbers separately rather than replacing or relabelling classic experiments.

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

## FOMO detection (Abrahan)

| Item | State | Where |
|---|---|---|
| Binary person detector on Pascal VOC, ESP32 and S3, sizing study | done | `benchmarks/conv2d/fomo_voc_person` |
| Detection score tests, `.u32` labels, classic ESP32 run, sizing by validation F1, one fixed random validation split | done | PRs #5, #6, #8, #9, #10 |
| Classic ESP32 without PSRAM: lut16 weights in RAM and swar over seeds 42/43/44, plus flash/RAM captures per step | open (verified 2026-10-30) | PR #15, head `b8ed19e`; measurements reported by Abrahan on his ESP32-D0WD-V3, previous validation split |
| Re-measure the published results with the new validation split on Abrahan's classic ESP32 (previous results used the old split) | next | `fomo_voc_person/README.md` |
| Measure FOMO on the shared S3, with board-labelled results separate from the classic measurements | next, board request required | Request Eden's approved board turn; skill section 2 |

Coordination evidence: [Abrahan reported](https://github.com/Eden-sudo/Symbex-1/pull/12#issuecomment-6526390086)
running the original skill steps 1.2 and 1.3 read-only on Linux/zsh: Team sync resolved to #13 and mention discovery
found #15. This is collaborator-reported evidence for the earlier commands, not our execution of the revised ones.

## Not assigned

| Item | State | Where |
|---|---|---|
| Re-measure `mlp_fashion` and `mlp_digits` with the current engine | next | `benchmarks/README.md` |
| Energy for every benchmark (no claims until a meter reading exists) | waiting for a meter | |

Take an unassigned item by saying so in the **Team sync** issue and moving it to your track here.
