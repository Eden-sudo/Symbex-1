# chip_kernels -- kernel measurements on the chips

Quick, separate microbenchmarks behind lib_symbex's choices. Each builds its own small firmware with random weights,
checks every method against plain C on the chip (otherwise its times are not trusted) and documents itself in its
docstring.

| Script | Question | Result |
|---|---|---|
| `pie_swar_bench.py` | a faster way to count bits with the ESP32-S3's multiplier? | 1.24-1.39× faster: lib_symbex's S3 count (`pie_swar`) |
| `pie_conv_bench.py` | the AND + PIE arithmetic of the convolutions on the ESP32-S3 | 1.36-1.49× faster than XOR + lut16 |
| `s3_vertical_count.py` | counting without popcount: vertical (bit-sliced) sums on the S3 | the basis of the vertical engine (`docs/vertical.md`) |
| `s8_dot_rows_chip_test.py` | is the S3 PIE kernel `symbex_s8_dot_rows` exact at every alignment? | 9,792 cases, weights in RAM and flash: all equal to C |
| `int8_first_bench.py` | does an int8 first layer speed up mlp_fashion and ResNet-8, as it does conv1d? | below |
| `stem_i8_pie_bench.py` | can the int8 stem use PIE better on the ESP32-S3? | 5.7× faster stem, now in lib_symbex (below) |

Results in `results/`; the generated firmware projects in `runs/` (not in git).

## An int8 first layer for ResNet-8 (`int8_first_bench.py`)

ResNet-8's stem alone (32×32, 3×3, 16 channels), random weights, every method checked against plain C on the chip,
weights in RAM, best of 7 runs (results in `results/int8_first_bench_<board>.json`):

| Chip | thermometer + binary stem | int8 stem | int8 / thermometer | int8 dot products with |
|---|---|---|---|---|
| ESP32-S3 | 8.13 ms | **4.55 ms** | **1.79× faster** | PIE, 16 per instruction |
| Pico 2, Cortex-M33 | 15.47 ms | **13.72 ms** | **1.13× faster** | SMLAD, 2 per instruction |
| Pico 2, Hazard3 | **11.66 ms** | 18.94 ms | 0.62× | plain C |
| ESP32 | **13.35 ms** | 15.44 ms | 0.86× (flash: 0.80×) | plain C |

The int8 stem wins only where the chip multiplies bytes in SIMD; it is also more accurate (ResNet-8 on CIFAR-10, 3
seeds: 75.6 % against 71.2 % on validation, `resnet8_symbex.py --variant`). It is now an option of the CDT's `boolean` method and of lib_symbex
(`symbex_bool_stem_i8`); the whole network with it is measured in `../../conv2d/cnn_cifar_resnet8`.

```
python int8_first_bench.py --port /dev/ttyACM0                      # ESP32-S3: weights in flash, then in RAM
python int8_first_bench.py --board esp32 --port /dev/ttyUSB0
python int8_first_bench.py --board pico2_arm --port /dev/ttyACM1    # Cortex-M33; pico2_riscv: Hazard3
```

## The int8 stem on PIE (`stem_i8_pie_bench.py`, Nov 5)

**Question.** The int8 stem is the most accurate first layer for ResNet-8 (75.6 % against 71.2 %), but on the
ESP32-S3 it was the slow part of the vertical engine (4.5 ms of 13.4). Can PIE do it faster?

**What we did.** The library's kernel built a 27-byte window per pixel and one dot product per channel. The new
kernel turns the loops around: the 16 lanes of a PIE register are 16 output channels of one pixel
(`ee.vsmulas.s8.qacc`, a ring of 3 padded rows, the 20-bit accumulator loaded with the threshold). It refuses a
layer that could leave 20 bits (|threshold| + 128·Σ|w| > 524287; real models are near 250k) and the library's
kernel runs instead.

**Result (measured, exact against plain C on the chip, best of 7, weights in flash or RAM alike):**

| Shape | library kernel | PIE, 16 channels per instruction | |
|---|---|---|---|
| ResNet-8 stem (32×32, 16 ch) | 4.19 ms | 0.73 ms | 5.7× |
| 40 ch, stride 2 | 2.15 ms | 0.85 ms | 2.5× |
| 48×48, 24 ch | 12.65 ms | 3.07 ms | 4.1× |

**Answer, and what went into lib_symbex (S3 HAL only, `src/hal/esp32s3`):**

- `symbex_s8_stem3_bits`: this kernel, used by the standard engine's `symbex_bool_stem_i8` for 3×3 stems.
- `symbex_s8_stem3_planes`, for the vertical engine: the 16 lanes are 16 *pixels* of one channel instead, so the
  16 signs go straight into the channel planes (no HWC map, no transpose). Stride 1, width a multiple of 16 up to 128.
  The vertical stem goes from 0.92 to 0.72 ms.
- Both return −1 on what they cannot run, and the portable kernel runs: same bits always (`run_tests.py`, 6 HALs).

Whole ResNet-8 (int8 stem + Bop + pool2), 3 seeds, every image equal to the simulation: standard engine 35.77 →
32.33 ms, vertical 13.37 → 9.73 ms, prepared 9.78 → 6.03 ms (details: `../training_gap`). The other boards keep
their kernels (checked, seed 42, exact: ESP32 and both Pico 2 cores the same or up to 1.4 % faster).

```
python stem_i8_pie_bench.py --port /dev/ttyACM0
```
