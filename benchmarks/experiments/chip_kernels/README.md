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
| `stem_i8_pico_bench.py` | where does the int8 stem's time go on the Pico 2, and can it go faster? | 1.4-1.6× faster stem, now in lib_symbex (below) |
| `dense_short_bench.py` | why is a first dense layer with few inputs (19 sensors → 64) slow? | 2.2-2.5× faster, now in lib_symbex (below) |
| `psram_bw_bench.py` | is tiny_llm on the ESP32-S3 limited by the PSRAM or by the kernels? | the head and the attention scores by the PSRAM, the attention values by plain C, the linear layers by both (below) |

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

## The int8 stem on the Pico 2 (`stem_i8_pico_bench.py`, Nov 6)

**Question.** On the Pico 2 the int8 stem took ~20 % of ResNet-8 (M33 14.6 ms, Hazard3 19.1 ms). Is the time in
building the 27-byte window per pixel, or in the multiplies? Can a kernel without windows do better?

**What we did.** Timed the library's kernel, its two halves alone (`window`, `dots`) and two candidates: `ring` (3
padded rows, no window, one pixel at a time) and `rows` (the same rows, then per channel and kernel row the 9 weights
stay in registers while 16 output pixels stream past; M33: the strip split once into SXTB16 halves, 5 SMLAD per
pixel; Hazard3: 2 channels per pass). Weights in flash, random, every kernel checked against plain C on the chip.

**Result (measured, best of 7, exact, code in RAM; ms):**

| Core | Shape | library | window alone | dots alone | ring | **rows** | |
|---|---|---|---|---|---|---|---|
| M33 | ResNet-8 stem (32×32, 16 ch) | 14.57 | 2.04 | 13.68 | 12.49 | **9.21** | 1.58× |
| M33 | 40 ch, stride 2 | 8.10 | 0.52 | 8.30 | 8.34 | **5.69** | 1.42× |
| M33 | 48×48, 24 ch | 45.79 | 4.61 | 45.02 | 41.84 | **30.32** | 1.51× |
| Hazard3 | ResNet-8 stem | 19.05 | 1.78 | 17.23 | 21.45 | **12.29** | 1.55× |
| Hazard3 | 40 ch, stride 2 | 11.08 | 0.45 | 10.66 | 14.22 | **7.74** | 1.43× |
| Hazard3 | 48×48, 24 ch | 61.71 | 4.04 | 57.72 | 72.16 | **41.16** | 1.50× |

Code in flash gives the same times within 2 % (`results/stem_i8_pico_bench_pico2_arm_flash.json`, measured after the
move: there `library` already runs `rows`, 9.17 ms).

**Answer.** The multiplies are the time (~94 % of the M33 stem), not the window (~2 ms). Keeping the weights in registers and the
pixels pre-split is what pays. `rows` is in lib_symbex as `stem3_rows` (`symbex_bool.c`), the 3×3 int8 stem of every
board but the S3 (which has its PIE kernel): rows up to 64 wide, the window path past that; same bits always
(`run_tests.py`, 6 HALs, rows up to 80 wide, sanitizers). Whole ResNet-8 (int8 stem + Bop + pool2), 3 seeds, exact:
M33 73.57 → 67.77 ms, Hazard3 65.52 → 57.94 ms, ESP32 69.31 → 62.52 ms (weights in flash; `../../conv2d/cnn_cifar_resnet8`).
Further: the M33 inner loop is 14 instructions per pixel and kernel row; assembly with software pipelining could go
lower (not tried).

```
python stem_i8_pico_bench.py --board pico2_arm --port /dev/ttyACM1 --code ram     # Cortex-M33
python stem_i8_pico_bench.py --board pico2_riscv --port /dev/ttyACM1 --code ram   # Hazard3
```

## Short first dense layers (`dense_short_bench.py`, Nov 7)

**Question.** A control network (19 sensor inputs → 64 → 64 → 3) measured on the ESP32-S3 spent 66 % of its time in
the 19 → 64 first layer, as slow with `symbex_dense_input_u8_bits` as with `_planes`. Why?

**What we found.** The 4-inputs-per-operation SWAR sum only took whole 32-input words: with fewer than 32 inputs
every input went one at a time. Two changes in `lib_symbex`, same bits:

- `symbex_swar_u8_masked_sum` (`ops/symbex_swar.h`): the partial last word also goes 4 inputs at a time.
- `symbex_dense_input_u8_bits` / `_mask`, layers of up to 64 inputs and at least 8 neurons: subset-sum tables. For
  each group of 4 inputs the 16 sums of its subsets are built once per image; each neuron then reads one table entry
  per 4 inputs. Any input alignment (the SWAR path needs 4-byte aligned inputs). 512 bytes of stack at most.

**Result (measured, exact against plain C on the chip, weights in flash, best of 7; `master` = before):**

| Layer | ESP32-S3 | ESP32 | Pico 2 M33 | Pico 2 Hazard3 |
|---|---|---|---|---|
| 19 → 64, `_bits` | 59.3 → **27.0 µs** (2.2×) | 66.6 → **31.0** (2.1×) | 97.8 → **38.8** (2.5×) | 92.9 → **39.4** (2.4×) |
| 64 → 64, `_bits` | 79.3 → **52.3 µs** (1.5×) | 91.9 → **57.1** (1.6×) | 136.3 → **77.0** (1.8×) | 127.7 → **73.0** (1.7×) |
| 64 → 64, input not aligned | 158.0 → **52.3 µs** (3.0×) | 178.5 → **57.1** (3.1×) | 256.8 → **77.0** (3.3×) | 251.2 → **73.0** (3.4×) |
| 64 → 3, 784 → 128 | same or up to 3 % faster (no tables below 8 neurons or above 64 inputs) | same | same | same |

One exception: 784 → 128 with an unaligned input on the M33, 6.00 → 6.65 ms (its code did not change: the
layout in flash behind the Pico 2's XIP cache, seen before). Code or weights in RAM give the same times on the S3. `_planes` is not changed (slower on these shapes except
784 → 128 on Hazard3, which has a popcount instruction).

Future work: the tables for wider layers (784 → 128 would need a 6 KB workspace; ESTIMATE ~1.3× from the 64-input
cost per group, not measured).

```
python dense_short_bench.py --port /dev/ttyACM0                      # ESP32-S3, the library in the work tree
python dense_short_bench.py --port /dev/ttyACM0 --lib-rev master     # the library before
python dense_short_bench.py --board pico2_arm --port /dev/ttyACM1    # esp32 /dev/ttyUSB0; pico2_riscv
```

## PSRAM or kernels: where tiny_llm's time goes (`psram_bw_bench.py`, Nov 12)

The 15.7 M-parameter model of `../tiny_llm` keeps its weights (1.77 MB binary, a 1.5 MB int8 head) and its KV
cache in the octal PSRAM of the ESP32-S3 N16R8. The same kernels ran on the same shapes with the data in PSRAM and
in internal RAM, so each part's memory cost is measured (best of 7, every PSRAM block read from a fresh copy, as
the model reads it once per token; `results/psram_bw_bench_s3_80.json` and `_120.json`):

| Test (one call) | KB | internal RAM | PSRAM 80 MHz | PSRAM 120 MHz | limited by |
|---|---|---|---|---|---|
| plain C read of 32-bit words | 1536 | 320 MB/s | 25.8 ms, 61 MB/s | 18.4 ms, 85 MB/s | the PSRAM |
| `symbex_u8_sum_bits_rows` 384 → 384 (a linear layer) | 18 | 0.267 ms (69 MB/s) | 0.465 ms (40 MB/s) | 0.412 ms (45 MB/s) | both, about half each |
| same, 384 → 1024 / 1024 → 384 | 48 | 0.713 / 0.651 ms | 1.234 / 1.176 ms | 1.093 / 1.050 ms | both |
| `symbex_s8_dot_rows`, the 4096 × 384 int8 head | 1536 | (512 rows: 452 MB/s) | 26.9 ms, 58 MB/s | 19.7 ms, 80 MB/s | the PSRAM only |
| attention scores, 6 heads, position 255 | 96 | 0.282 ms (348 MB/s) | 1.656 ms (59 MB/s) | 1.233 ms (80 MB/s) | the PSRAM only |
| attention values, 6 heads, position 255 (tinylm.c's int32 loop) | 96 | 5.141 ms (19 MB/s) | 6.325 ms (16 MB/s) | 6.016 ms | plain C: 7.7× slower than its PSRAM reads |

What it says about a token (138 ms measured, seed 42, `../tiny_llm`):

- The output head (28 ms) is pure PSRAM traffic: in internal RAM the same kernel runs 7.7× faster. A 1-bit head
  (196 KB, lib_symbex's 8-bit × 1-bit kernel) would read 8× fewer bytes: ~3-5 ms (ESTIMATE from the rates above;
  whether the model keeps its perplexity with a binary head is not measured).
- The linear layers (55.6 ms) are half kernel, half memory: 69 MB/s in internal RAM, 40 MB/s from PSRAM, close to
  the sum of the two times. The PSRAM at 120 MHz gains only ~10 % here.
- The attention scores (7.5 ms) are at PSRAM speed. The attention values (25 ms) are not: the plain C loop is 3.8×
  slower than the scores' PIE kernel over the same bytes, with the data in internal RAM or in PSRAM alike. A PIE
  kernel for the values could bring them near the scores' time: ~18 ms per token less (ESTIMATE).
- PSRAM at 120 MHz (`--speed 120`): +40 % raw bandwidth, head 26.9 → 19.7 ms, ~16 ms per token (ESTIMATE); it is
  experimental in ESP-IDF (octal PSRAM, crashes when the temperature drifts ~20 °C), needs the flash at 120 MHz too
  and ESP-IDF instead of the Arduino core tiny_llm's firmware uses. Measured here only to know its size.

```
python psram_bw_bench.py --port /dev/ttyACM0                 # PSRAM at 80 MHz (the default of every build)
python psram_bw_bench.py --port /dev/ttyACM0 --speed 120     # experimental 120 MHz, flash at 120 too
```
