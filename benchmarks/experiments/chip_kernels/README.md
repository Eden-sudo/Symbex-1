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
