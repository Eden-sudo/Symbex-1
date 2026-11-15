# vertical1d -- the vertical engine for 1D convolutions on the ESP32-S3 (`vertical1d.h`, `vertical1d.py`)

**Question:** on the ESP32-S3 the vertical (bit-sliced) engine made the 2D ResNet-8 about 3× faster than the standard
one (`docs/vertical.md`). The 1D networks (HAR, ECG) still run on the standard 1D layer there, and their margin over
TFLite Micro INT8 on the S3 is the smallest of all boards (1.47-1.90× for HAR). Does the same idea speed up the binary
1D convolutions?

**Answer:** yes, on the long layers. The HAR networks' second convolution (120-137 steps) runs **2.0-4.6× faster**,
the whole network **1.26-1.87× faster** with the int8 first layer, every run **exact** (100 % equal to the
simulation, 3 seeds). Against TFLite INT8 on the S3 the margin grows from **1.47-1.90×** to **1.85-3.52×**. The
price is RAM: a work block of 7.4-18.7 KB (`l`: 2.8 → 21.9 KB in total), still less than TFLite's whole RAM on
`s` / `m` / `l` (10-21 KB vs 14-29 KB), more on `xs` (9.0 vs 6.1 KB). Short layers (16
steps: the third convolution) are slower vertically (0.31 → 0.74 ms), so they stay on the standard layer.

## What

`vertical1d.h` is lib_symbex's vertical engine (`symbex_bool_vertical.c`) turned to one dimension, with the same
`struct SymbexBool1D`, the same time-major streams in and out and the same output bits as `symbex_bool1d_conv`:

- the input stream becomes channel planes (bit t of plane c = channel c at step t), then one shifted plane per kernel
  tap and channel: **128 conv steps per 128-bit register**, so the count needs no sum across the register;
- per output channel, the planes of its 1-bit weights are summed with carry-save adders (Harley-Seal), the threshold
  is a bit-sliced add, the OR pool folds neighbouring lanes, and the planes go back to the stream;
- on the S3 the counting runs on PIE (128-bit logic); elsewhere the same steps in portable C.

Limits: stride 1, kernel × in_ch ≤ 1016. The layers of one network share one work block (the list table sits first).

`vertical1d.py make <run>` takes a CDT `bool1d` run folder, embeds `vertical1d.h` into a copy of its `model_symbex.h`,
swaps the chosen layers to `v1d_conv` (`--layers`), counts the work block in the buffers and checks the C against the
run's integer simulation on every validation window. `vertical1d.py test` compares `v1d_conv` with
`symbex_bool1d_conv` on random layers (channels 1-70, kernels 1-17, pools 1-8, thresholds at the extremes) with
AddressSanitizer / UBSan: 5,000 layers, 0 different. Then `tools/chip_test.py` on the new folder, the same flags as the
source run (`--board s3 --usb --popcount swar --weights flash`). `vertical1d.py report` writes
`results/vertical1d.json`.

The networks are the HAR benchmark's (`../../conv1d/har_ultracompact`, xs / s / m / l, Adam, the runs of its README
tables), the same trained weights for both engines: only the second convolution is vertical (`--layers 1`).

## Result

ESP32-S3, weights in flash, 1,000 validation windows, median latency in ms, 3 seeds (42, 43, 44), mean ± standard
deviation. Every run 100 % equal to the simulation (PC and chip).

| Network | first layer | standard | **vertical** | speed-up | TFLite INT8 (seed 42) | vertical vs TFLite |
|---|---|---|---|---|---|---|
| xs | int8 | 0.487 ± 0.001 | **0.386 ± 0.002** | 1.26× | 0.715 | 1.85× |
| s | int8 | 1.832 ± 0.002 | **0.981 ± 0.005** | 1.87× | 2.913 | 2.97× |
| m | int8 | 3.757 ± 0.003 | **2.030 ± 0.011** | 1.85× | 7.138 | 3.52× |
| l | int8 | 3.835 ± 0.000 | **2.263 ± 0.007** | 1.69× | 7.107 | 3.14× |
| xs | binary (bits) | 0.829 ± 0.001 | 0.741 ± 0.011 | 1.12× | | |
| s | binary (bits) | 2.359 ± 0.003 | 1.514 ± 0.009 | 1.56× | | |
| m | binary (bits) | 4.865 ± 0.003 | 3.132 ± 0.007 | 1.55× | | |
| l | binary (bits) | 4.827 ± 0.002 | 3.264 ± 0.010 | 1.48× | | |

Per step (seed 42, int8 first layer, ms): the second convolution goes 0.207 → 0.101 (`xs`), 1.087 → 0.234 (`s`),
2.484 → 0.759 (`m`), 2.325 → 0.745 (`l`); the first layer and the classifier do not change; the third convolution
(on the standard layer) takes 0.006 ms more. With the third convolution vertical too (`l`, 16 steps in one 128-lane
tile): 0.31 → 0.735 ms, whole network 2.707 ms instead of 2.263.

RAM (`l`, measured on the chip): 2.8 KB standard, 21.9 KB vertical (work block 18.7 KB: the shifted planes of one
tile 7.8 KB, the list table 5.1 KB, the channel planes, counters and list the rest); `xs` 1.7 → 9.3 KB, `s` 1.9 → 10.5 KB, `m` 2.8 → 21.9 KB. Flash is the same
(the weights do not change).

### Whole RAM against TFLite Micro, weights in flash and in RAM

TFLite's RAM counted the way `chip_test.py` counts SYMBEX's (arena + interpreter objects + input buffer + heap taken
while building + stack used; `har_tflite.py`, new memory request `0x5A` in the firmware of
`mlp_fashion/fashion_tflite.py`), 3 seeds, every run 100 % equal to the PC interpreter
(`../../conv1d/har_ultracompact/results/tflite/memory/`). SYMBEX = int8 first layer, vertical second convolution.
Times in ms (mean over 3 seeds), RAM in KB.

| Network | weights | SYMBEX vertical | TFLite INT8 | speed-up | SYMBEX RAM | TFLite RAM |
|---|---|---|---|---|---|---|
| xs | flash | 0.386 | 0.665 | 1.72× | 9.0 | 6.1 |
| s | flash | 0.981 | 2.922 | 2.98× | 10.3 | 14.2 |
| m | flash | 2.030 | 7.123 | 3.51× | 21.4 | 29.2 |
| l | flash | 2.263 | 7.091 | 3.13× | 21.4 | 28.7 |
| xs | RAM | 0.386 | 0.677 | 1.75× | 9.8 | 13.4 |
| s | RAM | 0.981 | 2.919 | 2.98× | 13.4 | 34.8 |
| m | RAM | 2.030 | 6.355 | 3.13× | 27.2 | 71.0 |
| l | RAM | 2.263 | 6.114 | 2.70× | 27.9 | 71.2 |

The weights in RAM do not speed up SYMBEX on the S3 (the same times as in flash, ± 0.001 ms) and cost their size in
RAM; TFLite gains up to 14 % with them. This TFLite measurement is newer than the seed-42 column of the table above
(`xs` 0.665 vs 0.715 ms, same model: the cause of the difference is not checked), so `xs` is 1.72× here, 1.85× there.

## Open

- The binary first layer as ±1 bytes (`_bytes` runs, the faster all-binary variant on the S3) is not measured yet.
- ECG (`../../conv1d/ecg_mitbih`, the same layer type, 248 steps) and the other boards (the vertical engine gains
  1.14-1.29× on the ESP32 and the M33 in 2D; the Hazard3, with its native popcount, keeps the standard engine).
- Less RAM: the shifted planes could be built a few taps at a time.
- Into lib_symbex and the CDT (`--engine vertical` for `bool1d`) only after that.
