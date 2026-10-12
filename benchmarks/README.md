# SYMBEX-1 benchmarks

Every number SYMBEX-1 reports comes from a benchmark in this folder, and every benchmark can be re-run
from scratch. Each one compares the SYMBEX model against an **official INT8 baseline produced outside
SYMBEX** (TensorFlow Lite converter + TensorFlow Lite Micro on the chip), starting from the **same FP32
model** and evaluated on the **same images**.

| Benchmark | Model | Data | Status |
|---|---|---|---|
| [`mlp_fashion/`](mlp_fashion/) | MLP 784-256-128-10 | Fashion-MNIST | complete: ESP32 and ESP32-S3, 3 seeds, official TFLite Micro |

| `poc_s3_bin_vs_int8.py` | one 256 x 512 layer, every kernel | synthetic | ESP32-S3: binary with the PIE multiplier 1.65x faster per weight than ESP-NN INT8 |

Being prepared for publication: the convolutional benchmarks of the SYMBEX v3 boolean engine (CIFAR-10 and
Visual Wake Words, already measured on the ESP32 and the ESP32-S3 against the MLPerf Tiny INT8 models), and a
VGG-style CNN through the CDT once its convolutional export is integrated.

## Layout of a benchmark

```
<benchmark>/
  README.md            what is measured, the configuration, and the results
  modelo.py            the original FP32 model (and how it is trained)
  symbex.toml          the CDT configuration of this benchmark
  <name>_symbex.py     SYMBEX side: FP32 model -> CDT -> export -> chip   (main environment)
  <name>_tflite.py     official INT8 side: TFLite converter -> TFLite Micro on the chip   (env_bench)
  results/             the .json result files (the only outputs that are published)
  runs/                everything generated (models, exports, firmware projects): not published
```

## Two environments

The SYMBEX side runs in the main environment of the repository (`.venv`, PyTorch).
The TFLite side needs TensorFlow, which does not support every Python version, so it has its own
environment, `env_bench`, created with a Python version TensorFlow supports (3.11 or 3.12):

```
cd benchmarks
python3.12 -m venv env_bench
source env_bench/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_bench.txt
```
(PyTorch is only used to read the FP32 weights and the dataset; the CPU version is enough.)

The TFLite side does not import anything from SYMBEX: the INT8 baseline is independent of the work it is
compared against.

## Hardware used

- ESP32-D0WD-V3 (classic ESP32, 240 MHz, no SIMD), ESP32-S3 (240 MHz, PIE SIMD)
- Lenovo ThinkPad P16 Gen 2: Intel i9-13950HX, NVIDIA RTX 4000 Ada Laptop GPU
- Firmware: ESP-IDF through PlatformIO, one core, timings measured on the chip with `esp_timer`
