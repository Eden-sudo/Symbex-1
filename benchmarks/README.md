# SYMBEX-1 benchmarks

Every number SYMBEX-1 reports comes from a benchmark in this folder, and every benchmark can be re-run from scratch.
Each one compares the SYMBEX model against an **INT8 baseline produced outside SYMBEX** (TensorFlow Lite converter +
TensorFlow Lite Micro on the chip), on the **same images** and the **same chips**.

| Benchmark | Task | SYMBEX model | INT8 baseline | Status |
|---|---|---|---|---|
| [`cnn_cifar_resnet8/`](cnn_cifar_resnet8/) | image classification, CIFAR-10 | fully binary ResNet-8, and a version sized for binary | the official MLPerf Tiny ResNet-8 | complete: ESP32 and ESP32-S3, 3 seeds, sizing study |
| [`fomo_voc_person/`](fomo_voc_person/) | person detection (FOMO), Pascal VOC | fully binary detector | the same architecture, trained the same way | complete: ESP32 and ESP32-S3, sizing study |
| [`mlp_fashion/`](mlp_fashion/) | Fashion-MNIST | MLP, every layer at 1 bit | the same FP32 model, quantized | complete on ESP32 and ESP32-S3; being re-measured with the current engine |
| [`mlp_digits/`](mlp_digits/) | scikit-learn Digits | MLP, every layer at 1 bit | the same FP32 model, quantized | complete; being re-measured with the current engine |
| [`har_ultracompact/`](har_ultracompact/) | human activity recognition, 1D | binary 1D CNN | INT8, and the binary library of Daghero et al. (2021) in its own RISC-V environment | in preparation |
| [`experiments/`](experiments/) | kernels on the ESP32-S3 | `pie_swar_bench.py`, `pie_conv_bench.py` | | isolated measurements behind the library's choices |

## The rules every benchmark follows

- **Same architecture** whenever SYMBEX is compared with INT8: the official model, or the same network trained with the
  same recipe. A binary network resized for binary is compared with INT8 resized the same way.
- **Independent baseline:** the INT8 side imports nothing from SYMBEX.
- **Measured on the chip:** latency as the median over the test images (`esp_timer`, one core, 240 MHz); RAM measured on
  the chip; flash from the exported model.
- **Exact:** the SYMBEX model on the chip must give 100 % the result of its integer simulation on the PC, image by image
  (and the INT8 model is checked against the TFLite interpreter on the PC).
- **Several seeds** for SYMBEX, reported as mean ± standard deviation.
- **Test set once:** every choice (checkpoint, threshold, variant) is made on a validation split carved out of the
  training data. A review found places where the test set was used for a choice; those are being fixed and the
  affected figures re-measured.

## Layout of a benchmark

```
<benchmark>/
  README.md            what is measured, how, and the results
  model.py             the network (and its floating-point version where it applies)
  symbex.toml          the CDT configuration, for the benchmarks trained through the CDT
  <name>_symbex.py     SYMBEX side: train -> export -> check -> measure on the chip
  <name>_tflite.py     INT8 side: TFLite converter -> TFLite Micro on the chip
  <name>_sizing.py     the sizing study, where there is one
  results/             the .json result files (the only outputs that are published)
  runs/                everything generated (models, exports, firmware projects): not published
```

`clean.py` removes the build folders of every `runs/` (`python clean.py --apply`), keeping models and results.

## The environment

`env_bench` runs both sides, SYMBEX (PyTorch) and INT8 (TensorFlow), on the GPU. TensorFlow does not support every Python
version, so it is created with one it supports (3.11 or 3.12):

```
cd benchmarks
python3.12 -m venv env_bench
source env_bench/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "tensorflow[and-cuda]"
pip install -r requirements_bench.txt
python -c "import torch, tensorflow as tf; print(torch.cuda.is_available(), tf.config.list_physical_devices('GPU'))"
```
(Use the PyTorch CUDA version that matches your driver; the CPU versions also work, slower.)

The chips are programmed with PlatformIO (ESP-IDF and the Arduino framework). The 1D benchmark adds a Docker image with
the PULP RISC-V toolchain and the GVSoC simulator (`har_ultracompact/pulp/Dockerfile`).

## Hardware used

- ESP32-D0WD-V3 (classic ESP32, 240 MHz, no SIMD) and ESP32-S3 (240 MHz, PIE SIMD)
- Lenovo ThinkPad P16 Gen 2: Intel i9-13950HX, NVIDIA RTX 4000 Ada Laptop GPU
