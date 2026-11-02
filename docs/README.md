# SYMBEX-1 documentation

How SYMBEX-1 works, how to use it and what it measured. The [main README](../README.md) has the summary; these pages
go into the details.

| Page | What it explains |
|---|---|
| [How it works](how-it-works.md) | Binary networks, counting bits instead of multiplying, folding everything into integer thresholds, bit-exactness |
| [CDT](cdt.md) | The compiler: from a PyTorch model to a checked C header, on the PC and on the chip |
| [Library](library.md) | lib_symbex: the layers, the hardware layer of each chip, how to add it to a project |
| [Results](results.md) | Every benchmark in one place: speed, flash and accuracy against INT8, with the charts |

The diagrams are written in [Mermaid](https://mermaid.js.org/) inside the pages (GitHub draws them). Charts live in
[`assets/`](assets/) and are generated from the result files of the benchmarks, so they can always be re-created.

## At a glance

Fully binary networks against the official INT8 ones (TensorFlow Lite Micro with Espressif's ESP-NN kernels, the
deployment path almost everyone uses, not the fastest INT8 possible: see [Results](results.md#how-fast-could-int8-be)),
the same network on the same chip:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speedup_dark.svg">
  <img alt="SYMBEX speed-up over TensorFlow Lite Micro INT8 per benchmark and chip" src="assets/speedup.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/flash_dark.svg">
  <img alt="Flash used by each model, SYMBEX vs. INT8" src="assets/flash.svg">
</picture>

The gap depends on the chip and on where the weights are. The ESP32-S3 has a SIMD unit that speeds up INT8, so there
the gap is smallest when INT8 can use it fully (ResNet-8: 1.33×, the MLP with its weights in RAM: 2.14×); on the classic
ESP32, without SIMD, SYMBEX is 8.6-11.3× faster, and the INT8 detector does not fit at all. Accuracy is in each
benchmark's README. To re-create the charts after a new
measurement: `python tools/dev/docs_charts.py`.
