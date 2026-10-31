# fomo_voc_person — a fully binary person detector against the same detector in INT8

Person detection on microcontrollers, FOMO-style (Edge Impulse's "Faster Objects, More Objects"): the image is split
into a 12 × 12 grid and every cell answers one question — *does the center of a person fall here?* — which locates every
person in the image. The SYMBEX detector is binary end to end; the reference is **the same architecture** trained the
same way and quantized to INT8 with TensorFlow Lite, measured with TensorFlow Lite Micro on the same boards.

| | SYMBEX binary | TFLite INT8 |
|---|---|---|
| Input | 96 × 96 RGB, thermometer of 8 bits per color | 96 × 96 RGB, 8-bit pixels |
| Network | stem 3×3 stride 2 → 3 stages (48, 24, 12) → 1×1 head per cell | the same |
| Weights and activations | **1 bit** | 8 bits |
| Shortcuts | OR of bits | sum |
| Head | the integer classifier at every cell (`symbex_bool_cell_head`) | 1×1 convolution |
| Arithmetic on the chip | **counting bits: no multiplication at all** | 8-bit multiply-accumulate |

Data: Pascal VOC, class *person* — training on VOC 2007 + 2012 trainval (16,551 images, 13,256 persons), test on VOC 2007
test (4,952 images, 4,528 persons). Score: precision, recall and F1 of the detected person centers (a detection matches a
person whose center is within 1.5 cells). Each model gets its own decision threshold, calibrated on 1,000 training images
kept out of the training, chosen at random with a fixed seed (`model.SPLIT_SEED`): the same images for every model and
seed, binary and INT8. On the chip the binary threshold costs nothing (it is folded into the integer bias of the head).
The published results were calibrated on the previous split (the last 1,000 training images, all from VOC 2012) and are
to be re-measured with this one.

## Results (ESP32-S3, 240 MHz)

| | Original shape (32-64-128) | | **Resized for binary (32-32-128)** | |
|---|---|---|---|---|
| | **SYMBEX binary** | TFLite INT8 | **SYMBEX binary** | TFLite INT8 |
| F1 (VOC 2007 test persons) | **31.0 %** | 30.2 % | **31.2 ± 0.7 %** | 30.1 ± 0.2 % |
| Latency | **160.8 ms** | 379.1 ms | **126.3 ms** | 256.9 ms |
| Speed-up | **2.36×** | | **2.03×** | |
| Flash | **52.3 KB** | 315.7 KB | **41.2 KB** | 236.1 KB |
| RAM (with the 27 KB input image, as in the arena) | **81 KB** * | 223.3 KB (arena) | **79 KB** | 222.6 KB (arena) |
| Same result as its PC reference | **100 %** | 90 % | **100 %** | 88 % |

\* measured on the classic ESP32 with the same setting (no counting table): the buffers depend only on the shape.

**On the classic ESP32** (ESP32-D0WD-V3, 240 MHz, one core, no PSRAM) the resized binary detector runs in **226.3 ms**
(32-32-128, 3 seeds: 226.31-226.35 ms; lut16 counting, weights in flash, 143 KB of RAM), every seed 100 % equal to its
simulation on 1,000 test images; the original shape, in **296.2 ms** with the same setting. With the weights in RAM:
210.7 / 275.0 ms (184 / 197 KB of RAM); without the 64 KB counting table (swar): 274.5 / 364.6 ms (79 / 81 KB); the
32-32-128 figures over 3 seeds. Flash is the number to report (a real deployment keeps the weights there); RAM is 7 %
faster. Step by step (`tools/dev/fine_profile.py --plain`, seed 42, 100 images; captures for 32-32-128 in
[flash](results/fine_profile_esp32_lut16_flash_c32-32-128_b1-2-2_w40_e100_s42.txt) and
[RAM](results/fine_profile_esp32_lut16_ram_c32-32-128_b1-2-2_w40_e100_s42.txt), for 32-64-128 in
[flash](results/fine_profile_esp32_lut16_flash_c32-64-128_b1-2-2_w40_e100_s42.txt) and
[RAM](results/fine_profile_esp32_lut16_ram_c32-64-128_b1-2-2_w40_e100_s42.txt)), the steps with 11-21 KB of weights gain
10-11 % in RAM and those with 2-6 KB 5-7 %. We attribute the larger gain to the flash cache, as in `../cnn_cifar_resnet8`, but moving the weights does not
isolate the cause, and the 5 % on the smallest steps is not explained (perhaps the code sharing the same cache; not
measured). The
INT8 detector does not fit: its arena overflows the chip's RAM by 97 KB (original shape); for 32-32-128, by 87.9 KB
even with the arena cut to the 223 KB it needs (94.9 KB with the default 230 KB; `fomo_tflite.py --arena-kb 223`).

The resized shape is reported over 3 training seeds on both sides (42, 43, 44); the original shape, one seed. The
floating-point version of the same network reaches 32.0 % F1: **binarization costs about one point here**, and the
binary detector is as accurate as the INT8 one, or slightly more. Person detection in VOC at 96 × 96 is hard for networks this small — the
limit is the task, the same for every precision.

**On the Raspberry Pi Pico 2** (RP2350, 150 MHz, weights in flash, 3 seeds, every seed 100 % equal to its simulation):
32-32-128 runs in **338.5 ms** on the Hazard3 RISC-V cores (native popcount, 79 KB of RAM) and **416.4 ms** on the
Cortex-M33 (64 KB counting table, 143 KB). TFLite INT8 on the Pico 2 (CMSIS-NN on the M33) is not measured yet.

### Why detection binarizes so well here

- **Every cell answers yes or no.** Whether certain shapes are present in a region is exactly what 1-bit features express;
  classifying ten similar classes (CIFAR-10) needs the fine detail that 1 bit loses first.
- **FOMO turns detection into classification per cell.** It never asks the network for precise numbers (box
  coordinates), the part binary networks do worst.
- **The calibrated threshold.** A binary network tends to mark too many cells; moving one integer bias fixes it.

## Sizing the binary detector

`fomo_sizing.py` trains the same structure with other numbers of blocks and channels per stage (single seed: differences
of 1-2 F1 points are noise) and predicts the latency from the chip profiles of `tools/dev/count_bench.py`, calibrated with the
measured detector — the prediction for 32-32-128 was 125.5 ms, the chip measured 126.3 ms.

| Channels / blocks | Validation F1 (chooses) | Test F1 (reported) | S3 ms (predicted) | |
|---|---|---|---|---|
| 32-64-128 / 1-2-2 (the benchmark detector) | 30.4 % | 31.0 % | 160.8 | |
| **32-32-128 / 1-2-2** | **31.8 %** | **31.8 %** (3 seeds: 31.2 ± 0.7 %) | **125.5** | the best on validation, and 1.28× faster |
| 32-64-64 / 1-2-2 | 29.2 % | 31.0 % | 137.5 | |
| 32-32-32 / 1-2-2 | 29.9 % | 29.8 % | 96.3 | 1.67× faster |
| 16-32-64 / 1-2-2 | 28.9 % | 28.5 % | 59.7 | 2.69× faster |
| 32-64-128 / 2-2-2 | 31.2 % | 30.8 % | 197.7 | +0.8 on validation for 1.23× the time: within the noise of one seed |
| 32-64-128 / 1-2-1 | 27.6 % | 26.8 % | 140.3 | the second block of the last stage matters |

The variants are ranked on the **validation F1**: the 1,000 training images `fomo_train.py` keeps out of the training
to calibrate the threshold. The test F1 is reported, never used to choose. Ranked that way the choice holds:
32-32-128 / 1-2-2 is also the best of the 27 width variants on validation. (An earlier version of the study ranked on
the test F1; the chosen shape was then trained with two more seeds that took no part in the choice: 30.9 % on average,
against 31.8 % for the seed that chose it.) The validation F1 is slightly optimistic, because the threshold is
calibrated on those same images; it is equally so for every variant, which is enough to rank them, but the final
measure is still the test F1.

The middle stage needs few channels, the last stage and the stem pay their cost. The best shape differs from the
classifier's (`../cnn_cifar_resnet8`): it depends on the task, which is why it is measured and not assumed.

## Where the time goes (ESP32-S3, 32-32-128, 126.3 ms)

| Step | ms | % |
|---|---|---|
| thermometer + stem 24→32, 96×96 | 29.9 | 23.7 |
| block 0: conv 32→32, 48×48 | 36.4 | 28.8 |
| block 1: conv 32→32, stride 2 + OR pool + 1×1 | 13.7 | 10.9 |
| block 2: conv 32→32, 24×24 | 9.5 | 7.5 |
| block 3: conv 32→128, stride 2 + OR pool + 1×1 | 13.6 | 10.8 |
| block 4: conv 128→128, 12×12 | 23.0 | 18.2 |
| per-cell head | 0.15 | 0.1 |

Inside the library (`tools/dev/fine_profile.py`): counting bits is ~73 % of the time, at the speed the benchmarks measure.

## Reproduce

From this folder (environment `env_bench`, with PyTorch on CUDA and TensorFlow):

```
python fomo_data.py                                          # Pascal VOC -> data/fomo_*.npz (once)
python fomo_train.py --channels 32-32-128 --blocks 1-2-2 --weight 40 --epochs 100
python fomo_export.py runs/c32-32-128_b1-2-2_w40_e100_s42    # integers, header, C check (100 %)
python ../../../tools/dev/fine_profile.py runs/c32-32-128_b1-2-2_w40_e100_s42 --board s3 --port /dev/ttyUSB0 --plain
python ../../../tools/dev/fine_profile.py runs/c32-32-128_b1-2-2_w40_e100_s42 --board esp32 --port /dev/ttyUSB0 --popcount lut16 --weights ram --plain  # weights flash|ram: the captures in results/
python fomo_symbex.py --channels 32-32-128 --seeds 3 --board esp32 --port /dev/ttyUSB0                   # train (if needed) + export + chip
python fomo_symbex.py --channels 32-32-128 --seeds 1 --board esp32 --port /dev/ttyUSB0 --weights ram
python fomo_symbex.py --channels 32-32-128 --seeds 1 --board esp32 --port /dev/ttyUSB0 --popcount swar
python fomo_symbex.py --channels 32-32-128 --seeds 3 --board s3 --port /dev/ttyACM0 --usb --popcount swar  # native USB
python fomo_symbex.py --channels 32-32-128 --seeds 3 --board pico2_riscv --port /dev/ttyACM0             # Pico 2 (Hazard3)
python fomo_symbex.py --channels 32-32-128 --seeds 3 --board pico2_arm --port /dev/ttyACM0               # Pico 2 (M33)
python fomo_tflite.py --channels 32-32-128 --blocks 1-2-2 --board s3 --port /dev/ttyUSB0
python fomo_train.py --fp32 --channels 32-64-128 --blocks 1-2-2    # the floating-point reference
python fomo_sizing.py --phase depth ; python fomo_sizing.py --phase width
```

| File | What it does |
|---|---|
| `fomo_data.py` | Pascal VOC → 96 × 96 images, 12 × 12 grids of person centers |
| `model.py` | the binary detector (the binary pieces of `../cnn_cifar_resnet8/model.py`), its float version, the F1 score |
| `fomo_train.py` | trains a detector (binary or `--fp32`) and calibrates its threshold |
| `fomo_export.py` | the binary detector into lib_symbex through the CDT exporter, checked against the simulation |
| `fomo_symbex.py` | train + export + measure on the board, per seed; summary in `results/symbex_seeds_<board>_<popcount>[_ram]_<name>.json` |
| `fomo_tflite.py` | the same architecture in Keras → INT8 → TFLite Micro on the boards |
| `fomo_sizing.py` | the sizing study |

## Future work

- **Higher accuracy for every precision:** a larger input (128 × 128), more training data of persons at a distance.
- **Boxes, the binary way:** box sides as thermometer outputs (yes/no questions decoded by counting bits) instead of numbers.
- **The CDT:** detection as a CDT task (today the training runs in `fomo_train.py`, the export and the chip go through the
  CDT exporter and tools).
- **Same training budget:** the INT8 reference follows the floating-point recipe (person weight 20, 60 epochs, where it had
  already plateaued); the binary detector used weight 40 and 100 epochs. Training INT8 with the binary recipe too
  closes this difference.

## References

FOMO (Edge Impulse, 2022) · The PASCAL Visual Object Classes Challenge (Everingham et al., IJCV 2010) · TensorFlow Lite
Micro (David et al., MLSys 2021) · Binarized Neural Networks (Hubara et al., NeurIPS 2016) · FINN (Umuroglu et al.,
FPGA 2017) · FracBNN (Zhang et al., FPGA 2021).
