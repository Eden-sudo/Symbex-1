# Development tools

Tools that measure the chips, the library and the training, to decide with numbers: which way of counting bits a chip
should use, where the time of a network goes, and how fast the CDT trains. Run them from this folder.

## count_bench.py — the raw arithmetic of a chip

Every way of counting bits that the chip has (native popcount, 8- or 16-bit tables, SWAR, the ESP32-S3's SIMD unit...)
against int8 multiply-accumulate, on the same convolution pattern and several window sizes.

```
python count_bench.py                                   # this PC (x86)
python count_bench.py --board s3 --port /dev/ttyUSB0    # a board: esp32 or s3
```

Prints the cycles per output of every method and the "binary / int8" comparison, and saves the fastest method per window
size in `../chip_profiles/count_<chip>.json`: the HAL's choice per chip, and the latency predictor of the sizing studies
in `benchmarks/`, read these profiles.

## fine_profile.py — where the time of a network goes, on the chip

Builds an exported model (a run folder of the CDT or of a benchmark) with CPU cycle counters inside lib_symbex and runs
it on the board.

```
python fine_profile.py <export folder> --board s3 --port /dev/ttyUSB0 --plain     # latency per layer, no counters
python fine_profile.py <export folder> --board s3 --port /dev/ttyUSB0             # + time per library function
python fine_profile.py <export folder> --board esp32 --port /dev/ttyUSB0 --popcount lut16 --weights ram
```

`--plain` gives the latency to report (the normal library). Without it, every part of the library (thermometer, window
copy, bit count, thresholds, OR pool, OR shortcut, classifier) with its ms per image, % of the time, calls and cycles per
call; the counters' own cost is subtracted.

## train_bench.py — how fast the CDT trains, and what it uses

Trains the binary ResNet-8 of `benchmarks/cnn_cifar_resnet8` with the training step of the CDT's boolean method for a
few hundred steps, in several configurations, and measures images per second, GPU utilization and memory, CPU (the whole
process tree, data loader workers included), threads, processes, RAM, start-up time and total time.

```
python train_bench.py                                   # cpu loader, data on the GPU, batch 512, CUDA graphs, ...
python train_bench.py --parallel 1 2 3                  # several trainings at once
TORCHINDUCTOR_COMPILE_THREADS=4 python train_bench.py --parallel-config gpu+graphs --parallel 1 2 3
```

It measures speed only: before a configuration goes into the CDT, its final accuracy is checked with a full training.
What it found on the RTX 4000 Ada: data on the GPU 1.23×, CUDA graphs 2.45×; bf16, `channels_last` and batches of 512 do
not help a network this small; with CUDA graphs, several trainings at once add nothing (one already gets what the GPU
gives).

Results of all three: `runs/` (not published).

## The library tests

Not here, next to the library: `python ../../Symbex1_PIO/lib/lib_symbex/tests/run_tests.py` (every function of
lib_symbex against numpy, with four hardware layers; `--stress` adds sanitizers and boundary sizes), and `pytest` from the
repository root (the Python export and the C engine against numpy references).
