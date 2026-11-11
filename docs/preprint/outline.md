# SYMBEX-1 preprint: outline

Working outline. Every claim lists the file that backs it (`results/` JSON, a README section or a doc). A claim with
no source does not go into the paper. Numbers are copied from the JSON at writing time, never from memory; the final
pass re-reads every file (`tools/dev/docs_charts.py` and the bench-checker do the same).

Target: arXiv (cs.LG, cross-list eess.SP), about 8 pages + appendix. Working title candidates:

- *SYMBEX-1: exact binary neural networks on commercial microcontrollers, measured against the INT8 toolchain*
- *One bit, exactly: an end-to-end chain for binary networks on microcontrollers*

## 1. Introduction

| Claim | Source |
|---|---|
| Microcontrollers: a few hundred KB of RAM, often no SIMD / FPU; INT8 with TensorFlow Lite Micro is the deployment path almost everyone uses | README "Why binary?"; docs/en/how-it-works.md |
| One bit per weight: 8x smaller than INT8, bit operations + count instead of multiply-accumulate; on a chip without SIMD the count is 11-17x cheaper than int8 MAC | `tools/chip_profiles/count_*.json` (count_bench.py), README "Why binary?" |
| Contribution 1: end-to-end chain (PyTorch -> integer folding -> C header -> chip) with **exactness**: C on the PC and on the chip equals the integer simulation on 100 % of the inputs, every model | every `chip_*.json` (`exact` / equality field), `tests/test_header_format.py`, docs/en/methodology.md |
| Contribution 2: same network, same commercial chip, against TFLite Micro INT8 (ESP-NN / CMSIS-NN), 3 seeds, test set once; three ISAs (Xtensa LX6 / LX7, Cortex-M33, Hazard3 RISC-V) | docs/en/methodology.md; results folders below |
| Contribution 3: where binary pays and where it does not: accuracy cost 0 to >10 points by task; speed-up largest without SIMD | README "Every benchmark, side by side", section 5 below |
| Contribution 4: the binary ResNet-8 loses by capacity, not training (study) | `benchmarks/experiments/training_gap/README.md`, `results/train_gap/*.json` |

## 2. Related work (two lines, honest)

Verified Nov 10 (arXiv pages, repositories). Read the PDFs before writing the paragraph; the "exact C = simulation"
and "vs INT8" columns are **to confirm from the PDFs**, absence in an abstract is not proof.

| Work | Chip / platform | Task | Code | vs INT8 | Notes |
|---|---|---|---|---|---|
| BinaryConnect / BNN (Courbariaux, Hubara 2015-16), XNOR-Net 2016, Bi-Real 2018, ReActNet 2020, Bop (Helwegen 2019) | method papers | ImageNet, CIFAR | yes | - | training methods; we use Bop as an option |
| FINN (Umuroglu 2017) | FPGA | MNIST, CIFAR | yes | - | hardware line |
| daBNN (Zhang 2019), Larq Compute Engine (Bannink 2020) | ARM64 phones | ImageNet | yes | yes (TFLite) | not microcontrollers |
| **eBNN**, McDanel, Teerapittayanon, Kung, EWSN 2017, arXiv 1709.02260, `kunglab/ebnn` | Intel Curie (Arduino 101), ~15 KB usable | MNIST (95 %) | yes: Chainer -> C header -> Arduino | no (vs FP32 BNN implementations) | **already an open end-to-end chain**: we do not claim "first end-to-end" |
| Garofalo et al., PULP-NN, Phil. Trans. R. Soc. A 378 (2164), 2020, DOI 10.1098/rsta.2019.0155; XpulpNN (DATE 2020) | PULP RISC-V (GAP8); XpulpNN = ISA extension | CIFAR-10 kernels | yes | yes (CMSIS-NN) | kernels, 8-bit and sub-byte; no compiler |
| Cerutti et al., ISLPED 2020, DOI 10.1145/3370748.3406588, arXiv 2101.04446 | GAP8 (RISC-V, commercial) | sound event detection | partial | vs FP32 and Cortex-M4 | energy per inference reported |
| **ultracompactBNN**, Daghero et al., ACM CF 2021, DOI 10.1145/3457388.3458656, arXiv 2205.12781, `francescodaghero/ultracompactBNN` | PULPissimo (RI5CY, 1 core) on GVSoC | HAR, UniMiB-SHAR + private "Walk" | yes | **no** (vs random forest and padded PULP-NN BNN) | our HAR comparison runs their library on GVSoC |
| Sakr et al., IEEE EDGE 2022 ("Memory efficient binary CNNs on microcontrollers"); **CBin-NN**, Electronics 2024, 13, 1624, DOI 10.3390/electronics13091624 | STM32F746 (Cortex-M7), commercial | CIFAR-10; robotic touch | yes | to confirm (CMSIS-NN?) | inference engine in C |
| xTern, Rutishauser et al., IEEE ASAP 2024, arXiv 2405.19065, DOI 10.1109/ASAP61560.2024.00049 | RISC-V cluster with a custom ISA extension | CIFAR-10 ternary vs 2-bit | yes | vs 2-bit | not a commercial chip |

Gap statement (what no previous work combines): an open chain whose output is **verified exact** on every model,
measured on **commercial chips of three ISAs** against the **official INT8 toolchain** on **several tasks** (2D images,
detection, 1D signals: HAR, ECG) with a **fixed methodology** (same architecture, 3 seeds, test once). eBNN is the
closest ancestor (open chain, MNIST, one chip, no INT8 baseline); CBin-NN the closest engine (commercial Cortex-M).

Obed's PR #36 (`docs/related-research`) lists eBNN, ultracompactBNN, Daghero 2022 (arXiv 2209.00839), the BNN survey
(arXiv 2004.03333) and daBNN: reuse those links.

## 3. Method

| Claim | Source |
|---|---|
| Binary layer: XNOR/AND + popcount, one integer threshold per output (scale, BatchNorm, sign folded) | docs/en/how-it-works.md |
| Input: thermometer bits, or int8 first layer (`stem_i8`); OR pooling fused; pooled int8 classifier (`pool2`) | docs/en/how-it-works.md, cnn_cifar_resnet8 README |
| Engines: standard (bit planes), vertical (bit-sliced, 128 windows per PIE register) | docs/en/vertical.md, `results/symbex/s3_vertical_layers.json` |
| HAL per chip: lut16 / SWAR / PIE / Zbb cpop; count benchmark per chip | docs/en/library.md, `tools/chip_profiles/*.json` |
| Compiler (CDT): config, search of widths and bits, training (Bop option), folding, export, validate, chip | docs/en/cdt.md |
| Exactness check: fidelity (PyTorch vs integer simulation) and exactness (C vs simulation, PC and chip) | docs/en/methodology.md |

## 4. Experimental setup

| Item | Source |
|---|---|
| Boards: ESP32 (Xtensa LX6, no SIMD), ESP32-S3 (LX7, PIE), Raspberry Pi Pico 2 (RP2350: Cortex-M33 + Hazard3) | docs/en/methodology.md |
| Baselines: TFLite Micro INT8 with ESP-NN (ESP32, S3) and CMSIS-NN (M33), reference kernels (Hazard3); ESP-DL quick check on the S3; ultracompactBNN on GVSoC | `results/tflite/*.json` per benchmark; README "What we measured" |
| Recipe: same architecture, same epochs / seeds 42 43 44 / split, validation carved from training, test once; mean +- sd | docs/en/methodology.md; `symbex_seeds_*.json` |
| Low-bit baselines INT4 / INT2 / ternary: our own code (`tools/lowbit`), exploratory, not an official toolchain | `benchmarks/mlp/mlp_fashion/results/lowbit/*.json` |

## 5. Results (one subsection per benchmark)

| Benchmark | Accuracy binary / INT8 | Speed, memory | Source files |
|---|---|---|---|
| ResNet-8, CIFAR-10 | 71.3 (x1 fully binary), 76.7 (int8 stem + pool2) / 87.0 (MLPerf Tiny INT8): 10-16 points | S3 vertical 9.70 vs 52.0 ms (5.4x); ESP32 62.5 vs 717.8 (11.5x); M33 67.8 vs 453 CMSIS-NN (6.7x); 24-29 vs 96 KB flash | `cnn_cifar_resnet8/results/symbex/symbex_seeds_*_w1_int8stem_pool2*.json`, `symbex_seeds_pc_w1.json`, `results/tflite/tflite_{esp32,s3,pico2_arm}.json` |
| FOMO person detection | F1 31.2 / 30.1: 0 points | 2.0x faster on the S3; only binary fits the ESP32; 41 vs 236 KB | `fomo_voc_person/results/symbex/symbex_seeds_*.json`, `results/tflite/tflite_s3_*.json` |
| MLP Fashion-MNIST (784-128-512-10) | 87.5 / 89.8: 2 points | S3 flash 0.42 vs 11.7 ms (28x), RAM 0.41 vs 0.89 (2.1x); 27 vs 187 KB | `mlp_fashion/results/symbex/*.json`, `results/tflite/tflite_s3_h128_512*.json` |
| MLP low-bit | INT2 / INT4 1-2 points above binary per byte; binary fastest corner | `mlp_fashion/results/lowbit/lowbit_study.json`, `lowbit_chip_s3_*.json` |
| HAR 1D (4 networks of Daghero) | INT8 ~9 points better (large) to 23 (smallest) | ESP32 5.3-10.4x, S3 1.47-1.90x; 1.07-1.37x fewer instructions than ultracompactBNN on GVSoC | `har_ultracompact/results/symbex/har_accuracy.json`, `har_gvsoc_*.json`, `results/ucbnn/ucbnn_gvsoc.json`, `results/tflite/*.json` |
| ECG MIT-BIH (AAMI, inter-patient) | tied within sd, two operating points; clinically weak (S PPV 24-33 %) | ESP32 6.0-13.4x, S3 1.04-1.59x, M33 2.0-3.0x, Hazard3 39-153x (reference kernels) | **on master only** (`benchmarks/conv1d/ecg_mitbih/results/`, PR #35): merge master into the docs branch first |
| Digits on an Arduino Uno | 94.5 % | 5.3 ms on a 16 MHz AVR, 3.4 KB flash | `mlp_digits` README |

Accuracy-cost table for the paper (one row per benchmark, from the files above): 0 (FOMO, ECG), 2 (MLP), 9-23 (HAR),
10-16 (ResNet-8).

## 6. Analysis

| Claim | Source |
|---|---|
| Binary pays most without SIMD; on the S3 beating ESP-DL needed the vertical engine (flash 1.52x; ESP-DL wins in RAM) | README "What we learned" 1; docs/en/results.md "How fast could INT8 be" |
| Capacity, not training: twin fits 92 % of train, binary 72 %; Bop +2.5, two-step +2.35, not additive; width > 2-bit activations | `training_gap/README.md`, `results/train_gap/*.json`, `results/small_gap/` |
| Size the network for binary: 16-32-128 +1.8 points at the same speed; narrow first MLP layer | `cnn_cifar_resnet8/results/sizing/`, `mlp_fashion/results/sizing/` |
| An 8-bit input deserves an int8 first layer (accuracy, and speed with SIMD): stem alone S3 1.79x, Hazard3 0.62x | `benchmarks/experiments/chip_kernels/` (int8_first_bench), cnn_cifar_resnet8 README |
| Per-layer cost: the int8 stem dominates on the Pico | `symbex_seeds_pico2_*_int8stem_pool2*.json` (per-layer fields) |

## 7. Limitations (written, not hidden)

- Fine-grained classification stays 10-16 points behind INT8 (CIFAR-10); HAR up to 23 on the smallest network.
- ESP-DL beats the vertical engine with the weights in RAM on the S3 (quick check, Nov 2, not 3 seeds).
- Energy not measured yet (meter on the way): if it arrives, one table; if not, future work.
- Low-bit baselines are ours, not an official toolchain.
- ECG: inter-patient accuracy limited by the data (few patients per rare class); not a clinical result.
- Binary RNN probe (local, not in the repo): fragile to state flips; a "limits" paragraph at most.

## 8. Reproducibility

- Repository, one script per benchmark (`*_symbex.py`, `*_tflite.py`), `tools/chip_test.py`, Docker image for GVSoC.
- Every result JSON in `benchmarks/*/results/`; `tools/dev/docs_charts.py` draws the figures from them.

## Figures (from docs_charts.py, regenerate before submission)

1. ResNet-8 accuracy vs latency Pareto (S3 and ESP32): `docs/assets/pareto_resnet8.svg`
2. MLP accuracy vs latency, binary vs INT8 / INT4 / INT2: `docs/assets/pareto_mlp.svg`
3. Speed-up per benchmark and chip: `docs/assets/speedup.svg`; flash per model: `docs/assets/flash.svg`
4. New: accuracy cost per benchmark (points) against speed-up; the chain diagram (PyTorch -> integers -> C -> chip)

## To do before prose

- [ ] Merge master into `docs/structure` (ECG results, PR #35) and regenerate the charts.
- [ ] Read the PDFs of eBNN, ultracompactBNN, CBin-NN, Cerutti: fill the "vs INT8" and "exact" columns.
- [ ] Decide the title and the author list.
- [ ] Energy: meter arrives -> one script, one table (ESP32, S3, Pico 2), same models.
