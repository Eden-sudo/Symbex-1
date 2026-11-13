# Preprint outline: SYMBEX-1

Working outline (Nov 12, 2026). Every claim below points to the file that holds its number. A claim without a source
in the repository is marked **NOT IN REPO** and stays out of the paper until it is there.

## Framing (agreed Nov 10)

- SYMBEX-1 is an end-to-end framework for fully binary networks: a compiler (the CDT, Python) and a C engine
  (lib_symbex). Microcontrollers are its first target, not its limit.
- The gap it fills: an open chain from training to C, **exact** (the chip equals the integer simulation on 100 % of
  the inputs), measured with the same network on the same commercial chip against the official INT8 toolchain
  (TensorFlow Lite Micro), on several tasks, with one fixed methodology ([methodology](../methodology.md)).
- INT8 is the main comparison because it is what everyone deploys. INT8 and binary sit at different points:
  INT8 keeps accuracy, binary buys speed, flash and RAM. The accuracy cost of going binary goes from 0 to more than
  10 points depending on the task, and the paper says so.
- INT4 / INT2 / ternary (tools/lowbit) run on OUR kernels, not an official toolchain: they are approximations, shown
  as context, never as the objective comparison.

## Sections

1. Introduction: deployment on microcontrollers, the INT8 default, what binary promises and what it costs.
2. Related work ([related_work.md](related_work.md): protocol, table, gap): FINN (FPGA), daBNN and Larq Compute
   Engine (ARM64), McDanel 2017, Daghero 2021 and CBin-NN 2024 (MCUs; CBin-NN already beats TFLM / CMSIS-NN on one
   Cortex-M7 network). Our gap is breadth + verification: several chips / ISAs and tasks, per-input exactness, fixed
   methodology.
3. SYMBEX-1: the CDT (training recipes, integer simulation, export), lib_symbex (Boolean layers, HALs for ESP32,
   ESP32-S3 PIE, RP2350 M33 / Hazard3), the vertical engine.
4. Methodology: same architecture, validation split from training data, test set once, 3 seeds, same recipe,
   exactness check, weights in flash first.
5. Results: speed / flash / RAM per task and chip; accuracy cost; the S3 against ESP-DL (stronger INT8).
6. Limits and negative results: accuracy cost on fine-grained classes, ECG detectors, binary RNN, tiny LLM.
7. Conclusion and future work: energy, more chips, the int8-stem / binary-body hybrids.

## Claims and their sources

Paths relative to `benchmarks/`. "README" = that benchmark's README, which gives the command for each file.

| # | Claim | Number | Source | Status |
|---|---|---|---|---|
| 1 | ResNet-8 S3, fully binary vs MLPerf INT8 (TFLM), flash | 39.2 vs 52.0 ms, 1.33× | `conv2d/cnn_cifar_resnet8/results/symbex/symbex_seeds_s3_swar_w1.json`, `.../tflite/tflite_s3.json` | in repo |
| 2 | ResNet-8 classic ESP32 | 63.3 vs 717.8 ms, 11.3× | `.../symbex/symbex_seeds_esp32_lut16_w1.json`, `.../tflite/tflite_esp32.json` | in repo |
| 3 | ResNet-8 vertical engine S3 (default / compact / prepared) | 10.36 / 7.70 / 6.44 ms | `.../symbex/symbex_seeds_s3_swar_w1_vertical*.json` (+ `_ram_`) | in repo |
| 4 | Most accurate ResNet-8 (int8 stem + Bop + pool2): test accuracy | 76.68 ± 0.34 % | `.../symbex/symbex_seeds_pc_w1_int8stem_pool2.json` (`test_accuracy` per seed) | in repo, checked |
| 5 | Same, S3 standard / vertical | 32.33 ms / 9.73 / 7.09 / 6.03 ms | `.../symbex/symbex_seeds_s3_swar_w1_int8stem_pool2*.json` | in repo |
| 6 | ResNet-8 against ESP-DL (S3: 14.8 flash / 9.8 RAM; ESP32: 712 / 603 ms) | 1.43-1.52× (vertical) | `~/Proyectos/espdl_test` | **NOT IN REPO**: add the scripts + JSON or cite as a quick check |
| 7 | Pico 2 M33, ResNet-8 vs TFLM + CMSIS-NN | 453 / 211 ms vs 74.5 ms | `.../tflite/tflite_pico2_arm*.json`, `.../symbex/symbex_seeds_pico2_arm_lut16_w1.json` | in repo |
| 8 | MLP Fashion 784-128-512-10, S3 flash / RAM | 0.42 vs 11.67 ms (27.8×), 0.414 vs 0.888 ms (2.14×) | `mlp/mlp_fashion/results/sizing/fashion_sizing*.json`, `mlp/mlp_fashion/results/tflite/` | check exact file per row |
| 9 | MLP accuracy, binary vs INT8 QAT same shape | 87.5 vs 89.8 % | same as 8 | in repo |
| 10 | FOMO person: F1 binary vs INT8 | 31.2 ± 0.7 vs 30.1 ± 0.2 % | `conv2d/fomo_voc_person/results/symbex/symbex_seeds_pc_c32-32-128_*.json`, `.../tflite/tflite_pc_c32-32-128_*.json` | in repo |
| 11 | FOMO speed S3 2.03×; only binary fits the classic ESP32 | 126.3 vs 256.9 ms | `.../symbex/symbex_seeds_s3_swar_*.json`, `.../tflite/tflite_s3_*.json`, `..._esp32_*` | in repo |
| 12 | HAR 1D vs ultracompactBNN on GVSoC | 1.07-1.37× fewer instructions | `conv1d/har_ultracompact/results/symbex/har_gvsoc_*.json`, `.../ucbnn/ucbnn_gvsoc.json` | in repo |
| 13 | HAR vs TFLite INT8: ESP32 5.3-10.4×, S3 1.47-1.90×; INT8 ~9 points more accurate | | `conv1d/har_ultracompact/results/tflite/`, `.../symbex/har_accuracy.json` | in repo |
| 14 | ECG MIT-BIH, inter-patient: balanced accuracy tied with INT8 within sd; both operating points | README section 3b | `conv1d/ecg_mitbih/results/` | in repo (PR #35) |
| 15 | Exactness: every chip run 100 % equal to the integer simulation | | the `equal` fields of every chip JSON | write a script that checks them all |
| 16 | Accuracy cost of going binary, per task | 0 (FOMO, ECG) to ~10+ (ResNet-8) | [results.md "Accuracy: where binary costs"](../results.md#accuracy-where-binary-costs) | add the ECG row |
| 17 | Binary loses mostly by capacity, not training (ResNet-8) | twin 92 % train fit vs binary 72 % | `experiments/training_gap/results/` | in repo |
| 18 | tiny LLM W1A8, exact on the S3 | 8.48 tokens/s, val loss 1.522 vs 1.300 FP32 | `experiments/tiny_llm/results/` | seed 42 on the chip; 43 / 44 running (Nov 12) |
| 19 | Low-bit context: INT2 QAT more accurate per byte than binary; binary wins speed | | `mlp/mlp_fashion/results/lowbit/` | our kernels: context only |

## Related work notes (PDFs read Nov 12)

**McDanel, Teerapittayanon, Kung, "Embedded Binarized Neural Networks" (arXiv 1709.02260, 2017).**
- What: fused binary blocks (conv + pool + BN + sign with one float accumulator) so the temporaries between layers
  are bits, not floats: 32x less temporary memory. Intel Curie (Arduino 101, 32 MHz Quark, 15 KB usable SRAM).
- Results (Table 1): MNIST MLP-1 91.54 % in 17.35 ms, ConvPool-2 97.86 % in 886.53 ms; CIFAR10-Easy (the 10 % of
  test images a big net is most sure about, not full CIFAR-10) ConvPool-1 84.30 % in 847.72 ms. Energy from the
  board's power figures; compared with sending the input over BLE.
- Tooling: Chainer training, automatic export to a standalone C header, C checked against the Python version
  "at each stage", open source (gitlab.com/htkung/ddnn), plus a search over models that fit a memory budget.
- No INT8 baseline, one chip, accuracy on easy data.
- For us: we are NOT the first open Python -> C chain for BNNs on MCUs, nor the first to check C against Python.
  Our difference: the check is 100 % per input on the chip itself, every run; same network vs the official INT8
  toolchain on the same chip; full CIFAR-10 and several tasks; several chips (with and without SIMD).

**Daghero, Burrello, Jahier Pagliari, Benini, Macii, Poncino, "Ultra-compact BNNs for HAR on RISC-V" (CF '21,
arXiv 2205.12781).**
- What: a BNN library for very small nets (< 32 channels, no padding to 32), 1D convs, PULPissimo (RI5CY) single
  core. First layer takes 8-bit inputs (as our int8 first layer).
- Baselines: Random Forests (scikit-learn, C on the same core) and PULP-NN's binary convs adapted to 1D (up to 41-44 %
  fewer cycles for few channels). No INT8 network baseline.
- Measurement: cycles on GVSoC (simulator), energy computed from published power of the Quentin chip at 205 MHz;
  not measured on a board. Table 1: UniMiB-SHAR 33-68 %, Walk 83.7-94.6 %, latency < 3 ms, < 10 uJ.
- Open source (github, "Ultra-compact BNNs").
- For us: our claim 12 (1.07-1.37x fewer instructions on GVSoC, same networks) compares against this library; we add
  real boards and the INT8 baseline it does not have. Say clearly that it is GVSoC vs GVSoC.

**Gap sentence, after reading both (SUPERSEDED Nov 12: CBin-NN compares with INT8 on the same chip; see related_work.md):** prior MCU BNN work shows binary inference fits and runs (McDanel) and can be
compact (Daghero), but neither compares with an INT8 network on the same chip, and neither reports per-input
equality between the deployed C and the trained model on real hardware across tasks and chips.

## Before writing

- Related work: ~~read the McDanel 2017 and Daghero 2021 PDFs~~ done (notes above); check the rest of Obed's PR #36
  references (FINN, daBNN and Larq still to read).
- Claim 8: name the exact JSON per row.
- Claim 6: decide whether ESP-DL goes in (then its scripts and JSON go into the repo) or stays a footnote.
- Claim 15: one script that walks every `results/` folder and lists any chip run below 100 %.
- Then: LaTeX skeleton from this outline.
