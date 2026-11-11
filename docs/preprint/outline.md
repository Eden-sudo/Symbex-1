# SYMBEX-1 preprint: outline

Working outline. Every claim lists the file that backs it (`results/` JSON, a README section or a doc). A claim with
no source does not go into the paper. Numbers are copied from the JSON at writing time, never from memory; the final
pass re-reads every file (`tools/dev/docs_charts.py` and the bench-checker do the same).

Structure: the one Edén's professor asked for (front matter + 5 body sections). The same text serves the arXiv
version (cs.LG, cross-list eess.SP) with the section names in English. About 8 pages + appendix.

## Front matter

| Item | Content | Status |
|---|---|---|
| Title | *SYMBEX-1: exact binary neural networks on commercial microcontrollers, measured against the INT8 toolchain* / *SYMBEX-1: redes neuronales binarias exactas en microcontroladores comerciales, medidas contra la cadena INT8* | candidate; Edén decides |
| Acronym | SYMBEX-1 = **Sym**bolic **B**it **Ex**pansion, 1-bit engine (the original name). *Symbolic*: Boolean conditions instead of multiplications. *Bit expansion*: inputs and activations expanded into bit planes (thermometer stem, bit-sliced vertical engine). *1*: one bit per weight and activation. | draft; Edén decides whether to keep "expansion" |
| Authors, affiliation | Carlos Duarte, Obed (Obed0101), Abrahan Aparicio (FOMO). Universidad Latina de Panamá, Licenciatura en Mecatrónica. | order and contribution lines: Edén decides |
| Abstract (Spanish and English) | What: an end-to-end chain for binary networks, exact on the chip. How: trained, folded to integers, compiled to C, measured on three commercial chips against TensorFlow Lite Micro INT8, same network, 3 seeds, test once. Result: without SIMD (ESP32, Hazard3) whole networks 5-13x faster than INT8; with SIMD (S3, M33) 1.0-2x on the convolutional networks and up to 28x when the INT8 weights are read from flash (MLP, S3); accuracy cost 0 to 23 points by task; 3-10x less flash. Conclusion: where binary pays and where INT8 stays. | written last, from the body |
| Keywords (Spanish and English) | redes neuronales binarias, microcontroladores, cuantización, TinyML, inferencia en el borde / binary neural networks, microcontrollers, quantization, TinyML, edge inference | draft |

## 1. Introducción / Introduction (what was done, how, why)

| Claim | Source |
|---|---|
| Microcontrollers: a few hundred KB of RAM, often no SIMD / FPU; INT8 with TensorFlow Lite Micro is the deployment path almost everyone uses | README "Why binary?"; docs/en/how-it-works.md |
| One bit per weight: 8x smaller than INT8, bit operations + count instead of multiply-accumulate; on a chip without SIMD the count is 11-17x cheaper than int8 MAC | `tools/chip_profiles/count_*.json` (count_bench.py), README "Why binary?" |
| Contribution 1: end-to-end chain (PyTorch -> integer folding -> C header -> chip) with **exactness**: C on the PC and on the chip equals the integer simulation on 100 % of the inputs, every model | every `chip_*.json` (`exact` / equality field), `tests/test_header_format.py`, docs/en/methodology.md |
| Contribution 2: same network, same commercial chip, against TFLite Micro INT8 (ESP-NN / CMSIS-NN), 3 seeds, test set once; three ISAs (Xtensa LX6 / LX7, Cortex-M33, Hazard3 RISC-V) | docs/en/methodology.md; results folders below |
| Contribution 3: where binary pays and where it does not: accuracy cost 0 to >10 points by task; speed-up largest without SIMD | README "Every benchmark, side by side", section 5 below |
| Contribution 4: the binary ResNet-8 loses by capacity, not training (study) | `benchmarks/experiments/training_gap/README.md`, `results/train_gap/*.json` |

**The framing (Nov 10, Edén):** SYMBEX-1 is, first, a tool that makes developing binary networks for commercial
microcontrollers iterable: `symbex.toml` -> train -> fold -> C header -> 100 % check on the PC and on the board, in
minutes. The measurements are what the tool produced. Claim it as "an open chain for commercial MCUs with a verified
path", never as "the first" (eBNN, 2017, had an open Chainer -> C -> Arduino 101 chain for one chip, both discontinued).

| Claim (the loop) | Source |
|---|---|
| One configuration file re-creates a run; stages can start or stop anywhere (`--from`, `--to`) | docs/en/cdt.md |
| Every export is checked against the simulation on the PC; on the chip when a board is configured and the run reaches the chip stage (every chip result in this paper did) | docs/en/how-it-works.md section 6; tools/cdt.py main(); every chip JSON |
| Iteration time: ResNet-8 training in minutes on one GPU (data on GPU + CUDA graphs); export + validate in seconds | docs/en/cdt.md, tools/README.md: ResNet-8 150 epochs in 3.0 min (8.6 without data on GPU + CUDA graphs), same accuracy and exactness |
| What the loop produced: engines (standard, vertical), stem / shortcut / head variants, training-gap study, ECG recipe by CV | the benchmark and experiment READMEs |

### 1.1 Related work (two lines, honest)

**Framework landscape (verified Nov 10 with the sources in the last column):**

| Framework | Training front end | Target | Commercial MCUs | Open | Config -> C, checked | Source |
|---|---|---|---|---|---|---|
| Larq + LCE (Plumerai) | TF / Keras | Cortex-A / ARM64 (NEON) | no: "meant more for mobile-class devices such as the Arm Cortex-A ... ESP32 family isn't the target"; for MCUs "Plumerai does offer commercial solutions" | LCE Apache-2.0; the MCU engine has no public source (evaluation on request) | yes, for phones | github.com/larq/compute-engine/discussions/775 (CNugteren, 2023-07-05); plumerai.com ("optimized for Arm Cortex-M, Cortex-A and RISC-V") |
| Brevitas + FINN (AMD) | PyTorch | FPGA (HLS / RTL) | no | yes | yes, for FPGA | FINN paper / repo |
| TensorFlow Lite Micro | TF | MCUs | yes | yes | INT8 only, no 1-bit kernels | - |
| ExecuTorch (Meta), Cortex-M backend | PyTorch | Cortex-M | yes | yes | "in beta"; "symmetric INT8 (8w8a)"; 4w8a from CMSIS-NN "not yet integrated"; validated on the Corstone-300 FVP (simulator) | docs.pytorch.org/executorch/stable/backends/arm-cortex-m |
| eBNN (McDanel 2017) | Chainer (discontinued 2019) | Intel Curie / Arduino 101 (discontinued) | one chip | yes (gitlab.com/htkung/ddnn) | "automatically converted into a standalone C header"; "validated the correctness of the C implementation by comparing the output at each stage to the Python version" (no metric, not per input on the device) | arXiv 1709.02260 |
| ultracompactBNN (Daghero 2021) | not in the repo | PULPissimo (RI5CY) in GVSoC | no (research platform) | code yes, no license file | six kernel templates (linear, maxpool, conv1d 1-bit and 8-to-1-bit) with a `main.c` each; no converter, no check | github.com/francescodaghero/ultracompactBNN (one commit, 2021-03-23) |
| CBin-NN (Sakr 2024) | Larq / Keras | STM32F746 (Cortex-M7, 216 MHz) + F091, F401, H743 (all Cortex-M) | yes, four STM32 | yes (github.com/Edge-Learning-Machine/CBin-NN) | `model_converter.py` writes the parameters and the layer calls; no check of the C output against Keras in the paper or the code | Electronics 2024 13 1624, sections 5.3 and 6 |
| SYMBEX-1 | PyTorch | Xtensa LX6 / LX7, Cortex-M33, Hazard3 RISC-V, AVR | yes, five cores of four ISAs | yes | yes, 100 % exact on PC and chip, every model | this paper |

Verified Nov 10 (arXiv pages, repositories; eBNN, Cerutti and CBin-NN read in full, ultracompactBNN read from its
repository and its paper).

| Work | Chip / platform | Task | Code | vs INT8 | Notes |
|---|---|---|---|---|---|
| BinaryConnect / BNN (Courbariaux, Hubara 2015-16), XNOR-Net 2016, Bi-Real 2018, ReActNet 2020, Bop (Helwegen 2019) | method papers | ImageNet, CIFAR | yes | - | training methods; we use Bop as an option |
| FINN (Umuroglu 2017) | FPGA | MNIST, CIFAR | yes | - | hardware line |
| daBNN (Zhang 2019), Larq Compute Engine (Bannink 2020) | ARM64 phones | ImageNet | yes | yes (TFLite) | not microcontrollers |
| **eBNN**, McDanel, Teerapittayanon, Kung, EWSN 2017, arXiv 1709.02260, `gitlab.com/htkung/ddnn` | Intel Curie (Arduino 101), 32 MHz, ~15 KB usable | MNIST (Conv-1 94.56 %, 53.7 ms; MLP-1 91.54 %, 17.4 ms, 14.7 KB), CIFAR10-Easy | yes: Chainer -> C header -> Arduino | **no** (only vs a BNN with float temporaries, 32x memory, theoretical) | **already an open end-to-end chain**: we do not claim "first end-to-end". Correctness checked "at each stage" against Python, no metric |
| Garofalo et al., PULP-NN, Phil. Trans. R. Soc. A 378 (2164), 2020, DOI 10.1098/rsta.2019.0155; XpulpNN (DATE 2020) | PULP RISC-V (GAP8); XpulpNN = ISA extension | CIFAR-10 kernels | yes | yes (CMSIS-NN) | kernels, 8-bit and sub-byte; no compiler |
| Cerutti et al., ISLPED 2020, DOI 10.1145/3370748.3406588, arXiv 2101.04446 | GAP8 (8-core RISC-V, commercial) vs STM32F469I (Cortex-M4F) | sound event detection, 28 classes (BNN 77.9 % vs 16-bit CNN 85.1 %) | no code mentioned | **no**: the only on-device baseline is the same BNN on the Cortex-M4F (10.3x faster, 51x less energy on GAP8); the 16-bit CNN does not fit | energy per inference reported (25.5 mJ); no exactness check |
| **ultracompactBNN**, Daghero et al., ACM CF 2021, DOI 10.1145/3457388.3458656, arXiv 2205.12781, `francescodaghero/ultracompactBNN` | PULPissimo (RI5CY, 1 core) on GVSoC | HAR, UniMiB-SHAR + private "Walk" | yes | **no** (vs random forest and padded PULP-NN BNN) | our HAR comparison runs their library on GVSoC |
| Sakr et al., IEEE EDGE 2022 ("Memory efficient binary CNNs on microcontrollers"); **CBin-NN**, Electronics 2024, 13, 1624, DOI 10.3390/electronics13091624 | STM32F746 (Cortex-M7, 216 MHz) + 3 more STM32, commercial | CIFAR-10 "SmallCifar": CBin-NN 77.42 % (Bop + PReLU + 8-bit first layer) vs the same network INT8 79.90 % | yes, Larq + `model_converter.py` | **yes**: 3.6x faster than TFLite Micro, 1.4x MicroTVM, 20 % less than CMSIS-NN, 15 % less than TinyEngine; 14.17 vs 87.34 KB | the closest work: commercial MCU, same network, vs INT8 toolchains. One task, one ISA, no exactness check, no seeds. Their 8-bit first layer (QQConv2D) matches our int8 stem finding |
| xTern, Rutishauser et al., IEEE ASAP 2024, arXiv 2405.19065, DOI 10.1109/ASAP61560.2024.00049 | RISC-V cluster with a custom ISA extension | CIFAR-10 ternary vs 2-bit | yes | vs 2-bit | not a commercial chip |

Gap statement (what no previous work combines), after reading the papers: eBNN (2017) had an open chain and a
correctness check, on one discontinued chip, one task, with no INT8 baseline. CBin-NN (2024) has an open engine and a
converter, on commercial Cortex-M, and compares the same network with the INT8 toolchains (TFLite Micro, CMSIS-NN),
on one task, one ISA, one seed, with no check of the C output. ultracompactBNN and xTern run on research platforms;
Cerutti on a commercial GAP8 but with no INT8 baseline; LCE and FINN do not target microcontrollers. What SYMBEX-1
adds: a chain whose output is **verified exact on every model, on the PC and on the chip**, measured on **commercial
chips of four ISAs** against the **official INT8 toolchain** on **five tasks** (images, detection, HAR, ECG, digits)
with a **fixed methodology** (same architecture, 3 seeds, validation for every choice, test once), plus a second engine
for SIMD and the studies the chain made possible. Say it that way; do not say "first".

Obed's PR #36 (`docs/related-research`) lists eBNN, ultracompactBNN, Daghero 2022 (arXiv 2209.00839), the BNN survey
(arXiv 2004.03333) and daBNN: reuse those links.

## 2. Materiales y métodos / Materials and methods

### 2.1 Method: the chain

| Claim | Source |
|---|---|
| Binary layer: XNOR/AND + popcount, one integer threshold per output (scale, BatchNorm, sign folded) | docs/en/how-it-works.md |
| Input: thermometer bits, or int8 first layer (`stem_i8`); OR pooling fused; pooled int8 classifier (`pool2`) | docs/en/how-it-works.md, cnn_cifar_resnet8 README |
| Engines: standard (bit planes), vertical (bit-sliced, 128 windows per PIE register) | docs/en/vertical.md, `results/symbex/s3_vertical_layers.json` |
| HAL per chip: lut16 / SWAR / PIE / Zbb cpop; count benchmark per chip | docs/en/library.md, `tools/chip_profiles/*.json` |
| Compiler (CDT): config, search of widths and bits, training (Bop option), folding, export, validate, chip | docs/en/cdt.md |
| Exactness check: fidelity (PyTorch vs integer simulation) and exactness (C vs simulation, PC and chip) | docs/en/methodology.md |

### 2.2 Materials: boards, baselines, recipe

| Item | Source |
|---|---|
| Boards: ESP32 (Xtensa LX6, no SIMD), ESP32-S3 (LX7, PIE), Raspberry Pi Pico 2 (RP2350: Cortex-M33 + Hazard3) | docs/en/methodology.md |
| Baselines: TFLite Micro INT8 with ESP-NN (ESP32, S3) and CMSIS-NN (M33), reference kernels (Hazard3); ESP-DL quick check on the S3; ultracompactBNN on GVSoC | `results/tflite/*.json` per benchmark; README "What we measured" |
| Recipe: same architecture, same epochs / seeds 42 43 44 / split, validation carved from training, test once; mean +- sd. **Exceptions to write down:** FOMO INT8 trained with the FP32 recipe (person weight 20, 60 epochs) vs binary weight 40, 100 epochs (matching budget = future work, README); FOMO published results calibrated on the previous split; ResNet-8 sizing variants (16-32-128 etc.) ranked on one seed; FP32 sha256 not yet in every result file | docs/en/methodology.md; fomo README 22, 88-94, 153; resnet8 README 112-114; `symbex_seeds_*.json` |
| Low-bit baselines INT4 / INT2 / ternary: our own code (`tools/lowbit`), exploratory, not an official toolchain | `benchmarks/mlp/mlp_fashion/results/lowbit/*.json` |

## 3. Resultados y discusión / Results and discussion

### 3.1 Results (one subsection per benchmark)

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

### 3.2 Discussion: where binary pays and why it loses

| Claim | Source |
|---|---|
| Binary pays most without SIMD; on the S3 beating ESP-DL needed the vertical engine (flash 1.52x; ESP-DL wins in RAM) | README "What we learned" 1; docs/en/results.md "How fast could INT8 be" |
| Capacity, not training: twin fits 92 % of train, binary 72 %; Bop +2.5, two-step +2.35, not additive; width > 2-bit activations | `training_gap/README.md`, `results/train_gap/*.json`, `results/small_gap/` |
| Size the network for binary: 16-32-128 +1.8 points at the same speed; narrow first MLP layer | `cnn_cifar_resnet8/results/sizing/`, `mlp_fashion/results/sizing/` |
| An 8-bit input deserves an int8 first layer (accuracy, and speed with SIMD): stem alone S3 1.79x, Hazard3 0.62x | `benchmarks/experiments/chip_kernels/` (int8_first_bench), cnn_cifar_resnet8 README |
| Per-layer cost: the int8 stem dominates on the Pico | `symbex_seeds_pico2_*_int8stem_pool2*.json` (per-layer fields) |

### 3.3 Limitations (written, not hidden)

- Fine-grained classification stays 10-16 points behind INT8 (CIFAR-10); HAR up to 23 on the smallest network.
- ESP-DL beats the vertical engine with the weights in RAM on the S3 (quick check, Nov 2, not 3 seeds).
- Energy not measured yet (meter on the way): if it arrives, one table; if not, future work.
- Low-bit baselines are ours, not an official toolchain.
- ECG: inter-patient accuracy limited by the data (few patients per rare class); not a clinical result.
- Binary RNN probe (local, not in the repo): fragile to state flips; a "limits" paragraph at most.

## 4. Conclusiones / Conclusions

| Claim | Backed by |
|---|---|
| An open chain can deliver binary networks that are exact on commercial chips of three ISAs | every chip JSON (`exact` 100 %) |
| Binary pays most without SIMD (5-13x on whole networks); with SIMD the standard engine wins 1.0-2x on convolutional networks (28x on the MLP only because INT8 reads its weights from flash) and beating the vendor's INT8 library took the vertical engine | section 3.1, 3.2 |
| The accuracy cost is the task's: 0 (detection, ECG), 2 (MLP), 10-16 (CIFAR-10), 9-23 (HAR, worst on the smallest network) | accuracy-cost table |
| Future work: energy, keyword spotting (1D + energy), more chips, the vertical engine in 1D | CLAUDE-level plan; not claims |

## 5. Referencias / References

Every work in section 1.1, with DOI or arXiv id, plus TensorFlow Lite Micro, ESP-NN, CMSIS-NN, ESP-DL, MLPerf Tiny,
the datasets (CIFAR-10, Fashion-MNIST, PASCAL VOC, UniMiB-SHAR, MIT-BIH) and the training-method papers. BibTeX file:
`docs/preprint/references.bib` (created Nov 11; entries marked `verify` still to check).

## Appendix: reproducibility

- Repository, one script per benchmark (`*_symbex.py`, `*_tflite.py`), `tools/chip_test.py`, Docker image for GVSoC.
- Every result JSON in `benchmarks/*/results/`; `tools/dev/docs_charts.py` draws the figures from them.

## Figures (from docs_charts.py, regenerate before submission)

1. ResNet-8 accuracy vs latency Pareto (S3 and ESP32): `docs/assets/pareto_resnet8.svg`
2. MLP accuracy vs latency, binary vs INT8 / INT4 / INT2: `docs/assets/pareto_mlp.svg`
3. Speed-up per benchmark and chip: `docs/assets/speedup.svg`; flash per model: `docs/assets/flash.svg`
4. Accuracy cost per benchmark (points) against speed-up: `docs/assets/cost.svg` (done Nov 11; activity recognition
   left out: its SYMBEX chip times are in runs/, not results/); the chain diagram: mermaid in draft.md section 2.1
   (redraw for LaTeX)

## To do before prose

- [x] Merge master into `docs/structure` (ECG results, PR #35) and regenerate the charts.
- [ ] Read the PDFs of eBNN, ultracompactBNN, CBin-NN, Cerutti: fill the "vs INT8" and "exact" columns.
- [ ] Decide the title and the author list.
- [ ] Energy: meter arrives -> one script, one table (ESP32, S3, Pico 2), same models.
