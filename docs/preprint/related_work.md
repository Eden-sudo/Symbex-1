# Related work: systematic review (working notes)

Started Nov 12, 2026. This file holds the search protocol, every paper found, how far each one was checked, and the
gap sentence that the checked papers still allow. A paper counts as **read** only when its PDF was read; **abstract**
means only the abstract or a web summary was seen, so its numbers stay out of the paper until the PDF is read.

## Protocol

- **Question:** which published work runs binary (or ternary / sub-byte) neural networks on microcontrollers, and
  what do they compare against, on which chips, with what verification?
- **Sources:** OpenAlex (keyword search + forward citations), arXiv, web search; snowballing from every paper read.
- **Include:** inference on a microcontroller (no FPGA, no ASIC, no phone / Raspberry Pi as the main target),
  weights at 2 bits or fewer, measured on a real board or a cycle-accurate simulator, 2017-2026.
- **Context only (cited, not compared):** BNN engines for bigger CPUs (daBNN, Larq Compute Engine), FPGA (FINN),
  accelerators (XNOR Neural Engine, CUTIE), accuracy methods (Bop, ReActNet, BoolNet, PokeBNN), surveys.
- **Record per paper:** chip, tasks, baseline (same network in INT8 on the same chip?), exactness check, seeds,
  open source, measured vs simulated.
- **Queries run (Nov 12):** "binary neural network microcontroller", forward citations of CBin-NN (27) and Daghero
  2021 (23). Rate-limited before the rest; **still to run:** "binarized neural network inference engine
  microcontroller", "binary neural network ESP32", "binary neural network Cortex-M TensorFlow Lite Micro",
  "ternary neural network microcontroller", "binary neural network RISC-V microcontroller", "sub-byte quantization
  microcontroller", forward citations of McDanel 2017 and of Larq Compute Engine, Google Scholar for 2026.

## Papers on microcontrollers

| Paper | Chip(s) | Task(s) | Same-net INT8 on the same chip | Exactness | Measured | Open | Checked |
|---|---|---|---|---|---|---|---|
| McDanel et al., eBNN, arXiv 1709.02260 (2017) | Intel Curie (32 MHz) | MNIST, CIFAR10-Easy | no | C vs Python "at each stage" | board | yes | read |
| Daghero et al., CF '21, arXiv 2205.12781 | PULPissimo RI5CY | HAR (UniMiB-SHAR, Walk) | no (Random Forests, PULP-NN binary) | not reported | GVSoC simulator | yes | read |
| Cerutti et al., TCAS-I 2022, arXiv 2201.03386 | GAP8 (BNN), STM32L476 (acquisition) | keyword spotting (Speech Commands), binary analog features + BNN, fixed-point last layer | no (vs published KWS systems) | none | **BNN energy estimated** from earlier GMAC/s/W figures, not run end to end; front end measured | not stated | read |
| Sakr, Berta, Doyle, Younes, De Gloria, Bellotti, "Memory Efficient Binary Convolutional Neural Networks on Microcontrollers", EDGE 2022, doi 10.1109/edge55608.2022.00032 | Cortex-M7 (model not named in the abstract) | robotic touch classification | no (vs earlier touch-classification work: +2 % accuracy, footprint 15 %, latency 25 %, direction unclear) | not in abstract | board | C library (the base of CBin-NN) | abstract (PDF on iris.unige.it, site certificate expired, Nov 13) |
| **Sakr et al., CBin-NN, Electronics 13(9) 1624 (2024)** | STM32F746 (M7, 216 MHz); also F091 / F401 / H743 | CIFAR-10 (SmallCifar, 3 conv) | **yes:** TFLM, CMSIS-NN, MicroTVM, TinyEngine (79.90 %) | not reported (only "same assembly" across compilers) | board | yes | read |
| Mohammady et al., COINS 2026, doi 10.1109/coins69995.2026.11715584 | STM32H7B3 | person detection (Wake Vision) | to check | to check | board | to check | abstract (no open PDF found Nov 13; same Genoa group: NAS for BNNs on Wake Vision, Springer Feb 2026, and "Binarized Wake-Up of Conversational Agents", Sep 2026: Larq Compute Engine + CBin-NN) |
| Ballout et al., binary autoencoders on STM32 (2026) | STM32 | to check | to check | to check | board | to check | title |
| Camacho-Pérez, Rodríguez-Arreola, Drupelet, LNCS 16265 pp. 199-210 (2026), doi 10.1007/978-3-032-17933-3_21 | two MCUs (not named in abstract) | mixed binary / ternary framework; memory and time of single layers only | no | not in abstract | board | to check | abstract (paywalled, no open copy found Nov 13) |
| Romano et al., Neuro-C, EuroSys 2026, doi 10.1145/3767295.3769380 | STM32F072RB (Cortex-M0, 16 KB RAM) | MNIST, FashionMNIST, CIFAR5; fixed ternary adjacency + per-neuron scale (MLP-type) | no: vs their own MLPs and TNNs (MNIST >97 %: 5 ms / 3 KB vs MLP 43 ms / 31 KB); not TFLM / CMSIS-NN | none: accuracy computed on the PC, latency on the chip | board | yes (GitHub) | read |
| BitNetMCU (cpldcpu, GitHub, 2024-2026) | CH32V003 (RISC-V, no multiplier) | MNIST 16x16 | no | script compares C vs Python on 10k test images (result not reported) | board | yes | README |
| Ruciński et al., FTerViT, arXiv 2605.21171 (NeurIPS 2026) | ESP32-S3-EYE, 240 MHz, 8 MB PSRAM | ImageNet DeiT-III-S, ternary weights (2-bit packed), 8-bit activations; 5 seeds for training | no: only one FC1 kernel FP32 vs INT8 vs ternary (1.71x vs FP32); no ESP-DL / TFLite | none stated | board, pure C, **21 s per image**, FP32 activations on the chip | not stated | read |
| slvDev/esp32-ai (GitHub, 2026, not peer-reviewed) | ESP32-S3 N16R8, both cores | TinyStories LM: 6 layers d96, 0.56 M dense core int4 + 25 M flash lookup table + 3.1 M int8 head in PSRAM | no | C vs PyTorch float, max abs diff 1e-5 (not bit-exact) | board, 9.88 tok/s, scalar dot products | yes | RESULTS.md |
| Aizhee/BitNeural32 (GitHub, 2026, not peer-reviewed) | ESP32 / S2 / S3 / C3 / C2 | ternary weights, float or fixed-point activations; dense, conv, LSTM / GRU | no | none described | latencies are estimates, no accuracy | yes | README |
| Rutishauser et al., xTern, ASAP 2024, arXiv 2405.19065 | RISC-V + ISA extension | ternary nets | to check | to check | to check | to check | abstract |

## Context (not compared)

| Paper | What | Why only context |
|---|---|---|
| Umuroglu et al., FINN, arXiv 1612.07119 (2017) | BNN on FPGA (ZC706), CIFAR-10 80.1 % | FPGA |
| Zhang et al., daBNN, arXiv 1908.05858 (2019) | BNN engine, ARM64 phones | not an MCU |
| Bannink et al., Larq Compute Engine, arXiv 2011.09398 (2021) | BNN engine on TFLite, Pixel 1 / RPi 4 | not an MCU (but used on STM32H7 in the 2026 Wake-Up paper) |
| Helwegen et al., Bop (2019) | binary optimizer | method; we use it, so do CBin-NN |
| Kızılateş, FastGRNN on ATmega328P / MSP430, arXiv 2606.17249 (2026) | Q15 FastGRNN for HAR, 566 B of weights; 100 % prediction agreement with PyTorch on 3,399 windows (seed 0), 99.91-100 % over 5 seeds | not binary; another example that per-input agreement checks exist |
| Plumerai (company blog, 2021-2023) | BNNs and an INT8 engine on Cortex-M, MLPerf Tiny | not peer-reviewed, closed |
| Banbury et al., MLPerf Tiny, arXiv 2106.07597 | the ResNet-8 we run | benchmark |
| Lai et al., CMSIS-NN, arXiv 1801.06601 | INT8 kernels for Cortex-M | the INT8 baseline |
| Abushahla et al., arXiv 2508.15008 (2025) | survey of quantization on MCUs | says sub-byte support on Cortex-M is "largely absent" |
| Park et al., IEEE IoT-J 2025, doi 10.1109/jiot.2025.3633487 | survey of binary / ternary nets and compute-in-memory | survey |
| Systematic review of BNNs, IEEE Access 2023, doi 10.1109/access.2023.3258360 | survey | survey |

## Current state (from the papers read so far)

- Binary inference on MCUs is established: it runs (McDanel 2017), it can be very compact (Daghero 2021), and an
  open C engine already beats the INT8 toolchains on one Cortex-M7 for one CIFAR-10 network (CBin-NN 2024: 1.2-3.6x
  faster, 2.5 points less accurate, 6x smaller).
- Tricks we use are **not new**: Bop, an 8-bit first layer, OR max pooling, BN folded into an integer threshold all
  appear in CBin-NN (and the 8-bit first layer in Daghero).
- Activity 2024-2026 is on applications (structural health, wake-up, autoencoders on STM32 from the Genoa group),
  ternary / mixed frameworks (Drupelet, xTern, BitNetMCU) and ternary transformers (FTerViT on the ESP32-S3).
- Not seen in any paper read so far: (1) per-input equality of the chip output with the trained integer model,
  reported on every run; (2) several chips with different ISAs (Xtensa with and without SIMD, Cortex-M33, RISC-V);
  (3) several tasks under one methodology (validation split, test set once, 3 seeds); (4) the official MLPerf Tiny
  network against the official INT8 toolchain.

## Gap sentence (replaces the one in outline.md)

> Binary networks already run on microcontrollers, and CBin-NN showed that a binary engine can beat the INT8
> toolchains on a Cortex-M7 for one network. What is still missing is a broad, verified comparison: the same
> networks, trained with the same recipe, run on several commercial chips with different instruction sets, against
> the official INT8 toolchain, with the chip's output checked to be identical to the trained integer model on every
> test input, across several tasks and seeds. SYMBEX-1 provides this, and an open chain from training to C.

Claims still to verify before this sentence is final: CBin-NN's seeds and whether it checks outputs per input;
Drupelet, Wake-Up 2026 and Sakr 2022 full texts; the remaining queries above.

## Targeted searches for our strongest claims (Nov 12, web search)

| Claim | Found | Verdict so far |
|---|---|---|
| A binary engine beats the vendor's SIMD INT8 library (ESP-DL) on the ESP32-S3 | nothing compares a BNN with ESP-DL | not seen; keep searching (Gemini Deep Research list pending) |
| Fully binary RNN (binary weights and binary state) on an MCU | Brain-on-Switch (arXiv 2403.11090): binary GRU on programmable switches, not MCUs; BitNeural32: LSTM / GRU with ternary weights but float state | not seen on MCUs |
| 1-bit / W1A8 language model on an MCU | bitnet.cpp and BitNet papers: CPUs and Raspberry Pi only; esp32-ai: int4 LM on the S3, 9.88 tok/s; llama2.c ports to the S3 (260 K parameters, ~17 tok/s) | no 1-bit LM on an MCU seen. **Careful with tok/s:** esp32-ai is faster than our 8.48 tok/s, but its dense core is 0.56 M parameters vs our 14.16 M in 1-bit linear layers, it uses both cores, and its vocabulary (32 768) differs, so perplexities are not comparable. Both are bound by the head's PSRAM reads. Cite it; compare compute per token, not tok/s |

## Gemini Deep Research table, checked row by row (Nov 12)

Checked against the arXiv abstract pages (the PULP-NN DOI page refused access; its arXiv 1908.11263 was read).

| Gemini's row | Checked | Correction / note | Use |
|---|---|---|---|
| CBin-NN, Sakr et al., Electronics 2024 | yes (PDF read before) | "3.6x" is against TFLM only (CMSIS-NN 20 %, TinyEngine 15 %) | already in the table |
| xTern, Rutishauser, Mihali, Scherer, Benini, ASAP 2024, 2405.19065 | abstract | correct: ISA extension, 8-core cluster, +57.1 % energy efficiency and +67 % throughput vs 2-bit kernels, 0.9 % area; chip or simulation not stated | context (custom ISA, not a commercial MCU) |
| PULP-NN, Garofalo et al., Phil. Trans. R. Soc. A 2019, doi 10.1098/rsta.2019.0155 (arXiv 1908.11263) | abstract | correct: down to INT-1, GAP-8, CIFAR-10 30x / 19.6x fewer cycles than CMSIS-NN on STM32L4 / H7. Note: 8-core cluster vs one Cortex-M core | add to MCU table (sub-byte on a RISC-V MCU, not the same-net same-chip INT8 test) |
| Veldanda, "Baseline Shape Decides the Verdict", arXiv 2609.29397 (Sep 2026) | abstract | exists, but **no MCU**: 98 runs on one laptop; "MCU-class" is a motivation only | context (ternary LM at tiny scale) |
| FTerViT, arXiv 2605.21171 | abstract (before) | Gemini gives no authors (Ruciński et al.) | already in the table |
| "Fully Binary Training of Deep and Recurrent Neural Networks (BEP-TT)", "Bacellar et al.", ICLR 2026, 2512.04189 | abstract | **title and authors wrong**: "BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training", Colombo, Pittorino, Zambon, Baldassi, Roveri, Alippi, ICLR 2026. Binary weights AND binary state RNN trained end to end (BEP-TT); "up to +10.57 %" vs (likely) QAT; **no hardware** | cite in the RNN "limits" section: a fully binary RNN exists as a training method, not run on an MCU |

Prose items in Gemini's report (not in its table), not checked yet, treat as leads: a 7-board ESP32-S3 cluster
running a ~0.4 B BitNet (community post, ~0.11 tok/s); esp32-pie-asr (65.7 M transducer, 2 / 4 bits, on the
**ESP32-P4**, not the S3); t-rsi-ocr (ternary OCR in C); BNRV (RISC-V SIMD extension for BitNet); Hybrid Gated Flow
(1.58-bit LLM stabilization); motor-temp-edge (INT8 MLP, bit-exact on QEMU Cortex-M4).

Prose leads checked (Nov 13, project pages / arXiv abstracts):

| Lead | Checked | What it really is | Use |
|---|---|---|---|
| 7-board ESP32-S3 cluster | news posts + repo name (Low-Zi-Hong/ESP32s3-LLM-Cluster) | a sliced Qwen2-0.5B (not 0.4 B) with BitNet b1.58 ternary weights, 6 compute nodes over SPI; seconds per token; the author says the model emits random tokens (training stopped at loss ~8) | context only: proof of concept, ternary, does not work as a model |
| esp32-pie-asr | GitHub README | 65.7 M Zipformer2 transducer, 2 / 4-bit weights chosen by sensitivity without training (post-training), **ESP32-P4** (2 RISC-V cores, 400 MHz), 0.70x real time; **chip output equal to the PC build of the engine**, every test clip and kernel | cite: exact chip-vs-PC checking already exists for a low-bit engine (against its own C on the PC, not against a trained integer model; not binary; not vs an official INT8 toolchain) |
| t-rsi-ocr | Hugging Face model card | ternary BitNet text-line OCR, C engine, 312 KB packed weights; **all numbers on an Apple M4 Max**, no microcontroller port | drop (no MCU) |
| BNRV | search | **not found** under that name; nearest: RISC-V Summit Europe 2026 study (SIMD hardware for ternary LLMs, x2 tokens/s, FPGA, 1.25 % LUTs) | drop the name; maybe context for custom ISA |
| Hybrid Gated Flow, arXiv 2602.05269 | abstract | Trejo Pizzo, Feb 2026, single author: ternary backbone + low-rank FP16 correction, ~12-15 % memory overhead, TinyStories, "qualitative signals" by the author's own words; no hardware | context at most (1.58-bit LM training) |
| motor-temp-edge | GitHub README | INT8 MLP, C on **QEMU** Cortex-M4 (mps2-an386, "emulated, not real hardware"), bit-exact fraction 1.0 vs Python integer emulation, `-ffp-contract=off` | context: per-input integer exactness is common practice in small INT8 projects (emulator, one task) |

Effect: esp32-pie-asr weakens "exact on the chip" further: say "checked per input against the trained integer
model (not only the engine's PC build), binary, several chips and tasks, against the official INT8 toolchain".

Effect on our claims:
- Exactness: CMSIS-NN is designed to be bit-exact to the TFLite reference kernels, and small INT8 projects check
  bit-exactness; so "bit-exact on the chip" is **not new in general**. What stays unseen: a binary engine whose chip
  output is checked per input against the trained integer model, on several chips and tasks. Reword claim 15 so.
- Binary RNN: "fully binary RNN" exists (BEP, training only). Our claim must be "not seen running on an MCU".
- 1-bit LM on an MCU: still not seen (the ESP32 LM items are int4 / ternary / 1.58-bit, or a 7-board cluster).
