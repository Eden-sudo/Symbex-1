# SYMBEX-1 (V2 - New Block-Gated Architecture)

SYMBEX-1 is an optimized library for microcontrollers that enables the inference of Binarized Neural Networks (BNN). 

It incorporates training and direct conversion tools to transform large models (FP16 or INT) into highly optimized binarized topologies. This allows even limited 8-bit microcontrollers without an FPU (like the ATmega328P) or embedded processors (like the ESP32) to execute complex neural networks accurately and quickly.

## The Update (From V1 to V2): Static Outliers vs. Dynamic Branching

The first version of SYMBEX mitigated quantization error by storing *outliers* (the most critical weights) in a parallel channel of static integer matrices. While effective, these matrices were computationally heavy and consumed vital memory.

**The V2 update incorporates the conversion of those heavy, static weights into a much lighter dynamic binarized architecture (Block-Gating).** 
Instead of processing the entire network statically, the architecture evaluates peripheral data in real-time and performs **dynamic branching** (physical *Early-Exit*). It physically turns off the network branches and blocks that are irrelevant to the current input. This maintains the global accuracy of the original network while maximizing efficiency and drastically reducing latency and memory usage.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ Python Compiler (FP16/INT → Dynamic 1-Bit Transformation)   │
│ Teacher FP32 → Binarized Student QAT (Block-Gating)         │
│ ↓                                                           │
│ Exporter → symbex_gated_weights.h (Bit planes)              │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Optimized Inference (C++ / ESP32 / AVR)                     │
│ [Packed Input]                                              │
│    ├─> 1. Gate: Evaluates K blocks & sorts Top-K            │
│    └─> 2. Core: Dynamic skipping of inactive blocks         │
│ [Dense Output]                                              │
│    └─> 3. Argmax: Direct final decision without FP          │
└─────────────────────────────────────────────────────────────┘
```

### 1. Packing Engine (XNOR + Popcount)
Both weights and inputs are packed into pure bits. The processor never performs arithmetic multiplication; it applies a binary `XNOR` operation followed by a bit count (`popcount`). 
- On advanced hardware (PC, Xtensa), it leverages register parallelism.
- On simple microcontrollers, it uses the **SWAR** (SIMD Within A Register) algorithm injected via a HAL layer (`symbex_config.h`), ensuring a fast count (~15 cycles) purely in software.
- All math is simplified to be **strictly increasing** (shifts and subtractions are eliminated during inference), directly evaluating the raw accumulation of hits to maximize clock speed.

### 2. Block-Gating and Early Exit (Dynamic Branching)
A massive hidden layer (e.g., 512 neurons) is subdivided into isolated blocks. The binarized Gate performs a quick peripheral review of the input, scores the topological relevance of each block, and activates only the best ones (Top-K). The main inference loop reads these flags and executes a physical jump (`continue;`) if the block is not needed, evading dead computation cycles.

## SDK Modular Structure

```text
Symbex1/
├── lib_symbex/           # Optimized C++ engine, HAL, and examples (ESP32/AVR)
└── tools/                
    ├── symbex_compiler.py # Main universal compiler (CLI for FP32 -> Binarized V1/V2)
    └── compiler_core/     # Modular Python core (Models, Trainer, Validator, Exporter)
```

## Hardware Validation Results (SYMBEX-1 V2)

The SYMBEX-1 framework is designed to scale deterministically from 32-bit microcontrollers down to 8-bit architectures without an FPU (Floating-Point Unit). The validations guarantee strict mathematical equivalence regardless of the underlying architecture.

### 1. Model Baseline (PyTorch FP32 Reference)
*Feed-Forward model (MLP) trained with the Scikit-Learn Digits dataset (64 features), used as a "Teacher" for distilling the binary model.*
* **FP32 Topology:** 64 → 128 → 10
* **Total Parameters:** 37,888
* **RAM/Disk Size:** ~148 KB
* **Baseline Accuracy (Test Set, 20% split):** 94.17%

---

### 2. 32-bit Performance Profile (ESP32 - Xtensa, 240 MHz)
*Binarized network using Knowledge Distillation and Dynamic Block-Gating.*
* **Binary Topology ("Student"):** 64 → 512 → 10 (4x capacity expansion vs baseline)

| Metric | Value |
|---|---|
| Inference Accuracy | **95.28%** (343/360 hits on Test Set)* |
| Inference Latency | **~645 µs** (Average over 360 continuous runs) |
| Weights Size (ROM/Flash) | **~4.8 KB** (~31x compression rate vs FP32) |
| Algorithmic Fidelity | **100% agreement with binary simulator** in Python (see Notes) |
| Base Engine | XNOR + Popcount SWAR (Native 32-bit) |

---

### 3. 8-bit Extreme Constraint Profile (Arduino Uno - ATmega328P, 16 MHz)
*Same binary topology operating within the limitations of 2KB SRAM, proving engine portability.*

| Metric | Value |
|---|---|
| Inference Accuracy | **95.28%** (343/360 hits on Test Set) |
| Inference Latency | **~18.3 ms** (Average over 360 continuous runs) |
| ROM Consumption (Flash) | **~4.8 KB** of weights (Final binary occupies 8.7 KB / 27% of total) |
| SRAM Consumption (Dynamic)| **311 global bytes** + 72 bytes local peaks (buffers) |
| Algorithmic Fidelity | **100% agreement with binary simulator** in Python |
| Base Engine | Safe XNOR (255-XOR) + Precomputed LUT (256 bytes in PROGMEM) |

---

### 4. Testing Methodology and Technical Notes

*   **Beating the Baseline:** The binarized model achieves 95.28% compared to the 94.17% of the FP32 baseline. This is not inherent to binarization, but to the architecture: the "Student" binary model has a hidden layer of 512 neurons (compared to 128 in the FP32 "Teacher"). The improvement is a result of this capacity expansion (to compensate for the loss of numerical precision) combined with the Knowledge Distillation process that regulates training. Both models were evaluated on the same stratified partition of 360 static samples.
*   **Bit-Level Fidelity:** The term "100% fidelity" does not compare the original FP32 model against the hardware. It defines that the execution in C++ produces exactly the same internal accumulators, the same logits, and the same output class as the binary model simulator executed in Python using NumPy matrices. 
*   **Multi-Architecture Determinism:** The outputs and intermediate accumulations generated by the ESP32 (32-bit arithmetic, popcount) and the Arduino Uno (8-bit arithmetic, 256-byte LUT) are bit-for-bit identical. The framework abstracts differences in *endianness* and *Integer Promotion* to guarantee equivalence.
*   **SRAM Consumption:** In the ATmega328P profile, ROM stores the weights. In SRAM (2 KB available), global variables consume 311 bytes, leaving 1737 bytes free. During inference, the local dynamic footprint is strictly 72 additional bytes (8 bytes for the packed input buffer + 64 bytes for the 512-bit hidden activations buffer).
*   **Latency Measurement:** The reported latency exclusively covers computational execution (`forward` and `argmax`) timed in hardware using native timers (`micros()`) during an uninterrupted cycle of 360 inferences. Serial port transmission overhead and initial data reading are excluded from the reported time.

> **Performance Note (AVR):** Unlike 32-bit architectures, the ATmega328P lacks native instructions to count bits. To overcome this physical limitation and avoid catastrophic GCC integer promotion bugs (which corrupt binary math), SYMBEX-1 injects a virtual hardware engine based on a precomputed 256-byte table in Flash memory. This allows 512 binarized neurons to be processed in just 18 milliseconds in an invulnerable manner.

## Credits
Carlos Duarte

Inspired by the transition from dense networks to sparse approaches (like *Mixture of Experts*) and QAT, pushed to the extreme of embedded silicon.
