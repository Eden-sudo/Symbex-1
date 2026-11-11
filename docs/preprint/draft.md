# SYMBEX-1 preprint: draft text

Prose for the sections of [outline.md](outline.md), written in English first (the repository's language); the Spanish
version follows once the text is stable. Every number comes from a file named in the outline; where a figure is still
to be copied from its JSON the text says `[JSON]`. Sections not written yet are listed at the end.

## 2. Materials and methods

### 2.1 The chain: from a trained model to an exact binary network on the chip

SYMBEX-1 has three parts. **lib_symbex** is a portable C inference engine for binary networks: bit counting instead of
multiplication, one integer threshold per output, no floating point. **The CDT** (`tools/cdt.py`) is the compiler: it
takes a PyTorch model, the data and the limits of a microcontroller, trains the binary network, folds it into integers,
writes a C header for the engine and checks that header bit by bit, on the PC and, when a board is connected, on the
chip. **benchmarks/** holds the comparisons against the official INT8 toolchain on the same chips, one folder per task,
all following the same standard (section 2.2).

**The binary layer.** Every weight and every activation is +1 or −1, stored as one bit. The product of two such values
is +1 when the bits are equal and −1 when they differ, so the dot product of two vectors of *n* values is
raw = *n* − 2·popcount(*x* ⊕ *w*). One 32-bit word holds 32 values: one XOR and one bit count replace 32 multiplications
and 32 additions. Where a chip counts faster with AND than with XOR the engine uses the equivalent form
popcount(*x*) + popcount(*w*) − 2·popcount(*x* ∧ *w*). After the count, the scale, the BatchNorm and the sign that a
trained network applies in floating point are folded into one integer threshold *T* per output: the layer emits bit 1
when raw ≥ *T* and 0 otherwise. The threshold is *T* = ⌈μ − βσ/γ⌉ from the BatchNorm's running statistics and learned
parameters; when γ < 0 the exporter negates that output's weight row so that the C always evaluates the same comparison.
Because raw is an integer and the threshold is rounded up, the folded layer gives the same bit as the floating-point
layer for every possible input: it is not an approximation. Dense networks fold more into the same threshold (weight
scale, bias, input normalization).

**Input and output.** A sensor gives bytes, not bits. Convolutional networks enter through a *thermometer* encoding:
each value becomes *levels* bits, bit *k* set when the value is at least (*k* + 1)·256/(*levels* + 1); with 8 levels a
pixel becomes 24 bits and the first layer is binary like the rest. Optionally the first convolution keeps 8-bit weights
and reads the pixels directly (`stem_i8`); its BatchNorm and sign fold into one threshold on the integer dot product, and
its output is bits, so nothing else changes. Dense networks read the `uint8` input directly: a weight of +1 adds the
value and a weight of −1 subtracts it, still without multiplications. The last layer is the only one that does not end
in a bit: it computes one score per class, scale·raw + bias, in fixed point with 64-bit products, and the largest score
wins. For convolutional networks the classifier can be pooled first (`pool2`: the number of set bits in each 2×2 window
of the last map, then an int8 linear layer), which keeps the gain of an int8 classifier with a quarter of its weights.
Hybrid networks (an int8 first layer or classifier inside a binary network) are labelled as such and reported with
their average bits per weight; "fully binary" means one bit everywhere.

**A whole network.** 2×2 max pooling is the OR of four bits (the maximum of four ±1 values is +1 when any is +1); a
residual shortcut is the OR of two maps; a change of shape is a 1×1 binary convolution after the pooling. The maps store
the channels of each pixel together, 32 channels per word, so a 3×3 window is a short run of words.

**Two engines, one export.** The *standard* engine counts the bits of one window at a time against every weight row.
The *vertical* engine runs the same exported network, with the same bits in flash and the same output bits, but turns
the loops around: it keeps one bit of each of 128 windows in a 128-bit register and counts 128 windows at once. On the
ESP32-S3 this uses the PIE SIMD unit for 128-bit logic. Two optional modes build the engine's plane lists and
thresholds once into RAM at start (*compact*, *prepared*), trading RAM for time; the deployer chooses.

**Counting bits on each chip.** The hardware abstraction layer (HAL) of lib_symbex picks the counting method per chip:
the native popcount on x86 and AArch64, the `cpop` instruction of the Zbb extension on the Pico 2's Hazard3 RISC-V
core, a 16-bit table (`lut16`) on its Cortex-M33 core and on the ESP32 (or SWAR), the SIMD multiplier of the PIE unit
on the ESP32-S3 (`pie_swar`), an 8-bit table on AVR. `tools/dev/count_bench.py` measures every method on every chip and
the profiles in `tools/chip_profiles/` record the cost per operation. Every counting function has a `*_rows` form that
computes a whole layer in one call, so the setup cost is paid once per layer.

**The compiler.** One configuration file (`symbex.toml`) describes a run, so every result can be re-created from the
file it came from. The stages are: *inspect* (layers, parameters, precision of the model), *estimate* (dense networks
only: how much each layer suffers at 1, 2, 4 and 8 bits, measured by cosine similarity on calibration samples),
*train* (dense networks: a search of width and bits per layer within the memory budget, judged on validation, then the
final training; convolutional networks: training of the network as defined, labels only by default), *export* (folding
into integers and writing the header), *validate* (the C on the PC against the integer simulation) and *chip* (the
same check and the timing on a connected board). Binary weights train with latent FP32 weights and a clipped
straight-through gradient; the Bop optimizer (Helwegen et al. 2019) is an option for the 1-bit weights of
convolutional networks (`binary_optimizer = "bop"`). Distillation from the FP32 model is available and switched off in
the benchmarks ("labels only"); when a recipe favors one method, both recipes are shown.

**Exact, always.** Two checks separate what training costs from what the implementation costs. *Fidelity* compares
the trained PyTorch model with its integer simulation: the binary layers fold exactly, the fixed-point scores of the
classifier can change a few predictions, and the CDT reports the difference. *Exactness* compares the integer
simulation with the compiled C, image by image, on the PC and on the chip, and must be 100 %: anything else is a bug in
the export or in the engine. Every result file records whether the board equalled the simulation, and every number in
this paper comes from a run where it did.

### 2.2 Materials: chips, baselines, tasks and recipe

**Chips.** Three commercial microcontrollers of three instruction sets, at their stock clock, one core:

| Board | Core | SIMD | Clock | Bit count used |
|---|---|---|---|---|
| ESP32 (Espressif) | Xtensa LX6 | none | 240 MHz | lut16 / SWAR |
| ESP32-S3 (Espressif) | Xtensa LX7 | PIE (128-bit) | 240 MHz | PIE multiplier (pie_swar) |
| Raspberry Pi Pico 2 (RP2350) | Arm Cortex-M33 | DSP (SMLAD) | 150 MHz | lut16 |
| Raspberry Pi Pico 2 (RP2350) | Hazard3 RISC-V (RV32IMAC + Zbb) | none | 150 MHz | cpop |

The Arduino Uno (ATmega328P, 8-bit AVR, 16 MHz) appears in one benchmark only (Digits), to show the smallest chip
the chain reaches; no INT8 toolchain runs there. Latency is the median over the validation images sent to the board
(1000 by default). Weights are measured **in flash first**, the way a model is normally deployed, then in RAM; every
method compared has its weights in the same place.

**Baselines.** The main reference is **INT8 through the official toolchain**: the TensorFlow Lite converter and
TensorFlow Lite Micro with the vendor kernels, ESP-NN on the ESP32 and the ESP32-S3 and CMSIS-NN on the Cortex-M33;
on Hazard3 only the reference kernels exist. The kernels are used as shipped, with no tuning by us. INT8 is the
comparison because it is what everyone deploys; it is built for accuracy, not for the smallest cost, so binary and
INT8 are expected to sit at different points of the accuracy–cost curve. Espressif's ESP-DL, the most optimized
INT8 path we know on the ESP32-S3, was measured once outside the repository (post-training quantization, 100 test
inputs) and is reported as a quick check, not a benchmark. **INT4 and INT2** (`tools/lowbit`: Brevitas
quantization-aware training, integer simulation equal to ESP-NN, our C around ESP-NN's multiply-accumulate kernels)
answer whether counting bits beats multiplying at the same bit budget; they are fair by construction but our own
implementation, since no official toolchain runs INT2 on these chips, and are reported as exploratory baselines. The
only comparison with **another binary engine**, ultracompactBNN (Daghero et al. 2021), follows that paper's setup:
its four networks, its processor (RI5CY in GVSoC) and its own code, scored in instructions.

**Tasks.** Five benchmarks, chosen to cover dense, 1D and 2D convolutional networks and tasks that range from
fine-grained classification to detection and biosignals:

| Benchmark | Data | Network | Reference model |
|---|---|---|---|
| ResNet-8 | CIFAR-10 (10 classes, 32×32 RGB) | 3×3 stem, three stages of two 3×3 convolutions, 16-32-64 channels, linear classifier | the official MLPerf Tiny INT8 ResNet-8 |
| FOMO | PASCAL VOC 2007+2012, class *person*; 96×96 RGB, 12×12 grid | stem 3×3 stride 2, three stages (48, 24, 12), 1×1 head per cell | the same network trained the same way, INT8 |
| MLP Fashion | Fashion-MNIST | 784-256-128-10 (official shape) and shapes sized for binary (784-128-512-10) | INT8 QAT of the same shape; INT4, INT2 |
| HAR 1D | UniMiB-SHAR (17 activities, phone accelerometer) | the four 1D networks of Daghero et al., Table 1 | ultracompactBNN; TFLite Micro INT8 of the same networks |
| ECG | MIT-BIH arrhythmia, AAMI classes N/S/V/F, inter-patient split (DS1 train, DS2 test) | 1D networks s/m/l (8-16-16, 16-32-32, 32-64-64 channels, kernel 9) | the same networks in INT8 `[after master is merged: verify against the README]` |
| Digits | scikit-learn Digits (8×8, 17 grey levels, 10 classes) | 64-256-10, input 1 bit per pixel | the FP32 network on the same 1-bit input |

Same architecture is the rule whenever SYMBEX is compared with INT8: the official model where one exists, or the same
network trained with the same recipe; a network resized for binary is compared with INT8 resized the same way. What
binarization imposes is the only difference: the thermometer input, OR shortcuts and OR pooling instead of sums and
maxima, bits instead of bytes.

**Recipe.** One FP32 starting model per seed, shared by every method, its sha256 recorded. The same epochs,
optimizer, seeds (42, 43, 44) and data split for every method. A validation split is carved out of the training data
and every choice (checkpoint, threshold, variant, shape, recipe) is made on it; **the test set is used once**, on the
final models, and each README says when. ECG chose its recipe by 4-fold patient-wise cross-validation inside the
training set. Results are means ± standard deviation over the three seeds; a single seed is called a ranking, not a
result; differences smaller than the standard deviations are called a tie. Figures that are not measured are called
estimates and say where they come from.

**Repeating a benchmark.** Every benchmark has the same scripts (`<bench>_symbex.py`, `<bench>_tflite.py`,
`<bench>_lowbit.py`) and the same folders (`runs/`, `results/symbex/`, `results/tflite/`, `results/lowbit/`); the
figures are drawn from the result files by `tools/dev/docs_charts.py`; a `reference/` folder per benchmark holds one
ready model (seed 42) with its header, its `.tflite` and its measured card, so a board can be measured without
retraining. The 1D comparison with ultracompactBNN runs in a Docker image with GVSoC.

## Not written yet

- Front matter (abstract and keywords in both languages): last, from the body.
- 1 Introduction and 1.1 Related work: from the outline's verified table, after the four PDFs are read.
- 3 Results and discussion: after master is merged into this branch (ECG results) and the charts are regenerated.
- 4 Conclusions, 5 References (`references.bib`).
