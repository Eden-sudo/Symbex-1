# SYMBEX-1 preprint: draft text

Prose for the sections of [outline.md](outline.md), written in English first (the repository's language); the Spanish
version follows once the text is stable. Every number comes from a file named in the outline; where a figure is still
to be copied from its JSON the text says `[JSON]`. Sections not written yet are listed at the end.

## 1. Introduction

A microcontroller has a few hundred kilobytes of RAM, a few megabytes of flash and often no SIMD unit and no
floating-point unit. The way almost everyone runs a neural network on one is INT8: the model is trained in floating
point, quantized to 8-bit integers with the vendor's converter and executed by TensorFlow Lite Micro with the vendor's
kernels (ESP-NN on Espressif chips, CMSIS-NN on Arm Cortex-M). INT8 keeps the accuracy of the floating-point model
almost intact. It also multiplies and adds one byte per weight.

A binary network keeps one bit per weight and one bit per activation. The weights take eight times less memory than
INT8, and the multiply-accumulate becomes a bit operation and a count: XOR or AND over 32 to 128 weights at a time,
then a population count. On a chip without SIMD that arithmetic is an order of magnitude cheaper than an int8
multiply-accumulate (11-17× on the classic ESP32, measured with `tools/dev/count_bench.py`). Scale, BatchNorm and the
activation fold into one integer threshold per output, so the inference has no floating point at all. The price is
accuracy, and how much depends on the task.

Binary networks are not new: the training methods date from 2015-2016 (BinaryConnect, BNN, XNOR-Net) and have been
refined since (Bi-Real, ReActNet, the Bop optimizer). Nor is running them on small hardware new: FINN compiles them
to FPGAs, daBNN and Larq Compute Engine run them on 64-bit Arm phones, and several works have put them on
microcontrollers or on research RISC-V platforms (section 1.1). What was missing, in our reading of that work, is
a chain that a developer can actually iterate with on the chips they buy, whose output is checked rather than
trusted, and a set of measurements against the toolchain everyone deploys, on several tasks, with one methodology.

**What this paper does.** SYMBEX-1 is an open, end-to-end chain for binary networks whose first target is commercial
microcontrollers. A configuration file describes the network and the chip's limits; the compiler trains the binary
network in PyTorch, folds it into integers, writes a C header for a portable inference engine and checks that header
against the integer simulation of the model, image by image, on the PC and on the board. The check must give 100 %;
it has, for every model reported here. With that chain we measured the same networks, binary and INT8, on three
commercial boards of four instruction sets (Xtensa LX6 and LX7, Arm Cortex-M33, Hazard3 RISC-V), INT8 through the
official TensorFlow Lite toolchain, three seeds, every choice made on validation and the test set used once. The
tasks go from fine-grained image classification (CIFAR-10, the MLPerf Tiny ResNet-8) through person detection, a
dense network on Fashion-MNIST, and two 1D signals (activity recognition from an accelerometer, arrhythmia from the
ECG), to a digit classifier on an 8-bit Arduino.

**What we found.** Binary pays most where the chip has no SIMD: on the classic ESP32 whole networks run 5 to 13
times faster than INT8 with 3 to 10 times less flash. Where the chip has an int8 SIMD unit the standard engine wins
by 1.0 to 2 times on the convolutional networks (28 times on the dense network with its weights in flash, where INT8
pays for every byte read), and beating the vendor's own INT8 library took a second engine that counts 128 windows at once.
The accuracy cost is the task's, not the method's: none on person detection and on the ECG, two points on
Fashion-MNIST, ten points on CIFAR-10, where a study shows the binary network loses by capacity, not by training,
and 9 to 23 points on activity recognition, worst on the smallest network.
An 8-bit first layer and a pooled 8-bit classifier recover part of it at a small cost. `[every number in this paragraph
is copied from the results tables of section 3; re-read them in the final pass]`

**Contributions.** (1) An open chain from a PyTorch model to a C header that is verified exact on the chip, with a
configuration file that re-creates every run. (2) A portable engine with one hardware layer per chip family and a
second, bit-sliced engine for SIMD units. (3) Measurements of the same network, binary and INT8, on commercial chips
of four ISAs against the official INT8 toolchain, with one methodology across five tasks. (4) An account of where one
bit pays and where it does not, including the capacity study of the binary ResNet-8. Everything, code, models and
result files, is in the repository.

### 1.1 Related work

**Training binary networks.** BinaryConnect (Courbariaux et al. 2015) and BNN (Hubara et al. 2016) train with latent
real weights and a straight-through gradient; XNOR-Net (Rastegari et al. 2016) adds scaling factors; Bi-Real
(Liu et al. 2018) and ReActNet (Liu et al. 2020) change the block to keep information through the sign; Bop
(Helwegen et al. 2019) optimizes the bits directly. SYMBEX-1 trains with the straight-through gradient and offers Bop
as an option; the training-gap study of section 3 measures what those methods are worth on the binary ResNet-8.

**Binary networks on hardware.** Three lines of work precede ours. *Programmable logic:* FINN (Umuroglu et al. 2017)
compiles binary networks to FPGAs through Brevitas and high-level synthesis; it produces hardware, not code for a
microcontroller. *Application processors:* daBNN (Zhang et al. 2019) and the Larq Compute Engine (Bannink et al.
2021) run binary networks on 64-bit Arm phones with NEON. Larq's maintainers state that the engine "is meant more for
mobile-class devices with CPUs such as the Arm Cortex-A" and that the ESP32 family "isn't the target"; for
microcontrollers Plumerai "does offer commercial solutions", with no public source. *Research RISC-V platforms:*
Garofalo et al. (PULP-NN, 2020) wrote 8-bit and sub-byte kernels for PULP and XpulpNN extended the ISA; Daghero et
al. (ultracompactBNN, 2021) ran 1D binary networks for activity recognition on PULPissimo in the GVSoC simulator,
against a random forest and a padded PULP-NN network, not against INT8; xTern (Rutishauser et al. 2024) adds ternary
instructions to a RISC-V cluster. Cerutti et al. (2020) ran a binary sound-event detector on the commercial GAP8 and
reported energy per inference; the only on-device baseline is the same network on a Cortex-M4F.

**Binary networks on commercial microcontrollers.** eBNN (McDanel, Teerapittayanon and Kung, 2017) is the closest
ancestor: a Chainer model "automatically converted into a standalone C header file" and run on an Intel Curie at
32 MHz with about 15 KB of usable RAM, on MNIST (94.6 % at 53.7 ms with a small convolutional network). The authors
"validated the correctness of the C implementation by comparing the output at each stage to the Python version"; the
chain is open, and both the framework and the chip have since been discontinued. There is no INT8 baseline. CBin-NN
(Sakr et al. 2024) is the closest engine: a C library of binary layers for Cortex-M, fed by a Python script that
writes the parameters of a Larq model, measured on an STM32F746 on CIFAR-10 against the same network in INT8 on
TensorFlow Lite Micro, CMSIS-NN, MicroTVM and TinyEngine: 3.6× faster than TensorFlow Lite Micro and 20 % faster than
CMSIS-NN, 14 KB instead of 87 KB, 77.4 % against 79.9 %. Two of their findings agree with ours: Bop helps, and an
8-bit first layer on 8-bit input is worth its cost. Their comparison covers one task, one instruction set and one
seed, and the paper does not check the C output against the Keras model.

**The INT8 toolchains.** TensorFlow Lite Micro and, more recently, ExecuTorch's Cortex-M backend ("in beta", INT8
only, validated on a simulator) are the official paths to a microcontroller; neither has 1-bit kernels or folds a
layer into a threshold. Espressif's ESP-DL is the most optimized INT8 path we know on the ESP32-S3; we report one
measurement against it. INT4 and INT2 have kernels in CMSIS-NN and ESP-NN but no end-to-end toolchain on these
chips; the low-bit baselines in this paper are our own code and are labelled as such.

**What SYMBEX-1 adds** to this picture is the combination: a chain whose output is verified exact on every model, on
the PC and on the chip; commercial chips of four instruction sets; the official INT8 toolchain as the baseline;
five tasks under one methodology (same architecture, three seeds, validation for every choice, test once); and a
second engine for SIMD units. We do not claim to be the first chain, the first engine or the first comparison; eBNN,
CBin-NN and the works above did each part before. We claim the measurements and the tool that produced them.

## 2. Materials and methods

### 2.1 The chain: from a trained model to an exact binary network on the chip

**What a developer does.** The chain exists to make one loop short: describe the network and the chip's limits in a
configuration file, train, fold into integers, generate the C header, and check it against the integer simulation on
the PC and on the board, image by image. Everything in this paper (the engines, the recipes, the studies on the stem,
the shortcuts and the training gap) was produced by running that loop many times; without it, every variant would
have meant hand-written C and a round of debugging against Python. The loop is the tool; the measurements are what
it was used for.

SYMBEX-1 has three parts. **lib_symbex** is a portable C inference engine for binary networks: bit counting instead of
multiplication, one integer threshold per output, no floating point. **The CDT** (`tools/cdt.py`) is the compiler: it
takes a PyTorch model, the data and the limits of a microcontroller, trains the binary network, folds it into integers,
writes a C header for the engine and checks that header bit by bit, on the PC and, when a board is connected, on the
chip. **benchmarks/** holds the comparisons against the official INT8 toolchain on the same chips, one folder per task,
all following the same standard (section 2.2).

**The binary layer.** Every weight and every activation is +1 or −1, stored as one bit. The product of two such values
is +1 when the bits are equal and −1 when they differ, so the dot product of two vectors of *n* values is
raw = *n* − 2·popcount(*x* ⊕ *w*). One 32-bit word holds 32 values: one XOR and one bit count replace 32 multiplications
and 32 additions. Where a chip counts faster with AND than with XOR the engine replaces the XOR term by the identity
popcount(*x* ⊕ *w*) = popcount(*x*) + popcount(*w*) − 2·popcount(*x* ∧ *w*), that is,
raw = *n* − 2·(popcount(*x*) + popcount(*w*)) + 4·popcount(*x* ∧ *w*); popcount(*w*) is a constant per weight row. After the count, the scale, the BatchNorm and the sign that a
trained network applies in floating point are folded into one integer threshold *T* per output: the layer emits bit 1
when raw ≥ *T* and 0 otherwise. During training the layer computes BatchNorm and then the sign, bit = [γ·(raw − μ)/σ + β ≥ 0], where μ and σ are
the running mean and standard deviation of raw and γ and β are the learned scale and bias. For γ > 0 this is
raw ≥ μ − βσ/γ, and since raw is an integer the threshold is *T* = ⌈μ − βσ/γ⌉; when γ < 0 the inequality flips to raw ≤ μ − βσ/γ, and the
exporter negates that output's weight row (which negates raw) and uses *T* = ⌈−(μ − βσ/γ)⌉, so the C always
evaluates the same comparison; when γ = 0 the output is constant and the threshold is set to the smallest or the
largest integer according to the sign of β.
Because raw is an integer and the threshold is rounded up, the folded layer gives the same bit as the floating-point
layer for every possible input: it is not an approximation. Dense networks fold more into the same threshold (weight
scale, bias, input normalization).

**Input and output.** A sensor gives bytes, not bits. Convolutional networks enter through a *thermometer* encoding:
each value becomes *levels* bits, bit *k* set when the value is at least (*k* + 1)·256/(*levels* + 1); the levels are per
color channel, so with 8 levels an RGB pixel becomes 3 × 8 = 24 bits, and the first layer is binary like the rest. Optionally the first convolution keeps 8-bit weights
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
| ECG | MIT-BIH arrhythmia, AAMI classes N/S/V/F, inter-patient split (DS1 trains, DS2 is the test set); one 256-sample window per beat plus three constant RR channels | 1D networks s/m/l (three convolutions of width 9, 8-16-16, 16-32-32, 32-64-64 channels, two max pools, one dense layer) | the same networks in INT8 (Keras, full-integer post-training quantization) |
| Digits | scikit-learn Digits (8×8, 17 grey levels, 10 classes) | 64-256-10, input 1 bit per pixel | the FP32 network on the same 1-bit input |

Same architecture is the rule whenever SYMBEX is compared with INT8: the official model where one exists, or the same
network trained with the same recipe; a network resized for binary is compared with INT8 resized the same way. What
binarization imposes is the only difference: the thermometer input, OR shortcuts and OR pooling instead of sums and
maxima, bits instead of bytes.

**Recipe.** The standard is: one FP32 starting model per seed, shared by every method, its sha256 recorded; the
same epochs, optimizer, seeds (42, 43, 44) and data split for every method; a validation split carved out of the
training data where every choice (checkpoint, threshold, variant, shape, recipe) is made; **the test set used once**,
on the final models, with each README saying when. ECG chose its recipe by 4-fold patient-wise cross-validation
inside the training set. Results are means ± standard deviation over the three seeds; a single seed is called a
ranking, not a result; differences smaller than the standard deviations are called a tie. Figures that are not
measured are called estimates and say where they come from.

Not every published measurement meets the standard yet, and section 3 says so where it applies: the INT8 FOMO
detector was trained with the floating-point recipe (person weight 20, 60 epochs, where it had plateaued) while the
binary one used weight 40 and 100 epochs, and its published detection thresholds were calibrated on an earlier split;
the ResNet-8 shapes sized for binary were ranked on one seed; the sha256 of the FP32 model is not yet in every result
file. Where a result predates the standard, the paper labels it a ranking or an exception rather than a result.

**Repeating a benchmark.** Every benchmark has the same scripts (`<bench>_symbex.py`, `<bench>_tflite.py`,
`<bench>_lowbit.py`) and the same folders (`runs/`, `results/symbex/`, `results/tflite/`, `results/lowbit/`); the
figures are drawn from the result files by `tools/dev/docs_charts.py`; a `reference/` folder per benchmark holds one
ready model (seed 42) with its header, its `.tflite` and its measured card, so a board can be measured without
retraining. The 1D comparison with ultracompactBNN runs in a Docker image with GVSoC.

## 3. Results and discussion

Every number below is copied from a result file in the repository (`benchmarks/*/results/`) or from the table of a
benchmark README that is built from those files; the path is given with each table. Latencies are medians over the
inputs sent to the board, averaged over the three seeds; the seeds agree to 0.01-0.1 ms, so the standard deviation of
a latency is omitted. Accuracies are mean ± standard deviation over the three seeds unless the reference is a single
published model. "Exact" means that the chip gave the class of the integer simulation on every input of the run.

### 3.1 Results

**Image classification: ResNet-8 on CIFAR-10.** The reference is the official MLPerf Tiny INT8 ResNet-8 (87.0 % on
the 10,000 test images). Two SYMBEX networks of the same graph are measured: the fully binary one (every layer 1 bit,
the image entering as a thermometer of bits) and the most accurate recipe the chain produced (an int8 first layer on
the pixels, Bop for the 1-bit weights and a pooled int8 classifier; every other layer binary). Weights in flash, one
core (`cnn_cifar_resnet8/results/symbex/symbex_seeds_<board>_*.json`, `results/tflite/tflite_<board>.json`):

| | Fully binary | int8 stem + Bop + pooled classifier | MLPerf Tiny INT8 (TFLite Micro) |
|---|---|---|---|
| Test accuracy | 71.3 ± 0.8 % | 76.7 ± 0.3 % | 87.0 % |
| ESP32-S3, standard engine | 39.2 ms (1.33×) | 32.3 ms (1.61×) | 52.0 ms (ESP-NN) |
| ESP32-S3, vertical engine | 10.4 ms (5.0×) | 9.70 ms (5.4×) | |
| ESP32 | 63.3 ms (11.3×) | 62.5 ms (11.5×) | 717.8 ms (ESP-NN) |
| Pico 2, Cortex-M33 | 74.5 ms (6.1×) | 67.8 ms (6.7×) | 453.0 ms (CMSIS-NN; 211.4 ms with the model in RAM) |
| Pico 2, Hazard3 | 56.1 ms | 57.9 ms | 5702 ms (reference kernels, 100 images) |
| Flash (the model) | 24.3 KB | 29.0 KB | 96.2 KB |
| RAM | 15 KB (79 KB with the 64 KB count table) | 15 KB | 53 KB (ESP32), 80.5 KB (S3) |
| Chip equal to its PC reference | 100 % | 100 % | 98.1 % |

The vertical engine's modes trade RAM for time on the S3: default 10.4 ms with 63 KB, compact 7.70 ms with 151 KB,
prepared 6.44 ms with 238 KB (fully binary); 9.70 / 7.07 / 6.02 ms with the most accurate recipe. On the ESP32 and
the Pico 2 the standard engine stays the fastest (section 2.1). Sizing the network for binary rather than copying the
INT8 shape gives 73.1 ± 0.2 % (16-32-128 channels, 1-1-2 blocks) at the same S3 latency (39.6 ms) and 74.5 ± 0.2 %
(16-64-128) at 46.6 ms; those shapes were ranked on one seed (section 2.2), so they are reported as a ranking.

**Person detection: FOMO on Pascal VOC.** The same detector in INT8, trained with the recipe noted in section 2.2
(`fomo_voc_person/results/symbex/symbex_seeds_*.json`, `results/tflite/tflite_s3_*.json`):

| | SYMBEX binary | TFLite INT8 |
|---|---|---|
| F1, person cells (3 seeds) | 31.2 ± 0.7 % | 30.1 ± 0.2 % |
| ESP32-S3 | 126.3 ms (2.03×) | 256.9 ms |
| ESP32 | 226.3 ms | does not fit (the arena overflows the RAM) |
| Flash | 41.2 KB | 236.1 KB |
| RAM | 51 KB | 223 KB |
| Chip equal to its PC reference | 100 % | 88 % |

Detection binarizes without loss: every grid cell answers a yes/no question, and the binary detector is as accurate as
the INT8 one, twice as fast on the S3, 5.7× smaller, and the only one of the two that runs on the classic ESP32.

**Dense network: Fashion-MNIST.** The shape sized for binary (784-128-512-10: a narrow first layer, which is the
expensive one on the chip because it sums 8-bit pixels, and a wide binary layer), against INT8 of the same shape,
3 seeds, labels only (`mlp_fashion/results/sizing/fashion_sizing.json`, `results/tflite/tflite_s3_h128_512*.json`,
`results/tflite/tflite_esp32_h128_512.json`):

| 784-128-512-10 | SYMBEX binary | INT8 (same shape) |
|---|---|---|
| Test accuracy | 87.5 ± 0.1 % | 89.8 ± 0.1 % (quantization-aware training) |
| ESP32-S3, weights in flash | 0.420 ms (27.8×) | 11.67 ms (TFLite Micro) |
| ESP32-S3, weights in RAM | 0.414 ms (2.14×) | 0.887 ms |
| ESP32, weights in flash | 3.55 ms (8.8×) | 31.11 ms |
| Flash | 27.3 KB | 187.0 KB |
| RAM | 4.6 KB | 6.7 KB (arena), plus the weights when in RAM |

The 27.8× is a flash effect, not an arithmetic one: the INT8 model reads 8× more bytes through the flash cache. With
both models in RAM the gap is 2.14×, and Espressif's ESP-DL runs the same INT8 MLP in 0.317 ms, 1.31× faster than
SYMBEX (quick check, section 2.2). On the Pico 2's Cortex-M33, TFLite Micro with CMSIS-NN takes 31.35 ms in flash and
2.89 ms in RAM against SYMBEX's 4.57 and 1.53 ms (6.9× and 1.9×).

The low-bit study on the official shape (784-256-128-10, `mlp_fashion/results/lowbit/`) answers the question of
section 2.2, bits for bits: per byte, the low-bit integers are more accurate. INT2 QAT (57.3 KB) reaches
89.3 ± 0.1 %, one point above a binary network widened to a similar size (SYMBEX x2, 65.3 KB, 88.3 ± 0.2 %) and 1.6
above the x1 binary network (28.7 KB, 87.6 ± 0.4 %); INT4 (114.6 KB) 90.0 %, INT8 QAT (229.2 KB) 90.0 %. On the S3
with the weights in flash the x1 binary network runs in 1.24 ms against 13.4 ms for INT2 and 10.3 ms for INT4 on
ESP-NN's kernels (the sub-byte weights are unpacked to int8 and multiplied); in RAM INT4 / INT2 / INT8 all run in
0.556 ms against 0.525 ms for binary. Binary keeps the smallest and fastest corner; INT2 and INT4 keep accuracy per byte.

**Activity recognition: 1D convolutions on UniMiB-SHAR.** The four networks of Daghero et al. (min, max-10 %,
max-5 %, max), 17 activities, 3 seeds (`har_ultracompact/results/symbex/`, `results/ucbnn/`, `results/tflite/`).
Against their library, in their environment (RI5CY core in GVSoC, their code unchanged), SYMBEX needs 1.07× (min),
1.37× (max-10 %), 1.23× (max-5 %) and 1.23× (max) fewer instructions for the whole network, and 1.06-1.37× when it
runs their own trained network (int8 first layer, same weights); both are exact. Against TFLite Micro INT8 on the
boards, weights in flash (the fastest form of the binary first layer in each row):

| Network | Test accuracy, binary / INT8 | ESP32-S3 (ESP-NN) | ESP32 (ESP-NN) | Pico 2 M33 (CMSIS-NN) | Flash, binary / INT8 |
|---|---|---|---|---|---|
| min | 32.7 ± 2.7 / 56.1 ± 4.2 % | 0.49 / 0.72 ms (1.47×) | 0.78 / 4.17 (5.3×) | 0.82 / 2.71 (3.3×) | 0.7 / 7.3 KB |
| max-10 % | 53.4 ± 1.9 / 62.4 ± 2.0 % | 1.83 / 2.93 (1.60×) | 4.01 / 31.23 (7.8×) | 3.64 / 12.00 (3.3×) | |
| max-5 % | 56.8 ± 0.7 / 66.2 ± 2.6 % | 3.76 / 7.14 (1.90×) | 10.99 / 114.22 (10.4×) | 9.89 / 78.04 (7.9×) | |
| max | 58.8 ± 0.9 / 67.9 ± 0.9 % | 3.83 / 7.11 (1.85×) | 12.78 / 115.62 (9.0×) | 11.72 / 77.70 (6.6×) | 5.6 / 42.5 KB |

Here INT8 is clearly more accurate: about 9 points on the three large networks and 23 on the smallest. The
accelerometer is an 8-bit input, and the benchmark's experiment with an int8 first layer (the option `symbex_bool1d`
offers) is both more accurate (+4.5-4.8 points on the two small networks, equal on the large ones) and faster on every
board, because a byte-wide input costs about one operation per sample whether the weights have 1 bit or 8, and the
chips multiply bytes faster than they count bit planes. The published rows keep the fully binary network because that
is what the benchmark set out to measure; the int8 first layer is the recipe the chain recommends for byte inputs.

**ECG arrhythmia: 1D convolutions on MIT-BIH, unseen patients.** AAMI classes N / S / V / F, the inter-patient split
of de Chazal (DS1 trains, DS2 is the test set), three networks, the recipe (three RR channels, no augmentation) chosen
by 4-fold patient-wise cross-validation inside DS1 for both methods (`ecg_mitbih/results/symbex/*_rr/`,
`results/tflite/tflite_<board>_<net>_s<seed>_rr.json`). Balanced accuracy (mean of the four sensitivities; N is 89 %
of the test beats) and latency per beat, weights in flash:

| Network | Balanced accuracy, binary / INT8 | ESP32-S3 | ESP32 | Pico 2 M33 | Pico 2 Hazard3 (reference kernels) | Model size |
|---|---|---|---|---|---|---|
| s (8-16-16) | 67.0 ± 4.6 / 70.9 ± 6.1 % | 3.03 / 3.14 ms (1.04×) | 5.47 / 32.95 (6.0×) | 5.70 / 11.66 (2.0×) | 5.75 / 225.5 (39×) | 1.3 / 11.4 KB |
| m (16-32-32) | 71.2 ± 3.3 / 71.9 ± 4.6 % | 6.06 / 6.86 (1.13×) | 11.69 / 94.48 (8.1×) | 11.74 / 35.10 (3.0×) | 10.55 / 800.0 (76×) | 3.6 / 23.7 KB |
| l (32-64-64) | 75.0 ± 5.5 / 72.4 ± 8.8 % | 13.55 / 21.59 (1.59×) | 30.55 / 408.46 (13.4×) | 29.60 / 89.52 (3.0×, both in RAM) | 23.25 / 3554.5 (153×) | 10.1 / 68.6 KB |

The two methods are tied: no gap is larger than one standard deviation, and they err differently (the binary networks
find more S beats, 74-85 % against 53-63 %; the INT8 ones more N and V). The spread is the data's: the rare classes
rest on very few patients (75 % of the test S beats come from one record, and F is learned from one patient and
tested on another), so neither method is a clinical result, and the positive predictive value of S is 24-33 % for
both at this operating point. A second recipe chosen by the same cross-validation (class weights raised to 0.5,
checkpoint on macro F1) cuts the binary networks' false S alarms by 40-60 % at the cost of sensitivity; the benchmark
reports both operating points and the three scorings of the test set. The Hazard3 column measures TFLite Micro's
reference kernels, the only ones it has for that core, not a tuned INT8. All 39 chip runs (4 cores × 3 networks × 3
seeds, plus the M33 RAM variant) are exact.

**Digits on an Arduino Uno.** The smallest chip of the study, where no INT8 toolchain runs: scikit-learn Digits
(8×8, 10 classes), input 1 bit per pixel, network 64-256-10 binary from end to end (`mlp_digits/README.md`,
`results/`). 94.5 ± 1.3 % on the test images, the accuracy an FP32 network reaches on the same 1-bit input (94.4 %);
5.3 ms per image at 16 MHz, 3.4 KB of flash and 0.21 KB of RAM, exact. The FP32 original (64-128-10, 98.3 %) needs
38.9 KB of weights and does not fit the Uno's flash.

**The cost of going binary, task by task.** The accuracy gap against INT8 of the same network, from the tables above:

| Task | Binary | INT8 | Cost |
|---|---|---|---|
| Person detection (FOMO), F1 | 31.2 ± 0.7 % | 30.1 ± 0.2 % | none |
| ECG beats, balanced accuracy, 3 networks | 67.0-75.0 % | 70.9-72.4 % | none (tied within one standard deviation) |
| Digits, 1-bit input (reference: FP32 on the same input) | 94.5 ± 1.3 % | 94.4 % | none |
| Fashion-MNIST MLP, same shape | 87.5 ± 0.1 % | 89.8 ± 0.1 % | 2.3 points |
| CIFAR-10 ResNet-8 (reference: the MLPerf Tiny model) | 71.3 ± 0.8 / 76.7 ± 0.3 % | 87.0 % | 10.3-15.7 points |
| Activity recognition, 4 networks | 32.7-58.8 % | 56.1-67.9 % | 9.0-23.4 points |

### 3.2 Discussion: where binary pays, and why it loses where it loses

**The chip decides the speed-up.** Without SIMD (ESP32, Hazard3) whole networks run 5-13× faster than INT8, because
counting bits is an order of magnitude cheaper than 8-bit multiply-accumulate there (the raw kernel measurement of
section 1: 11-17×). With an 8-bit SIMD unit (ESP32-S3) or DSP instructions (Cortex-M33) INT8 gets most of that back:
the standard engine wins by 1.0-2× on the convolutional networks and by 2-3× on the M33, and on the S3 the vendor's
own INT8 library (ESP-DL) is faster than the standard engine. Beating it took a second engine written for that unit
(the vertical engine, 1.4-1.5× faster than ESP-DL with the weights in flash, 1.06× slower with both in RAM at a smaller
RAM budget, faster again in the compact and prepared modes at a larger one). The exception to "1-2× with SIMD" is the
MLP in flash (28×), which is a memory effect: the binary weights fit the flash cache and the INT8 ones do not. A
sub-byte INT4 / INT2 model does not recover that, because the kernels unpack it to int8 before multiplying.

**Flash and RAM pay everywhere.** 3-10× less flash on every benchmark (the model itself), and less RAM: ResNet-8
15 KB against 53-80 KB, FOMO 51 against 223 KB, the 1D networks 2-10× less (the exception is the classic ESP32's
optional 64 KB bit-count table, which the ESP32 rows of ResNet-8 and FOMO use for speed). FOMO is the clearest case:
the INT8 detector's arena does not fit the classic ESP32 and the binary one runs there.

**Capacity, not training, sets the accuracy cost.** The training-gap study (`experiments/training_gap/`) trains three
networks of one graph with one recipe: the binary ResNet-8 reaches 70.0 ± 1.1 % on validation and fits 72 % of its
training images; its "twin" (the real-valued network the straight-through estimator pretends to train) reaches
83.6 % and fits 92 %; a ReLU network 85.4 % / 95 %. No training change closes that: Bop +2.5 points, the two-step
schedule +2.35, clipping +1.7, longer training and distillation less, and they do not add up. Twice the channels adds
+10.8 points (80.8 %) with 3.3× the weights; an int8 first layer +5.4 (75.4 %, 76.1 % with Bop). Half of the
binary network's activation inputs sit where the estimator passes no gradient (46 % against 21 % in the twin). The
practical reading is the one the benchmarks show: tasks that need fine detail (CIFAR-10's ten classes, 17 activities
from an accelerometer) pay 9-16 points, and the smallest networks pay most; detection, beat classification and a
binary input pay nothing.

**Size the network for binary, and give byte inputs a byte layer.** The best binary shape is not the INT8 one: a
narrow first layer and wide binary layers in the MLP (784-128-512-10 runs 13× faster than the wider 784-512-256-10 for
1.2 points), thinner early stages and a wider last one in the ResNet (+1.8 points at the same speed). Where the input
is 8-bit (pixels, accelerometer, ECG samples), the first layer with int8 weights is more accurate on the image and
activity tasks (+5.4 and +4.5-4.8 points) and faster wherever the chip has SIMD (S3 stem 1.79×, M33 1.13×,
`experiments/chip_kernels/`); on plain
RISC-V and the classic ESP32 it is slower (0.62× and 0.86× for the stem alone) and the chain keeps the binary first
layer there. On the ECG it did not pay (67.9 / 65.4 / 68.8 % against 67.0 / 71.2 / 75.0), so the rule is measured per
task, which the chain makes cheap.

**What the loop produced.** Everything above that is not the first fully binary network came out of iterations of
the chain, each a configuration change, a training run, an export and a 100 % check: the int8 stem (a study of four
stem / shortcut variants), Bop as an option, the pooled int8 classifier, the vertical engine and its modes, the
sizing studies, the ECG recipe chosen by cross-validation. The exactness check is what made this safe: an export that
disagreed with its simulation was a bug to fix, never a result to report.

### 3.3 Limitations

- **Fine-grained classification stays behind INT8.** 10-16 points on CIFAR-10, 9-23 on activity recognition; the
  training-gap study says the cause is capacity, so closing it costs width (and speed), not a better optimizer.
- **ESP-DL is faster than the standard engine on the S3**, and faster than the vertical engine with both models in RAM
  at its RAM budget. The ESP-DL numbers are a quick check (one run, 100-200 inputs), not a 3-seed benchmark.
- **The INT8 side is the official path, not the fastest possible.** The vendor kernels are used as shipped; on
  Hazard3 only reference kernels exist, so its INT8 column measures what runs today.
- **Energy is not measured.** Latency and memory are; a current meter is the next measurement.
- **INT4 / INT2 / ternary are our own implementation**, fair by construction but not an official toolchain; they are
  reported as exploratory.
- **Some published measurements fall short of the recipe standard** (section 2.2): the FOMO INT8 training budget, the
  FOMO thresholds calibrated on an earlier split, the one-seed ranking of the sized ResNet-8 shapes.
- **ECG accuracy is bounded by the data**, not the method: few patients per rare class, so the standard deviations
  reach 9 points and the positive predictive value of S and F is low for both methods. It is not a clinical result.
- **Recurrent networks are out of scope.** A local probe of a binary Elman network on the activity task trained
  (38 % against 39 % for int8, far below the convolutional networks) but was fragile: flipping 1 % of the state bits
  changed 28 % of its predictions. Nothing of it is in the chain. `[local probe only (LOGBOOK.md outside the
  repository), no result file: Edén decides whether this bullet stays]`

## Not written yet

- Front matter (abstract and keywords in both languages): last, from the body.
- 4 Conclusions (from the outline's table), 5 References (`references.bib`).
- Final pass: re-read the "What we found" paragraph of the Introduction against the tables of section 3.
