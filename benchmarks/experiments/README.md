# experiments -- pairs of bits in binary networks

**Question:** in a network where everything is binary, does giving each neuron a *pair* of bits help it learn
more than a flat 1-bit network?

**Answer:** yes, when the pair goes to the **activations**. Two bits per activation, combined as a plain
**sum of counts** (no multiplications), give **+2 to +3 points** of accuracy with **the same memory**, in both a
fully connected network and a convolutional one. In the fully connected network they are also better per
operation: a pair network beats a flat network twice its size.

## What is compared

Every network is binary from end to end: input, weights and activations are bits, and every product is an
XNOR/AND between bits followed by a count.

| | Weights | Activations |
|---|---|---|
| **flat** | 1 bit | 1 bit: `s = sign(x - t)` |
| **pair (activations)** | 1 bit | 2 bits: `s1 = sign(x - t1)`, `s2 = sign(x - t2)`, with `t1 < t2` learned per neuron |

With `--act-scale equal` the two activation bits count the same, so a layer computes
`count(s1, w) + count(s2, w)`: two binary counts and one addition, no multiplication anywhere. The weights do not
change, so the memory does not change either.

Networks: an MLP on Fashion-MNIST (784 -> H -> H/2 -> 10) and a binary ResNet-8 on CIFAR-10 (channels c/2c/4c,
input encoded as 24 bit planes). Plain training, no distillation, 20 epochs, 3 seeds, NVIDIA RTX 4000 Ada.

## Results

### Same memory: pair vs flat of the same size

| Network | Flat | Pair (activations, sum of counts) | Gain |
|---|---|---|---|
| MLP, Fashion-MNIST (28.7 KB) | 86.11% | **88.30%** | **+2.19** |
| ResNet-8, CIFAR-10 (9.5 KB) | 48.70% | **50.80%** | **+2.10** |
| ResNet-8, CIFAR-10, thermometer input (9.5 KB) | 51.34% | **54.34%** | **+3.00** |

### Accuracy vs binary operations (MLP, Fashion-MNIST)

| Network | Accuracy | Binary ops | Memory |
|---|---|---|---|
| flat, H = 128 | 85.15% | 109 k | 13.3 KB |
| **pair, H = 128** | **87.71%** | 218 k | 13.3 KB |
| flat, H = 256 | 86.11% | 235 k | 28.7 KB |
| flat, H = 384 | 86.37% | 377 k | 46.0 KB |
| **pair, H = 256** | **88.30%** | 470 k | 28.7 KB |
| flat, H = 512 | 86.53% | 535 k | 65.3 KB |
| **pair, H = 384** | **88.75%** | 753 k | 46.0 KB |
| **pair, H = 512** | **88.89%** | 1.07 M | 65.3 KB |

The smallest pair network (H = 128) is more accurate than the largest flat one (H = 512), with a fifth of the
memory and less than half of the operations.

### Accuracy vs binary operations (ResNet-8, CIFAR-10)

| Network | Accuracy | Binary ops | Memory |
|---|---|---|---|
| flat, c = 8 | 41.32% | 4.7 M | 2.5 KB |
| pair, c = 8 | 42.64% | 7.7 M | 2.5 KB |
| flat, c = 16 | 48.89% | 15.3 M | 9.5 KB |
| pair, c = 16 | 50.69% | 27.1 M | 9.5 KB |
| flat, c = 24 | 53.09% | 31.9 M | 21.0 KB |
| flat, c = 32 | 55.95% | 54.3 M | 37.0 KB |
| pair, c = 24 | 55.35% | 58.4 M | 21.0 KB |
| pair, c = 32 | 58.30% | 101.5 M | 37.0 KB |

In the convolutional network the pairs win at the same memory; at the same number of operations, extra channels
are worth about as much. Pairs are the choice when memory is the limit (the usual case on a microcontroller).

### Where the pair goes

| ResNet-8, same memory | Gain |
|---|---|
| pair in the activations | **+2.5** |
| pair in the weights | about 0 |

| MLP, same operations | Gain |
|---|---|
| pair in the activations | **+1.7** |
| pair in the weights | about 0 |

The information a binary network loses is in its activations: that is where the second bit pays off.

### Input encoding (ResNet-8)

Encoding each pixel as a **thermometer** (8 levels per channel) instead of its 8 bit planes gives **+2.6 points**
(flat) and **+3.5 points** (pair), with the same 24 input planes.

### How the two activation bits are combined

| MLP, Fashion-MNIST | Accuracy |
|---|---|
| learned scale per neuron (needs multiplications) | 88.38% |
| **equal: sum of counts (no multiplications)** | **88.30%** |

The count-only version keeps the whole gain.

## Related work

FracBNN (Zhang et al., FPGA 2021) also improves binary networks with up to two bits per activation and a
thermometer-encoded input, on FPGAs. These experiments reach the same conclusion independently, with a
count-only arithmetic aimed at microcontrollers.

## Reproduce

From `benchmarks/experiments/`, in the main environment (`.venv`):

```
python pair_bits.py --pair-on activations --act-scale equal
python pair_bits.py --pair-on activations --act-scale equal --widths 0.5,1,1.5,2
python pair_bits.py --arch resnet8 --pair-on activations --act-scale equal
python pair_bits.py --arch resnet8 --pair-on activations --act-scale equal --widths 0.5,1,1.5,2
python pair_bits.py --arch resnet8 --pair-on activations --act-scale equal --input thermometer
python pair_bits.py --arch resnet8 --pair-on weights
```

Results land in `results/`. Measured in October 2026.

## Other probes in this folder

Quick, separate tests, each documented in its own docstring:

| Script | Question |
|---|---|
| `pie_conv_bench.py` | the AND + PIE arithmetic of the convolutions on the ESP32-S3 |
| `pie_swar_bench.py` | a faster way to count bits with the ESP32-S3's multiplier? |
| `int8_first_bench.py` | does an int8 first layer speed up mlp_fashion and ResNet-8, as it does conv1d? Every board (see below) |
| `s8_dot_rows_chip_test.py` | is the S3 PIE kernel `symbex_s8_dot_rows` exact at every alignment? (9,792 cases on the chip, weights in RAM and flash: all equal to C) |
| `poplut_probe.py` | PopLUT (codebook-compressed binary weights) on mlp_fashion: a falsification probe |
| `layer_probe_study.py` | layer by layer: linear probes, threshold margins, shuffled-filter null test, bit-flip noise (see below) |
| `train_gap_study.py` | how much of the binary ResNet-8's accuracy loss is training, not bits? (see below) |
| `small_gap_study.py` | the same two studies on conv1d (UniMiB-SHAR) and the Fashion-MNIST MLP (see below) |
| `gpu_bench.py` | every arithmetic (FP32, TF32, FP16, BF16, INT8, 1 bit) on the CUDA GPU, same matrix product |

## An int8 first layer for ResNet-8 (`int8_first_bench.py`)

ResNet-8's stem alone (32×32, 3×3, 16 channels), random weights, every method checked against plain C on the chip,
weights in RAM, best of 7 runs (results in `results/int8_first_bench_<board>.json`):

| Chip | thermometer + binary stem | int8 stem | int8 / thermometer | int8 dot products with |
|---|---|---|---|---|
| ESP32-S3 | 8.13 ms | **4.55 ms** | **1.79× faster** | PIE, 16 per instruction |
| Pico 2, Cortex-M33 | 15.47 ms | **13.72 ms** | **1.13× faster** | SMLAD, 2 per instruction |
| Pico 2, Hazard3 | **11.66 ms** | 18.94 ms | 0.62× | plain C |
| ESP32 | **13.35 ms** | 15.44 ms | 0.86× (flash: 0.80×) | plain C |

The int8 stem wins only where the chip multiplies bytes in SIMD; it is also more accurate (ResNet-8 on CIFAR-10, 3
seeds: 75.6 % against 71.2 % on validation). It is now an option of the CDT's `boolean` method and of lib_symbex
(`symbex_bool_stem_i8`); the whole network with it is measured in `../conv2d/cnn_cifar_resnet8`.

## How much does training cost the binary ResNet-8? (`train_gap_study.py`)

**Question:** the fully binary ResNet-8 of `../conv2d/cnn_cifar_resnet8` loses ~15 points against a floating-point
ResNet-8 of the same graph. How much of that comes from the training (Adam on latent weights + the straight-through
estimator) and how much from the bits themselves?

**Answer:** most of the loss is the network's **capacity**, not its training. The same binary network with twice the
channels gains **+10.8 points** (80.81 %) and learns its training images far better; no training recipe on the small
network comes close. A better optimizer for the 1-bit weights, **Bop**, gives **+2.5 points** on the small network
(+0.7 with the int8 stem, nothing at twice the channels): the less capacity, the more it helps. Bop is now an option
of the CDT (`[training] binary_optimizer = "bop"`).

Three networks of one graph, the benchmark's recipe for all (Adam lr 5e-3 + cosine, 150 epochs, batch 128, crop +
flip, labels only, the same starting weights, the CDT's 45,000 / 5,000 split). Validation accuracy (the test set is
never read), 3 seeds (42, 43, 44), mean ± standard deviation:

| Network | Validation | Training images |
|---|---|---|
| relu: real weights, ReLU, added shortcuts | 85.42 ± 0.31 % | 95.18 % |
| twin: real weights, hardtanh (what the estimator pretends to train) | 83.59 ± 0.86 % | 91.92 % |
| **binary** (the benchmark's network) | 70.00 ± 1.12 % | 72.19 % |

Recipes for the binary network (same epochs, seeds and split; each a change of the training only):

| Recipe | Validation | Training images |
|---|---|---|
| base (above) | 70.00 ± 1.12 % | 72.19 % |
| clip: latent weights clamped to [-1, 1] (BinaryConnect) | 71.67 ± 0.68 % | 73.65 % |
| twostep: binary activations first, then binary weights (Liu et al. 2021) | 72.35 ± 0.35 % | 73.47 % |
| twostep + Bop | 71.03 ± 0.20 % | 73.29 % |
| **Bop** for the 1-bit weights (Helwegen et al. 2019, γ 1e-4, τ 1e-8) | **72.49 ± 0.51 %** | 74.47 % |
| base, 300 epochs | 70.79 ± 0.70 % | 72.99 % |
| Bop, 300 epochs | 72.68 ± 0.73 % | 75.24 % |
| base + distillation from the relu network | 70.62 ± 0.99 % | 71.87 % |
| Bop + distillation from the relu network | 72.81 ± 0.51 % | 73.37 % |
| twostep + distillation from the relu network | 70.41 ± 0.36 % | 71.95 % |

Capacity and the int8 stem (`--width 2`, `--stem int8`; the same recipe, 3 seeds):

| Network | base | Bop | Training images (base) |
|---|---|---|---|
| binary, thermometer stem | 70.00 ± 1.12 % | 72.49 ± 0.51 % | 72.19 % |
| binary, int8 stem | 75.39 ± 0.20 % | **76.08 ± 0.24 %** | 76.81 % |
| binary, twice the channels | **80.81 ± 0.45 %** | 80.52 ± 0.67 % | 83.17 % |
| relu, int8 stem | 87.77 ± 0.56 % | | 96.47 % |

What the per-epoch records show (`results/train_gap/*.json`, `history`):

- **The twin network learns what the binary one cannot:** 91.9 % on its training images against 72-75 % for every
  binary recipe, also with twice the epochs. Training longer or with a teacher does not lift that ceiling.
- **Signs oscillate:** with Adam a weight changes sign ~7.8 times per epoch for each time it really moves. Bop does not
  change that ratio (7.7) and still wins, so its gain is not the end of the oscillation.
- **Half of the activations get no gradient:** 46 % of the binary network's activation inputs lie where the estimator
  passes none (|x| > 1), against 21 % in the twin network.

Bop closes ~2.5 of the 13.6 points between the twin and the binary network (~18 %): the training part found so far, a
lower bound. Twice the channels closes ~10.8 of them, with about four times the weights (an estimate from the channel
count, not measured in an export). Bop in the CDT stores the weights as ±0.5 (inside every estimator's window); its tests are in
`tests/test_bop.py`.

```
python train_gap_study.py --model all --seeds 42 43 44
python train_gap_study.py --model binary --recipe clip bop twostep twostep_bop --seeds 42 43 44
python train_gap_study.py --model binary --recipe base bop --epochs 300 --seeds 42 43 44
python train_gap_study.py --model binary --recipe base bop twostep --distill --seeds 42 43 44
python train_gap_study.py --model binary --width 2 --recipe base bop --seeds 42 43 44
python train_gap_study.py --model binary --stem int8 --recipe base bop --seeds 42 43 44
python train_gap_study.py --model relu --stem int8 --seeds 42 43 44
```

Measured in November 2026 (NVIDIA RTX 4000 Ada, ~4 minutes per run).

### Layer by layer (`layer_probe_study.py`)

The trained networks above, measured on the validation images, 3 seeds. A linear probe (a logistic regression on
each layer's output, average-pooled to 4×4) shows where each network adds class information:

| Network | stem | block 1 | block 2 (stride 2) | block 3 | block 4 (stride 2) | block 5 | network |
|---|---|---|---|---|---|---|---|
| binary | 52.8 | 57.7 | 56.8 | 60.1 | 57.1 | 60.2 | 70.0 |
| binary, int8 stem | 58.6 | 62.7 | 61.3 | 63.9 | 62.4 | 66.5 | 75.4 |
| binary, twice the channels | 53.1 | 59.8 | 58.3 | 64.0 | 70.4 | 74.6 | 80.8 |
| twin | 45.6 | 60.1 | 63.7 | 67.6 | 71.3 | 79.3 | 83.6 |
| relu | 46.9 | 60.5 | 64.3 | 67.3 | 72.3 | 81.7 | 85.4 |

- **The binary stem keeps the image well** (52.8 % against 46.9 % for relu): the thermometer + 1-bit stem loses
  nothing a linear readout can see; the int8 stem adds ~6 points there and keeps them to the end.
- **The small binary network stops adding information after block 1** (+2.5 points over four blocks, and it drops
  at both stride-2 blocks), where twin and relu add ~21. Twice the channels recovers it in the last stage (blocks
  4-5: +16): the capacity the binary network lacks is in its last stage, which is what the sizing study's
  16-32-128 shape (`../conv2d/cnn_cifar_resnet8`) gives it.
- **Counting noise is not what width fixes.** 25-43 % of the block outputs (60 % at the stem) lie within one noise unit
  (√N on the ±1 count, for random bits) of their threshold, and flipping 1 % of the stem's output bits changes
  ~13.5 % of the last block's bits; both are the same with twice the channels, which is still 10.8 points better.
- **No sign of more spurious detections than in floating point:** with each filter's weights shuffled (same values,
  19 shuffles), 5-33 % of the binary channels are more class-selective than all their shuffles, against 2-21 % in
  twin and relu.

```
python layer_probe_study.py --nets binary binary_i8 binary_w2 twin relu relu_i8 --seeds 42 43 44
```

### The same on 1D convolutions and an MLP (`small_gap_study.py`)

The binary / twin / relu networks with the shapes of `../conv1d/har_ultracompact` (UniMiB-SHAR, Daghero's `max`
and `min` networks, subject split) and `../mlp/mlp_fashion` (784-256-128-10, the CDT's split), one recipe for the
three (Adam 5e-3, cosine, 100 / 50 epochs, labels only, last epoch), 3 seeds, validation:

| Network | binary | twin | relu | binary train | binary probe, layer by layer | 1 % flipped → predictions changed |
|---|---|---|---|---|---|---|
| HAR `max` (32 ch) | 67.2 ± 1.3 | 69.2 ± 0.7 | 72.8 ± 0.5 | 87.2 | 57.9 → 66.4 → 66.7 | 17.9 % |
| HAR `min` (4 ch) | 38.3 ± 6.7 | 56.8 ± 2.0 | 61.3 ± 1.5 | 38.8 | 34.7 → 46.4 → 44.0 | 17.6 % |
| MLP 256-128 | 88.5 ± 0.2 | 89.4 ± 0.1 | 90.0 ± 0.2 | 96.9 | 87.3 → 88.0 | 3.2 % |
| ResNet-8 x1 (above) | 70.0 | 83.6 | 85.4 | | 52.8 → ... → 60.2 | 17.8 % |

- **The same answer: capacity, not training.** Where the binary network has room (HAR `max`, the MLP) it is within
  1-2 points of twin and fits its training data (87-97 %); where it is narrow (HAR `min`, ResNet-8 x1) it loses
  13-18 points and does not even fit its training data (38.8 % train, 38.3 % validation).
- **The narrow binary networks stall the same way:** HAR `min` loses information in its last layer (46.4 → 44.0),
  as ResNet-8 x1 does at its stride-2 blocks; the wide ones keep adding it.
- **The margins are the same everywhere:** 19-54 % of the binary outputs lie within one noise unit of their
  threshold (25-43 % in ResNet-8), in the networks that lose 1 point and in the ones that lose 18.
- **Noise grows with depth and convolution, not with being narrow:** 1 % of flipped first-layer bits becomes 8 %
  of the last layer's bits in HAR `max` (13.7 % in ResNet-8) but 3.7 % in the MLP, and changes 3 % of the MLP's
  predictions against ~18 % in both convolutional networks.
- **No more spurious detections than floating point:** 30-34 % (HAR `max`) and 79 % (the MLP's second layer) of the
  binary neurons beat all 19 shuffles of their weights, against 3-32 % and 54-56 % in twin and relu.

```
python small_gap_study.py --bench har --arch max --seeds 42 43 44
python small_gap_study.py --bench har --arch min --seeds 42 43 44
python small_gap_study.py --bench mlp --hidden 256 128 --seeds 42 43 44
```

## Next

A kernel for two-plane layers in `lib_symbex` that reads every weight word once for both planes, measured on the
ESP32, the ESP32-S3 and the Arduino Uno.
