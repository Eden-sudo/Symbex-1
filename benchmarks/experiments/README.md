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

### More than two bits per element (`--bits`)

Is two the right number? The same MLP (784-256-128-10, Fashion-MNIST, 20 epochs, 3 seeds) with 4 and 8 independent
bits per weight and per activation (`--pair-on both`, learned scales). A layer costs k × k binary products, so the
comparison is against a flat network of the same weight memory and one of about the same binary operations (searched
by width):

| Bits per element | Accuracy | Memory | Binary ops | vs flat, same memory | vs flat, same ops |
|---|---|---|---|---|---|
| 1 (flat) | 86.27 ± 0.21 % | 28.7 KB | 1x | | |
| 2 (pairs) | 88.63 ± 0.24 % | 57.3 KB | 4x | +1.91 | +1.47 |
| **4** | **89.13 ± 0.08 %** | 114.6 KB | 16x | +1.97 | +1.36 |
| 8 | 88.90 ± 0.11 % | 229.2 KB | 64x | +1.48 | +1.00 (the search stopped at width 4096: 11.6 M ops against 15.0 M) |

- **The gain saturates fast:** +2.4 points from 1 to 2 bits, +0.5 from 2 to 4, and 8 bits is no better than 4.
- **The cost does not:** 16x and 64x the binary operations. At 8 bits this is INT8 arithmetic done one bit plane at a
  time, slower than a multiply-accumulate on any of our chips: two bits is where this idea pays.
- These runs report the test images (the script's quick-experiment protocol, no validation split): a ranking, not a
  result. The 2-bit row is `pair_bits.py` with its defaults (`results/pair_bits_mlp_fashion.json`).

```
python pair_bits.py --bits 4
python pair_bits.py --bits 8
```

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

Results land in `results/`. Measured in October 2026; the 4- and 8-bit runs (`--bits`, above) in November 2026.

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
seeds: 75.6 % against 71.2 % on validation, `resnet8_symbex.py --variant`). It is now an option of the CDT's `boolean` method and of lib_symbex
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
lower bound. Twice the channels closes ~10.8 of them, with 3.26x the weights (counted in the trained network, see the
width table below). Bop in the CDT stores the weights as ±0.5 (inside every estimator's window); its tests are in
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

Measured in November 2026 (NVIDIA RTX 4000 Ada, 4-19 minutes per run).

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
  16-32-128 shape (`../conv2d/cnn_cifar_resnet8`) gives it (the width study below finds the middle stage worth even
  more).
- **Counting noise is not what width fixes.** 25-43 % of the block outputs (60 % at the stem) lie within one noise unit
  (√N on the ±1 count, for random bits) of their threshold, and flipping 1 % of the stem's output bits changes
  ~13.5 % of the last block's bits; both are the same with twice the channels, which is still 10.8 points better.
- **No sign of more spurious detections than in floating point:** with each filter's weights shuffled (same values,
  19 shuffles), 5-33 % of the binary channels are more class-selective than all their shuffles, against 2-21 % in
  twin and relu.

```
python layer_probe_study.py --nets binary binary_i8 binary_w2 twin relu relu_i8 --seeds 42 43 44
```

### Width or 2-bit activations, and where the width goes (`train_gap_study.py --last-stage`, `--channels`)

The probes point at the binary network's last stage. Two ways to spend the same extra binary operations there (~1.23x
the network's, counted layer by layer), same recipe, 3 seeds, validation:

| Network | validation | train | binary ops | weights |
|---|---|---|---|---|
| binary x1 (16-32-64) | 70.00 ± 1.12 | 72.19 | 1.00x | 1.00x |
| last stage 2-bit activations (`act2`) | 72.27 ± 0.22 | 74.58 | 1.24x | 1.00x |
| last stage 96 channels (`wide`, 16-32-96) | 73.67 ± 0.27 | 75.53 | 1.23x | 1.64x |
| 16-48-96 | 75.37 ± 0.22 | 77.33 | 1.53x | 1.88x |
| 16-32-128 | 74.60 ± 0.50 | 76.91 | 1.54x | 2.43x |
| 24-48-64 | 75.07 ± 0.73 | 77.30 | 1.83x | 1.30x |
| 24-48-96 | 76.81 ± 1.22 | 79.75 | 2.08x | 1.97x |
| twice the channels everywhere (x2) | 80.81 ± 0.45 | 83.17 | 3.54x | 3.26x |

- **At the same operations, width wins** (+3.7 points against +2.3), as pairs of bits did in the weights; the 2-bit
  activations win only if the weights' memory is the limit (no extra weights). `act2` is `pair_bits.py`'s pair of
  activation bits with equal scales (two nested thresholds, a sum of counts on a chip) inside the benchmark's network
  and recipe; `pair_bits.py`'s own ResNet (20 epochs) gave the same order: flat at the same operations 51.6-52.9,
  pairs in the weights 52.4, in the activations 50.8 (equal scales) / 51.7 (learned scales).
- **The last stage is not all of it:** both give a third of what x2 gives; the other stages need width too. At the
  same operations the middle stage is worth more than the last (16-48-96 75.37 against 16-32-128 74.60, with fewer
  weights), and widening the early stages costs operations but few weights (24-48-64: 75.07 with 1.30x the weights).
- These are the same 2-2-2 blocks; the sizing study's 1-1-2 shapes (`../conv2d/cnn_cifar_resnet8`) trade depth too (next sections).

```
python train_gap_study.py --model binary --last-stage act2 --seeds 42 43 44
python train_gap_study.py --model binary --last-stage wide --seeds 42 43 44
python train_gap_study.py --model binary --channels 16-48-96 --seeds 42 43 44     # 16-32-128, 24-48-64, 24-48-96
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

### Depth, and the stride-2 shortcut (`train_gap_study.py --channels a-b-c/x-y-z`, `--shortcut2 avg`)

Widening helps, but every point costs time on the chip. The next question was whether the same neurons give more
accuracy arranged differently: fewer or more blocks per stage, and a different shortcut where the resolution halves.
Same recipe, 3 seeds, validation. The times are **predicted** from the binary operations of each layer with the cost
curves of `tools/dev/count_bench.py`, calibrated on the measured x1 network (the method of
`../conv2d/cnn_cifar_resnet8/resnet8_sizing.py`; it was within 3 % of the S3 on the sizing study's shapes):

| Network (channels / blocks per stage) | validation | train | weights | S3 ms (predicted) | ESP32 ms (predicted) |
|---|---|---|---|---|---|
| x1, 16-32-64 / 2-2-2 | 70.00 ± 1.12 | 72.19 | 1.00x | 39.2 (measured) | 63.3 (measured) |
| x1 + average shortcut in the stride-2 blocks | 71.84 ± 0.99 | 73.47 | 0.98x | 34.6 | 55.7 |
| 16-48-96 / 1-1-2 | 72.92 ± 0.29 | 75.29 | 1.69x | 37.3 | 63.7 |
| 16-48-96 / 1-2-2 | 74.70 ± 0.47 | 76.97 | 1.86x | 46.7 | 79.7 |
| 16-64-128 / 1-1-2 | 74.78 ± 0.23 | 77.41 | 2.71x | 48.0 | 85.5 |
| 16-48-96 / 2-2-2 | 75.37 ± 0.22 | 77.33 | 1.88x | 52.8 | 89.8 |
| 16-48-96 + average shortcut | 74.65 ± 0.49 | 76.60 | 1.84x | 46.0 | 78.5 |
| **16-48-96 / 1-3-3** | **77.33 ± 0.32** | 79.93 | 2.72x | 63.1 | 109.9 |

- **Depth where the maps are small pays:** one block less in the first stage and one more in the other two
  (1-3-3) is the most accurate shape below x2's cost (x2: 80.81 % with 3.54x the operations; 1-3-3: 2.05x), level
  with 24-48-96 (76.81 ± 1.22, the same operations) and steadier across seeds. One block less
  in the first stage alone (1-2-2) costs 0.7 points and saves ~12 % of the time.
- **The average shortcut** (Bi-Real: `sign(BN(conv(x)) + s · avgpool2(x))`, no 1×1 convolution) gives +1.8 on x1
  and is faster, but loses 0.7 on 16-48-96. It needs a few integer thresholds per channel that `lib_symbex` does not
  have yet, so it stays a PC result.

The winner on accuracy, measured on the boards with the benchmark's own pipeline (`resnet8_symbex.py --spec
16-48-96/1-3-3`, 3 seeds, test images, 1,000 images on each chip, **chip = integer simulation on 100 %** in all 12
runs, `../conv2d/cnn_cifar_resnet8/results/symbex_seeds_*_c16-48-96_b1-3-3.json`):

| ResNet-8 | test | S3 flash / RAM | ESP32 flash / RAM | weights + thresholds |
|---|---|---|---|---|
| MLPerf Tiny INT8 (TFLite Micro) | 87.0 % | 52.0 ms | 717.8 ms | 96.2 KB |
| SYMBEX x1 | 71.3 ± 0.8 % | 39.2 ms (flash) | 63.3 / 59.7 ms | 24.3 KB |
| SYMBEX x1, int8 stem | 75.41 ± 0.02 % | 35.5 ms (flash) | 68.1 / 63.5 ms | 24.0 KB |
| **SYMBEX 16-48-96 / 1-3-3** | **76.82 ± 0.45 %** | 61.3 / 57.8 ms | 110.5 / 98.2 ms | 56.2 KB |

The prediction held (63.1 → 61.3 ms on the S3, 109.9 → 110.5 ms on the ESP32). On the ESP32 the deeper network is
still 6.5x faster than INT8 for +5.5 points over x1; on the S3 it is slower than INT8, and the int8 stem on x1 is the
better point there.

### Distillation, three losses (`--distill --distill-loss kl|mse|cosine`)

The literature on binary networks (Real-to-Bin, ReActNet) suggests that Hinton's KL loss carries too little signal
through binary layers, and that matching the raw logits works better. Same teacher (the relu network of each seed),
same recipe, 3 seeds:

| Loss | validation | train |
|---|---|---|
| none (labels only) | 70.00 ± 1.12 | 72.19 |
| KL at T = 4 (0.9 · T² · KL + 0.1 · cross-entropy) | 70.62 ± 0.99 | 71.87 |
| cosine of the logit vectors + cross-entropy | 70.75 ± 0.86 | 72.47 |
| MSE of the raw logits + cross-entropy | 67.49 ± 0.87 | 68.41 |

No loss helps beyond the seed noise; MSE hurts (it asks the binary network to match the teacher's logit scale too).
The binary network does not lack a better target, it lacks room: the same answer as the capacity study.

### Dead channels in the stride-2 shortcuts (`--density`, `--dead-fix`)

Is the capacity the network has really used? `--density` measures the share of ones of every channel of every binary
map of the trained networks (2,000 validation images, before the OR with the shortcut). In forward order: stem (L0),
then per block its 3×3 convolution, and in the two stride-2 blocks also their 1×1 shortcut convolution (L4, L7).
Channels with < 10 % or > 90 % ones (nearly constant), 3 seeds:

| Map | channels | mean ones | nearly constant |
|---|---|---|---|
| L0 stem | 16 | 22-24 % | 1-3 |
| L1-L2 blocks 1-2 | 16 | 13-17 % | 5-10 |
| L3 block 3, 3×3 (stride 2) | 32 | 17-19 % | 6-10 |
| **L4 block 3, 1×1 shortcut** | 32 | 8-10 % | **19-22** |
| L5 block 4 | 32 | 18-21 % | 8-13 |
| L6 block 5, 3×3 (stride 2) | 64 | 15-16 % | 19-28 |
| **L7 block 5, 1×1 shortcut** | 64 | 5-6 % | **53 in every seed** |
| L8 block 6 | 64 | 18 % | 3-6 |

Low densities are expected with OR shortcuts (dense maps would saturate the OR to all ones), but the 1×1 shortcuts
are nearly off: the network has turned off the path that carries the previous stage across the stride. Three fixes
that would not change the network on a chip, 3 seeds:

| `--dead-fix` | what it changes in training | validation | L7 nearly constant |
|---|---|---|---|
| none | | 70.00 ± 1.12 | 53 / 53 / 53 |
| `or_grad` | the OR passes the whole gradient to both branches (`torch.maximum` gives it to the larger one only) | **58.65 ± 2.12** | 3 / 8 / 2 |
| `wstd` | weight standardization per output channel before sign (IR-Net) | 66.31 ± 0.88 | 49 / 48 / 48 |
| `dead_loss` | a penalty on channels with < ~10 % or > ~90 % ones (soft density: mean of sigmoid) | 70.75 ± 1.15 | 54 / 52 / 52 |

- **Reviving the channels does not help, it hurts:** `or_grad` brings them back (L7 2-8 nearly constant, every map
  at 32-58 % ones) and loses 11 points: with dense maps on both sides, the OR saturates and the block output carries
  less. The 1×1 shortcut is off because it is not useful, not because it was starved.
- **What the average shortcut gets right:** it replaces that 1×1 convolution with the input itself, averaged over the
  stride, which is why it is the only change in this section that helps (+1.8 on x1).
- `wstd` hurts; `dead_loss` did not revive the channels (the penalty was too weak), so it says nothing either way.

```
python train_gap_study.py --model binary --channels 16-48-96/1-3-3 --seeds 42 43 44     # 1-1-2, 1-2-2; 16-64-128/1-1-2
python train_gap_study.py --model binary --shortcut2 avg --seeds 42 43 44               # and --channels 16-48-96
python train_gap_study.py --model binary --distill --distill-loss mse --seeds 42 43 44  # cosine
python train_gap_study.py --model binary --dead-fix or_grad --seeds 42 43 44            # wstd, dead_loss
python train_gap_study.py --model binary --density --seeds 42 43 44                     # + the same --dead-fix
cd ../conv2d/cnn_cifar_resnet8 && python resnet8_symbex.py --spec 16-48-96/1-3-3 --seeds 3
python ../../../tools/chip_test.py runs/c16-48-96_b1-3-3_s42 --board s3 --port /dev/ttyACM0 --usb --popcount swar
```

### Where accuracy stands, and how we got here

The path of this study, each step answering the question the previous one raised:

1. **Is the loss in the training?** binary vs twin vs relu on one graph: the twin network learns 92 % of its training
   images, the binary one 72 %, with every recipe. Mostly not: Bop (+2.5) and twostep (+2.35) are the training part
   found, and they do not add up (71.03 together).
2. **Then it is capacity: where?** The layer probes: the small binary network stops adding information after its
   first block and loses it at both stride-2 blocks; twice the channels recovers it in the last stage.
3. **Is it the same in other networks?** conv1d (HAR) and the MLP: where the binary network has room it is within
   1-2 points of twin; where it is narrow it loses 13-18.
4. **How to spend the extra operations?** Width beats 2-bit activations at the same operations; the middle stage is
   worth more than the last; more blocks in the small-map stages match more width, steadier (1-3-3: 77.3 %). Two bits per
   element is the sweet spot of `pair_bits.py`; 4 and 8 cost far more for little.
5. **Is there a better teacher signal?** Three distillation losses: no.
6. **Is the capacity used?** The 1×1 shortcuts are nearly off; forcing them on costs 11 points. The one shortcut
   change that helps (average) replaces them.

What is left, as measured: against the INT8 MLPerf model (87.0 %) the best binary ResNet-8 is ~10 points behind
(76.8 % with 1-3-3, 75.4 % with the int8 stem at x1's cost). Binary networks win where the arithmetic is the limit
(ResNet-8 against TFLite Micro INT8: 6.5-12x faster on the classic ESP32, 2.8-6.1x on the Pico 2's M33; x1 uses a
quarter of INT8's flash, 1-3-3 ~58 %), and pay in accuracy on hard tasks.
The options that help and cost nothing on the chip are all kept: Bop (`[training] binary_optimizer = "bop"`), the
int8 stem (`--variant int8stem`), and the shapes (`--spec`).

Open, if you want to help:

- **The average shortcut in `lib_symbex`** (a few integer thresholds per channel in the stride-2 blocks): +1.8 on x1,
  and faster. The one architecture change with evidence behind it.
- **The int8 stem with `--spec`** (1-3-3 + int8 stem): both gains measured apart, not together.
- **Bop with the benchmark's own pipeline and test images:** measured here on validation only.
- **A stronger `dead_loss`, or another way to use the shortcut path** across the stride.
- Ideas we have not tried and would like to see measured: grouped binary convolutions, channels passed through
  without a convolution (BinaryDenseNet style), a multi-bit classifier.

## Next

A kernel for two-plane layers in `lib_symbex` that reads every weight word once for both planes, measured on the
ESP32, the ESP32-S3 and the Arduino Uno.
