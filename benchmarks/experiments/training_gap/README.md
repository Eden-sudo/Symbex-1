# training_gap -- how much does training cost the binary ResNet-8? (`train_gap_study.py`)

**Question:** the fully binary ResNet-8 of `../../conv2d/cnn_cifar_resnet8` loses ~15 points against a floating-point
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

## Layer by layer (`layer_probe_study.py`)

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
  16-32-128 shape (`../../conv2d/cnn_cifar_resnet8`) gives it (the width study below finds the middle stage worth even
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

## Width or 2-bit activations, and where the width goes (`train_gap_study.py --last-stage`, `--channels`)

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
- These are the same 2-2-2 blocks; the sizing study's 1-1-2 shapes (`../../conv2d/cnn_cifar_resnet8`) trade depth too (next sections).

```
python train_gap_study.py --model binary --last-stage act2 --seeds 42 43 44
python train_gap_study.py --model binary --last-stage wide --seeds 42 43 44
python train_gap_study.py --model binary --channels 16-48-96 --seeds 42 43 44     # 16-32-128, 24-48-64, 24-48-96
```

## The same on 1D convolutions and an MLP (`small_gap_study.py`)

The binary / twin / relu networks with the shapes of `../../conv1d/har_ultracompact` (UniMiB-SHAR, Daghero's `max`
and `min` networks, subject split) and `../../mlp/mlp_fashion` (784-256-128-10, the CDT's split), one recipe for the
three (Adam 5e-3, cosine, 100 / 50 epochs, labels only, last epoch), 3 seeds, validation:

| Network | binary | twin | relu | binary train | binary probe, layer by layer | 1 % flipped → predictions changed |
|---|---|---|---|---|---|---|
| HAR `max` (32 ch) | 67.5 ± 1.3 | 69.1 ± 1.0 | 72.8 ± 0.3 | 88.2 | 59.2 → 64.6 → 67.1 | 18.3 % |
| HAR `min` (4 ch) | 40.8 ± 5.2 | 57.7 ± 3.4 | 58.2 ± 2.2 | 39.8 | 39.0 → 46.6 → 45.9 | 22.3 % |
| MLP 256-128 | 88.6 ± 0.1 | 89.6 ± 0.2 | 89.9 ± 0.1 | 97.1 | 87.6 → 88.1 | 3.2 % |
| ResNet-8 x1 (above) | 70.0 | 83.6 | 85.4 | | 52.8 → ... → 60.2 | 17.8 % |

- **The same answer: capacity, not training.** Where the binary network has room (HAR `max`, the MLP) it is within
  1-2 points of twin and fits its training data (87-97 %); where it is narrow (HAR `min`, ResNet-8 x1) it loses
  13-17 points and does not even fit its training data (39.8 % train, 40.8 % validation).
- **The narrow binary networks stall the same way:** HAR `min` loses information in its last layer (46.6 → 45.9),
  as ResNet-8 x1 does at its stride-2 blocks; the wide ones keep adding it.
- **The margins are the same everywhere:** 33-47 % of the binary outputs lie within one noise unit of their
  threshold (25-43 % in ResNet-8), in the networks that lose 1 point and in the ones that lose 18.
- **Noise grows with depth and convolution, not with being narrow:** 1 % of flipped first-layer bits becomes 8 %
  of the last layer's bits in HAR `max` (13.7 % in ResNet-8) but 3.6 % in the MLP, and changes 3 % of the MLP's
  predictions against ~18 % in both convolutional networks.
- **No more spurious detections than floating point:** 22-33 % (HAR `max`) and 81 % (the MLP's second layer) of the
  binary neurons beat all 19 shuffles of their weights, against 8-45 % and 54-58 % in twin and relu.

```
python small_gap_study.py --bench har --arch max --seeds 42 43 44
python small_gap_study.py --bench har --arch min --seeds 42 43 44
python small_gap_study.py --bench mlp --hidden 256 128 --seeds 42 43 44
```

## Depth, and the stride-2 shortcut (`train_gap_study.py --channels a-b-c/x-y-z`, `--shortcut2 avg`)

Widening helps, but every point costs time on the chip. The next question was whether the same neurons give more
accuracy arranged differently: fewer or more blocks per stage, and a different shortcut where the resolution halves.
Same recipe, 3 seeds, validation. The times are **predicted** from the binary operations of each layer with the cost
curves of `tools/dev/count_bench.py`, calibrated on the measured x1 network (the method of
`../../conv2d/cnn_cifar_resnet8/resnet8_sizing.py`; it was within 3 % of the S3 on the sizing study's shapes):

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
runs, `../../conv2d/cnn_cifar_resnet8/results/symbex_seeds_*_c16-48-96_b1-3-3.json`):

| ResNet-8 | test | S3 flash / RAM | ESP32 flash / RAM | weights + thresholds |
|---|---|---|---|---|
| MLPerf Tiny INT8 (TFLite Micro) | 87.0 % | 52.0 ms | 717.8 ms | 96.2 KB |
| SYMBEX x1 | 71.3 ± 0.8 % | 39.2 ms (flash) | 63.3 / 59.7 ms | 24.3 KB |
| SYMBEX x1, int8 stem | 75.41 ± 0.02 % | 35.5 ms (flash) | 68.1 / 63.5 ms | 24.0 KB |
| **SYMBEX 16-48-96 / 1-3-3** | **76.82 ± 0.45 %** | 61.3 / 57.8 ms | 110.5 / 98.2 ms | 56.2 KB |

The prediction held (63.1 → 61.3 ms on the S3, 109.9 → 110.5 ms on the ESP32). On the ESP32 the deeper network is
still 6.5x faster than INT8 for +5.5 points over x1; on the S3 it is slower than INT8, and the int8 stem on x1 is the
better point there.

## Distillation, three losses (`--distill --distill-loss kl|mse|cosine`)

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

## Dead channels in the stride-2 shortcuts (`--density`, `--dead-fix`)

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
cd ../../conv2d/cnn_cifar_resnet8 && python resnet8_symbex.py --spec 16-48-96/1-3-3 --seeds 3
python ../../../tools/chip_test.py runs/c16-48-96_b1-3-3_s42 --board s3 --port /dev/ttyACM0 --usb --popcount swar
```

## The shortcut itself: XOR, option A, and what makes the average work (`--shortcut-op`, `--shortcut2 pad|addpool`)

The dead 1×1 shortcuts raised two questions. Is the OR the problem (it can only switch bits on, so the maps must stay
sparse: an OR of two maps with 50 % ones has 75 %)? And does the stride-2 block need its 1×1 at all? Same recipe, 3
seeds, validation (XOR: 1 seed, a ranking):

| Shortcut | what it is | x1 | 16-48-96 / 1-3-3 | S3 / ESP32 ms, x1 (cost model; measured below) |
|---|---|---|---|---|
| OR + 1×1 (the benchmark's) | `sign(BN(conv)) OR sign(BN(1x1(orpool(x))))` | 70.00 ± 1.12 | **77.33 ± 0.32** | 39.2 / 63.3 (measured) |
| XOR in every block (`--shortcut-op xor`) | the conv can switch a shortcut bit off too | 62.48 (1 seed) | | 39.2 / 63.3 |
| XOR + pad | | 58.92 (1 seed) | | 34.6 / 55.7 |
| **pad** (ResNet's option A) | no 1×1: the OR-pooled input for the first C_in channels, bit 0 for the new ones | 69.74 ± 0.26 | 75.36 ± 0.19 | **34.6 / 55.7** |
| avg (Bi-Real) | `sign(BN(conv) + s · avg2x2(x))`, channels repeated | **71.84 ± 0.99** | 74.65 ± 0.49 (16-48-96 / 2-2-2) | 34.6 / 55.7 |
| addpool | avg with the OR-pool (one bit) instead of the 2×2 count | 71.47 ± 0.62 | | 34.6 / 55.7 |

- **XOR loses 7.5 points:** letting the conv switch bits off does not help; the sparse maps of the OR network are
  its choice, not a trap. XNOR is XOR with the bits inverted, which the network can learn, so it is the same.
  An AND shortcut is the OR one with every bit inverted (De Morgan), so not tried either.
- **The 1×1 is worth nothing in x1 and 2 points in 1-3-3.** In x1 the channels double at each stride, and the input
  covers half the outputs. In 1-3-3 the first stride goes 16 → 48: the input covers a third, and the 1×1 fills the
  rest. pad is a speed option for x1: about the same accuracy, ~8 % faster on the S3 (an estimate from the measured
  blocks below), and it needs only an OR with the pooled input on a chip, no new thresholds.
- **avg differs from the benchmark's shortcut in four ways:** it passes a count (0-4, ~2.3 bits) instead of one OR-ed
  bit, it adds it to the threshold instead of OR-ing it, it has no 1×1, and it repeats channels. pad shows that the
  1×1 alone is not it. addpool keeps the addition and drops the count, and keeps
  nearly all of the gain (71.47 against 71.84, inside the seed noise). **The addition is what helps:** the shortcut
  bit moves the conv's threshold instead of forcing a 1. On a chip addpool is two integer thresholds per channel
  (one per shortcut bit) in the two stride-2 blocks, no 1×1: simpler than avg (five) and the cheapest shortcut that
  helps.

`add` (`resnet8_symbex.py --variant add`: every block `sign(BN(conv) + shortcut)`, two thresholds per channel, so it
can learn OR, AND, copy or ignore) gave 69.39 against 71.17 for OR. So the addition helps where the resolution
halves, and the OR is better in the other blocks.

### The shortcuts on the chip (`shortcut_bench.py`)

The two stride-2 blocks of x1 (16 → 32 at 32×32 → 16×16, 32 → 64 at 16×16 → 8×8), random weights, lib_symbex's
conv, every way checked against plain C on the chip (all exact, flash and RAM), best of 7 runs. addpool is a copy of
`symbex_bool_conv` made by the script, whose threshold step picks one of two thresholds by the pooled shortcut bit
(nothing added to lib_symbex). Both blocks together, ms per image:

| Chip | weights | conv alone | OR + 1×1 (today) | pad | addpool | addpool saves |
|---|---|---|---|---|---|---|
| ESP32-S3 | flash | 5.60 | 8.82 | **5.66** | 6.11 | 2.71 |
| ESP32-S3 | RAM | 5.60 | 8.82 | **5.66** | 6.11 | 2.71 |
| ESP32 | flash | 8.50 | 11.77 | **8.66** | 9.03 | 2.75 |
| ESP32 | RAM | 8.15 | 11.27 | **8.22** | 8.68 | 2.59 |

- **The 1×1 shortcut costs ~3.2 ms per image** on both chips; pad removes nearly all of it, addpool all but 0.5 ms
  (the threshold choice per output).
- **The cost model overestimated the saving:** it predicted 4.6 ms (S3) and 7.6 ms (ESP32) for the whole network;
  the blocks measure ~2.7 ms on both. For the whole network that is an estimate of 39.2 → ~36.5 ms on the S3 (−7 %)
  and 63.3 → ~60.6 ms on the ESP32 (−4 %), not measured.
- **addpool:** +1.5 points (71.47 ± 0.62 against 70.00 ± 1.12) and ~7 % faster on the S3: a real but small gain.

```
python shortcut_bench.py --port /dev/ttyACM0                  # ESP32-S3 (results/shortcut_bench_s3.json)
python shortcut_bench.py --board esp32 --port /dev/ttyUSB0    # classic ESP32
```

## Bop with twostep, again (`--recipe twostep_bop --bop-tau`, `--phase2-lr`)

Bop and twostep each give ~+2.4 but 71.03 together. The guess: step 2 starts from the good signs of step 1, and Bop's
τ = 1e-8 flips them too easily. A larger τ flips only the weights with a strong, steady gradient. One seed (42) per τ,
a ranking:

| τ | 1e-8 | 1e-7 | 1e-6 | 1e-5 | 1e-4 |
|---|---|---|---|---|---|
| validation | **71.22** | 70.48 | 70.98 | 66.84 | 53.94 |

A larger τ only hurts, so Bop needs its flips. Step 2 with Adam at a small learning rate (`--phase2-lr 1e-4`, 3 seeds)
gives 69.06 ± 0.46, against 72.35 ± 0.35 for twostep with the full rate. Twostep alone stays the best recipe.

```
python train_gap_study.py --model binary --shortcut-op xor --seeds 42                 # + --shortcut2 pad
python train_gap_study.py --model binary --shortcut2 pad --seeds 42 43 44             # + --channels 16-48-96/1-3-3
python train_gap_study.py --model binary --shortcut2 addpool --seeds 42 43 44
python train_gap_study.py --model binary --recipe twostep_bop --bop-tau 1e-6 --seeds 42   # 1e-7, 1e-5, 1e-4
python train_gap_study.py --model binary --recipe twostep --phase2-lr 1e-4 --seeds 42 43 44
```

## The rest of the block: padding, BatchNorm, the classifier (`--pad`, `--head`, `--density`)

Four more parts of the network, x1 binary, 3 seeds, validation (base 70.00 ± 1.12):

| Part | Variant | validation | Result |
|---|---|---|---|
| 3×3 convolutions | `--density`: channels never on / always on (x1 and 1-3-3, 2,000 images) | | 1 of 2,832 never on, 0 always on |
| BatchNorm | `--density`: channels with gamma < 0 | | 0 of 2,832 |
| Padding | `--pad minus` (border = −1) | 70.90 ± 0.99 | inside the noise |
| Padding | `--pad plus` (border = +1) | 71.09 ± 0.76 | inside the noise |
| Classifier | `--head int8` (int8 weights, binary input bits) | **72.84 ± 0.13** | **+2.8**, and steadier |
| Classifier | `--head pool` (ones per channel, then int8) | 67.09 ± 1.36 | −2.9 |

- **The "dead" channels are sparse, not dead:** every nearly constant channel is nearly *off* (< 10 % ones), none
  nearly on, and only one is never on. With the OR an off channel passes its shortcut through, so there is nothing
  to prune for free.
- **No gamma < 0 anywhere**, and the exporter already turns one into a negated weight row at no cost: nothing to
  gain in the BatchNorm.
- **Block order (BN before or after the convolution):** on ±1 inputs a BatchNorm before a binary convolution folds
  into it; the real alternatives are how the shortcut is added, already measured (`add` 69.39, addpool +1.5).
- **Padding does not matter:** the chip's zero padding (border pixels left out of the count) is as good as −1 or +1.
- **The int8 classifier is the one that helps, but it is not free:** it reads the whole last map (64 × 8 × 8 = 4,096
  bits → 10 classes), so its weights grow from 5 KB to 40 KB, against 24 KB of flash for the whole x1 network today.
  Pooling first (64 counts) is tiny but loses 2.9 points: the classifier needs where in the image a feature is.

### The int8 classifier on the chip (`int8_head_bench.py`), and on the MLP and conv1d

These times are **the classifier alone** (the last layer, which votes the class), not the whole network: an int8
classifier makes the network more accurate but slower, never faster. The whole ResNet-8 x1 runs in 39.2 ms (S3) /
59.7 ms (ESP32) with the standard engine and 6.44 ms with the vertical engine (the record, unchanged), of which the
binary classifier is 0.06 / 0.11 ms. The whole network with an int8 classifier was not measured.

Four ways to run the classifier, random weights at the real shapes, exact against plain C on the chip, µs (best of 7):
`binary` is lib_symbex's `symbex_bool_classify`; the int8 ways add the weights of every bit (`dense`), only of the
bits that are 1 (`sparse`, weights stored per bit; the final map of the trained x1 networks has 31 % ones), or run 8
AND + popcount passes over the weights' bit planes (`planes`).

| ResNet-8 classifier only (4,096 bits → 10) | binary | i8_dense | i8_sparse | i8_planes | extra, best int8 |
|---|---|---|---|---|---|
| ESP32-S3, flash | 60 | 2,858 | **1,063** | 2,814 | +1.0 ms (~2.6 % of 39.2 ms, estimated) |
| ESP32-S3, RAM | 60 | 1,602 | 488 | **301** | +0.24 ms (~0.6 %) |
| ESP32, flash | 109 | 4,724 | **2,095** | 3,832 | +2.0 ms (~3.3 % of 59.7 ms) |
| ESP32, RAM | 109 | 1,613 | **552** | 656 | +0.44 ms (~0.7 %) |

The percentages are estimates (classifier time over the standard-engine network). Over the vertical engine's 10.36 ms
(default) the same extra would be ~10 % in flash and ~2.3 % in RAM, also estimated.

The small heads (128 bits → 17 or 10 classes) cost 22-48 µs in int8 against 4-10 µs binary, on both chips.
The same `--head int8` in `small_gap_study.py` (binary networks, 3 seeds, validation):

| Network | binary head | int8 head | Δ | head weights |
|---|---|---|---|---|
| MLP 256-128 | 88.64 ± 0.13 | 88.79 ± 0.11 | +0.2 | 160 B → 1.3 KB |
| HAR `max` | 67.48 ± 1.34 | 66.63 ± 1.36 | −0.9 | 272 B → 2.2 KB |
| HAR `max`, int8 first layer | 67.17 ± 2.51 | 68.90 ± 0.29 | +1.7 | |
| HAR `min` | 40.82 ± 5.18 | 40.93 ± 2.92 | +0.1 | 60 B → 476 B |
| HAR `min`, int8 first layer | 35.79 ± 5.92 | 43.32 ± 3.73 | +7.5 (noisy) | |

- **The int8 classifier pays where the classifier is large:** ResNet-8 reads 4,096 bits and gains 2.8 points for
  ~1 ms more on the S3 (weights in flash) and 35 KB more (5 → 40 KB; the whole x1 network is 24 KB). The MLP's last hidden layer already holds its accuracy (probe 88.1 %),
  so an int8 head adds nothing there; HAR gains only with the int8 first layer too, steadier rather than higher.

```
python train_gap_study.py --model binary --pad minus --seeds 42 43 44                 # plus
python train_gap_study.py --model binary --head int8 --seeds 42 43 44                 # pool
python train_gap_study.py --model binary --density --seeds 42 43 44                   # + --channels 16-48-96/1-3-3
python small_gap_study.py --bench har --arch max --first int8 --head int8 --kinds binary --seeds 42 43 44
python int8_head_bench.py --port /dev/serial/by-id/usb-Espressif_USB_JTAG_...   # --board esp32 --port /dev/ttyUSB0
```

## The pooled int8 classifier (`--head pool2|pool4`), now in `lib_symbex`

The int8 classifier gains +2.8 but weighs 40 KB, because it reads every bit of the last map. Pooling the whole map
first (`--head pool`) loses 2.9 points. In between: count the ones of every channel per 2 × 2 (or 4 × 4) window of the
8 × 8 map, then an int8 layer on those counts. `pool2` reads 64 channels × 16 windows = 1,024 counts →
10 classes: **10 KB of int8 weights** instead of 40 KB. x1 network, 3 seeds, validation:

| Network | binary head | int8 head | pool2 | pool4 |
|---|---|---|---|---|
| thermometer stem | 70.00 ± 1.12 | 72.84 ± 0.13 | 72.81 ± 0.59 | 71.19 ± 0.92 |
| thermometer stem + Bop | 72.49 ± 0.51 | | 73.72 ± 1.04 | |
| int8 stem | 75.39 ± 0.20 | | | |
| int8 stem + Bop | 76.08 ± 0.24 | 77.66 ± 0.63 | **77.48 ± 0.39** | |
| int8 stem + addpool + Bop | 75.71 ± 0.68 | 77.21 ± 0.45 | 76.89 ± 0.16 | 75.93 ± 0.46 |

- **pool2 keeps the int8 classifier's gain at a quarter of its weights:** 72.81 vs 72.84 on x1, 77.48 vs 77.66 with
  the int8 stem and Bop. pool4 (4 windows) already loses 1-1.6 points: the classifier still needs some of *where*.
- **int8 stem + Bop + pool2 is the most accurate x1 network so far:** 77.48 %, +1.9 over the int8 stem record of the
  benchmark (75.63, without Bop; the two recipes differ, so this is not the gain of the head alone), for 29.6 KB of flash.
- addpool does not add up with the int8 stem (75.71 vs 76.08 with Bop).
- **BatchNorm recalibration** (`--recalibrate-bn`: the BN statistics measured again on training images after
  training, `bnrecal_*.json`) moves validation by −0.2 to +0.5 points, inside the noise: not used.

The chip runs it as `symbex_bool_pool_head` (lib_symbex, `SymbexBoolNet.pool_head`): the counts of each window are
kept bit-sliced (a few count planes), and the dot product with the int8 weights is AND + popcount of every count plane
against each of the weights' 8 bit planes, exact in int64. The exporter folds the BatchNorm and the per-class scale
into an int32 scale and bias, as for the binary classifier. `--export` writes the header and checks it in C.

**Whole network on the chip**, int8 stem + Bop + pool2 against the int8-stem record (same flags, standard engine,
3 seeds × 1,000 images, every image equal to the integer simulation; `results/train_gap/pool_head_chip.json`), ms:

| Board | weights | int8 stem record | + pool2 | extra | classifier alone |
|---|---|---|---|---|---|
| ESP32-S3 (swar) | flash | 35.54 | 35.77 | +0.23 (+0.7 %) | 0.06 → 0.28 |
| ESP32-S3 (swar) | RAM | 35.54 | 35.76 | +0.23 (+0.6 %) | 0.06 → 0.28 |
| ESP32 (lut16) | flash | 68.07 | 69.31 | +1.24 (+1.8 %) | 0.35 → 1.00 |
| ESP32 (lut16) | RAM | 63.52 | 63.98 | +0.46 (+0.7 %) | 0.11 → 0.56 |
| Pico 2 M33 | flash | 72.21 | 73.57 | +1.36 (+1.9 %) | 0.57 → 2.25 |
| Pico 2 M33 | RAM | 68.54 | 69.14 | +0.60 (+0.9 %) | 0.14 → 0.76 |
| Pico 2 Hazard3 | flash | 64.34 | 65.52 | +1.18 (+1.8 %) | 0.53 → 1.93 |
| Pico 2 Hazard3 | RAM | 59.45 | 60.16 | +0.71 (+1.2 %) | 0.09 → 0.55 |

Flash: 24.6 → 29.6 KB. The standard deviations are ≤ 0.004 ms.

**On the vertical engine (ESP32-S3)**, the int8 stem and the pooled classifier run there too (exact on every image,
3 seeds, weights in flash; `results/train_gap/pool_head_vertical_s3.json`), ms (RAM):

| Network | validation | default | compact | prepared |
|---|---|---|---|---|
| record: thermometer, binary classifier | 71.17 ± 0.32 | 10.36 (63 KB) | 7.70 (151 KB) | 6.44 (238 KB) |
| thermometer + Bop + pool2 | 73.72 ± 1.04 | 10.67 ± 0.04 (62 KB) | 7.92 ± 0.04 (151 KB) | 6.74 ± 0.02 (239 KB) |
| int8 stem + Bop + pool2 | 77.48 ± 0.39 | 13.37 ± 0.04 (52 KB) | 10.84 ± 0.04 (136 KB) | 9.78 ± 0.02 (215 KB) |

- **The pooled classifier costs ~0.3 ms** on the vertical engine (0.35 ms against 0.10).
- **The int8 stem is now the slow part:** 4.5 ms against 1.9 ms for the thermometer + binary stem. In the standard
  engine it was faster (4.2 vs 8.1 ms), but the vertical engine made the thermometer path 4× faster. Against ESP-DL
  (14.8 ms flash, 9.8 ms RAM with 147 KB) the int8-stem network is 1.11× faster in flash and only ties in RAM.
- **Thermometer + Bop + pool2 keeps the speed** (1.39× ESP-DL in flash, 1.45× in RAM) and gains 2.5 points.

**The int8 stem on PIE (Nov 6).** *Question:* the int8 stem is the slow part on the S3; can PIE take it away?
*What we did:* two S3 HAL kernels (`../chip_kernels`): 16 output channels per instruction for the standard engine,
16 pixels per instruction written straight into the channel planes for the vertical one. *Result*, measured, 3
seeds × 1,000 images, every image equal to the simulation (`results/train_gap/stem_pie_s3.json`), ms (RAM):

| Engine | before | int8 stem on PIE |
|---|---|---|
| standard, weights in flash / RAM | 35.77 / 35.76 | **32.33 / 32.32** (15 / 44 KB) |
| vertical default, flash | 13.37 | **9.73 ± 0.04** (54 KB) |
| vertical default, weights in RAM | — | 9.72 ± 0.04 (83 KB) |
| vertical default, flash, code in RAM | — | 9.43 ± 0.04 (65 KB) |
| vertical compact, flash (code in RAM) | 10.84 | **7.09 ± 0.04** (138 KB); 6.96 (149 KB) |
| vertical prepared, flash (code in RAM) | 9.78 | **6.03 ± 0.02** (217 KB); 5.91 (228 KB) |

*Answer:* the most accurate x1 network (77.48 %) is now also the fastest on the S3: 9.73 ms in flash with 54 KB of
RAM, faster than the earlier record (10.36, thermometer, 71.2 %) and 1.52× faster than ESP-DL in flash (14.8 ms);
the compact mode (7.09 ms, 138 KB) is 1.38× faster than ESP-DL in RAM (9.8 ms, 147 KB). The other boards keep
their times (seed 42, exact: ESP32 and both Pico 2 cores the same or up to 1.4 % faster).

**The test set, once (Nov 6).** The network chosen on validation (int8 stem + Bop + pool2), its 3 seeds on the
10,000 CIFAR-10 test images, the integer simulation that the chip runs (the compiled C equal on every image, standard
and the three vertical modes; `--test`, `results/train_gap/test_binary_i8_hpool2_bop_g0.0001_t1e-08.json`):
**77.26 ± 0.09 %** (validation 77.48 ± 0.39; the int8-stem record of the benchmark: 75.41 ± 0.02 on test). The
recipe is now in the CDT (Nov 6): `resnet8_symbex.py --variant int8stem_pool2` in `../../conv2d/cnn_cifar_resnet8`,
measured there on every board.

```
python train_gap_study.py --model binary --stem int8 --recipe bop --head pool2 --seeds 42 43 44
python train_gap_study.py --model binary --stem int8 --recipe bop --head pool2 --seeds 42 43 44 --export
python train_gap_study.py --model binary --stem int8 --recipe bop --head pool2 --seeds 42 43 44 --test   # once
python ../../../tools/chip_test.py runs/train_gap/export/binary_i8_hpool2_bop_g0.0001_t1e-08_s42 --board s3 --usb --popcount swar
```

## Where accuracy stands, and how we got here

The path of this study, each step answering the question the previous one raised:

1. **Is the loss in the training?** binary vs twin vs relu on one graph: the twin network learns 92 % of its training
   images, the binary one 72 %, with every recipe. Mostly not: Bop (+2.5) and twostep (+2.35) are the training part
   found, and they do not add up (71.03 together).
2. **Then it is capacity: where?** The layer probes: the small binary network stops adding information after its
   first block and loses it at both stride-2 blocks; twice the channels recovers it in the last stage.
3. **Is it the same in other networks?** conv1d (HAR) and the MLP: where the binary network has room it is within
   1-2 points of twin; where it is narrow it loses 13-17.
4. **How to spend the extra operations?** Width beats 2-bit activations at the same operations; the middle stage is
   worth more than the last; more blocks in the small-map stages match more width, steadier (1-3-3: 77.3 %). Two bits per
   element is the sweet spot of `../pair_bits`; 4 and 8 cost far more for little.
5. **Is there a better teacher signal?** Three distillation losses: no.
6. **Is the capacity used?** The 1×1 shortcuts are nearly off; forcing them on costs 11 points. The one shortcut
   change that helps (average) replaces them.
7. **Is the OR the problem, and is the 1×1 needed?** XOR is worse (−7.5). Without the 1×1 (option A), x1 keeps its
   accuracy and gets faster, but 1-3-3 loses 2 points. Adding the OR-pooled bit to the threshold (addpool) gives
   +1.5, nearly all of avg's gain. Bop and twostep still do not combine, at any τ.
8. **The rest of the block?** Padding, block order and BatchNorm give nothing; an int8 classifier gives +2.8 on
   ResNet-8 (~1 ms, 35 KB more), nothing on the MLP or HAR, whose classifiers are small.

What is left, as measured: against the INT8 MLPerf model (87.0 %) the best binary ResNet-8 is ~10 points behind
(76.8 % with 1-3-3, 75.4 % with the int8 stem at x1's cost). Binary networks win where the arithmetic is the limit
(ResNet-8 against TFLite Micro INT8: 6.5-12x faster on the classic ESP32, 2.8-6.1x on the Pico 2's M33; x1 uses a
quarter of INT8's flash, 1-3-3 ~58 %), and pay in accuracy on hard tasks.
The options that help and cost nothing on the chip are all kept: Bop (`[training] binary_optimizer = "bop"`), the
int8 stem (`--variant int8stem`), and the shapes (`--spec`).

Open, if you want to help:

- **The addpool shortcut in `lib_symbex`** (two integer thresholds per channel in the stride-2 blocks, no 1×1):
  +1.5 on x1 and ~7 % faster on the S3 (estimated from the measured blocks). The one architecture change with
  evidence behind it; avg (five thresholds) adds
  +0.4 more, inside the noise.
- **The pooled int8 classifier with 1-3-3** (int8 stem + Bop + pool2 is measured on test: 77.26 %).
  The pooled classifier itself is now in `lib_symbex` (above).
- **The int8 stem with `--spec`** (1-3-3 + int8 stem): both gains measured apart, not together.
- **Bop with the benchmark's own pipeline and test images:** measured here on validation only.
- **A stronger `dead_loss`, or another way to use the shortcut path** across the stride.
- **Option A (`--shortcut2 pad`) in `lib_symbex` for x1:** about the same accuracy without the 1×1, ~8 % faster on
  the S3 (estimated from the measured blocks).
- Ideas we have not tried and would like to see measured: grouped binary convolutions, channels passed through
  without a convolution (BinaryDenseNet style).
