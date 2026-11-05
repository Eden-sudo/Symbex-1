# pair_bits -- pairs of bits in binary networks

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

From this folder (`benchmarks/experiments/pair_bits/`), in the main environment (`.venv`):

```
python pair_bits.py --pair-on activations --act-scale equal
python pair_bits.py --pair-on activations --act-scale equal --widths 0.5,1,1.5,2
python pair_bits.py --arch resnet8 --pair-on activations --act-scale equal
python pair_bits.py --arch resnet8 --pair-on activations --act-scale equal --widths 0.5,1,1.5,2
python pair_bits.py --arch resnet8 --pair-on activations --act-scale equal --input thermometer
python pair_bits.py --arch resnet8 --pair-on weights
```

Results land in `results/`. Measured in October 2026; the 4- and 8-bit runs (`--bits`, above) in November 2026.

## Next

A kernel for two-plane layers in `lib_symbex` that reads every weight word once for both planes, measured on the
ESP32, the ESP32-S3 and the Arduino Uno.
