# tiny_llm -- a small language model with 1-bit weights, for the microcontroller

## Question

Does a small language model still write stories when its weights are 1-bit (or ternary), and can it run as integers
only, exact in C, small enough for an ESP32-S3 with 8 MB of PSRAM?

## What

- **Data:** TinyStories (GPT-4 version, Eldan & Li 2023), short English stories. `prepare.py` trains a 4096-token
  SentencePiece vocabulary and writes the token files: train 542 M tokens, validation 5.5 M (every 100th training
  story), test 5.5 M (the official valid file, not read yet).
- **Model:** a Llama-style transformer (RMSNorm, rotary positions, SwiGLU), 8 layers, width 384, 6 heads, 1024 hidden,
  256-token context: 15.74 M parameters, 14.16 M of them in the linear layers that get quantized.
- **Training (`train.py`):** same recipe for the three weight formats: 10,000 steps of 128 x 256 tokens (328 M
  tokens), AdamW, learning rate 1.5e-3 with warm-up and cosine decay, seeds 42, 43, 44, the checkpoint with the lowest
  validation loss. Quantized runs: weights `ternary` ({-1, 0, +1} x one scale per row, BitNet b1.58) or `binary`
  ({-1, +1} x one scale per row), 8-bit inputs to every linear layer, straight-through estimator.
- **Integer model (`intsim.py`):** no floating point after the export. Residual stream int32 (Q12), RMSNorm through an
  integer square root, int8 inputs, 1-bit weights with one integer multiplier per row, rotary tables in Q14, attention
  with an int8 KV cache and a 2048-entry exp table, SwiGLU with a sigmoid table, int8 embedding shared with the output
  layer. The output of every step is an int64 score per token; its argmax is the next token.
- **C (`c/tinylm.c`, plain C99):** the same model one token at a time with a KV cache. `c/check_pc.c` compares every
  score of every token of the vocabulary (4096 per step) and the final max with `intsim.py`, on 4 validation windows
  of 256 tokens, then the 120-token greedy story.

```
python prepare.py
python train.py --weights binary --seeds 42 43 44          # also fp32, ternary
python intsim.py runs/binary_d384l8_s42 --export          # integer loss + model.bin / check.bin for the C
(cd c && gcc -O2 -Wall -Wextra -o check_pc check_pc.c tinylm.c && ./check_pc ../runs/binary_d384l8_s42)
python chip_run.py runs/binary_d384l8_s42 --port /dev/ttyACM0                   # ESP32-S3 N16R8, native USB port
python chip_run.py runs/binary_d384l8_s42 --port /dev/ttyACM0 --linear symbex   # linear layers on lib_symbex
```

- **Chip (`chip/`, `chip_run.py`):** ESP32-S3 N16R8 at 240 MHz. `model.bin` goes to a flash partition and is copied
  into PSRAM at start; the KV cache is in PSRAM, the per-token buffers in internal RAM. The firmware checks every
  token of 2 validation windows (a hash of the 4096 scores and the final max, against `intsim.py`) and the greedy
  story token by token, and times each part of a step. Kernels for the 1-bit linear layers: `lut8` (tinylm.c: a table
  of the 256 sums of every 8 inputs, one lookup per weight byte) or `--linear symbex` (`c/tinylm_symbex.h`: lib_symbex's
  8-bit x 1-bit kernel, the one its benchmarks use for their first layer, `symbex_u8_sum_bits_rows`, PIE on the S3;
  the int8 inputs go in as x + 128 and the row's count of 1 bits takes the 128 back out, so the integers are the same).
  The output layer and the attention scores use lib_symbex's PIE int8 row products (`symbex_s8_dot_rows`).

**What is binary here, and what is not.** This is a 1-bit-*weight* model (W1A8, like BitNet), not a fully binary
network like the other SYMBEX benchmarks:

| Part | Format | Share of the parameters |
|---|---|---|
| weights of the linear layers (attention q, k, v, o and SwiGLU) | **1 bit** (+ one integer multiplier per row) | 14.16 M of 15.74 M (90 %) |
| inputs (activations) of those layers | int8 | |
| embedding / output layer (shared) | int8 | 1.57 M (10 %) |
| attention: KV cache, scores, softmax | int8 cache, integer scores, exp table | |
| norms, SwiGLU, residual stream | integer (int32, tables) | 0.01 M |

So the linear layers count with additions and subtractions of int8 values, not with XNOR + popcount: with 8-bit
inputs that is the path the other benchmarks use only for their first layer.

## Result

Validation loss (cross-entropy per token, 5,120 windows of 256 tokens, mean ± sd over seeds 42, 43, 44;
[`results/train_*.json`](results/)), about 33 minutes per run on an RTX 4000 Ada laptop GPU:

| Weights | Validation loss | Perplexity | Linear-layer weights |
|---|---|---|---|
| fp32 | 1.300 ± 0.002 | 3.67 | 56.6 MB |
| ternary | 1.425 ± 0.003 | 4.16 | 3.5 MB at 2 bits |
| binary | 1.522 ± 0.003 | 4.58 | 1.77 MB |

Integer model of the binary networks (1,024 validation windows, [`results/intsim_*.json`](results/)):

| | Float model | Integer model |
|---|---|---|
| validation loss | 1.5306 ± 0.0013 | 1.5307 ± 0.0013 |
| next token equal to the float model's | | 98.57 ± 0.04 % |

C on the PC ([`results/c_check_pc_*.txt`](results/)): **1,024 / 1,024 tokens with every score equal, and the same
greedy story, on all three seeds.** `model.bin` is 3.55 MB (1.77 MB of 1-bit weights, 1.57 MB of int8 embedding,
tables and multipliers); the KV cache for 256 tokens is 1.5 MB.

On the ESP32-S3 ([`results/chip_s3_binary_d384l8_s4*.json`](results/)): **512 / 512 tokens with every score equal
and the story 120 / 120 tokens equal: exact**, with both kernels on seed 42 and with lib_symbex's kernel on seeds 43
and 44. With lib_symbex's kernel the story runs at **8.48 ± 0.00 tokens per second** (118 ms per token, 3 seeds). A step costs more as the context fills (attention over more
tokens). Model load from flash 0.5-0.6 s; ~3 MB of PSRAM and ~246 KB of internal RAM stay free.

| Part of a step (ms, mean of 512 tokens, seed 42) | `lut8` | `--linear symbex` |
|---|---|---|
| 1-bit linear layers (q, k, v, o, w1, w3, w2) | 115.1 | **55.6** |
| attention (rope + int8 cache 9.7, scores 7.8, softmax 2.0, values 13.2, division 1.6) | 43.8 | **34.3** |
| output layer (4096 x 384 int8) | 28.0 | 28.0 |
| norms, embedding, SwiGLU | 10.5 | 10.5 |
| **whole step** (position 0 / 255) | 197.3 (163 / 231) | **128.5** (109 / 149) |
| **story, tokens per second** | 5.55 | **8.48** |

The attention keeps the integers of `intsim.py` and only changes how they are computed: the KV cache is stored
head by head (a head's keys one after another), and the softmax index needs a 32-bit division only. The values are
row products on the PIE: each head's values are stored transposed (one row per dimension, as value + 128 in a byte),
the weights of the values (p x vm < 2^32, checked; a larger one falls back to plain C in int64) are split into at most
4 bytes, one unsigned 8-bit row product per byte (`ee.vmulas.u8.accx`), and the sum takes 128 x the weights back out.
Values 25.2 -> 13.2 ms; writing the transposed cache costs 2.2 ms more (rope + cache 7.5 -> 9.7). The `lut8` column
was measured before these attention changes.

**Which layers are sensitive, and does the CDT's scan see it?** (`layer_sensitivity.py`,
[`results/layer_sensitivity.json`](results/layer_sensitivity.json), fp32 models, 3 seeds.) The CDT's sensitivity scan
(`tools/compiler_core/sensitivity.py`) gives one cosine per layer: the layer's output with fewer bits against its fp32
output, one layer at a time. Here it runs as it is ("bin": 1-bit weights and 1-bit input) and with tiny_llm's own
format (1-bit weights, 8-bit input), and both are compared with the real damage: binarize only that layer (W1A8) in
the fp32 model and measure the validation loss of the whole model (512 windows, no retraining).

| Layer type (8 blocks) | CDT cosine, bin | cosine, W1A8 | loss damage, one layer | loss damage, all 8 |
|---|---|---|---|---|
| q | 0.613 ± 0.006 | 0.887 ± 0.010 | 0.020 ± 0.002 | 0.32 ± 0.05 |
| k | 0.444 ± 0.006 | 0.888 ± 0.015 | 0.028 ± 0.004 | 0.59 ± 0.15 |
| v | 0.674 ± 0.001 | 0.807 ± 0.006 | 0.022 ± 0.002 | 0.29 ± 0.05 |
| o | 0.669 ± 0.005 | 0.873 ± 0.002 | 0.022 ± 0.001 | 0.28 ± 0.01 |
| w1 | 0.642 ± 0.007 | 0.871 ± 0.000 | 0.029 ± 0.001 | 0.34 ± 0.01 |
| w3 | 0.686 ± 0.003 | 0.847 ± 0.001 | 0.025 ± 0.000 | 0.28 ± 0.01 |
| w2 | 0.408 ± 0.002 | 0.833 ± 0.004 | 0.035 ± 0.000 | 0.47 ± 0.01 |
| output layer (head) | 0.378 ± 0.012 | 0.912 ± 0.005 | **0.513 ± 0.012** | |

- The real damage has a clear shape: the output layer alone costs 15× a block layer, block 0 costs 1.24 ± 0.18
  (blocks 1-7: 0.13-0.24), and the last block is the next worst. The top 10 layers by damage are the same on the 3
  seeds: the head, block 0's w1 / w2 / w3, block 7's w1 / w2 / w3, then single k layers.
- The cosines do not see it. Rank correlation with the real damage: CDT bin 0.24 ± 0.04, W1A8 0.06 ± 0.06. They
  share 2-3 of the top 10 (bin) and 1 (W1A8). The W1A8 cosine even ranks the head as the safest layer (highest
  cosine) while it is the most damaging one.
- Damages do not add up: all the blocks binarized together cost 4.63 ± 0.24, the sum of the single layers 1.44;
  block 0 alone 1.24 against 0.29 summed.
- This is damage before retraining, so it was checked with training: the same recipe as the binary base, 3 seeds
  ([`results/train_binary_d384l8_*_s*.json`](results/), `train.py --head binary` / `--int8-blocks`):

| Variant | Validation loss | Paired change vs binary | Linear-layer weights |
|---|---|---|---|
| binary (base) | 1.5224 ± 0.0026 | | 1.77 MB |
| 1-bit output layer | 1.6071 ± 0.0030 | +0.085 ± 0.005 | 1.77 MB (+ head 1.57 -> 0.20 MB) |
| blocks 0 and 7 with int8 weights (the most damaged) | 1.4205 ± 0.0027 | -0.102 ± 0.004 | 4.87 MB |
| blocks 3 and 4 with int8 weights (control, same bytes) | 1.4272 ± 0.0017 | -0.095 ± 0.003 | 4.87 MB |
| ternary, every block | 1.425 ± 0.003 | | 3.54 MB |

- The 1-bit output layer stays costly after training (+0.085, a third of the binary-fp32 gap): not used.
- Blocks 0 and 7 beat the middle blocks 3 and 4 on every seed, but only by 0.007 ± 0.005: 93 % of the gain comes
  from the extra bytes, not from picking the sensitive blocks. And per byte, ternary everywhere does as well with
  fewer bytes. So the damage ranking was right about the head and right in sign for the blocks, but too weak to
  matter there. The scan is kept as it is (it measures one layer in isolation, as its own notes say).

### A 4-bit or 2-bit output layer

The output layer (4096 x 384, shared with the embedding) is the slowest part of a step on the S3 (28 ms): it is
bound by the PSRAM, so fewer bytes make it faster. `train.py --head int4` (levels -7..7 x max|w| / 7 per row) and
`--head int2` (levels {-1.5, -0.5, 0.5, 1.5} x mean|w| per row), with the same recipe and seeds; the embedding lookup
uses the same quantized table, so the chip keeps a single table. Validation loss over 3 seeds (paired change vs. the
float output layer of the same seed), then intsim, C and the ESP32-S3 for each seed
([`results/train_binary_d384l8_head*_s*.json`](results/), [`results/chip_s3_binary_d384l8_head*_s*_symbex.json`](results/)):

| Output layer | Validation loss | Paired change | Output layer on the S3 | Tokens per second | `model.bin` |
|---|---|---|---|---|---|
| int8 (after training) | 1.522 ± 0.003 | | 28.0 ms | 8.48 ± 0.00 | 3.55 MB |
| **int4** | 1.5376 ± 0.0031 | +0.015 ± 0.003 | **14.7 ms** | **9.52 ± 0.00** | 2.77 MB |
| int2 | 1.5709 ± 0.0044 | +0.049 ± 0.003 | **8.4 ms** | **10.13 ± 0.00** | 2.37 MB |
| 1 bit | 1.6071 ± 0.0030 | +0.085 ± 0.005 | not taken to the chip | | |

- Integer model equal to the float one (int4 1.546 / 1.546 on seed 42, int2 1.583 / 1.583; 1,024 windows) and the C
  **exact on every seed, on the PC and on the S3** (512 / 512 tokens, story 120 / 120).
- The S3 kernels read the packed table straight from PSRAM: int4 (two weights per byte) is split with one mask and one
  shift (`ee.andq`, `ee.vsl.32`) into two int8 vectors, int2 (four per byte) into four, each multiplied by the PIE
  (`ee.vmulas.s8.accx`); `model.bin` version 3 / 4 (`intsim.pack_nibbles` / `pack_crumbs`).
- **int4 is the choice:** half the time of the output layer for a sixth of the 1-bit layer's cost. int2 is 6 % faster
  and 0.39 MB smaller again, for three times int4's cost in loss: an option when speed or flash matter most.

### 1-bit activations (W1A1) on the chip

The 1-bit activations were made to train (`w1a1_diag.py`, `damage_map.py`, `hadamard_probe.py`; LOGBOOK.md): the
clipped straight-through estimator moves W1A1 from 3.82 to 2.04, w2's input at 8 bits gives 1.8096 ± 0.0054 (3 seeds),
and the best recipe found (a Hadamard transform before every 1-bit input, block 0 at 8 bits) 1.782 (seed 42, a
ranking), against W1A8 1.522. That model is exact in C and on the S3 (XOR + popcount on the PIE through lib_symbex):
**8.88 tokens per second against W1A8's 8.48**, the same flash (+896 bytes) and 8.9 KB more internal RAM, for +0.26
loss. In a model that runs one token at a time the activations are only ~1 KB, so binarizing them saves no memory,
and the time goes to the output layer and the attention, not the linear layers.

### Rules learned for language models

- Keep the activations at 8 bits (W1A8): 1-bit activations cost loss and give no speed or memory back here.
- The output layer matters most: at 1 bit it costs the most, and on the chip it is the slowest part. Quantize it to
  int4, not to 1 bit.
- The memory bandwidth (PSRAM), not the arithmetic, limits a step: fewer bytes in what is read every token (output
  layer, attention cache) is the lever.
- The per-layer cosine scan does not predict the trained loss here: check sensitive layers by training (3 seeds).

A story from the binary model (seed 42, greedy, from "Once upon a time"):

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and have fun. One day, Lily
> found a big, red ball in her room. She was very happy and wanted to show it to her friends. Lily went to her friend
> Tom and said, "Look at my big, red ball! Let's play with it!" Tom was excited and said, "Yes, let's play!" ...

## Answer

Yes: the 1-bit model writes stories with dialogue and a plot, the integer version loses nothing (1.5307 vs
1.5306), and the C equals it on every score, on the PC and on the ESP32-S3, where it writes 8.48 tokens per second
(3 seeds) with lib_symbex's own 8-bit x 1-bit kernel, and 9.52 with a 4-bit output layer (+0.015 loss, 3 seeds,
exact). It costs perplexity against fp32 (4.58 vs 3.67): the binary model
repeats itself more. Ternary sits between the two at twice the weight memory.

## Future work

- The chip, faster. `../chip_kernels/psram_bw_bench.py` ran the same kernels with the data in PSRAM and in internal
  RAM: the output head (28 ms) and the attention scores are at the PSRAM's speed (58 MB/s), the linear layers are
  half kernel and half memory (69 MB/s internal, 40 MB/s from PSRAM), and the attention values were plain C (25 ms,
  now 13.2 ms on the PIE, exact). The int4 / int2 output layer is done (above; 1 bit rejected). Next: the PSRAM at
  120 MHz (+40 % bandwidth, experimental in ESP-IDF), a cheaper transposed cache write.
- Ideas from [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) (an int4 TinyStories model on the same chip,
  9.88 tokens/s on both cores, float; reviewed Nov 12): a 4-bit or 2-bit output head (done, above: measured 14.7 /
  8.4 ms, 9.52 / 10.13 tokens/s); per-layer embeddings (a large lookup table in
  flash, a few rows read per token) gave them +0.098 nats at vocabulary 32768 but only +0.025 at 4096, ours, so a
  PC check at most; `noinline` on the linear kernel (inlining at -O3 slowed their step from 95 to 155 ms).

- The ternary integer model and C; a binary model with as many bytes as the ternary one.
- Binarizing more is research: 1-bit activations were measured (above: +0.26 loss with the best recipe, one seed,
  no gain on the chip); a binary embedding and binary attention are not measured: an open question.
- The test split, once, for the final numbers.
