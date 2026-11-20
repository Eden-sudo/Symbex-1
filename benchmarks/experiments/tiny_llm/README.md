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

## KV cache below 8 bits (rule Nov 18 2026, before measuring)

**Question:** the KV cache is int8 (1.5 MB for 256 tokens here; ~328 KB for a 260K-size model with 512 tokens,
computed). It grows with the context, unlike the weights. How much loss does it cost at 1, 2 and 4 bits?

**What:** `intsim.py --kv-bits 1|2|4` on the trained binary W1A8 models (no retraining), keys and values quantized per
token and head (1 bit: sign x mean|x|; 2 bits: train.py's int2 levels; 4 bits: -7..7 x max / 7), the rest of the
integer model unchanged. Validation, 1024 windows, 3 seeds; reference: the same runs at `--kv-bits 8`. PC only: no
speed measured here (the C keeps int8).

**Rule:** d = mean over seeds of (loss at b bits - loss at 8 bits). d < 0.003 (the base sd): free. d < 0.03 (a third
of the 1-bit output layer's +0.085): usable without training. Larger: it needs training with the KV quantized (a
follow-up, only if Edén wants it), not "the KV cannot be binary".

**Result (val loss of the integer model, 3 seeds, `results/kv_bits.json`):**

| KV cache | loss | d vs 8 bits | KV bytes, 256 tokens (computed) |
|---|---|---|---|
| 8 bits | 1.5307 ± 0.0013 | | 1.57 MB |
| 4 bits | 1.5405 ± 0.0011 | +0.0098 ± 0.0003 | 0.79 MB |
| 2 bits | 1.9163 ± 0.0205 | +0.386 ± 0.021 | 0.39 MB |
| 1 bit | 2.7450 ± 0.0390 | +1.214 ± 0.039 | 0.20 MB |

**Answer:** without retraining, 4 bits is usable (not free); 2 and 1 bit break the model (1 bit: the next token
agrees with the float model 49 % of the time). By the rule this does not say a binary KV is impossible, only that it
needs training with the KV quantized.

## KV cache: keys vs values, per channel, QJL, first token (rule Nov 18 2026, before measuring)

**Question:** the 1- and 2-bit KV above were per token for both sides. The literature (KIVI, AsymKV, QJL, BitNet
a4.8; LOGBOOK Nov 18) says keys hurt more than values, keys want per-channel scales, a 1-bit key can be
sign(S k) + ||k|| with the query unquantized (QJL), and the first token is worth more bits. Which of these lets a
1-bit (or 2-bit) KV work without retraining?

**What:** `intsim.py --k-bits / --v-bits / --k-mode token|channel|qjl / --qjl-m / --keep-first` (kv_store: the
non-token modes rebuild an integer vector and pass it through int8, one extra rounding; QJL uses float inside:
experiment only, no C). Same models, windows and seeds as above. Phase 1, fixed now:
(a) K1 V8, K8 V1, K2 V8, K8 V2 (per token); (b) K2 per channel (groups of 32 tokens) V8, K1 per channel V8,
K2 per channel V2; (c) K1 QJL with 64 / 128 / 256 bits per key, V8; (d) K1 V8 and K1 V1 with the first token int8.
Phase 2: the 1-bit K with the lowest d, with V at 1 and 2 bits, with and without the first token int8.

**Rule:** d as above (mean over seeds vs `--kv-bits 8`); same thresholds: < 0.003 free, < 0.03 usable without
training, larger: needs training with the KV quantized. A mode "helps" if its d is below the per-token d of the same
bits by more than 2x the larger d sd.

**Result (val loss, 3 seeds, `results/kv_modes.json`; bits per dimension; reference 8 bits 1.5307):**

| Keys | Values | d vs 8 bits | Rule |
|---|---|---|---|
| 1 bit per token | 8 | +0.698 ± 0.039 | needs training |
| 1 bit per channel | 8 | +0.657 ± 0.011 | does not help (0.041 < 0.078) |
| 1 bit QJL, 64 bits per key (1 per dim) | 8 | +0.409 ± 0.009 | **helps** (-0.289) |
| 2 bits per token | 8 | +0.200 ± 0.014 | needs training |
| 2 bits per channel | 8 | +0.233 ± 0.010 | worse |
| 1 bit QJL, 128 bits per key (2 per dim) | 8 | +0.164 ± 0.004 | helps, barely (-0.036 vs 0.028) |
| 4 bits per token (control, added after phase 1) | 8 | +0.008 ± 0.001 | usable |
| 1 bit QJL, 256 bits per key (4 per dim) | 8 | +0.069 ± 0.002 | worse than 4 bits per token |
| 8 | 1 bit | +0.193 ± 0.001 | needs training |
| 8 | 2 bits | +0.058 ± 0.001 | needs training |
| QJL 256 | 1 / 2 bits | +0.295 / +0.138 | needs training |
| first token kept int8 (K1 V8, K1 V1, QJL 256 V1 / V2) | | changes < 0.002 | does not help |

**Answer:** keys hurt about 3.5x more than values at 1 bit, as the literature says. Per-channel keys and an int8
first token do nothing here. At the same bytes, QJL is the only mode that helps (1 bit per dimension: +0.41 instead
of +0.70), but at 4 bits per dimension plain int4 wins. Without training, nothing below 4 bits is usable: a 1- or
2-bit KV needs training with the KV quantized.

## KV cache: changing the 1-bit key (rule Nov 18 2026, before measuring)

**Question:** QJL was the only mode that helped, but its Gaussian S needs float multiplies and was tried with one
S only. Does a +-1 S (signed Hadamard: S q with additions only), a rotation before plain 1- or 2-bit quantization, or
a residual (1 bit per token plus QJL of what it missed, as TurboQuant) beat the best key of the same bytes?

**What:** `intsim.py --k-mode hadqjl | rot | res --qjl-seed s` (kv_store; LOGBOOK Nov 18). All with V8, same models,
windows and seeds. Fixed now: (a) hadqjl 64 and 128 bits per key; (b) rot at 1 and 2 bits; (c) res with 64 bits
(1 bit per token + 64 = 2 bits per dimension); (d) Gaussian QJL 64 with S seeds 1 and 2 (is S seed 0 typical?).

**Rule:** d vs `--kv-bits 8` (1.5307). Compared by bits per dimension with the best mode so far: 1 bit -> QJL 64
+0.409 ± 0.009; 2 bits -> QJL 128 +0.164 ± 0.004. A change "helps" if its d is below that by more than 2x the larger
sd. (d): if seeds 1 / 2 move d by more than 2x the sd from seed 0's +0.409, the single-S result is fragile and the
comparison uses the mean of the 3 S. Same thresholds for "usable" (< 0.03) as above.

**Result (val loss, 3 seeds, V8, `results/kv_modes2.json`; reference 8 bits 1.5307):**

| Keys | Bits per dim | d vs 8 bits | Rule (vs the best of the same bits) |
|---|---|---|---|
| Gaussian QJL 64, S seed 0 / 1 / 2 | 1 | +0.409 / +0.407 / +0.435 (mean +0.417) | seed 2 moves d by 0.026 (= 2x its sd 0.013): fragile, use the mean |
| rot, 1 bit | 1 | +0.376 ± 0.002 | helps, barely (-0.041 vs 0.026) |
| **hadqjl 64** | 1 | **+0.198 ± 0.005** | **helps (-0.219)**; same as 2 bits per token (+0.200) |
| **rot, 2 bits** | 2 | **+0.076 ± 0.001** | **helps (-0.088 vs 0.008)** |
| hadqjl 128 | 2 | +0.084 ± 0.005 | helps (-0.080); ties rot 2 bits |
| res (1 bit per token + hadqjl 64) | 2 | +0.090 ± 0.003 | helps (-0.074) |

**Answer:** a +-1 Hadamard S is better than a Gaussian one (+0.20 instead of +0.42 at 1 bit per dimension) and
needs only additions. At 2 bits per dimension a rotation before plain 2-bit quantization halves the damage
(+0.076 instead of +0.164); the TurboQuant-style residual is no better than that. Nothing is usable yet (< 0.03):
training is still needed, but it now starts from a key at half the damage. Why hadqjl 64 beats rot 1 bit: both
store the same bits, sign(R k); only the scale differs, by pi / 2 (next section).

## KV cache: is the 1-bit key's damage a scale bias? (rule Nov 18 2026, before measuring)

**Question:** hadqjl 64 and rot 1 bit store the same bits, sign(R k). Their rebuilt keys differ only by a factor
pi / 2: rot uses the least-squares scale mean|y| (y = R k), QJL the unbiased one sqrt(pi / 2) ||k|| / sqrt(d), and
mean|y| ~ sqrt(2 / pi) ||k|| / sqrt(d) for Gaussian y. A least-squares scale shrinks every score by ~2 / pi, which
flattens the softmax. Is that most of the 1-bit key's damage, and does a fixed factor fix plain 1 bit per token too?

**What:** `intsim.py --k-scale f` multiplies the stored key scale (integer, one multiply per head and token).
Fixed now: K1 per token x pi / 2; rot 1 bit x pi / 2; V8, same models, windows and seeds.

**Rule:** rot 1 bit x pi / 2 should equal hadqjl 64 (+0.198) within 2x the larger sd if the explanation holds.
K1 per token x pi / 2 "helps" if its d is below +0.698 by more than 2x the larger sd, and is compared with hadqjl 64
(same bits; hadqjl needs S q per query, the plain one does not).

**Result (val loss, 3 seeds, V8, `results/kv_modes2.json`):**

| Keys (1 bit per dim) | d vs 8 bits | Rule |
|---|---|---|
| 1 bit per token (before) | +0.698 ± 0.039 | |
| **1 bit per token x pi / 2** | **+0.244 ± 0.008** | **helps (-0.454)**; hadqjl 64 still lower by 0.046 (> 2x 0.008) |
| rot 1 bit x pi / 2 | +0.203 ± 0.005 | = hadqjl 64 (+0.198, difference 0.005 < 0.010): the explanation holds |

**Answer:** yes. About two thirds of the 1-bit key's damage was a scale bias, not lost information: the shrunk
key flattens the softmax. One integer multiply on the key scale takes plain 1 bit per token from +0.70 to +0.24,
with no rotation and no new kernel. The rotation adds the rest (+0.20). Still not usable without training (> 0.03).
pi / 2 is the Gaussian value, not a tuned one.

## KV cache trained at 1-bit keys and 2-bit values (rule Nov 18 2026, before measuring)

**Question:** without training, the best plain KV below 4 bits is the 1-bit key x pi / 2 (no rotation, no new
kernel). Does training with the KV quantized (QAT) make 1-bit keys (x pi / 2) and 2-bit values usable? This KV is
0.59 MB instead of 1.67 MB for 256 tokens (computed), what lets the 15M fit a 4 MB PSRAM or the 30M an 8 MB one.

**What:** `train.py --k-bits 1 --v-bits 2 --k-scale 1.5708` (quant_kv: the same levels as intsim.py per token and
head, identity straight-through gradient; train / intsim levels checked equal except ~1e-4 of the values at a
rounding edge). Same recipe as the base binary model (10,000 steps, seeds 42-44, name d384l8_k1v2). Scored by
intsim.py with `--k-bits 1 --v-bits 2 --k-scale 1.5708` (validation, 1024 windows). Also measured: the base models
with the same KV and no training (the reference to beat), and the QAT models with an int8 KV (informative).

**Rule:** d = QAT model with its KV minus the base model with KV int8 (1.5307). QAT "helps" if d is below the
no-training d of the same KV by more than 2x the larger sd. Same thresholds: < 0.03 usable. Next step (C, exact,
S3) only if usable; between 0.03 and 0.10 Edén decides; above, the result is reported and the KV stays int8.

**Result (val loss of the integer model, 3 seeds, `results/kv_qat.json`; reference: base model, KV int8, 1.5307):**

| Model | KV | loss | d vs 8 bits | Rule |
|---|---|---|---|---|
| base binary (no KV training) | K1 x pi / 2, V2 | 1.8977 | +0.367 ± 0.015 | |
| **trained with that KV (QAT)** | K1 x pi / 2, V2 | 1.8514 | **+0.321 ± 0.022** | helps, barely (-0.046 vs 0.044); > 0.10: KV stays int8 |
| the same QAT model | int8 | 13.78 | +12.25 ± 3.53 | (informative) it only works with the KV it was trained with |

The float loss during training is the same (1.825-1.865 best val): the gap is what the model can learn with this
KV, not an integer / float mismatch. Training took 22.3-22.4 min per seed. Correction to the question above: K1 / V2
is 0.29 MB for 256 tokens (0.39 MB with the int32 scales), not 0.59 MB (that is K4 / V2).

**Answer:** no. Training with 1-bit keys recovers only 0.05 of the 0.37: the 1-bit key (64 bits per head and
token) loses information the model cannot work around in 10,000 steps, not a bias it can learn away. The KV stays
int8 for K1 / V2. What is usable today is the 4-bit KV without training (+0.0098, above): half the bytes.

## KV cache: 3 bits, mixed bits, a recent int8 window, tuned scales (rule Nov 18 2026, before measuring)

**Question:** the 4-bit KV is usable without training (+0.0098). Three ideas from the papers were not tried: 3 bits
and mixed K / V bits (keys hurt more), KIVI's recent window (the last N tokens of every query kept at full
precision, the rest at 2 bits), and a scale factor tuned beyond pi / 2 (2-bit levels at mean / 2 and 3 mean / 2 are
also shrunk for Gaussian data: ESTIMATE ~1.2x). Does any of them reach "usable" (< 0.03) with fewer bytes than KV4?

**What:** `intsim.py --k-bits / --v-bits 3` (codes -3..3 x max / 3, per token and head), `--kv-recent N` (per query
the N nearest keys and values, itself included, read int8, the others at the low bits; checked: N = 256 gives
exactly the int8 loss), `--v-scale f` (as `--k-scale`). No training, same models, 1024 validation windows, seeds
42-44. Fixed now:
(a) per token: K3 V8, K8 V3, K8 V4, K3 V3, K3 V4, K4 V3, K4 V2, K3 V2, K2 V4.
(b) recent window N = 16 / 32 / 64 with K2 V2, K1 x pi / 2 V2 and K3 V3.
(c) scale factor, chosen on seed 42 only: K1 {1.4, 1.5708, 1.75, 1.9}, K2 and K2 rot {1.1, 1.2, 1.3, 1.4} (V8), V2
{1.1, 1.2, 1.3, 1.4} (K8). The best factor of each then on 3 seeds; then the best K2 and V2 factors together, with and
without N = 32. Seed 42 chose the factor: its own number is slightly optimistic; seeds 43 / 44 are reported apart.

**Bytes** (computed, 15M model, 256 tokens, int32 scale per token and head included: 49 KB per side): int8 1.67 MB,
KV4 0.88 MB. A recent window of N tokens is counted as N int8 tokens + 256 - N low-bit ones (a ring buffer).

**Rule:** d = mean over seeds of (loss - loss at int8), reference 1.5307. Usable: d < 0.03 (as above). A config
**beats KV4** if it is usable and has fewer bytes than 0.88 MB. A scale factor "helps" if its 3-seed d is below the
same config at factor 1 (or pi / 2 for K1) by more than 2x the larger sd. Next step (C, exact, S3) for the usable
config with the fewest bytes; none -> KV4 goes to C.

### Why keys break: a diagnosis, then keys per channel before RoPE (rule Nov 18 2026, before measuring the loss)

**Question:** is the low-bit key damage the softmax "amplifying any error exponentially", or something we can fix?

**What:** `kv_diag.py` (measurement only: the int8 model's q, k, v on 32 validation windows, every layer and head; the
low-bit keys from `kv_store`; floats only to measure; results/kv_diag_binary_d384l8_s4*.json), 3 seeds.

**Result (diagnosis, 3 seeds, mean over layers; s42 / s43 / s44 agree within ~5 %):**

| keys | KL(p8 ‖ pq) | ½ Var δ | slope (temperature) | entropy int8 → quant |
|---|---|---|---|---|
| K4 per token | 0.027 | 0.027 | +0.01 | 2.60 → 2.59 |
| K3 per token | 0.143 | 0.142 | -0.02 | 2.60 → 2.56 |
| K3 per channel (after RoPE) | 0.049 | 0.049 | 0.00 | 2.60 → 2.58 |
| K3 per channel **before** RoPE | 0.027 | 0.027 | +0.01 | 2.60 → 2.59 |
| K2 per token | 0.291 | 0.188 | -0.30 | 2.60 → 3.38 |
| K2 before RoPE | 0.156 | 0.139 | -0.18 | 2.60 → 2.97 |
| K1 per token | 0.575 | 0.365 | -0.44 | 2.60 → 3.65 |

(seed 42 shown; δ = score error in logits with the per-query shift removed, weighted by the int8 attention.)
- Keys have outlier channels, values do not: max / median channel RMS 3-10 for K before RoPE, 2-7 after, V 1.1-1.6;
  the top-4 key channels are the same in both halves of the windows 92-100 % (KIVI and KVQuant see the same).
- At 3-4 bits KL = ½ Var δ to 1 %: the error enters the softmax **quadratically**, not exponentially. At 1-2 bits KL is
  1.3-1.6x that: the large-error regime, mostly a **temperature bias** (scores shrink, slope -0.3 / -0.44, entropy up).
- No "Jensen bias": keys with a larger error lose weight (Spearman -0.1 to -0.4), they do not steal it. The sink
  (position 0) takes 1 % of the moved attention; the recent 32 keys 60-80 %.
- Per channel before RoPE gives K3 the same softmax damage as K4 per token.
Smoke check (64 windows, seed 42, NOT a result): K3 per token +0.050, K3 before RoPE +0.005.

**Test (fixed now, before measuring):** `intsim.py --k-mode pre` = the channel mode (one int32 scale per channel and
group of 32 tokens, KIVI's grouping; its scale sees the later tokens of its group: a slight leak, as `channel`) on the
keys before RoPE, rotated after, then int8 for the attention. 1024 validation windows, seeds 42-44: K4 pre V8, K3 pre
V8, K2 pre V8, K3 pre V3, K3 pre V4, K4 pre V3; K2 pre at the best K2 factor of (c) (chosen on seed 42). Added the same day,
before any of them ran (Edén): pre with the recent int8 window, K3 pre V3 N = 16 / 32, K2 pre V2 N = 32 (K2 at the
factor of (c), V2 x 1.2 = (c)'s choice on seed 42).
**Bytes** (computed as above; per-channel scales: 64 x 4 B per 32 tokens = 8 B per token and head, 98 KB per side):
K3 pre V3 0.74 MB, K3 pre V4 0.84, K4 pre V3 0.84, K2 pre V8 1.13, K3 pre V8 1.23 (KV4 0.88). Same rule: usable d < 0.03, beats KV4
if usable with fewer bytes than 0.88 MB. Cost on the chip (not measured): RoPE on every key read (KVQuant fuses it).

**Result (both rules, 3 seeds, validation, results/kv_modes3.json; reference int8 1.5307, KV4 +0.0098 at 0.885 MB):**

| config | d ± sd | MB (computed) | |
|---|---|---|---|
| K8 V4 | +0.0017 ± 0.0001 | 1.28 | free |
| K8 V3 | +0.0092 ± 0.0003 | 1.18 | usable |
| K3 V8 / K3 V4 / K3 V3 (per token) | +0.048 / +0.050 / +0.058 | | no |
| K4 V3 | +0.0173 ± 0.0006 | 0.79 | beats KV4 |
| K4 V2 / K3 V2 / K2 V4 | +0.067 / +0.117 / +0.202 | | no |
| K3 V3, recent N = 16 / 32 / 64 | +0.0079 / +0.0039 / +0.0014 | 0.75 / 0.81 / 0.93 | N 16, 32 beat KV4 |
| K2 V2, N = 16 / 32 / 64 | +0.109 / +0.069 / +0.033 | | no |
| K1 x pi/2 V2, N = 16 / 32 / 64 | +0.160 / +0.119 / +0.072 | | no |
| K1 x 1.75 (s42 choice) | +0.232 ± 0.007 | | pi/2 +0.244 ± 0.008: does not help |
| K2 x 1.4 (s42 choice) | +0.075 ± 0.003 | | factor 1 +0.200: helps |
| K2 rot x 1.3 (s42 choice) | +0.043 ± 0.002 | | factor 1 +0.076: helps |
| V2 x 1.2 (s42 choice) | +0.025 ± 0.001 | 1.08 | factor 1 +0.058: helps, usable |
| K2 rot x 1.3 + V2 x 1.2 | +0.070 ± 0.000 | 0.49 | no |
| **K2 rot x 1.3 + V2 x 1.2, N = 32** | **+0.0079 ± 0.0003** | **0.64** | **beats KV4, fewest bytes** |
| K4 pre / K3 pre / K2 pre (V8) | +0.0010 / +0.0054 / +0.140 | 1.33 / 1.23 / 1.13 | per token: +0.0079 / +0.048 / +0.200 |
| K2 pre x 1.4 (V8) | +0.074 ± 0.002 | 1.13 | = K2 per token x 1.4 |
| K3 pre V3 / V4, K4 pre V3 | +0.0148 / +0.0070 / +0.0103 | 0.74 / 0.84 / 0.84 | beat KV4 |
| K3 pre V3, N = 16 / 32 | +0.0029 / +0.0016 | 0.80 / 0.85 | beat KV4, ~free |
| K2 pre x 1.4 + V2 x 1.2, N = 32 | +0.0181 ± 0.0003 | 0.68 | beats KV4 |

Seeds 43 / 44 alone (the factors were chosen on 42): within 0.0045 of seed 42 for every factor config, and seed 42
is the same or worse in all but V2 x 1.2 (0.0242 vs 0.0256): the choice did not flatter its own seed.

**Answer:** the diagnosis was right. Keys per channel before RoPE make K3 / K4 nearly free (9x / 8x less damage than per
token); at 2 bits the temperature bias dominates and the scale factor fixes most of it, rotation the rest. The recent
int8 window (32 tokens) takes most of what is left: K2 rot x 1.3 + V2 x 1.2 + N 32 is usable at 0.64 MB (int8 1.67 MB,
2.6x less; KV4 0.88). By the rule it goes to C next. Close second with ~no loss: K3 pre V3 N 16 (+0.003, 0.80 MB).
Not measured yet: the chip cost of the window (two read paths), the key rotation (rotate q once per query: ESTIMATE
cheap) and RoPE on every key read (pre).

**Chip form of the rotation (rule Nov 18 2026, before measuring):** `--k-mode rot` rebuilds every key in the original
basis in floats (not what a chip does). `--k-mode rotq` is the chip form, all integers: keys stored as quant(H k / 8)
(H = +-1 signed Sylvester Hadamard), the query rotated once, H q (int16, additions only), score = (H q) . code x scale
/ 8. Smoke check (64 windows, s42, NOT a result): K8 rotq 1.5419 vs int8 1.5421; K2 x 1.3 V2 x 1.2 N 32: rotq 1.5489
vs rot 1.5491. Run: rotq K2 x 1.3 + V2 x 1.2 + N 32, 3 seeds, 1024 windows. Rule: rotq replaces rot if its d is within
2x the larger sd of rot's (+0.0079 ± 0.0003); otherwise the C has to rebuild keys (or the result does not hold).
**Result (3 seeds, 1024 windows, results/kv_modes3.json `k2Rg1.3_v2tg1.2_r32`):** rotq 1.5399 / 1.5388 / 1.5371,
d = +0.0079 ± 0.0002, the same as rot (+0.0079 ± 0.0003). **Answer:** the rule passes: the chip form gives the
same loss; the C rotates the query once and never rebuilds a key. Chip cost: ../chip_kernels (kv_attn_bench.py).

## Blocks run twice: fewer distinct weights (rule Nov 18 2026, before measuring)

**Question:** MobileLLM runs every block twice in a row with the same weights. A chip then reads each block's
weights once from the PSRAM and uses them twice. Does a binary model keep its quality with half the distinct blocks?

**What:** `train.py --repeat 2` (immediate block-wise sharing; --layers = the depth that runs). Same recipe as the base
(10,000 steps, batch 128, fp32 output layer, KV int8), seeds 42 / 43 / 44, validation loss of train.py (float
model; the integer model and the C do not support --repeat yet):

| run | distinct blocks | depth | 1-bit weights |
|---|---|---|---|
| base `d384l8` (done) | 8 | 8 | 1.77 MB |
| `d384l8r2` | 4 | 8 | 0.89 MB |
| `d384l4` (control: the same bytes, no repeat) | 4 | 4 | 0.89 MB |
| `d384l16r2` (informative: the same bytes as the base) | 8 | 16 | 1.77 MB |

Reference: base 1.5252 / 1.5219 / 1.5200 (mean 1.5224). **Rules:** d = loss - base, mean of 3 seeds.
(1) `d384l8r2` is usable if d < 0.03 (the KV rule). (2) The repeat helps if `d384l8r2` beats `d384l4` by more than 2x
the larger sd. (3) `d384l16r2` beats the base (depth for no extra bytes) if it wins by more than 2x the larger sd.
Speed on the chip is not part of this test. ESTIMATE: `d384l8r2` gives ~10.5-11 tokens/s on the S3, `d384l16r2` is
slower (twice the computation).

**Result** (validation loss, mean ± sd of 3 seeds, `results/repeat_blocks.json`):

| run | 42 / 43 / 44 | mean ± sd | d vs base |
|---|---|---|---|
| base `d384l8` | 1.5252 / 1.5219 / 1.5200 | 1.5224 ± 0.0026 | |
| `d384l8r2` | 1.6406 / 1.6459 / 1.6454 | 1.6440 ± 0.0029 | +0.1216 |
| `d384l4` | 1.6573 / 1.6620 / 1.6637 | 1.6610 ± 0.0033 | +0.1386 |
| `d384l16r2` | 1.5108 / 1.5156 / 1.5124 | 1.5129 ± 0.0024 | -0.0095 |

Speed, measured afterwards on the S3 (`chip_run.py --repeat-probe`, seed 42 model with the int4 head run in the
`--repeat 2` schedule: timing only, the outputs then differ from intsim.py):

| schedule | 1-bit linear layers | block copies | ms per token (windows) | tokens/s (story) |
|---|---|---|---|---|
| base (8 blocks from PSRAM, exact 100 %) | 55.9 ms | | 115.5 | 9.53 |
| pairs, the second pass reads PSRAM again | 55.9 ms | | 115.4 | 9.54 |
| pairs, block copied to internal RAM, both passes there | 37.0 ms | 15.1 ms | 112.2 | 9.84 |

**Answer:** (1) fails: `d384l8r2` is 0.12 worse, four times the limit. (2) passes: the repeat beats the control with
the same bytes by 0.017 (2x sd = 0.0066; every seed, -0.016 to -0.018). (3) passes: `d384l16r2` is 0.0095 better
than the base (2x sd = 0.0053) for the same bytes and twice the computation. On the chip the repeat does not pay:
reading a block twice from PSRAM costs the same as two blocks (the 64 KB cache cannot hold 221 KB), and copying it
to internal RAM first costs 15 ms per token (884 KB at ~59 MB/s, the PSRAM's speed), which eats most of the 19 ms
the second pass saves: +3 % tokens/s for +0.12 loss. Overlapping the copy with the computation (DMA, two block
buffers = 442 KB) does not fit the ~250 KB of internal RAM. Not pursued; the ESTIMATE above (10.5-11 tokens/s) was
wrong. The memory that limits the ESP32 and the Pico 2 is the KV cache, which repeating blocks does not shrink.

## Attention in bits: sign scores, a learned temperature, buckets (rule Nov 19 2026, before measuring)

**Question:** can attention itself run on bits? With the query and the key as signs (64 bits each per head), a
score is q̂·k̂ = 64 - 2h, h = popcount(q̂ XOR k̂): XOR + popcount, no multiplication. A softmax weight is then one of
65 values, 2^(-τ(h - hmin)) from a table, and the values can be summed per h first ("buckets": 65 sums, then 65
multiplications). The KV cache drops from 1.57 MB (int8) to ~0.2 MB (keys 8 bytes + 2-bit values 16 bytes per
position, head and layer). Our earlier QAT with 1-bit keys (section "KV cache trained at 1-bit keys and 2-bit values")
lost +0.30 float / +0.32 integer: it had no learned temperature, no recent window, and the query stayed int8.

**Literature (searched Nov 19):** each piece exists, not the whole: binary attention with a learnable bias against
the flat softmax of binary scores (BinaryAttention, CVPR 2026, arXiv 2603.09582; HAD 2502.01770, BiBERT 2203.06390:
encoders, fine-tuned or distilled), 1-bit key sketches with 4-bit values (QJL AAAI 2025, H1B-KV 2510.05529), base-2
softmax (Softermax 2103.09301), few attention levels (Ji et al. 2106.01335), multiplication-free attention by
gather-and-add (2605.01910), hashing attention (EcoFormer 2209.09004). Not found: summing the values per Hamming
distance, nor a generative LLM trained this way and run exact on a microcontroller. Note: a norm before the sign
(QK-norm) changes nothing, the sign does not see the scale.

**What:** `train.py` options (float model; intsim.py and the C do not have them yet): `--attn-temp` (one learned
temperature per head, exp of a parameter starting at 0), `--kv-recent N` (the last N positions unquantized, as
intsim --kv-recent), `--q-bits 1` (query sign x mean|q|), `--k-unit` (key = pure sign: then the score is an integer
Hamming count times one factor per query: the bucket form). Same recipe as the base (10,000 steps, batch 128,
binary weights, fp32 output layer). Screening, seed 42, validation:

| run | query | key | values | temperature | recent |
|---|---|---|---|---|---|
| T1 `attn_k1v2_tau` | int8 | sign x mean\|k\| x π/2 | 2 bits | learned | |
| T2 `attn_k1v2_tau_r32` | int8 | as T1 | 2 bits | learned | 32 |
| T3 `attn_bits_tau` | sign x mean\|q\| | pure sign | 2 bits | learned | |
| T4 `attn_bits_tau_r32` | as T3 | as T3 | 2 bits | learned | 32 |

**Rules** (d = validation loss - base; base 1.5252 seed 42, 1.5224 mean of 3): (1) a run passes the screening if d <
0.10 at seed 42; the best one that passes (lowest loss) gets seeds 43 and 44 the same night. (2) Usable if the
3-seed mean d < 0.03 (the KV rule). (3) The learned temperature helps if T1 is below the earlier QAT 1.826 by more than
0.05 (one seed: a ranking, not a result). (4) Only a usable run goes on to intsim.py (integer, buckets) and the C;
the float loss is not the final word (the earlier QAT lost 0.02 more in integers). Chips, the same night, separate
from the loss: one head, random data, exact against numpy (`chip_kernels/bit_attn_bench.py`): int8 attention vs
XOR + popcount scores, values summed directly vs by buckets. ESTIMATE: buckets are not faster on chips with a one-cycle
multiply (the additions stay 256 x 64); the gain is the memory.

**Result** (validation loss, float model, `results/attn_bits.json`; learned temperature = mean of exp(τ) per block):

| run | 42 / 43 / 44 | d vs base | temperature, blocks 0 -> 7 |
|---|---|---|---|
| T1 K1 V2 τ | 1.5952 | +0.070 | 0.43, 0.09, 0.06, 0.06, 0.02, 0.01, 0.01, 0.01 |
| **T2 K1 V2 τ recent 32** | 1.5355 / 1.5361 / 1.5345 = **1.5354 ± 0.0008** | **+0.013** | 3.52, 0.42, 0.32, 0.15, 0.34, 0.23, 0.26, 0.13 |
| T3 pure bits τ | 4.5241 | +3.00 | ~1.0 everywhere (did not learn) |
| T4 pure bits τ recent 32 | 2.0752 | +0.55 | 0.82, 0.26, 0.25, 0.23, 0.29, 0.16, 0.25, 0.07 |

Chips (`chip_kernels/bit_attn_bench.py`, chip_kernels README): sign scores 8.7-22x faster than int8, 2-bit values
1.2-1.3x slower (unpacking), buckets no gain; scores + values 1.42-1.98x per head, exact on 4 cores.
Note: these chip times are the PURE bit form (T3 / T4, which fails in training). T2 keeps an int8 query: its score
is a masked sum of the query, sum over bit planes b of 2^b x popcount(q_b AND k) (8 planes x 2 words per key), not
yet timed (ESTIMATE ~4x faster than int8, to be measured). Measured in the whole model on the S3 (below): not
faster than the PIE int8 dot.

**Answer:** rule 1: T1 and T2 pass, T3 / T4 fail; T2 got seeds 43 / 44. Rule 2: T2 is usable in float (d +0.013 ±
0.0008 < 0.03): 1-bit keys, 2-bit values and a learned temperature, with the last 32 positions unquantized. Rule 3:
the temperature helps (T1 1.595 vs the earlier QAT 1.826, one seed: a ranking). The temperature goes DOWN, not up
(the opposite of BinaryAttention): without the window, blocks 1-7 flatten their attention almost to an average. The
pure bit form (sign query, key without its scale) fails: the scale of k and q matters. Next: T2 in intsim.py
(integer, 1-bit keys with mean|k|, 32 int8 recent) and its KV bytes; the chip kernel for the values (bit planes).

**Integer model (intsim.py, Nov 19, `results/attn_bits_intsim.json`):** intsim reads the learned temperature (exp(τ)
per head in Q16, applied to the exp-table index) and runs T2 with the same KV it was trained with (`--k-bits 1
--v-bits 2 --k-scale 1.5708 --kv-recent 32`), 1024 windows. Integer loss 1.5432 / 1.5445 / 1.5446, d vs the base
integer KV8 of the same seed **+0.0134 ± 0.0020: usable in integers too** (float -> integer +0.0002; next-token
agreement 98.2-98.3 %). KV bytes (computed, kv_modes3.json's formula: the int32 scales and the int8 window count)
**0.55 MB** vs int8 1.67 MB: the "~0.2 MB" above ignored the scales and the window (1-bit K + 2-bit V alone: 0.39
MB). Against the best KV without retraining (K2 rot + V2 + recent 32: +0.0079 at 0.64 MB), T2 has 14 % fewer bytes
for a larger loss and needs training; what it adds is a key of 1 bit (scores without multiplications).

**On the ESP32-S3 (Nov 19, seed 42, `chip_run.py --linear symbex`, first firmware: `results/chip_s3_binary_attn_k1v2_tau_r32_s42_symbex_wide.json`):**
tinylm.c runs T2 when model.bin ends with a trailer (`intsim.py --export` writes it: key bits 1, value bits 2, the
window, k_scale, exp(τ) per head): keys as signs + one scale, values transposed 2 bits each, a ring of the last 32
int8 keys and values, the tempered exp-table index (two 32-bit divisions). Without the trailer the int8 path is
unchanged. **Exact: 512 / 512 validation tokens and the 120-token story equal to intsim.py**, the base too (re-run
on the same firmware). Both with the int8 output head.

| | base (int8 KV) | T2 (K1 V2 τ + 32 int8) |
|---|---|---|
| val loss (integer) | 1.5318 | 1.5432 (+0.011) |
| KV cache in PSRAM (allocated) | 1.67 MB | **0.60 MB** (PSRAM free +1.07 MB, measured) |
| internal RAM free | 233 KB | 215 KB (-17 KB: one head's values decoded) |
| tokens / s (story) | **8.47** | 7.56 (-11 %) |
| ms per token, position 0 / 255 | 108.6 / 148.5 | 116.7 / 178.1 |
| attention: cache write / scores / softmax / values (ms) | 9.7 / 7.8 / 2.0 / 13.3 | 12.1 / 9.2 / 2.9 / 25.1 |

0.60 MB, not the 0.55 MB computed above: the C keeps every position also in 1 / 2 bits (a position leaving the
window needs them; the int8 copy cannot rebuild them exactly). **T2 saves memory, not time, on the S3:** the PIE
int8 dot already makes the scores cheap (the sign scores through a subset-sum table, plain C, are not faster), and
the 2-bit values cost +12 ms per token (copying each head from PSRAM to internal RAM first did not help, 7.50
tokens/s). **Correction (Nov 19, measured on the chip, next section):** that +12 ms was mostly not the decoding: in
26 % of T2's head steps the weights p x v overflowed 32 bits and the head fell back to a plain-C loop. Fixed, T2 runs
7.87 tokens/s (7.94 with the cut). Where it could pay more: a chip without SIMD (the bench: sign scores 8.7-22x
faster than plain int8) or a value kernel on the packed bits (future work).

### Reading the cache: memory down, time up (Nov 19)

**What T2 achieved:** the KV cache is 2.8x smaller (1.67 -> 0.60 MB), exact on the chip. **What it costs:** every new
token reads every earlier position, and reading the packed bits takes time: the cost per position grows from 0.157
(base) to 0.241 ms (T2, both measured on the S3: 108.6 -> 148.5 and 116.7 -> 178.1 ms from position 0 to 255).
The time is in unpacking the values (4 positions share a byte), not in the PSRAM. We are working on reading fewer
positions:

- **Exact zeros are rare.** The exp table gives p = 0 only ~11.8 nats below the maximum: 73-100 % of the keys before
  the last 32 keep p > 0 (seed 42, 8 windows), so skipping them saves only ~5-10 % of the value decoding.
- **An approximate cut** drops the keys with p < p_max e^-n (`attn_cut.py`, intsim `kv["p_cut"]`). Seed 42 (explored
  first, a ranking), validation, 1024 windows: n = 6 keeps 39 % of T2's far keys for +0.0005 loss, n = 4 keeps 21 %
  for +0.005.
- **Page summaries do not work with 1-bit keys** (`quest_probe.py`, seed 42, 64 windows, a ranking): the Quest bound
  (per-channel max / min of a page's keys) is almost never below the cut, because inside a page nearly every channel
  has both a +1 and a -1. At n = 6 it skips 6.5 / 0.4 / 0.0 % of the far pages of 4 / 8 / 16 positions, while 46 /
  39 / 32 % could be skipped (oracle). Not pursued: the scores of every position are still computed; the cut saves
  only the values.

**Rule (written before the 3 seeds):** for base and T2, cuts n = 8, 6, 4, 3, 2, 1, validation only. The chosen n is
the smallest whose mean loss rise over no cut (paired by seed) is below 0.003, a tenth of the 0.03 "usable" limit
of the KV study. Only then the C:
values stored per position (so a skipped position is not decoded), the cut in tinylm.c, exact against intsim.py with
the same cut, then the S3 time per position. Speed is claimed only from the chip.

**Result (3 seeds, validation, 1024 windows, `results/attn_cut.json`):** loss rise over no cut (paired) and share of
the far keys still read.

| cut n (nats) | base: rise | base: far keys read | T2: rise | T2: far keys read |
|---|---|---|---|---|
| none | 0 (1.5307 ± 0.0013) | 90.0 % | 0 (1.5441 ± 0.0008) | 87.8 % |
| 8 | +0.0000 ± 0.0001 | 63.8 % | +0.0000 ± 0.0001 | 61.6 % |
| **6** | **+0.0002 ± 0.0002** | **42.6 %** | **+0.0002 ± 0.0002** | **41.3 %** |
| 4 | +0.0051 ± 0.0011 | 22.1 % | +0.0045 ± 0.0007 | 21.9 % |
| 3 | +0.0171 ± 0.0026 | 13.7 % | +0.0149 ± 0.0014 | 14.1 % |
| 2 | +0.0522 ± 0.0057 | 6.8 % | +0.0454 ± 0.0025 | 7.5 % |
| 1 | +0.1607 ± 0.0128 | 1.9 % | +0.1485 ± 0.0063 | 2.2 % |

**Answer:** by the rule, **n = 6 for both**: the loss does not move (+0.0002) and less than half of the far
positions need their values. Then the C and the S3 (below).

**On the S3 (Nov 19, seed 42, `chip_run.py --linear symbex [--p-cut 6] [--fine-prof]`, `results/chip_s3_*_symbex*.json`):**
tinylm.c drops the weights past the cut (`-DTINYLM_P_CUT=768`) and, with the low-bit cache, decodes only the groups
of 4 positions (one byte of the 2-bit cache) with some p != 0: the cache format stays (storing values per position
would need 4x the stores per decoded position, an estimate). **Exact: 512 / 512 tokens and the story equal to
intsim.py, with and without the cut** (PC: every build, ASan / UBSan clean; the cut firmware against the no-cut
reference does differ, so the check sees it).

The first cut firmware gained little (T2 7.56 -> 7.64 tokens/s, `..._c6_wide.json`; skipping only exact-zero groups
7.54, `..._skip0.json`). A fine profile on the chip (`--fine-prof`: cycle marks inside the attention, their cost
~0) showed where the value time went: **10.4 of 20.4 ms in a plain-C fallback**, not in the decoding (`..._prof_wide.json`).
In 26 % of T2's head steps p x v >= 2^32 (PC count), so the head left the PIE path. Fix: the weights split into
up to 6 bytes, only the bytes the largest weight needs written (every head on the PIE; the sum <= 2^63 is checked).

| | T2 first | T2 | T2, cut 6 | base | base, cut 6 |
|---|---|---|---|---|---|
| attention values (ms per token, mean of 2 x 256) | 25.1 | 15.3 | **11.7** | 13.0 | 12.9 |
| ms per token at position 255 | 178.1 | 151.9 | **142.9** | 148.1 | 146.9 |
| ms per token, mean of positions 0-255 | 145.4 | 135.6 | **131.6** | 128.4 | 127.9 |
| tokens / s (story, positions 5-125) | 7.56 | 7.87 | **7.94** (+5 %) | 8.47 | 8.48 |

Where a T2 + cut token goes (fine profile, mean of positions 0-255, ms): output head 28.0, w1 / w3 24.8, qkv 14.3,
cache write + RoPE 12.1, w2 11.5, SwiGLU 5.9, o 5.4, norms + embedding 4.7; scores 9.2 (subset table 4.0, far keys
3.5, int8 window 1.7), softmax + division 4.0, values 11.7 (PIE products 4.7, decoding 3.6, window copy 1.6, weight
bytes 1.2, packing 0.6). **Answer: the cache is 0.60 MB instead of 1.67 MB at 6 % fewer tokens per second than the
base (was 11 %); at position 255 T2 is now faster than the base (142.9 vs 148.1 ms).** The weights (84 of 132 ms)
dominate the token: the attention has at most ~10 ms left to gain (estimate). Next ideas: a value kernel on the
packed 2-bit codes (no decoding), a cheaper subset table, a longer context where the cut pays more.

## KV cache: conclusions (Nov 19 2026)

The work of this branch answers three questions. Model: 15.7 M W1A8, ESP32-S3, context 256; loss on validation
(3 seeds), chip numbers seed 42, every chip run exact 100 %.

**1. How do we lower the cache memory? Answered.**

| cache | KV memory | val loss rise | retraining |
|---|---|---|---|
| int8 (base) | 1.67 MB | 0 | no |
| K2 rotq + V2 + 32 recent int8 | 0.64 MB | +0.008 | no |
| **T2: K 1 bit + V 2 bits + learned temperature + 32 recent int8** | **0.60 MB** | +0.011 | yes |

What made it work: keys have outlier channels (values do not), so keys need the rotation; at 1-2 bits the damage
behaves like a temperature bias, so a learned temperature fixes most of it; the last 32 positions in int8 cost
little memory and keep the quality. A 1-bit key without retraining was never usable.

**2. How do we make it faster? Partly.**

| step | on the S3 |
|---|---|
| cut at 6 nats (p = 0 for far positions) | free in loss (+0.0002); 41-43 % of far keys read |
| overflow fix (found by the chip profile) | attention values 25.1 -> 11.7 ms per token |
| T2 + cut, whole model | 7.94 vs base 8.47 tokens/s (-6 %); position 255: 142.9 vs 148.1 ms (faster) |
| Quest page bound | useless on 1-bit keys (skips 0-6.5 % of pages; oracle 32-46 %) |

The limit on the S3: the PIE int8 dot already makes the int8 scores cheap, and the weights take 84 of the 132 ms of
a token. The cache has at most ~10 ms left to gain (estimate). **This method lowers memory; speed must come from
reading fewer weight bytes per token.**

**3. Where can it help other models? Where memory decides.**

- Small models on chips without PSRAM (classic ESP32, Pico 2): the cache must fit in internal RAM; a third of the
  bytes decides whether the model runs at all.
- Long contexts: the cache grows with every token, the weights do not; T2 with the cut gains more the longer the
  context (estimate: measured only up to 256).
- Chips without SIMD (classic ESP32, Hazard3): sign scores ran 8.7-22x faster than plain int8 (one kernel bench,
  not a whole model).
- Any softmax attention: the cut in nats and the int8 window carry over as they are.

**Answer: T2 makes the model fit; speed comes from the weights.** Next (agreed Nov 19 2026): a smaller model (the
260K shape: dim 64, 5 layers, 8 heads, vocabulary 512), weights from flash, cache in internal RAM, on the ESP32 and
the Pico 2, where these conclusions are tested on a second model and on chips without wide SIMD. Repeated blocks
are not a speed line (section above: on the chip a block read twice costs as two). Where the weights could still
gain (`../chip_kernels/results/psram_bw_bench_s3_80.json`, measured Nov 12; this model has the int8 head, 3.55 MB):
the output head (28 ms) already reads at the PSRAM's speed (58 MB/s); the 1-bit linears (56 ms) run at 40 MB/s from
PSRAM, against 61 MB/s for a plain sequential read and 69 MB/s for the same kernels with the weights in internal
RAM. Overlapping the reads with the computation could bring them toward ~61 MB/s, ~-19 ms per token (estimate from
those measured rates); the int4 head (14.7 ms, above) and the PSRAM at 120 MHz (head 58 -> 80 MB/s) add to it.

## Reading the weights while computing (Nov 19 2026)

**Question.** The 1-bit linears read their weights from PSRAM at 40 MB/s, the same kernels on internal RAM at 69
MB/s (section above). Do they wait on each cache miss, and can the reads overlap the computation?

**What.** `../chip_kernels/psram_prefetch_bench.py` (S3, PSRAM 80 MHz, the model's three linear shapes, 32 fresh
copies per layer so nothing stays in the 32 KB data cache, best of 7, outputs of every variant hashed against the
plain kernel's) tried five ways; the best one went into the firmware as `chip_run.py --autoload 3`
(`-DTINYLM_AUTOLOAD`: the data cache's autoload, ROM `Cache_Config_DCache_Autoload`, over the model buffer, set once
at start). Then the whole model on the chip, exact against intsim.py, seeds 42 / 43 / 44.

**Result.** Kernel alone, MB/s (384 -> 384 / 384 -> 1024 / 1024 -> 384; `results/psram_prefetch_bench_s3.json`):

| | MB/s |
|---|---|
| plain (today) | 39.7 / 39.7 / 41.9 |
| internal RAM (ceiling) | 68.9 / 69.0 / 75.5 |
| cache preload of the next 4 KB block | 51.8 / 57.5 / 57.8 |
| cache autoload, size 3 | 55.8 / 55.9 / 56.7 |
| memcpy to internal RAM, then compute | 31.8 / 31.8 / 33.1 |
| GDMA copy of the next block (double buffer) | 22.8 / 23.4 / 23.7 |

Whole model, S3, 3 seeds, every run exact 100 % (512 / 512 validation tokens + story 120 / 120):

| model | ms per token | tokens/s (story) | autoload: ms per token | tokens/s |
|---|---|---|---|---|
| int8 head (3.55 MB) | 128.5 ± 0.0 | 8.48 ± 0.00 | 114.2 ± 0.0 | **9.62 ± 0.00** |
| int4 head (2.77 MB) | 115.6 ± 0.1 | 9.52 ± 0.01 | 101.6 ± 0.0 | **10.96 ± 0.00** |

Seed 42 by part (int8 head): linears 55.8 -> 43.7 ms, head 28.0 -> 26.2, attention 34.0 -> 34.0. T2 + cut 6 (seed
42): 7.94 -> 8.90 tokens/s. The autoload over the KV cache too (`--autoload-kv`, region 1) moves the attention by
+-1.5 ms both ways (seed 42, three models): not kept. A first try over the whole external data range (it also maps
the flash) rebooted the board in a loop: not saved.

**Answer: yes, the linears waited on PSRAM; one call at start gives -12 to -14 ms per token, 1.14-1.15x, exact.**
The estimate before measuring was -19 ms; measured -12 on the linears. Weak point: the ROM cache functions are marked
"do not call in your SDK application"; they work on this Arduino core (ESP-IDF 4.4) and may change with another.

## A 260K model for chips without PSRAM (rule Nov 19 2026, before measuring)

**Question.** The 15.7 M model needs PSRAM (KV cache 1.67 MB): the classic ESP32 (346 KB heap, no PSRAM) and the
Pico 2 (520 KB RAM) cannot run it. Does llama2.c's 260K shape train as a binary W1A8 model, and how far is it from its
fp32 twin? The goal of this line is **a model that runs exact on a chip without PSRAM** (weights from flash), not
better stories.

**What.** `train.py --vocab 512 --dim 64 --layers 5 --heads 8 --kv-heads 4 --hidden 172 --seq 512` (llama2.c's
stories260K shape; `--kv-heads 4` = grouped-query attention, each key / value head serves 2 query heads),
tokenizer `prepare.py --vocab 512` (ours, trained on TinyStories: llama2.c's val loss 1.297 uses its own tokenizer
and is not comparable). The same recipe as every other run here (10000 steps, batch 128, lr 1.5e-3, warmup 500),
fp32 and binary (1-bit weights, 8-bit inputs, fp32 output layer tied to the embedding), seeds 42, 43, 44, validation
only. KV cache int8, computed: 5 layers x 512 positions x 4 heads x 8 dims x 2 (K, V) = 164 KB (328 KB with 8 KV heads;
plus the int32 scales per token and head, 80 KB: see step 2).

**Rule.** Report val loss mean ± sd of both and the gap. No accuracy threshold decides the next step: the binary
model goes to intsim.py / the C (step 2) unless its sample stories are not sentences at all (then the recipe is
looked at first, on validation). The test split is not read.

**Result** (results/v512d64l5kv4_train.json, 0.26 M parameters, ~4.7 min per run on the RTX GPU):

| weights | val loss (42 / 43 / 44) | mean ± sd |
|---|---|---|
| fp32 | 1.3433 / 1.3542 / 1.3417 | 1.3464 ± 0.0068 |
| binary W1A8 | 2.1567 / 2.1279 / 2.1365 | 2.1404 ± 0.0148 |

Gap +0.794 (the 15.7 M model: +0.222). The binary curve is flat from step ~4000 (2.23 -> 2.14). Its stories are
sentences with weak sense ("She lived in a big box. She wanted to play with his friends and play with the park").

**Answer.** It trains, and by the rule it goes to step 2 (intsim.py / the C). Binary costs much more at this size:
with dim 64 every layer has few bits to spend. Better recipes for this size stay future work, chosen on validation.

### Step 2: the 260K model as integers and C (Nov 19 2026)

**Question.** Is the integer model of the 260K shape as good as its float one, and does the C give the same integers?

**What.** intsim.py and c/tinylm.c learned what the shape needs: heads of any width that is a multiple of 8 (the score
scale 1 / sqrt(hd) becomes the integer divisor `att_div(hd) << 20`, `att_div = (isqrt(16129^2 hd) + 4) >> 3`: 16129
for 64-wide heads, the same integers as before; 5702 for 8-wide), grouped-query attention (the cache keeps the 4 key /
value heads; model.bin's heads field = kv_heads << 16 | heads, old files unchanged), a hidden width that is not a
multiple of 8 (172: the last table group counts the missing inputs as 0), and contexts past 258 tokens (the int32
value sums run over blocks of 256 keys; the attention scratch is sized by the context, not by `hidden`). The low-bit
cache keeps its limits (64-wide heads, no GQA, 258 tokens). Validation, 1024 windows of 512 tokens; the C on 4
windows (2048 tokens, every one of the 512 outputs) + a 120-token greedy story, 3 kernels (lut8, lib_symbex,
lib_symbex + the row-product hooks), built with ASan + UBSan.

**Result** (results/intsim_binary_v512d64l5kv4_s*.json, results/c_check_pc_binary_v512d64l5kv4_s*.txt):

| seed | val loss float | val loss integer | next token equal float / integer | C = intsim.py |
|---|---|---|---|---|
| 42 | 2.1671 | 2.1676 | 97.5 % | 2048 / 2048, story 120 / 120 |
| 43 | 2.1397 | 2.1402 | 97.6 % | 2048 / 2048, story 120 / 120 |
| 44 | 2.1457 | 2.1463 | 97.7 % | 2048 / 2048, story 120 / 120 |

The integer model costs +0.0005 ± 0.0001 nats. (The float loss here is over 1024 windows; train.py's best checkpoint
loss, 2.1404 ± 0.0148, over its own 40 x 128.) The older models (15.7 M with int8 / int4 / int2 head, W1A1, the
low-bit cache) are still exact with the new C, and the S3 firmware builds.

**Answer.** Yes: exact 100 %, 3 seeds. model.bin is 0.09 MB. RAM on a chip, computed: int8 KV 164 KB + its int32
scales 80 KB (5 layers x 512 x 4 heads x 4 bytes x 2) = 244 KB, plus roughly 30 KB of scratch (estimate): it should fit the classic
ESP32 (346 KB heap) and the Pico 2 (520 KB), to be measured. Next (step 3): firmware for both, weights from flash.

### Step 3: the 260K model on the classic ESP32 and the Pico 2 (Nov 19 2026)

**Question.** Does the 260K binary model run exact on chips without PSRAM, weights read from flash, and how fast?

**What.** `chip_mcu/` (PlatformIO, Arduino): model.bin is a 16-byte aligned const array in the firmware
(`src/model_data.h`, written by `chip_run.py`): the ESP32 reads it through its flash cache, the RP2350 through XIP
(16 KB cache). The int8 KV cache and the buffers sit in internal RAM. Plain C everywhere (lut8 linear layers, no
SIMD). The classic ESP32 needed two changes: its byte heap is 279 KB in regions (largest block 115 KB), so
`-DTINYLM_KV_SPLIT` allocates the cache per layer (16 KB blocks), and `TINYLM_MALLOC_WORDS` puts the int32 scales
in the 32-bit-only IRAM heap (~72 KB). Both are opt-in: the PC and the S3 builds are unchanged, and the C is exact with or
without them (PC, ASan + UBSan). `python chip_run.py runs/<run> --board esp32|pico2_arm|pico2_riscv --port ...`:
2 validation windows of 512 tokens (a hash of every output equal to intsim.py's) + the 120-token greedy story.
ESP32 at 240 MHz, Pico 2 at 150 MHz (Arduino-Pico default). Seeds 42, 43, 44.

**Result** (results/chip_{esp32,pico2_arm,pico2_riscv}_binary_v512d64l5kv4_s*.json):

| chip | exact (3 seeds) | story tokens/s | ms per token, windows mean | at position 0 / 511 |
|---|---|---|---|---|
| ESP32 (classic, 240 MHz) | 3072 / 3072, story 360 / 360 | 40.15 ± 0.06 | 35.38 ± 0.03 | 19.9 / 49.1 |
| Pico 2 Cortex-M33 (150 MHz) | 3072 / 3072, story 360 / 360 | 31.84 ± 0.06 | 41.36 ± 0.05 | 27.0 / 54.1 |
| Pico 2 Hazard3 (150 MHz) | 3072 / 3072, story 360 / 360 | 35.38 ± 0.88 | 39.52 ± 0.75 | 23.3 / 53.6 |

Flash: the firmware is 0.39 MB on the ESP32 (model 0.09 MB). Free RAM after loading: ESP32 72 KB (most of it IRAM),
Pico 2 228-235 KB. The time per token grows with the position (the attention reads the whole cache): ~2.5x from
the first to the last token. The story runs 125 positions, so its speed is above the window mean. The Hazard3 spread
(seed 42 34.8, 43 36.4, 44 34.9 tokens/s, same code) is code layout under the XIP cache, as seen before with
SYMBEX on this core.

Tokens per second of the story, per seed (each run's JSON, `tokens_per_second_story`):

| chip | seed 42 | seed 43 | seed 44 | mean ± sd |
|---|---|---|---|---|
| ESP32 (classic, 240 MHz) | 40.200 | 40.090 | 40.166 | 40.15 ± 0.06 |
| Pico 2 Cortex-M33 (150 MHz) | 31.835 | 31.776 | 31.896 | 31.84 ± 0.06 |
| Pico 2 Hazard3 (150 MHz) | 34.823 | 36.394 | 34.909 | 35.38 ± 0.88 |

Memory (bytes; RAM used = free heap at start - free heap after loading, from the firmware's own lines, the same on
all 3 seeds; KV cache and scales computed from the shape; scratch = used - the two):

| | ESP32 | Pico 2 M33 | Pico 2 Hazard3 |
|---|---|---|---|
| model.bin in flash | 93 120 | 93 120 | 93 120 |
| free RAM at start | 350 764 | 513 944 | 506 928 |
| free RAM after loading | 71 672 | 235 144 | 228 128 |
| **RAM used** | **279 092** | **278 800** | **278 800** |
| of it: int8 KV cache (5 x 512 x 4 x 8 x 2) | 163 840 | 163 840 | 163 840 |
| of it: int32 KV scales (5 x 512 x 4 x 4 x 2) | 81 920 | 81 920 | 81 920 |
| of it: scratch, buffers, heap overhead | 33 332 | 33 040 | 33 040 |

The ~30 KB scratch estimate of step 2 is now measured: 33 KB.

**Answer.** Yes: a binary W1A8 language model runs exact (100 %, 3 seeds, 3 cores) on a classic ESP32 and on a
Pico 2, weights in flash, at 32-40 tokens/s. Its stories are weak (val loss 2.14 vs fp32 1.35, above): this is a
memory and exactness result, not a quality one. Not measured yet: an fp32 or int8 baseline on the same chips, and
the S3's PIE kernels on this model.


### Step 4: the training recipe at 260K (rule Nov 19 2026, before measuring)

**Question.** The binary 260K model was trained with the 15.7 M recipe (lr 1.5e-3, 10000 steps) and its curve is flat
from step ~4000. Does another learning rate or a longer run close part of the gap to fp32 (2.14 vs 1.35)?

**What.** Binary W1A8, the same shape, seeds 42, 43, 44, validation only. A: lr 1e-3 and 3e-3 at 10000 steps (base
1.5e-3 = 2.1404 ± 0.0148). B: 20000 steps (1.3 B tokens, 1.25 passes over the 1.05 B training tokens) at the best lr of
A (or of the base if none wins). C: fp32 with the recipe chosen in B, so the gap is measured with one recipe for
both (the fp32 base 1.3464 stays as the reference too).

**Rule.** A new lr is kept only if its mean val loss beats the base by more than the larger sd; 20000 steps is kept
only if it beats 10000 steps at that lr by more than the larger sd. Otherwise the base recipe stays. The test split is
not read.

**Result** (results/v512d64l5kv4_recipe.json, val loss, seeds 42 / 43 / 44):

| weights | recipe | val loss | mean ± sd | minutes per run |
|---|---|---|---|---|
| binary | lr 1.5e-3, 10k (base) | 2.1567 / 2.1279 / 2.1365 | 2.1404 ± 0.0148 | 4.9 |
| binary | lr 1e-3, 10k | 2.1688 / 2.1617 / 2.1642 | 2.1649 ± 0.0036 | 4.8 |
| binary | lr 3e-3, 10k | 2.1007 / 2.0956 / 2.1003 | 2.0989 ± 0.0028 | 4.9 |
| binary | **lr 3e-3, 20k** | 2.0632 / 2.0519 / 2.0688 | **2.0613 ± 0.0086** | 9.7 |
| fp32 | lr 1.5e-3, 10k (base) | 1.3433 / 1.3542 / 1.3417 | 1.3464 ± 0.0068 | 4.5 |
| fp32 | **lr 3e-3, 20k** | 1.2616 / 1.2787 / 1.2725 | **1.2709 ± 0.0087** | 9.2 |

Both rules pass: lr 3e-3 beats the base by 0.042 (sd 0.015), 20000 steps beat 10000 by 0.038 (sd 0.009).

**Answer.** The new recipe (lr 3e-3, 20000 steps) improves the binary model by 0.079, but fp32 gains the same
(0.076): the gap stays +0.790 (was +0.794). The recipe was not what held the binary model back; the gap is the
size (dim 64 at 1 bit). The new recipe is the one for the next steps (the chip models above stay the base ones
until a combined model is exported).

### Step 5: ternary weights at 260K (rule Nov 19 2026, before measuring)

**Question.** If the gap is the size (step 4), ternary weights (-1, 0, 1: up to ~1.58 bits each, 2 bits stored) give
each layer more to spend at the same shape. How much of the gap do they close?

**What.** `--weights ternary`, the same shape, the step 4 recipe (lr 3e-3, 20000 steps), seeds 42, 43, 44,
validation only. Compared with binary at the same recipe (2.0613 ± 0.0086) and fp32 (1.2709 ± 0.0087).

**Rule.** Ternary counts as better if its mean val loss beats binary by more than the larger sd. Weights in flash
double (2 bits, computed: ~0.09 -> ~0.15 MB model.bin, still small). Whether it is carried to the chips is decided
after the next steps, with the other winners. The test split is not read.

**Result** (results/v512d64l5kv4_ternary.json, lr 3e-3, 20000 steps, val loss):

| weights | seeds 42 / 43 / 44 | mean ± sd | gap to fp32 |
|---|---|---|---|
| binary | 2.0632 / 2.0519 / 2.0688 | 2.0613 ± 0.0086 | +0.790 |
| **ternary** | 1.8315 / 1.8416 / 1.8394 | **1.8375 ± 0.0053** | **+0.567** |
| fp32 | 1.2616 / 1.2787 / 1.2725 | 1.2709 ± 0.0087 | 0 |

**Answer.** Yes: ternary beats binary by 0.224 (larger sd 0.009), 28 % of the gap. The rest (+0.57) is still large.
The C has no ternary weights yet (intsim.py's C check packs 1-bit weights only): carrying it to the chips needs that
kernel.

### Step 6: a learned temperature per head at 260K (rule Nov 19 2026, before measuring)

**Question.** With quantized weights the attention scores come out flat (README "Attention in bits"); one learned
temperature per head (`--attn-temp`) sharpens them. Does it help the 260K model, binary and ternary?

**What.** `--attn-temp`, the step 4 recipe (lr 3e-3, 20000 steps), binary and ternary, seeds 42, 43, 44, validation
only. Compared with steps 4 / 5 without it (binary 2.0613 ± 0.0086, ternary 1.8375 ± 0.0053). The temperature is not
in intsim.py / the C yet (one integer factor per head on the scores: small work, estimate).

**Rule.** Kept for a weight kind if its mean val loss beats the same kind without it by more than the larger sd.
The test split is not read.

**Result** (3 seeds, validation; [results/v512d64l5kv4_temp.json](results/v512d64l5kv4_temp.json)):

| weights | without temperature | with `--attn-temp` | difference | rule |
|---|---|---|---|---|
| binary | 2.0613 ± 0.0086 | 2.0510 ± 0.0080 (2.0448 / 2.0601 / 2.0482) | -0.0103 | passes (just: 0.0103 > 0.0086) |
| ternary | 1.8375 ± 0.0053 | **1.8199 ± 0.0059** (1.8186 / 1.8148 / 1.8263) | -0.0176 | passes (3x the sd) |

**Answer.** Kept for both. Small for binary (barely above the spread), clearer for ternary; the best 260K is now
ternary + temperature 1.8199 (gap to fp32 1.2709: 0.549). Training took ~56 min per run instead of ~10 (attention
by hand, no fused kernel). On a chip the temperature is one factor per head on the scores (folded into the score
scale: estimate, not in intsim.py / the C yet).

### Step 7: the key taken apart (rule Nov 20 2026, before the 3-seed runs and before any training)

**Question.** At 15.7 M the key K was the part of the cache that broke in few bits. Which variable hurts the 260K
model most, does the residual sum pile the errors up, and what in K makes it hard, and can that be fixed?

**What.** Two no-training diagnoses, 3 seeds, validation windows (fixed scales / means from TRAINING windows):
`error_anatomy.py` (one quantization source at a time in the trained fp32 model: W1A8 on each linear kind, K or V
cache in few bits; loss rise, residual error after each block, joint vs single, one block at a time and how its
error grows through the later blocks) and `k_anatomy.py` (K statistics per layer and key head, and the key cache at
4 / 3 / 2 bits stored six ways in the fp32, binary and ternary models). One-seed smoke runs of both were seen before
this rule was written (the key mean holds 50-97 % of the key's energy; storing key - mean cut the 4-bit cache loss
rise of the binary model from 0.10 to 0.01): the training below was chosen from them.

Then training, binary weights, the step 4 recipe, 3 seeds, validation only:
- key cache at 2 and at 4 bits, plain vs `--k-center` (key - a fixed mean, added back when read);
- `--int8-linears k` vs `--int8-linears v` (the same bytes: W_k and W_v have the same shape).

**Rule.** `--k-center` is kept at a bit width if its mean val loss beats the plain cache at the same width by more
than the larger sd. K is "special in training" only if int8 W_k beats int8 W_v by more than the larger sd (else the
15.7 M rule "bytes, not where" holds at 260K). The test split is not read.

**Result: which variable hurts most** (`error_anatomy.py`, fp32 model, 3 seeds, 256 validation windows; loss rise,
no retraining; [results/v512d64l5kv4_error_anatomy_int4.json](results/v512d64l5kv4_error_anatomy_int4.json),
[_binary.json](results/v512d64l5kv4_error_anatomy_binary.json)):

| source, in every block | int4 weights / 4-bit cache | W1A8 / 2-bit cache |
|---|---|---|
| W_q | 0.025 ± 0.004 | 0.87 ± 0.08 |
| W_k | 0.027 ± 0.001 | 2.28 ± 0.21 |
| W_v | 0.165 ± 0.040 | 5.79 ± 1.57 |
| W_o | 0.032 ± 0.003 | 1.89 ± 0.20 |
| W_1 / W_3 | 0.071 / 0.071 | 4.01 / 4.03 |
| W_2 | 0.106 ± 0.003 | 4.54 ± 0.27 |
| **K cache** | **1.682 ± 0.114** | 3.45 ± 0.13 |
| V cache | 0.011 ± 0.000 | 0.44 ± 0.01 |

- The key CACHE is the one variable that breaks: at 4 bits it costs 1.68 nats, 160x the value cache (0.011) and 10x
  the worst weight matrix at 4 bits. The key WEIGHTS W_k are ordinary (like W_q). Among weights, W_v and W_2 hurt most.
- One-shot W1A8 destroys the 260K fp32 model whatever is switched (rises of 1-6 nats, residual error ~1): this column
  only ranks; the binary model is usable because it is trained binary.
- Do errors add? With small errors (int4) the seven weight errors combine like independent ones: joint residual error
  / root sum of squares = 0.98-1.01 at every block, not a linear sum (that would be up to sqrt 7 = 2.6x larger). The
  loss is 1.32x the sum of single rises (0.651 vs 0.495): the loss grows faster than the error (quadratic), as at
  15.7 M (2.2-2.5x).
- Does the residual stream pile errors up? An error made in block 0 alone is 2.00 ± 0.06x larger at the last block
  (block 1: 1.70, block 2: 1.51, block 3: 1.31): later blocks AMPLIFY an early error, they do not damp it. This
  supports Edén's hypothesis for the LLM (the sum residual carries every error forward and the next blocks react to
  it); the matching measure on ResNet-8's OR shortcuts is not done.

**Result: the key taken apart** (`k_anatomy.py`, 3 seeds, 256 validation windows, fixed means / scales from 64
training windows; [results/v512d64l5kv4_k_anatomy.json](results/v512d64l5kv4_k_anatomy.json)):

| layer (mean over key heads) | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| key energy in its mean, fp32 | 0.77 | 0.96 | 0.88 | 0.92 | 0.89 |
| key energy in its mean, binary | 0.41 | 0.74 | 0.63 | 0.85 | 0.75 |
| score variance from q . RoPE(mean), fp32 | 0.54 | 0.87 | 0.75 | 0.82 | 0.75 |
| score variance from q . RoPE(mean), binary | 0.22 | 0.58 | 0.51 | 0.71 | 0.57 |

- A key is mostly one fixed vector: its mean over tokens holds 41-96 % of its energy. Through RoPE that mean becomes
  a term that depends only on the key's POSITION, and it carries 22-87 % of the score variance: much of what the
  attention decides is "how far back", learned through the key mean. The content part is the small rest.
- A per-token scale must cover mean + content, so the content (small) is what few bits erase. Outlier channels are
  mild here (max / median channel RMS 2.3-4.9) and a smooth (per-channel) scale barely helps.

Key cache loss rise by storage (binary model; fp32 and ternary in the JSON, same order):

| storage | 4 bits | 3 bits | 2 bits | on a chip |
|---|---|---|---|---|
| token (now) | 0.100 ± 0.015 | 0.474 ± 0.013 | 0.803 ± 0.065 | as now |
| smooth | 0.088 ± 0.010 | 0.366 ± 0.046 | 0.690 ± 0.055 | cheap |
| hadamard | 0.090 ± 0.011 | 0.551 ± 0.047 | 1.198 ± 0.015 | cheap |
| pre (before RoPE) | 0.103 ± 0.040 | 0.306 ± 0.082 | 0.707 ± 0.110 | RoPE per key read |
| pre-center | 0.011 ± 0.005 | **0.039 ± 0.011** | **0.122 ± 0.022** | RoPE per key read |
| **center** | **0.009 ± 0.003** | 0.056 ± 0.017 | 0.128 ± 0.021 | + one dot per key (estimate) |
| center + hadamard | 0.007 ± 0.002 | 0.045 ± 0.011 | 0.189 ± 0.053 | + one dot per key |

- Storing key - mean (the mean rotated to the key's position, added back when read) cuts the cache damage 6-11x in
  the binary model (4 bits 0.100 -> 0.009; 2 bits 0.80 -> 0.13), and 30x in fp32 at 4 bits (1.68 -> 0.056). The
  usable 4-bit key cache needs no training at all with it. Rotations alone (Hadamard) do not fix it: the problem is
  the mean, not the outliers.
- Binary and ternary models are far less fragile than fp32 (4-bit token: 0.10 / 0.21 vs 1.68): their keys hold less
  of their energy in the mean (41-85 % vs 77-96 %).

**Result: training** (binary weights, step 4 recipe, 3 seeds, validation;
[results/v512d64l5kv4_k_train.json](results/v512d64l5kv4_k_train.json); the int8 key cache baseline is step 4's
2.0613 ± 0.0086):

| run | seeds 42 / 43 / 44 | val loss | vs int8 cache |
|---|---|---|---|
| key cache 2 bits, plain | 2.4176 / 2.4385 / 2.3452 | 2.4004 ± 0.0490 | +0.339 |
| key cache 2 bits, `--k-center` | 2.1314 / 2.0998 / 2.0991 | **2.1101 ± 0.0184** | +0.049 |
| key cache 4 bits, plain | 2.1227 / 2.1037 / 2.1061 | 2.1108 ± 0.0103 | +0.050 |
| key cache 4 bits, `--k-center` | 2.0851 / 2.0705 / 2.0659 | **2.0738 ± 0.0100** | +0.013 |
| int8 W_k (cache int8) | 2.0471 / 2.0250 / 2.0297 | 2.0339 ± 0.0116 | -0.027 |
| int8 W_v (cache int8) | 1.9615 / 1.9708 / 1.9948 | **1.9757 ± 0.0172** | -0.086 |

- `--k-center` passes the rule at both widths: 2 bits -0.290 (larger sd 0.049), 4 bits -0.037 (0.010). Trained with
  it, a 2-bit key cache is as good as a plain 4-bit one (2.110 vs 2.111) and the 4-bit one is within 0.013 of int8.
  Key cache bytes (computed, values only, scales unchanged): int8 82 KB, 4 bits 41 KB, 2 bits 20.5 KB.
- K is NOT special in training: int8 W_v beats int8 W_k by 0.058 (larger sd 0.017), the reverse of the "K special"
  test. And "bytes, not where" (the 15.7 M rule) does NOT hold at 260K either: the same bytes help 3x more in W_v
  than in W_k, as the no-training ranking said (W_v the worst weight at int4, 0.165 vs 0.027).

**Answer.** The variable that breaks the 260K model in few bits is the key CACHE, not the key weights; its problem
is the key's mean (a position term through RoPE) eating the per-token scale. Storing key - mean fixes most of it,
with or without training. Errors from different weights combine like independent ones, and an early error grows
through the residual stream (2x from block 0 to the last). For the weights, the value projection W_v is where extra
bits pay most. Not done: `--k-center` in intsim and the C (an extra dot of q with RoPE(mean) per key, estimate),
center combined with the learned temperature or ternary weights, the growth measure on ResNet-8's OR shortcuts.

### Step 7b: the key at two sizes, and what the literature says (Nov 20 2026, no training)

**Question.** Is the key's mean a 260K problem or does it stay when the model grows, and which published key fixes
match it?

**What.** `k_anatomy.py` on the 260K (head width 8) and the 15.7 M d384l8 (head width 64), fp32 / binary / ternary,
3 seeds, 256 validation windows, calibration from 64 training windows. New: `mean_slow` (share of the key mean's
energy in the RoPE pairs that turn less than once over the context: 260K 2 of 4 pairs, 15.7 M 19 of 32) and `sage`
(subtract the post-RoPE key mean, not added back: free, exact without quantization, SageAttention's K smoothing).
[results/v512d64l5kv4_k_anatomy.json](results/v512d64l5kv4_k_anatomy.json),
[results/d384l8_k_anatomy.json](results/d384l8_k_anatomy.json).

**Result** (loss rise of the key cache at 4 / 2 bits, mean ± sd over 3 seeds; mean_share / mean_slow = range over layers):

| model | mean_share | mean_slow | token 4b / 2b | sage 4b / 2b | center 4b / 2b | pre-center 4b / 2b |
|---|---|---|---|---|---|---|
| 260K fp32 | 0.77-0.96 | 0.87-0.92 | 1.68 / 3.45 | 0.44 / 2.91 | 0.056 / 1.03 | 0.062 / 0.72 |
| 260K binary | 0.41-0.85 | 0.79-0.88 | 0.100 / 0.80 | 0.045 / 0.66 | 0.009 / 0.13 | 0.011 / 0.12 |
| 260K ternary | 0.57-0.83 | 0.82-0.91 | 0.208 / 1.15 | 0.059 / 0.91 | 0.017 / 0.30 | 0.017 / 0.24 |
| 15.7 M fp32 | 0.42-0.74 | 0.64-0.84 | 0.0094 / 0.48 | 0.0049 / 0.43 | 0.0020 / 0.087 | 0.0018 / 0.082 |
| 15.7 M binary | 0.29-0.76 | 0.58-0.84 | 0.0082 / 0.20 | 0.0048 / 0.18 | 0.0014 / 0.056 | 0.0016 / 0.052 |
| 15.7 M ternary | 0.32-0.69 | 0.65-0.83 | 0.0085 / 0.23 | 0.0048 / 0.20 | 0.0019 / 0.059 | 0.0019 / 0.054 |

- The mean is there at both sizes (a third to three quarters of the key's energy at 15.7 M) and lives in the slow
  RoPE pairs (58-92 %): the "rotary offset features" of Jonasson 2025 and the low-frequency bias zones of the
  "massive spikes are bias vectors" paper. It is not a 260K artifact.
- What changes with size is the fragility: a plain 4-bit key cache costs 1.68 nats on the 260K fp32, 0.0094 on the
  15.7 M fp32 (~180x). Removing the mean (center / pre-center) cuts it 4-30x at both sizes.
- `sage` (free) helps 2-4x at 4 bits and almost nothing at 2 bits: subtracting a constant does not shrink the
  per-token scale enough. SageAttention's own table shows no gain on a Llama at INT8 either.
- Hadamard helps on 64-wide heads (15.7 M 4b 0.0094 -> 0.0036) and hurts on 8-wide ones (260K 0.100 -> 0.090
  binary, 2b worse): too few channels to spread an outlier.
- Literature read (PDFs): KVQuant (pre-RoPE per channel keys + zero point = our pre-center; first token kept in
  fp16), SageAttention (K = shared bias + small token signal), rotary offset features (the mean of q and k makes a
  position-only attention pattern in slow pairs), Bondarenko 2023 (outliers from "no-op" heads; clipped softmax /
  gated attention), HAD and BinaryAttention (1-bit Q and K: lose magnitude, fixed with standardization, a learned
  bias and distillation).

**Answer.** The key's mean is a general feature of RoPE models (it is the position term in the slow pairs) and
removing it before quantizing is the fix that works at both sizes, as in the literature. The small model is ~100x
more fragile to cache bits, so the limit search is a size question: where between 260K and 15.7 M does a plain
low-bit key cache become cheap. Not done: intermediate sizes (d128 / d256), first-token (sink) kept at int8.

### Step 8: a softmax that does not need the key's mean, and the size ladder (rule Nov 20 2026, before measuring)

**Question 1.** The key's mean builds a position-only score in the slow RoPE pairs: a learned "attention sink"
(rotary offset features; massive spikes are bias vectors). If the softmax can attend to "nothing" by itself, does the
model stop building that mean, and does the key then survive a plain low-bit cache? Until now only the softmax's
temperature was tried (step 6, a small loss gain); the softmax does not amplify key errors exponentially (KL = ½ Var,
section "Why keys break"), so a softmax change can only help by changing what the key has to carry.

**What 1.** `train.py --attn-sink` (one learned logit per head joins each softmax with a zero value; starts at 0 =
softmax with +1 in the denominator) and `--attn-gate` (sigmoid gate per head and token on the attention output, a
dim -> heads linear in the blocks' weight format; Qiu et al. 2025 report it removes attention sinks). Binary 260K,
same recipe as the base (lr 3e-3, 20,000 steps), 3 seeds, validation; then `k_anatomy.py` on each. C cost
(ESTIMATE): sink = one add to the softmax sum per head; gate = a 64 -> 8 linear and a sigmoid table per token.

**Rule 1 (written before the runs).** A change helps the key if (a) the plain per-token 4-bit key cache rise falls
below the base's by more than the larger sd (base binary 0.100 ± 0.015) and (b) the as-trained validation loss is not
worse than the base by more than 0.03 (base 2.0613 ± 0.0086). Also reported, not part of the rule: mean_share,
mean_slow, 2-bit rise, center with it.

**Question 2.** The plain 4-bit key cache costs 1.68 nats at 260K fp32 and 0.009 at 15.7 M. Where in between does
it become cheap, and how does the binary - fp32 gap move?

**What 2.** Only the width grows, everything else as the 260K (5 layers, 8 heads, 4 key heads, vocab 512, context
512, hidden = 2.6875 x dim, lr 3e-3, 20,000 steps): d128 (head width 16) and d256 (head width 32), fp32 and binary, 3
seeds, validation; `k_anatomy.py` on each. Head width grows with the width (as in 260K -> 15.7 M).

**Rule 2 (written before the runs).** The size limit is the smallest width whose binary model has a plain per-token
4-bit key cache rise below 0.03 nats (the "usable" bound of the KV cache study), mean over 3 seeds. Also reported:
binary and fp32 validation loss, the gap, center 4 / 2 bits.

### Step 9: a 1-bit key cache at 260K, for chips without int8 SIMD (rule Nov 20 2026, before measuring)

**Question.** On the classic ESP32 and the Pico 2 (no int8 SIMD), a 1-bit key makes the score q . k a sum of
plus / minus the query's 8 values. Can the 260K be trained with it? A no-training probe (LOGBOOK, Nov 20) found that
84 % of the key's sign bits only copy the sign of the key's mean (0.46 bit of information per stored bit; 0.98 after
subtracting the mean), and a 1-bit cache costs +1.29 plain vs +0.39 with center.

**What.** Binary 260K, base recipe (lr 3e-3, 20,000 steps), value cache as trained (8 bits), 3 seeds, validation:
`k1` (`--k-bits 1`: sign x mean|k| per token and head), `k1c` (+ `--k-center`), `k1tr` (`--k-bits 1 --attn-temp
--kv-recent 32`: the 15.7 M T2 recipe with the value at 8 bits), `k1ctr` (k1tr + `--k-center`), `k1uctr` (k1ctr +
`--k-unit`: the pure sign of key - mean, no per-token scale; the learned temperature carries the scale). Key cache
bytes per key head and position (computed): int8 8 + 4 (int32 scale) = 12, 2 bits 2 + 4 = 6, 1 bit 1 + 4 = 5,
`--k-unit` 1 (plus 32 int8 positions per head with the window). At head width 8 the per-token scale is most of a
low-bit key: only `--k-unit` removes it.

**Rule (written before the runs).** (1) Center helps a 1-bit key if `k1c` < `k1` and `k1ctr` < `k1tr`, each by more
than the larger sd. (2) A 1-bit key is usable if its 3-seed loss is within 0.03 of the base (2.0613 ± 0.0086, the KV
rule); if none is, the best is compared with the trained 2-bit center key (2.1101 ± 0.0184, step 7) at its bytes.
(3) `k1uctr` goes to intsim / the C / the chips only if it passes (2) or ties the 2-bit center key within the larger
sd.

## Future work

- The chip, faster. `../chip_kernels/psram_bw_bench.py` ran the same kernels with the data in PSRAM and in internal
  RAM: the output head (28 ms) and the attention scores are at the PSRAM's speed (58 MB/s), the linear layers are
  half kernel and half memory (69 MB/s internal, 40 MB/s from PSRAM), and the attention values were plain C (25 ms,
  now 13.2 ms on the PIE, exact). The int4 / int2 output layer is done (above; 1 bit rejected), the cache autoload
  too (section above: 10.96 tokens/s). Next: the PSRAM at 120 MHz with the autoload (+40 % bandwidth, experimental
  in ESP-IDF), a cheaper transposed cache write.
- Ideas from [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) (an int4 TinyStories model on the same chip,
  9.88 tokens/s on both cores, float; reviewed Nov 12): a 4-bit or 2-bit output head (done, above: measured 14.7 /
  8.4 ms, 9.52 / 10.13 tokens/s); per-layer embeddings (a large lookup table in
  flash, a few rows read per token) gave them +0.098 nats at vocabulary 32768 but only +0.025 at 4096, ours, so a
  PC check at most; `noinline` on the linear kernel (inlining at -O3 slowed their step from 95 to 155 ms).

- The ternary integer model and C; a binary model with as many bytes as the ternary one.
- Binarizing more is research: 1-bit activations were measured (above: +0.26 loss with the best recipe, one seed,
  no gain on the chip); a binary embedding and binary attention are not measured: an open question.
- The test split, once, for the final numbers.
