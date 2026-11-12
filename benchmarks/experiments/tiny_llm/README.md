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

On the ESP32-S3 (seed 42, [`results/chip_s3_binary_d384l8_s42*.json`](results/)): **512 / 512 tokens with every
score equal and the story 120 / 120 tokens equal, with both kernels: exact.** With lib_symbex's kernel the story runs
at **8.28 tokens per second** (121 ms per token). A step costs more as the context fills (attention over more
tokens). Model load from flash 0.5-0.6 s; ~3 MB of PSRAM and ~246 KB of internal RAM stay free.

| Part of a step (ms, mean of 512 tokens) | `lut8` | `--linear symbex` |
|---|---|---|
| 1-bit linear layers (q, k, v, o, w1, w3, w2) | 115.1 | **55.6** |
| attention (rope + int8 cache 7.5, scores 7.5, softmax 2.1, values 25.2, division 1.6) | 43.8 | 43.9 |
| output layer (4096 x 384 int8) | 28.0 | 28.0 |
| norms, embedding, SwiGLU | 10.5 | 10.5 |
| **whole step** (position 0 / 255) | 197.3 (163 / 231) | **138.0** (104 / 172) |
| **story, tokens per second** | 5.55 | **8.28** |

The attention keeps the integers of `intsim.py` and only changes how they are computed: the KV cache is stored
head by head (a head's keys one after another), the weights of the values (p x vm < 2^32, checked; a larger one
falls back to int64) are summed as two 16-bit halves in int32, and the softmax index needs a 32-bit division only.

A story from the binary model (seed 42, greedy, from "Once upon a time"):

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and have fun. One day, Lily
> found a big, red ball in her room. She was very happy and wanted to show it to her friends. Lily went to her friend
> Tom and said, "Look at my big, red ball! Let's play with it!" Tom was excited and said, "Yes, let's play!" ...

## Answer

Yes: the 1-bit model writes stories with dialogue and a plot, the integer version loses nothing (1.5307 vs
1.5306), and the C equals it on every score, on the PC and on the ESP32-S3, where it writes 8.28 tokens per second
(one seed) with lib_symbex's own 8-bit x 1-bit kernel. It costs perplexity against fp32 (4.58 vs 3.67): the binary model
repeats itself more. Ternary sits between the two at twice the weight memory.

## Future work

- The chip, faster: the linear layers read their weights at ~32 MB/s with lib_symbex's kernel while the output
  layer reads PSRAM at ~55 MB/s; the attention values (25 ms) are plain C (a PIE version needs wider lanes than
  QACC's 20 bits) (the MB/s are computed from the measured times and the bytes read). Seeds 43 and 44 on
  the chip.
- The ternary integer model and C; a binary model with as many bytes as the ternary one.
- Binarizing more is research, not done here: 1-bit (or few-bit) activations for the linear layers (XNOR + popcount,
  the fully binary path of the other benchmarks), a binary embedding, binary attention. How much a model this small
  loses that way is not measured: an open question.
- The test split, once, for the final numbers.
