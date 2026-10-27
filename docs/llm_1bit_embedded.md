# A native 1-bit LLM on ESP32-class chips

Design note for [`benchmarks/edge_llm_probe/`](../benchmarks/edge_llm_probe/). **Nothing here has been measured on a
chip.** Figures are arithmetic from the model shapes or are labelled as unverified; the probe exists to replace them with
measurements.

## 1. Why a native 1-bit LLM fits SYMBEX

SYMBEX-1 runs networks whose weights are +-1 (one bit) with XNOR + popcount on the chip. A **natively trained** 1-bit
LLM (BitNet b1.58 style, or +-1 weights trained from scratch with low-bit activations; not a post-training quantisation
of an FP model) has exactly that weight format in every linear layer (attention Q/K/V/O, MLP up/down). Weights cost
1 bit, so a 15M-parameter model is about 1.8 MB, small enough for the memory of the cheap boards SYMBEX already targets.

Two things are *not* 1-bit and stay in higher precision: the token embedding / output head (kept at int8 here, tied) and
the activations (4 to 8 bits), plus the KV cache.

## 2. Where the engine does not fit yet

SYMBEX's XNOR/popcount path needs **1-bit activations** too. In an LLM they are not: the inputs of every linear layer
are 4 to 8 bit. In today's SYMBEX the layer whose input is multi-bit (the first layer, input uint8 x weights 1-bit) takes
a slower path than the binary-binary layers, and the repo owner reports that this first-layer path is about 96-98 % of
the inference time in the current benchmarks (not reproduced in this change; reproduce with `tools/fine_profile.py`).
In an LLM **every** layer would take that path.

So the kernel that matters is **activation (4-8 bit) x weight (1 bit)**. `kernels/onebit_gemv.c` holds the candidates, all
portable C99 and bit-exact against the reference in `tests/test_edge_llm_kernels.py`:

| Kernel | Idea | Cost shape |
|---|---|---|
| `ref` | sum of +-x, one branch/multiply per weight | correctness oracle |
| `mask` | y = 2*S - sum(x), S = sum of x where bit = 1 | roughly the current multi-bit-input path |
| `lut4` | T-MAC style: per 4 activations build a 16-entry table of all +-sums once per vector; each row then does one lookup per 4 weights | 1 table lookup per 4 weights; table build amortised over the rows |
| `bitserial4` | 4-bit activations as 4 bit-planes; y = 2*sum_b 2^b * popcount(A_b & W) - sum(a) - 8*rowsum | 4 AND+popcount per 32 weights; **SWAR popcount** because Xtensa has no popcount instruction |
| `xnor` | 1-bit activations, y = N - 2*popcount(x ^ w) | lower bound, not usable for an LLM |

`lut4` and `bitserial4` are the candidate fixes; the probe measures cycles per weight of each from SRAM, PSRAM and flash.
Whichever wins on the S3 then needs a PIE (128-bit SIMD) version (phase 2 below).

## 3. Token rate

Every weight is read once per token (batch 1). With `P` weights (the 1-bit parameters), `B` = bytes read per token
(`P/8` plus the int8 head and embedding row), `BW` = measured sequential read bandwidth of the tier that holds the
weights, `cpw` = measured cycles per weight of the best verified kernel, `f` = CPU clock:

```
t_mem   = B / BW
t_cpu   = P * cpw / f
s/token = max(t_mem, t_cpu)    optimistic: reads overlap compute perfectly
s/token = t_mem + t_cpu        pessimistic: no overlap
```

Memory tiers the probe measures (capacities are from board datasheets; bandwidths are **pending: run on hardware**):

| Tier | Typical capacity | Used for | Bandwidth |
|---|---|---|---|
| internal SRAM | 520 KB (ESP32), 512 KB (S3), under 300 KB usable | activations, LUT, scratch, hot layers | pending: run on hardware |
| PSRAM | 4 MB (ESP32-CAM, mapped), 8 MB (S3 N8R8) | weights, KV cache | pending: run on hardware |
| flash, memory-mapped | 4 MB (ESP32-CAM), 8-16 MB (S3) | weights (read-only) | pending: run on hardware |
| SD card (SDMMC) | GBs | weights of models too large for the rest | pending: run on hardware |

PSRAM and flash go through the data cache, so the probe times them cache-cold (a weight matrix is not reused within a
token). `parse_log.py` applies these formulas to the measured numbers and labels the result PROJECTION. Attention
compute, KV-cache traffic and the head's compute are not in it.

## 4. Arithmetic available per chip

From datasheets and architecture knowledge, **not verified in this change**; check against the technical reference
manual of each chip before relying on a row.

| Chip | Core, clock | SIMD | Popcount | Memory | Consequence |
|---|---|---|---|---|---|
| ESP32 | Xtensa LX6, 240 MHz, 2 cores | none; 32-bit MUL, MAC16 | none (SWAR) | 520 KB SRAM, external SPI PSRAM (4 MB on ESP32-CAM) | scalar kernels; `lut4` avoids multiplies and popcounts |
| ESP32-S3 | Xtensa LX7, 240 MHz, 2 cores | PIE, 128-bit | no scalar popcount; whether PIE gives a byte-popcount or table-lookup path is to be checked in the TRM | 512 KB SRAM, octal PSRAM (up to 8 MB or more), flash up to 16 MB | main target: a PIE `lut4`/`bitserial4` is the phase-2 work |
| RP2040 | Cortex-M0+, 133 MHz, 2 cores | none | none | 264 KB SRAM, XIP flash, no PSRAM on most boards | too little RAM for KV cache; flash-resident model only |
| RP2350 | Cortex-M33 at 150 MHz (or Hazard3 RISC-V), 2 cores | DSP extension (packed 8/16-bit) on M33 | the Hazard3 core lists bit-manipulation extensions that include popcount (to be verified); M33 none | 520 KB SRAM, optional PSRAM | worth a probe port |
| Pi Zero 2 W / Pi 4 / Pi 5 | Cortex-A53 / A72 / A76, 1 to 2.4 GHz, 4 cores | NEON; A76 is ARMv8.2 | NEON vector byte-count (CNT) | 512 MB to several GB of RAM | a different class: runs much larger models, not comparable to the MCU numbers |

The Pi rows are included to bound expectations, not as targets.

## 5. Model sizing

~4k BPE vocabulary (TinyStories needs few distinct tokens), tied int8 embedding / head, 1-bit linear layers. Linear
parameters = `12 * d_model^2 * layers` (4 attention matrices + 8 d^2 for a 4x MLP). KV cache at int8 = `2 * layers *
d_model` bytes per token.

| Model | d_model, layers | 1-bit linear weights | 1-bit size | int8 embedding / head (4096 x d) | KV per token | KV at 512 tokens |
|---|---|---|---|---|---|---|
| 15M | 384, 8 | 14.2 M | 1.77 MB | 1.57 MB | 6 144 B | 3.1 MB |
| 30M | 512, 9 | 28.3 M | 3.54 MB | 2.10 MB | 9 216 B | 4.7 MB |
| 110M | 768, 15 | 106 M | 13.3 MB | 3.15 MB | 23 040 B | 11.8 MB |
| 200M | 1024, 16 | 201 M | 25.2 MB | 4.19 MB | 32 768 B | 16.8 MB |

(`parse_log.py` uses these shapes, rounded to 15M / 30M / 110M / 200M parameters.) The int8 head is read once per token
and is as large as the 1-bit weights for the 15M model, so it counts as bandwidth: `B` for 15M is about 3.3 MB, not 1.8 MB.
The KV cache belongs in PSRAM (it does not fit SRAM), which shares the PSRAM bus with the weights when those are there too.

## 6. Storage plan

- Up to roughly 100M 1-bit parameters (about 12.5 MB) fit in a 16 MB flash and are read through the memory-mapped
  window: no copy into RAM. The probe leaves 3 MB for the application; the 4 MB ESP32-CAM flash holds only the small model.
- A model that fits PSRAM (8 MB on the S3 N8R8, less the KV cache and 1 MB reserve) can be copied there at boot if PSRAM
  bandwidth beats flash.
- The SD card is only for models that fit nowhere else (110M to 200M and above). Every token streams the whole model
  from the card, so the rate is `card MB/s / model MB`: seconds per token, a demo, not interactive use.

## 7. Prior art to check

Ports of llama2.c to ESP32 and ESP32-S3 exist (the maintainers' repositories are public), as do BitNet reference
implementations and T-MAC-style LUT kernels for CPUs. They must be **looked up and read before the training phase**:
what model sizes, which precision, and what they report on which board. No figures from them are quoted here because none
have been verified.

## 8. Roadmap

1. **This probe:** memory bandwidth per tier, cycles per weight per kernel and location, energy per phase, on ESP32-S3
   and ESP32-CAM. Output: the first real table to replace sections 3 and 6.
2. **Kernel:** a `lut4` / `bitserial4` kernel in `lib_symbex`, plus a PIE version on the S3, verified bit-exact against
   the reference.
3. **Model:** train a 15-30M 1-bit TinyStories model with a ~4k tokenizer; check perplexity against an FP baseline of
   the same shape.
4. **On-chip:** run it from PSRAM or flash; measure tokens/s and mJ/token (board-level, with an idle baseline).
5. **Demo:** a larger model from SD at seconds per token.

## 9. Kill criteria

Target for the 15M model: at least 1 tok/s (acceptable), 2 tok/s (comfortable), 5 tok/s (good). The time budget is split
evenly between memory and compute (the pessimistic sum); the table uses `B` = 3.34 MB, the rounded `P` = 15M and
`f` = 240 MHz on one core:

| Target | Time per token | Memory share: BW needed (3.34 MB) | Compute share: max cycles per weight (15M) |
|---|---|---|---|
| 1 tok/s | 1.0 s | 6.7 MB/s | 8 |
| 2 tok/s | 0.5 s | 13.4 MB/s | 4 |
| 5 tok/s | 0.2 s | 33.4 MB/s | 1.6 |

Re-plan if, on the S3:

- the best **verified** kernel, cache-cold from the tier that will hold the weights, exceeds **8 cycles/weight** (it
  alone uses the whole 1 tok/s budget of the 15M model on one core; the limit is about 16 if memory time is ignored, and
  the 2-core kernel the probe also measures may lower the measured value), or
- cache-cold sequential read bandwidth of that tier is below **~7 MB/s** (the same 1 tok/s floor), or
- for the 30M model (28.3M linear weights, `B` = 5.64 MB) the 1 tok/s limits are about 4 cycles/weight and 11 MB/s.

Thresholds such as "more than 3 cycles/weight or PSRAM under 20 MB/s" are tighter than 1 tok/s requires: for 15M,
3 cycles/weight is about 5 tok/s from compute alone and 20 MB/s about 6 tok/s from memory alone, so they mark the 5 tok/s
target, not the 1 tok/s floor. Also re-plan if no kernel passes the on-chip bit-exact check, or if the measured
board-level mJ/token is no better than the same model on a Pi Zero 2 W (to be measured, not assumed).
