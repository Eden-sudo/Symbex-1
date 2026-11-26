# Synthesis: what makes a binary network lose, and what wins it back (Nov 20, 2026; updated Nov 26, 2026)

A summary of the measured results so far, and the conjectures they suggest. **Measured** means 3 seeds and validation
unless the text says otherwise; **conjecture** means not tested yet. Sources: this folder's README / LOGBOOK, and
the benchmark READMEs named in each row. Section 5 lists the experiments that would turn the conjectures into results.

## 1. The binary key cache: the damage can be mostly removed (measured)

260K binary model (d64, 5 layers, 8 heads of width 8, GQA 4), trained with the cache it is evaluated with. Base =
int8 key cache, loss 2.0613.

| key cache | plain | centered (key - RoPE(mean)) | damage removed by centering |
|---|---|---|---|
| 4 bits | +0.050 | +0.013 | 74 % |
| 2 bits | +0.339 | +0.049 | 86 % |
| 1 bit (sign x scale) | +0.798 | +0.081 | 90 % |
| 1 bit + temperature + int8 window of 32 (step 9) | +0.024 | +0.030 | center adds nothing: the window does the work |
| 1 bit, pure sign of key - mean + temperature + window (1 byte per head and position) | — | +0.033 | — |

- **Cause:** the key carries a large mean (41-96 % of its energy). It lives in the slow RoPE pairs, so it becomes a
  score term that depends only on position. 84 % of the plain 1-bit key bits only copy the sign of that mean
  (bit entropy 0.46 bit; 0.98 after centering).
- **Fix:** subtract the mean before quantizing and add it back exactly. Each head stores one vector, at no cost
  per token.
- **Training matters:** without training, the 1-bit centered cache costs +0.39; trained with that cache, +0.08.
- **What is left:** 0.03-0.08 nats at 1 bit, against 0.79 nats for binarizing the weights.
- **Correction (step 9, Nov 21):** with the temperature and an int8 window of the last 32 positions, a plain 1-bit
  key is already within +0.024, and centering adds nothing (+0.030). Center is the fix only when there is no window.
- **Conjecture:** the key cache is no longer the main problem of the binary LLM; the weights are.

## 2. Width helps, and other levers win part of it back without widening (measured)

| width (head width) | fp32 | binary | gap | plain 4-bit key, binary | KV cache at ctx 512 (computed) |
|---|---|---|---|---|---|
| d64 (8), 260K | 1.2709 | 2.0613 | 0.79 | +0.100 | 246 KB |
| d128 (16), 973K | 0.9677 | 1.4943 | 0.53 | +0.036 | 410 KB |
| d256 (32), 3.76 M | 0.7946 | 1.0985 | 0.30 | +0.021 | 737 KB |

- At about the same weight memory, a wide binary network beats a narrow fp32 one: binary d256 1.10 (~0.6 MB of
  weights, computed) vs fp32 d64 1.27 (1.04 MB).
- The KV cache grows with width. On chips without PSRAM the cache, not the weights, sets the largest model.
- Rule (Edén): no widening to chase accuracy. So the levers below matter.

Levers at fixed width (260K binary unless stated):

| lever | loss change | cost |
|---|---|---|
| ternary weights (-1, 0, +1) | -0.22 | 1.58 bits per weight |
| int8 W_v, everything else 1-bit | -0.09 (int8 W_k: -0.03) | +7 bits in one layer per block |
| recipe lr 3e-3, 20k steps | -0.08 | training only (fp32 gains alike: the gap stays) |
| attention gate per head | -0.027 | one 64 -> 8 layer per block (changes the architecture) |
| learned temperature per head | -0.010 (ternary -0.018) | 8 numbers per layer |
| int4 output head (15.7 M) | +0.015 vs int8 head, 9.52 tok/s on the S3 | smaller than int8 |
| attention sink, Hadamard on 8-wide heads, W1A1 | nothing or worse | — |

**Conjecture (estimate, not measured):** ternary + int8 W_v + gate together could bring the 260K gap from 0.79 to
about 0.45-0.5. The gains will probably not add up in full.

## 3. Binarizing a given model without changing its design

- **One-shot binarization does not work.** Damage from several layers together is 2.2-2.5x the sum of the
  single-layer damages (llm_layers.py). The CDT's one-shot cosine estimate does not predict the trained loss of the
  LLM (Spearman 0.24 / 0.06). The model must be retrained binary with the same architecture (QAT).
- **What helps without touching the design:**
  - a good QAT recipe;
  - ternary weights;
  - int8 / int4 in a few layers (W_v, output head);
  - the centered key cache;
  - low-bit V with an int8 recent window (15.7 M: 0.60 vs 1.67 MB, +0.013);
  - systems work such as the S3 cache autoload (9.52 -> 10.96 tok/s).
- **Changes the design:** the gate, the sink, and repeated blocks.
- **On the chip (measured):** the binary 260K runs 3.6-14.7x faster than the same net in fp32 on llama2.c, with
  11x less flash. The price is 0.79 nats.

## 4. Cross-benchmark pattern: binary loses where each sum has few terms (conjecture with support)

Fan-in = how many terms one neuron (or one dot product) adds. With 1-bit weights a short sum can take only a few
distinct values, so the sign throws away a larger share of the information. This agrees with known theory:
Anderson & Berg 2017 show that binarized vectors keep their direction only in high dimension.

| case (measured) | fan-in | binary gap |
|---|---|---|
| LLM d64 / d128 / d256 / 15.7 M (weights) | 64 / 128 / 256 / larger | 0.79 / 0.53 / 0.30 / 0.22 nats |
| ECG s / m / l (8 / 16 / 32 channels x width 9), balanced accuracy, base recipe | 72 / 144 / 288 | -3.9 / -0.7 / +2.6 points |
| HAR smallest vs largest net | small / large | -23 / -9 points |
| ResNet-8 first layer (3 channels x 9 = 27), int8 instead of binary | smallest in the net | +4.5 points |
| LLM key cache at 4 bits, by head width 8 / 16 / 32 | 8 / 16 / 32 | +0.100 / +0.036 / +0.021 |

Three families (LLM, conv1d, conv2d) point the same way. The fixes that worked without widening fit two causes:

1. **Few terms per sum.** Fixes add terms or add levels per term:
   - width;
   - ternary weights;
   - int8 exactly in the shortest-sum layer (ResNet-8 stem, HAR first layer);
   - 2-bit activations (HAR train fit +3..+14).
2. **A common component (a mean) dominates the signal**, so the sign spends its bits on it. Fixes remove it:
   - centering the key (90 % of the damage removed without a window; an int8 window of recent keys does the same);
   - BN thresholds in the conv nets.

**Against the pattern:**
- **HAR:** widening x2 did not close the gap. There the loss is generalization to new people: binary drops ~10
  points from validation to test, INT8 ~6. That is a different cause.
- **The LLM ladder is confounded:** widening changed the fan-in, the head width and the parameter count together.
- **ECG:** the larger net also has more total capacity, not only more fan-in.

The table compares different tasks and networks: it shows a correlation, not a proof.

## 5. Tests that would confirm or refute section 4 (C done Nov 26, A and B not started)

| test | cost | prediction if the pattern holds |
|---|---|---|
| A. per-layer damage vs fan-in, inside one network, from existing JSONs (ResNet-8 drop table, llm_layers.py) | no GPU, ~1 h (estimate) | lower fan-in -> more damage, within the same net |
| B. step 10: wider heads at the same parameter count (head width 8 -> 16 -> 32, fewer heads) | 9 runs | the key cache suffers less; loss about the same |
| C. deep-narrow vs shallow-wide at the same parameter count, fp32 and binary | 12 runs | fp32 prefers deep-narrow (MobileLLM); binary reverses or ties. **DONE Nov 26: confirmed, strongly** (README "Test C"): fp32 deep 1.2796 vs wide 1.3055, binary deep 2.1144 vs wide 2.0514 |

C is the strongest. It held (Nov 26, 3 seeds, validation): at 260K and a fixed parameter count, choose width over
depth for binary weights. On the chips without PSRAM the wide binary shape is also the fastest and needs the least
RAM (the deep shape does not fit the ESP32). Caveat: the non-binary token embedding is larger in the wide shape
(49K vs 25K values); not separated. Not yet checked at other sizes or on vision nets.

Test C on the chips (Nov 26, 2026; weights in flash, exact 100 %, README "Test C"):

| binary shape (260K budget, 3 seeds) | val loss | model.bin (flash) | KV cache int8 + scales (computed) | RAM free ESP32 / M33 / Hazard3 | tok/s ESP32 / M33 / Hazard3 |
|---|---|---|---|---|---|
| deep-narrow d48 l9 | 2.1144 | 96,480 B | 221 + 111 KB | does not fit / 152 / 145 KB | does not fit / 25.86 / 29.11 |
| base d64 l5 | 2.0613 | 93,120 B | 164 + 82 KB | 72 / 235 / 228 KB | 40.60 / 31.88 / 34.80 |
| shallow-wide d96 l2 | **2.0514** | 100,896 B | **98 + 49 KB** | **164 / 327 / 320 KB** | **46.49 / 41.12 / 41.92** |

What each shape gains (+) and loses (-) against the base:

| vs base | loss | flash | RAM free | speed |
|---|---|---|---|---|
| shallow-wide | + 0.0099 nats better (just above the larger sd, 0.0086) | - 7,776 B more (+8.4 %) | + 92 KB more on every core | + 1.15x ESP32, 1.29x M33, 1.20x Hazard3 |
| deep-narrow | - 0.0531 nats worse | - 3,360 B more (+3.6 %) | - 83 KB less (M33, Hazard3); ESP32 does not fit | - 0.81x M33, 0.84x Hazard3 |


## 6. Pending: place the findings in the literature and confirm them (not started)

The literature below is cited from memory. Before any of it reaches the README or the preprint, each citation must
be checked: read the PDF and quote the exact claim.

| our finding | prior work that seems to cover it (to verify) | what may be ours | still to confirm |
|---|---|---|---|
| the key has a large mean; centering fixes the low-bit key cache | SageAttention ("smooth K": subtract the mean over tokens); KIVI / KVQuant (key outlier channels, per-channel key quantization) | 1-bit, trained with the cache, binary weights | that SageAttention's smoothing is the same operation as ours (RoPE(mean), added back exactly) |
| 1-bit key cache as the pure sign | QJL 2024 (sign of a random projection of the key + its norm) | sign of key - mean, 1 byte per head, no per-token scale | how QJL's loss compares at the same bytes |
| the binary gap shrinks with size | BitNet, Spectra / TriLM | the ladder at microcontroller sizes | — |
| ternary much better than binary | BitNet b1.58 | — | — |
| first / last layer at higher precision | XNOR-Net and later BNN practice | measured on ResNet-8, HAR, LLM head | — |
| binary needs high dimension | Anderson & Berg 2017 | the same pattern in LLM, conv1d, conv2d | tests A / B / C (section 5) |
| wide binary beats narrow fp32 at the same memory | Spectra (large models) | microcontroller sizes, KV cache as the real limit | — |

Findings that look new (NO literature search done yet):

1. **Binary weights make the key cache more robust.** Plain 4-bit key, no retraining: fp32 260K +1.68 vs binary
   260K +0.100 (~17x). Weak point: our fp32 260K may be under-trained (the reason distillation was dropped). Step 12
   (Nov 25) trained both 4x longer (80000 steps): fp32 improved (1.2709 -> 1.2255) and its 4-bit key rise grew
   (+1.68 -> +2.18) while binary stayed (+0.100 -> +0.096): ratio 16.8x -> 22.9x, NOT under-training. Also check whether the effect holds at d128 /
   d256 (fp32 0.13 / 0.035 vs binary 0.036 / 0.021: the ratio shrinks to ~1.7-3.6x).
2. **"Bytes, not where" depends on size:** it held at 15.7 M and failed at 260K (int8 W_v gains 3x more than W_k).
   To confirm: the same W_v / W_k control at d128 / d256.
3. **Additivity depends on the statistic** (ResNet-8: accuracy 0.45, clipped 0.81-0.86, mean loss 0.92-0.98, median
   2.1-3.1). To confirm: the median for tiny_llm (never measured).
4. **Binary prefers width over depth** at 260K (test C, section 5: confirmed strongly Nov 26; embedding caveat).
   To confirm: other sizes (d128 / d256 budgets) and a vision net.

Order proposed (Edén decides): systematic literature search for 1-4 (OpenAlex / arXiv); test A (no GPU); then B /
C and the fp32 recipe check for finding 1.
