# tiny_llm logbook

Every step of the experiment in order: what was tried, what came out, what was decided. Numbers come from the files
in `results/` (named on each line); the README holds the summary. Validation split only; the test split is not read.

## Nov 12, 2026

### Model and integer version (commit 2a50399)
- Trained fp32 / ternary / binary (W1A8), seeds 42, 43, 44, same recipe (`results/train_*.json`). Validation loss:
  fp32 1.300 ± 0.002, ternary 1.425 ± 0.003, binary 1.522 ± 0.003.
- Integer model of the binary runs (`results/intsim_*.json`): 1.5307 vs float 1.5306; 98.57 % of next tokens equal.
- C on the PC (`results/c_check_pc_*.txt`): 1,024 / 1,024 tokens with every score equal, 3 seeds.

### On the ESP32-S3, seed 42 (all runs exact: 512 / 512 tokens, story 120 / 120)

| Commit | Change | Story, tokens/s |
|---|---|---|
| 5f087ab | firmware, `lut8` kernel for the 1-bit layers | 4.0 |
| 4a68d12 | lib_symbex PIE int8 dot for the output layer and the attention scores | 5.25 |
| a43788d (`results/chip_s3_binary_d384l8_s42.json`) | `lut8` with the faster attention below | 5.55 |
| d88977b | linear layers on lib_symbex's 8-bit x 1-bit kernel (`--linear symbex`) | 7.66 |
| a43788d | attention: head-major KV cache, int32 value sums, 32-bit softmax division | 8.28 |
| 4aad9e3 (`results/chip_s3_binary_d384l8_s42_symbex.json`) | attention values on the PIE (below) | **8.48** |

- **PSRAM or kernels?** (01e9d49, `../chip_kernels/results/psram_bw_bench_s3_80.json` and `_120.json`). The same
  kernels with their data in PSRAM and in internal RAM. The output layer and the scores run at the PSRAM's speed
  (58-59 MB/s, 452 / 348 MB/s internal). The 1-bit layers are half kernel and half memory (69 MB/s internal, 40 from
  PSRAM). The attention values were plain-C bound. PSRAM at 120 MHz: +40 % bandwidth, experimental (ESP-IDF only).
  The whole model does not fit in internal RAM (3.55 MB model + 1.5 MB KV cache, ~245 KB free).
- **Attention values on the PIE.** First version: 7-bit signed chunks, 5 passes: values 25.2 -> 15.6 ms, 8.37
  tokens/s. Final: V cache transposed and stored as value + 128, weights split into at most 4 unsigned bytes,
  `ee.vmulas.u8.accx`: values 13.2 ms; writing the transposed cache costs +2.2 ms (rope + cache 7.5 -> 9.7). Whole
  step 138.0 -> 128.5 ms.
- Seeds 43 and 44 on the chip, `--linear symbex` (`results/chip_s3_binary_d384l8_s4{3,4}_symbex.json`): exact
  (512 / 512, story 120 / 120), 8.48 tokens/s each, step 128.47 / 128.50 ms (seed 42: 128.49). 3 seeds: 8.48 ± 0.00.
- Review fixes (MendCode): no signed left shift in the values (01e9d49); the 4-vector scratch check only with the
  PIE values path, and psram_bw_bench exits nonzero on a mismatch (6e84cf0).

### Layer sensitivity (3ddaa6f, `results/layer_sensitivity.json`)
- Question: does the CDT's sensitivity scan (cosine per layer) point at the layers that hurt the model most?
- Real damage, one layer binarized (W1A8) in the fp32 model, no retraining, 3 seeds: output layer 0.513 ± 0.012
  (15x a block layer), block 0 1.24 ± 0.18, blocks 1-7 0.13-0.24, last block next worst. The top 10 is the same on
  the 3 seeds.
- The scan does not see it: Spearman with the real damage, CDT bin 0.24 ± 0.04, W1A8 cosine 0.06 ± 0.06. The W1A8
  cosine ranks the output layer as the safest.
- Damages do not add up: all blocks together 4.63 ± 0.24 vs 1.44 summed.
- Open: does the ranking hold after training? -> next entry.

### Training check of the most sensitive layers (running; `runs/ablation.log`)
- Same recipe as the binary base (10,000 steps, seeds 42, 43, 44), two variants (`train.py --head binary`,
  `--int8-blocks 0 7`). Not supported by intsim.py / the C yet.
- Binary base (`results/train_binary_d384l8_s*.json`): 1.5252 / 1.5219 / 1.5200.
- Results (`results/train_binary_d384l8_{head1,b07i8}_s*.json`), validation loss, 3 seeds:

| Variant | s42 | s43 | s44 | mean ± sd | vs base |
|---|---|---|---|---|---|
| binary base | 1.5252 | 1.5219 | 1.5200 | 1.5224 ± 0.0026 | |
| 1-bit output layer | 1.6038 | 1.6096 | 1.6078 | 1.6071 ± 0.0030 | +0.085 |
| blocks 0 and 7 int8 | 1.4219 | 1.4174 | 1.4223 | 1.4205 ± 0.0027 | -0.102 |
| (ternary, all blocks) | | | | 1.425 ± 0.003 | |

- The 1-bit output layer costs 0.085 even with training: rejected for now (it would save ~24 ms per token on the S3,
  ESTIMATE from the PSRAM bench).
- Blocks 0 and 7 at int8 reach the ternary model's loss, but with more bytes: linear weights ~4.87 MB (2 blocks x
  1.77 MB int8 + 6 blocks at 1 bit) vs ternary 3.54 MB and binary 1.77 MB (computed from the layer sizes).
- Control (`results/train_binary_d384l8_b34i8_s*.json`): blocks 3 and 4 at int8, same bytes: 1.4262 / 1.4292 /
  1.4262 = 1.4272 ± 0.0017, paired -0.095 ± 0.003. Blocks 0 / 7 minus blocks 3 / 4, paired: -0.0043 / -0.0118 /
  -0.0039 = -0.007 ± 0.005. Same sign on the 3 seeds but small: ~93 % of the gain is the extra bytes.
- Decision: no 1-bit head; mixed int8 blocks not worth it (ternary everywhere gives the same loss with fewer bytes).
  The damage ranking is right in sign, too weak to choose blocks; the CDT scan stays as it is.
- Queued after the control (session symbex1-1b, Edén's request): W1A1 probe, `train.py --acts 1` (1-bit inputs to
  every block's linear layer; output layer, norms and attention unchanged), same recipe, `runs/w1a1.log`.

### 2026-11-12: ideas from slvDev/esp32-ai (reviewed, not started)
- esp32-ai: TinyStories on the ESP32-S3, 0.56 M dense core in group-wise int4 (float math), 25 M per-layer embedding
  table in flash, 32768 x 96 int8 head staged in PSRAM, both cores, scalar dot products, 9.88 tokens/s; checked
  against PyTorch to 1e-5 (not bit-exact). Their head is PSRAM-bound (57.6 ms) like ours (28 ms).
- Worth testing later (Edén: "por el momento no"): `--head int4` / `int2` in train.py (3 seeds; to the chip only if
  the loss rises < ~0.02). ESTIMATE int4 head 28 -> ~14 ms per token.
- Probably not worth it: per-layer embeddings (+0.025 nats at vocabulary 4096 in their ablation), skipping padded
  vocabulary rows (we have none), LZ4-compressed weights (1-bit weights barely compress), dual core (our rule).
- Small checks: `noinline` on the linear kernel; RoPE cos/sin once per token.

### 2026-11-12: W1A1 result and the plan to make it train (plan only, nothing run)
- W1A1 (`--acts 1`, same recipe, `runs/binary_d384l8_a1_s*/train.json`): best validation loss 3.8965 (step 2500) /
  4.1432 (step 9500) / 3.4058 (step 10000) = 3.82 ± 0.37, vs W1A8 1.5224 ± 0.0026. Unstable across seeds; the
  stories repeat themselves. JSONs not copied to results/ (the W1A1 probe belongs to another session).
- Known weak points of the current `quant_act1`: identity straight-through gradient (never clipped), the sign
  threshold fixed at 0 (SwiGLU's output is not centered), training from scratch, labels only.
- Plan:
  1. Diagnosis, no training: on the saved W1A1 models, per layer, the share of positive inputs, the cosine between
     sign(x) x scale and x, and the gradient norm. Tells which remedy is needed.
  2. Screening, seed 42, ~3000 steps, validation, one change at a time, criterion fixed beforehand (best validation
     loss): (a) clipped STE (|x| <= 1) or ApproxSign (Bi-Real Net); (b) learned per-channel threshold and scale
     before the sign (ReActNet RSign, BiT elastic binarization); (c) two-step: start from the W1A8 checkpoint;
     (d) distillation from W1A8 / fp32 (BiT, BiBERT).
  3. The best 2 together, then 3 seeds, full recipe (10,000 steps).
- Literature check before any claim: is there any fully W1A1 generative language model that works? (BitNet keeps
  8-bit activations; BiT / BiBERT are BERT classifiers: my reading, to verify). If nothing gets close to W1A8,
  that is a "limits" result for the paper.
- Before running: reboot the PC (NVML "Driver/library version mismatch" after a driver update, Nov 12).

### 2026-11-12: W1A1 diagnosis (step 1, `w1a1_diag.py`, results/w1a1_diag.json)
- Saved models, 3 seeds each, 16 validation windows; W1A8 measured for reference. Mean over blocks and seeds,
  W1A1 | W1A8:

| input | share >= 0 | channel bias | cos sign(x) | cos with per-channel threshold (upper bound) | share \|x\| > 1 |
|---|---|---|---|---|---|
| attn_in (q k v) | 0.503 \| 0.498 | 0.300 \| 0.153 | 0.822 \| 0.807 | 0.849 \| 0.819 | 0.484 \| 0.110 |
| o_in | 0.496 \| 0.499 | 0.340 \| 0.162 | 0.779 \| 0.736 | 0.808 \| 0.751 | 0.887 \| 0.132 |
| ffn_in (w1 w3) | 0.502 \| 0.499 | 0.325 \| 0.139 | 0.828 \| 0.804 | 0.855 \| 0.815 | 0.723 \| 0.107 |
| w2_in (SwiGLU out) | 0.485 \| 0.499 | 0.385 \| 0.128 | **0.456 \| 0.450** | 0.580 \| 0.462 | 0.704 \| 0.051 |

- **Gradients explode at the bottom, same in the 3 seeds.** One training batch, seed 44: total gradient norm 216
  (W1A8: 0.39, ~550x), of which the embedding 208 and block 0's q / o / w2 ~15-29; blocks 1-7 0.01-0.7. With
  `clip_grad_norm_(1.0)` the whole update is divided by ~200, so blocks 1-7 learn ~200x slower than in W1A8 (they
  are starved, not unstable themselves). Relative gradient of block 0's q: 1.3-6.9 (W1A8 0.0001). Cause not
  isolated (identity straight-through through 8 blocks and the norms are suspects).
- **w2's input is the big information loss:** sign keeps only 0.46 of its direction (the other inputs ~0.8); the
  SwiGLU product is heavy-tailed. A per-channel threshold would recover at most 0.58.
- The sign threshold at 0 is fine on average (share >= 0 is 0.50); a learned per-channel threshold gains little
  (+0.03 in cosine) except on w2_in.
- Screening order (step 2) changed by this: (1) gradient fixes: straight-through clipped RELATIVE to the scale
  (|x| <= k x mean|x|; at |x| <= 1 it would cut 48-89 % of the gradients) or ApproxSign, and per-group clipping
  so the embedding cannot starve the blocks; (2) two-step from W1A8; (3) w2's input at 8 bits (all else W1A1);
  (4) learned threshold; (5) distillation.

### 2026-11-12: W1A1 screening (step 2; seed 42, 3000 steps, warmup 500, best validation loss; a RANKING)
- `train.py --acts 1` plus one option each (runs/binary_a1s_*_s42, log runs/w1a1_screen.log, ~7 min per run):

| option | best val loss | vs base |
|---|---|---|
| base (identity straight-through) | 4.731 (step 2500) | |
| `--ste clip --ste-k 2` | **2.233** | -2.50 |
| `--a8-inputs w2` (w2's input at 8 bits) | **2.332** | -2.40 |
| `--init-from` W1A8 (two-step) | 3.315 | -1.42 |
| `--learn-thr` | 3.651 | -1.08 |
| `--clip-groups` | 4.333 | -0.40 |

- The two causes of the diagnosis are confirmed: the gradient (clip relative to the scale) and w2's input. Two-step
  helps less than expected (with the identity straight-through the gradient still explodes from a good start).
- Running (runs/w1a1_combo.log): clip2 + w2a8, clip2 + two-step, and a W1A8 reference at the same 3000 steps.
- Combinations (seed 42, 3000 steps; runs/w1a1_combo.log):

| run | best val loss |
|---|---|
| clip2 + w2a8 | **1.981** |
| clip2 + two-step | 2.115 |
| W1A8 reference, same 3000 steps | 1.657 |

- At the same budget the gap to W1A8 falls from 3.07 (base) to 0.32 (clip2 + w2a8); the story is coherent again.
  Two-step adds nothing on top of the clipped straight-through. Next (step 3): 3 seeds x 10,000 steps of clip2 +
  w2a8, then clip2 alone (fully W1A1) for the paper (runs/w1a1_full.log).
- 3 seeds x 10,000 steps, clip2 + w2a8 (runs/binary_d384l8_a1_clip2_w2a8_s*): 1.9010 / 1.8983 / 1.8925 =
  **1.897 ± 0.004** (W1A8 1.5224 ± 0.0026: +0.375; W1A1 base 3.82 ± 0.37). Stable across seeds now.
- Pure W1A1 clip2, seed 42 (10,000 steps): 2.1661 (perplexity 8.72); seeds 43 / 44 running. w2's input at 8 bits is
  worth ~0.27 (a ranking until 3 seeds).
- Opinion (Nov 12): the clipped straight-through fixes the instability, but W1A1 stays far from W1A8 (perplexity
  ~8.7 vs 4.58) and the gap grows with steps (0.32 at 3000, 0.375 at 10,000). ESTIMATE, not measured: on the S3
  1-bit inputs would only speed up the linear layers (55.6 of 128.5 ms per token, half of it PSRAM-bound), ~20-25 %
  per token. For tiny_llm W1A8 stays the choice; W1A1 is a "limits" result.
- New option: `train.py --distill <teacher.pt> [--distill-alpha 0.5] [--distill-t 1]` (KL to the teacher's
  next-token distribution; the logged train loss is then the mixed loss, the validation loss stays cross-entropy).
- Night queue (/tmp/w1a1_night.py, log runs/w1a1_night.log, summary results/w1a1_night.json): screening 2 (seed 42,
  3000 steps, fully W1A1): k 1, k 3, clip2 + learned threshold, clip2 + distillation from fp32 / from W1A8; choice on
  validation (an add-on only if > 0.02 better); then 3 seeds x 10,000 of the chosen recipe, alone and with w2a8.
- 3 seeds x 10,000 steps, fully W1A1 clip2 (runs/binary_d384l8_a1_clip2_s*): 2.1661 / 2.1776 / 2.1796 =
  **2.174 ± 0.007** (perplexity 8.80; W1A8 1.522: +0.652; clip2 + w2a8 1.897: w2's input at 8 bits is worth 0.277).
- Screening 2 (seed 42, 3000 steps, fully W1A1): `--ste-k 1` **2.034** (k 2: 2.233, -0.20).
- `--ste-k 3` 2.767 (+0.53 vs k 2): the narrower the gradient window, the better so far (k 1 < 2 < 3).
- clip2 + `--learn-thr` 2.214 (-0.02 vs clip2: within noise, not an add-on by the > 0.02 rule).
- clip2 + distillation from fp32 (alpha 0.5, T 1) **2.174** (-0.06 vs clip2).
- clip2 + distillation from W1A8 2.182 (-0.05; fp32 teacher slightly better). Next by the queue's rule: k 1 +
  fp32 distillation screened, then the best of {k1, k2 + distill, k1 + distill} gets 3 seeds x 10,000.

### 2026-11-12 (night, auto-written by the queue; reviewed in the morning): W1A1 screening 2
- Fully W1A1 clip2, 3 seeds x 10,000: [2.1661, 2.1776, 2.1796]
- Screening (seed 42, 3000 steps, best val loss; clip2 baseline 2.2326): clip2 2.2326, k1 2.0337, k3 2.7671, clip2_thr 2.2135, clip2_dfp32 2.1742, clip2_dw1a8 2.1815, k1_dfp32 2.007
- Chosen on validation: k1_dfp32 (--ste clip --ste-k 1 --distill runs/fp32_d384l8_s{seed}/model.pt)
- k 1 + fp32 distillation **2.007** (k 1 alone 2.034: -0.03; gains roughly additive with k but small). Chosen on
  validation: k 1 + fp32 distillation -> 3 seeds x 10,000, fully W1A1, then with w2a8.
- Final k 1 + fp32 distillation, seed 42: best 2.0839 at step **3000** (k 2 alone: 2.1661 at step 10,000). Its
  validation loss stalls at ~2.09 from step 3000 on, while k 2 alone keeps falling to the end (2.317 at 3000 ->
  2.166). So the 3000-step screening favors recipes that learn fast early; k 1 (or the distillation) seems to cap
  what the model reaches later. To check after the night: k 1 vs k 2 vs distillation alone at 10,000 steps
  (which of the two causes the plateau), and an intermediate k (1.5).
- Seed 43: best 2.0699 at step 4000 (plateau again).
- Seed 44: 2.0866 (step 9000). 3 seeds x 10,000, fully W1A1, k 1 + fp32 distillation
  (runs/binary_d384l8_a1_k1_dfp32_s*): **2.080 ± 0.009** (perplexity 8.01; k 2 alone 2.174 ± 0.007: -0.094;
  W1A8 1.522: gap +0.558). Running: the same recipe with w2a8 (3 seeds).
- k 1 + fp32 distillation + w2a8, seed 42: 1.8167 at step 9500 (clip2 + w2a8 seed 42: 1.9010).
- Seed 43: 1.7971 (step 9500).
- Finals (3 seeds x 10,000, best val loss): k1_dfp32 [2.0839, 2.0699, 2.0866], k1_dfp32_w2a8 [1.8167, 1.7971, 1.8133]
- Seed 44: 1.8133. 3 seeds x 10,000, k 1 + fp32 distillation + w2a8 (runs/binary_d384l8_a1_k1_dfp32_w2a8_s*):
  **1.809 ± 0.011** (perplexity 6.10; clip2 + w2a8 1.897 ± 0.004: -0.088; W1A8 1.522: gap +0.287).

Night summary (3 seeds x 10,000 steps, best validation loss; results/w1a1_night.json):

| recipe | fully W1A1 | w2's input at 8 bits |
|---|---|---|
| identity straight-through (first try) | 3.82 ± 0.37 | |
| clip k 2 | 2.174 ± 0.007 (ppl 8.80) | 1.897 ± 0.004 (ppl 6.67) |
| clip k 1 + fp32 distillation | **2.080 ± 0.009** (ppl 8.01) | **1.809 ± 0.011** (ppl 6.10) |
| W1A8 reference | 1.522 ± 0.003 (ppl 4.58) | |

- k 1 + distillation gains ~0.09 in both columns; w2a8 gains ~0.27-0.28 in both rows (the two are additive).
- The fully W1A1 run with k 1 + distillation plateaus at step 3000-4000 (seed 44 best at 9000 but flat).
- Next (started after the night queue, seed 42, 10,000 steps, rankings): k 1 alone, k 2 + fp32 distillation,
  k 1.5 + fp32 distillation: which of the two causes the plateau (runs/w1a1_plateau.log).
- Plateau check, seed 42 (ranking): k 1 alone, 10,000 steps: **2.0433** (step 9000), better than k 1 + fp32
  distillation (2.0839, same seed). Its curve keeps falling slowly to the end (2.088 at 3500 -> 2.053 at 8500).
  So the distillation causes the plateau: it helps early (screening at 3000 steps) and hurts later. The queue's
  choice was right on its 3000-step rule but wrong for 10,000 steps; lesson: screen at the final length, or at
  least check the curve. If k 2 + distillation also plateaus, k 1 alone becomes the recipe (3 seeds needed).
- k 2 + fp32 distillation, 10,000 steps, seed 42: 2.1776 (k 2 alone 2.1661): no gain at the final length either.
  Distillation (alpha 0.5, T 1) is dropped. The k 1.5 + distillation run was stopped (no longer informative); queue
  replaced by: k 1 alone seeds 43 / 44 (3 seeds), then k 0.5 seed 42 (k 1 is the edge of the range so far)
  (runs/w1a1_k1.log).
- k 1 alone, seed 43: 2.0575 (step 9500).
- Seed 44: 2.0491. 3 seeds x 10,000, fully W1A1, clip k 1 alone (runs/binary_a1p_k1_s*): **2.050 ± 0.007**
  (perplexity 7.77): the best fully W1A1 recipe (k 2 2.174 ± 0.007, k 1 + distillation 2.080 ± 0.009; W1A8 1.522:
  gap +0.528). Running: k 0.5, seed 42.
- k 0.5, seed 42: best 2.396 at step 1500, then the training loss climbs back (2.33 -> 2.55) and settles at ~2.46:
  too little gradient passes, unstable. k 1 is the best of {0.5, 1, 2, 3}.
- Running: clip k 1 + w2a8, 3 seeds x 10,000 (completes the w2a8 column; runs/w1a1_k1w2.log).
- clip k 1 + w2a8, seed 42: 1.8158 (k 1 alone 2.0433: w2a8 worth 0.228; k 1 + distill + w2a8 1.8167).
- k 1 + w2a8 seeds 43 / 44 stopped (they only add the sd of a known result), moved to the end of the next queue.

### 2026-11-13: where is the rest of the W1A1 gap? (seed 42, 10,000 steps, clip k 1; RANKINGS)
- Known: train loss = validation loss in every model (no overfitting, all underfit); the straight-through
  estimator alone moves W1A1 3.82 -> 2.05 at the same capacity; w2's input (SwiGLU output, cos_sign 0.46, upper
  bound with a threshold 0.58) is ~half the gap (0.23-0.28 of ~0.53). Unexplained: the other ~0.26-0.30.
- New train.py options: `--w2-acts t|2` (w2's input ternary or 2 bits), `--ffn-act relu` (relu gate).
- Queue (runs/w1a1_where.log), baselines k 1 = 2.0433, k 1 + w2a8 = 1.8158:
  (1) one input at 8 bits at a time: q k v / o / w1 w3; (2) w2's input ternary, 2 bits, relu gate;
  (3) capacity: FFN 1024 -> 1536 for W1A1 and for W1A8 (does widening close the gap or help both alike?);
  then k 1 + w2a8 seeds 43 / 44.

### 2026-11-13: Gemini's suggestions, checked (abstracts / full text on arXiv)
- BitNet v2 (Wang, Ma, Wei, arXiv 2504.18415): H-BitLinear = online Hadamard before activation quantization, for
  the outliers of attention and FFN intermediate states. Correct, but its target is 4-bit activations (8-bit
  matches BitNet b1.58), NOT 1 bit.
- QuEST (arXiv 2502.05003): Hadamard on weights and activations + MSE-optimal fit + trust gradient estimator
  (mask |x_hat - x| <= s x alpha, s ~ 1.30 at 1 bit: like our clip at ~2.3 alpha). W1A1 trained at 30M / 50M
  non-embedding (C4 loss 3.945 / 3.791; plain STE 5.256 / 6.867). Correct. No BF16 number for the W1A1 gap.
- BitNet a4.8 (2411.04965): 4-bit inputs of attention / FFN, sparsified intermediate states at 8 bits. Correct.
- Activation spikes in GLU (2405.14428): spikes in the FFN of early and late layers, a few tokens. Correct; in our
  W1A1 model the w2 cosine is low in EVERY block (0.58-0.64), not only at the ends (block 0 is lower for q / o).
- Not checked yet: ReLU^2 + SubLN (BitNet b1.58 2B4T, "An extra RMSNorm"), StableQAT, ReSTE.
- Probe (hadamard_probe.py, no training, 16 validation windows), mean 1-bit cosine over blocks, sign vs Hadamard:

| model | q | o | w1 | w2 |
|---|---|---|---|---|
| W1A1 k 1 (seed 42) | 0.780 -> 0.799 | 0.730 -> 0.776 | 0.800 -> 0.799 | **0.615 -> 0.798** |
| W1A8 (seed 42) | 0.807 -> 0.798 | 0.736 -> 0.768 | 0.804 -> 0.798 | **0.451 -> 0.805** |

  After the rotation every input sits at ~0.798 = sqrt(2 / pi), the value for a Gaussian: the Hadamard turns the
  heavy-tailed SwiGLU output into a Gaussian-like one. Prediction checked (Gemini said ~0.78-0.80).
  Note: the W1A1 model had already adapted its w2 input (0.615 vs 0.451 in W1A8).
- New option `train.py --hadamard <layers>`. Queue reordered (runs/w1a1_where2.log): Hadamard on w2, on w2 + o,
  on every input first, then the rest. On the chip a 1024-point fast Hadamard is ~10 k additions per block and
  token on the 8-bit values before the sign (ESTIMATE, small next to the matrix products), weights rotated offline.
- Gemini's follow-up, checked: "cos 0.82 -> the noise has more than half the energy" is wrong (sqrt(1 - 0.82^2) =
  0.57 is the amplitude; the energy is 0.33). Right: the cosine does not measure how sensitive the next operation
  is; a 1-bit Gaussian has a ceiling of sqrt(2 / pi) with one scale per token; rotating only x (as train.py does)
  needs no weight rotation at export (correction of the note above). Hypotheses, NOT measured: q / k noise is
  amplified by QK^T and the softmax; "attention has a precision ceiling, the FFN has spikes". Falsifiable checks:
  Hadamard on every input should NOT recover the q k v at 8 bits gain; q k at 8 bits should give most of it, v
  at 8 bits little (both added after the queue: runs/w1a1_where3.log). Also compare with w1 w3 at 8 bits (more
  weights than q k v: 786 k vs 442 k).
- Result (1) q k v at 8 bits, seed 42: **1.8905** (k 1 alone 2.0433: -0.153; w2a8 -0.228).
- Hadamard on w2, seed 42: **1.9887** (k 1 alone 2.0433: -0.055; w2 at 8 bits: -0.228). The rotation raises the
  w2 cosine to 0.80 but recovers only a quarter of what 8 bits give: the cosine is not the whole story here either.
- Gemini on the Hadamard result, checked: "-0.055 is 8 sd, 100 % real" overstates it (0.007 is the 3-seed sd; two
  single-seed runs differ with sd ~0.010: ~5.5 sd, very likely real, still a ranking). 1 - 0.798^2 = 0.363 of the
  energy lost: right. "Loss is not linear in the cosine": matches q k v and Hadamard w2. Its contrast "Gaussian
  shape (Hadamard) vs a zero (ternary)" is good; added the missing cell, Hadamard + ternary w2
  (runs/w1a1_where4.log, after the queue).
- Hadamard on w2 + o, seed 42: **1.9375** (k 1 alone 2.0433: -0.106; Hadamard w2 alone -0.055, so o adds -0.051
  although its cosine only moved 0.730 -> 0.776). Still fully 1-bit inputs. Next: Hadamard on every input.
- o at 8 bits, seed 42: **1.8753** (k 1 alone 2.0433: -0.168; q k v at 8 bits -0.153, w2 at 8 bits -0.228). One
  layer (147 k weights' inputs, 1/3 of q k v) gives more than q k v together: the attention output is the second
  bottleneck after w2. Hadamard on o recovered -0.051 of these -0.168. Ranking (one seed).

### 2026-11-13: damage map without training (damage_map.py, seed 42, ~1 min per model)
- Question: can a no-training map replace the 25-min runs to find the bottleneck?
- W1A8 model, one input group binarized (delta val loss; base 1.532): w2 block 0 **+2.79** (all w2 +4.79), w13
  block 0 +0.30, qkv block 0 +0.13; middle blocks +0.03-0.09, block 7 second highest. All at once +5.44 vs sum of
  groups +8.23: the damages overlap, they do not add.
- W1A1 model, one group back to 8 bits: every cell gets WORSE (+0.001 to +0.83): the trained network expects
  sign(x) x mean|x|; an untrained switch is a distribution shift, not a recovery. Unusable.
- Against the trained runs (w2 -0.228 > o -0.168 > q k v -0.153): the map agrees on w2 first but puts o last
  (+0.62 vs q k v +1.42). It does not predict what training recovers, except the top one.
- Trained effects do not add either: w2 + o + q k v = -0.549 > the whole W1A1 - W1A8 gap (2.043 - 1.522 = 0.521).
  No single bottleneck: w2 is the largest, block 0 dominates, the rest overlap.
- Gemini's follow-up ideas, checked against our data: (1) only block 0 at 8 bits: supported by the damage map,
  added `--a8-blocks` to train.py, queued; (2) two 1-bit planes on w2 ~ the 2-bit w2 run, kept; (3) learned
  threshold already screened (-0.019 < 0.02 rule); (4) SubLN before w2 / o: a positive-gain norm does not change
  sign(x), so it can only matter together with Hadamard; (5) QuEST MSE scale: for sign() the MSE-optimal scale is
  mean|x|, what we use; clip s 1.30 lies between our k 1 and k 2 (fine-tuning at most).
- Queue trimmed (Eden agreed): runs/w1a1_where5.log = Hadamard everywhere, + block 0 all inputs at 8 bits,
  + block 0 w2 / o at 8 bits, w2 at 2 bits, seeds 43 44 of w2a8. Dropped: width 1536, w2 ternary, relu, q k / v,
  Hadamard + ternary.
- w1 w3 at 8 bits, seed 42: **1.8497** (k 1 alone 2.0433: -0.194). Per input group at 8 bits (one seed, a
  ranking): w2 -0.228 > w1 w3 -0.194 > o -0.168 > q k v -0.153. Every group recovers a large share; the four sum
  to -0.743 vs a whole gap of 0.521: the errors overlap, no single bottleneck. The no-training map ranked w1 w3
  third and o last: it does not predict the trained recovery beyond w2.
- Hadamard on every input (q k v o w1 w3 w2), seed 42, all inputs still 1 bit: **1.8720** (k 1 alone 2.0433:
  -0.171; Hadamard w2 + o -0.106). The best fully 1-bit model so far, as good as putting o (1.875) or q k v
  (1.891) at 8 bits, and 0.35 above W1A8 (1.522). Cost on the chip, ESTIMATE: one block Hadamard per input and token
  (additions only), not measured. Ranking (one seed); next the block-0 runs on top of it.
- Hadamard everywhere + block 0 with every input at 8 bits (blocks 1-7 fully 1-bit), seed 42: **1.7821**
  (Hadamard everywhere 1.8720: -0.090; k 1 alone 2.0433: -0.261). Better than w2 at 8 bits in all 8 blocks
  (1.8158) with 1 block of 8 at 8 bits; 0.26 above W1A8 (1.522). Best W1A1-family model so far. Ranking (one seed).
- Hadamard everywhere + block 0 with only w2 / o at 8 bits, seed 42: **1.8359** (Hadamard everywhere 1.8720:
  -0.036; block 0 all inputs at 8 bits -0.090). In block 0, q k v and w1 w3 carry more than half of the gain:
  protect the whole block, not two inputs. Ranking (one seed).

### 2026-11-13: integer model of the best W1A1 recipe (intsim.py, PC)
- intsim.py now handles --acts 1 (sign bits, scale 127 x mean|input| per token), --a8-blocks and --hadamard (integer
  Sylvester block, +-1 additions; 1 / sqrt(n) folded into the per-row multiplier). Loss only, no C export yet.
  W1A8 seed 42 unchanged (float 1.5316, integer 1.5318, agreement 0.9854: same as before the change).
- Hadamard everywhere + block 0 at 8 bits, seed 42, 1024 validation windows: float **1.7951**, integer **1.7948**
  (int8 embedding / output, int8 attention, integer norms, SiLU table). Going integer costs nothing in loss.
  Next-token agreement float vs integer only 85.1 % (W1A8 98.5 %): 1-bit signs near 0 flip between bf16 and exact
  integers, changing the top token without changing the loss. (1.7951 vs train.py's 1.7821: different windows.)
- w2 input at 2 bits (4 levels {-1.5, -0.5, 0.5, 1.5} x mean|x|), no Hadamard, seed 42: **1.8981** (k 1 alone 2.0433:
  -0.145; w2 at 8 bits -0.228; Hadamard w2 at 1 bit -0.055). Two bits on w2 recover ~64 % of what 8 bits give.
  Ranking (one seed).
- w2 at 8 bits (no Hadamard, k 1), 3 seeds: 1.8158 / 1.8056 / 1.8074 = **1.8096 ± 0.0054** (k 1 alone, seed 42, 2.0433). A result now (3 seeds); W1A8 1.522 ± 0.003.

### 2026-11-13: the best W1A1 recipe in C and on the ESP32-S3 (seed 42)
- model.bin version 2: 16 bytes before each linear layer's weights (int32 acts, had, 0, 0; keeps the 16-byte
  alignment the PIE needs). Version 1 files still load. tinylm.c: integer FWHT per block, 1-bit inputs packed as
  bits, 1-bit linear = n_in - 2 popcount(input XOR weight); hook TINYLM_XOR_ROWS -> lib_symbex
  symbex_xor_popcount_rows (PIE on the S3).
- PC: exact in every build (plain, REFERENCE, DOT_ROWS_C, symbex generic, symbex + DOT_ROWS_C), 1024 / 1024 tokens +
  story; the W1A8 model still exact.
- S3 N16R8, --linear symbex, plain C popcount: exact 512 / 512 + story 120 / 120, 3.395 tokens/s (294.5 ms).
- S3 with the PIE XOR popcount: exact 512 / 512 + story 120 / 120, **8.881 tokens/s** (112.6 ms per story token;
  W1A8 8.48 ± 0.00). Per token (ms): qkv 12.0, w1 w3 21.3, w2 9.3, o 4.2, output head 28.0, attention values 13.3,
  rope / cache 10.1, scores 7.8, swiglu 8.3, embedding / norms 5.4. Load 3,551,648 bytes, internal RAM free 236 KB.
  The 1-bit linears (46.8 ms) are no longer the bottleneck: head + attention (~59 ms) are, as in W1A8. One seed.

### 2026-11-13: output layer at int4 / int2 (queued first, runs/w1a1_where7.log; W1A1 seeds 43 44 moved after it)
- The output layer is the slowest single part on the S3 (28.0 ms of ~113 per token, PSRAM-bound) and half of
  model.bin (1.57 MB int8). train.py --head int4 (absmax per row, 15 levels) / int2 (four levels {-1.5, -0.5, 0.5,
  1.5} x mean|w| per row), W1A8 blocks, 3 seeds x 10,000, same recipe as d384l8_head1. Unlike --head binary, the
  embedding lookup uses the quantized table too (one table on the chip). Compare: int8 head (post-training) 1.522 ±
  0.003, 1-bit head 1.6071 ± 0.0030. Size ESTIMATE: head 0.79 MB (int4) / 0.39 MB (int2) + scales.
- Decision (Eden, Nov 13): tiny_llm keeps 8-bit activations (W1A8). W1A1 gives no real gain here: +5 % speed on the S3
  (8.88 vs 8.48 tokens/s, one seed), no flash saved (+896 bytes), +8.9 KB internal RAM (int64 Hadamard buffer;
  one token's activations are <= 1 KB), +0.26 loss. Valid for token-by-token LLMs, not for conv nets (whole maps).
- int4 output layer, seed 42: **1.5373** (same seed: float head 1.5252, +0.012; 1-bit head 1.6038). Ranking (one seed).
- int4 output layer, seed 43: **1.5408**.
- int4 output layer, seed 44: 1.5347. **3 seeds: 1.5376 ± 0.0031**; paired vs the float head (1.5252 / 1.5219 /
  1.5200): **+0.015 ± 0.003** (1-bit head +0.085). Half the head's bytes for a sixth of the 1-bit head's cost.
  Next: int2 (running); then int4 in intsim + C + S3 (speed of the head is an ESTIMATE until measured).

### 2026-11-13: int4 output layer in intsim, C and on the ESP32-S3 (seed 42)
- intsim.py: --head int4 runs (train.py's levels, e ~ emb_q x emb_m / 2^20 as for int8): float 1.546 = integer 1.546
  (1024 windows), next-token agreement 98.7 %. model.bin version 3 = version 2 + the int4 table packed (per 32
  columns, byte j = column j low nibble, column 16 + j high nibble): **2,765,216 bytes (W1A8 int8 head 3,550,752)**.
- tinylm.c: emb4 lookup, int4 head (hook TINYLM_DOT_ROWS4, plain C otherwise). PC exact (plain, DOT_ROWS_C emulating
  the PIE trick, symbex), 1024 / 1024 + story, for the int4 model, W1A8 and W1A1 alike.
- S3 kernel (chip/src/tinylm_esp.c): per 16 packed bytes ee.andq 0xF0 (high) and ee.vsl.32 by 4 + ee.andq (low),
  both 16 x value as int8, two ee.vmulas.s8.accx, sum >> 4. Exact 512 / 512 + story 120 / 120.
  **Output head 28.0 -> 14.7 ms; 9.52 tokens/s (105.0 ms per story token; W1A8 8.48)**, +12 %. Per token (ms): qkv
  14.4, w1 w3 24.9, w2 11.5, o 5.2, head 14.7, attention values 13.3, rope / cache 9.7, scores 7.8, swiglu 5.9,
  embedding / norms 4.6. Internal RAM free 235 KB, PSRAM free 3.82 MB. One seed (loss: 3 seeds, +0.015 ± 0.003).
- Seeds 43 / 44 on the S3 (intsim float = integer: 1.5486 / 1.5486, 1.5432 / 1.5430): exact 512 / 512 + story
  120 / 120 each. **3 seeds: 9.52 ± 0.00 tokens/s, head 14.68 ms, model.bin 2,765,216 bytes, exact on every seed.**
- int2 output layer, seed 42: **1.5756** (same seed: float head 1.5252, +0.050; int4 1.5373, 1-bit 1.6038). Ranking (one seed).
- int2 output layer, seed 43: **1.5670** (float head 1.5219, +0.045).
- Queue (Eden, Nov 13): W1A1 seeds 43 / 44 of the best recipe dropped (tiny_llm stays W1A8); hadall_b0a8 stays one seed (a ranking).
- int2 output layer, seed 44: 1.5700. **3 seeds: 1.5709 ± 0.0044; paired vs the float head +0.049 ± 0.003**
  (int4 +0.015 ± 0.003, 1-bit +0.085 ± 0.005). Head bytes ESTIMATE 0.39 MB vs int4 0.79 MB: int2 triples the cost
  for 0.4 MB more. Decision candidate: int4 head (measured on the S3, 3 seeds, exact). int2 not taken to the chip.

### 2026-11-13: int2 output layer in intsim, C and on the ESP32-S3 (3 seeds; Eden: "measure it anyway")
- intsim.py: --head int2 runs (train.py's levels: e ~ (2c + 1) x mean|w| / 2, emb_q = 2c + 1 odd -3..3, emb_m =
  mean|w| / 2 x 2^20). Float = integer: 1.5831 / 1.5833, 1.5779 / 1.5780, 1.5777 / 1.5776 (1024 windows); next-token
  agreement 98.6-98.7 %. model.bin version 4 = version 2 + the int2 table packed (per 64 columns, byte j = c of
  columns j, 16 + j, 32 + j, 48 + j in bits 1:0, 3:2, 5:4, 7:6): **2,372,000 bytes (int4 2,765,216; int8 3,550,752)**.
- tinylm.c: emb2 lookup, int2 head (hook TINYLM_DOT_ROWS2, plain C otherwise). PC exact (plain and DOT_ROWS_C
  emulating the PIE trick), 1024 / 1024 + story, every seed; int4 and W1A8 still exact.
- S3 kernel: per 16 packed bytes, crumb k to the top bits (ee.vsl.32 by 6 - 2k, ee.andq 0xC0 = 64 c as int8), four
  ee.vmulas.s8.accx, dot = 2 (sum >> 6) + sum x. **3 seeds exact 512 / 512 + story 120 / 120; output head 8.43 ms
  (int4 14.68, int8 28.0); 10.13 ± 0.00 tokens/s (int4 9.52, int8 8.48)**. Internal RAM free 234 KB, PSRAM free
  4.22 MB (int4 3.82). Trade-off: +6 % speed and -0.39 MB vs int4 for +0.034 loss (int2 +0.049 vs int4 +0.015).
