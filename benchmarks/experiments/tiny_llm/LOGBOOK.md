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
