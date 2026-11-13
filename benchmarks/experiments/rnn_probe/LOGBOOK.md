# rnn_probe logbook: does an error grow step after step in a binary recurrent network?

Run locally on Nov 7, moved into the repo on Nov 12 (see the last entry); the summary is in [README.md](README.md).
Script: `rnn_probe.py`. Results: `results/rnn_h<H><tag>_<kind>_s<seed>.json` (`.pth` weights stay local, gitignored).
Logs: `runs/run_*.log` (local, gitignored).
Every number is validation (subjects carved out of the training data); the test set was never read.

## Question

In a feed-forward binary network, flipping 1 % of the first layer's bits changes few bits downstream (HAR max 8 %
of the last layer, ResNet-8 13.7 %, MLP 3.6 %; training_gap/small_gap_study). In a recurrent network the same weights
are reused at every step: does the error grow without bound (the "exponential" argument), die out, or settle?

## Setup

- Elman RNN on UniMiB-SHAR: each window read as 151 steps x 3 axes, 17 classes, hidden 128, class from the last
  state. One BatchNorm shared by every step (one threshold per neuron, what a binary chip runs).
- Kinds, one recipe (60 epochs, Adam lr 5e-3, cosine, batch 128, seeds 42 43 44):
  - binary: +-1 weights, +-1 state (sign, straight-through)
  - twin: real weights, hardtanh state
  - tanh: real weights, tanh state
  - int8: QAT, 8-bit weights (symmetric, per tensor), 8-bit hardtanh state; BatchNorm and input gain real
- Input gain (`--in-gain`): a learned per-neuron scale on the input term, started at sqrt(H). Used in every run
  below (see 2026-11-07, entry 2).
- Measurements:
  - flip: at t0 = T/4, the sign of 1 % of the state (1 neuron of 128 = 0.78 %) is flipped; we report the share of
    state signs that differ from the clean run afterwards, and the share of predictions changed.
  - Derrida curve: d0 = 0.78-20 % flipped at T/4; d after 1, 5, 20 steps and at the end; "driven" (input on) and
    "autonomous" (input = 0 from t0 + 1); for the trained network and for the same network before training.
  - margin2: share of neurons whose count is within 2 (one flipped input bit) / 4 of the threshold.
  - probe: a linear probe on h_t (how much the state knows at step t).

## 2026-11-07

### 1. First run: the binary network did not train

60 epochs, no input gain: binary val 14.4 % (chance ~6 %, majority class ~14 %), twin 36 %. Changing lr (5e-3, 1e-3)
or the sequence length (stride 5: 31 steps) did not help (12.9-16.2 %).
Cause: the input term is 3 values x +-1 weights, while the recurrent term adds 128 bits (sd ~11). The network
barely hears the input.

### 2. Fix: the input gain

With `--in-gain` (seed 42, 15 epochs): binary 39.7 % (151 steps), 42.8 % (31 steps); twin 36.4 / 49.8 %.
On a chip this gain is the int8 first layer SYMBEX already has.

### 3. Base run, 3 seeds (hidden 128, 151 steps, 60 epochs, input gain)

| kind | val accuracy | state bits different at the end (1 % flipped) | predictions changed |
|---|---|---|---|
| binary | 37.9 +- 5.3 % | 8.8 +- 1.1 % | **27.8 +- 3.1 %** |
| twin | 40.8 +- 5.1 % | ~0 % | 0.1 % |
| tanh | 35.0 +- 5.8 % | ~0 % | 0.2 % |
| int8 | 38.9 +- 8.3 % | 0.01 % (autonomous 0.09 %) | 0.07 % |

Binary Derrida curve (state bits different at the end, mean of 3 seeds):

| network | d0 = 0.78 % | d0 = 20 % |
|---|---|---|
| untrained, autonomous | 49.3 % (chaos: fully decorrelated) | 49.3 % |
| untrained, driven | ~35 % | ~35 % |
| trained, driven | 8.7 % | 9.5 % |
| trained, autonomous | 11.7 +- 5.6 % | ~12.6 % |

- Binary margin: 6.0 % of neurons within 2 of the threshold, 12 % within 4 (same in the 3 seeds).
- Reading: no exponential growth to 50 %. The damage grows for 5-20 steps (step 1: 0.78 -> 2.8 %, lambda_D ~3.6), then
  settles at ~9 % whatever the size of the kick (0.78 % or 20 %): a floor of the system, not of the perturbation.
  Training and the input both hold it down (input pinning / entrainment); the random autonomous network is chaotic
  as Derrida theory predicts.
- But it is bad for binary: the binary state cannot shrink an error, it carries it. twin, tanh and int8 forget it.
  1 % flipped changes 28 % of the predictions (feed-forward HAR: ~18 % conv, 3 % MLP).
- int8 reaches the same accuracy as binary (38.9 vs 37.9, within the noise) and is as robust as the float networks.

### 4. Variants of the binary network, 3 seeds (same recipe + input gain)

| binary | val accuracy | bits different at the end (driven) | autonomous | predictions changed | within 2 of threshold |
|---|---|---|---|---|---|
| base, K = 128 | 37.9 +- 5.3 % | 8.8 +- 1.1 % | 11.7 +- 5.6 % | 27.8 +- 3.1 % | 6.0 % |
| margin loss M = 2, lambda 3e-3 | 39.8 +- 1.9 | 8.9 +- 0.4 | 12.8 +- 6.7 | 28.0 +- 1.0 | 6.0 |
| STE triangular max(0, 1-\|x\|) | 38.3 +- 1.6 | 8.5 +- 0.6 | 8.6 +- 4.2 | 28.4 +- 1.3 | 6.0 |
| sparse K = 32 (fixed random mask on W_h) | 33.1 +- 0.8 | 0.8 +- 0.3 | 3.2 +- 1.6 | 2.8 +- 0.9 | 10.3 |
| sparse K = 16 | 30.5 +- 2.7 | 0.0 | 0.6 +- 0.1 | 0.1 +- 0.1 | 12.4 |

- Margin loss at lambda 3e-3: did not act (the share near the threshold did not move): not tested yet, not negative.
  Rerun with lambda 0.1 and 1 (seed 42 first).
- STE triangular: nothing.
- Sparsity stops the spreading, at a cost of 5-7 points of accuracy. On SYMBEX unstructured zeros are not free: the
  weights are packed in words, zeros need the ternary kernel's extra mask plane (or a word-structured sparsity).

### 5. Why sparsity works (measured on the saved weights, 3 seeds)

| K | Var(input term) / Var(recurrent term) | damage at step 1 (0.78 % flipped) | probe at t = 0 | probe at t = 150 |
|---|---|---|---|---|
| 128 | 0.13 | 2.78 % (lambda_D ~3.6) | 19.3 % | 37.3 % |
| 32 | 0.48 | 1.20 % (~1.5) | 20.5 % | 31.7 % |
| 16 | 0.87 | 0.72 % (~0.9, below 1: ordered) | 20.2 % | 29.9 % |

- A simple model: a neuron flips if it receives a damaged input (A) AND its count is near the threshold (B) AND the
  damaged bit pushes toward it (~1/2). ESTIMATE P(flip) ~ P(A) x P(B) x 1/2, with P(A) = 1 - (1 - 0.0078)^K:
  - K = 16: 0.118 x 0.124 x 0.5 = 0.73 % (measured 0.72 %).
  - K = 128: 0.63 x 0.060 x 0.5 = 1.9 % (measured 2.78 %: the model misses neurons hit by 2+ damaged bits, which
    count within 4).
  The sparse networks have MORE neurons near the threshold (12 % vs 6 %) and still do not spread damage: the number
  of damaged inputs a neuron receives matters, not its distance to the threshold.
- The input does NOT dominate in the sparse networks (an 8x stronger input was a hypothesis): the learned ratio
  goes from 0.13 (recurrence 7.7x stronger) to 0.87 (balanced).
- Memory is lost: the probe at the last step falls from 37.3 % (K = 128) to 29.9 % (K = 16). The sparse network
  knows less at the end of the sequence; that is where its accuracy goes.

### 6. Margin loss with a real weight (seed 42 only: a ranking, not a result)

| binary, K = 128 | val accuracy | within 2 / 4 of threshold | damage at step 1 | bits different at the end | predictions changed |
|---|---|---|---|---|---|
| base | 43.3 % | 6.0 / 12.1 % | 2.83 % | 8.97 % | 27.98 % |
| M = 2, lambda 0.1 | 40.1 | 5.7 / 11.4 | 2.53 | 6.86 | 22.36 |
| M = 2, lambda 1 | 38.2 | 4.0 / 8.5 | 1.71 | 2.88 | 9.26 |

- The margin loss acts from lambda ~1: fewer neurons near the threshold, less spreading (28 -> 9 % of predictions
  changed), for ~5 points of accuracy on this seed. Running: lambda 1 on seeds 43 44, lambda 3 and 10 on 3 seeds.

### 7. Margin loss, 3 seeds (M = 2, K = 128)

| binary | val accuracy | within 2 of threshold | damage at step 1 | bits different at the end | autonomous | predictions changed | probe at t = 150 |
|---|---|---|---|---|---|---|---|
| base | 37.9 +- 5.3 % | 6.0 % | 2.8 % | 8.8 +- 1.1 % | 11.7 +- 5.6 % | 27.8 +- 3.1 % | 37.2 +- 1.5 % |
| lambda 1 | 36.2 +- 1.9 | 4.0 | 1.7 | 2.7 +- 0.2 | 4.8 +- 2.4 | **9.3 +- 0.5** | 36.4 +- 1.2 |
| lambda 3 | 27.4 +- 10.2 | 2.0 | 0.8 | 0.4 +- 0.4 | 4.7 +- 3.8 | 1.6 +- 1.6 | 29.9 +- 3.0 |
| lambda 10 | 14.0 +- 0.0 | 0.1 | 0 | 0 | 0 | 0 | 14.0 (collapsed: majority class) |
| sparse K = 32 (for reference) | 33.1 +- 0.8 | 10.3 | 1.2 | 0.8 +- 0.3 | 3.2 +- 1.6 | 2.8 +- 0.9 | 31.7 +- 1.6 |

- lambda 1 is the best trade measured: predictions changed 28 -> 9 % with the accuracy within the noise of the base
  (36.2 +- 1.9 vs 37.9 +- 5.3) and the memory kept (probe at the end 36.4 vs 37.2). Unlike sparsity it costs nothing
  on the chip (only training changes).
- lambda 3: less spreading but training becomes unstable (one seed falls). lambda 10: the network stops learning.
- Still 9 % vs 0.07 % for int8 at the same accuracy: the margin narrows the gap, it does not close it.

## Open

- Margin loss + K = 32 together; lambda between 1 and 3; M = 4.
- Not done: structured (word-level) sparsity, a balanced K = 128 with a smaller input gain, LSTM / gated units.

## Decision (2026-11-07)

Closed for now. Recurrent networks stay out of lib_symbex and the CDT; the paper can carry them as a short
"scope and limits" section with these numbers (no explosion, a ~9 % floor; same accuracy as int8 but 28 % of
predictions change for 1 % of the state flipped, 9 % with the margin loss; int8 0.07 %). Before claiming that no
fully binary RNN runs on a microcontroller, do a real literature search (not done yet).


## 2026-11-12: moved into the repo

- Copied from `~/Proyectos/rnn_probe` to `benchmarks/experiments/rnn_probe`. Only change to the script: the path to
  `training_gap/small_gap_study.py` is now relative (it was `~/Proyectos/Symbex1/...`); nothing else, no rerun.
- Every 3-seed number in entries 3, 4 and 7 recomputed from the 34 result JSONs: all equal to this logbook.
- The JSONs of the first runs (binary / twin / tanh base) predate the `--ste`, `--margin-loss`, `--sparse` options and
  do not record them (they ran with the defaults). The file tag says which option a run used (README, table "Files").
- The literature search for binary RNNs on microcontrollers goes with the preprint review
  (`docs/preprint/related_work.md`).
