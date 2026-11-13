# rnn_probe: does a flipped bit grow step after step in a fully binary recurrent network?

> [!NOTE]
> A parallel, exploratory probe, **incomplete**: PC only, no C export, no chip run, no literature search yet. It is
> kept as evidence for the "limits" section, not as a supported feature.

PC study, validation only (subjects carved out of the training data; the test set is never read), seeds 42, 43, 44.
Full record, entry by entry: [LOGBOOK.md](LOGBOOK.md).

## Question

In a feed-forward binary network, flipping 1 % of a layer's bits changes few bits downstream (HAR at most 8 % of
the last layer; [`training_gap`](../training_gap/)). A recurrent network reuses the same weights at every step: does
the error grow without bound, die out, or settle? And what does that mean for running binary RNNs on a chip?

## What

- An Elman RNN on UniMiB-SHAR (the windows of [`conv1d/har_ultracompact`](../../conv1d/har_ultracompact/), read as
  151 steps of 3 axes, 17 classes), hidden 128, class from the last state. One BatchNorm shared by every step: one
  threshold per neuron, what a binary chip runs.
- Four kinds, one recipe (60 epochs, Adam 5e-3, cosine, batch 128): **binary** (±1 weights and state), **twin** (real
  weights, hardtanh state), **tanh**, **int8** (QAT: 8-bit weights and state).
- A learned input gain on the input term (an int8 first layer on a chip): without it the binary network does not
  train (14 % vs 38 %), because 3 input values are drowned by a 128-bit recurrent count.
- Measured: flip the sign of 1 % of the state (1 neuron) at step T/4 and count the state bits and predictions that
  change; the Derrida curve (0.78-20 % flipped, input on or off, trained and untrained); the share of neurons near
  their threshold; a linear probe on the state over time.
- Binary variants: a triangular straight-through gradient, sparse recurrence (each neuron hears K = 32 or 16 fixed
  neurons), and a margin loss that pushes each count at least M = 2 away from its threshold.

## Result

| kind | val accuracy | state bits different at the end | predictions changed |
|---|---|---|---|
| binary | 37.9 ± 5.3 % | 8.8 ± 1.1 % | **27.8 ± 3.1 %** |
| twin | 40.8 ± 5.1 % | ~0 % | 0.1 % |
| tanh | 35.0 ± 5.8 % | ~0 % | 0.2 % |
| int8 | 38.9 ± 8.3 % | ~0 % | 0.07 % |

1 % of the state flipped at T/4, mean of 3 seeds.

| binary variant | val accuracy | bits different at the end | predictions changed | probe at the last step |
|---|---|---|---|---|
| base | 37.9 ± 5.3 % | 8.8 ± 1.1 % | 27.8 ± 3.1 % | 37.2 ± 1.5 % |
| triangular gradient | 38.3 ± 1.6 | 8.5 ± 0.6 | 28.4 ± 1.3 | 36.0 ± 0.7 |
| sparse K = 32 | 33.1 ± 0.8 | 0.8 ± 0.3 | 2.8 ± 0.9 | 31.7 ± 1.6 |
| sparse K = 16 | 30.5 ± 2.7 | 0.0 | 0.1 ± 0.1 | 29.9 ± 1.3 |
| margin loss, λ 1 | 36.2 ± 1.9 | 2.7 ± 0.2 | **9.3 ± 0.5** | 36.4 ± 1.2 |
| margin loss, λ 3 | 27.4 ± 10.2 | 0.4 ± 0.4 | 1.6 ± 1.6 | 29.9 ± 3.0 |
| margin loss, λ 10 | 14.0 (collapsed) | 0 | 0 | 14.0 |

- **No explosion.** The damage grows for 5-20 steps, then settles at ~9 % of the state whatever the kick (0.78 % or
  20 %). The same network before training, with no input, is chaotic (49 %, fully decorrelated); training and the
  input hold it down.
- **But a binary state carries an error instead of shrinking it.** twin, tanh and int8 forget a flipped bit; binary
  keeps it, and 28 % of the predictions change. int8 reaches the same accuracy and is as robust as float.
- **Sparsity stops the spreading** by giving each neuron fewer damaged inputs. A simple count model (a neuron flips if
  it hears the one flipped neuron, K/128, and sits within 2 of its threshold, and the bit pushes the right way, 1/2)
  predicts the first step: K = 128 3.0 % (measured 2.78), K = 32 1.3 % (1.20), K = 16 0.78 % (0.72). It costs 5-7
  points of accuracy, and the state remembers less at the end. On lib_symbex unstructured zeros are not free (they need the ternary kernel's mask plane).
- **The margin loss at λ 1** cuts the changed predictions from 28 to 9 % at the same accuracy (within the noise) and
  costs nothing on the chip (training only). Larger λ makes training unstable or collapses it. These runs used the
  first version of the loss (`--margin-ref mixed`), whose distance is measured to neither threshold exactly, so
  "M = 2 counts" is approximate; the effect itself was measured with the inference threshold. Not rerun yet with the
  corrected default (`--margin-ref batch`).

## Answer

A fully binary RNN does not blow up; its damage settles. But it is fragile where int8 is not: at the same accuracy,
one flipped state neuron changes 28 % of its predictions (9 % with the margin loss) against 0.07 % for int8.
Binary recurrent networks stay out of the CDT and lib_symbex for now; this study is the "limits" evidence for the
paper. Open: margin loss + sparse together, λ between 1 and 3, word-structured sparsity, gated units (LSTM / GRU).

## Files

| result tag | command (from this folder, `../../env_bench/bin/python`) |
|---|---|
| `rnn_h128_{binary,twin,tanh}` | `rnn_probe.py --in-gain --kinds binary twin tanh --seeds 42 43 44` |
| `rnn_h128_int8` | `rnn_probe.py --in-gain --kinds int8 --seeds 42 43 44` |
| `rnn_h128_tri_binary` | `rnn_probe.py --in-gain --kinds binary --ste tri --tag _tri --seeds 42 43 44` |
| `rnn_h128_k32_binary`, `_k16_` | `rnn_probe.py --in-gain --kinds binary --sparse 32 --tag _k32 --seeds 42 43 44` (16: `_k16`) |
| `rnn_h128_m2_binary` | `rnn_probe.py --in-gain --kinds binary --margin-loss 2 --margin-ref mixed --tag _m2 --seeds 42 43 44` (λ 3e-3: no effect) |
| `rnn_h128_m2l1_binary`, `_m2l3_`, `_m2l10_` | same with `--margin-lambda 1 --tag _m2l1` (3: `_m2l3`, 10: `_m2l10`) |
| `rnn_h128_m2l01_binary` | `--margin-lambda 0.1 --tag _m2l01 --seeds 42` (one seed: a ranking) |

The first runs (binary / twin / tanh) predate the `--ste`, `--margin-loss` and `--sparse` options and do not record
them in their JSON; they used the defaults. No JSON records `margin_ref` (added Nov 12): every margin run used
`mixed`.
