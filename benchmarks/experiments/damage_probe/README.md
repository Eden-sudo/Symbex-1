# damage_probe -- which cheap measure predicts the damage that training leaves?

## Question

When one part of a network goes to fewer bits (the output layer, two blocks, the inputs of w2...), how much worse is
the model **after training**? A full training run answers it (25-35 min per run for tiny_llm), so choosing what to
quantize by training every option is slow. Which cheap measure, taken on a model that is already trained, ranks the
changes the way full training does?

The measures we had do not (tiny_llm/LOGBOOK.md): the CDT's per-layer cosine has Spearman 0.24 with the damage
without retraining, and that damage itself ranks the trained recoveries right only at the top (w2 first, o last
instead of before q k v).

## What

**The truth (already measured, no new long training):** 14 pairs of tiny_llm runs, A and B trained with the same
recipe, B = A with one part at fewer bits. Damage = validation loss of B - validation loss of A (`train.json`).

| Pair | A (trained) | change to get B | seeds |
|---|---|---|---|
| head_bin | binary base | output layer fp32 -> 1 bit | 42 43 44 |
| head_int2 | binary base | output layer + embedding fp32 -> int2 | 42 43 44 |
| head_int4 | binary base | output layer + embedding fp32 -> int4 | 42 43 44 |
| b07_bin | blocks 0, 7 int8 | blocks 0, 7 int8 -> 1 bit | 42 43 44 |
| b34_bin | blocks 3, 4 int8 | blocks 3, 4 int8 -> 1 bit | 42 43 44 |
| tern_bin | ternary | every block ternary -> 1 bit | 42 43 44 |
| fp32_bin | fp32 | every block fp32 -> 1-bit weights, 8-bit inputs | 42 43 44 |
| fp32_tern | fp32 | every block fp32 -> ternary weights, 8-bit inputs | 42 43 44 |
| acts_all | binary base (W1A8) | every block input 8 -> 1 bit (W1A1) | 42 43 44 |
| acts_w2 | W1A1 with w2 at 8 bits | w2 input 8 -> 1 bit | 42 43 44 |
| acts_o | W1A1 with o at 8 bits | o input 8 -> 1 bit | 42 |
| acts_qkv | W1A1 with q k v at 8 bits | q k v inputs 8 -> 1 bit | 42 |
| acts_w13 | W1A1 with w1 w3 at 8 bits | w1 w3 inputs 8 -> 1 bit | 42 |
| acts_w2_2to1 | W1A1 with w2 at 2 bits | w2 input 2 -> 1 bit | 42 |

**The candidates,** all computed on the trained A only (B's weights are never used):

1. **cos:** sum over the changed layers of (1 - cosine) between the layer's output before and after the change, same
   input (what the CDT's scan does, summed over the changed layers).
2. **loss0:** validation loss of A with the change applied, no retraining, minus A's loss (damage_map.py's measure).
3. **fisher:** ½ Σ g² Δy² per token, summed over the changed layers: Δy = change of the layer's output (same input),
   g = gradient of the loss with respect to that output (second-order Taylor estimate with the diagonal Fisher, as in
   BRECQ / HAWQ). The embedding lookup of int4 / int2 counts as one more changed layer.
4. **recover:** apply the change and fine-tune 500 steps (5 % of a full run; AdamW, peak learning rate 1.5e-4 = the
   end of train.py's schedule, 50 warm-up steps, cosine to 10 %); the same fine-tune of A unchanged is the control;
   recover = loss(B fine-tuned) - loss(A fine-tuned). Also recorded at 100 and 250 steps (not used by the rule).

Calibration batches for cos / fisher come from the TRAINING split; every loss is train.py's `evaluate` on the
validation split (the same windows as `train.json`). The test split is never read.

**Rule (written before measuring, Nov 16 2026):** a measure passes if its Spearman correlation with the truth is
**≥ 0.8 on each seed** (seed 42: 14 pairs, seeds 43 / 44: the 10 pairs with 3 seeds) and it costs **≤ 10 % of one
full run** (~3 min for tiny_llm). The cheapest passing measure is the one to try next on ResNet-8 (training_gap's
trained variants) before anything reaches the CDT. If none passes, that is the result.

Expectation written beforehand (to be checked, not a result): cos fails (as in tiny_llm), loss0 overrates the large
changes (fp32_bin, acts_all) that training recovers well, fisher is close to loss0, recover passes.

Limits known beforehand: 14 pairs, 4 of them single seed; the pairs mix weight and input changes, small and large.

    python damage_probe.py --seeds 42 43 44 --runs ~/Proyectos/Symbex1/benchmarks/experiments/tiny_llm/runs \
        --data ~/Proyectos/Symbex1/benchmarks/data/tinystories          # results/damage_probe.json

## Result

(running)

## Answer

(after the result)
