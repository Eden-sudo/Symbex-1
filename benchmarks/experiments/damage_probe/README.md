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

`results/damage_probe.json`, 34 pair x seed rows, Nov 16-17 2026, one RTX 4000 Ada laptop GPU.

Spearman correlation with the truth (rule: ≥ 0.8 on each seed):

| Measure | seed 42 (14 pairs) | seed 43 (10) | seed 44 (10) | seed 42, the same 10 pairs | time per pair |
|---|---|---|---|---|---|
| cos | 0.670 | 0.867 | 0.818 | 0.903 | 13-20 s (with loss0 and fisher) |
| loss0 | 0.618 | 0.927 | 0.952 | 0.927 | |
| fisher | 0.007 | 0.200 | 0.248 | 0.188 | |
| **recover** (500 steps) | **0.815** | **0.988** | **0.976** | 0.964 | 210-226 s + one control per model A |
| recover at 100 / 250 steps (not in the rule) | 0.745 / 0.802 | 0.976 / 0.976 | 0.976 / 0.976 | | ~1/5, ~1/2 of that |

Truth and measures per pair (mean over the seeds the pair has):

| Pair | seeds | truth | cos | loss0 | fisher | recover |
|---|---|---|---|---|---|---|
| acts_all | 3 | 0.528 | 9.23 | 5.60 | 0.74 | 1.024 |
| acts_w2 | 3 | 0.240 | 3.00 | 2.44 | 0.17 | 0.353 |
| fp32_bin | 3 | 0.222 | 7.93 | 4.62 | 0.77 | 0.352 |
| acts_w13 | 1 | 0.194 | 1.64 | 0.38 | 0.03 | 0.140 |
| acts_o | 1 | 0.168 | 1.44 | 0.73 | 0.17 | 0.230 |
| acts_qkv | 1 | 0.153 | 2.35 | 2.22 | 0.11 | 0.189 |
| acts_w2_2to1 | 1 | 0.145 | 0.63 | 0.09 | 0.03 | 0.047 |
| fp32_tern | 3 | 0.125 | 4.29 | 3.51 | 0.69 | 0.190 |
| b07_bin | 3 | 0.102 | 1.91 | 2.37 | 0.63 | 0.192 |
| tern_bin | 3 | 0.097 | 6.03 | 0.52 | 0.54 | 0.133 |
| b34_bin | 3 | 0.095 | 1.75 | 1.67 | 0.51 | 0.155 |
| head_bin | 3 | 0.085 | 0.09 | 0.54 | 2.27 | 0.101 |
| head_int2 | 3 | 0.049 | 0.10 | 0.19 | 0.72 | 0.041 |
| head_int4 | 3 | 0.015 | 0.01 | 0.015 | 0.016 | 0.005 |

- **Cost:** a full tiny_llm run takes 22.6-33.5 min (torch.compile on), so 10 % is 136-201 s. recover took 210-226 s
  per pair (no compile, three validation passes), plus the same again once per model A for the control: 11-17 % of
  a full run without the control, more with it.
- fisher fails everywhere: the output layer at 1 bit gets the largest estimate (2.27) and is one of the smallest
  damages (0.085). The second-order estimate breaks on large output changes (logits at 1 bit).
- loss0 and cos are good on the 10 pairs with 3 seeds (0.82-0.95) and fall on seed 42's four extra pairs, the inputs
  of one group in a W1A1 model. Among those five input pairs (one seed, a ranking) no measure gets the order:
  truth w2 > w1 w3 > o > q k v > w2 2->1 bit; recover w2 > o > q k v > w1 w3 > w2 2->1, loss0 w2 > q k v > o > w1 w3
  > w2 2->1 (Spearman 0.7, 0.6, cos 0.7).
- recover overestimates large changes (acts_all 1.02 vs 0.53) and underestimates w2 2->1 bit (0.047 vs 0.145), but
  keeps the order.

## Answer

**No measure passes the rule as written.** recover (a 500-step fine-tune) is the only one that ranks the changes the
way full training does on every seed (0.815 / 0.988 / 0.976), but it costs 11-17 % of a full run, over the 10 %
limit. The cheap ones (cos, loss0) rank the coarse choices right (output layer vs blocks vs inputs, 0.82-0.95 on
the pairs with 3 seeds) and fail on the fine one (which input group to keep at 8 bits); fisher does not work here.

What it means for the tools: the CDT's cosine is fine for coarse decisions and must not be trusted for fine ones;
the fine-tune check is the measure to make cheaper and then test on ResNet-8 (new data, the same rule).

Limits: 14 pairs, the five input pairs single seed; several truths differ by less than the seed spread (b07 / b34 /
tern_bin 0.095-0.102); the 100-step curve was recorded but is not part of the rule, so it is a lead, not a result.

---

# Part 2 -- per-layer accuracy table: how much does each layer of the binary ResNet-8 cost?

## Question

The binary ResNet-8 is ~13 points below its real-valued twin (training_gap: twin 83.6, binary 70.0 % validation).
**Which layers lose that accuracy?** A table with one row per layer, in two directions (as AIMET's QuantAnalyzer does
for INT8; Larq has nothing like it): the twin with only that layer binary (drop), and the binary network with only
that layer back to real values (recovery). Can that table, measured in seconds without training, be trusted to choose
which layer to give more bits?

## What

**Networks (already trained, no new long training):** training_gap's `twin_s<seed>` (real weights, hardtanh) and
`binary_s<seed>` (1-bit weights and activations) of the same graph (thermometer stem, OR shortcuts, binary
classifier), seeds 42 43 44.

**Layers (10):** stem, the six block convs b0..b5, the two 1x1 shortcut convs ds2 and ds4 (blocks 2 and 4, stride 2),
the classifier fc. **Parts:** the layer's weights (w), its output activation (a: sign vs hardtanh after its BatchNorm),
both (wa). fc has weights only.

**The table (one shot, no training):**
- drop = accuracy(twin) - accuracy(twin with only that part binary)
- recovery = accuracy(binary with only that part real) - accuracy(binary). Real weights = the binary network's own
  latent weights.

After every change the BatchNorm statistics are recomputed on 10,000 fixed training images (1-bit weights are ~1/|w|
times larger than the real ones: the stored statistics are meaningless for the other kind); the reference networks get
the same recalibration. Accuracy on the 5,000 validation images (the CDT's split); the test set is never read.

**The check (the truth, approximately):** for every layer, the binary network with that layer real (wa) is
fine-tuned 5 epochs (Adam lr 5e-4 + cosine, batch 128, crop + flip, the benchmark's BatchNorm momentum 0.3); the same
fine-tune of the unchanged binary network is the control; gain = accuracy(layer real, fine-tuned) - accuracy(control).
A full training per layer (10 x 3 seeds x 9.4 min) is the real truth; it is future work if this check is not enough.

**Rule (written before measuring, Nov 17 2026):** the one-shot table can be used to choose layers if its recovery (wa)
has Spearman **≥ 0.8 with the fine-tuned gain on each seed** and the whole table costs **≤ 10 % of one full run**
(~56 s for the 9.4-min binary ResNet-8). The drop column and drop vs recovery agreement are reported, not judged.

Expectation written beforehand (not a result): the stem is first in recovery (independent evidence: the int8 stem
trained gave +4.5 points, training_gap), then fc and the last stage; the shortcut convs last. Drop and recovery
disagree somewhere, because the twin and the binary network are different networks.

Limits known beforehand: one network, 10 layers (Spearman on 10 points is coarse); 5 fine-tune epochs are not a full
training; the recalibration itself changes the reference accuracy (reported).

    python layer_table.py --seeds 42 43 44 \
        --runs ~/Proyectos/Symbex1/benchmarks/experiments/training_gap/runs/train_gap \
        --data ~/Proyectos/Symbex1/benchmarks/data/cifar10            # results/layer_table.json

## Result

`results/layer_table.json`, 3 seeds, Nov 17 2026, one RTX 4000 Ada laptop GPU. Mean ± sd over the seeds, points of
validation accuracy. References after the recalibration: twin 81.7 (stored statistics 83.6), binary 69.9 (70.0).

| Layer | drop w | drop a | drop wa | recovery wa (one shot) | fine-tuned gain (check) |
|---|---|---|---|---|---|
| **stem** | **19.6 ± 7.7** | **16.2 ± 4.2** | **37.2 ± 7.3** | -0.05 ± 0.91 | **+2.21 ± 0.76** |
| b0 | 4.9 ± 1.8 | 12.6 ± 3.4 | 20.4 ± 5.6 | +0.66 ± 0.73 | +0.61 ± 0.80 |
| b1 | 3.1 ± 0.8 | 8.5 ± 2.6 | 14.0 ± 2.9 | +0.41 ± 0.55 | +0.14 ± 0.86 |
| b2 | 4.3 ± 0.9 | 7.7 ± 1.2 | 13.2 ± 1.1 | +0.26 ± 0.24 | +1.19 ± 0.90 |
| ds2 | 2.9 ± 0.9 | 5.0 ± 0.2 | 9.4 ± 1.2 | +0.15 ± 0.56 | +0.40 ± 1.20 |
| b3 | 4.5 ± 0.3 | 4.8 ± 0.6 | 11.0 ± 1.5 | +0.15 ± 0.18 | +0.28 ± 0.46 |
| b4 | 4.3 ± 0.5 | 7.5 ± 1.3 | 13.2 ± 2.4 | +0.26 ± 0.33 | +1.27 ± 0.24 |
| ds4 | 2.9 ± 0.9 | 5.6 ± 0.6 | 9.8 ± 1.4 | -0.58 ± 0.30 | +0.57 ± 0.67 |
| b5 | 5.1 ± 0.3 | 5.4 ± 0.4 | 11.2 ± 0.8 | -0.28 ± 0.27 | -0.13 ± 0.55 |
| fc | 8.6 ± 0.1 | -- | 8.6 ± 0.1 | +0.25 ± 0.15 | +0.43 ± 0.33 |

Spearman (10 layers) per seed 42 / 43 / 44:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| one-shot recovery vs fine-tuned gain (the rule: ≥ 0.8) | -0.04 | 0.16 | 0.24 |
| drop vs fine-tuned gain (not judged) | 0.66 | 0.07 | 0.52 |
| drop vs one-shot recovery (not judged) | -0.22 | 0.86 | -0.06 |

Agreement of each column with itself across seeds (Spearman of seed pairs): drop 0.68-0.83, fine-tuned gain 0.29-0.53,
one-shot recovery -0.02-0.45.

- **Cost:** the whole table (10 layers x 3 parts x 2 directions, each with its recalibration) takes 11-12 s per seed
  (2 % of a full run): it passes the cost. The fine-tune check took 152-191 s per seed (11 fine-tunes).
- The one-shot recovery is noise: putting one layer back to real values in a network trained around 1-bit layers
  changes the accuracy by less than a point either way, and the order changes from seed to seed.
- The drop is stable across seeds and has one clear answer: the **stem** (37 points when only it is binary, about
  twice the next layer). The fine-tuned check agrees on the stem (+2.2, the largest gain on every seed) and so does
  the independent trained result (int8 stem +4.5 points, training_gap).
- In the drop, the **activations cost more than the weights** in every block and shortcut conv (b0: 12.6 vs 4.9; b3 and b5 by only 0.3); the stem
  and fc are the layers whose weights cost most.
- After the stem, nobody ranks the rest: the fine-tuned gains are 0-1.3 points with sd up to 1.2, and they agree with
  themselves across seeds only at 0.29-0.53. Five epochs are too short (or one fine-tune per layer too noisy) to be a
  truth for the small layers. b2, b4 and fc also have mean gains above their sd (+1.19 ± 0.90, +1.27 ± 0.24,
  +0.43 ± 0.33); only b4 clearly (sd 0.24).

## Answer

**The one-shot table does not pass the rule:** its recovery column (one layer of the binary network back to real
values) has Spearman -0.04 / 0.16 / 0.24 with the fine-tuned gain. It costs 12 s (it passes the cost), but in a
network trained with every layer binary, one real layer alone changes almost nothing until the rest is retrained.

The **drop** column (the twin with one layer binary) is the useful half: stable across seeds, cheap, and it finds the
one layer that matters, the stem, which the fine-tune and the trained int8 stem confirm. It also says that the
activations cost more than the weights inside the blocks. Below the stem, neither column nor the 5-epoch check can
rank the layers.

What it means for the tools: for a binary network, a per-layer table must use the drop direction (start from the
real network), not the recovery one; it is good at finding the dominant layer, not at ordering the rest. Ordering the
rest needs a real truth (a full training per layer, 30 runs, ~4.7 h) before any other tool is judged on it.

Expectation check: the stem first was right (drop, fine-tune); fc and the last stage second was wrong (b5 recovers
nothing; b4 and b2 do); the shortcut convs last was right for the drop, not for the fine-tune (ds4 +0.57).

Limits: one network; the twin and the binary network are different networks (drop and recovery disagree: -0.22 /
0.86 / -0.06); the recalibration lowers the twin from 83.6 to 81.7 (the stored statistics come from augmented
images); a single 5-epoch fine-tune per layer is noisy; the MLP is not measured yet.
