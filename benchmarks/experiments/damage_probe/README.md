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

---

# Part 3 -- do cheap measures rank whole mixed configurations the way training does?

## Question

Part 2 changed one layer at a time: the trained differences (0-1.3 points) were mostly seed noise, so no measure
could be judged below the stem. FIT (Zandonati et al., ICLR 2023) validated its measure differently: many random
mixed-precision configurations, each trained a short time from the full-precision model, ranked by the measure vs
their trained accuracy. **Done the same way at 1 bit: which cheap measure ranks random binary / real configurations of
ResNet-8 the way a 30-epoch training does?**

## What

- **Configurations:** 40 distinct random subsets of the 10 layers of part 2 made binary (weights and activation
  together), each layer in with probability 1/2 (generator seed 0, the same 40 for every seed); the other layers stay
  real (twin).
- **Truth:** each configuration trained 30 epochs from the twin checkpoint of the seed (as FIT: 30 epochs from the
  full-precision model; Adam lr 5e-4 = the twin's 5e-3 / 10, cosine, crop + flip, BatchNorm momentum 0.3), BatchNorm
  recalibrated, validation accuracy. Seeds 42 43 44.
- **Measures** (on the trained twin, no training; `config_table.py` docstring has the formulas):
  drop_sum (sum of part 2's per-layer drops), fit_w / fit_a / fit (FIT on weights, activations, both; empirical
  Fisher from 512 training images; noise = x - alpha sign(x) with the scale a BatchNorm absorbs), acc0 (accuracy of
  the configuration without training), cos_end (1 - cosine of its logits with the twin's).
- **Rule (written before measuring, Nov 17 2026):** a measure passes the **coarse** test if its Spearman with the
  trained accuracy is ≥ 0.8 on each seed over the 40 configurations, and the **fine** test if it is also ≥ 0.6 on each
  seed inside each stem group (stem real / stem binary): the stem alone splits the configurations (part 2: 37
  points), so the coarse test could pass by seeing only the stem. Cost: the per-layer measures once ≤ 10 % of a full
  run (56 s); the per-configuration ones ≤ 1 s each.
- **Noise ceiling (reported, not judged):** Spearman of the trained accuracies between seeds.

Expectation written beforehand (not a result): every measure passes the coarse test (the stem); drop_sum and acc0
pass the fine one at best partly; fit_a weakest (FIT's own table: activation terms 0.33-0.39 with BatchNorm).

Limits known beforehand: one network; 30 epochs from the twin is not the benchmark's recipe (150 epochs from the
start); only binary vs real per layer (no int8 / 2-bit); 40 configurations out of 1,024.

    python config_table.py --seeds 42 43 44 \
        --runs ~/Proyectos/Symbex1/benchmarks/experiments/training_gap/runs/train_gap \
        --data ~/Proyectos/Symbex1/benchmarks/data/cifar10            # results/config_table.json

    python config_analysis.py                                          # results/config_analysis.json (checks below)

## Result

120 configurations trained (40 x 3 seeds), ~2.7 h on the GPU, no errors. The trained accuracies spread 56-79 %
(sd ~4.2 per seed); 25 configurations have the stem binary, 15 real. Spearman with the trained accuracy, seeds
42 / 43 / 44:

| measure | all 40 (≥ 0.8) | stem real (≥ 0.6) | stem binary (≥ 0.6) | cost | rule |
|---|---|---|---|---|---|
| drop_sum | 0.92 / 0.93 / 0.94 | 0.96 / 0.98 / 0.94 | 0.93 / 0.94 / 0.92 | 12 s once (part 2 table) | **passes** |
| fit_w | 0.79 / 0.70 / 0.85 | 0.94 / 0.88 / 0.94 | 0.82 / 0.75 / 0.89 | 3-6 s once | fails (coarse) |
| fit_a | 0.88 / 0.91 / 0.91 | 0.91 / 0.92 / 0.94 | 0.95 / 0.93 / 0.92 | (same pass) | **passes** |
| fit | 0.82 / 0.76 / 0.87 | 0.95 / 0.90 / 0.94 | 0.87 / 0.81 / 0.91 | (same pass) | fails (coarse) |
| acc0 | 0.87 / 0.90 / 0.92 | 0.96 / 0.96 / 0.86 | 0.93 / 0.91 / 0.95 | ≤ 0.5 s per configuration | **passes** |
| cos_end | 0.87 / 0.86 / 0.93 | 0.97 / 0.96 / 0.87 | 0.94 / 0.89 / 0.95 | (same pass) | **passes** |
| *noise ceiling* (seed vs seed) | 0.97-0.98 | 0.95-0.97 | 0.97-0.98 | | |

Checks written AFTER the run (`config_analysis.py`, not part of the rule):

| measure | all 40 | stem real | stem binary | **number of binary layers held fixed** |
|---|---|---|---|---|
| **count** (number of binary layers) | 0.84 / 0.87 / 0.87 | 0.98 / 0.96 / 0.98 | 0.91 / 0.93 / 0.94 | 0.00 (by construction) |
| drop_sum | 0.92 / 0.93 / 0.94 | | | **0.83 / 0.82 / 0.72** |
| acc0 | | | | **0.84 / 0.78 / 0.67** |
| cos_end | | | | **0.84 / 0.78 / 0.75** |
| drop_cov (Hill's coverage model) | 0.81 / 0.85 / 0.92 | 0.96 / 0.98 / 0.92 | 0.94 / 0.94 / 0.90 | 0.82 / 0.80 / 0.72 |
| fit_a | | | | 0.47 / 0.64 / 0.54 |
| fit_w | | | | -0.06 / -0.45 / 0.11 |
| fit | | | | 0.05 / -0.32 / 0.19 |
| *noise ceiling* | | | | 0.86-0.92 |

- **The rule was too easy.** Counting the binary layers passes it too: in random configurations the accuracy
  depends mostly on HOW MANY layers are binary, and every measure sees that. The useful question is WHICH layers, at
  the same count (last column: ranks centered inside each group of equal count, residuals correlated; groups of
  3-9 binary layers, 1-9 configurations each). A first version removed only a linear trend of the count, which
  left counting itself at 0.10-0.17 (MendCode review); corrected.
- **At the same count:** drop_sum, acc0 and cos_end tie (0.67-0.84, ceiling 0.86-0.92; none is ahead on every
  seed), then fit_a (0.47-0.64). FIT on the weights sees nothing (-0.45 to 0.19): what it measures at 1 bit is how
  many layers are binary.
- **The stem after 30 epochs:** weighting it more than the count (count + 5 for the stem) ranks WORSE (0.72-0.74
  on all 40): training repairs much of its one-shot 37-point drop. drop_sum still ranks well because the rest of
  its terms carry the order.
- **Hill's coverage model** (drop_cov) does not beat the plain sum: ≤ on every seed over all 40 and at the same count.
- Expectation (written before): every measure passes the coarse test, right for 4 of 6 (fit_w and fit fail);
  drop_sum and acc0 fail the fine one at least partly, wrong (both pass); fit_a weakest, wrong (fit_w is).

## Answer

**Yes for the pre-registered question, with an honest correction:** drop_sum, fit_a, acc0 and cos_end pass the
rule, but so does counting the binary layers. The part that matters for choosing layers is the ranking at the same
number of binary layers: there **part 2's drop table (12 s per network), the accuracy of the configuration without
training and its logit cosine (≤ 0.5 s each) tie, 0.67-0.84 against a seed-to-seed ceiling of 0.86-0.92**; FIT on
the activations is weaker; FIT on the weights does not work at 1 bit. The drop table is the only one that needs no
pass per configuration (one table, then sums).

Limits: one network (ResNet-8), 40 random configurations per seed, the truth is a 30-epoch training from the twin
(not the 150-epoch recipe), binary vs real only. The "same count" check was designed after seeing the results:
it needs its own pre-registered test on another network (MLP, or tiny_llm's layers) before drop_sum goes to tools/dev.

# Part 4 -- do the interactions between layers rank the configurations better than the plain sum?

## Question

The damage does not add up (tiny_llm: all units binary at once costs 2.2-2.5x the sum of the single units). Part 3
ranked its trained configurations with the plain sum of single-layer drops. Do models that include the interactions
(pairs, Shapley values, a count model), all measured without training, rank them better?

## What

- **One shot table:** the twin with each of the 2^10 = 1,024 subsets of the 10 layers binary (weights and
  activation together, as part 3), BatchNorm recalibrated (10,000 training images), validation accuracy. Seeds 42 43 44.
- **Models** (`interactions.py` docstring): additive (sum of single drops), pairs (+ every pair interaction
  I_ij = d_ij - d_i - d_j inside the configuration), shapley (sum of the exact Shapley values of its layers),
  popcount (mean one-shot drop of the subsets with the same stem state and the same number of other binary layers),
  and acc0 (the configuration's own one-shot accuracy, which all the others approximate).
- **Truth:** part 3's trained accuracies (40 configurations x 3 seeds, results/config_table.json, no new training).
  Statistic: Spearman at the same number of binary layers (config_analysis.partial), and over all 40.
- **Rule (written before measuring, Nov 17 2026):** a model is useful if its same-count Spearman beats additive's by
  > 0.05 on each of the 3 seeds. popcount's assumption (the non-stem layers interchangeable) holds if the mean spread
  of the one-shot accuracy inside its (stem, count) groups is not larger than the spread of the same subset between
  seeds.

Expectation written beforehand (not a result): pairs and shapley close to acc0 (they approximate it) and not above
it; none beats additive by > 0.05 on every seed (part 3: acc0 did not beat drop_sum at the same count); popcount
fails its assumption (part 2: b0 drops 12, b5 ~1).

Limits known beforehand: one-shot interactions may not survive training (co-adaptation, quick_check); one network;
the truth is part 3's 30-epoch training.

    python interactions.py --seeds 42 43 44 --runs ~/Proyectos/Symbex1/benchmarks/experiments/training_gap/runs/train_gap \
        --data ~/Proyectos/Symbex1/benchmarks/data/cifar10          # results/interactions.json

## Result

Same-count Spearman with part 3's trained accuracy (seeds 42 / 43 / 44; over all 40 in brackets):

| model | same count | all 40 | beats additive by > 0.05 on every seed? |
|---|---|---|---|
| additive | 0.83 / 0.82 / 0.72 | (0.92 / 0.93 / 0.94) | (reference) |
| pairs | 0.59 / 0.58 / 0.47 | (0.89 / 0.89 / 0.89) | no (worse on all 3) |
| shapley | 0.84 / 0.81 / 0.72 | (0.92 / 0.93 / 0.94) | no (+0.01 / -0.01 / 0.00) |
| popcount | 0.79 / 0.76 / 0.79 | (0.83 / 0.89 / 0.95) | no (-0.04 / -0.06 / +0.07) |
| acc0 | 0.84 / 0.78 / 0.67 | (0.87 / 0.90 / 0.92) | no |

- The one-shot table reproduces part 3's acc0 exactly (max difference 0.00 on every seed).
- In ResNet-8 the damage is **sub-additive**, the opposite of tiny_llm: the sum of the 10 single drops is 151 / 146 /
  147 points, all 10 binary at once drop 68 / 68 / 69. Accuracy cannot fall below chance (10 %), so the drops saturate.
  The largest pair interactions are negative and almost all involve the stem (stem + b0: -10.3 / -9.3 / -8.1).
- popcount's assumption: mean spread inside a (stem, count) group 2.08 / 2.08 / 2.96 points vs 2.32 for the same subset
  between seeds: it holds on 2 seeds and fails on 1.

## Answer

No. Using the interactions does not rank the trained configurations better than the plain sum. Shapley ties
additive. pairs is clearly worse: adding every negative pair term overcorrects a game that saturates. The count model
is better on one seed and worse on two. As expected beforehand, nothing beats additive; pairs fell below acc0
(expected: close to it). popcount's assumption is borderline, not clearly broken. For the ResNet-8 the single-layer
table plus a sum stays the cheapest useful ranking. The super-additivity seen in tiny_llm does not appear here (sub-additive, bounded by
chance).

# Part 4b -- is the sub-additive damage only the 10 % floor? The same table with the validation loss

## Question

Accuracy cannot fall below chance (10 %), so the drops of part 4 may saturate only because of that floor. The
cross-entropy has no such ceiling (tiny_llm's damage, super-additive, was measured in loss). With the loss, is the
ResNet-8 damage still sub-additive?

## What

- The same one-shot table (1,024 subsets, BatchNorm recalibrated on the same 10,000 images, seeds 42 43 44), now
  recording the mean validation cross-entropy (natural log) in the same pass as the accuracy. Game: v(S) = loss(S) -
  loss(no layer binary). `interactions.py --metric loss --out results/interactions_loss.json`.
- **Rule (written before measuring, Nov 17 2026):** ratio = loss increase with all 10 layers binary / sum of the 10
  single increases. Super-additive if ratio > 1 on each of the 3 seeds (then the part 4 "sub-additive" was the floor:
  ResNet-8 and tiny_llm tell the same story); sub-additive if ratio < 1 on each seed (a property of the network, not
  of the metric); otherwise mixed. Also reported, not judged: the share of the 45 pair interactions that are positive,
  and part 4's ranking table with the loss models.

Expectation written beforehand (not a result): no expectation either way; the stem's pair interactions are the place to look.

## Result

The accuracies recorded in the same pass are identical to part 4's (max difference 0.0). Validation loss with no layer
binary: 0.56 / 0.59 / 0.54. With all 10 binary: 7.01 / 7.05 / 6.61, far above chance (ln 10 = 2.30): the network
is confidently wrong, so the loss does not saturate the way accuracy does.

| seed | sum of the 10 single increases | all 10 binary | ratio | positive pair interactions |
|---|---|---|---|---|
| 42 | 6.98 | 6.45 | 0.92 | 43 / 45 |
| 43 | 6.66 | 6.46 | 0.97 | 42 / 45 |
| 44 | 6.20 | 6.07 | 0.98 | 43 / 45 |
| (accuracy, part 4) | 151 / 146 / 147 points | 68 / 68 / 69 | 0.45 / 0.47 / 0.47 | 24 / 23 / 24 of 45 |

Ranking with the loss models (same count, s42 / s43 / s44): additive 0.83 / 0.80 / 0.68, pairs 0.87 / 0.79 / 0.68,
shapley 0.84 / 0.80 / 0.68, popcount 0.79 / 0.76 / 0.76, acc0 0.87 / 0.77 / 0.67. None beats additive by > 0.05 on
every seed.

## Answer

By the rule written beforehand: **sub-additive** (ratio < 1 on all 3 seeds). But only barely: 0.92-0.98, almost
exactly additive. In accuracy the ratio was 0.45. Most of part 4's sub-additivity was the floor. In loss, almost every
pair adds damage (42-43 of 45 pair interactions positive; in accuracy about half). Taken all together the 10 layers
come out close to the plain sum. It is not tiny_llm's 2.2-2.5x. In loss, ResNet-8 is close to additive and the LLM is
clearly super-additive. The two networks do not tell the same story, but the difference is much smaller than accuracy
suggested. The interaction models still do not rank the trained configurations better than the sum.

## Check: do a few extreme images drive the ratio?

With a mean loss of ~7, a few confidently wrong images could dominate. `loss_tails.py` recomputes the 57 subsets the
ratio and the pair count need (none, the 10 singles, the 45 pairs, all 10), keeping every image's loss, and gives
the ratio with three statistics: the mean (must equal part 4b), the median, and the mean of the loss clipped at 10
per image.

**Rule (written before measuring, Nov 17 2026):** the 4b reading (close to additive, not super-additive) is solid if
the median and clipped ratios stay in 0.85-1.05 on every seed. If either goes > 1.05 on any seed, the mean hid a
super-additive bulk; if < 0.85, the mean hid a sub-additive one.

Note on 4b's comparison with tiny_llm (Edén): the ratio has no units, so it compares across models; what differs is
the task and the architecture.

### Result

The mean ratio reproduces 4b exactly (0.924 / 0.971 / 0.979).

| seed | mean | median | clipped at 10 | images above 10 (all binary) | loss share of the worst 5 % (all binary) |
|---|---|---|---|---|---|
| 42 | 0.92 | 2.10 | 0.81 | 27.5 % | 13.5 % |
| 43 | 0.97 | 2.47 | 0.84 | 28.0 % | 13.6 % |
| 44 | 0.98 | 3.14 | 0.86 | 24.1 % | 13.6 % |

Positive pair interactions: mean 42-43 / 45, median 45 / 45, clipped 41-43 / 45.

### Answer

The rule fails: the median goes > 1.05 on every seed, and the clipped ratio goes < 0.85 on 2 seeds. The 4b reading
"close to additive" is not solid. The ratio depends on the statistic: 0.45 (accuracy), 0.81-0.86 (clipped),
0.92-0.98 (mean), 2.1-3.1 (median).

A few images are NOT the cause: with all 10 binary the worst 5 % of images carry only 13.6 % of the loss, and the
median increase (6.1-6.5) is close to the mean one. What moves the ratio is the single layers. One layer alone
leaves most images right, so the median image's loss barely rises. All 10 together push it past the decision.
The median, like accuracy, has a threshold inside it, so it sees super-additivity where the mean sees about the sum.
The clipped loss caps 24-28 % of the images: another floor.

So "ResNet-8 sub- or super-additive" has no answer apart from the statistic. The only like-for-like comparison with
tiny_llm is mean vs mean loss: 0.92-0.98 vs 2.2-2.5. That gap is still there, and the OR-shortcut hypothesis
(LOGBOOK) is still unmeasured.

## Part 4 in one paragraph: magnitude vs order

Additivity is a property of the network **and** the statistic, not of the network alone. A statistic with a floor or
a threshold bends the sum: accuracy saturates at chance and looks sub-additive (0.45); the median has a threshold that
a single layer does not cross and looks super-additive (2.1-3.1); the mean loss, with neither, sits near the sum
(0.92-0.98). A claim that per-layer damage is (or is not) additive must name its statistic. For the CDT two claims
are separate:

- **Magnitude** (does the sum predict how much the network drops?): depends on the statistic. Not used.
- **Order** (does the sum rank the configurations?): holds. Same-count Spearman with the trained accuracy 0.72-0.83
  (accuracy table) and 0.68-0.83 (loss table), and no interaction model beats it. Choosing the better configuration
  is what the CDT needs, not the exact number.

# Part 5 -- does a layer's damage follow its fan-in and its distance to the output?

## Question

Two hypotheses (Edén, Nov 17) for where the one-shot damage comes from, both checked on tables already measured (part
2 for ResNet-8, llm_layers for tiny_llm), no training:

- **A, count resolution:** a binary neuron with K inputs gives only K + 1 count values; a small fan-in loses more.
  Prediction: the drop falls as the fan-in grows.
- **B, distance to the output:** an early error travels through every later layer. Prediction: the drop grows with
  the number of layers left.

## What

- `fanin_depth.py`. Drops: ResNet-8 part 2 `drop[wa]` (accuracy points; `w` and `a` also reported), tiny_llm
  llm_layers `before.units[*].dloss` (validation loss increase of the fp32 model with only that unit 1-bit).
- Fan-in (read from the checkpoints): ResNet-8 stem 216 (24 thermometer channels x 3x3, not 54: this net has no
  BPR), b0 / b1 / b2 144, ds2 16, b3 / b4 288, ds4 32, b5 576, fc 4096; tiny_llm qkv / o / w13 384, w2 1024.
- Layers left: ResNet-8 weight layers after it on the main path (stem 7, b0 6, b1 5, b2 / ds2 4, b3 3, b4 / ds4 2,
  b5 1, fc 0); tiny_llm blocks after it (7 - block).
- Known confound: in ResNet-8 the fan-in grows with depth (channels double per stage), so A and B overlap there;
  the cleanest tests are in tiny_llm (same block = same distance, different fan-in; same group = same fan-in,
  different distance). w13 is two matrices, so its size works against A; o (one 384 x 384 matrix, fewer weights than
  w2, smaller fan-in) separates fan-in from size.

**Rule (written before measuring, Nov 17 2026), each on every seed:**
- A supported if Spearman(drop, fan-in) <= -0.5 in each model, and in tiny_llm w2 drops less than o in >= 6 of 8
  blocks. Against A if the sign is positive in either model.
- B supported if Spearman(drop, layers left) >= 0.5 in each model (tiny_llm: also >= 0.5 inside each of the 4
  groups, mean over groups).
- The model drop = a + b log(fan-in) + c (layers left), least squares per model and seed, is useful if its
  leave-one-out R^2 > 0.5.
With 10 (ResNet-8) or 32 (tiny_llm) points, a Spearman of 0.5 is a modest bar; the per-seed repetition is the check.

## Result

Spearman per seed (42 / 43 / 44), results/fanin_depth.json:

| | ResNet-8 drop wa | ResNet-8 drop w only | ResNet-8 drop a only | tiny_llm dloss |
|---|---|---|---|---|
| vs fan-in (A predicts < 0) | 0.11 / -0.10 / -0.17 | **+0.63 / +0.70 / +0.60** | 0.09 / 0.01 / -0.25 | -0.32 / -0.30 / -0.31 |
| vs layers left (B predicts > 0) | **0.54 / 0.80 / 0.78** | 0.06 / -0.07 / 0.04 | 0.39 / 0.72 / 0.89 | 0.04 / 0.04 / 0.03 |
| model a + b log(fan-in) + c left, LOO R^2 | -0.20 / 0.13 / 0.59 | | | -0.07 / -0.11 / -0.07 |

tiny_llm: w2 drops less than o in 3 / 3 / 2 of 8 blocks (A needs >= 6). w2 drops less than w13 in 8 / 8 / 8, but
w13 is two matrices. Inside each group across the 8 blocks, the Spearman with blocks left averages 0.01 / 0.05 /
-0.08. Mean dloss by group: qkv 0.105 (3 matrices), w13 0.079 (2), w2 0.035 (1, 1024 x 384), o 0.022 (1, 384 x 384).

## Answer

- **A (fan-in): not supported.** In ResNet-8 the whole-layer drop has no relation to the fan-in. With the weights
  alone it is the opposite: the larger the fan-in, the larger the drop. In tiny_llm the sign is right but weak
  (-0.31), and o, with the smallest fan-in, drops less than w2 in most blocks. The stem is not a small-fan-in layer
  in this network (216 inputs, more than b0-b2).
- **B (distance): supported in ResNet-8 only.** It passes on every seed, carried by the activations, not the weights.
  In ResNet-8 the distance overlaps with the stage, the channel count and the stem, so this does not single out
  distance as the cause. In tiny_llm, the one-shot damage of a unit does not depend on its block (inside every group
  about 0). Block 0's weight in part 3 came from training, not from this table.
- **The model drop ≈ f(fan-in) x g(layers left): not useful** (LOO R^2 > 0.5 only on ResNet-8 seed 44).
- Seen, not tested (after looking at the numbers): in tiny_llm the order of the groups follows how many matrices a
  unit binarizes (qkv 3 > w13 2 > w2, o 1), then their size (w2 > o). A test written beforehand would need units of
  equal size.

## Checks of part 5 (Nov 17, rules written before measuring)

1. **ResNet-8 B without the stem and at equal fan-in** (`fanin_depth.py`, no run): Spearman(drop wa, layers left)
   on the 9 layers without the stem, and partial Spearman(drop wa, layers left | fan-in) on all 10 (ranks, residuals
   of a linear fit on the fan-in rank). B survives the confound if both are >= 0.5 on every seed.
2. **tiny_llm, one matrix at a time** (`llm_matrices.py`, GPU, no training): the fp32 model with only one of the 56
   matrices (8 blocks x q k v o w1 w3 w2) 1-bit, validation loss increase, same 20 batches as llm_layers (the base
   loss must equal llm_layers'). Shapes: q k v o 384 x 384; w1 w3 384 -> 1024 and w2 1024 -> 384 have the same number
   of weights.
   - A at equal size: w2 (fan-in 1024) drops less than the mean of w1 and w3 (fan-in 384) in >= 6 of 8 blocks on
     every seed.
   - Matrix count: the post hoc order (qkv > w13 > w2, o) is explained by the number of matrices if q, k, v alone
     each drop like o (mean of q, k, v within 0.5x-2x of o's mean on every seed); the role matters if not.
   - Also reported: Spearman of each unit's dloss with the sum of its matrices' single drops (32 units).

### Results of the checks

1. **ResNet-8 B:** without the stem 0.36 / 0.72 / 0.70; at equal fan-in (partial) 0.69 / 0.88 / 0.81. The rule (both
   >= 0.5 on every seed) **fails** on seed 42 without the stem. Distance survives the fan-in, but on one seed in
   three most of it is the stem.
2. **tiny_llm, one matrix at a time** (base loss equal to llm_layers' on every seed):

| seed | q | k | v | o | w1 | w3 | w2 | w2 < mean(w1, w3) | mean(q, k, v) / o | unit vs sum of its matrices: Spearman, ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.020 | 0.029 | 0.024 | 0.022 | 0.029 | 0.025 | 0.035 | 0 / 8 | 1.09 | 0.97, 1.34 |
| 43 | 0.018 | 0.024 | 0.020 | 0.022 | 0.029 | 0.025 | 0.035 | 0 / 8 | 0.95 | 0.97, 1.38 |
| 44 | 0.021 | 0.032 | 0.024 | 0.024 | 0.028 | 0.025 | 0.034 | 0 / 8 | 1.08 | 0.98, 1.31 |

(mean loss increase over the 8 blocks)

- **A at equal size: rejected, the other way.** w2 (fan-in 1024) drops MORE than w1 / w3 (fan-in 384, same weights)
  in 8 of 8 blocks, every seed. Same direction as ResNet-8's weights-only drop (+0.6 to +0.7 with fan-in). Binarizing
  the weights of a large fan-in layer costs more, not less. Not tested why.
- **Matrix count: holds.** A 384 x 384 matrix costs about the same whatever its role (q, k, v vs o: 0.95-1.09). The
  qkv unit is the most damaging because it binarizes three matrices at once.
- A unit costs 1.31-1.38x the sum of its matrices alone: the super-additivity is already there inside one block.

# Part 5c -- does the weight distribution (cos(w, sign w)) explain the fan-in effect?

## Question

Hypothesis C (Edén, Nov 17): sign() gives every weight the same magnitude, even the ones near zero. If a trained
layer's signal sits in a few large weights, binarizing turns many small weights into +-1 noise, and a layer with
more inputs has more of them. Measure per output row: cos(w, sign w) = ||w||_1 / (sqrt(n) ||w||_2) (1 = all weights
of equal magnitude; lower = signal concentrated in few weights; the per-layer quantity RBNN plots).

## What

- `weight_cosine.py`, no GPU: the saved fp32 / twin weights (ResNet-8 twin_s<seed>.pth; tiny_llm
  fp32_d384l8_s<seed>), cosine of every output row (conv: the C_in x k x k weights of one output channel; linear:
  one row, length = fan-in), mean over the rows of a layer. Invariant to any per-row scale, so it does not depend on
  how the binary layer is scaled.
- Drops: ResNet-8 part 2 `drop[w]` (weights only, primary: the cosine is about weights; `wa` also reported);
  tiny_llm part 5's single matrices (`llm_matrices.json`, 56 per seed).

**Rule (written before measuring, Nov 17 2026), each on every seed:**
- C1: Spearman(cosine, fan-in) <= -0.5 in each model.
- C2: Spearman(cosine, drop) <= -0.5 (lower cosine, larger drop) and |Spearman(cosine, drop)| beats |Spearman(fan-in,
  drop)| by > 0.1, in each model.

## Result

| seed | ResNet-8 cos vs fan-in | ResNet-8 drop w: cos / fan-in | ResNet-8 drop wa: cos / fan-in | tiny_llm cos vs fan-in | tiny_llm drop: cos / fan-in |
|---|---|---|---|---|---|
| 42 | 0.17 | -0.12 / 0.63 | 0.37 / 0.11 | -0.06 | -0.24 / 0.15 |
| 43 | -0.61 | -0.38 / 0.70 | -0.15 / -0.10 | -0.01 | -0.09 / 0.25 |
| 44 | -0.23 | -0.46 / 0.60 | -0.62 / -0.17 | -0.04 | -0.19 / 0.12 |

Every layer and matrix has a cosine of 0.77-0.81 (ResNet-8) and 0.79-0.80 (tiny_llm). A Gaussian gives sqrt(2/pi) =
0.798. The trained weights are close to Gaussian everywhere: the signal is not concentrated in a few weights in any
layer.

## Answer

**C fails.** C1: the cosine does not fall with the fan-in (only ResNet-8 seed 43 passes). C2: the cosine does not
predict the drop better than the fan-in. In ResNet-8 weights-only, the fan-in (+0.6 to +0.7) beats the cosine (-0.1
to -0.5). The cosine barely varies between layers (±2 %), so it cannot explain differences in damage. The fan-in
effect (larger fan-in, larger weight-binarization drop) stays unexplained.

Code review (Nov 17): `fanin_depth.partial_rho` broke ties (equal fan-ins / depths) by order instead of average
ranks; fixed with rankdata. The check values change slightly (given fan-in 0.69 / 0.88 / 0.81 -> 0.71 / 0.89 /
0.83), and the verdicts do not change.

# Part 5d -- the fan-in effect: the scale, the 8-bit input, or the layer type?

## Question

Part 5's checks: binarizing the weights of a larger fan-in layer costs more (tiny_llm w2 > w1 / w3 at equal size in
8 / 8 blocks; ResNet-8 weights-only drop +0.6 to +0.7 with fan-in). Edén's candidates (Nov 17):
- **tiny_llm, the scale:** without a scale factor, sign(w) inflates the output by ~sqrt(n). **Read in the code
  first:** `train.quant_weight` already scales every row by alpha = mean|w| (XNOR-Net / BWN's optimal scale), so the
  sqrt(n) inflation is not there. But `kind = "binary"` also quantizes the layer's INPUT to int8 (absmax per token,
  `quant_act8`), and w2's input (silu(w1 x) * w3 x) is the place where LLMs carry massive activations (DuQuant,
  Super Weight). New candidate: the w2 excess comes from its 8-bit input, not its weights.
- **ResNet-8, the layer type:** the fan-in goes with the stage and with conv vs 1x1 shortcut. Check the fan-in with
  stage and type held fixed.

## What

- `llm_matrices.py --part w | a`: the same 56 single matrices with only the 1-bit weights (x alpha, input in floating
  point) or only the int8 input (weights in floating point); `wa` = part 5 (both).
- `fanin_depth.py`: on the 8 block convs and shortcuts (stem and fc are their own types), partial Spearman(drop w,
  fan-in | stage, type) (ranks, residuals of a linear fit on stage and type dummies); the only same-stage same-type
  pairs with different fan-in: b3 (288) vs b2 (144), b5 (576) vs b4 (288).

**Rule (written before measuring, Nov 17 2026), each on every seed:** r = mean w2 drop / mean of w1, w3 drops (part 5
wa: 1.19-1.40).
- The 8-bit input explains the excess if r(w only) <= 1.05 and w2's input-only drop is larger than mean(w1, w3)'s
  in >= 6 of 8 blocks. The weights explain it if r(w only) >= 1.2. Otherwise both share it.
- ResNet-8: the fan-in effect survives stage and type if the partial Spearman >= 0.5 and b3 > b2 and b5 > b4 (drop w).
