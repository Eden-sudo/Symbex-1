# test_math -- the math of the binary neuron, checked on trained networks

Each binary neuron is `bit = H(z - T)`: `z` is the integer sum of the ±1 products and `T` its integer threshold.
This folder measures questions about that formula on trained networks, in the CDT's exact integer simulation.
The questions come from writing the model out for a math review.

## 1. Does H(0) matter? (`tie_rule.py`, Nov 24, 2026)

**Question.** The library uses H(0) = 1: when z equals T, the bit is +1. What changes if a trained network uses
H(0) = 0 (equivalent to threshold T + 1) or H(0) = ½ (0, which makes it a ternary value)?

**What we did.** We changed only the ties, with no retraining. With H(0) = 1, every network reproduces the CDT
simulation prediction by prediction. Everything is measured on the validation split: for ResNet-8, the 2000 images
the export saved; for HAR and ECG, the whole split. Seeds 42, 43 and 44. Data: `results/tie_rule.json`.

| Network | Ties z = T | H(0) = 0 | H(0) = ½ | Predictions changed (0 / ½) |
|---|---|---|---|---|
| MLP MNIST 64-256-10 (seed 42, the touch demo's model) | 0.05 % | +0.09 | +0.05 | 0.15 % / 0.08 % |
| ResNet-8 base | 1.3-1.8 % | −4.5 | −0.6 | 26 % / 19 % |
| ResNet-8 int8 stem + pool2 | 0.9-1.2 % | −2.4 | −0.3 | 18 % / 13 % |
| HAR, binary first layer (4 nets) | 0.2-2.2 % | −0.7 | +0.05 | 18 % / 12 % |
| HAR, int8 first layer (4 nets) | 0.4-2.2 % | −1.0 | +0.1 | 19 % / 12 % |
| ECG, binary first layer (s / m / l) | 0.6-1.4 % | −0.3 (−3.1 to +3.7) | −0.8 | 13 % / 9 % |
| ECG, int8 first layer (s / m / l) | 0.5-2.3 % | −1.7 (−5.2 to +2.5) | −1.5 | 9 % / 6 % |

The H(0) columns give the change in points against H(0) = 1: accuracy for ResNet-8 and HAR, balanced accuracy
for ECG. The MLP row was measured with an ad-hoc numpy script on the demo's header; it is not in the JSON.

**Answer.** H(0) is part of the model.
- In the convolutions, about 1 % of the neurons tie, 20-30x more often than in the MLP. A neuron sums only a few
  bits (a 3x3xC window), so z often equals T exactly. In the MLP, the 1-bit input is 0 or 128 and few thresholds
  fall on a multiple of 128.
- Changing only those ties changes 9-26 % of the final predictions. The deepest network (ResNet-8) changes most.
- The total accuracy can hide this: in HAR and ECG, fixed and broken predictions partly cancel out.
- ½ does less harm than 0, but it does not win, and it would cost 2 bits per activation.
- The network was trained with H(0) = 1 (PyTorch's `x >= 0 -> +1`), and the C uses the same rule (`raw >= T`).
  That is why the chip matches the simulation 100 %.

```
python tie_rule.py resnet8 ../../conv2d/cnn_cifar_resnet8/runs/w1_base_s4{2,3,4} --out results/tie_rule.json
python tie_rule.py har ../../conv1d/har_ultracompact/runs/max_binary_e100_s4{2,3,4} --out results/tie_rule.json
python tie_rule.py ecg ../../conv1d/ecg_mitbih/runs/m_binary_e100_s4{2,3,4}_rr --out results/tie_rule.json
```

## 2. Is the binary network fragile, and does that explain its accuracy gap? (`fragility.py`, Nov 24, 2026)

**Question.** Section 1 showed that moving only the ties changes 9-26 % of the predictions. Is that sensitivity
special to the binary network, or does its comparison network share it? And does it explain why the binary network
is less accurate?

**What we did.** One perturbation that does not depend on the architecture: noise of ±1 step on every 8-bit input
value (random sign per value, clipped to the input range), the same inputs for both networks. We count the
predictions that change, among the right and among the wrong ones, on the validation split and on training data
(the first 5000 samples). Comparison network: for ResNet-8 the training_gap twin (same net, real weights, hardtanh),
for HAR and ECG the TFLite INT8 model of the same net and seed. Seeds 42, 43 and 44; the test set is never read.
Noise of ±2, 4 and 8 steps is in the JSON too and gives the same picture. Data: `results/fragility.json`.

| Network | Binary: accuracy val / train | Binary: right / wrong that change (val) | Control: accuracy val / train | Control: right / wrong that change (val) |
|---|---|---|---|---|
| ResNet-8 base (control: twin) | 70.0 / 72.5 | 9.1 ± 0.3 / 40.1 ± 1.0 | 83.6 / 92.3 | 2.0 ± 0.1 / 18.1 ± 0.5 |
| HAR min | 44.9 / 45.1 | 2.8 ± 0.7 / 9.4 ± 4.0 | 61.1 / 84.6 | 1.8 ± 0.5 / 7.4 ± 1.0 |
| HAR max10 | 62.8 / 81.3 | 6.9 ± 0.0 / 25.1 ± 2.5 | 70.2 / 100.0 | 1.2 ± 0.3 / 5.5 ± 1.7 |
| HAR max5 | 69.2 / 90.7 | 6.8 ± 1.3 / 33.9 ± 2.7 | 73.7 / 99.9 | 0.6 ± 0.3 / 5.1 ± 1.0 |
| HAR max | 68.6 / 91.1 | 6.0 ± 1.0 / 28.2 ± 3.8 | 73.6 / 99.9 | 0.5 ± 0.1 / 3.2 ± 0.8 |
| ECG s | 64.2 / 92.3 | 9.6 ± 1.7 / 23.3 ± 5.2 | 70.6 / 99.0 | 2.4 ± 0.3 / 6.2 ± 2.3 |
| ECG m | 65.6 / 95.7 | 9.4 ± 4.4 / 23.2 ± 3.8 | 71.3 / 99.7 | 1.7 ± 0.2 / 5.7 ± 1.6 |
| ECG l | 70.0 / 96.1 | 5.4 ± 2.8 / 21.2 ± 4.1 | 60.5 / 98.7 | 1.4 ± 0.5 / 3.0 ± 1.3 |

Accuracy is plain accuracy on the validation split of each benchmark (mean of 3 seeds); the change columns are
mean ± sd in % of the right (or wrong) predictions. On training data the right binary predictions that change are
8.5 % for ResNet-8 (the same as on validation) and 0.9-3.9 % for HAR max* and ECG (fewer than on validation). The
samples that change in the binary and in the control network barely overlap: Jaccard 0.005-0.076, about what two
independent choices would give (lift 0.5-2.8).

**Answer.** The binary network is more fragile, but that does not explain its accuracy gap.
- On every network, ±1 step changes more of the binary network's right predictions than the control's: 1.6x on
  HAR min, 4-6x on ResNet-8 and ECG, 6-12x on HAR max*.
- ECG is as fragile as the others and ties INT8 in accuracy on the test set (ECG README). If fragility caused the
  gap, ECG would show one too.
- Fragility follows how well the network fits its data. ResNet-8 binary fits only 72.5 % of its training data and
  is as fragile there as on validation (capacity). HAR max* and ECG fit 91-96 % and are less fragile on what they
  fit. HAR min fits 45 % and its few right predictions change little: it is wrong on most samples anyway.
- The two networks are fragile on different samples, so the binary network's fragile samples are not simply the
  "hard" ones.
- All networks here are convolutional. The MLP ties 20-30x less often (section 1); its fragility is not measured.

```
python fragility.py resnet8 --seeds 42 43 44 --out results/fragility.json
python fragility.py har --nets min max5 max10 max --seeds 42 43 44 --out results/fragility.json
python fragility.py ecg --nets s m l --seeds 42 43 44 --out results/fragility.json
```

## 3. Where does the fragility live: inside the network or at its output? (`damage.py`, rule Nov 25, 2026, before measuring)

**Question.** Section 2 found the binary network 1.6-12x more fragile than its control, and fragility tied to how
well it fits its data. Two places can cause it: the damage grows layer by layer (propagation), or it stays small
inside and only the output has little margin between the two top classes. And is it a property of the
convolutions? The MLP ties 20-30x less often (section 1).

**What.** Only binary networks, in the CDT's exact integer simulation (the clean predictions checked equal to it),
seeds 42, 43, 44; validation and training data, the test set never read. Networks: ResNet-8 base (CDT `w1_base`),
HAR max, ECG m (the section 2 runs) and, as the non-convolutional control, the Fashion MLP 784-128-512-10 (labels
only, the sized model of the mlp_fashion README). Two perturbations:
- **input**: ±1 step on every 8-bit input value, as in section 2;
- **inside**: 1 % of the first binary layer's output bits flipped at random, the same share for every network
  (it does not depend on how the input is read, so the MLP and the convolutions compare directly).

Measured: the share of bits that change in each binary layer, the predictions that change, and the output margin
(top logit minus the second, before the perturbation) of the samples that change and of those that do not.

**Rule** (means over 3 seeds, validation):
1. Propagation, input noise: share of changed bits in the last binary layer / in the first. Above 2: the damage
   grows; below 0.5: it fades; in between: it stays.
2. Margin: median margin of the predictions that change / of those that do not. Below 0.5: a small output margin
   decides which samples move.
3. Convolutions vs MLP, inside perturbation: if the MLP's changed predictions are less than half of the smallest
   convolutional network's, the architecture matters; otherwise it does not.

**Result** (mean ± sd over 3 seeds, validation; data: `results/damage.json`):

| Network | Binary layers | Changed bits, first → last layer (input ±1) | Last / first | Predictions changed: input ±1 / inside 1 % | Margin, changed / unchanged |
|---|---|---|---|---|---|
| ResNet-8 base | 7 | 2.58 → 14.35 % | **5.6 ± 0.2** | 18.5 / 17.9 % | 0.22 |
| HAR max | 3 | 0.40 → 5.47 % | **13.6 ± 0.6** | 13.0 / 16.9 % | 0.16 |
| ECG m | 3 | 1.14 → 7.17 % | **6.6 ± 1.4** | 14.0 / 15.1 % | 0.26 |
| MLP 784-128-512-10 | 2 | 0.25 → 0.40 % | **1.6 ± 0.0** | 0.9 / **3.3** % | 0.09 |

With 1 % of the first layer flipped, last / first is 13.5 (ResNet-8), 7.7 (HAR), 8.3 (ECG) and 1.14 (MLP). On
training data the bits change exactly as much as on validation in every network, but the predictions that change
fall where the network fits its data: HAR max 13.0 -> 5.9 %, ECG m 14.0 -> 2.7 %, MLP the same (0.9 %), and
ResNet-8 the same (18.5 -> 17.6 %, it fits only 73 % of its training data).

**Answer.**
1. Propagation: in the three convolutional networks the damage **grows**, 5.6-13.6x from the first to the last
   binary layer. In the MLP it stays (1.6x).
2. Margin: in every network the predictions that change had 4-11x less margin than those that did not (ratio
   0.09-0.26). The margin decides which samples move; training data move less because the network gave them more
   margin, not because less damage reaches the output.
3. Convolutions vs MLP: the same 1 % inside changes 3.3 % of the MLP's predictions and 15-18 % of the convolutional
   networks': less than half, so by the rule the architecture matters.

Limit: the MLP also has fewer binary layers (2 against 3 and 7), so this does not separate "convolution" from
"depth". HAR and ECG have 3 layers and already grow 7-14x, which points at the convolution (a neuron that sums a
few bits of a small window, section 1) more than at depth; a deeper MLP or a 2-layer convolutional network would
separate the two.

```
python damage.py resnet8 --seeds 42 43 44 --out results/damage.json
python damage.py har --nets max --seeds 42 43 44 --out results/damage.json
python damage.py ecg --nets m --seeds 42 43 44 --out results/damage.json
python damage.py mlp --seeds 42 43 44 --out results/damage.json
```

## 4. Convolution or depth? (`deep_mlp.py` + `damage.py --depth`, rule Nov 25, 2026, before measuring)

**Question.** In section 3 the damage grows through the convolutional networks and stays in the MLP, but that MLP
also has fewer binary layers (2 against 3 and 7). Does a deeper MLP grow the damage too?

**What.** The same MLP made deeper with more 512-wide layers: 784-128-512-512-10 (3 binary layers, like HAR and ECG)
and 784-128-512 x 6-10 (7, like ResNet-8). Same recipe as the section 3 MLP: its own FP32 model (30 epochs, seed 42),
then the CDT, every layer 1 bit, 15 epochs, labels only, seeds 42, 43, 44. Then `damage.py`, the same two
perturbations and measures as section 3.

**Rule** (means over 3 seeds, validation):
1. Depth, input noise: changed bits last / first layer. If both deep MLPs stay below 2 (as the 2-layer MLP, 1.6),
   depth does not make the damage grow: the convolution does. If they reach the convolutional range (5.6 or more),
   depth explains it. In between: both play a part.
2. Inside 1 %: if the predictions changed by each deep MLP are less than half of those of the convolutional networks
   with the same number of binary layers (HAR / ECG 15.1-16.9 % for 3, ResNet-8 17.9 % for 7), the architecture
   matters at equal depth.

**Result** (mean ± sd over 3 seeds, validation; data: `results/damage.json`, keys `mlp_d3_*` / `mlp_d7_*`):

| Network | Binary layers | Validation accuracy | Changed bits, last / first (input ±1) | Largest / first | Predictions changed: input ±1 / inside 1 % |
|---|---|---|---|---|---|
| MLP 784-128-512-10 (section 3) | 2 | 88.27 ± 0.35 | 1.60 ± 0.02 | 1.60 | 0.9 / 3.3 % |
| MLP 784-128-512-512-10 | 3 | 88.30 ± 0.35 | **2.61 ± 0.09** | 2.61 | 1.4 / **3.9** % |
| MLP 784-128-512 x 6-10 | 7 | 87.95 ± 0.36 | **0.78 ± 0.11** | 2.29 | 1.7 / **4.2** % |
| HAR max / ECG m (section 3) | 3 | 68.6 / 65.6 | 13.6 / 6.6 | — | 13.0-14.0 / 15.1-16.9 % |
| ResNet-8 base (section 3) | 7 | 70.9 | 5.6 | — | 18.5 / 17.9 % |

In the 7-layer MLP the changed bits rise to 2.3x the first layer by the third, then fall: 0.26 -> 0.38 -> 0.58 ->
0.54 -> 0.44 -> 0.24 -> 0.22 % (seed 42). Training data give the same numbers. The output margin ratio is 0.11 and
0.15 (section 3: 0.09-0.26). Both deep MLPs are exact (compiled C = simulation, 100 %).

**Answer.**
1. Depth, by the rule: in between. One more layer raises the growth from 1.6x to 2.6x, but at 7 layers the damage
   rises to 2.3x and then fades back to 0.8x. No MLP comes near the convolutional networks (5.6-13.6x).
2. At equal depth the convolutional networks change 4x more predictions (3 layers: 3.9 % vs 15.1-16.9 %; 7 layers:
   4.2 % vs 17.9 %): the architecture matters, not only the depth.

So the growth of the damage comes mainly from the convolution; depth adds a little. A likely cause, not measured:
a convolutional bit is read by every window that covers it (9 positions in 3x3, plus the next layer's channels),
so one changed bit reaches many neurons of the next layer, and each of those neurons sums few terms (section 1).
In the MLP every neuron sums 512 bits, so one changed bit moves each sum by only 2 out of up to 512.

```
python deep_mlp.py --depths 3 7 --seeds 42 43 44
python damage.py mlp --depth 3 --seeds 42 43 44 --out results/damage.json
python damage.py mlp --depth 7 --seeds 42 43 44 --out results/damage.json
```

## 5. Is it the convolution, or the binary convolution? And how does one bit spread? (rule Nov 25, 2026, before measuring)

**Questions.** (a) Section 4 found that the damage grows through the binary convolutions. Does it grow through the
INT8 convolutions of the same network too? (b) Why does it grow? A changed bit reaches some neurons of the next
layer (its **reach**: the sums that read it) and each of those flips only if its sum was close to the threshold
(**flip chance**). In an MLP one bit reaches every neuron of the next layer, but each sum has hundreds of terms; in a
convolution it reaches only the windows that cover it. Which of the two factors makes the convolution amplify?

**What.**
- (a) `damage.py --int8`: HAR max and ECG m, the TFLite INT8 model of the same net and seed (the section 2
  control), ±1 step on every 8-bit input, as section 3. Measure per layer (output of each convolution, after its
  pool) the relative error ‖changed − clean‖ / ‖clean‖, and the same measure for the binary network (for ±1 bits it
  is 2·√(share of changed bits)). Growth = last layer / first.
- (b) `spread.py`: one random bit of the output of binary layer l flipped per sample (validation, 3 seeds), the
  next binary layer run once. Per transition l -> l+1: reach = sums of layer l+1 that change, flip chance = share of
  those whose output bit flips, amplification = changed bits that layer l+2 reads (after the pool / OR) per flipped
  bit, share growth = amplification × bits of layer l / bits of layer l+1. Also the share of layer l+1 neurons with
  z − T in [−2, 1] (a ±2 step can flip them) on the clean pass.
  Networks: ResNet-8 base, HAR max, ECG m, MLP 2, 3 and 7 binary layers.

**Rule** (means over 3 seeds, validation):
1. (a) If the INT8 growth is above 2 and at least half the binary growth on the same net, the convolution amplifies
   whatever its arithmetic; if it is below 2, the amplification is a property of the binary convolution.
2. (b) Share growth above 1 grows the damage as sections 3-4 measure it. If the convolutions' mean share growth per
   transition is above 1 and the MLPs' below 1 (or clearly lower), the per-layer spread explains section 4. Then the
   factor with the largest conv / MLP ratio is the mechanism: share growth = reach x flip chance x pool keep (bits
   left after the pool / OR) x size ratio (bits of layer l / bits of layer l+1).

   *Correction before the 3-seed run:* the first version of rule 2 used the amplification (changed bits per flipped
   bit) against 1. A one-seed check (HAR max and the 2-layer MLP, seed 42) showed it does not measure what sections 3-4
   measure: the MLP's amplification is 7.1 and HAR's 1.4-1.8, but sections 3-4 count the share of changed bits, and the
   MLP's next layer has 4x more bits (128 -> 512). Rule 2 now uses the share growth, written before the full run.

**Result** (mean ± sd over 3 seeds, validation; data: `results/damage.json` key `int8`, `results/spread.json`):

(a) Relative error per layer, ±1 on every input, the same inputs through both networks:

| Network | Binary: first → last layer | Binary growth | INT8: first → last layer | INT8 growth |
|---|---|---|---|---|
| HAR max | 0.13 → 0.47 | **3.69 ± 0.08** | 0.030 → 0.026 | **0.85 ± 0.06** |
| ECG m | 0.21 → 0.53 | **2.56 ± 0.27** | 0.06 → 0.07 | **1.23 ± 0.15** |

(b) One bit flipped, per transition l -> l+1 (mean over the transitions of each network):

| Network | Share growth | Reach (sums) | Flip chance | Pool / OR keep | Size ratio | Near threshold |
|---|---|---|---|---|---|---|
| HAR max | 10.2 (6.3, 14.0) | 345 | 1.9 % | 0.27 | 6.0 | 3.7 % |
| ECG m | 7.5 (4.9, 10.1) | 265 | 3.1 % | 0.28 | 3.3 | 6.1 % |
| ResNet-8 base | 4.6 (3.1-7.9) | 204 | 2.3 % | 0.95 | 1.33 | 4.5 % |
| MLP 2 layers | 1.8 | 512 | 1.4 % | 1 | 0.25 | 2.8 % |
| MLP 3 layers | 3.9 (1.3, 6.6) | 512 | 1.2 % | 1 | 0.63 | 2.3 % |
| MLP 7 layers | 2.7 (1.5-5.0) | 512 | 0.7 % (1.4 → 0.3) | 1 | 0.88 | 1.4 % (2.7 → 0.6) |

**Answer.**
1. (a) INT8 does not amplify: its error stays the same from the first to the last convolution (0.85x, 1.23x), while
   the binary one grows 2.6-3.7x on the same nets and inputs. By the rule, the amplification is a property of the
   **binary** convolution, not of the convolution.
2. (b) Rule 2 is **not met**: one flipped bit grows the share of damage in the MLPs too (1.8-3.9x per transition),
   not below 1. The convolutions grow it more (4.6-10.2x), but the one-bit spread alone does not explain why the
   7-layer MLP's real damage fades (section 4) while its one-bit growth stays above 1. When many bits change together
   (sections 3-4) the growth per transition is lower than one bit predicts in every network (e.g. HAR 3.5-3.9x
   measured vs 6.3-14x).
   Factor by factor (descriptive, since the rule's condition failed):
   - **Reach does not make the difference:** a bit reaches *fewer* sums in a convolution (69-482) than in the MLP
     (all 512). The idea that the convolution spreads one bit wider is not what we see.
   - **Flip chance does:** a reached sum flips 2-4x more often in a convolution (1.9-3.1 % vs 0.7-1.4 %), and it
     follows the share of neurons close to their threshold (3.7-6.1 % vs 1.4-2.8 %). The same cause as the ties of
     section 1: the convolutional sums sit closer to the threshold.
   - In the 1D nets the pools keep 27-28 % of the changed bits but shrink the layer 3-6x, so the damage concentrates.
   - In the 7-layer MLP the deeper layers have fewer neurons near the threshold (2.7 % -> 0.6 %): that is why its
     damage fades.

Not measured: why the binary convolution's sums sit closer to the threshold than the MLP's (and than INT8, whose
outputs are not thresholded); and how many changed bits interact (one bit vs several at once).

```
python damage.py har --nets max --seeds 42 43 44 --int8 --out results/damage.json
python damage.py ecg --nets m --seeds 42 43 44 --int8 --out results/damage.json
python spread.py har --nets max --seeds 42 43 44 --out results/spread.json      # also: ecg --nets m, resnet8,
python spread.py mlp --depth 7 --seeds 42 43 44 --out results/spread.json      # mlp, mlp --depth 3
```

## 6. Why do the binary convolution's sums sit closer to the threshold? (`near.py`, rule Nov 25, 2026, before measuring)

**Question.** Section 5 found that a reached sum flips 2-4x more often in a binary convolution than in the MLP,
following the share of neurons near their threshold. Two things set that share. The **spread** of the sum: a neuron
that adds n terms of ±1 has values on a grid of step 2; the wider its spread σ, the less often it lands next to T. And
the **placement** of the threshold: in the middle of the values (bits half +1, half −1) or out in a tail (the bit
almost always the same). For a bell-shaped sum, near ≈ 4·φ(μ/σ)/σ, with μ = mean of z − T and φ the normal density.
Which of the two makes the difference? And is the spread set by the number of terms (σ = c·√n, c = 1 for
independent terms) or by how correlated they are (c)?

**What.** Every binary layer that reads bits (as section 5), clean pass, validation, 3 seeds, networks of section 5.
Per neuron (an output channel of a convolution, all its positions; a unit of the MLP): n = its nonzero weights, μ
and σ of z − T, near = share of z − T in [−2, 1]. Per network: means over the neurons of its layers. Check: predicted
near (4·φ(μ/σ)/σ per neuron) against measured near.

**Rule** (means over 3 seeds; conv = mean of ResNet-8, HAR max, ECG m; MLP = mean of the three MLPs):
1. If predicted near is within 1.5x of measured near in every network, the bell model holds and rule 2 applies;
   otherwise only the measured near is reported.
2. Split the conv / MLP ratio of near into spread (σ_MLP / σ_conv) and placement (φ_conv / φ_MLP): the larger factor
   is the cause. If spread: the larger of √n_MLP / √n_conv and c_MLP / c_conv says whether it is the number of terms or
   their correlation.

**Result** (results/near.json, means ± sd over 3 seeds; σ, √n, c geometric means and φ a mean over the neurons whose
bit moves; "constant" = hidden neurons of these layers whose bit never changes on validation):

| network | near | predicted | constant neurons | near among moving | σ | √n | c | mean φ(μ/σ) |
|---|---|---|---|---|---|---|---|---|
| MLP 2 layers | 2.81 ± 0.04 % | 3.22 % | 62.6 ± 0.1 % | 7.50 ± 0.09 % | 11.9 | 11.3 | 1.06 | 0.262 |
| MLP 3 layers | 2.30 ± 0.03 % | 2.75 % | 63.1 ± 0.1 % | 6.22 ± 0.09 % | 13.8 | 16.9 | 0.82 | 0.263 |
| MLP 7 layers | 1.39 ± 0.03 % | 1.80 % | 71.7 ± 0.1 % | 4.91 ± 0.10 % | 19.8 | 19.3 | 1.02 | 0.307 |
| HAR max | 3.73 ± 0.18 % | 3.91 % | 0 % | 3.73 % | 35.3 | 21.9 | 1.61 | 0.329 |
| ECG m | 6.06 ± 0.92 % | 5.93 % | 0 % | 6.06 % | 19.8 | 14.3 | 1.39 | 0.289 |
| ResNet-8 | 4.01 ± 0.05 % | 4.16 % | 0 % | 4.01 % | 22.4 | 17.0 | 1.32 | 0.229 |

Correction (Nov 25): the first version (commit 1017de5) of this table took the moments of
z − T; the CDT gives a dead neuron a threshold of ±2^31, and E[z²] − μ² then lost all precision, so about a third of
the dead neurons were counted as "moving" with a meaningless σ. The moments are now taken on the raw sum (small
integers, exact). near and predicted did not change; the constant share and the split below did.

1. Predicted / measured near: 0.98-1.29 per network (0.86-1.32 per seed), within 1.5x: the bell model holds.
2. Near, conv / MLP: 4.60 / 2.17 = 2.12. Spread σ_MLP / σ_conv = 15.2 / 25.8 = 0.59 and placement φ_conv / φ_MLP =
   0.282 / 0.277 = 1.02: neither factor makes the convolutions' sums nearer (the rule's "larger factor" is placement,
   at 1.02, i.e. no effect). Among the neurons that move, the MLP's sums are NEARER the threshold (6.21 % vs 4.60 %,
   ratio 0.74 ≈ 0.59 x 1.02 x the rest), because they are narrower (c ≈ 1: independent terms; conv c 1.3-1.6). The
   whole 2.12 comes from the **constant neurons**: 63-72 % of the MLPs' 512-wide hidden neurons give the same bit on
   every sample; the convolutions have none (2.12 / 0.74 = 2.9 ≈ 1 / (1 − 0.66)).

**Answer.** The binary convolution's sums are not nearer the threshold: per neuron that moves, the MLP's are nearer
(its sums are narrower). The MLP looks robust only because about two thirds of its hidden neurons never change, so
they can neither flip nor pass a flip on. Section 7 asks what those neurons are.

## 7. Are the MLP's constant neurons dead weight? (`prune.py`, rule Nov 25, 2026, before measuring)

**Question.** Section 6 found 46-60 % of the binary MLPs' hidden neurons giving the same bit on every validation
sample. If they are constant on the training data too, they can be removed: their bit is a fixed ±1, so its weight
moves into the next layer's threshold (raw − s·c; for the classifier, a fixed integer offset per class on its sum).
Does that change the accuracy?

**What.** The three MLPs of section 4 (2, 3 and 7 binary layers), 3 seeds. A hidden neuron (every hidden layer,
the first included) is constant if its bit is the same on the WHOLE CDT training split. Remove them all, fold their
bits as above, run the exact integer network on validation. Count: neurons removed, validation accuracy before /
after, predictions that change, and the weight bits saved (computed: the neuron's row of weights plus its column in
the next layer; thresholds not counted).

**Rule.** Constant on training data = dead weight if, in every seed, removing them changes validation accuracy by
less than 0.1 points and changes fewer than 0.5 % of predictions. Otherwise report how much they carry. Flash and
speed on a chip are not measured here: the bits saved are a count, not a flash measurement.

**Result** (results/prune.json, 3 seeds; hidden = all hidden neurons, the 128-wide first layer included):

| MLP | removed (constant on train) | of them threshold ±2^31 | validation accuracy before = after | predictions changed | logits equal | weight bits saved |
|---|---|---|---|---|---|---|
| 2 layers (128-512) | 51.3 % (326-330 / 640) | 325-330 | 88.27 ± 0.35 % | 0 % | 100 % | 28.8-31.5 % |
| 3 layers | 56.2 % (645-649 / 1152) | 643-648 | 88.30 ± 0.35 % | 0 % | 100 % | 63.7-64.4 % |
| 7 layers | 69.0 % (2204-2210 / 3200) | 2202-2205 | 87.95 ± 0.36 % | 0 % | 100 % | 83.6-83.8 % |

The rule passes in every seed: the pruned network gives the same integer logits on every validation sample. Almost
all removed neurons (all but 0-5 per network) are ones the exporter already marks as always 1 (threshold −2^31:
`_threshold` in compiler_core/export/dense.py, "ReLU(z) >= 0 >= beta"): they are constant on any input, so removing
those is exact everywhere, not only on these data.

Where they come from (seed 42, FP32 starting model vs trained binary model): the FP32 MLPs have NO dead ReLU unit
(0 % of units always 0 on 10000 training samples, every layer). In the binary model the learned binarization
threshold `beta` of RSign (compiler_core/methods/base.py) is ≤ 0 in 3.9 / 62.5 % of the 128 / 512 channels (2 layers),
0.8 / 68.8 / 56.6 % (3 layers), 2.3-86.5 % growing with depth (7 layers). After a ReLU, beta ≤ 0 makes the bit always
1, and the straight-through gradient then no longer moves that channel: it dies DURING binary training (calibration
places every beta between the median and the next value, above 0 for a live channel).

**Pruned header on the chip** (prune.py writes runs/pruned/{full,pruned}_<d>_s<seed>/model_symbex.h with the
CDT's own exporter; the classifier offset becomes K always-1 neurons in the last hidden layer, K = 51-99, so the
library needs no change; C on the PC 100 % equal to the simulation, 9 / 9 networks). ESP32 classic, seed 42, weights
in flash, standard engine, lut16, 1000 validation images (`chip_test.py --no-int8`), equal to the simulation 100 %
on all 6 runs, same accuracy before and after:

| MLP (seed 42) | hidden widths after | model data in flash (header arrays) | latency ESP32 | model buffers (RAM) |
|---|---|---|---|---|
| 2 layers | 128-512 -> 123-242 | 27.9 -> 20.6 KB | 3.38 -> 2.29 ms (1.47x) | 3.09 -> 2.04 KB |
| 3 layers | 128-512-512 -> 127-160-298 | 64.8 -> 31.1 KB | 11.00 -> 3.99 ms (2.76x) | 3.09 -> 2.26 KB |
| 7 layers | 128-512x6 -> 125-195-216-177-139-72-160 | 212.3 -> 47.7 KB | 32.08 -> 8.38 ms (3.83x) | 3.09 -> 1.94 KB |

Total RAM on the chip stays ~67-68 KB: it is dominated by the 64 KB lut16 popcount table, which does not depend on
the model (`--popcount swar` removes it). Without the K bias neurons (a per-class integer bias in the C classifier,
not written) the flash would be 19.3 / 27.9 / 45.3 KB (computed). The flash numbers are the exporter's count of the
header arrays; the app image shrank more (191.2 -> 173.6 KB, 217.6 -> 190.8 KB, esptool), not understood. In flash the
time also depends on the flash cache: the same 784 -> 128 first layer takes 2.43 ms in the 2-layer net and 3.77 ms
in the 3-layer one, so part of the gain is a smaller model in the cache (not separated). One seed: a ranking, not a
result.

**Answer.** Yes: they are dead weight, removed with no change at all. And they are not a property of binary MLPs but
of the plain dense recipe: binary training drives the 512-wide layers' beta to ≤ 0 and kills 60-70 % of them, so the
Fashion MLP 784-128-512-10 really computes with ~190 live neurons in its second layer. This is the same family of
problem as the frozen latent weights of PR #47, and binary_health does not see it (it checks weights, not beta).
Consequences, NOT measured: (1) a recipe that keeps beta > 0 or binarizes before the ReLU might give back that
capacity, and the MLP's accuracy cost (87.5 vs INT8 89.8 %) might shrink; (2) pruning would cut flash and time of the
512 layers by the saved share (a count of weight bits, not a measurement on a chip); (3) the MLP's lower fragility in
sections 2-6 is partly these dead neurons. Each needs its own rule and, (1), GPU training.
