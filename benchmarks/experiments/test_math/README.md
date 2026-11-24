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
