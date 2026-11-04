"""
bop.py -- Bop (Helwegen et al. 2019, "Latent Weights Do Not Exist"), an optimizer for 1-bit weights.

A binary weight has no magnitude that matters: only its sign reaches the forward pass. Adam on a "latent" real weight
makes the signs oscillate (the binary ResNet-8 of cnn_cifar_resnet8: a weight changes sign ~7.5 times per epoch for each
time it really moves, benchmarks/experiments/train_gap_study.py). Bop keeps no latent weight: an exponential moving
average m of each weight's gradient, and the weight flips only when m is larger than a threshold tau and pushes
against its current sign.

    Bop(weights, gamma=1e-4, tau=1e-8)    the 1-bit weights (the other parameters stay with Adam)
    bop.step(scale)                       after loss.backward(); scale multiplies gamma (the learning rate's schedule)
    bop.zero_grad()

The weights are stored as +-MAGNITUDE (0.5): the same signs as +-1, and inside the window where every straight-through
estimator in use passes gradient (model.py's |x| <= 1, Bi-Real's 2 - 2|x|, which is 0 at |x| = 1).
"""
import torch

MAGNITUDE = 0.5


class Bop:
    def __init__(self, weights, gamma=1e-4, tau=1e-8):
        if not 0 < gamma <= 1:
            raise ValueError(f"Bop: gamma {gamma} must be in (0, 1]")
        if tau < 0:
            raise ValueError(f"Bop: tau {tau} must be >= 0")
        self.weights, self.gamma, self.tau = list(weights), gamma, tau
        self.m = [torch.zeros_like(w) for w in self.weights]
        with torch.no_grad():
            for w in self.weights:
                w.copy_(torch.where(w >= 0, MAGNITUDE, -MAGNITUDE))      # sign 0 -> +1, as the export reads it

    @torch.no_grad()
    def step(self, scale=1.0):
        g = self.gamma * scale
        for w, m in zip(self.weights, self.m):
            if w.grad is None:
                continue
            m.mul_(1 - g).add_(w.grad, alpha=g)
            flip = (m.abs() > self.tau) & ((m >= 0) == (w >= 0))       # descent wants the weight to change sign
            w.copy_(torch.where(flip, -w, w))

    def zero_grad(self):
        for w in self.weights:
            w.grad = None
