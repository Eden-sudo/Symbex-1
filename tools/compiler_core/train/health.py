"""
health.py -- is a binary network still learning? Two checks on the latent (float) weights behind the 1-bit ones.

    clip_latent(weights)                    clamp every latent weight to [-1, 1] in place (BinaryConnect clipping):
                                            call it after each optimizer step
    binary_health(weights, initial=None)    % of latent weights beyond |w| > 1 ("frozen"), largest |w|, and, given
                                            the weights at the start, % whose sign differs from it now (bits that moved)
    warn(health, where, limit=1.0)          prints a warning when more than `limit` % of the weights are frozen

Why: the straight-through sign of these benchmarks passes the gradient only where |w| <= 1. A latent weight that an
Adam step pushes past 1 gets zero gradient from then on and its bit never flips again: the bit is frozen for the rest
of the training. Nothing in the exactness tests sees this (the frozen network is still exact), and it went unnoticed
in ResNet-8 (52 % frozen), HAR, ECG and FOMO until November 13, 2026. Bop (bop.py) keeps no latent weights (every
weight is +-MAGNITUDE): 0 % frozen by construction.
"""
import torch


@torch.no_grad()
def clip_latent(weights):
    for w in weights:
        w.clamp_(-1, 1)


@torch.no_grad()
def binary_health(weights, initial=None):
    """dict(n, frozen, max_abs[, flipped]): frozen and flipped in % of all the latent weights. `initial` is a list of
    tensors (the weights before training, same order); flipped counts the bits that differ from it now."""
    n = sum(w.numel() for w in weights)
    frozen = sum(int((w.abs() > 1).sum()) for w in weights)
    out = dict(n=n, frozen=round(100.0 * frozen / max(n, 1), 3),
               max_abs=round(max((float(w.abs().max()) for w in weights), default=0.0), 4))
    if initial is not None:
        flipped = sum(int(((w >= 0) != (w0.to(w.device) >= 0)).sum()) for w, w0 in zip(weights, initial))
        out["flipped"] = round(100.0 * flipped / max(n, 1), 3)
    return out


def warn(health, where, limit=1.0):
    if health["frozen"] > limit:
        print(f"    [!] {where}: {health['frozen']:.1f} % of the 1-bit weights are frozen (|w| > 1, no gradient: "
              f"their bit can no longer flip; max |w| {health['max_abs']:.3f}). Clip them (clip_latent) or use Bop.")
