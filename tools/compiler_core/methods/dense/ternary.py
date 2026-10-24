"""Ternary dense layer (~1.58 bit): weights in {-1, 0, +1} times a per-neuron scale, binarized (+-1) input.

Weights:
    delta_layer = ratio * mean(|W|)                  ONE learnable threshold per layer (ratio starts at 0.7, the
                                                     usual TWN value). mean(|W|) is detached: the ratio is what
                                                     learns, so Adam moves it at a pace independent of the
                                                     weight magnitude.
    t_ij  = sign(W_ij) * [|W_ij| > delta]            0 -> +1 in the sign (never matters: a zero has no sign)
    alpha_j = mean of |W_ij| over the NONZERO weights of neuron j (0 if the row is all zero)
    weight used in the forward pass = alpha_j * t_ij

Gradients (straight-through):
    dt/dW     = 1                                   (identity STE; the weights stay small, no clipping needed)
    dt/dratio = -sign(W) * mean|W| * k(|W| - delta) with k a triangular kernel of half-width delta around the
                                                     threshold: only the weights close to it decide whether
                                                     raising the threshold helps (more zeros) or hurts.

On the chip each weight is TWO bits (sign plane + nonzero mask), and with a +-1 input bit a:
    dot = popcount(mask & ~(a ^ sign)) - popcount(mask & (a ^ sign))
Same integer threshold scheme as the plain method: export/ternary.py folds alpha, bias, BatchNorm and the next
layer's RSign into one int32 per neuron.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import RSign

INITIAL_RATIO = 0.7


class _Ternarize(torch.autograd.Function):
    """Forward: sign(w) * [|w| > delta]. Backward: identity to w, triangular-kernel surrogate to the ratio."""
    @staticmethod
    def forward(ctx, w, ratio, scale):
        delta = ratio * scale
        ctx.save_for_backward(w, delta, scale)
        sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        return sign * (w.abs() > delta).to(w.dtype)

    @staticmethod
    def backward(ctx, grad):
        w, delta, scale = ctx.saved_tensors
        width = delta.clamp(min=1e-8)
        kernel = (1 - ((w.abs() - delta) / width).abs()).clamp(min=0) / width
        sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        grad_ratio = -(grad * sign * kernel).sum() * scale
        return grad, grad_ratio.view_as(delta), None


def ternary_delta(weight, ratio):
    """The layer's zeroing threshold (a scalar tensor), as the forward pass uses it."""
    return ratio.clamp(min=0) * weight.detach().abs().mean()


def ternary_codes(weight, ratio):
    """{-1, 0, +1} codes, computed EXACTLY as the forward pass does (same dtype, same comparison): the exporter
    calls this, so a weight sitting on the threshold cannot end up on different sides in PyTorch and in C."""
    scale = weight.detach().abs().mean()
    return _Ternarize.apply(weight, ratio.clamp(min=0), scale)


def ternary_alpha(weight, codes):
    """Per-neuron scale: mean |W| over the nonzero weights of each row (0 for an all-zero row)."""
    mask = (codes != 0).to(weight.dtype)
    count = mask.sum(dim=1, keepdim=True)
    return (weight.abs() * mask).sum(dim=1, keepdim=True) / count.clamp(min=1)


def ternarize_weights(weight, ratio):
    """alpha_j * t_ij: the weight the forward pass multiplies by."""
    t = ternary_codes(weight, ratio)
    return t * ternary_alpha(weight, t)


class DenseTernary(nn.Module):
    def __init__(self, in_features, out_features, bias=False, binarize_input=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.delta_ratio = nn.Parameter(torch.tensor(INITIAL_RATIO))
        self.rsign = RSign(in_features, dims=2) if binarize_input else nn.Identity()

    def forward(self, x):
        return F.linear(self.rsign(x), ternarize_weights(self.weight, self.delta_ratio), self.bias)

    @torch.no_grad()
    def sparsity(self):
        """Fraction of zero weights right now."""
        return float((ternary_codes(self.weight, self.delta_ratio) == 0).float().mean())

    @classmethod
    def from_float(cls, layer, binarize_input=True):
        """Build from a regular nn.Linear, copying its weights as the starting point."""
        new = cls(layer.in_features, layer.out_features,
                  bias=layer.bias is not None, binarize_input=binarize_input)
        new.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            new.bias.data.copy_(layer.bias.data)
        return new
