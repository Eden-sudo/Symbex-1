"""
Pieces shared by ALL binarization methods.
If something is used by more than one method, it lives here (not copied around).

Sign conventions (both exist because each C engine was written with one of them):
    binarize(x)                       0 -> +1, Bi-Real triangular gradient (default; plain methods, SYMBEX v3 engine)
    binarize(x, zero=-1)              0 -> -1 (gated / bitslice C++ engines, see dense/core.py)
    binarize(x, gradient="straight")  clipped straight-through gradient: 1 where |x| <= 1, else 0
The defaults are unchanged: binarize(x) behaves exactly as before.
(The old Spanish keyword names cero= / gradiente= and the value "recto" are still accepted.)
"""
import torch
import torch.nn as nn


def _sign(x, zero):
    one = torch.ones_like(x)
    return torch.where(x >= 0, one, -one) if zero > 0 else torch.where(x > 0, one, -one)


class _Sign(torch.autograd.Function):
    """0 -> +1, Bi-Real triangular gradient (the original behavior of this file)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _sign(x, +1)                             # 0 -> +1 (same as the C engine)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (2 - 2 * x.abs()).clamp(min=0)   # Bi-Real triangular gradient


class _SignClipped(torch.autograd.Function):
    """Clipped straight-through gradient (1 inside |x| <= 1, 0 outside). `zero` chooses where 0 goes."""
    @staticmethod
    def forward(ctx, x, zero):
        ctx.save_for_backward(x)
        return _sign(x, zero)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        g = grad.clone()
        g[x.abs() > 1.0] = 0
        return g, None


class _SignBiRealNeg(torch.autograd.Function):
    """0 -> -1 with the Bi-Real triangular gradient (only for completeness of the options)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _sign(x, -1)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (2 - 2 * x.abs()).clamp(min=0)


def binarize(x, zero=1, gradient="bireal", cero=None, gradiente=None):
    """x -> {-1, +1}.
    zero: +1 (default, 0 -> +1) or -1 (0 -> -1). gradient: "bireal" (default) or "straight".
    cero / gradiente: old names of the same arguments (kept until every caller is translated)."""
    if cero is not None:
        zero = cero
    if gradiente is not None:
        gradient = gradiente
    if gradient == "recto":
        gradient = "straight"
    if zero not in (1, -1):
        raise ValueError(f"zero must be +1 or -1, got {zero}")
    if gradient == "bireal":
        return _Sign.apply(x) if zero > 0 else _SignBiRealNeg.apply(x)
    if gradient == "straight":
        return _SignClipped.apply(x, zero)
    raise ValueError(f"gradient must be 'bireal' or 'straight', got {gradient!r}")


class RSign(nn.Module):
    """Binarizes an INPUT with a learned per-channel threshold: sign(x - beta).
    dims=4 for conv maps (N,C,H,W), dims=2 for flat features (N,F)."""
    def __init__(self, channels, dims=4):
        super().__init__()
        shape = (1, channels, 1, 1) if dims == 4 else (1, channels)
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return binarize(x - self.beta)


def binarize_weights(w):
    """sign(w) * alpha, where alpha = mean(|w|) per output channel."""
    alpha = w.abs().mean(dim=list(range(1, w.dim())), keepdim=True)
    return binarize(w) * alpha

