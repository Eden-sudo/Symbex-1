"""
Sign function of the gated / bitslice C++ engines: 0 -> -1, clipped straight-through gradient.

Kept as its own name (BipolarSTE, BipolarStepSTE) because symbex_compiler.py, gated.py and the
simulators use it, and because those engines were written with the 0 -> -1 convention.
It no longer duplicates code: it is the shared implementation in ../base.py with cero=-1,
gradiente="recto". Behavior is identical to the previous version of this file.
"""
import torch.nn as nn

from ..base import _SignRecto


class BipolarSTE:
    """Same interface as before: BipolarSTE.apply(x). 0 -> -1, gradient 1 inside |x| <= 1, else 0."""
    @staticmethod
    def apply(x):
        return _SignRecto.apply(x, -1)


class BipolarStepSTE(nn.Module):
    """PyTorch Module wrapper for BipolarSTE."""
    def forward(self, x):
        return BipolarSTE.apply(x)
