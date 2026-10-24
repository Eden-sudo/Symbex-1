"""Plain 1-bit dense layer: binarize the input and the weights. Nothing else."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import RSign, binarize_weights


class DensePlain(nn.Module):
    def __init__(self, in_features, out_features, bias=False, binarize_input=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rsign = RSign(in_features, dims=2) if binarize_input else nn.Identity()

    def forward(self, x):
        return F.linear(self.rsign(x), binarize_weights(self.weight), self.bias)

    @classmethod
    def from_float(cls, layer, binarize_input=True):
        """Build from a regular nn.Linear, copying its weights as the starting point."""
        new = cls(layer.in_features, layer.out_features,
                  bias=layer.bias is not None, binarize_input=binarize_input)
        new.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            new.bias.data.copy_(layer.bias.data)
        return new

