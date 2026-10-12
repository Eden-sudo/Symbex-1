"""Plain 1-bit Conv2d: binarize the input and the weights. Nothing else."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import RSign, binarize_weights


class ConvPlain(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, bias=False, binarize_input=True):
        super().__init__()
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.rsign = RSign(in_channels) if binarize_input else nn.Identity()

    def forward(self, x):
        return F.conv2d(self.rsign(x), binarize_weights(self.weight),
                        self.bias, self.stride, self.padding)

    @classmethod
    def from_float(cls, layer, binarize_input=True):
        """Build from a regular nn.Conv2d, copying its weights as the starting point."""
        if layer.groups != 1 or layer.dilation != (1, 1):
            raise NotImplementedError("grouped / dilated convolutions are not supported")
        if layer.kernel_size[0] != layer.kernel_size[1] or isinstance(layer.padding, str):
            raise NotImplementedError("only square kernels with numeric padding are supported")
        if layer.stride[0] != layer.stride[1] or layer.padding[0] != layer.padding[1]:
            raise NotImplementedError(f"only equal stride / padding in both directions are supported "
                                      f"(got stride {layer.stride}, padding {layer.padding})")
        new = cls(layer.in_channels, layer.out_channels, layer.kernel_size[0],
                  layer.stride[0], layer.padding[0],
                  bias=layer.bias is not None, binarize_input=binarize_input)
        new.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            new.bias.data.copy_(layer.bias.data)
        return new
