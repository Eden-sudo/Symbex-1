"""
k-bit layers (k >= 2) for mixed-precision training: the layers the estimate assigns 2, 4 or 8 bits.
Layers assigned "bin" (1 bit) keep using the plain method.

Quantization simulated during training (fake quantization, straight-through estimator for the rounding):
    weights      symmetric, per output channel:  q = clamp(round(w / s), -L, L),  s = max|w| / L,  L = 2^(k-1) - 1
    activations  asymmetric, per tensor, with running min / max (EMA, like BatchNorm statistics):
                 q = clamp(round((x - lo) / s), 0, 2^k - 1),  s = (hi - lo) / (2^k - 1)
The first layer of the model keeps its input untouched (quantize_input=False), the same rule as
apply_method(): how the raw input is encoded is decided upstream.

This simulates the arithmetic to measure ACCURACY in PyTorch. Running k-bit layers on the chip is a
separate step: the C dense layer with k > 1 is not verified yet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_ste(x):
    """round() in the forward pass, identity gradient (straight-through)."""
    return x + (torch.round(x) - x).detach()


def quantize_weight(w, k):
    """Symmetric per-output-channel k-bit fake quantization."""
    L = 2 ** (k - 1) - 1
    s = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp(min=1e-8) / L
    return torch.clamp(_round_ste(w / s), -L, L) * s


class ActivationQuantizer(nn.Module):
    """Asymmetric per-tensor k-bit fake quantization with running min / max."""
    def __init__(self, k, momentum=0.1):
        super().__init__()
        self.k, self.momentum = k, momentum
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "iniciado" in state_dict:                 # checkpoints saved before the translation
            state_dict[prefix + "initialized"] = state_dict.pop(prefix + "iniciado")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                lo, hi = x.min(), x.max()
                if not self.initialized:
                    self.lo.copy_(lo)
                    self.hi.copy_(hi)
                    self.initialized.fill_(True)
                else:
                    self.lo.mul_(1 - self.momentum).add_(self.momentum * lo)
                    self.hi.mul_(1 - self.momentum).add_(self.momentum * hi)
        levels = 2 ** self.k - 1
        s = (self.hi - self.lo).clamp(min=1e-8) / levels
        return torch.clamp(_round_ste((x - self.lo) / s), 0, levels) * s + self.lo


def _old_input_keys(state_dict, prefix):
    """Checkpoints saved before the translation call the input quantizer 'cuant_entrada'."""
    for k in [k for k in state_dict if k.startswith(prefix + "cuant_entrada.")]:
        state_dict[prefix + "input_quant." + k[len(prefix + "cuant_entrada."):]] = state_dict.pop(k)


class DenseKBit(nn.Module):
    def __init__(self, in_features, out_features, k, bias=True, quantize_input=True):
        super().__init__()
        if not (2 <= k <= 16):
            raise ValueError(f"k must be between 2 and 16 (use the plain method for 1 bit), got {k}")
        self.k = k
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.input_quant = ActivationQuantizer(k) if quantize_input else nn.Identity()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _old_input_keys(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        return F.linear(self.input_quant(x), quantize_weight(self.weight, self.k), self.bias)

    @classmethod
    def from_float(cls, layer, k, quantize_input=True):
        """Build from a regular nn.Linear, copying its weights as the starting point."""
        new = cls(layer.in_features, layer.out_features, k, bias=layer.bias is not None,
                  quantize_input=quantize_input)
        new.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            new.bias.data.copy_(layer.bias.data)
        return new


class ConvKBit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, k, stride=1, padding=0, bias=True,
                 quantize_input=True):
        super().__init__()
        if not (2 <= k <= 16):
            raise ValueError(f"k must be between 2 and 16 (use the plain method for 1 bit), got {k}")
        self.k, self.stride, self.padding = k, stride, padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.input_quant = ActivationQuantizer(k) if quantize_input else nn.Identity()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _old_input_keys(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        return F.conv2d(self.input_quant(x), quantize_weight(self.weight, self.k), self.bias,
                        self.stride, self.padding)

    @classmethod
    def from_float(cls, layer, k, quantize_input=True):
        """Build from a regular nn.Conv2d, copying its weights as the starting point."""
        if layer.groups != 1 or layer.dilation != (1, 1):
            raise NotImplementedError("grouped / dilated convolutions are not supported")
        if layer.kernel_size[0] != layer.kernel_size[1] or isinstance(layer.padding, str):
            raise NotImplementedError("only square kernels with numeric padding are supported")
        if layer.stride[0] != layer.stride[1] or layer.padding[0] != layer.padding[1]:
            raise NotImplementedError("only equal stride / padding in both directions are supported")
        new = cls(layer.in_channels, layer.out_channels, layer.kernel_size[0], k, layer.stride[0],
                  layer.padding[0], bias=layer.bias is not None, quantize_input=quantize_input)
        new.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            new.bias.data.copy_(layer.bias.data)
        return new

