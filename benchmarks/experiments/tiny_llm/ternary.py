"""BitNet-style absmean ternary weights, using tiny_llm's per-row scale.

Float master weights remain trainable. Codes are {-1, 0, +1}; round uses
ties-to-even, including +/-0.5 -> 0. This is not a packed checkpoint format.
"""
import torch


def quantize(weight):
    """Return int8 codes [out, in] and detached scales [out, 1].

    Preserve the training recipe's 1e-5 scale floor, including all-zero rows.
    The scale is per output row, not the original paper's whole-tensor scale.
    """
    if weight.ndim != 2 or min(weight.shape) == 0 or not weight.is_floating_point():
        raise ValueError("weight must be a nonempty floating-point matrix")
    with torch.no_grad():
        scale = weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        codes = (weight / scale).round().clamp(-1, 1).to(torch.int8)
    return codes, scale


def quant_weight(weight):
    """Dequantized weights with an identity straight-through derivative."""
    codes, scale = quantize(weight)
    quantized = codes.to(weight.dtype) * scale
    return weight + (quantized - weight).detach()
