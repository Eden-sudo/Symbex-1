"""Host ternary linear reference: exact int64 accumulation, float scales.

This is a projection simulator, not the legacy integer transformer's exporter.
"""
import torch


def quantize_input(x):
    """Match train.quant_act8, returning int8 inputs and per-token scales."""
    if not x.is_floating_point() or x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x must have a nonempty floating-point last axis")
    with torch.no_grad():
        inverse = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        q = (x * inverse).round().clamp(-128, 127).to(torch.int8)
    return q, 1.0 / inverse


def accumulate(x, codes):
    """Compute [..., in] @ [out, in].T exactly on CPU, without scales.

    Accept full int8 activation range, including -128. Width is bounded so
    even worst-case accumulation cannot overflow signed int64.
    """
    if x.device.type != "cpu" or codes.device.type != "cpu":
        raise ValueError("integer reference requires CPU tensors")
    if x.dtype != torch.int8 or codes.dtype != torch.int8:
        raise ValueError("x and codes must be int8")
    if codes.ndim != 2 or x.ndim < 1 or not 0 < codes.shape[1] <= (2**63 - 1) // 128:
        raise ValueError("invalid matrix dimensions")
    if x.shape[-1] != codes.shape[1]:
        raise ValueError("input width mismatch")
    if ((codes < -1) | (codes > 1)).any():
        raise ValueError("codes must be ternary")
    return x.to(torch.int64) @ codes.to(torch.int64).T


def dequantize(acc, input_scale, weight_scale):
    """Apply per-token and per-row scales to an exact accumulation.

    Scales come from quantize_input and ternary.quantize, respectively.
    Float32 arithmetic here is deliberately outside the integer kernel.
    """
    if input_scale.shape != (*acc.shape[:-1], 1) or weight_scale.shape != (acc.shape[-1], 1):
        raise ValueError("scale shape mismatch")
    return acc.to(torch.float32) * input_scale * weight_scale[:, 0]
