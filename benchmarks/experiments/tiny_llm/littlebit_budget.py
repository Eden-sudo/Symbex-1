"""Packed factor payload budgets; headers excluded, scales default to FP16."""
import math


def storage_bytes(rank, in_features, out_features, residual=False, scale_bits=16):
    for name, value in (("rank", rank), ("in_features", in_features), ("out_features", out_features), ("scale_bits", scale_bits)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if rank > min(in_features, out_features) or scale_bits % 8:
        raise ValueError("rank exceeds dimensions or scale_bits is not byte aligned")
    if type(residual) is not bool:
        raise ValueError("residual must be bool")
    # U rows are outputs; V^T rows are latent channels, matching the C kernel.
    signs = out_features * ((rank + 7) // 8) + rank * ((in_features + 7) // 8)
    scales = (in_features + out_features + rank) * (scale_bits // 8)
    return (1 + residual) * (signs + scales)


def bits_per_weight(rank, in_features, out_features, residual=False, scale_bits=16):
    return 8 * storage_bytes(rank, in_features, out_features, residual, scale_bits) / (in_features * out_features)


def rank_for_bpw(target_bpw, in_features, out_features, residual=False, scale_bits=16):
    """Largest legal rank whose padded payload fits; raise if rank one cannot fit."""
    minimum = bits_per_weight(1, in_features, out_features, residual, scale_bits)
    if isinstance(target_bpw, bool) or not isinstance(target_bpw, (int, float)) or not math.isfinite(target_bpw):
        raise ValueError("target_bpw must be finite")
    if target_bpw < minimum:
        raise ValueError(f"target_bpw cannot fit rank one ({minimum:g} BPW)")
    low, high = 1, min(in_features, out_features)
    while low < high:
        mid = (low + high + 1) // 2
        if bits_per_weight(mid, in_features, out_features, residual, scale_bits) <= target_bpw:
            low = mid
        else:
            high = mid - 1
    return low
