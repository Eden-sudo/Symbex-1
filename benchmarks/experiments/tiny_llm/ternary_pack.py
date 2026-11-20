"""Canonical row-local ternary payloads; not GGML TQ block layouts.

Map -1,0,+1 to digits 0,1,2. The first weight is the least significant
digit. Unused final digits are zero (the code for -1), not extra weights.
"""
import operator


def _layout(format):
    if format == "2bit":
        return 4, 4
    if format == "base3":
        return 3, 5
    raise ValueError("format must be 2bit or base3")


def payload_bytes(count, format="base3"):
    """Bytes for one row; callers sum rows to account for row padding."""
    count = operator.index(count)
    if count < 0:
        raise ValueError("negative weight count")
    _, group = _layout(format)
    return (count + group - 1) // group


def bytes_per_weight(count, format="base3", scale_bytes=0):
    """Effective row cost including optional scale bytes; empty rows cost 0."""
    size = payload_bytes(count, format)
    scale_bytes = operator.index(scale_bytes)
    if scale_bytes < 0:
        raise ValueError("negative scale size")
    if count == 0:
        if scale_bytes:
            raise ValueError("scale on an empty row")
        return 0.0
    return (size + scale_bytes) / count


def pack(weights, format="base3"):
    """Pack a flat iterable of integral ternary weights into bytes."""
    radix, group = _layout(format)
    values = [operator.index(w) for w in weights]
    if any(w not in (-1, 0, 1) for w in values):
        raise ValueError("weights must be ternary")
    return bytes(sum((w + 1) * radix**j for j, w in enumerate(values[i:i + group]))
                 for i in range(0, len(values), group))


def unpack(data, count, format="base3"):
    """Decode exactly count weights, rejecting invalid codes and padding."""
    expected = payload_bytes(count, format)
    radix, group = _layout(format)
    data = bytes(data)
    if len(data) != expected:
        raise ValueError("payload length mismatch")
    weights = []
    for byte in data:
        if format == "base3" and byte >= 243:
            raise ValueError("invalid base3 byte")
        for _ in range(group):
            digit, byte = byte % radix, byte // radix
            if digit == 3:
                raise ValueError("invalid 2bit code")
            if len(weights) < count:
                weights.append(digit - 1)
            elif digit:
                raise ValueError("noncanonical padding")
    return weights
