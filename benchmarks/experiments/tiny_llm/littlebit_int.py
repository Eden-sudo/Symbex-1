"""Integer reference for byte-padded, LSB-first packed sign factors."""

import operator
import sys


def _integer(value, name, low, high):
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must contain integers") from None
    if not low <= value <= high:
        raise ValueError(f"{name} is out of range")
    return value


def _sequence(values, name, length, low, high):
    try:
        if len(values) != length:
            raise ValueError(f"{name} has an invalid length")
        return [_integer(value, name, low, high) for value in values]
    except TypeError:
        raise ValueError(f"{name} must be a sized integer sequence") from None


def _requantize(acc, multiplier, shift):
    product = acc * multiplier
    magnitude = abs(product)
    if shift:
        magnitude = (magnitude + (1 << (shift - 1))) // (1 << shift)
    magnitude = min(magnitude, 127)
    return -magnitude if product < 0 else magnitude


def factor_int(x, v, u, m1, m2, n_in, rank, n_out, sh1):
    """Return int64-contract outputs as Python integers, without mutating inputs.

    Each packed row starts on a byte boundary, LSB first: 1 is +1, 0 is -1.
    Unused high bits are ignored.
    """
    n_in = _integer(n_in, "n_in", 1, (2**31 - 1) // 128)
    rank = _integer(rank, "rank", 1, (2**31 - 1) // 127)
    n_out = _integer(n_out, "n_out", 1, 2**31 - 1)
    sh1 = _integer(sh1, "sh1", 0, 63)
    vs, us = (n_in + 7) // 8, (rank + 7) // 8
    # sys.maxsize is the maximum length of a Python sequence.
    for name, count in (("v", rank * vs), ("u", n_out * us),
                        ("m1", rank * 4), ("m2", n_out * 4),
                        ("y", n_out * 8)):
        if count > 2 * sys.maxsize + 1:
            raise ValueError(f"{name} size is out of range")
    x = _sequence(x, "x", n_in, -128, 127)
    v = _sequence(v, "v", rank * vs, 0, 255)
    u = _sequence(u, "u", n_out * us, 0, 255)
    m1 = _sequence(m1, "m1", rank, -(2**31), 2**31 - 1)
    m2 = _sequence(m2, "m2", n_out, -(2**31), 2**31 - 1)
    z = []
    for r in range(rank):
        acc = sum(value if (v[r * vs + i // 8] >> (i % 8)) & 1
                  else -value for i, value in enumerate(x))
        z.append(_requantize(acc, m1[r], sh1))
    return [sum(value if (u[o * us + r // 8] >> (r % 8)) & 1
                else -value for r, value in enumerate(z)) * m2[o]
            for o in range(n_out)]
