"""Integer reference for byte-padded, LSB-first packed sign factors."""

import operator
import sys
import math

SCALE_SHIFT = 20
OUTPUT_SHIFT = 12


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


def _round_shift(value, shift):
    magnitude = (abs(value) + (1 << (shift - 1))) >> shift
    return -magnitude if value < 0 else magnitude


def _pack_rows(rows):
    packed = bytearray()
    for row in rows:
        for start in range(0, len(row), 8):
            packed.append(sum((value >= 0) << bit for bit, value in enumerate(row[start:start + 8])))
    return bytes(packed)


def integerize(layer, input_bound=1.0):
    """Convert a FactorLinear; input q represents q*input_bound/127.

    Conservative per-path bounds avoid saturation for |q|<=127. All three
    learned scales survive: g before V, l between factors, h after U.
    Multipliers are signed int32/Q20; output is int32/Q12. No float arithmetic
    is used by forward_int. Quantization uses half-away rounding at runtime.
    """
    if not math.isfinite(input_bound) or input_bound <= 0:
        raise ValueError("input_bound must be finite and positive")
    result = {"n_in": layer.in_features, "rank": layer.rank, "n_out": layer.out_features,
              "input_scale": input_bound / 127, "paths": []}

    def multipliers(values):
        output = []
        for value in values:
            scaled = float(value) * (1 << SCALE_SHIFT)
            if not math.isfinite(scaled) or abs(scaled) > 2**31 - 1:
                raise ValueError("scale multiplier exceeds int32")
            output.append(int(math.copysign(math.floor(abs(scaled) + 0.5), scaled)))
        return output

    for path in layer.paths:
        g, l, h = (getattr(path, name).detach().double().cpu().tolist() for name in ("g", "l", "h"))
        if not all(math.isfinite(v) for values in (g, l, h) for v in values):
            raise ValueError("scales must be finite")
        # Nonzero fallback units make zero factors exactly zero without division by zero.
        g_step = max(map(abs, g)) * input_bound / 127 or 1.0
        latent_step = layer.in_features * g_step * max(map(abs, l)) or 1.0
        u, v = path.U.detach().cpu(), path.V.detach().cpu()
        if not bool(u.isfinite().all() and v.isfinite().all()):
            raise ValueError("factors must be finite")
        result["paths"].append({"v": _pack_rows(v.T.tolist()), "u": _pack_rows(u.tolist()),
                                "mg": multipliers(x * result["input_scale"] / g_step for x in g),
                                "m1": multipliers(x * g_step / latent_step for x in l),
                                "m2": multipliers(x * latent_step * (1 << OUTPUT_SHIFT) for x in h)})
    validate_integer_layer(result)
    return result


def validate_integer_layer(layer):
    n = _integer(layer["n_in"], "n_in", 1, (2**31 - 1) // 128)
    r = _integer(layer["rank"], "rank", 1, min(n, (2**31 - 1) // 127))
    o = _integer(layer["n_out"], "n_out", r, 2**31 - 1)
    if not math.isfinite(layer["input_scale"]) or layer["input_scale"] <= 0:
        raise ValueError("input_scale must be finite and positive")
    if len(layer["paths"]) not in (1, 2):
        raise ValueError("expected one or two paths")
    bounds = [0] * o
    for path in layer["paths"]:
        for name, count, low, high in (("v", r * ((n + 7) // 8), 0, 255),
                                       ("u", o * ((r + 7) // 8), 0, 255),
                                       ("mg", n, -2**31, 2**31 - 1),
                                       ("m1", r, -2**31, 2**31 - 1),
                                       ("m2", o, -2**31, 2**31 - 1)):
            _sequence(path[name], name, count, low, high)
        for j, mult in enumerate(path["m2"]):
            bounds[j] += _round_shift(r * 127 * abs(mult), SCALE_SHIFT)
    if any(bound > 2**31 - 1 for bound in bounds):
        raise ValueError("output bound exceeds int32")


def forward_int(layer, x):
    """One int8 vector -> Q12 output, including signed scales and residual pair."""
    validate_integer_layer(layer)
    n, r, o = (layer[key] for key in ("n_in", "rank", "n_out"))
    x = _sequence(x, "x", n, -128, 127)
    output = [0] * o
    for path in layer["paths"]:
        scaled = [_requantize(value, mult, SCALE_SHIFT) for value, mult in zip(x, path["mg"])]
        raw = factor_int(scaled, path["v"], path["u"], path["m1"], path["m2"], n, r, o, SCALE_SHIFT)
        output = [a + _round_shift(b, SCALE_SHIFT) for a, b in zip(output, raw)]
    return output
