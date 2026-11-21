"""Deterministic integer top-k sampling with caller-owned RNG state."""

_TABLE = (65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256,
          128, 64, 32, 16, 8, 4, 2, 1)


def _integer(value, name, low, high):
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in range {low}..{high}")


def xorshift32(state):
    """Advance a nonzero 32-bit xorshift state."""
    _integer(state, "state", 1, 0xffffffff)
    state ^= (state << 13) & 0xffffffff
    state ^= state >> 17
    state ^= (state << 5) & 0xffffffff
    return state & 0xffffffff


def exp_q(distance, shift):
    """Q16 staircase 2^(-floor(distance/shift)); no interpolation."""
    _integer(distance, "distance", 0, 0xffffffffffffffff)
    _integer(shift, "shift", 0, 0xffffffff)
    if shift == 0:
        return 0
    bucket = distance // shift
    return _TABLE[bucket] if bucket < len(_TABLE) else 0


def sample_topk(scores, k, shift, state):
    """Return (token, next_state), ranking equal scores by lower token ID."""
    from collections.abc import Sequence

    if not isinstance(scores, Sequence):
        raise TypeError("scores must be a finite sequence of built-in integers")
    if not 1 <= len(scores) <= 2147483647:
        raise ValueError("scores length must be in range 1..2147483647")
    _integer(k, "k", 1, 2147483647)
    _integer(shift, "shift", 1, 0xffffffff)
    _integer(state, "state", 1, 0xffffffff)
    for score in scores:
        _integer(score, "scores", -(1 << 63), (1 << 63) - 1)
    ids = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    maximum = scores[ids[0]]
    weights = [exp_q(maximum - scores[i], shift) for i in ids]
    next_state = xorshift32(state)
    r = next_state % sum(weights)
    cumulative = 0
    for token, weight in zip(ids, weights):
        cumulative += weight
        if cumulative > r:
            return token, next_state
