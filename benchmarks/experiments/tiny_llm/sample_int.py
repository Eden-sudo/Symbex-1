"""Deterministic integer top-k sampling with caller-owned RNG state."""

_TABLE = (65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256,
          128, 64, 32, 16, 8, 4, 2, 1)


def _integer(value, name, low, high):
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in range {low}..{high}")


def xorshift32(state):
    """Xorshift32 (13,17,5), modulo 2^32; reject seed zero, no global state.

    Seed 1 begins 270369, 67634689, 2647435461. C: tinylm_rng.h.
    """
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


def sample(scores, k, temperature, top_p, state):
    """Integer temperature = score units per log2 weight halving; 0 = greedy.

    k=0 means whole vocabulary. top_p is 1..65536 (Q16), applied after
    top-k: keep the shortest prefix reaching ceil(total * top_p / 65536).
    Ties prefer smaller IDs. Greedy does not advance RNG; stochastic draws
    advance once, using next_state % retained_total (intentional modulo bias).
    """
    from collections.abc import Sequence

    if not isinstance(scores, Sequence):
        raise TypeError("scores must be a finite sequence")
    if not 1 <= len(scores) <= 2147483647:
        raise ValueError("scores must not be empty")
    _integer(k, "k", 0, 2147483647)
    _integer(temperature, "temperature", 0, 0xffffffff)
    _integer(top_p, "top_p", 1, 65536)
    _integer(state, "state", 1, 0xffffffff)
    for score in scores:
        _integer(score, "scores", -(1 << 63), (1 << 63) - 1)
    ids = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k or len(scores)]
    if temperature == 0:
        return ids[0], state
    weights = [exp_q(scores[ids[0]] - scores[i], temperature) for i in ids]
    target = (sum(weights) * top_p + 65535) // 65536
    total = 0
    for end, weight in enumerate(weights, 1):
        total += weight
        if total >= target:
            break
    state = xorshift32(state)
    draw = state % total
    for token, weight in zip(ids[:end], weights[:end]):
        if draw < weight:
            return token, state
        draw -= weight
