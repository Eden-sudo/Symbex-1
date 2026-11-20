"""Additive presence penalty in the original int64 logit units."""


def penalize(scores, history, penalty):
    """Subtract once per distinct seen token; saturate, never mutate inputs."""
    if type(penalty) is not int or not 0 <= penalty <= (1 << 63)-1:
        raise ValueError("penalty must be a nonnegative int64")
    if not scores or any(type(z) is not int or not -(1 << 63) <= z < (1 << 63) for z in scores):
        raise ValueError("scores must be nonempty int64 values")
    if any(type(t) is not int or not 0 <= t < len(scores) for t in history):
        raise ValueError("history token out of range")
    seen = set(history)
    return [max(-(1 << 63), z-penalty) if i in seen else z for i,z in enumerate(scores)]
