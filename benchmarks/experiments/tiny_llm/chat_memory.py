"""Whole-turn context retention; never silently cut role markers or the system."""


def retain(system, turns, suffix, reserve=1, context=256):
    """Return (flat IDs, dropped turn count).

    system includes BOS and any system turn; suffix is the open assistant marker.
    turns are complete framed turns (or whole user/assistant exchanges if paired
    retention is desired). Keep a contiguous newest suffix, reserve generation
    space. Oversized system or newest turn is an error, not a partial prompt.
    Rebuild/reset and replay the KV cache after eviction; never slide positions.
    """
    if type(context) is not int or not 1 <= context <= 256 or type(reserve) is not int or not 0 <= reserve <= context:
        raise ValueError("invalid context/reserve")
    spans = [system, *turns, suffix]
    if any(type(t) is not int or not 0 <= t <= 2147483647 for s in spans for t in s):
        raise ValueError("invalid token ID")
    available = context-reserve-len(system)-len(suffix)
    if available < 0 or (turns and len(turns[-1]) > available):
        raise ValueError("system/newest turn does not fit")
    first = len(turns)
    while first and len(turns[first-1]) <= available:
        first -= 1
        available -= len(turns[first])
    return list(system)+[t for turn in turns[first:] for t in turn]+list(suffix), first
