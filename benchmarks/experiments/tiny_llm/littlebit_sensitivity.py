"""Weight-reconstruction proxy and greedy packed-payload allocation, without training."""
import importlib.util
import math
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("littlebit", Path(__file__).with_name("littlebit.py"))
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


@torch.no_grad()
def probe(model, bpws, *, residual=False, itq_iters=None, seed=0):
    """Return named candidate lists: rank, actual payload bits, squared Frobenius error.

    Includes every nn.Linear (including a linear head); requires floating,
    bias-free, unquantized projections. Infeasible requested BPWs raise.
    Does not mutate weights or training mode. This is not loss sensitivity.
    """
    bpws = tuple(bpws)
    if not bpws:
        raise ValueError("at least one BPW is required")
    result = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        if layer.bias is not None or getattr(layer, "kind", "fp32") != "fp32":
            raise ValueError("probe requires bias-free float projections")
        candidates = {}
        for bpw in bpws:
            rank = lb.rank_for_bpw(bpw, layer.in_features, layer.out_features, residual)
            if rank not in candidates:
                factor = lb.FactorLinear.from_dense(layer.weight, rank, residual, itq_iters=itq_iters, seed=seed)
                candidates[rank] = {"rank": rank, "bits": 8 * factor.storage_bytes(),
                                    "error": float((factor.dense() - layer.weight).double().square().sum())}
        result[name] = sorted(candidates.values(), key=lambda c: c["bits"])
    if not result:
        raise ValueError("model has no float projections")
    return result


def allocate(candidates, budget_bits):
    """Greedily buy the largest error reduction per added bit that fits.

    Starts at each layer's cheapest candidate; considers jumps over bad ranks.
    Heuristic, not a globally optimal knapsack solver. Returns (selection, used).
    Unreachable budgets leave slack; never invent padding as model capacity.
    """
    if type(budget_bits) is not int or budget_bits < 0 or not candidates:
        raise ValueError("expected nonnegative integer budget and nonempty candidates")
    choices = {}
    for name, options in candidates.items():
        if not options:
            raise ValueError("empty candidate list")
        for c in options:
            if (type(c["bits"]) is not int or c["bits"] <= 0
                    or not math.isfinite(c["error"]) or c["error"] < 0):
                raise ValueError("invalid candidate cost or error")
        choices[name] = dict(min(options, key=lambda c: (c["bits"], c["error"])))
    used = sum(c["bits"] for c in choices.values())
    if used > budget_bits:
        raise ValueError("budget cannot fit cheapest candidates")
    while True:
        best = None
        for name, options in candidates.items():
            old = choices[name]
            for c in options:
                cost, gain = c["bits"] - old["bits"], old["error"] - c["error"]
                if cost > 0 and gain > 0 and used + cost <= budget_bits:
                    if best is None or gain / cost > best[0]:
                        best = (gain / cost, name, c, cost)
        if best is None:
            return choices, used
        _, name, candidate, cost = best
        choices[name] = dict(candidate)
        used += cost
