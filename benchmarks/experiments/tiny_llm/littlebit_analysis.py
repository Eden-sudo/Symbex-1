"""Arithmetic-only storage/operation estimates, not latency or quality results."""
import argparse
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("littlebit_budget", Path(__file__).with_name("littlebit_budget.py"))
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def estimate(target_bpw=None, *, dim=384, layers=8, hidden=1024, vocab=4096, residual=False):
    """Count seven block projections and one shared int8 embedding/head.

    Binary accumulation counts one signed add per stored sign; int8 head
    counts two operations per MAC. Scale applications count one multiply.
    Excludes attention, norms, nonlinearities, quantization, KV, tables and
    headers except the 40-byte header of each standalone factored blob.
    Float payload uses FP16 scales; integer blob uses int32 multipliers.
    """
    for value in (dim, layers, hidden, vocab):
        if type(value) is not int or value <= 0:
            raise ValueError("model dimensions must be positive integers")
    shapes = [(dim, dim)] * 4 + [(dim, hidden)] * 2 + [(hidden, dim)]
    signs = scales = payload = integer_bytes = weights = 0
    ranks = []
    for n, o in shapes:
        weights += n * o
        if target_bpw is None:
            ranks.append(None)
            packed = o * ((n + 7) // 8)
            payload += packed + 2 * o
            integer_bytes += packed + 4 * o
            signs += n * o
            scales += o
        else:
            r = budget.rank_for_bpw(target_bpw, n, o, residual)
            ranks.append(r)
            payload += budget.storage_bytes(r, n, o, residual)
            integer_bytes += 40 + budget.storage_bytes(r, n, o, residual, 32)
            signs += (1 + residual) * r * (n + o)
            scales += (1 + residual) * (n + r + o)
    # Residual summation adds one operation per output projection channel.
    residual_adds = sum(o for _, o in shapes) if target_bpw is not None and residual else 0
    head_bytes = dim * vocab + 4 * vocab
    head_ops = 2 * dim * vocab + vocab
    return {"ranks": ranks, "linear_weights": layers * weights,
            "linear_payload_bytes": layers * payload,
            "integer_weight_bytes": layers * integer_bytes + head_bytes,
            "linear_signed_adds": layers * signs,
            "linear_scale_multiplies": layers * scales,
            "estimated_ops_per_token": layers * (signs + scales + residual_adds) + head_ops,
            "head_bytes": head_bytes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--targets", nargs="+", type=float, default=[0.1, 0.3, 0.5, 1.0])
    args = parser.parse_args(argv)
    print("d384, 8 layers, hidden1024, vocab4096; shared int8 head")
    print("Estimates only: cached latent execution; excludes attention/KV/norms/tables/quantization.")
    print("target_bpw ranks(q,w1,w2) FP16_scale_payload_B integer_weights_B estimated_ops/token")
    for target in [None, *args.targets]:
        try:
            row = estimate(target, residual=args.residual)
        except ValueError as error:
            print(f"{target:g} infeasible: {error}")
            continue
        ranks = row["ranks"]
        label = "1-bit" if target is None else f"{target:g}"
        print(label, (ranks[0], ranks[4], ranks[6]), row["linear_payload_bytes"],
              row["integer_weight_bytes"], row["estimated_ops_per_token"])


if __name__ == "__main__":
    main()
