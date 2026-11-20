"""Synthetic reconstruction/output proxy, not perplexity or a trained-model result."""
import argparse
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("littlebit", Path(__file__).with_name("littlebit.py"))
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)
BPWS = (0.1, 0.3, 0.55, 1.0)


@torch.no_grad()
def compare(weight, inputs, *, bpws=BPWS, itq_iters=3, seed=0):
    """Compare a float linear teacher to binary and all ITQ/residual combinations.

    Costs include FP16 scales and padded signs, exclude headers. Report raw
    errors: increasing rank does not guarantee monotone binary factor quality.
    """
    n_out, n_in = weight.shape
    target = inputs @ weight.T
    def metrics(approx):
        return {"weight_mse": float((approx - weight).square().mean()),
                "output_mse": float((inputs @ approx.T - target).square().mean())}
    binary = lb._train.quant_weight(weight, "binary")
    rows = [dict(kind="1-bit", target_bpw=1., actual_bpw=8 * (n_out * ((n_in + 7)//8) + 2*n_out)/(n_out*n_in),
                 itq=False, residual=False, rank=None, **metrics(binary))]
    for residual in (False, True):
        for itq in (False, True):
            for bpw in bpws:
                row = dict(kind="LittleBit", target_bpw=bpw, itq=itq, residual=residual)
                try:
                    rank = lb.rank_for_bpw(bpw, n_in, n_out, residual)
                except ValueError as exc:
                    row.update(rank=None, actual_bpw=None, weight_mse=None, output_mse=None, reason=str(exc))
                else:
                    layer = lb.FactorLinear.from_dense(weight, rank, residual,
                                                      itq_iters=itq_iters if itq else None, seed=seed)
                    row.update(rank=rank, actual_bpw=layer.bits_per_weight(), **metrics(layer.dense()))
                rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.size <= 0:
        parser.error("size must be positive")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        gen = torch.Generator().manual_seed(args.seed)
        weight = torch.randn(args.size, args.size, generator=gen) / args.size**.5
        inputs = torch.randn(4, args.size, generator=gen)
        rows = compare(weight, inputs, seed=args.seed)
    finally:
        torch.set_num_threads(previous)
    print("Synthetic random linear teacher; no training. BPW includes FP16 scales, excludes headers.")
    print("kind       target actual ITQ residual rank weight_MSE output_MSE")
    for r in rows:
        prefix = f"{r['kind']:10} {r['target_bpw']:.2f}"
        if r["actual_bpw"] is None:
            print(f"{prefix} infeasible ITQ={r['itq']} residual={r['residual']}: {r['reason']}")
        else:
            print(f"{prefix} {r['actual_bpw']:.4f} {int(r['itq'])} {int(r['residual'])} "
                  f"{str(r['rank']):>4} {r['weight_mse']:.6g} {r['output_mse']:.6g}")
    return rows


if __name__ == "__main__":
    main()
