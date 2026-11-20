"""Compare float and dequantized intsim logits on bounded CPU windows."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("agreement_loss", Path(__file__).with_name("eval_loss.py"))
loss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loss)


@torch.inference_mode()
def report(run, tokens, n_windows, max_tokens, offset=0):
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    model, args, sim, params = loss.load_run(run, integer=True)
    cap = min(max_tokens, n_windows * args["seq"])
    count, matches, windows, max_diff = 0, 0, 0, 0.0
    for x, _ in loss.token_windows(tokens, args["seq"], cap, offset):
        if int(x.max()) >= model.emb.num_embeddings:
            raise ValueError("token ID outside model vocabulary")
        floating = model(x).double()
        integer = sim.float_logits(*sim.forward(params, x))
        if not torch.isfinite(floating).all() or not torch.isfinite(integer).all():
            raise ValueError("nonfinite logits")
        matches += int((floating.argmax(-1) == integer.argmax(-1)).sum())
        count += x.numel()
        max_diff = max(max_diff, float((floating - integer).abs().max()))
        windows += 1
    return {"device": "cpu", "tokens": count, "windows": windows,
            "requested_windows": n_windows, "max_tokens": max_tokens, "offset": offset,
            "context": args["seq"], "top1_matches": matches,
            "top1_agreement": matches / count, "max_logit_difference": max_diff,
            "logit_units": "dequantized logits (not raw integer scores)"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("tokens")
    p.add_argument("--windows", type=int, required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--offset", type=int, default=0)
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    print(json.dumps(report(a.run, a.tokens, a.windows, a.max_tokens, a.offset), allow_nan=False))


if __name__ == "__main__":
    main()
