"""Bounded CPU next-token evaluation of stock TinyLM run directories."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


def local(name):
    spec = importlib.util.spec_from_file_location(
        "tiny_eval_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    # intsim's legacy absolute import must resolve to this worktree's train.
    if name == "intsim":
        previous = sys.modules.get("train")
        sys.modules["train"] = local("train")
        try:
            spec.loader.exec_module(module)
        finally:
            if previous is None:
                del sys.modules["train"]
            else:
                sys.modules["train"] = previous
    else:
        spec.loader.exec_module(module)
    return module


def load_run(run, integer=False):
    """Load tensor-only stock checkpoints on CPU; never infer vocabulary from data."""
    ck = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=True)
    a = ck["args"]
    t = local("train")
    model = t.TinyLM(
        ck["model"]["emb.weight"].shape[0], a["dim"], a["layers"], a["heads"],
        a["hidden"], a["seq"], a["weights"], head=a.get("head", "fp32"),
        int8_blocks=tuple(a.get("int8_blocks", [])), acts=a.get("acts", 8),
        act_opts={"ste_mode": a.get("ste", "identity"), "ste_k": a.get("ste_k", 2.0),
                  "learn_thr": a.get("learn_thr", False)},
        a8_inputs=tuple(a.get("a8_inputs", [])),
        w2_acts=t.W2_ACTS[a.get("w2_acts", "same")], ffn_act=a.get("ffn_act", "silu"),
        rotate=tuple(a.get("hadamard", [])), a8_blocks=tuple(a.get("a8_blocks", [])),
        kv_heads=a.get("kv_heads")).eval()
    model.load_state_dict(ck["model"], strict=True)
    sim, params = None, None
    if integer:
        if a["weights"] not in ("binary", "ternary") or a["dim"] != 64 * a["heads"]:
            raise ValueError("intsim requires binary/ternary weights and 64-wide heads")
        sim = local("intsim")
        # Reuse the legacy loader's unsupported-option guards.
        checked, _ = sim.load_model(str(run), model.emb.num_embeddings, "cpu")
        params = sim.integerize(checked, a["weights"])
    return model, a, sim, params


def token_windows(path, seq, max_tokens, offset=0):
    """Read at most max_tokens targets plus one context token, little-endian u16.

    Adjacent windows share one boundary token; context resets each window.
    A short final window contributes its actual target count, not a window mean.
    """
    if seq <= 0 or max_tokens <= 0 or offset < 0:
        raise ValueError("seq/max_tokens must be positive and offset nonnegative")
    if Path(path).stat().st_size % 2:
        raise ValueError("token file has an incomplete uint16")
    with open(path, "rb") as stream:
        stream.seek(offset * 2)
        data = np.fromfile(stream, dtype="<u2", count=max_tokens + 1)
    if len(data) < 2:
        raise ValueError("slice needs at least two tokens")
    for start in range(0, len(data) - 1, seq):
        row = torch.from_numpy(data[start:start + seq + 1].astype(np.int64))
        yield row[:-1].unsqueeze(0), row[1:].unsqueeze(0)


@torch.inference_mode()
def evaluate(run, tokens, max_tokens, backend="float", offset=0):
    if backend not in ("float", "intsim"):
        raise ValueError("backend must be float or intsim")
    model, args, sim, params = load_run(run, backend == "intsim")
    total, count, windows = 0.0, 0, 0
    for x, y in token_windows(tokens, args["seq"], max_tokens, offset):
        if max(int(x.max()), int(y.max())) >= model.emb.num_embeddings:
            raise ValueError("token ID outside model vocabulary")
        logits = model(x) if sim is None else sim.float_logits(*sim.forward(params, x))
        loss = torch.nn.functional.cross_entropy(
            logits.double().flatten(0, 1), y.flatten(), reduction="sum")
        total += loss.item()
        count += y.numel()
        windows += 1
    loss = total / count
    if not math.isfinite(loss):
        raise ValueError("nonfinite loss")
    return {"backend": backend, "device": "cpu", "loss": loss,
            "perplexity": math.exp(loss), "tokens": count, "windows": windows,
            "offset": offset, "max_tokens": max_tokens, "context": args["seq"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("tokens")
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--backend", choices=("float", "intsim"), default="float")
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    print(json.dumps(evaluate(a.run, a.tokens, a.max_tokens, a.backend, a.offset), allow_nan=False))


if __name__ == "__main__":
    main()
