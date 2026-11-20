"""CPU sensor-text training entry point; reuses TinyLM without changing TinyStories."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location("sensor_modules", Path(__file__).with_name("_modules.py"))
_modules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_modules)
base = _modules.load("../tiny_llm/train.py")

DEFAULTS = dict(dim=64, layers=2, heads=4, hidden=128, seq=128, kind="binary", head="fp32")


def build_model(vocab, **overrides):
    config = dict(DEFAULTS, **overrides)
    for key in ("dim", "layers", "heads", "hidden", "seq"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 3 <= vocab <= 32768 or config["dim"] % (2 * config["heads"]):
        raise ValueError("vocabulary must fit int16 and head width must be even")
    if config["kind"] not in ("fp32", "binary", "ternary") or config["head"] not in ("fp32", "int4", "int2"):
        raise ValueError("unsupported weight format")
    return base.TinyLM(vocab=vocab, **config), config


def _loss(model, data, seq):
    # Fixed bounded validation prefix: a smoke metric, not a full benchmark.
    usable = min(4, (len(data) - 1) // seq)
    idx = torch.arange(usable)[:, None] * seq + torch.arange(seq + 1)
    window = data[idx].long()
    logits = model(window[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), window[:, 1:].reshape(-1))


def train(data_dir, out, *, vocab, steps=5, batch_size=2, lr=0.003, seed=0, **overrides):
    """Optimize causal loss over prompt + sentence; never read test.bin.

    Default is five smoke steps, not a trained-model recipe. Longer work requires
    explicitly increasing steps. Checkpoints are chosen only by the fixed val prefix.
    """
    if type(steps) is not int or steps <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive integers")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    torch.manual_seed(seed)
    model, config = build_model(vocab, **overrides)
    previous_data = base.DATA
    try:
        base.DATA = str(data_dir)
        splits = {name: base.load_split(name, "cpu") for name in ("train", "val")}
    finally:
        base.DATA = previous_data
    for name, data in splits.items():
        if len(data) <= config["seq"] + 1:
            raise ValueError(f"{name} needs more than seq+1 tokens")
        if data.min().item() < 0 or data.max().item() >= vocab:
            raise ValueError(f"{name} contains token IDs outside the vocabulary")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    best = float("inf")
    for step in range(steps + 1):
        model.eval()
        with torch.no_grad():
            metrics = {name: _loss(model, data, config["seq"]).item() for name, data in splits.items()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("nonfinite loss")
        history.append(dict(step=step, **metrics))
        if metrics["val"] < best:
            best = metrics["val"]
            torch.save({"model": model.state_dict(), "config": dict(vocab=vocab, **config),
                        "step": step, "val_loss": best, "seed": seed}, out / "model.pt")
        if step == steps:
            break
        model.train()
        x, y = base.batch(splits["train"], batch_size, config["seq"], gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    (out / "train.json").write_text(json.dumps({"history": history, "seed": seed,
        "config": dict(vocab=vocab, **config), "steps": steps, "batch_size": batch_size,
        "lr": lr, "device": "cpu", "validation": "first at most 4 nonoverlapping windows"}, indent=2) + "\n")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab", type=int, required=True)
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name, type=type(default), default=default)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())
    data, out = args.pop("data"), args.pop("out")
    torch.set_num_threads(1)
    train(data, out, **args)


if __name__ == "__main__":
    main()
