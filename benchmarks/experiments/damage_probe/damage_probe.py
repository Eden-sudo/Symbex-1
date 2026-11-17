"""
damage_probe.py -- which cheap measure, taken on a trained model, ranks precision changes the way full training does?
README.md has the question, the 14 pairs of trained tiny_llm runs (the truth), the four measures and the rule written
before measuring. For each pair and seed: load the trained A, apply B's change and compute

    cos      sum over the changed layers of (1 - cosine) of the layer output, same input
    loss0    validation loss with the change, no retraining, minus A's
    fisher   1/2 sum g^2 dy^2 per token over the changed layers (dy: output change, g: loss gradient at the output)
    recover  loss after a 500-step fine-tune with the change minus the same fine-tune of A unchanged

then the Spearman correlation of each measure with the truth (val loss of B - val loss of A, from train.json).

    python damage_probe.py --seeds 42 43 44 --runs <main>/benchmarks/experiments/tiny_llm/runs \
        --data <main>/benchmarks/data/tinystories                   # results/damage_probe.json
    python damage_probe.py --seeds 42 --no-recover                  # static measures only (minutes)
    python damage_probe.py --seeds 42 --pairs head_bin --ft-steps 20 --eval-batches 2 --out smoke.json   # quick check
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tiny_llm"))
import train as T  # noqa: E402

LIN = ("q", "k", "v", "o", "w1", "w3", "w2")
CLIP1 = {"ste_mode": "clip", "ste_k": 1.0}  # the W1A1 runs' straight-through gradient (only matters in fine-tuning)

# pair: (run A, run B, change). change: ("head", kind) | ("kind", kind, blocks or None = all) |
# ("acts", bits, layer names)
PAIRS = {
    "head_bin": ("binary_d384l8", "binary_d384l8_head1", ("head", "binary")),
    "head_int2": ("binary_d384l8", "binary_d384l8_head2", ("head", "int2")),
    "head_int4": ("binary_d384l8", "binary_d384l8_head4", ("head", "int4")),
    "b07_bin": ("binary_d384l8_b07i8", "binary_d384l8", ("kind", "binary", (0, 7))),
    "b34_bin": ("binary_d384l8_b34i8", "binary_d384l8", ("kind", "binary", (3, 4))),
    "tern_bin": ("ternary_d384l8", "binary_d384l8", ("kind", "binary", None)),
    "fp32_bin": ("fp32_d384l8", "binary_d384l8", ("kind", "binary", None)),
    "fp32_tern": ("fp32_d384l8", "ternary_d384l8", ("kind", "ternary", None)),
    "acts_all": ("binary_d384l8", "binary_a1p_k1", ("acts", 1, LIN)),
    "acts_w2": ("binary_a1p_k1_w2a8", "binary_a1p_k1", ("acts", 1, ("w2",))),
    "acts_o": ("binary_a1x_o8", "binary_a1p_k1", ("acts", 1, ("o",))),
    "acts_qkv": ("binary_a1x_qkv8", "binary_a1p_k1", ("acts", 1, ("q", "k", "v"))),
    "acts_w13": ("binary_a1x_w13a8", "binary_a1p_k1", ("acts", 1, ("w1", "w3"))),
    "acts_w2_2to1": ("binary_a1x_w2b2", "binary_a1p_k1", ("acts", 1, ("w2",))),
}


def load(runs, run, dev):
    """A trained tiny_llm run as train.py built it (same constructor arguments as damage_map.py)."""
    ck = torch.load(os.path.join(runs, run, "model.pt"), map_location=dev, weights_only=False)
    a = ck["args"]
    m = T.TinyLM(4096, a["dim"], a["layers"], a["heads"], a["hidden"], a["seq"], a["weights"], a.get("head", "fp32"),
                 tuple(a.get("int8_blocks", [])), a.get("acts", 8),
                 {"ste_mode": a.get("ste", "identity"), "ste_k": a.get("ste_k", 2.0),
                  "learn_thr": a.get("learn_thr", False)},
                 tuple(a.get("a8_inputs", [])), T.W2_ACTS[a.get("w2_acts", "same")], a.get("ffn_act", "silu"),
                 tuple(a.get("hadamard", [])), tuple(a.get("a8_blocks", []))).to(dev)
    m.load_state_dict(ck["model"])
    return m, a


def settings(model, change):
    """The (object, attribute, value) list that turns A into B's format."""
    if change[0] == "head":
        return [(model, "head_kind", change[1])]
    if change[0] == "kind":
        blocks = [b for i, b in enumerate(model.blocks) if change[2] is None or i in change[2]]
        return [(getattr(b, n), "kind", change[1]) for b in blocks for n in LIN]
    out = []
    for b in model.blocks:
        for n in change[2]:
            lin = getattr(b, n)
            out += [(lin, "acts", change[1]), (lin, "act_opts", dict(CLIP1))]
    return out


def apply(sets):
    old = [(o, k, getattr(o, k)) for o, k, _ in sets]
    for o, k, v in sets:
        setattr(o, k, v)
    return old


def embed(model, idx, kind):
    if kind in ("int4", "int2"):
        return F.embedding(idx, T.quant_weight(model.emb.weight, kind))
    return model.emb(idx)


def head(model, h, kind):
    if kind in ("binary", "int4", "int2"):
        return F.linear(T.quant_act8(h), T.quant_weight(model.emb.weight, kind))
    return F.linear(h, model.emb.weight)


def static_measures(model, change, train, seq, n_batches, bsz, seed):
    """cos and fisher on training windows. Each changed layer is computed twice on the same input (A's format, then
    B's): no effect of one change reaches another layer's input, as in the CDT's per-layer scan."""
    model.eval()
    acc = {"cos": 0.0, "fisher": 0.0, "tokens": 0}
    units = []  # (output A with grad, output B) per changed layer, filled during the forward

    def note(y_a, y_b):
        y_a.retain_grad()
        units.append((y_a, y_b.detach()))

    lin_sets = {}
    if change[0] != "head":
        for o, k, v in settings(model, change):
            lin_sets.setdefault(id(o), (o, []))[1].append((o, k, v))

    def hook(mod, inp, out):
        with torch.no_grad():
            old = apply(lin_sets[id(mod)][1])
            y_b = mod.forward(inp[0])  # forward(), not __call__: the hook must not fire again
            apply(old)
        note(out, y_b)

    handles = [o.register_forward_hook(hook) for o, _ in lin_sets.values()]
    gen = torch.Generator(device=train.device).manual_seed(seed)
    try:
        for _ in range(n_batches):
            units.clear()
            x, y = T.batch(train, bsz, seq, gen)
            t = x.shape[1]
            hk = model.head_kind
            e = embed(model, x, hk)
            if change[0] == "head" and change[1] in ("int4", "int2"):
                with torch.no_grad():
                    e_b = embed(model, x, change[1])
                e = e.detach().requires_grad_(True)
                note(e, e_b)
            for blk in model.blocks:
                e = blk(e, model.cos[:t], model.sin[:t])
            h = model.norm(e)
            logits = head(model, h, hk)
            if change[0] == "head":
                with torch.no_grad():
                    lb = head(model, h, change[1])
                note(logits, lb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
            model.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                for y_a, y_b in units:
                    dy = (y_b - y_a).float()
                    acc["fisher"] += 0.5 * (y_a.grad.float() ** 2 * dy ** 2).sum().item()
                    acc["cos"] += (1 - F.cosine_similarity(y_a.float(), y_b.float(), dim=-1)).sum().item()
            acc["tokens"] += y.numel()
    finally:
        for hd in handles:
            hd.remove()
    model.zero_grad(set_to_none=True)
    n = acc["tokens"]
    return {"cos": acc["cos"] / n, "fisher": acc["fisher"] / n}


def finetune(model, train, val, a, args, seed):
    """args.ft_steps of AdamW from the trained weights; validation loss at each step of args.ft_report."""
    model.train()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a["wd"]}, {"params": rest, "weight_decay": 0.0}],
                            lr=args.ft_lr, betas=(0.9, 0.95), fused=True)
    gen = torch.Generator(device=train.device).manual_seed(seed + 1000)
    out = {}
    for step in range(1, args.ft_steps + 1):
        lr = args.ft_lr * min(1.0, step / args.ft_warmup) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.ft_steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = T.batch(train, a["batch"], a["seq"], gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in args.ft_report or step == args.ft_steps:
            out[str(step)] = T.evaluate(model, val, a["seq"], args.eval_batches, a["batch"])
    return out


def truth(runs, pa, pb, seed):
    rd = {}
    for r in (pa, pb):
        p = os.path.join(runs, f"{r}_s{seed}", "train.json")
        if not os.path.exists(p):
            return None
        rd[r] = json.load(open(p))["val_loss"]
    return rd[pb] - rd[pa]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=list(PAIRS))
    ap.add_argument("--runs", default=os.path.join(HERE, "..", "tiny_llm", "runs"), help="tiny_llm runs/ folder")
    ap.add_argument("--data", default=T.DATA, help="tinystories token folder (prepare.py)")
    ap.add_argument("--calib-batches", type=int, default=16, help="training batches for cos / fisher")
    ap.add_argument("--calib-bsz", type=int, default=8, help="windows per calibration batch")
    ap.add_argument("--eval-batches", type=int, default=40, help="validation batches of 128 (train.py's default)")
    ap.add_argument("--no-recover", dest="recover", action="store_false", help="skip the fine-tune measure")
    ap.add_argument("--ft-steps", type=int, default=500)
    ap.add_argument("--ft-report", type=int, nargs="*", default=[100, 250])
    ap.add_argument("--ft-lr", type=float, default=1.5e-4)
    ap.add_argument("--ft-warmup", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "damage_probe.json"))
    args = ap.parse_args()
    T.DATA = args.data
    dev = "cuda"
    train, val = T.load_split("train", dev), T.load_split("val", dev)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    # resuming only with the same measurement settings: seeds, pairs and the output may differ
    same = {k: v for k, v in vars(args).items() if k not in ("seeds", "pairs", "out", "recover")}
    old = {k: v for k, v in res.get("args", {}).items() if k in same}
    if old and old != same:
        diff = {k: (old.get(k), same[k]) for k in same if old.get(k) != same[k]}
        sys.exit(f"{args.out} was measured with other settings {diff} (old, new): use another --out")
    res.setdefault("args", vars(args))
    rows = res.setdefault("rows", {})
    controls = res.setdefault("controls", {})
    for seed in args.seeds:
        for name in args.pairs:
            pa, pb, change = PAIRS[name]
            key = f"{name}_s{seed}"
            tr = truth(args.runs, pa, pb, seed)
            if tr is None or (key in rows and (not args.recover or "recover" in rows[key])):
                continue
            t0 = time.time()
            model, a = load(args.runs, f"{pa}_s{seed}", dev)
            row = {"pair": name, "seed": seed, "truth": round(tr, 4)}
            base = T.evaluate(model, val, a["seq"], args.eval_batches, a["batch"])
            row["loss_a"] = round(base, 4)
            row.update({k: v for k, v in static_measures(model, change, train, a["seq"], args.calib_batches,
                                                       args.calib_bsz, seed).items()})
            old = apply(settings(model, change))
            row["loss0"] = T.evaluate(model, val, a["seq"], args.eval_batches, a["batch"]) - base
            apply(old)
            row["static_seconds"] = round(time.time() - t0, 1)
            if args.recover:
                ck = f"{pa}_s{seed}"
                if ck not in controls:
                    ctl, _ = load(args.runs, ck, dev)
                    controls[ck] = finetune(ctl, train, val, a, args, seed)
                    del ctl
                t1 = time.time()
                apply(settings(model, change))
                ft = finetune(model, train, val, a, args, seed)
                row["recover_curve"] = {s: ft[s] - controls[ck][s] for s in ft}
                row["recover"] = row["recover_curve"][str(args.ft_steps)]
                row["recover_seconds"] = round(time.time() - t1, 1)
            rows[key] = row
            print(f"{key:20s} truth {tr:+.4f}  cos {row['cos']:.4f}  loss0 {row['loss0']:+.4f}  "
                  f"fisher {row['fisher']:.4f}" + (f"  recover {row['recover']:+.4f}" if "recover" in row else "")
                  + f"  ({time.time() - t0:.0f} s)", flush=True)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(res, open(args.out, "w"), indent=2)
            del model
            torch.cuda.empty_cache()

    measures = ["cos", "loss0", "fisher"] + (["recover"] if args.recover else [])
    summary = {}
    for seed in args.seeds:
        rs = [r for r in rows.values() if r["seed"] == seed]
        if len(rs) < 3:
            continue
        summary[str(seed)] = {"pairs": len(rs)}
        for m in measures:
            if all(m in r for r in rs):
                summary[str(seed)][m] = round(spearmanr([r[m] for r in rs], [r["truth"] for r in rs])[0], 3)
    res["spearman"] = summary
    json.dump(res, open(args.out, "w"), indent=2)
    print("\nSpearman with the truth, per seed:")
    for s, d in summary.items():
        print(f"  seed {s} ({d['pairs']} pairs): " + "  ".join(f"{m} {d[m]:+.3f}" for m in measures if m in d))


if __name__ == "__main__":
    main()
