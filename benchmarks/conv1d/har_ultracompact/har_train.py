"""
har_train.py -- trains one binary HAR network of model.py on UniMiB-SHAR (data/har_unimib.npz from har_data.py).

    python har_train.py --arch max --first binary --seed 42          # lib_symbex's network (1-bit first weights)
    python har_train.py --arch max --first int8 --seed 42            # ultracompactBNN's network (int8 first weights)
    python har_train.py --all                                        # 4 architectures x 2 first layers x seeds 42-44
    python har_train.py --arch max --optimizer bop --bop-gamma 1e-3 --no-test    # Bop for the 1-bit weights, screening
    python har_train.py --arch max --optimizer bop --bop-gamma 1e-3 --seeds 42,43,44 --score-test   # chosen recipe:
                                                                     # score its saved checkpoints on the test subjects

Run from benchmarks/conv1d/har_ultracompact/ (main environment). The same recipe for every network: Adam, learning rate
--lr with cosine decay, --epochs epochs, batches of 128, cross-entropy, no augmentation; the whole data set lives on
the GPU. The checkpoint is chosen on the VALIDATION subjects (the epoch with the best validation accuracy); the test
subjects are scored once, at the end, with that checkpoint (--no-test: not scored, for screening runs). Saves
runs/<arch>_<first>_e<epochs>_s<seed>/model.pth and train.json.

--optimizer bop: the 1-bit weights (every conv but an int8 first one, and the classifier) are trained by Bop
(tools/compiler_core/train/bop.py, the CDT's binary_optimizer = "bop"), its gamma following the learning rate's cosine;
the rest (BatchNorms, int8 first-layer weights) stays with Adam. The run is named <arch>_<first>_e<epochs>_bop<gamma>_s<seed>
(plus _tau<tau> when --bop-tau is not the default 1e-8, so runs with different thresholds never share a folder).
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from har_data import load                                                                    # noqa: E402
from model import ARCHS, TABLE1, BinaryHAR, evaluate                                         # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
from compiler_core.train.bop import Bop                                                      # noqa: E402

SEEDS = (42, 43, 44)
BOP_TAU = 1e-8                                                       # the default threshold: not written in run names


def run_name(arch, first, epochs, seed, optimizer="adam", bop_gamma=1e-3, bop_tau=BOP_TAU):
    """the run folder's name: the optimizer's settings are part of it, so different recipes never overwrite each other"""
    # repr: the shortest text that reads back as the same float (0.001 -> "0.001", as the recorded runs are named)
    bop = (f"_bop{bop_gamma!r}" + (f"_tau{bop_tau!r}" if bop_tau != BOP_TAU else "")) if optimizer == "bop" else ""
    return f"{arch}_{first}_e{epochs}{bop}_s{seed}"


def binary_weights(model):
    """the 1-bit weights of a BinaryHAR: every conv (the first one only when first = "binary") and the classifier"""
    return [c.weight for i, c in enumerate(model.convs) if i > 0 or model.first == "binary"] + [model.fc.weight]


def train(arch, first, seed, epochs, lr, dev, data, optimizer="adam", bop_gamma=1e-3, bop_tau=BOP_TAU, test=True):
    (xtr, ytr), (xva, yva), (xte, yte) = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BinaryHAR(arch, first).to(dev)
    bop = None
    if optimizer == "bop":
        bits = binary_weights(model)
        bop = Bop(bits, bop_gamma, bop_tau)
        ids = {id(w) for w in bits}
        opt = torch.optim.Adam([p for p in model.parameters() if id(p) not in ids], lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    name = run_name(arch, first, epochs, seed, optimizer, bop_gamma, bop_tau)
    out = os.path.join(HERE, "runs", name)
    best, best_state, history, t0 = -1.0, None, [], time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            if bop:
                bop.zero_grad()
            loss.backward()
            opt.step()
            if bop:                                                  # gamma follows the learning rate's cosine
                bop.step(opt.param_groups[0]["lr"] / lr)
        sched.step()
        va = evaluate(model, xva, yva)
        if va > best:
            best, best_epoch, best_state = va, ep + 1, copy.deepcopy(model.state_dict())
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            history.append(dict(epoch=ep + 1, validation=va))
            print(f"    {name} | epoch {ep + 1:3d}/{epochs} | validation {va:5.1f}% (best {best:5.1f}% at {best_epoch})"
                  f" | {(time.time() - t0) / 60:.1f} min")
    model.load_state_dict(best_state)
    te = evaluate(model, xte, yte) if test else None
    os.makedirs(out, exist_ok=True)
    torch.save(best_state, os.path.join(out, "model.pth"))
    json.dump(dict(arch=arch, layers=ARCHS[arch], first=first, seed=seed, epochs=epochs, lr=lr, optimizer=optimizer,
                   **(dict(bop_gamma=bop_gamma, bop_tau=bop_tau) if bop else {}), best_epoch=best_epoch,
                   validation=best, test=te, features=model.features, history=history),
              open(os.path.join(out, "train.json"), "w"), indent=1)
    print(f"[*] {name}: validation {best:.1f}% (epoch {best_epoch}) | "
          + (f"TEST {te:.1f}%" if test else "test not scored") + f" -> {out}/")
    return best, te


def score_test(arch, first, seed, epochs, optimizer, bop_gamma, bop_tau, dev, data):
    """scores a screening run's saved checkpoint (chosen on validation) on the test subjects, without training again,
    and records the result in its train.json; refuses a run whose recorded recipe is not the one asked for"""
    name = run_name(arch, first, epochs, seed, optimizer, bop_gamma, bop_tau)
    out = os.path.join(HERE, "runs", name)
    if not os.path.exists(os.path.join(out, "train.json")):
        raise SystemExit(f"{out}: no such run (train it first, with --no-test)")
    info = json.load(open(os.path.join(out, "train.json")))
    info.setdefault("optimizer", "adam")                             # runs from before --optimizer existed
    asked = dict(arch=arch, first=first, seed=seed, epochs=epochs, optimizer=optimizer,
                 **(dict(bop_gamma=bop_gamma, bop_tau=bop_tau) if optimizer == "bop" else {}))
    wrong = {k: (info.get(k), v) for k, v in asked.items() if info.get(k) != v}
    if wrong:
        raise SystemExit(f"{out}: recorded recipe differs from the one asked for (recorded, asked): {wrong}")
    if info.get("test") is not None:
        print(f"[*] {name}: test already scored ({info['test']:.1f}%), not scored again")
        return info["validation"], info["test"]
    model = BinaryHAR(arch, first).to(dev)
    model.load_state_dict(torch.load(os.path.join(out, "model.pth"), map_location=dev))
    te = evaluate(model, *data[2])
    info["test"] = te
    json.dump(info, open(os.path.join(out, "train.json"), "w"), indent=1)
    print(f"[*] {name}: validation {info['validation']:.1f}% (epoch {info['best_epoch']}) | TEST {te:.1f}% -> {out}/")
    return info["validation"], te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="max", choices=list(ARCHS))
    ap.add_argument("--first", default="binary", choices=["binary", "int8"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--all", action="store_true", help="every architecture, both first layers, seeds 42, 43, 44")
    ap.add_argument("--optimizer", default="adam", choices=["adam", "bop"], help="for the 1-bit weights (default adam)")
    ap.add_argument("--bop-gamma", type=float, default=1e-3, help="Bop's moving-average rate (default 1e-3)")
    ap.add_argument("--bop-tau", type=float, default=BOP_TAU, help="Bop's flip threshold (default 1e-8)")
    ap.add_argument("--seeds", help="comma-separated seeds instead of --seed, e.g. 42,43,44")
    ap.add_argument("--no-test", action="store_true", help="do not score the test subjects (screening runs)")
    ap.add_argument("--score-test", action="store_true",
                    help="do not train: score the saved checkpoints of these runs on the test subjects (once)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = [tuple(torch.as_tensor(a, device=dev) for a in load(s)) for s in ("train", "validation", "test")]
    data = [(x, y.long()) for x, y in data]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    jobs = ([(a, f, s) for a in TABLE1 for f in ("binary", "int8") for s in SEEDS] if args.all
            else [(args.arch, args.first, s) for s in seeds])
    results = {}
    for a, f, s in jobs:
        if args.score_test:
            results.setdefault((a, f), []).append(score_test(a, f, s, args.epochs, args.optimizer, args.bop_gamma,
                                                             args.bop_tau, dev, data))
            continue
        results.setdefault((a, f), []).append(train(a, f, s, args.epochs, args.lr, dev, data, args.optimizer,
                                                    args.bop_gamma, args.bop_tau, not args.no_test))
    if len(jobs) > 1:
        print(f"\n    {'network':<16}{'validation %':>16}{'test %':>16}")
        for (a, f), r in results.items():
            r = np.array(r, dtype=float)                                  # test None (--no-test) -> nan
            print(f"    {a + ' ' + f:<16}{r[:, 0].mean():>10.1f} ± {r[:, 0].std(ddof=1):.1f}{r[:, 1].mean():>10.1f} ± {r[:, 1].std(ddof=1):.1f}")


if __name__ == "__main__":
    main()
