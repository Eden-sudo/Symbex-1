"""
har_train.py -- trains one binary HAR network of model.py on UniMiB-SHAR (data/har_unimib.npz from har_data.py).

    python har_train.py --arch max --first binary --seed 42          # lib_symbex's network (1-bit first weights)
    python har_train.py --arch max --first int8 --seed 42            # ultracompactBNN's network (int8 first weights)
    python har_train.py --all                                        # 4 architectures x 2 first layers x seeds 42-44

Run from benchmarks/conv1d/har_ultracompact/ (main environment). The same recipe for every network: Adam, learning rate
--lr with cosine decay, --epochs epochs, batches of 128, cross-entropy, no augmentation; the whole data set lives on
the GPU. The checkpoint is chosen on the VALIDATION subjects (the epoch with the best validation accuracy); the test
subjects are scored once, at the end, with that checkpoint. Saves runs/<arch>_<first>_e<epochs>_s<seed>/model.pth and
train.json.
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
from model import ARCHS, BinaryHAR, evaluate                                                 # noqa: E402

SEEDS = (42, 43, 44)


def train(arch, first, seed, epochs, lr, dev, data):
    (xtr, ytr), (xva, yva), (xte, yte) = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BinaryHAR(arch, first).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    name = f"{arch}_{first}_e{epochs}_s{seed}"
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
            loss.backward()
            opt.step()
        sched.step()
        va = evaluate(model, xva, yva)
        if va > best:
            best, best_epoch, best_state = va, ep + 1, copy.deepcopy(model.state_dict())
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            history.append(dict(epoch=ep + 1, validation=va))
            print(f"    {name} | epoch {ep + 1:3d}/{epochs} | validation {va:5.1f}% (best {best:5.1f}% at {best_epoch})"
                  f" | {(time.time() - t0) / 60:.1f} min")
    model.load_state_dict(best_state)
    te = evaluate(model, xte, yte)
    os.makedirs(out, exist_ok=True)
    torch.save(best_state, os.path.join(out, "model.pth"))
    json.dump(dict(arch=arch, layers=ARCHS[arch], first=first, seed=seed, epochs=epochs, lr=lr, best_epoch=best_epoch,
                   validation=best, test=te, features=model.features, history=history),
              open(os.path.join(out, "train.json"), "w"), indent=1)
    print(f"[*] {name}: validation {best:.1f}% (epoch {best_epoch}) | TEST {te:.1f}% -> {out}/")
    return best, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="max", choices=list(ARCHS))
    ap.add_argument("--first", default="binary", choices=["binary", "int8"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--all", action="store_true", help="every architecture, both first layers, seeds 42, 43, 44")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = [tuple(torch.as_tensor(a, device=dev) for a in load(s)) for s in ("train", "validation", "test")]
    data = [(x, y.long()) for x, y in data]
    jobs = ([(a, f, s) for a in ARCHS for f in ("binary", "int8") for s in SEEDS] if args.all
            else [(args.arch, args.first, args.seed)])
    results = {}
    for a, f, s in jobs:
        results.setdefault((a, f), []).append(train(a, f, s, args.epochs, args.lr, dev, data))
    if len(jobs) > 1:
        print(f"\n    {'network':<16}{'validation %':>16}{'test %':>16}")
        for (a, f), r in results.items():
            r = np.array(r)
            print(f"    {a + ' ' + f:<16}{r[:, 0].mean():>10.1f} ± {r[:, 0].std():.1f}{r[:, 1].mean():>10.1f} ± {r[:, 1].std():.1f}")


if __name__ == "__main__":
    main()
