"""
ecg_train.py -- trains one binary ECG network (har_ultracompact/model.BinaryHAR with the layers of ecg_data.ARCHS) on
the MIT-BIH beats (data/ecg_mitbih.npz from ecg_data.py).

    python ecg_train.py --arch m --first binary --seed 42           # lib_symbex's network (1-bit first weights)
    python ecg_train.py --arch m --first int8 --seed 42             # int8 first-layer weights
    python ecg_train.py --all                                       # every architecture x both first layers x seeds
    python ecg_train.py --arch m --augment --seed 42                # the same with ecg_data.augment every epoch
    python ecg_train.py --arch m --cv                               # patient-wise cross-validation inside DS1
    python ecg_train.py --arch m --rr --seed 42                     # + the RR-interval input channels
    python ecg_train.py --arch m --rr --cv --weight-power 0.5 --select f1   # softer class weights, checkpoint on F1

Run from benchmarks/conv1d/ecg_mitbih/ (main environment). The same recipe for every network (and for the INT8 side,
ecg_tflite.py): Adam, learning rate --lr with cosine decay, --epochs epochs, batches of 128, cross-entropy weighted by
the inverse class frequency of the training beats (ecg_data.class_weights), no augmentation unless --augment (a new
ecg_data.augment draw of the training beats every epoch, numpy seed = the seed); the whole data set lives on the GPU.
The checkpoint is the epoch with the best BALANCED accuracy on the validation records; the test records (DS2) are
scored once, at the end, with that checkpoint. Saves runs/<arch>_<first>_e<epochs>_s<seed>[_aug]/model.pth and
train.json. --fold k: trains on the other DS1 folds and validates on fold k (ecg_data.cv_folds), the test set is NOT
scored, runs/..._f<k>/; --cv: every fold, then the balanced accuracy of the pooled out-of-fold predictions (written to
runs/cv/<name>.json): the recipe is chosen on it, never on the test set. --rr: the ECG plus the ecg_data.rr_channels
rhythm channels as the input (runs/..._rr). --weight-power p: the class weights ** p (ecg_data.class_weights, runs/..._wp<p>);
--select f1: the checkpoint of the best validation macro F1 instead of balanced accuracy (runs/..._self1); the
cross-validation scores both.
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
sys.path.insert(1, os.path.join(HERE, "..", "har_ultracompact"))
from ecg_data import (ARCHS, CLASSES, CV_FOLDS, RR_CHANNELS, WINDOW, augment, class_weights, load,  # noqa: E402
                      recipe_tag, scores)
from model import BinaryHAR                                                                  # noqa: E402

SEEDS = (42, 43, 44)


def make_model(arch, first, rr=False):
    return BinaryHAR(arch, first, layers=ARCHS[arch], length=WINDOW, channels=1 + (RR_CHANNELS if rr else 0),
                     classes=len(CLASSES))


@torch.no_grad()
def predict(model, x, batch=4096):
    model.eval()
    return torch.cat([model(x[i:i + batch]).argmax(1) for i in range(0, len(x), batch)]).cpu().numpy()


def run_name(arch, first, epochs, seed, aug=False, fold=None, rr=False, wp=1.0, select="balanced"):
    return (f"{arch}_{first}_e{epochs}_s{seed}" + ("_aug" if aug else "") + ("_rr" if rr else "")
            + recipe_tag(wp, select) + ("" if fold is None else f"_f{fold}"))


def train(arch, first, seed, epochs, lr, dev, data, aug=False, fold=None, rr=False, wp=1.0, select="balanced"):
    (xtr, ytr), (xva, yva), (xte, yte) = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng, xtr_np = np.random.default_rng(seed), xtr.cpu().numpy()
    model = make_model(arch, first, rr).to(dev)
    weight = torch.as_tensor(class_weights(ytr.cpu().numpy(), wp), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    name = run_name(arch, first, epochs, seed, aug, fold, rr, wp, select)
    out = os.path.join(HERE, "runs", name)
    best, best_state, history, t0 = -1.0, None, [], time.time()
    yva_np, yte_np = yva.cpu().numpy(), yte.cpu().numpy()
    for ep in range(epochs):
        model.train()
        if aug:
            xtr = torch.as_tensor(augment(xtr_np, rng), device=dev)
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx], weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        va = scores(yva_np, predict(model, xva))
        if va[select] > best:
            best, best_epoch, best_state = va[select], ep + 1, copy.deepcopy(model.state_dict())
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            history.append(dict(epoch=ep + 1, validation_balanced=va["balanced"], validation_accuracy=va["accuracy"]))
            print(f"    {name} | epoch {ep + 1:3d}/{epochs} | validation balanced {va['balanced']:5.1f}% "
                  f"F1 {va['f1']:5.1f}% (best {select} {best:5.1f}% at {best_epoch}) | {(time.time() - t0) / 60:.1f} min")
    model.load_state_dict(best_state)
    p_va = predict(model, xva)
    va = scores(yva_np, p_va)
    te = None if fold is not None else scores(yte_np, predict(model, xte))     # folds never look at the test set
    os.makedirs(out, exist_ok=True)
    torch.save(best_state, os.path.join(out, "model.pth"))
    json.dump(dict(arch=arch, layers=ARCHS[arch], first=first, seed=seed, epochs=epochs, lr=lr, augment=aug, fold=fold,
                   rr=rr, weight_power=wp, select=select,
                   best_epoch=best_epoch, validation=va, test=te, features=model.features, history=history,
                   validation_pred=p_va.tolist() if fold is not None else None),
              open(os.path.join(out, "train.json"), "w"), indent=1)
    if te is None:
        print(f"[*] {name}: validation (fold {fold}) balanced {va['balanced']:.1f}% F1 {va['f1']:.1f}% (epoch "
              f"{best_epoch}), sensitivity {va['sensitivity']} -> {out}/")
    else:
        print(f"[*] {name}: validation balanced {va['balanced']:.1f}% (epoch {best_epoch}) | TEST balanced "
              f"{te['balanced']:.1f}%, accuracy {te['accuracy']:.1f}%, sensitivity {te['sensitivity']} -> {out}/")
    return va, te


def to_dev(fold, dev, rr=False):
    data = [tuple(torch.as_tensor(a, device=dev) for a in load(s, fold, rr)) for s in ("train", "validation", "test")]
    return [(x, y.long()) for x, y in data]


def cross_validate(args, dev):
    """every fold of DS1 (a fold already trained is read back: the folds can run in parallel with --fold), then the
    scores of the pooled out-of-fold predictions -> runs/cv/<name>.json (only when every fold is there)"""
    ys, ps, per_fold, per_fold_f1 = [], [], [], []
    folds = range(CV_FOLDS) if args.fold is None else [args.fold]
    for k in folds:
        path = os.path.join(HERE, "runs", run_name(args.arch, args.first, args.epochs, args.seed, args.augment, k,
                                                    args.rr, args.weight_power, args.select), "train.json")
        if not os.path.exists(path):
            train(args.arch, args.first, args.seed, args.epochs, args.lr, dev, to_dev(k, dev, args.rr), args.augment,
                  k, args.rr, args.weight_power, args.select)
        info = json.load(open(path))
        ys.append(load("validation", k)[1])
        ps.append(np.array(info["validation_pred"]))
        per_fold.append(info["validation"]["balanced"])
        per_fold_f1.append(scores(ys[-1], ps[-1])["f1"])          # the runs written before F1 existed too
    if args.fold is not None:
        return
    pooled = scores(np.concatenate(ys), np.concatenate(ps))
    name = run_name(args.arch, args.first, args.epochs, args.seed, args.augment, rr=args.rr, wp=args.weight_power,
                    select=args.select)
    os.makedirs(os.path.join(HERE, "runs", "cv"), exist_ok=True)
    out = os.path.join(HERE, "runs", "cv", name + ".json")
    json.dump(dict(name=name, arch=args.arch, first=args.first, seed=args.seed, epochs=args.epochs, augment=args.augment,
                   rr=args.rr, weight_power=args.weight_power, select=args.select,
                   folds=list(folds), fold_balanced=per_fold, fold_f1=per_fold_f1, pooled=pooled),
              open(out, "w"), indent=1)
    print(f"[*] CV {name}: pooled out-of-fold balanced {pooled['balanced']:.1f}% F1 {pooled['f1']:.1f}%, folds "
          f"{' / '.join(f'{b:.1f}' for b in per_fold)} (F1 {' / '.join(f'{b:.1f}' for b in per_fold_f1)}), "
          f"sensitivity {pooled['sensitivity']} ppv {pooled['ppv']} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="m", choices=list(ARCHS))
    ap.add_argument("--first", default="binary", choices=["binary", "int8"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--all", action="store_true", help="every architecture, both first layers, seeds 42, 43, 44")
    ap.add_argument("--augment", action="store_true", help="a new ecg_data.augment draw of the training beats every epoch")
    ap.add_argument("--cv", action="store_true", help="patient-wise cross-validation inside DS1 (every fold, no test)")
    ap.add_argument("--fold", type=int, choices=range(CV_FOLDS), help="with --cv: only this fold")
    ap.add_argument("--rr", action="store_true", help="+ the RR-interval channels (ecg_data.rr_channels) as input")
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="class weights = inverse frequency ** this (1 default, 0.5 square root, 0 none)")
    ap.add_argument("--select", choices=["balanced", "f1"], default="balanced",
                    help="validation metric of the checkpoint: balanced accuracy (default) or macro F1")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cv:
        return cross_validate(args, dev)
    data = to_dev(None, dev, args.rr)
    jobs = ([(a, f, s) for a in ARCHS for f in ("binary", "int8") for s in SEEDS] if args.all
            else [(args.arch, args.first, args.seed)])
    results = {}
    for a, f, s in jobs:
        results.setdefault((a, f), []).append(train(a, f, s, args.epochs, args.lr, dev, data, args.augment,
                                                    rr=args.rr, wp=args.weight_power, select=args.select))
    if len(jobs) > 1:
        print(f"\n    {'network':<12}{'val balanced %':>18}{'test balanced %':>18}{'test accuracy %':>18}")
        for (a, f), r in results.items():
            vb = np.array([v["balanced"] for v, _ in r])
            tb = np.array([t["balanced"] for _, t in r])
            ta = np.array([t["accuracy"] for _, t in r])
            print(f"    {a + ' ' + f:<12}{vb.mean():>12.1f} ± {vb.std():.1f}{tb.mean():>12.1f} ± {tb.std():.1f}"
                  f"{ta.mean():>12.1f} ± {ta.std():.1f}")


if __name__ == "__main__":
    main()
