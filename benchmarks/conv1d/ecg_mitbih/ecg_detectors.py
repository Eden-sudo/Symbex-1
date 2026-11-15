"""
ecg_detectors.py -- experiment: one small binary detector per class (N, S, V, F: "this class or not") instead of one
4-class network. Cross-validation inside DS1 only: the test set (DS2) is never scored.

    python ecg_detectors.py --arch s --rr --weight-power 0.5 --select f1 --cv            # seed 42, every fold
    python ecg_detectors.py --arch s --rr --weight-power 0.5 --select f1 --cv --fold 2   # one fold

Run from benchmarks/conv1d/ecg_mitbih/ (main environment). Question: does splitting the task into one binary network
per class (each specializes: S lives mostly in the rhythm, V in the beat shape) beat one network of about the same
cost? Four detectors of --arch s cost about one m network (the inner layers have a quarter of the weights).
The recipe of ecg_train.py, unchanged: the same folds, Adam, --lr with cosine decay, --epochs, batches of 128, class
weights ** --weight-power (here over "this class" / "the rest"), the RR channels with --rr. The four detectors train
side by side, one epoch each in turn; the predicted class is the detector with the largest margin (score of "yes"
minus score of "no"), no thresholds tuned. ONE checkpoint for the four (the epoch with the best combined --select on
the fold's validation records): the same freedom of choice as one network. Writes runs/det_<name>_f<k>/
(model_<class>.pth, train.json) and, with every fold, runs/cv/det_<name>.json (pooled out-of-fold scores and the
mean fold F1, to compare with runs/cv/<name>.json of ecg_train.py --cv).
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
from ecg_data import ARCHS, CLASSES, CV_FOLDS, RR_CHANNELS, WINDOW, class_weights, load, scores  # noqa: E402
from ecg_train import fold_is_current, run_name                                                               # noqa: E402
from model import BinaryHAR                                                                   # noqa: E402


@torch.no_grad()
def margins(models, x, batch=4096):
    """N x len(CLASSES): per detector, the score of "this class" minus the score of "the rest\""""
    out = []
    for m in models:
        m.eval()
        s = torch.cat([m(x[i:i + batch]) for i in range(0, len(x), batch)])
        out.append(s[:, 1] - s[:, 0])
    return torch.stack(out, 1).cpu().numpy()


def train_fold(arch, first, seed, epochs, lr, dev, fold, rr, wp, select):
    (xtr, ytr), (xva, yva) = [tuple(torch.as_tensor(a, device=dev) for a in load(s, fold, rr))
                              for s in ("train", "validation")]
    ytr, yva_np = ytr.long(), yva.cpu().numpy()
    torch.manual_seed(seed)
    np.random.seed(seed)
    models, opts, scheds, weights = [], [], [], []
    for c in range(len(CLASSES)):
        m = BinaryHAR(arch, first, layers=ARCHS[arch], length=WINDOW, channels=1 + (RR_CHANNELS if rr else 0),
                      classes=2).to(dev)
        yb = (ytr == c).long()
        w = class_weights(yb.cpu().numpy(), wp)[:2]                  # "the rest" / "this class"
        models.append(m)
        weights.append(torch.as_tensor(w, dtype=torch.float32, device=dev))
        opts.append(torch.optim.Adam(m.parameters(), lr=lr))
        scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(opts[-1], epochs))
    name = "det_" + run_name(arch, first, epochs, seed, False, fold, rr, wp, select)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(epochs):
        for c, (m, opt, sched, w) in enumerate(zip(models, opts, scheds, weights)):
            m.train()
            perm = torch.randperm(len(xtr), device=dev)
            for i in range(0, len(perm), 128):
                idx = perm[i:i + 128]
                if len(idx) < 2:
                    continue
                loss = F.cross_entropy(m(xtr[idx]), (ytr[idx] == c).long(), weight=w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
        va = scores(yva_np, margins(models, xva).argmax(1))
        if va[select] > best:
            best, best_epoch, best_state = va[select], ep + 1, [copy.deepcopy(m.state_dict()) for m in models]
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"    {name} | epoch {ep + 1:3d}/{epochs} | validation balanced {va['balanced']:5.1f}% F1 "
                  f"{va['f1']:5.1f}% (best {select} {best:5.1f}% at {best_epoch}) | {(time.time() - t0) / 60:.1f} min",
                  flush=True)
    for m, s in zip(models, best_state):
        m.load_state_dict(s)
    p_va = margins(models, xva).argmax(1)
    va = scores(yva_np, p_va)
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    for c, s in zip(CLASSES, best_state):
        torch.save(s, os.path.join(out, f"model_{c}.pth"))
    json.dump(dict(arch=arch, layers=ARCHS[arch], first=first, seed=seed, epochs=epochs, lr=lr, fold=fold, rr=rr,
                   weight_power=wp, select=select, detectors=CLASSES, best_epoch=best_epoch, validation=va,
                   validation_pred=p_va.tolist()),
              open(os.path.join(out, "train.json"), "w"), indent=1)
    print(f"[*] {name}: validation (fold {fold}) balanced {va['balanced']:.1f}% F1 {va['f1']:.1f}% (epoch "
          f"{best_epoch}), sensitivity {va['sensitivity']} -> {out}/", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="s", choices=list(ARCHS), help="the size of EACH detector (s default)")
    ap.add_argument("--first", default="binary", choices=["binary", "int8"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--cv", action="store_true", help="every fold of DS1 (required: this experiment never scores the test)")
    ap.add_argument("--fold", type=int, choices=range(CV_FOLDS), help="with --cv: only this fold")
    ap.add_argument("--rr", action="store_true", help="+ the RR-interval channels (ecg_data.rr_channels) as input")
    ap.add_argument("--weight-power", type=float, default=1.0, help="class weights = inverse frequency ** this")
    ap.add_argument("--select", choices=["balanced", "f1"], default="balanced",
                    help="validation metric of the checkpoint: balanced accuracy (default) or macro F1")
    args = ap.parse_args()
    if not args.cv:
        sys.exit("[X] use --cv: the detectors experiment runs inside DS1 only")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = range(CV_FOLDS) if args.fold is None else [args.fold]
    ys, ps, fold_f1, fold_bal = [], [], [], []
    for k in folds:
        path = os.path.join(HERE, "runs", "det_" + run_name(args.arch, args.first, args.epochs, args.seed, False, k,
                                                             args.rr, args.weight_power, args.select), "train.json")
        if not fold_is_current(path, args.lr):                       # a fold already trained is read back
            train_fold(args.arch, args.first, args.seed, args.epochs, args.lr, dev, k, args.rr, args.weight_power,
                       args.select)
        info = json.load(open(path))
        ys.append(load("validation", k)[1])
        ps.append(np.array(info["validation_pred"]))
        fold_f1.append(info["validation"]["f1"])
        fold_bal.append(info["validation"]["balanced"])
    if args.fold is not None:
        return
    pooled = scores(np.concatenate(ys), np.concatenate(ps))
    name = "det_" + run_name(args.arch, args.first, args.epochs, args.seed, False, rr=args.rr, wp=args.weight_power,
                             select=args.select)
    os.makedirs(os.path.join(HERE, "runs", "cv"), exist_ok=True)
    out = os.path.join(HERE, "runs", "cv", name + ".json")
    json.dump(dict(name=name, arch=args.arch, first=args.first, seed=args.seed, epochs=args.epochs, rr=args.rr,
                   weight_power=args.weight_power, select=args.select, detectors=CLASSES, folds=list(folds),
                   fold_balanced=fold_bal, fold_f1=fold_f1, mean_fold_f1=float(np.mean(fold_f1)), pooled=pooled),
              open(out, "w"), indent=1)
    print(f"[*] CV {name}: mean fold F1 {np.mean(fold_f1):.2f}% (folds {' / '.join('%.1f' % f for f in fold_f1)}), "
          f"pooled balanced {pooled['balanced']:.1f}% F1 {pooled['f1']:.1f}%, sensitivity {pooled['sensitivity']} "
          f"ppv {pooled['ppv']} -> {out}")


if __name__ == "__main__":
    main()
