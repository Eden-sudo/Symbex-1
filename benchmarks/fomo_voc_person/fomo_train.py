"""
fomo_train.py -- does the binary FOMO detector learn? A quick training on the PC, before the CDT integration.

    python fomo_train.py                                    # 16-32-64 channels, 1-1-1 blocks, 60 epochs
    python fomo_train.py --channels 32-64-128 --blocks 1-1-2 --weight 30 --epochs 100
    python fomo_train.py --fp32                             # the same architecture in floating point (the ceiling)

Run from benchmarks/fomo_voc_person/ (main environment; needs data/fomo_*.npz from fomo_data.py). The whole data set
goes to the GPU once (uint8, ~600 MB). Loss: cross-entropy per cell, the person cells weighted --weight times more
(only ~0.5% of the cells hold a person: without the weight, "nobody anywhere" is a 99.5%-accurate answer). Augmentation:
horizontal flip of the image AND its grid. Every 10 epochs: precision, recall and F1 of the detected person centers on
the 4,952 VOC 2007 test images (model.f1_score), with the default decision (the more likely class of every cell).
At the end, the decision THRESHOLD is calibrated, fairly: --val training images are kept out of the training; on them,
a range of thresholds on (person logit - background logit) is tried and the one with the best F1 is kept; the test F1
is then measured with that fixed threshold (the test images never choose anything). On the chip the threshold is free:
it only shifts the integer bias of the person class in the head. Saves runs/<name>/model.pth and train.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import BoolFOMO, RealFOMO, f1_score, load                                                      # noqa: E402


@torch.no_grad()
def margins(model, x, batch=512):
    """person logit - background logit, per cell: > threshold means "a person center here" (argmax = threshold 0)"""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        z = model(x[i:i + batch])
        out.append((z[:, 1] - z[:, 0]).float())
    return torch.cat(out).cpu().numpy()


def calibrate(m, centers, owner):
    """the threshold with the best F1 on (m, centers, owner): candidates where 0.1% to 4% of the cells are positive"""
    qs = np.quantile(m, 1 - np.geomspace(0.001, 0.04, 40))
    best_t, best = 0.0, f1_score(m > 0, centers, owner)
    for th in np.concatenate([qs, [0.0]]):
        s = f1_score(m > th, centers, owner)
        if s["f1"] > best["f1"]:
            best_t, best = float(th), s
    return best_t, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="16-32-64")
    ap.add_argument("--blocks", default="1-1-1")
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--weight", type=float, default=20.0, help="weight of the person cells in the loss (default 20)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32", action="store_true", help="the same architecture in floating point (the reference)")
    ap.add_argument("--val", type=int, default=1000, help="training images kept out to calibrate the threshold (1000)")
    args = ap.parse_args()
    os.chdir(HERE)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch = tuple(int(v) for v in args.channels.split("-"))
    bl = tuple(int(v) for v in args.blocks.split("-"))
    xtr, ytr, ctr, otr = load("train")
    nv = args.val                                   # the LAST nv training images: the calibration set, never trained on
    xva, cva, ova = xtr[-nv:], ctr[otr >= len(xtr) - nv], otr[otr >= len(xtr) - nv] - (len(xtr) - nv)
    xtr, ytr = xtr[:-nv], ytr[:-nv]
    xte, yte, cte, ote = load("test")
    xtr = torch.as_tensor(xtr, device=dev).permute(0, 3, 1, 2).contiguous()        # uint8 N x 3 x 96 x 96
    ytr = torch.as_tensor(ytr, device=dev).long()
    xte_t = torch.as_tensor(xte, device=dev).permute(0, 3, 1, 2).contiguous()
    xva_t = torch.as_tensor(xva, device=dev).permute(0, 3, 1, 2).contiguous()
    model = (RealFOMO(ch, bl) if args.fp32 else BoolFOMO(ch, bl, args.levels)).to(dev)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.momentum = 0.3
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    cw = torch.tensor([1.0, args.weight], device=dev)
    name = f"{'fp32_' if args.fp32 else ''}c{args.channels}_b{args.blocks}_w{args.weight:g}_e{args.epochs}_s{args.seed}"
    out = os.path.join("runs", name)
    print(f"[*] {'FP32' if args.fp32 else 'binary'} FOMO {args.channels} / {args.blocks}, thermometer {args.levels} levels, person weight {args.weight:g},"
          f" {args.epochs} epochs ({dev}) -> {out}/")
    history, t0 = [], time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm) - 127, 128):
            idx = perm[i:i + 128]
            x, y = xtr[idx], ytr[idx]
            flip = torch.rand(len(idx), device=dev) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            y = torch.where(flip.view(-1, 1, 1), y.flip(2), y)
            loss = F.cross_entropy(model(x), y, weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            s = f1_score(margins(model, xte_t) > 0, cte, ote)
            history.append(dict(epoch=ep + 1, **s))
            print(f"    epoch {ep + 1:3d}/{args.epochs} | precision {s['precision']:5.1f}% | recall {s['recall']:5.1f}% | "
                  f"F1 {s['f1']:5.1f}% | {(time.time() - t0) / 60:.1f} min")
    th, sva = calibrate(margins(model, xva_t), cva, ova)
    ste = f1_score(margins(model, xte_t) > th, cte, ote)
    print(f"\n    THRESHOLD  calibrated on {nv} held-out training images: {th:+.3f} (default 0)  -> validation F1 {sva['f1']:.1f}%")
    print(f"    TEST       default threshold: F1 {history[-1]['f1']:5.1f}% (precision {history[-1]['precision']:.1f}%, recall "
          f"{history[-1]['recall']:.1f}%)")
    print(f"               calibrated       : F1 {ste['f1']:5.1f}% (precision {ste['precision']:.1f}%, recall {ste['recall']:.1f}%)")
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pth"))
    json.dump(dict(fp32=args.fp32, channels=ch, blocks=bl, levels=args.levels, weight=args.weight, epochs=args.epochs, seed=args.seed,
                   final=history[-1], threshold=th, calibrated_validation=sva, calibrated_test=ste, history=history),
              open(os.path.join(out, "train.json"), "w"), indent=1)
    print(f"[*] saved {out}/model.pth -- test F1 {ste['f1']:.1f}% with the calibrated threshold "
          f"({history[-1]['f1']:.1f}% with the default one)")


if __name__ == "__main__":
    main()
