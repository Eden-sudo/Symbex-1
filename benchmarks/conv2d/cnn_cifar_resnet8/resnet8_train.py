"""
resnet8_train.py -- trains the binary ResNet-8 of this benchmark (model.py) on CIFAR-10.

    python resnet8_train.py --width 1                  # the shape of the MLPerf INT8 reference (16-32-64 channels)
    python resnet8_train.py --width 2                  # twice the channels (32-64-128)
    python resnet8_train.py --width 1 --seeds 3        # seeds 0, 1, 2
    python resnet8_train.py --check ../../convul_symbex_v3/bool_p2_a1_termo8/modelo.pth --width 1
                                                       # evaluate an already trained model (also the old ones)

Run from benchmarks/conv2d/cnn_cifar_resnet8/ (main environment, .venv; a GPU helps). The recipe of the earlier experiments
(convul_symbex_v3/purga_symbex.py): Adam, learning rate 5e-3 with cosine decay, 150 epochs, batch 128, random crop +
horizontal flip, BatchNorm momentum 0.3, labels only (no distillation). Each run lands in
runs/w<width>_t<levels>_s<seed>/: model.pth (the last epoch) and train.json (the test accuracy every 10 epochs).
A run already trained is skipped (--redo to train it again).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import BoolResNet8, cifar_loaders, evaluate, load_model          # noqa: E402


def run_name(width, levels, seed):
    return os.path.join("runs", f"w{width:g}_t{levels}_s{seed}")


def train(width, levels, seed, epochs, device):
    torch.manual_seed(seed)
    model = BoolResNet8(width, levels).to(device)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.momentum = 0.3
    train_loader, test_loader = cifar_loaders()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            acc = evaluate(model, test_loader, device)
            history.append(dict(epoch=ep + 1, test_accuracy=acc))
            print(f"    epoch {ep + 1:3d}/{epochs} | test {acc:.2f}% | {(time.time() - t0) / 60:.1f} min")
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=1.0, help="channel multiplier (1: 16-32-64, as the INT8 reference)")
    ap.add_argument("--levels", type=int, default=8, help="thermometer levels per color (default 8)")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds: 0, 1, ... (default 1)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--redo", action="store_true", help="train again runs that already exist")
    ap.add_argument("--check", default="", help="only evaluate this trained model (model.pth or an old modelo.pth)")
    args = ap.parse_args()
    os.chdir(HERE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.check:
        _, test_loader = cifar_loaders()
        acc = evaluate(load_model(args.check, args.width, args.levels, device), test_loader, device)
        print(f"[*] {args.check}: {acc:.2f}% on the 10,000 test images (width x{args.width:g}, {args.levels} levels)")
        return
    for seed in range(args.seeds):
        out = run_name(args.width, args.levels, seed)
        if os.path.exists(os.path.join(out, "model.pth")) and not args.redo:
            print(f"[*] {out}: already trained (--redo to train it again)")
            continue
        print(f"\n[*] binary ResNet-8, width x{args.width:g}, thermometer {args.levels} levels, seed {seed}, "
              f"{args.epochs} epochs -> {out}/  ({device})")
        model, history = train(args.width, args.levels, seed, args.epochs, device)
        os.makedirs(out, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out, "model.pth"))
        json.dump(dict(width=args.width, levels=args.levels, seed=seed, epochs=args.epochs,
                       final_accuracy=history[-1]["test_accuracy"], history=history),
                  open(os.path.join(out, "train.json"), "w"), indent=1)
        print(f"[*] saved {out}/model.pth -- final test accuracy {history[-1]['test_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
