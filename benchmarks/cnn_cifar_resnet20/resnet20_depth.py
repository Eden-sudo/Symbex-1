"""
resnet20_depth.py -- does a published ResNet-20 need all its depth? (Most of it, it turns out, it does not.)

    python resnet20_depth.py                  # everything, 3 epochs of adaptation (about 10-15 min on a GPU)
    python resnet20_depth.py --finetune 5     # longer adaptation
    python resnet20_depth.py --no-train       # only the strict test (seconds)

Run from benchmarks/cnn_cifar_resnet20/ (main environment, .venv). The model is the published FP32 ResNet-20 (model.py,
92.59% on CIFAR-10): 3 stages of 3 residual blocks; the first block of stages 2 and 3 halves the map (downsample).
A block is removed by keeping only its shortcut (the identity, or its 1x1 downsample convolution).

    1. STRICT TEST    remove one block, measure at once. The layers after it were trained for its output, so they are
                      thrown off too: this overstates what each block holds.
    2. CONTROL        the full model trained N more epochs, nothing removed: the reference for 3 and 4.
    3. ONE BLOCK      remove one block, let the rest of the network adapt N epochs, measure: what was ONLY in it.
    4. SHALLOWER      remove one block per stage (ResNet-14) and two per stage (ResNet-8), N epochs: are the losses
                      additive, or does the network keep absorbing them?
Every run starts again from the published weights. All accuracies on the 10,000 test images.
Writes results/resnet20_depth.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("layer1", "layer2", "layer3")
SHALLOWER = {"ResNet-14 (removes layer1.2, layer2.2, layer3.2)": ["layer1.2", "layer2.2", "layer3.2"],
             "ResNet-8  (removes blocks 1 and 2 of every stage)": ["layer1.1", "layer1.2", "layer2.1", "layer2.2",
                                                                   "layer3.1", "layer3.2"]}


class ShortcutOnly(nn.Module):
    """what is left of a residual block without its two convolutions: its shortcut"""
    def __init__(self, block):
        super().__init__()
        self.down = block.downsample

    def forward(self, x):
        return torch.relu(self.down(x)) if self.down is not None else x


class Bench:
    """the published model, the CIFAR-10 loaders, and the three operations every experiment needs"""

    def __init__(self, classes):
        from model import ResNet20, cifar_loaders, get_pretrained
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet20()
        self.model.load_state_dict(torch.load(get_pretrained(), map_location="cpu"))
        self.model.to(self.dev).eval()
        self.published = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.classes = classes
        self.train, self.test = cifar_loaders()

    def blocks(self):
        return [f"{s}.{i}" for s in STAGES for i in range(len(getattr(self.model.net, s)))]

    def block_params(self, name):
        stage, i = name.split(".")
        b = getattr(self.model.net, stage)[int(i)]
        return sum(p.numel() for p in b.parameters()) - (sum(p.numel() for p in b.downsample.parameters())
                                                         if b.downsample is not None else 0)

    @torch.no_grad()
    def evaluate(self):
        """accuracy and accuracy per class"""
        self.model.eval()
        hits, totals = np.zeros(len(self.classes)), np.zeros(len(self.classes))
        for x, y in self.test:
            p = self.model(x.to(self.dev)).argmax(1).cpu().numpy()
            y = y.numpy()
            np.add.at(totals, y, 1)
            np.add.at(hits, y[p == y], 1)
        return 100 * hits.sum() / totals.sum(), 100 * hits / totals

    def adapt(self, epochs):
        """a few epochs of SGD on the whole network (cosine from 0.01)"""
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(self.train))
        self.model.train()
        for _ in range(epochs):
            for x, y in self.train:
                opt.zero_grad()
                nn.functional.cross_entropy(self.model(x.to(self.dev)), y.to(self.dev)).backward()
                opt.step()
                sched.step()
        self.model.eval()

    def run(self, removed, epochs):
        """remove the given blocks, measure (and again after `epochs` of adaptation), restore the published model"""
        originals = {}
        for name in removed:
            stage, i = name.split(".")
            seq = getattr(self.model.net, stage)
            originals[name] = seq[int(i)]
            seq[int(i)] = ShortcutOnly(seq[int(i)]).to(self.dev)
        at_once = self.evaluate()
        adapted = None
        if epochs:
            self.adapt(epochs)
            adapted = self.evaluate()
        for name, block in originals.items():
            stage, i = name.split(".")
            getattr(self.model.net, stage)[int(i)] = block
        self.model.load_state_dict(self.published)
        return at_once, adapted


def bar(drop):
    return "#" * max(0, min(40, int(round(drop))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", type=int, default=3, help="epochs of adaptation after removing blocks (default 3)")
    ap.add_argument("--no-train", action="store_true", help="only the strict test (no adaptation, seconds)")
    args = ap.parse_args()
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    from model import CLASSES
    epochs = 0 if args.no_train else args.finetune
    t0 = time.time()
    b = Bench(CLASSES)
    n_params = sum(p.numel() for p in b.model.parameters())
    base, base_cls = b.evaluate()
    print(f"\n    THE PUBLISHED RESNET-20: {base:.2f}% on the 10,000 CIFAR-10 test images, {n_params:,} parameters")
    out = dict(published=base, parameters=n_params, finetune_epochs=epochs, blocks=[], shallower=[])

    reference = base
    if epochs:
        print(f"\n[*] control: the full model, {epochs} more epoch(s), nothing removed ...")
        _, (ctrl, _) = b.run([], epochs)
        reference = ctrl
        out["control"] = ctrl
        print(f"    CONTROL: {ctrl:.2f}%  -- the reference for everything below (training itself moves it "
              f"{ctrl - base:+.2f} points)")

    print(f"\n[*] one block at a time{f', then {epochs} epoch(s) of adaptation' if epochs else ''} ...")
    for name in b.blocks():
        (acc, _), adapted = b.run([name], epochs)
        row = dict(block=name, params=b.block_params(name), at_once=acc, drop_at_once=base - acc)
        if adapted:
            row.update(adapted=adapted[0], drop_adapted=reference - adapted[0],
                       worst_class=CLASSES[int(np.argmax(base_cls - adapted[1]))])
        out["blocks"].append(row)
        print(f"      {name:<9} removed: {acc:6.2f}%" + (f"  ->  adapted: {adapted[0]:6.2f}%" if adapted else ""))

    print("\n    1. STRICT TEST and 3. ONE BLOCK  (drop in points; adapted: against the control)\n")
    hdr = f"{'adapted':>9}{'drop':>7}  {'':<22}" if epochs else ""
    print(f"    {'block':<10}{'params':>9}{'of model':>10}{'at once':>10}{'drop':>7}  {hdr}")
    for r in out["blocks"]:
        line = f"    {r['block']:<10}{r['params']:>9,}{100 * r['params'] / n_params:>9.1f}%{r['at_once']:>9.2f}%{r['drop_at_once']:>7.1f}  "
        if epochs:
            line += f"{r['adapted']:>8.2f}%{r['drop_adapted']:>7.2f}  {bar(4 * r['drop_adapted']):<22}"
        print(line)

    if epochs:
        print(f"\n[*] shallower networks, {epochs} epoch(s) of adaptation ...")
        for title, removed in SHALLOWER.items():
            (acc, _), adapted = b.run(removed, epochs)
            params = n_params - sum(b.block_params(n) for n in removed)
            out["shallower"].append(dict(network=title, removed=removed, params=params, at_once=acc,
                                         adapted=adapted[0], drop_adapted=reference - adapted[0]))
        print("\n    4. SHALLOWER NETWORKS  (several blocks removed at once, then adapted)\n")
        print(f"    {'network':<52}{'params':>9}{'of model':>10}{'accuracy':>10}{'drop':>7}")
        print(f"    {'ResNet-20 (control)':<52}{n_params:>9,}{100.0:>9.1f}%{reference:>9.2f}%{0:>7.2f}")
        for r in out["shallower"]:
            print(f"    {r['network']:<52}{r['params']:>9,}{100 * r['params'] / n_params:>9.1f}%{r['adapted']:>9.2f}%"
                  f"{r['drop_adapted']:>7.2f}")
        cheap = [r for r in out["blocks"] if r["drop_adapted"] < 1.0]
        print(f"\n    {len(cheap)} of {len(out['blocks'])} blocks cost less than 1 point once the network adapts "
              f"({100 * sum(r['params'] for r in cheap) / n_params:.1f}% of the parameters).")
    path = os.path.join("results", "resnet20_depth.json")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n    -> {path}   ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
