"""
boolean.py -- training of the CDT's "boolean" method (fully binary convolutional networks): the network is trained AS
DEFINED (it is already binary), labels only. The model's contract and the export are in export/boolean.py.

    train(model, train_ds, val_ds, cfg, out_dir)       labels only, Adam + cosine, crop + flip on the GPU; with
                                                       [training] binary_optimizer = "bop", Bop for the 1-bit weights
    binary_weights(model)                              the 1-bit weights of the model (export/boolean.py's contract)
    load_trained(model, out_dir)                       a trained model (model_binarized.pth)
"""
import copy
import json
import os
import time

import torch
import torch.nn.functional as F


def _augment(x):
    """random crop (4 zero pixels of padding, as torchvision's RandomCrop(32, padding=4)) + horizontal flip"""
    n, _, h, w = x.shape
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (n,), device=x.device)
    j = torch.randint(0, 9, (n,), device=x.device)
    rows = (i.view(n, 1) + torch.arange(h, device=x.device).view(1, h))            # n x h
    cols = (j.view(n, 1) + torch.arange(w, device=x.device).view(1, w))            # n x w
    out = xp[torch.arange(n, device=x.device).view(n, 1, 1, 1), torch.arange(x.shape[1], device=x.device).view(1, -1, 1, 1),
             rows.view(n, 1, h, 1), cols.view(n, 1, 1, w)]
    flip = torch.rand(n, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


@torch.no_grad()
def _evaluate(model, loader, dev):
    model.eval()
    hits = total = 0
    for x, y in loader:
        hits += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
        total += y.numel()
    return 100.0 * hits / total


def binary_weights(model):
    """the 1-bit weights of a boolean model, by the contract above: the stem (unless it is int8), every block's conv
    and shortcut conv, the classifier or per-cell head"""
    ws = [] if getattr(model, "stem_kind", "thermometer") == "int8" else [model.stem.weight]
    for b in model.blocks:
        ws.append(b.conv.weight)
        if getattr(b, "changes", False):
            ws.append(b.ds_conv.weight)
    ws.append((model.head if hasattr(model, "head") else model.fc).weight)
    return ws


def train(model, train_ds, val_ds, cfg, out_dir, test_ds=None):
    """Trains the network as defined (labels only). Returns dict(model, plan={}, width) like stage_train.
    val_ds (carved out of the training data) is what the training reports while it runs; test_ds, when the data has
    one, is evaluated ONCE at the end (nothing is chosen with it) and saved as acc_test in train.json."""
    from torch.utils.data import DataLoader
    tr = cfg["training"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tr["seed"])
    student = copy.deepcopy(model).to(dev)
    for m in student.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.momentum = 0.3
    gen = torch.Generator().manual_seed(tr["seed"])
    on_gpu = bool(tr.get("data_on_gpu")) and dev.type == "cuda"
    if on_gpu:                                       # the whole training set on the GPU, converted once
        xs, ys = [], []
        for x, y in DataLoader(train_ds, batch_size=1024, num_workers=4):
            xs.append(x.to(dev))
            ys.append(y.to(dev))
        xs, ys = torch.cat(xs), torch.cat(ys)
        bs = tr["batch_size"]

        def epoch_batches():
            perm = torch.randperm(len(xs), device=dev, generator=torch.Generator(device=dev).manual_seed(
                int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen))))
            for i in range(0, len(perm) - bs + 1, bs):  # whole batches only (drop_last)
                idx = perm[i:i + bs]
                yield xs[idx], ys[idx]
    else:
        trl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, generator=gen, num_workers=4,
                         persistent_workers=True, drop_last=True, pin_memory=True)

        def epoch_batches():
            for x, y in trl:
                yield x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
    val_l = DataLoader(val_ds, batch_size=512, num_workers=2, persistent_workers=True)
    mode = tr.get("compile") or ""
    step_model = student
    if mode and dev.type == "cuda":                  # CUDA graphs: the training step recorded once, then replayed
        import torch._inductor.config as inductor_config
        inductor_config.compile_threads = min(4, os.cpu_count() or 1)    # the compiler's peak RAM: ~6 GB, not ~24
        step_model = torch.compile(student, mode=None if mode == "default" else mode)
    bop = None
    if tr.get("binary_optimizer", "adam") == "bop":  # the 1-bit weights to Bop, everything else (BatchNorms...) Adam
        from compiler_core.train.bop import Bop
        bits = binary_weights(student)
        bop = Bop(bits, tr.get("bop_gamma", 1e-4), tr.get("bop_tau", 1e-8))
        ids = {id(w) for w in bits}
        opt = torch.optim.Adam([p for p in student.parameters() if id(p) not in ids], lr=tr["learning_rate"])
    else:
        opt = torch.optim.Adam(student.parameters(), lr=tr["learning_rate"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tr["epochs"])
    if tr.get("distill"):
        print("    [!] method 'boolean' trains on the labels only: [training] distill is ignored")
    print(f"\n[3] TRAIN  (method 'boolean': the network as defined, {tr['epochs']} epochs, labels only, Adam "
          f"lr {tr['learning_rate']} + cosine"
          + (f", Bop for the 1-bit weights (gamma {bop.gamma:g}, tau {bop.tau:g})" if bop else "")
          + f", crop + flip | {dev}"
          + (", data on the GPU" if on_gpu else "") + (f", compile {mode}" if mode and dev.type == "cuda" else "") + ")")
    history, t0 = [], time.time()
    for ep in range(tr["epochs"]):
        student.train()
        for x, y in epoch_batches():
            loss = F.cross_entropy(step_model(_augment(x)), y)
            opt.zero_grad(set_to_none=True)
            if bop:
                bop.zero_grad()
            loss.backward()
            opt.step()
            if bop:                                  # gamma follows the learning rate's cosine
                bop.step(opt.param_groups[0]["lr"] / tr["learning_rate"])
        sched.step()
        if (ep + 1) % 10 == 0 or ep == tr["epochs"] - 1:
            acc = _evaluate(student, val_l, dev)
            history.append([ep + 1, None, None, acc])
            print(f"    epoch {ep + 1:3d}/{tr['epochs']} | validation {acc:.2f}% | {(time.time() - t0) / 60:.1f} min")
    acc_test = None
    if test_ds is not None:                          # held-out test: once, after training, never used to choose
        acc_test = _evaluate(student, DataLoader(test_ds, batch_size=512, num_workers=2), dev)
        print(f"    HELD-OUT TEST ({len(test_ds)} samples, not used for any choice): {acc_test:.2f}%")
    student = student.cpu().eval()
    torch.save(student.state_dict(), os.path.join(out_dir, "model_binarized.pth"))
    width = float(getattr(student, "width", 1.0))
    info = dict(method="boolean", width=width, levels=int(student.levels),
                binary_optimizer="bop" if bop else "adam",
                stem=getattr(student, "stem_kind", "thermometer"), plan={}, epochs=tr["epochs"],
                acc_last=history[-1][3], acc_test=acc_test, n_train=len(train_ds), n_validation=len(val_ds),
                n_test=None if test_ds is None else len(test_ds), history=history)
    json.dump(info, open(os.path.join(out_dir, "train.json"), "w"), indent=1)
    print(f"    -> {os.path.join(out_dir, 'model_binarized.pth')} | validation accuracy (last epoch) {info['acc_last']:.2f}%")
    return dict(model=student, plan={}, width=width)


def load_trained(model, out_dir):
    path = os.path.join(out_dir, "model_binarized.pth")
    if not os.path.exists(path):
        raise SystemExit(f"[X] {path} not found: there is no trained model to resume from (run from the train stage)")
    m = copy.deepcopy(model).cpu()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    print(f"\n[3] TRAIN  skipped: resumed the trained boolean network from {out_dir}")
    return dict(model=m, plan={}, width=float(getattr(m, "width", 1.0)))
