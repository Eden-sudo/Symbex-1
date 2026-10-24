"""
boolean.py -- the CDT's "boolean" method: fully binary convolutional networks (lib_symbex layers/conv/symbex_bool).

The network is trained AS DEFINED (it is already binary: no plan search, no widening), then folded into integers and
exported to the library. The model (e.g. benchmarks/cnn_cifar_resnet8/model.py, BoolResNet8) must expose:
    levels                    thermometer levels per color of its input encoding
    stem, stem_bn             binary 3x3 conv + BatchNorm (then sign)
    blocks                    Blocks: conv + bn (stride 1 or 2), and, where the shape changes, ds_conv (1x1) + ds_bn
                              after an OR pool when the block has stride 2; out = sign(BN(conv)) OR shortcut
    fc, fc_bn                 binary linear on the flattened final bits (CHW order) + BatchNorm1d
Every conv computes raw = sum of +-1 products over the in-image window; BatchNorm + sign fold into raw >= threshold
(weights negated where the BatchNorm scale is negative). The classifier keeps an int32 scale and bias per class
(2^16 fixed point) and the C picks the first class with the largest int64 logit.

    train(model, train_ds, val_ds, cfg, out_dir)       labels only, Adam + cosine, crop + flip on the GPU
    load_trained(model, out_dir)                       a trained model (model_binarized.pth)
    fold(model) / simulate(net, x_u8)                  the integer network, and its exact simulation
    export_and_validate(model, val_ds, out_dir, ...)   header + validation files + the C check on this PC (gcc)
The header offers the interface of the MLP headers (symbex_mlp_infer, SYMBEX_MLP_N_INPUTS / N_CLASSES / N_STEPS,
symbex_mlp_steps, SYMBEX_MLP_BYTES_BUFFERS), so firmware/model_test.c and the chip stage measure it unchanged.
"""
import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1
FC_SHIFT = 16


# ---------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------
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
    trl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, generator=gen, num_workers=4,
                     persistent_workers=True, drop_last=True, pin_memory=True)
    val_l = DataLoader(val_ds, batch_size=512, num_workers=2, persistent_workers=True)
    opt = torch.optim.Adam(student.parameters(), lr=tr["learning_rate"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tr["epochs"])
    if tr.get("distill"):
        print("    [!] method 'boolean' trains on the labels only: [training] distill is ignored")
    print(f"\n[3] TRAIN  (method 'boolean': the network as defined, {tr['epochs']} epochs, labels only, Adam "
          f"lr {tr['learning_rate']} + cosine, crop + flip | {dev})")
    history, t0 = [], time.time()
    for ep in range(tr["epochs"]):
        student.train()
        for x, y in trl:
            x, y = _augment(x.to(dev, non_blocking=True)), y.to(dev, non_blocking=True)
            loss = F.cross_entropy(student(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
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
    info = dict(method="boolean", width=width, levels=int(student.levels), plan={}, epochs=tr["epochs"],
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


# ---------------------------------------------------------------------------------------------------
# folding into integers
# ---------------------------------------------------------------------------------------------------
def _uses_bias(layer):
    """a bias counts only where the layer's forward adds it: the standard Conv2d / Linear do; the binary layers of
    the SYMBEX models (BinaryConv, BinaryLinear) override forward and never use it, by design"""
    standard = (torch.nn.Conv2d.forward, torch.nn.Linear.forward)
    return getattr(layer, "bias", None) is not None and type(layer).forward in standard


def _check_conv(conv, where, pointwise=False):
    """what lib_symbex runs, checked instead of assumed: a square kernel (3x3, or 1x1 for a shortcut), the same stride
    on both axes (1 or 2; 1 for a shortcut, which follows an OR pool), 'same' padding (k // 2), no dilation, no groups,
    no bias added (BatchNorm carries the offset). A model outside this would otherwise be exported as a DIFFERENT network:
    the C-vs-simulation check could not see it, since both would share the wrong assumption."""
    k, s, p = tuple(conv.kernel_size), tuple(conv.stride), conv.padding
    problems = []
    if k[0] != k[1]:
        problems.append(f"kernel {k} is not square")
    if pointwise and k[0] != 1:
        problems.append(f"a shortcut must be a 1x1 convolution (got {k[0]}x{k[1]})")
    if s[0] != s[1] or s[0] not in (1, 2) or (pointwise and s[0] != 1):
        problems.append(f"stride {s}: lib_symbex runs 1 or 2 on both axes (a shortcut: 1, after the OR pool)")
    if p != "same" and tuple(p if isinstance(p, tuple) else (p, p)) != (k[0] // 2, k[1] // 2):
        problems.append(f"padding {p}: lib_symbex pads k // 2 = {k[0] // 2} on every side")
    if tuple(conv.dilation) != (1, 1) or conv.groups != 1:
        problems.append(f"dilation {tuple(conv.dilation)} / groups {conv.groups}: not supported")
    if _uses_bias(conv):
        problems.append("the convolution adds a bias: use bias=False (the BatchNorm after it carries the offset)")
    if problems:
        raise ValueError(f"boolean export, {where}: " + "; ".join(problems))


def _fold_conv(conv, bn, where="conv", pointwise=False):
    """binary conv + BatchNorm + sign -> sign weights (O, C, k, k) in +-1 (rows negated where the scale < 0) and
    integer thresholds on raw: bit = raw >= threshold"""
    _check_conv(conv, where, pointwise)
    if bn.running_mean is None or not bn.affine:
        raise ValueError(f"boolean export, {where}: its BatchNorm needs running statistics and affine parameters")
    w = torch.where(conv.weight.detach() >= 0, 1.0, -1.0).double()
    g = bn.weight.detach().double()
    sd = torch.sqrt(bn.running_var.detach().double() + bn.eps)
    t = bn.running_mean.detach().double() - bn.bias.detach().double() * sd / torch.where(g == 0, torch.ones_like(g), g)
    thr = np.empty(len(g), dtype=np.int64)
    for o in range(len(g)):
        if g[o] == 0:
            thr[o] = INT_MIN if bn.bias[o] >= 0 else INT_MAX
        elif g[o] > 0:                                       # BN(raw) >= 0  <=>  raw >= T
            thr[o] = math.ceil(float(t[o]))
        else:                                                # raw <= T  <=>  (-raw) >= -T
            w[o] = -w[o]
            thr[o] = math.ceil(-float(t[o]))
    return dict(w=w.numpy(), thr=np.clip(thr, INT_MIN, INT_MAX), k=conv.kernel_size[0], stride=conv.stride[0])


def fold(model, threshold=0.0):
    """the trained BoolResNet8-like model -> integer network (dict of numpy arrays). A model with head / head_bn (a
    per-cell detector, FOMO-style) gets a per-cell head; threshold = the calibrated decision threshold on
    (logit of class 1 - logit of class 0), folded into the integer bias of class 1"""
    model = model.cpu().eval()
    net = dict(levels=int(model.levels), stem=_fold_conv(model.stem, model.stem_bn, "stem"), blocks=[])
    if model.stem.in_channels != 3 * net["levels"]:
        raise ValueError(f"boolean export, stem: {model.stem.in_channels} input channels, the thermometer gives "
                         f"3 x {net['levels']} = {3 * net['levels']}")
    for i, b in enumerate(model.blocks):
        blk = dict(conv=_fold_conv(b.conv, b.bn, f"block {i}"), shortcut=None, pool=False)
        if getattr(b, "changes", False):
            blk["shortcut"] = _fold_conv(b.ds_conv, b.ds_bn, f"block {i} shortcut", pointwise=True)
            blk["pool"] = b.conv.stride[0] == 2
        elif b.conv.stride[0] != 1 or b.conv.in_channels != b.conv.out_channels:
            raise ValueError(f"boolean export, block {i}: it changes the shape (stride {b.conv.stride[0]}, "
                             f"{b.conv.in_channels} -> {b.conv.out_channels}) but has no shortcut convolution")
        net["blocks"].append(blk)
    lin, bn = (model.head, model.head_bn) if hasattr(model, "head") else (model.fc, model.fc_bn)
    g = bn.weight.detach().double()
    sd = torch.sqrt(bn.running_var.detach().double() + bn.eps)
    a = (g / sd).numpy()
    c = (bn.bias.detach().double() - bn.running_mean.detach().double() * g / sd).numpy()
    if hasattr(model, "head") and threshold:
        c = c.copy()
        c[1] -= threshold                                  # class 1 wins only where its margin passes the threshold
    for name, v in (("scale", a), ("bias", c)):
        if np.abs(v).max() * 2 ** FC_SHIFT >= 2 ** 31:
            raise ValueError(f"boolean export: the classifier {name} does not fit in int32 with {FC_SHIFT} fraction bits")
    w = lin.weight.detach()
    if hasattr(model, "head"):
        _check_conv(lin, "per-cell head", pointwise=True)
    elif _uses_bias(lin):
        raise ValueError("boolean export, classifier: it adds a bias; use bias=False (its BatchNorm carries the offset)")
    net["cells"] = hasattr(model, "head")
    net["fc_w"] = torch.where((w.flatten(1) if net["cells"] else w) >= 0, 1.0, -1.0).double().numpy()
    # classifier: (classes, C*H*W) in CHW order | per-cell head: (classes, C), applied at every cell
    net["fc_scale"] = np.round(a * 2 ** FC_SHIFT).astype(np.int64)
    net["fc_bias"] = np.round(c * 2 ** FC_SHIFT).astype(np.int64)
    return net


def _grid_checksum(grid):
    """symbex_bool_grid_checksum: FNV-1a over the class bytes of the grid (row by row), 31 bits"""
    h = 2166136261
    for v in np.asarray(grid, dtype=np.uint8).ravel():
        h = ((h ^ int(v)) * 16777619) & 0xFFFFFFFF
    return h & 0x7FFFFFFF


@torch.no_grad()
def simulate(net, x_u8, batch=500, device="cpu", return_bits=False):
    """the integer network on uint8 images (N, H, W, 3), exactly as lib_symbex computes it (float64 holds every
    integer here exactly). Returns the classes (first largest logit); for a per-cell head, the grid checksum of every
    image (what the chip reports), or the grids themselves with return_bits=True."""
    dev = torch.device(device)
    t = torch.as_tensor((np.arange(net["levels"]) + 1) * 256 // (net["levels"] + 1), dtype=torch.float64, device=dev)

    def layer(x, c):
        raw = F.conv2d(x, torch.as_tensor(c["w"], device=dev), stride=c["stride"], padding=c["k"] // 2)
        return torch.where(raw >= torch.as_tensor(c["thr"], dtype=torch.float64, device=dev).view(1, -1, 1, 1), 1.0, -1.0).double()

    preds = []
    for i in range(0, len(x_u8), batch):
        x = torch.as_tensor(np.ascontiguousarray(x_u8[i:i + batch]), device=dev).permute(0, 3, 1, 2).double()
        x = (x.unsqueeze(2) >= t.view(1, 1, -1, 1, 1)).flatten(1, 2).double() * 2 - 1      # thermometer, +-1
        x = layer(x, net["stem"])
        for b in net["blocks"]:
            y = layer(x, b["conv"])
            if b["shortcut"] is not None:
                src = F.max_pool2d(x, 2) if b["pool"] else x                             # OR of bits
                a = layer(src, b["shortcut"])
            else:
                a = x
            x = torch.maximum(y, a)                                                      # OR of bits
        if net.get("cells"):                                                             # per cell: (N, classes, h, w)
            raw = torch.einsum("nchw,kc->nkhw", x, torch.as_tensor(net["fc_w"], device=dev)).cpu().numpy().astype(np.int64)
            logit = raw * net["fc_scale"][None, :, None, None] + net["fc_bias"][None, :, None, None]
            grids = np.argmax(logit, axis=1).astype(np.uint8)                            # the first largest
            if return_bits:
                preds.append(grids)
            else:
                preds.append(np.array([_grid_checksum(g) for g in grids]))
            continue
        raw = x.flatten(1) @ torch.as_tensor(net["fc_w"], device=dev).T                  # CHW order, as PyTorch
        logit = raw.cpu().numpy().astype(np.int64) * net["fc_scale"][None] + net["fc_bias"][None]
        preds.append(np.argmax(logit, axis=1))                                           # the first largest
    return np.concatenate(preds)


# ---------------------------------------------------------------------------------------------------
# the C header
# ---------------------------------------------------------------------------------------------------
def _words(bits, n_words):
    """(..., n) 0/1 -> (..., n_words) uint32, bit i of word i // 32"""
    bits = np.asarray(bits, dtype=np.uint8)
    pad = np.zeros(bits.shape[:-1] + (n_words * 32,), dtype=np.uint8)
    pad[..., :bits.shape[-1]] = bits
    return np.packbits(pad, axis=-1, bitorder="little").view("<u4").reshape(bits.shape[:-1] + (n_words,))


def _conv_arrays(c):
    """weights PACKED as lib_symbex builds its windows: the C bits of kernel pixel q = kh * k + kw start at bit q * C
    of the row (with C a multiple of 32 this is the plain [kh][kw][words] layout); rows padded to a multiple of 4
    words; counts [o][kh*k+kw]"""
    w = c["w"] > 0                                                                       # (O, C, k, k) bits
    o, cin, k, _ = w.shape
    n = -(-(k * k * cin) // 32)
    rw = -(-n // 4) * 4
    hwc = w.transpose(0, 2, 3, 1)                                                        # (O, k, k, C)
    rows = np.zeros((o, rw), dtype=np.uint32)
    rows[:, :n] = _words(hwc.reshape(o, k * k * cin), n)
    return rows, rw, hwc.reshape(o, k * k, cin).sum(2)


def _c_array(f, ctype, name, values, attr=""):
    values = np.asarray(values).ravel()
    f.write(f"static SYMBEX_ROM {ctype} {name}[{len(values)}]{attr} = {{")
    for i, v in enumerate(values):
        lit = f"0x{int(v):08X}u" if ctype == "uint32_t" else ("(-2147483647-1)" if int(v) == INT_MIN else str(int(v)))
        f.write(("\n    " if i % 12 == 0 else " ") + lit + ",")
    f.write("\n};\n")


def write_header(net, path, in_size=32):
    """model_symbex.h for lib_symbex's symbex_bool_net_run, with the interface of the MLP headers. Returns
    (flash bytes of weights + parameters, RAM bytes of the buffers, step names, layer shapes)."""
    convs, flash, steps, shapes = [], 0, [], []
    res = in_size

    def add(c, cin_h):
        nonlocal flash
        rows, rw, cnt = _conv_arrays(c)
        k, s = c["k"], c["stride"]
        out_h = (cin_h + 2 * (k // 2) - k) // s + 1
        convs.append(dict(rows=rows, rw=rw, cnt=cnt, thr=c["thr"], cin=c["w"].shape[1], cout=c["w"].shape[0], k=k, s=s,
                          h=cin_h, oh=out_h))
        flash += rows.size * 4 + len(c["thr"]) * 4 + cnt.size * 2
        return len(convs) - 1, out_h

    stem_i, res = add(net["stem"], res)
    steps.append(f"thermometer ({net['levels']} levels) + stem {convs[stem_i]['cin']}->{convs[stem_i]['cout']} {in_size}x{in_size}")
    blocks = []
    for bi, b in enumerate(net["blocks"]):
        h_in = res
        ci, res = add(b["conv"], h_in)
        si = None
        if b["shortcut"] is not None:
            si, _ = add(b["shortcut"], h_in // 2 if b["pool"] else h_in)
        blocks.append((ci, si, b["pool"]))
        c = convs[ci]
        steps.append(f"block {bi}: conv {c['cin']}->{c['cout']} {h_in}x{h_in}{' s2' if c['s'] == 2 else ''}"
                     + (" + OR pool + 1x1" if b["pool"] else (" + 1x1" if si is not None else "")) + " | OR")
    cells = bool(net.get("cells"))
    steps.append("per-cell head" if cells else "classifier")
    last = convs[blocks[-1][0]] if blocks else convs[stem_i]
    ch, fh = last["cout"], last["oh"]
    ppx = (ch + 31) // 32
    fcw = net["fc_w"] > 0
    n_cls = fcw.shape[0]
    if cells:                                                                            # (classes, C): one row per class
        fc_words = ppx
        fc_rw = -(-ppx // 4) * 4
        fc_rows = np.zeros((n_cls, fc_rw), dtype=np.uint32)
        fc_rows[:, :ppx] = _words(fcw, ppx)
        fc_bits = ch
    else:                                                                                # (classes, C*H*W), CHW
        fc_words = fh * fh * ppx
        fc_rw = -(-fc_words // 4) * 4
        hwc = fcw.reshape(n_cls, ch, fh, fh).transpose(0, 2, 3, 1).reshape(n_cls * fh * fh, ch)
        fc_rows = np.zeros((n_cls, fc_rw), dtype=np.uint32)
        fc_rows[:, :fc_words] = _words(hwc, ppx).reshape(n_cls, fc_words)
        fc_bits = fh * fh * ch
    fc_cnt = fcw.sum(1)
    flash += fc_rows.size * 4 + n_cls * 12
    # every map buffer as large as what symbex_bool_net_run writes into IT (it rotates 4 buffers): far less RAM than
    # four buffers as large as the largest map (the thermometer input of a 96 x 96 image alone is 36 KB)
    words_of = lambda c, h, ch_: h * h * ((ch_ + 31) // 32)
    size = [in_size * in_size * ((3 * net["levels"] + 31) // 32), 0, 0, 0]
    size[1] = words_of(None, convs[stem_i]["oh"], convs[stem_i]["cout"])
    x, y, p, s = 1, 0, 2, 3
    for ci, si, pool in blocks:
        c = convs[ci]
        size[y] = max(size[y], words_of(None, c["oh"], c["cout"]))
        if si is not None:
            if pool:
                size[p] = max(size[p], words_of(None, c["h"] // 2, c["cin"]))
            sc = convs[si]
            size[s] = max(size[s], words_of(None, sc["oh"], sc["cout"]))
        x, y = y, x
    size[x] = max(size[x], fc_rw if not cells else fh * fh * ppx + fc_rw)   # the head may read a row past the end
    size = [max(-(-v // 4) * 4, 4) for v in size]
    max_rw = max(max(c["rw"] for c in convs), 4)
    max_out = max(max(c["cout"] for c in convs), n_cls)
    ram = 4 * (sum(size) + max_rw + 2 * max_out) + (fh * fh if cells else 0)
    guard = "SYMBEX_BOOL_MODEL_H"
    with open(path, "w") as f:
        f.write(f"/* generated by the CDT (compiler_core/export/boolean.py): a fully binary network for lib_symbex's\n"
                f"   symbex_bool_net_run, with the interface of the MLP headers (symbex_mlp_infer). Do not edit. */\n"
                f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n#include \"layers/conv/symbex_bool.h\"\n"
                f"#ifndef SYMBEX_ROM\n#define SYMBEX_ROM const           /* -DSYMBEX_ROM= : weights in RAM */\n#endif\n\n")
        for i, c in enumerate(convs):
            _c_array(f, "uint32_t", f"BW{i}", c["rows"], " __attribute__((aligned(16)))")
            _c_array(f, "int32_t", f"BT{i}", c["thr"])
            _c_array(f, "int16_t", f"BC{i}", c["cnt"])

        def conv_init(i):
            c = convs[i]
            return (f"{{{c['cin']}, {c['cout']}, {c['k']}, {c['s']}, {c['h']}, {c['h']}, {c['oh']}, {c['oh']}, {c['rw']}, "
                    f"BW{i}, BT{i}, BC{i}}}")
        for bi, (ci, si, pool) in enumerate(blocks):
            if si is not None:
                f.write(f"static const struct SymbexBoolConv BS{bi} = {conv_init(si)};\n")
        f.write("static const struct SymbexBoolBlock B_BLOCKS[] = {\n")
        for bi, (ci, si, pool) in enumerate(blocks):
            f.write(f"    {{{conv_init(ci)}, {'&BS' + str(bi) if si is not None else 'NULL'}, {1 if pool else 0}}},\n")
        f.write("};\n")
        _c_array(f, "uint32_t", "FC_W", fc_rows, " __attribute__((aligned(16)))")
        _c_array(f, "int32_t", "FC_C", fc_cnt)
        _c_array(f, "int32_t", "FC_S", net["fc_scale"])
        _c_array(f, "int32_t", "FC_B", net["fc_bias"])
        for i in range(4):
            f.write(f"static uint32_t B_MAP{i}[{size[i]}] __attribute__((aligned(16)));\n")
        f.write(f"static uint32_t B_WINDOW[{max_rw}] __attribute__((aligned(16)));\n"
                f"static int32_t B_SCRATCH[{2 * max_out}];\n")
        if cells:
            f.write(f"static uint8_t B_CELLS[{fh * fh}];      /* the class of every cell of the {fh} x {fh} grid */\n")
        f.write(f"static const struct SymbexBoolNet symbex_net = {{{in_size}, {in_size}, {net['levels']}, {conv_init(stem_i)}, "
                f"B_BLOCKS, {len(blocks)}, FC_W, {fc_rw}, {n_cls}, {fc_bits}, {fc_words}, FC_C, FC_S, FC_B, "
                f"{{B_MAP0, B_MAP1, B_MAP2, B_MAP3}}, B_WINDOW, B_SCRATCH, "
                + (f"B_CELLS, {fh}, {fh}}};   /* per-cell head: returns the grid checksum */\n\n" if cells else
                   "NULL, 0, 0};   /* no per-cell head: a classifier */\n\n"))
        f.write(f"#define SYMBEX_MLP_N_INPUTS {in_size * in_size * 3}      /* uint8 RGB, HWC */\n"
                f"#define SYMBEX_MLP_N_CLASSES {n_cls}\n#define SYMBEX_MLP_N_STEPS {len(steps)}\n"
                f"#define SYMBEX_MLP_BYTES_BUFFERS {ram}u\n"
                "static const char* const symbex_mlp_steps[SYMBEX_MLP_N_STEPS] = {\n"
                + "".join(f"    \"{s}\",\n" for s in steps) + "};\n"
                "static inline int symbex_mlp_infer(const uint8_t* x) { return symbex_bool_net_run(&symbex_net, x); }\n"
                f"#endif /* {guard} */\n")
    for c in convs:
        shapes.append(dict(cin=c["cin"], cout=c["cout"], k=c["k"], stride=c["s"], h=c["h"], out_h=c["oh"],
                           bit_ops=c["cout"] * c["oh"] * c["oh"] * c["k"] * c["k"] * c["cin"]))
    return flash, ram, steps, shapes


def validate_c(header_path, lib_path, x_u8, expected):
    """gcc on this PC: the header + lib_symbex (symbex_bool.c, the generic HAL) on x_u8; (equal, total, message)"""
    from .dense import _TEST_MAIN
    if shutil.which("gcc") is None:
        return None, 0, "gcc not found: the C exactness check was skipped"
    inc = os.path.join(lib_path, "include")
    sources = [os.path.join(lib_path, "src", "layers", "conv", "symbex_bool.c"),
               os.path.join(lib_path, "src", "hal", "generic", "symbex_hal_impl.c")]
    for r in [inc] + sources:
        if not os.path.exists(r):
            return None, 0, f"lib_symbex not found at {lib_path} ({r} missing): set [output] lib_path"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(header_path, os.path.join(d, "model_symbex.h"))
        open(os.path.join(d, "main.c"), "w").write(_TEST_MAIN)
        exe = os.path.join(d, "check")
        r = subprocess.run(["gcc", "-O2", "-std=gnu11", "-I", inc, "-I", d, "-o", exe, os.path.join(d, "main.c")] + sources,
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C did not compile:\n" + r.stderr[-2000:]
        data = os.path.join(d, "x.u8")
        np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(data)
        r = subprocess.run([exe, data, str(len(x_u8))], capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C program failed while running"
        pred_c = np.array([int(v) for v in r.stdout.split()])
    return int((pred_c == expected).sum()), len(expected), ""


def _val_u8(val_ds, n):
    """the first n validation images as uint8 HWC (the CDT gives tensors in [0, 1], C x H x W) + labels"""
    xs, ys = [], []
    for i in range(min(n, len(val_ds))):
        x, y = val_ds[i]
        xs.append(torch.round(x * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy())
        ys.append(int(y))
    return np.stack(xs), np.array(ys)


def export_and_validate(model, val_ds, out_dir, lib_path, n_validation=2000):
    from .dense import save_validation
    x_u8, y = _val_u8(val_ds, n_validation)
    model = model.cpu().eval()
    with torch.no_grad():
        p_torch = np.concatenate([model(torch.as_tensor(x_u8[i:i + 250]).permute(0, 3, 1, 2)).argmax(1).numpy()
                                  for i in range(0, len(x_u8), 250)])
    net = fold(model)
    p_sim = simulate(net, x_u8)
    header = os.path.join(out_dir, "model_symbex.h")
    flash, ram, steps, shapes = write_header(net, header, x_u8.shape[1])
    equal, total, msg = validate_c(header, lib_path, x_u8, p_sim)
    save_validation(out_dir, x_u8.reshape(len(x_u8), -1), y, p_sim)
    acc_t, acc_s = 100 * (p_torch == y).mean(), 100 * (p_sim == y).mean()
    fid = 100 * (p_sim == p_torch).mean()
    print(f"    network     : thermometer {net['levels']} levels -> stem -> {len(net['blocks'])} blocks (conv OR "
          f"shortcut) -> classifier | every layer 1 bit, no multiplication")
    print(f"    header      : {header}  ({flash / 1024:.1f} KB of weights + thresholds, {ram / 1024:.1f} KB of buffers)")
    print(f"    validation  : {len(x_u8)} samples")
    print(f"    accuracy    : PyTorch {acc_t:.2f}%   integer simulation {acc_s:.2f}%")
    print(f"    FIDELITY    : {fid:.2f}%  (same prediction PyTorch vs. integers)")
    if equal is None:
        print(f"    EXACTNESS   : not checked ({msg})")
    else:
        print(f"    EXACTNESS   : {100 * equal / total:.2f}%  (compiled C vs. integer simulation, must be 100%) "
              + ("OK" if equal == total else "[X]"))
    return dict(method="boolean", header=header, flash_bytes=flash, ram_buffers=ram, steps=steps, layers=shapes,
                accuracy_torch=acc_t, accuracy_sim=acc_s, fidelity=fid,
                exactness=None if equal is None else 100 * equal / total, arithmetic="and (HAL rows)",
                _x_u8=x_u8.reshape(len(x_u8), -1), _y=y, _pred_sim=p_sim)
