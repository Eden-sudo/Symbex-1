"""
boolean.py -- the CDT's "boolean" method: fully binary convolutional networks (lib_symbex layers/conv/symbex_bool).

The network is trained AS DEFINED (it is already binary: no plan search, no widening), then folded into integers and
exported to the library. The model (e.g. benchmarks/conv2d/cnn_cifar_resnet8/model.py, BoolResNet8) must expose:
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
Built with -DSYMBEX_BOOL_VERTICAL, the same header runs the vertical engine (symbex_bool_vertical.h, the fastest on
the ESP32-S3; docs/vertical.md) on the same data, with the same results: chip_test.py --engine vertical.
Adding -DSYMBEX_BOOL_VERTICAL_PREPARED (optional) builds its plane lists and thresholds once into RAM: more RAM,
faster (chip_test.py --engine vertical_prepared). -DSYMBEX_BOOL_VERTICAL_PREPARED_COMPACT instead: the lists prepared,
the thresholds kept compact (one value per output channel and border kind): about half that RAM, a little slower
(chip_test.py --engine vertical_compact).
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
    opt = torch.optim.Adam(student.parameters(), lr=tr["learning_rate"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tr["epochs"])
    if tr.get("distill"):
        print("    [!] method 'boolean' trains on the labels only: [training] distill is ignored")
    print(f"\n[3] TRAIN  (method 'boolean': the network as defined, {tr['epochs']} epochs, labels only, Adam "
          f"lr {tr['learning_rate']} + cosine, crop + flip | {dev}"
          + (", data on the GPU" if on_gpu else "") + (f", compile {mode}" if mode and dev.type == "cuda" else "") + ")")
    history, t0 = [], time.time()
    for ep in range(tr["epochs"]):
        student.train()
        for x, y in epoch_batches():
            loss = F.cross_entropy(step_model(_augment(x)), y)
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


def _vplane_words(ch, h, w):
    """words of a channel-plane map of the vertical engine (symbex_bool_vertical.h: plane stride + 4 zero words)"""
    return ch * (((h * w // 32 + 3) & ~3) + 4) + 4


def vertical_problem(convs, levels, in_size):
    """None if the vertical engine runs these layers, else why not (symbex_bool_vertical_check, the same rules)"""
    if 3 * levels > 32:
        return "thermometer with more than 32 bits per pixel"
    if 32 % in_size or in_size * in_size % 32:
        return "input size (width must divide 32, h * w a multiple of 32)"
    for c in convs:
        k, s, h, oh, K = c["k"], c["s"], c["h"], c["oh"], c["k"] * c["k"] * c["cin"]
        if k not in (1, 3):
            return "kernel other than 1 x 1 or 3 x 3"
        if (s == 1 and oh != h) or (s == 2 and (k != 3 or h != 2 * oh)) or s not in (1, 2):
            return f"stride {s} with kernel {k} ({h} -> {oh})"
        if 32 % h or 32 % oh or h * h % 32 or oh * oh % 32:
            return f"map size {h} -> {oh} (width must divide 32, h * w a multiple of 32)"
        if oh * oh > 256 * 128:
            return f"output map {oh} x {oh} above 32768 pixels (more than 256 tiles of 128)"
        if K % 8 or K >= 2048:
            return f"k * k * in_ch = {K}: not a multiple of 8 or not below 2048"
    return None


def vertical_work_bytes(convs, blocks, levels, in_size, fc_rw):
    """bytes of the vertical engine's work block: symbex_bool_vertical_work_bytes (net_sizes + carve), the same sums"""
    sz = dict(map=_vplane_words(3 * levels, in_size, in_size), phases=0, xq=0, ap=0, km=0, bt=0, ba=0, list=0, tiles=0)
    for c in convs:
        nwo = c["oh"] * c["oh"] // 32
        TL, K = (nwo + 3) // 4, c["k"] * c["k"] * c["cin"]
        K8 = (K + 8) // 8 * 8
        for key, v in (("map", _vplane_words(c["cout"], c["oh"], c["oh"])),
                       ("phases", 4 * _vplane_words(c["cin"], c["oh"], c["oh"]) if c["s"] == 2 else 0),
                       ("xq", TL * K8), ("ap", TL * 12), ("km", TL * 9), ("bt", TL * 12),
                       ("ba", ((K8 // 8 + 7) & ~7) + 8), ("list", K + 18), ("tiles", TL)):
            sz[key] = max(sz[key], v)
    for ci, si, pool in blocks:
        if si is not None and pool:
            c = convs[ci]
            sz["map"] = max(sz["map"], _vplane_words(c["cin"], c["h"] // 2, c["h"] // 2))
    last = convs[blocks[-1][0]] if blocks else convs[0]
    hwc = last["oh"] * last["oh"] * ((last["cout"] + 31) // 32) + fc_rw + 4
    r16 = lambda b: (b + 15) & ~15
    shared = max(r16(sz["km"] * 16) + r16(sz["bt"] * 16), r16(sz["phases"] * 4))    # the phases share km and bt
    return (4 * r16(sz["map"] * 4) + shared + r16(sz["xq"] * 16) + r16(sz["ap"] * 16) + 2 * r16(sz["ba"] * 16) + r16(16 * 16) + r16(15 * 16)
            + r16(sz["list"] * 2 + 4) + 3 * r16(sz["tiles"]) + r16(sz["tiles"] * 9) + r16(256 * 20) + r16(hwc * 4))


def _vertical_tile_types(c, with_kinds=False):
    """tile types of a conv in the vertical engine: border_kinds in symbex_bool_vertical.c (the tiles of 128 output
    pixels with the same border-kind masks), the same steps; with_kinds: (tile types, border kinds)"""
    k, s, H, OH = c["k"], c["s"], c["h"], c["oh"]
    pad, OW, W = k // 2, OH, H
    nwo = OH * OW // 32
    TL = (nwo + 3) // 4

    def valid(o, n):
        return sum(1 << i for i in range(k) if 0 <= o * s - pad + i < n)

    def kinds(n_out, n_in):
        vals, cnt = [], []
        for o in range(n_out):
            m = valid(o, n_in)
            if m not in vals:
                vals.append(m)
                cnt.append(0)
            cnt[vals.index(m)] += 1
        return vals, cnt
    cv, cc = kinds(OW, W)
    rv, rc = kinds(OH, H)
    cmask = [0] * len(cv)
    for ox in range(OW):
        for j in range(ox, 32, OW):
            cmask[cv.index(valid(ox, W))] |= 1 << j
    nc, NK = len(cv), len(rv) * len(cv)
    kbase = cc.index(max(cc)) + rc.index(max(rc)) * nc       # the first commonest, as the C
    rows_per_word, rowbits = 32 // OW, (1 << OW) - 1
    km = [[[0] * 4 for _ in range(NK)] for _ in range(TL)]
    for wd in range(4 * TL):
        t, j = divmod(wd, 4)
        if wd >= nwo:
            km[t][kbase][j] = 0xFFFFFFFF
            continue
        rbits = [0] * len(rv)
        for r in range(rows_per_word):
            rbits[rv.index(valid(wd * rows_per_word + r, H))] |= rowbits << (r * OW)
        for a in range(len(rv)):
            for b in range(nc):
                km[t][a * nc + b][j] = rbits[a] & cmask[b]
    ntt = len({tuple(tuple(m) for m in tile) for tile in km})
    return (ntt, NK) if with_kinds else ntt


def vertical_prepared_bytes(convs, compact=False):
    """bytes of the vertical engine's prepared lists and thresholds: symbex_bool_vertical_prepared_bytes (compact:
    symbex_bool_vertical_prepared_compact_bytes), the same sums (convs in the network's order: stem, then every
    block's conv and shortcut)"""
    r16 = lambda b: (b + 15) & ~15
    total = r16(12 * len(convs))
    for c in convs:
        ntt, nk = _vertical_tile_types(c, with_kinds=True)
        if compact:                          # header, kind masks per tile type, their kinds, the values per channel
            total += 16 + ntt * nk * 16 + r16(ntt * 10) + r16(c["cout"] * nk * 2)
        else:
            total += c["cout"] * ntt * 12 * 16
        ones = [sum(bin(int(w)).count("1") for w in row) for row in np.asarray(c["rows"]).reshape(c["cout"], -1)]
        total += r16(sum(2 * (2 + (n + 7) // 8 * 8) for n in ones))
        total += r16(1 + (c["oh"] * c["oh"] // 32 + 3) // 4)     # NTT and the type of every tile
    return total


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
    v_why = vertical_problem(convs, net["levels"], in_size)
    v_bytes = None if v_why else vertical_work_bytes(convs, blocks, net["levels"], in_size, fc_rw)
    v_ram = None if v_why else v_bytes + 4 * (4 * 4 + 4 + 2 * max_out) + (fh * fh if cells else 0)
    v_prep = None if v_why else vertical_prepared_bytes(convs)
    v_prep_c = None if v_why else vertical_prepared_bytes(convs, compact=True)
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
        # -DSYMBEX_BOOL_VERTICAL: the vertical engine works in its own block (B_VWORK); the standard engine's maps and
        # window shrink to a placeholder (symbex_net still points at them)
        f.write("#if defined(SYMBEX_BOOL_VERTICAL)\n")
        if v_why:
            f.write(f"#error \"the vertical engine does not run this network: {v_why}\"\n")
        f.write("#define SYMBEX_BOOL_MAPW(n) 4\n#else\n#define SYMBEX_BOOL_MAPW(n) (n)\n#endif\n")
        for i in range(4):
            f.write(f"static uint32_t B_MAP{i}[SYMBEX_BOOL_MAPW({size[i]})] __attribute__((aligned(16)));\n")
        f.write(f"static uint32_t B_WINDOW[SYMBEX_BOOL_MAPW({max_rw})] __attribute__((aligned(16)));\n"
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
                "static const char* const symbex_mlp_steps[SYMBEX_MLP_N_STEPS] = {\n"
                + "".join(f"    \"{s}\",\n" for s in steps) + "};\n")
        if not v_why:
            f.write(f"#if defined(SYMBEX_BOOL_VERTICAL)   /* the vertical engine: same data, same results */\n"
                    f"#include \"layers/conv/symbex_bool_vertical.h\"\n"
                    f"static uint32_t B_VWORK[{v_bytes // 4}] __attribute__((aligned(16)));\n"
                    f"#if defined(SYMBEX_BOOL_VERTICAL_PREPARED_COMPACT)   /* optional: the lists built once into RAM, the\n"
                    f"   thresholds compact (symbex_bool_vertical_prepare_compact): {v_prep_c / 1024:.1f} KB more RAM, faster */\n"
                    f"#define SYMBEX_MLP_BYTES_BUFFERS {v_ram + v_prep_c}u\n"
                    f"static uint32_t B_VPREP[{v_prep_c // 4}] __attribute__((aligned(16)));\n"
                    f"#elif defined(SYMBEX_BOOL_VERTICAL_PREPARED)   /* optional: the lists and thresholds built once into RAM\n"
                    f"   (symbex_bool_vertical_prepare): {v_prep / 1024:.1f} KB more RAM, faster still */\n"
                    f"#define SYMBEX_MLP_BYTES_BUFFERS {v_ram + v_prep}u\n"
                    f"static uint32_t B_VPREP[{v_prep // 4}] __attribute__((aligned(16)));\n"
                    f"#else\n#define SYMBEX_MLP_BYTES_BUFFERS {v_ram}u\n#endif\n"
                    "static struct SymbexBoolVertical B_V;\nstatic int B_V_READY;\n"
                    "static inline int symbex_mlp_infer(const uint8_t* x) {\n"
                    "    if (!B_V_READY) {                        /* once: carve the work block, build the list table */\n"
                    "        if (symbex_bool_vertical_init(&symbex_net, B_VWORK, sizeof B_VWORK, &B_V)) return -1;\n"
                    "#if defined(SYMBEX_BOOL_VERTICAL_PREPARED_COMPACT)\n"
                    "        if (symbex_bool_vertical_prepare_compact(&B_V, B_VPREP, sizeof B_VPREP)) return -1;\n"
                    "#elif defined(SYMBEX_BOOL_VERTICAL_PREPARED)\n"
                    "        if (symbex_bool_vertical_prepare(&B_V, B_VPREP, sizeof B_VPREP)) return -1;\n"
                    "#endif\n"
                    "        B_V_READY = 1;\n    }\n"
                    "    return symbex_bool_net_run_vertical(&symbex_net, &B_V, x);\n}\n#else\n")
        f.write(f"#define SYMBEX_MLP_BYTES_BUFFERS {ram}u\n"
                "static inline int symbex_mlp_infer(const uint8_t* x) { return symbex_bool_net_run(&symbex_net, x); }\n"
                + ("" if v_why else "#endif\n") + f"#endif /* {guard} */\n")
    for c in convs:
        shapes.append(dict(cin=c["cin"], cout=c["cout"], k=c["k"], stride=c["s"], h=c["h"], out_h=c["oh"],
                           bit_ops=c["cout"] * c["oh"] * c["oh"] * c["k"] * c["k"] * c["cin"]))
    return flash, ram, steps, shapes


def validate_c(header_path, lib_path, x_u8, expected, vertical=False, prepared=False):
    """gcc on this PC: the header + lib_symbex (symbex_bool.c, the generic HAL) on x_u8; (equal, total, message).
    vertical=True: built with -DSYMBEX_BOOL_VERTICAL (the vertical engine, in its portable C form); prepared=True:
    also -DSYMBEX_BOOL_VERTICAL_PREPARED (its lists and thresholds prepared in RAM); prepared="compact":
    -DSYMBEX_BOOL_VERTICAL_PREPARED_COMPACT (the lists prepared, the thresholds compact)"""
    pflag = ("-DSYMBEX_BOOL_VERTICAL_PREPARED_COMPACT" if prepared == "compact" else "-DSYMBEX_BOOL_VERTICAL_PREPARED")
    from .dense import _TEST_MAIN
    if shutil.which("gcc") is None:
        return None, 0, "gcc not found: the C exactness check was skipped"
    inc = os.path.join(lib_path, "include")
    sources = [os.path.join(lib_path, "src", "layers", "conv", "symbex_bool.c"),
               os.path.join(lib_path, "src", "hal", "generic", "symbex_hal_impl.c")]
    if vertical:
        sources.append(os.path.join(lib_path, "src", "layers", "conv", "symbex_bool_vertical.c"))
    for r in [inc] + sources:
        if not os.path.exists(r):
            return None, 0, f"lib_symbex not found at {lib_path} ({r} missing): set [output] lib_path"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(header_path, os.path.join(d, "model_symbex.h"))
        open(os.path.join(d, "main.c"), "w").write(_TEST_MAIN)
        exe = os.path.join(d, "check")
        r = subprocess.run(["gcc", "-O2", "-std=gnu11", "-I", inc, "-I", d, *(["-DSYMBEX_BOOL_VERTICAL"] if vertical else []),
                            *([pflag] if vertical and prepared else []),
                            "-o", exe, os.path.join(d, "main.c")] + sources, capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C did not compile:\n" + r.stderr[-2000:]
        data = os.path.join(d, "x.u8")
        np.ascontiguousarray(x_u8, dtype=np.uint8).tofile(data)
        r = subprocess.run([exe, data, str(len(x_u8))], capture_output=True, text=True)
        if r.returncode != 0:
            return None, 0, "the C program failed while running"
        pred_c = np.array([int(v) for v in r.stdout.split()])
        if vertical:                             # the header's buffers must be the sizes the library asks for, exactly
            open(os.path.join(d, "sizes.c"), "w").write(
                "#include <stdio.h>\n#include \"model_symbex.h\"\nint main(void) {\n"
                "    struct SymbexBoolVertical v;\n"
                "    if (symbex_bool_vertical_init(&symbex_net, B_VWORK, sizeof B_VWORK, &v)) return 1;\n"
                "    printf(\"%zu %zu\\n\", symbex_bool_vertical_work_bytes(&symbex_net), sizeof B_VWORK);\n"
                "#if defined(SYMBEX_BOOL_VERTICAL_PREPARED_COMPACT)\n"
                "    printf(\"%zu %zu\\n\", symbex_bool_vertical_prepared_compact_bytes(&v), sizeof B_VPREP);\n"
                "#elif defined(SYMBEX_BOOL_VERTICAL_PREPARED)\n"
                "    printf(\"%zu %zu\\n\", symbex_bool_vertical_prepared_bytes(&v), sizeof B_VPREP);\n"
                "#endif\n    return 0;\n}\n")
            r = subprocess.run(["gcc", "-O2", "-std=gnu11", "-I", inc, "-I", d, "-DSYMBEX_BOOL_VERTICAL",
                                *([pflag] if prepared else []),
                                "-o", exe + "_sizes", os.path.join(d, "sizes.c")] + sources, capture_output=True, text=True)
            r = subprocess.run([exe + "_sizes"], capture_output=True, text=True) if r.returncode == 0 else r
            pairs = [l.split() for l in r.stdout.strip().splitlines()] if r.returncode == 0 else []
            if not pairs or any(a != b for a, b in pairs):
                return None, 0, ("[X] the header's vertical buffers differ from the library's sizes (library, header): "
                                 f"{pairs or r.stderr[-500:]}")
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
    with open(header) as f:
        has_vertical = "symbex_bool_net_run_vertical" in f.read()
    v_equal, v_total, v_msg = validate_c(header, lib_path, x_u8, p_sim, vertical=True) if has_vertical else (None, 0, "")
    p_equal, p_total, p_msg = (validate_c(header, lib_path, x_u8, p_sim, vertical=True, prepared=True) if has_vertical
                               else (None, 0, ""))
    c_equal, c_total, c_msg = (validate_c(header, lib_path, x_u8, p_sim, vertical=True, prepared="compact")
                               if has_vertical else (None, 0, ""))
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
    if not has_vertical:
        print("    vertical    : not available for this network (see the header's #error)")
    elif v_equal is None:
        print(f"    vertical    : not checked ({v_msg})")
    else:
        print(f"    vertical    : {100 * v_equal / v_total:.2f}%  (-DSYMBEX_BOOL_VERTICAL, the vertical engine, vs. the "
              f"integer simulation) " + ("OK" if v_equal == v_total else "[X]"))
    if has_vertical and p_equal is None:
        print(f"    prepared    : not checked ({p_msg})")
    elif has_vertical:
        print(f"    prepared    : {100 * p_equal / p_total:.2f}%  (+ -DSYMBEX_BOOL_VERTICAL_PREPARED, lists and thresholds "
              f"prepared in RAM, vs. the integer simulation) " + ("OK" if p_equal == p_total else "[X]"))
    if has_vertical and c_equal is None:
        print(f"    compact     : not checked ({c_msg})")
    elif has_vertical:
        print(f"    compact     : {100 * c_equal / c_total:.2f}%  (+ -DSYMBEX_BOOL_VERTICAL_PREPARED_COMPACT, lists prepared, "
              f"thresholds compact, vs. the integer simulation) " + ("OK" if c_equal == c_total else "[X]"))
    return dict(method="boolean", header=header, flash_bytes=flash, ram_buffers=ram, steps=steps, layers=shapes,
                accuracy_torch=acc_t, accuracy_sim=acc_s, fidelity=fid,
                exactness=None if equal is None else 100 * equal / total, arithmetic="and (HAL rows)",
                exactness_vertical=None if v_equal is None else 100 * v_equal / v_total,
                exactness_vertical_prepared=None if p_equal is None else 100 * p_equal / p_total,
                exactness_vertical_compact=None if c_equal is None else 100 * c_equal / c_total,
                _x_u8=x_u8.reshape(len(x_u8), -1), _y=y, _pred_sim=p_sim)
