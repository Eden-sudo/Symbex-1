"""
boolean.py -- the CDT's "boolean" method: fully binary convolutional networks (lib_symbex layers/conv/symbex_bool).

The network is trained AS DEFINED (it is already binary: no plan search, no widening), then folded into integers and
exported to the library. The model (e.g. benchmarks/conv2d/cnn_cifar_resnet8/model.py, BoolResNet8) must expose:
    levels                    thermometer levels per color of its input encoding
    stem, stem_bn             binary 3x3 conv + BatchNorm (then sign)
    stem_kind (optional)      "thermometer" (the default) or "int8": then stem is an 8-bit conv on the centered
                              pixels (pixel - 128) / 128 and stem.int8_weights() gives its integers q (O, 3, k, k) in
                              -127..127 and per-channel scales s (O,), weight = q * s (model.py's Int8Conv). The chip
                              runs it as lib_symbex's optional int8 first layer (symbex_bool_stem_i8): one integer
                              dot product per output and the BatchNorm + sign as one threshold on it. More accurate
                              (ResNet-8: +4.5 points), faster on chips with int8 SIMD (ESP32-S3, Cortex-M33), slower
                              on the others (ESP32, Hazard3): benchmarks/experiments/chip_kernels/int8_first_bench.py. The vertical
                              engine runs the thermometer stem only (for now).
    blocks                    Blocks: conv + bn (stride 1 or 2), and, where the shape changes, ds_conv (1x1) + ds_bn
                              after an OR pool when the block has stride 2; out = sign(BN(conv)) OR shortcut
    fc, fc_bn                 binary linear on the flattened final bits (CHW order) + BatchNorm1d
Every conv computes raw = sum of +-1 products over the in-image window; BatchNorm + sign fold into raw >= threshold
(weights negated where the BatchNorm scale is negative). The classifier keeps an int32 scale and bias per class
(2^16 fixed point) and the C picks the first class with the largest int64 logit.

    train / binary_weights / load_trained              the training: train/boolean.py (still importable from here)
    fold(model) / simulate(net, x_u8)                  the integer network, and its exact simulation
    export_and_validate(model, val_ds, out_dir, ...)   header + validation files + the C check on this PC (gcc)
The header offers the interface of the MLP headers (symbex_mlp_infer, SYMBEX_MLP_N_INPUTS / N_CLASSES / N_STEPS,
symbex_mlp_steps, SYMBEX_MLP_BYTES_BUFFERS), so firmware/model_test.c and the chip stage measure it unchanged.
Built with -DSYMBEX_BOOL_VERTICAL, the same header runs the vertical engine (symbex_bool_vertical.h, the fastest on
the ESP32-S3; docs/vertical.md) on the same data, with the same results: chip_test.py --engine vertical.
Adding -DSYMBEX_BOOL_VERTICAL_PREPARED (optional) builds its plane lists and thresholds once into RAM: more RAM,
faster (chip_test.py --engine vertical_prepared). -DSYMBEX_BOOL_VERTICAL_PREPARED_COMPACT instead: the lists prepared,
the thresholds kept compact (one value per output channel and border kind): about half that RAM, a little slower
(chip_test.py --engine vertical_compact). The engine's buffer sizes and limits: export/vertical.py.
"""
import math
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from . import c_build, c_header
from .vertical import vertical_prepared_bytes, vertical_problem, vertical_work_bytes
from ..train.boolean import _augment, _evaluate, binary_weights, load_trained, train   # noqa: F401 (the old home)

_words, _c_array = c_header.words, c_header.c_array   # the names these helpers had here before c_header.py

INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1
FC_SHIFT = 16


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


def _fold_stem_i8(conv, bn):
    """int8 stem + BatchNorm + sign -> int8 weights (O, 3, k, k) (rows negated where the BatchNorm scale < 0) and an
    integer threshold on D = sum of (pixel - 128) * q (zero padding = pixel 128): the float conv is D * s / 128, so
    BN(D * s / 128) >= 0  <=>  D >= ceil(T * 128 / s)"""
    _check_conv(conv, "int8 stem")
    k = conv.kernel_size[0]
    problems = []
    if conv.in_channels != 3:
        problems.append(f"{conv.in_channels} input channels (the int8 stem reads the 3 colors)")
    if k % 2 == 0 or k > 3:
        problems.append(f"kernel {k}: the int8 stem runs 1x1 or 3x3 (a window of at most 27 bytes)")
    if not hasattr(conv, "int8_weights"):
        problems.append("the stem has no int8_weights() (q, s): see model.py's Int8Conv")
    if bn.running_mean is None or not bn.affine:
        problems.append("its BatchNorm needs running statistics and affine parameters")
    if problems:
        raise ValueError("boolean export, int8 stem: " + "; ".join(problems))
    q, s = conv.int8_weights()
    q, s = q.detach().double().clone(), s.detach().double().flatten()
    if q.abs().max() > 127 or not torch.equal(q, torch.round(q)) or (s <= 0).any():
        raise ValueError("boolean export, int8 stem: int8_weights() must give integers in -127..127 and scales > 0")
    g = bn.weight.detach().double()
    sd = torch.sqrt(bn.running_var.detach().double() + bn.eps)
    t = bn.running_mean.detach().double() - bn.bias.detach().double() * sd / torch.where(g == 0, torch.ones_like(g), g)
    thr = np.empty(len(g), dtype=np.int64)
    for o in range(len(g)):
        u = float(t[o]) * 128 / float(s[o])
        if g[o] == 0:
            thr[o] = INT_MIN if bn.bias[o] >= 0 else INT_MAX
        elif g[o] > 0:                                       # D >= U
            thr[o] = math.ceil(u)
        else:                                                # D <= U  <=>  (-D) >= -U
            q[o] = -q[o]
            thr[o] = math.ceil(-u)
    return dict(w=q.numpy(), thr=np.clip(thr, INT_MIN, INT_MAX), k=k, stride=conv.stride[0], int8=True)


def fold(model, threshold=0.0):
    """the trained BoolResNet8-like model -> integer network (dict of numpy arrays). A model with head / head_bn (a
    per-cell detector, FOMO-style) gets a per-cell head; threshold = the calibrated decision threshold on
    (logit of class 1 - logit of class 0), folded into the integer bias of class 1"""
    model = model.cpu().eval()
    kind = getattr(model, "stem_kind", "thermometer")
    if kind not in ("thermometer", "int8"):
        raise ValueError(f"boolean export: stem_kind {kind!r} (thermometer or int8)")
    if any(getattr(b, "add", False) for b in model.blocks):
        raise ValueError("boolean export: added shortcuts (shortcut=\"add\") are not in lib_symbex; use OR shortcuts")
    stem = (_fold_stem_i8(model.stem, model.stem_bn) if kind == "int8"
            else _fold_conv(model.stem, model.stem_bn, "stem"))
    net = dict(levels=int(model.levels), stem=stem, blocks=[])
    if kind == "thermometer" and model.stem.in_channels != 3 * net["levels"]:
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
        if net["stem"].get("int8"):
            x = layer(x - 128, net["stem"])                                              # int8 stem: D >= thr
        else:
            x = (x.unsqueeze(2) >= t.view(1, 1, -1, 1, 1)).flatten(1, 2).double() * 2 - 1  # thermometer, +-1
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
    rows[:, :n] = c_header.words(hwc.reshape(o, k * k * cin), n)
    return rows, rw, hwc.reshape(o, k * k, cin).sum(2)


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

    i8 = bool(net["stem"].get("int8"))
    if i8:      # the int8 stem: rows of k * k * 3 int8 [kh][kw][color], zero padded to a multiple of 4 bytes
        st = net["stem"]
        k, s = st["k"], st["stride"]
        q = st["w"].astype(np.int64).transpose(0, 2, 3, 1).reshape(len(st["w"]), -1)
        rb = -(-q.shape[1] // 4) * 4
        i8_rows = np.zeros((len(q), rb), dtype=np.int64)
        i8_rows[:, :q.shape[1]] = q
        out_h = (res + 2 * (k // 2) - k) // s + 1
        convs.append(dict(rows=None, rw=0, cnt=None, thr=st["thr"], cin=3, cout=len(q), k=k, s=s, h=res, oh=out_h,
                          int8=True, row_bytes=rb, i8_rows=i8_rows))
        flash += i8_rows.size + len(st["thr"]) * 4
        stem_i, res = 0, out_h
        steps.append(f"int8 stem {3}->{len(q)} {in_size}x{in_size}{' s2' if s == 2 else ''}")
    else:
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
        fc_rows[:, :ppx] = c_header.words(fcw, ppx)
        fc_bits = ch
    else:                                                                                # (classes, C*H*W), CHW
        fc_words = fh * fh * ppx
        fc_rw = -(-fc_words // 4) * 4
        hwc = fcw.reshape(n_cls, ch, fh, fh).transpose(0, 2, 3, 1).reshape(n_cls * fh * fh, ch)
        fc_rows = np.zeros((n_cls, fc_rw), dtype=np.uint32)
        fc_rows[:, :fc_words] = c_header.words(hwc, ppx).reshape(n_cls, fc_words)
        fc_bits = fh * fh * ch
    fc_cnt = fcw.sum(1)
    flash += fc_rows.size * 4 + n_cls * 12
    # every map buffer as large as what symbex_bool_net_run writes into IT (it rotates 4 buffers): far less RAM than
    # four buffers as large as the largest map (the thermometer input of a 96 x 96 image alone is 36 KB)
    words_of = lambda c, h, ch_: h * h * ((ch_ + 31) // 32)
    size = [0 if i8 else in_size * in_size * ((3 * net["levels"] + 31) // 32), 0, 0, 0]   # map 0: the thermometer
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
    max_rw = max(max(c["rw"] for c in convs), 8 if i8 else 4)     # the int8 stem's window: 32 bytes
    max_out = max(max(c["cout"] for c in convs), n_cls)
    ram = 4 * (sum(size) + max_rw + 2 * max_out) + (fh * fh if cells else 0)
    v_why = ("int8 stem (the vertical engine runs the thermometer stem only, for now)" if i8
             else vertical_problem(convs, net["levels"], in_size))
    v_bytes = None if v_why else vertical_work_bytes(convs, blocks, net["levels"], in_size, fc_rw)
    v_ram = None if v_why else v_bytes + 4 * (4 * 4 + 4 + 2 * max_out) + (fh * fh if cells else 0)
    v_prep = None if v_why else vertical_prepared_bytes(convs)
    v_prep_c = None if v_why else vertical_prepared_bytes(convs, compact=True)
    guard = "SYMBEX_BOOL_MODEL_H"
    with open(path, "w") as f:
        f.write(f"/* generated by the CDT (compiler_core/export/boolean.py): a fully binary network for lib_symbex's\n"
                f"   symbex_bool_net_run, with the interface of the MLP headers (symbex_mlp_infer). Do not edit. */\n"
                f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n#include \"layers/conv/symbex_bool.h\"\n"
                + c_header.prologue() +
                f"#ifndef SYMBEX_ROM\n#define SYMBEX_ROM const           /* -DSYMBEX_ROM= : weights in RAM */\n#endif\n\n")
        for i, c in enumerate(convs):
            if c.get("int8"):
                c_header.c_array(f, "int8_t", f"BI{i}", c["i8_rows"], " __attribute__((aligned(16)))")
                c_header.c_array(f, "int32_t", f"BT{i}", c["thr"])
                f.write("static const struct SymbexBoolStemI8 B_STEM_I8 = "
                        + c_header.struct(out_ch=c['cout'], kernel=c['k'], stride=c['s'], in_h=c['h'], in_w=c['h'],
                                          out_h=c['oh'], out_w=c['oh'], row_bytes=c['row_bytes'], weights=f"BI{i}",
                                          thresholds=f"BT{i}") + ";   /* the int8 stem */\n")
                continue
            c_header.c_array(f, "uint32_t", f"BW{i}", c["rows"], " __attribute__((aligned(16)))")
            c_header.c_array(f, "int32_t", f"BT{i}", c["thr"])
            c_header.c_array(f, "int16_t", f"BC{i}", c["cnt"])

        def conv_init(i):
            c = convs[i]
            shapes = dict(in_ch=c['cin'], out_ch=c['cout'], kernel=c['k'], stride=c['s'], in_h=c['h'], in_w=c['h'],
                          out_h=c['oh'], out_w=c['oh'])
            if c.get("int8"):                        # only the shapes: the network runs B_STEM_I8 (stem_i8)
                return c_header.struct(**shapes)
            return c_header.struct(**shapes, row_words=c['rw'], weights=f"BW{i}", thresholds=f"BT{i}",
                                   weight_counts=f"BC{i}")
        for bi, (ci, si, pool) in enumerate(blocks):
            if si is not None:
                f.write(f"static const struct SymbexBoolConv BS{bi} = {conv_init(si)};\n")
        f.write("static const struct SymbexBoolBlock B_BLOCKS[] = {\n")
        for bi, (ci, si, pool) in enumerate(blocks):
            f.write("    " + c_header.struct(conv=conv_init(ci), shortcut='&BS' + str(bi) if si is not None else 'NULL',
                                             shortcut_pool=1 if pool else 0) + ",\n")
        f.write("};\n")
        c_header.c_array(f, "uint32_t", "FC_W", fc_rows, " __attribute__((aligned(16)))")
        c_header.c_array(f, "int32_t", "FC_C", fc_cnt)
        c_header.c_array(f, "int32_t", "FC_S", net["fc_scale"])
        c_header.c_array(f, "int32_t", "FC_B", net["fc_bias"])
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
        extra = (dict(cells="B_CELLS", cells_h=fh, cells_w=fh) if cells else {}) | (dict(stem_i8="&B_STEM_I8") if i8 else {})
        f.write("static const struct SymbexBoolNet symbex_net = "
                + c_header.struct(in_h=in_size, in_w=in_size, levels=net['levels'], stem=conv_init(stem_i),
                                  blocks="B_BLOCKS", n_blocks=len(blocks), fc_weights="FC_W", fc_row_words=fc_rw,
                                  n_classes=n_cls, fc_bits=fc_bits, fc_words=fc_words, fc_weight_counts="FC_C",
                                  fc_scale="FC_S", fc_bias="FC_B", maps="{B_MAP0, B_MAP1, B_MAP2, B_MAP3}",
                                  window="B_WINDOW", scratch="B_SCRATCH", **extra)
                + (";   /* per-cell head: returns the grid checksum */\n\n" if cells else
                   ";   /* no per-cell head: a classifier */\n\n"))
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
                + ("" if v_why else "#endif\n") + c_header.epilogue() + f"#endif /* {guard} */\n")
    for c in convs:
        ops = c["cout"] * c["oh"] * c["oh"] * c["k"] * c["k"] * c["cin"]
        shapes.append(dict(cin=c["cin"], cout=c["cout"], k=c["k"], stride=c["s"], h=c["h"], out_h=c["oh"],
                           **(dict(int8_macs=ops) if c.get("int8") else dict(bit_ops=ops))))
    return flash, ram, steps, shapes


def validate_c(header_path, lib_path, x_u8, expected, vertical=False, prepared=False):
    """gcc on this PC: the header + lib_symbex (symbex_bool.c, the generic HAL) on x_u8; (equal, total, message).
    vertical=True: built with -DSYMBEX_BOOL_VERTICAL (the vertical engine, in its portable C form); prepared=True:
    also -DSYMBEX_BOOL_VERTICAL_PREPARED (its lists and thresholds prepared in RAM); prepared="compact":
    -DSYMBEX_BOOL_VERTICAL_PREPARED_COMPACT (the lists prepared, the thresholds compact)"""
    pflag = ("-DSYMBEX_BOOL_VERTICAL_PREPARED_COMPACT" if prepared == "compact" else "-DSYMBEX_BOOL_VERTICAL_PREPARED")
    sources = ["layers/conv/symbex_bool.c", "hal/generic/symbex_hal_impl.c"]
    flags = []
    if vertical:
        sources.append("layers/conv/symbex_bool_vertical.c")
        flags = ["-DSYMBEX_BOOL_VERTICAL"] + ([pflag] if prepared else [])

    def sizes_match(build, d):                   # the header's buffers must be the sizes the library asks for, exactly
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
        err = build("sizes.c", "check_sizes")
        r = subprocess.run([os.path.join(d, "check_sizes")], capture_output=True, text=True) if err is None else None
        pairs = [l.split() for l in r.stdout.strip().splitlines()] if r is not None and r.returncode == 0 else []
        if not pairs or any(a != b for a, b in pairs):
            return ("[X] the header's vertical buffers differ from the library's sizes (library, header): "
                    f"{pairs or (err or '')[-500:]}")
        return ""

    return c_build.check(header_path, lib_path, x_u8, expected, sources, flags=flags,
                         after=sizes_match if vertical else None)


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
    if net["stem"].get("int8"):
        print(f"    network     : int8 stem (pixel - 128, int8 weights) -> {len(net['blocks'])} blocks (conv OR shortcut) "
              f"-> classifier | every layer after the stem 1 bit")
    else:
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
