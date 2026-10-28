"""
bool1d.py -- binary 1D convolutional networks (time series: sensors, audio features) into lib_symbex's symbex_bool1d.

The model (e.g. benchmarks/conv1d/har_ultracompact/model.py, BinaryHAR) must expose:
    plan                      the layers in order: ("conv", i) -> convs[i] + bns[i] + sign, ("pool", p) -> an OR pool of p
                              steps (max of +-1), which must follow a conv (it is fused into it on the chip)
    convs, bns                Conv1d (no padding, no dilation, no groups, no bias) + BatchNorm1d
    first                     "binary": the first conv has 1-bit weights on the 8-bit input (lib_symbex's uint8 layer);
                              "int8": int8 weights, given by model.first_weights() as integers (folded and simulated
                              here, but lib_symbex does not run it)
    fc, fc_bn                 binary linear on the flattened final bits (PyTorch order: channel-major) + BatchNorm1d
The input is int8 (N, length, channels), time-major; the first conv sees x / 127 in PyTorch (only a scale: folded here).
On the chip the input is uint8 = int8 + 128; the offset folds into the first thresholds.

    fold(model)                                the integer network (dict of numpy arrays)
    simulate(net, x_int8)                      its exact simulation: the classes (first largest logit)
    write_header(net, path)                    model_symbex.h, with the interface of the MLP headers (symbex_mlp_infer)
    validate_c(header, lib_path, x_u8, expected)   gcc on this PC: header + lib_symbex == simulation
"""
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from .boolean import FC_SHIFT, INT_MAX, INT_MIN, _c_array, _words

INPUT_SCALE = 127                                 # the model divides the int8 input by this before its first conv


def _check_conv(conv, where):
    problems = []
    if conv.padding not in (0, (0,)) or tuple(conv.dilation) != (1,) or conv.groups != 1:
        problems.append(f"padding {conv.padding} / dilation {tuple(conv.dilation)} / groups {conv.groups}: "
                        "lib_symbex runs no padding, no dilation, no groups")
    if conv.bias is not None:
        problems.append("the convolution has a bias: use bias=False (the BatchNorm after it carries the offset)")
    if problems:
        raise ValueError(f"bool1d export, {where}: " + "; ".join(problems))


def _fold_conv(conv, bn, w_int, in_scale, where):
    """integer weights (O, C, k) + BatchNorm + sign -> weights with rows negated where the BN scale < 0, and integer
    thresholds on raw = sum of input x weight: bit = raw >= threshold. BN sees raw * in_scale."""
    _check_conv(conv, where)
    w = w_int.detach().double().clone()
    g = bn.weight.detach().double()
    sd = torch.sqrt(bn.running_var.detach().double() + bn.eps)
    t = (bn.running_mean.detach().double() - bn.bias.detach().double() * sd / torch.where(g == 0, torch.ones_like(g), g)) / in_scale
    thr = np.empty(len(g), dtype=np.int64)
    for o in range(len(g)):
        if g[o] == 0:
            thr[o] = INT_MIN if bn.bias[o] >= 0 else INT_MAX
        elif g[o] > 0:                                       # BN(raw) >= 0  <=>  raw >= T
            thr[o] = math.ceil(float(t[o]))
        else:                                                # raw <= T  <=>  (-raw) >= -T
            w[o] = -w[o]
            thr[o] = math.ceil(-float(t[o]))
    return dict(w=w.numpy(), thr=np.clip(thr, INT_MIN, INT_MAX), k=conv.kernel_size[0], stride=conv.stride[0], pool=1)


def fold(model):
    model = model.cpu().eval()
    layers = []
    for kind, i in model.plan:
        if kind == "pool":
            if not layers or layers[-1]["pool"] != 1:
                raise ValueError("bool1d export: an OR pool must follow a convolution (it is fused into it)")
            layers[-1]["pool"] = int(i)
            continue
        conv = model.convs[i]
        if i == 0 and model.first == "int8":
            w = torch.round(model.first_weights())
        else:
            w = torch.where(conv.weight.detach() >= 0, 1.0, -1.0)
        layers.append(_fold_conv(conv, model.bns[i], w, 1 / INPUT_SCALE if i == 0 else 1.0, f"conv {i}"))
    bn = model.fc_bn
    g = bn.weight.detach().double()
    sd = torch.sqrt(bn.running_var.detach().double() + bn.eps)
    a = (g / sd).numpy()
    c = (bn.bias.detach().double() - bn.running_mean.detach().double() * g / sd).numpy()
    for name, v in (("scale", a), ("bias", c)):
        if np.abs(v).max() * 2 ** FC_SHIFT >= 2 ** 31:
            raise ValueError(f"bool1d export: the classifier {name} does not fit in int32 with {FC_SHIFT} fraction bits")
    if model.fc.bias is not None:
        raise ValueError("bool1d export, classifier: it has a bias; use bias=False (its BatchNorm carries the offset)")
    cout = layers[-1]["w"].shape[0]
    w = torch.where(model.fc.weight.detach() >= 0, 1.0, -1.0).double().numpy()        # (classes, C * L), channel-major
    length = w.shape[1] // cout
    return dict(first=model.first, layers=layers, length=length,
                fc_w=w.reshape(len(w), cout, length).transpose(0, 2, 1).reshape(len(w), -1),  # time-major: t * C + c
                fc_scale=np.round(a * 2 ** FC_SHIFT).astype(np.int64), fc_bias=np.round(c * 2 ** FC_SHIFT).astype(np.int64))


@torch.no_grad()
def simulate(net, x_int8, batch=1000, return_bits=False):
    """the integer network on int8 windows (N, length, channels): the classes (first largest logit), exactly as the C
    computes them (float64 holds every integer here exactly). return_bits: also the final bits (N, L, C) in 0/1."""
    preds, finals = [], []
    for i in range(0, len(x_int8), batch):
        x = torch.as_tensor(np.asarray(x_int8[i:i + batch], dtype=np.float64)).transpose(1, 2)    # N x C x L
        for c in net["layers"]:
            raw = F.conv1d(x, torch.as_tensor(c["w"]), stride=c["stride"])
            x = torch.where(raw >= torch.as_tensor(c["thr"], dtype=torch.float64).view(1, -1, 1), 1.0, -1.0).double()
            if c["pool"] > 1:
                x = F.max_pool1d(x, c["pool"])                                                  # OR of bits
        feats = x.transpose(1, 2).flatten(1)                                                    # time-major
        raw = (feats @ torch.as_tensor(net["fc_w"]).T).numpy().astype(np.int64)
        logit = raw * net["fc_scale"][None] + net["fc_bias"][None]
        preds.append(np.argmax(logit, axis=1))
        if return_bits:
            finals.append((x.transpose(1, 2).numpy() > 0).astype(np.uint8))
    return (np.concatenate(preds), np.concatenate(finals)) if return_bits else np.concatenate(preds)


def _rows(w):
    """(O, C, k) +-1 -> rows of the time-major window (bit q * C + c = the weight of step q, channel c), padded to a
    multiple of 4 words; their counts of ones"""
    bits = (w > 0).transpose(0, 2, 1).reshape(len(w), -1)                                     # (O, k * C)
    n = -(-bits.shape[1] // 32)
    rw = -(-n // 4) * 4
    rows = np.zeros((len(w), rw), dtype=np.uint32)
    rows[:, :n] = _words(bits, n)
    return rows, rw, bits.sum(1)


def write_header(net, path, in_len, in_ch):
    """model_symbex.h: the layers as SymbexBool1D, the classifier of symbex_bool.h, and symbex_mlp_infer (uint8 input,
    time-major) calling them in order, with SYMBEX_MARK after every layer. Returns (flash bytes, RAM bytes of the
    buffers, step names, layer shapes)."""
    if net["first"] not in ("binary", "int8"):
        raise ValueError(f"bool1d export: unknown first layer {net['first']!r}")
    i8 = net["first"] == "int8"                         # first layer with int8 weights (SymbexBool1D.weights_i8)
    flash, steps, shapes, convs = 0, [], [], []
    length, ch = in_len, in_ch
    for i, c in enumerate(net["layers"]):
        w8 = None
        if i == 0 and i8:                                # rows of kernel * in_ch bytes, time-major like the bit rows
            w8 = c["w"].transpose(0, 2, 1).reshape(len(c["w"]), -1)
            if np.abs(w8).max() > 127 or np.any(w8 != np.round(w8)):
                raise ValueError("bool1d export: the int8 first-layer weights are not integers in [-127, 127]")
            w8 = w8.astype(np.int8)
            rows, rw, cnt = np.zeros((len(w8), 0), dtype=np.uint32), -(-(-(-w8.shape[1] // 32)) // 4) * 4, None
        else:
            rows, rw, cnt = _rows(c["w"])
        thr = c["thr"].copy()
        if i == 0:                                       # uint8 = int8 + 128: raw_u8 = raw + 128 * sum of the weights
            thr = np.clip(thr + 128 * c["w"].reshape(len(thr), -1).sum(1).astype(np.int64), INT_MIN, INT_MAX)
        out_len = ((length - c["k"]) // c["stride"] + 1) // c["pool"]
        convs.append(dict(rows=rows, rw=rw, cnt=cnt, thr=thr, cin=ch, cout=len(thr), k=c["k"], s=c["stride"],
                          p=c["pool"], h=length, oh=out_len, w8=w8))
        flash += rows.size * 4 + (w8.size if w8 is not None else 0) + len(thr) * 4 + (0 if i == 0 else len(thr) * 4)
        steps.append(f"conv {ch}->{len(thr)} k{c['k']} L{length}"
                     + ((" (uint8 input, int8 weights)" if i8 else " (uint8 input)") if i == 0 else "")
                     + (f" + OR pool {c['pool']}" if c["pool"] > 1 else ""))
        shapes.append(dict(cin=ch, cout=len(thr), k=c["k"], stride=c["stride"], pool=c["pool"], length=length,
                           out_length=out_len, bit_ops=len(thr) * out_len * c["pool"] * c["k"] * ch))
        length, ch = out_len, len(thr)
    if length != net["length"]:
        raise ValueError(f"bool1d export: the final map has {length} steps, the classifier expects {net['length']}")
    fcw = net["fc_w"] > 0
    n_cls, fc_bits = fcw.shape[0], fcw.shape[1]
    fc_words = -(-fc_bits // 32)
    fc_rw = -(-fc_words // 4) * 4
    fc_rows = np.zeros((n_cls, fc_rw), dtype=np.uint32)
    fc_rows[:, :fc_words] = _words(fcw, fc_words)
    flash += fc_rows.size * 4 + n_cls * 12
    steps.append("classifier")
    out_words = [-(-c["oh"] * c["cout"] // 32) + 1 for c in convs]
    pingpong = max([w for w in out_words[:-1]] + [4])
    final = max(out_words[-1], fc_rw)                   # the classifier reads whole rows: zero past the last map
    max_rw = max(c["rw"] for c in convs)
    u8_rw = convs[0]["rw"]
    max_out = max(max(c["cout"] for c in convs), n_cls)
    ram = 4 * (2 * pingpong + final + 2 * max_rw + max_out) + u8_rw * 32
    guard = "SYMBEX_BOOL1D_MODEL_H"
    with open(path, "w") as f:
        f.write(f"/* generated by the CDT (compiler_core/export/bool1d.py): a fully binary 1D network for lib_symbex's\n"
                f"   symbex_bool1d layers, with the interface of the MLP headers (symbex_mlp_infer). Do not edit. */\n"
                f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n#include \"layers/conv/symbex_bool.h\"\n"
                f"#include \"layers/conv/symbex_bool1d.h\"\n"
                f"#ifndef SYMBEX_ROM\n#define SYMBEX_ROM const           /* -DSYMBEX_ROM= : weights in RAM */\n#endif\n\n")
        for i, c in enumerate(convs):
            if c["w8"] is not None:
                _c_array(f, "int8_t", f"W{i}", c["w8"], " __attribute__((aligned(16)))")
            else:
                _c_array(f, "uint32_t", f"W{i}", c["rows"], " __attribute__((aligned(16)))")
            _c_array(f, "int32_t", f"T{i}", c["thr"])
            if i:
                _c_array(f, "int32_t", f"C{i}", c["cnt"])
            wb, w8 = ("NULL", f"W{i}") if c["w8"] is not None else (f"W{i}", "NULL")
            f.write(f"static const struct SymbexBool1D L{i} = {{{c['cin']}, {c['cout']}, {c['k']}, {c['s']}, {c['p']}, "
                    f"{c['h']}, {c['oh']}, {c['rw']}, {wb}, T{i}, {'NULL' if i == 0 else f'C{i}'}, {w8}}};\n")
        _c_array(f, "uint32_t", "FC_W", fc_rows, " __attribute__((aligned(16)))")
        _c_array(f, "int32_t", "FC_C", fcw.sum(1))
        _c_array(f, "int32_t", "FC_S", net["fc_scale"])
        _c_array(f, "int32_t", "FC_B", net["fc_bias"])
        f.write(f"static uint32_t M_A[{pingpong}] __attribute__((aligned(16)));\n"
                f"static uint32_t M_B[{pingpong}] __attribute__((aligned(16)));\n"
                f"static uint32_t M_F[{final}] __attribute__((aligned(16)));   /* the last map: only its layer writes it */\n"
                f"static uint32_t B_WINDOW[{2 * max_rw}] __attribute__((aligned(16)));\n"
                f"static uint32_t B_WORK[{u8_rw * 8}] __attribute__((aligned(16)));   /* the uint8 layer's workspace */\n"
                f"static int32_t B_SCRATCH[{max_out}];\n\n")
        f.write(f"#define SYMBEX_MLP_N_INPUTS {in_len * in_ch}      /* uint8 (int8 + 128), time-major: {in_len} x {in_ch} */\n"
                f"#define SYMBEX_MLP_N_CLASSES {n_cls}\n#define SYMBEX_MLP_N_STEPS {len(steps)}\n"
                f"#define SYMBEX_MLP_BYTES_BUFFERS {ram}u\n"
                "static const char* const symbex_mlp_steps[SYMBEX_MLP_N_STEPS] = {\n"
                + "".join(f"    \"{s}\",\n" for s in steps) + "};\n")
        f.write("static inline int symbex_mlp_infer(const uint8_t* x) {\n    SYMBEX_MARK(0);\n")
        n = len(convs)
        for i in range(n):
            dst = "M_F" if i == n - 1 else ("M_A" if i % 2 == 0 else "M_B")
            src = "M_A" if (i - 1) % 2 == 0 else "M_B"
            if i == 0:
                f.write(f"    symbex_bool1d_conv_u8(&L0, x, {dst}, (uint8_t*)B_WORK, B_SCRATCH);\n")
            else:
                f.write(f"    symbex_bool1d_conv(&L{i}, {src}, {dst}, B_WINDOW, B_SCRATCH);\n")
            f.write(f"    SYMBEX_MARK({i + 1});\n")
        f.write(f"    const int c = symbex_bool_classify(M_F, {fc_words}, {fc_bits}, FC_W, {fc_rw}, {n_cls}, FC_C, FC_S, "
                f"FC_B, B_SCRATCH);\n    SYMBEX_MARK({n + 1});\n    return c;\n}}\n#endif /* {guard} */\n")
    return flash, ram, steps, shapes


def validate_c(header_path, lib_path, x_u8, expected, native_popcount=False, extra_sources=(), defines=()):
    """gcc on this PC: the header + lib_symbex (symbex_bool1d.c, symbex_bool.c, the generic HAL) on x_u8 (N, inputs).
    native_popcount: -mpopcnt, so the 1D layers take their inline two-window path. extra_sources / defines: more C
    files and -D flags (e.g. an application's own first-layer kernels with -DSYMBEX_U8_EXTERNAL).
    Returns (equal, total, message)."""
    from .dense import _TEST_MAIN
    if shutil.which("gcc") is None:
        return None, 0, "gcc not found: the C exactness check was skipped"
    inc = os.path.join(lib_path, "include")
    sources = [os.path.join(lib_path, "src", "layers", "conv", f) for f in ("symbex_bool1d.c", "symbex_bool.c")]
    sources.append(os.path.join(lib_path, "src", "hal", "generic", "symbex_hal_impl.c"))
    sources += list(extra_sources)
    for r in [inc] + sources:
        if not os.path.exists(r):
            return None, 0, f"lib_symbex not found at {lib_path} ({r} missing)"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(header_path, os.path.join(d, "model_symbex.h"))
        open(os.path.join(d, "main.c"), "w").write(_TEST_MAIN)
        exe = os.path.join(d, "check")
        r = subprocess.run(["gcc", "-O2", "-std=gnu11"] + (["-mpopcnt"] if native_popcount else [])
                           + [f"-D{d}" for d in defines] + ["-I", inc, "-I", d, "-o", exe, os.path.join(d, "main.c")] + sources,
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
