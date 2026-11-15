"""
acts1d.py -- har_ultracompact benchmark (README section 10): the two-bit / ternary networks of README section 9
(model.BinaryHAR with acts "2" or "ternary", int8 first layer) into integers and into C on lib_symbex's HAL
(planes1d.h, embedded in the header). An experiment: it lives here, not in the CDT, until it is proven. har_export.py
calls it for those runs.

    fold(model)                                     the integer network (dict of numpy arrays)
    simulate(net, x_int8)                           its exact simulation: the classes (first largest logit)
    write_header(net, path, in_len, in_ch)          model_symbex.h with symbex_mlp_infer (the interface every chip
                                                    test reads)
    validate_c(header, lib_path, x_u8, expected)    gcc on this PC: header + lib_symbex HAL == simulation

Integers: a layer's raw is the sum of input x weight on integer inputs: int8 for the first layer (x / 127 in PyTorch),
3v (-3, -1, 1, 3) after a two-bit layer, v (-1, 0, 1) after a ternary one. The level of an output = the number of its
ascending thresholds raw reaches; the thresholds are where the BatchNorm output crosses the level boundaries of
model._Levels (2 bits: -2/3, 0, 2/3; ternary: -1/2, 1/2), with the row negated where the BN scale is negative (as
bool1d.py does for sign). Pooling: max of the levels.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools")))
from compiler_core.export import c_build, c_header                                          # noqa: E402
from compiler_core.export.bool1d import INPUT_SCALE, _check_conv, _rows                     # noqa: E402
from compiler_core.export.boolean import FC_SHIFT, INT_MAX, INT_MIN                          # noqa: E402

BOUNDS = {"2": (-2 / 3, 0.0, 2 / 3), "ternary": (-0.5, 0.5)}
LEVEL_SCALE = {"2": 3, "ternary": 1}                 # the integer value of a level = LEVEL_SCALE * its float value


def _fold_conv(conv, bn, w_int, in_scale, bounds, where):
    """BN(raw * in_scale) >= b for each boundary b -> integer thresholds on raw, ascending, rows negated where the BN
    scale < 0"""
    _check_conv(conv, where)
    w = w_int.detach().double().clone()
    g = bn.weight.detach().double()
    sd = torch.sqrt(bn.running_var.detach().double() + bn.eps)
    mu, beta = bn.running_mean.detach().double(), bn.bias.detach().double()
    thr = np.empty((len(g), len(bounds)), dtype=np.int64)
    for o in range(len(g)):
        for k, b in enumerate(bounds):
            if g[o] == 0:                                    # BN output = beta: the level is fixed
                thr[o, k] = INT_MIN if beta[o] >= b else INT_MAX
                continue
            t = float((mu[o] + (b - beta[o]) * sd[o] / g[o]) / in_scale)
            thr[o, k] = math.ceil(t) if g[o] > 0 else math.ceil(-t)   # raw >= t  /  (-raw) >= -t
        if g[o] < 0:
            w[o] = -w[o]
    return dict(w=w.numpy(), thr=np.clip(thr, INT_MIN, INT_MAX), k=conv.kernel_size[0], stride=conv.stride[0], pool=1)


def fold(model):
    model = model.cpu().eval()
    acts = model.acts
    if acts not in BOUNDS:
        raise ValueError(f"acts1d export: activations {acts!r} (only 2 bits or ternary; 1 bit is bool1d.py)")
    if model.first != "int8":
        raise ValueError("acts1d export: only the int8 first layer (the networks README section 9 chose)")
    if model.pool_first:
        raise ValueError("acts1d export: pool_first runs are not supported")
    layers = []
    for kind, i in model.plan:
        if kind == "pool":
            if not layers or layers[-1]["pool"] != 1:
                raise ValueError("acts1d export: a pool must follow a convolution (it is fused into it)")
            layers[-1]["pool"] = int(i)
            continue
        conv = model.convs[i]
        w = torch.round(model.first_weights()) if i == 0 else torch.where(conv.weight.detach() >= 0, 1.0, -1.0)
        scale = 1 / INPUT_SCALE if i == 0 else 1 / LEVEL_SCALE[acts]
        layers.append(_fold_conv(conv, model.bns[i], w, scale, BOUNDS[acts], f"conv {i}"))
    bn = model.fc_bn
    g = bn.weight.detach().double()
    sd = torch.sqrt(bn.running_var.detach().double() + bn.eps)
    a = (g / sd).numpy() / LEVEL_SCALE[acts]            # the classifier sees the integer levels
    c = (bn.bias.detach().double() - bn.running_mean.detach().double() * g / sd).numpy()
    for name, v in (("scale", a), ("bias", c)):
        if np.abs(v).max() * 2 ** FC_SHIFT >= 2 ** 31:
            raise ValueError(f"acts1d export: the classifier {name} does not fit in int32 with {FC_SHIFT} fraction bits")
    cout = layers[-1]["w"].shape[0]
    w = torch.where(model.fc.weight.detach() >= 0, 1.0, -1.0).double().numpy()
    length = w.shape[1] // cout
    return dict(acts=acts, layers=layers, length=length,
                fc_w=w.reshape(len(w), cout, length).transpose(0, 2, 1).reshape(len(w), -1),     # time-major
                fc_scale=np.round(a * 2 ** FC_SHIFT).astype(np.int64), fc_bias=np.round(c * 2 ** FC_SHIFT).astype(np.int64))


def _value(level, acts):
    """level (0..3 or 0..2) -> the integer input of the next layer"""
    return 2 * level - 3 if acts == "2" else level - 1


@torch.no_grad()
def simulate(net, x_int8, batch=1000):
    """the integer network on int8 windows (N, length, channels): the classes, as the C computes them"""
    preds = []
    for i in range(0, len(x_int8), batch):
        x = torch.as_tensor(np.asarray(x_int8[i:i + batch], dtype=np.float64)).transpose(1, 2)    # N x C x L
        for c in net["layers"]:
            raw = F.conv1d(x, torch.as_tensor(c["w"]), stride=c["stride"])
            thr = torch.as_tensor(c["thr"], dtype=torch.float64)                                 # (O, K)
            level = (raw[:, :, :, None] >= thr[None, :, None, :]).sum(-1).double()
            if c["pool"] > 1:
                level = F.max_pool1d(level, c["pool"])
            x = _value(level, net["acts"])
        feats = x.transpose(1, 2).flatten(1)
        raw = (feats @ torch.as_tensor(net["fc_w"]).T).numpy().astype(np.int64)
        preds.append(np.argmax(raw * net["fc_scale"][None] + net["fc_bias"][None], axis=1))
    return np.concatenate(preds)


def write_header(net, path, in_len, in_ch):
    """model_symbex.h: planes1d.h, the layers as Planes1D, symbex_mlp_infer. Returns (flash bytes, RAM bytes of the
    buffers, step names)"""
    acts = net["acts"]
    levels, n_thr = (4, 3) if acts == "2" else (3, 2)
    convs, steps, flash = [], [], 0
    length, ch = in_len, in_ch
    for i, c in enumerate(net["layers"]):
        out_len = ((length - c["k"]) // c["stride"] + 1) // c["pool"]
        thr = c["thr"].copy()
        if i == 0:
            w8 = c["w"].transpose(0, 2, 1).reshape(len(c["w"]), -1)
            if np.abs(w8).max() > 127 or np.any(w8 != np.round(w8)):
                raise ValueError("acts1d export: the int8 first-layer weights are not integers in [-127, 127]")
            w8 = w8.astype(np.int8)
            sums = w8.astype(np.int64).sum(1)
            thr = np.clip(thr + 128 * sums[:, None], INT_MIN, INT_MAX)   # uint8 = int8 + 128
            rows, rw = None, 0
            flash += w8.size + sums.size * 4
        else:
            w8, sums = None, None
            rows, rw, _ = _rows(c["w"])
            flash += rows.size * 4
        flash += thr.size * 4
        convs.append(dict(w8=w8, sums=sums, rows=rows, rw=rw, thr=thr, cin=ch, cout=len(thr), k=c["k"], s=c["stride"],
                          p=c["pool"], h=length, oh=out_len))
        steps.append(f"conv {ch}->{len(thr)} k{c['k']} L{length}" + (" (uint8 input, int8 weights)" if i == 0 else "")
                     + f" -> {acts} acts" + (f" + max pool {c['pool']}" if c["pool"] > 1 else ""))
        length, ch = out_len, len(thr)
    if length != net["length"]:
        raise ValueError(f"acts1d export: the final map has {length} steps, the classifier expects {net['length']}")
    fcw = net["fc_w"] > 0
    n_cls, fc_bits = fcw.shape
    fc_words = -(-fc_bits // 32)
    fc_rw = -(-fc_words // 4) * 4
    fc_rows = np.zeros((n_cls, fc_rw), dtype=np.uint32)
    fc_rows[:, :fc_words] = c_header.words(fcw, fc_words)
    flash += fc_rows.size * 4 + n_cls * 8
    steps.append("classifier")
    # words per plane of each layer's output map (+1: a window may read one word past); the last one padded to the
    # classifier's rows (the PIE count reads whole rows)
    pw = [-(-c["oh"] * c["cout"] // 32) + 1 for c in convs]
    pw[-1] = max(pw[-1], fc_rw)
    pingpong = max([2 * w for w in pw[:-1]] + [8])
    max_rw = max([c["rw"] for c in convs] + [4])
    max_out = max(max(c["cout"] for c in convs), n_cls)
    work = -(-convs[0]["k"] * convs[0]["cin"] // 16) * 16
    n_pingpong = min(2, len(convs) - 1)
    ram = 4 * (n_pingpong * pingpong + 2 * pw[-1] + 2 * max_rw + 3 * max_out) + work
    kernel = open(os.path.join(HERE, "planes1d.h")).read()
    guard = "HAR_ACTS1D_MODEL_H"
    with open(path, "w") as f:
        f.write(f"/* generated by benchmarks/conv1d/har_ultracompact/acts1d.py: a {acts} activation 1D network (1-bit "
                f"weights, int8 first layer)\n   on lib_symbex's HAL row functions, with the interface of the MLP "
                f"headers (symbex_mlp_infer). Do not edit. */\n#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n"
                + c_header.prologue() + "#ifndef SYMBEX_ROM\n#define SYMBEX_ROM const           /* -DSYMBEX_ROM= : "
                "weights in RAM */\n#endif\n#ifndef SYMBEX_MARK\n#define SYMBEX_MARK(i)\n#endif\n\n"
                "/* ---- planes1d.h (embedded) ---- */\n" + kernel + "/* ---- end of planes1d.h ---- */\n\n")
        for i, c in enumerate(convs):
            if i == 0:
                c_header.c_array(f, "int8_t", "W0", c["w8"], " __attribute__((aligned(16)))")
                c_header.c_array(f, "int32_t", "S0", c["sums"])
            else:
                c_header.c_array(f, "uint32_t", f"W{i}", c["rows"], " __attribute__((aligned(16)))")
            c_header.c_array(f, "int32_t", f"T{i}", c["thr"])
            f.write(f"static const struct Planes1D L{i} = "
                    + c_header.struct(in_ch=c["cin"], out_ch=c["cout"], kernel=c["k"], stride=c["s"], pool=c["p"],
                                      in_len=c["h"], out_len=c["oh"], row_words=c["rw"],
                                      in_words=0 if i == 0 else pw[i - 1], out_words=pw[i],
                                      levels_in=0 if i == 0 else levels, n_thr=n_thr,
                                      weights="NULL" if i == 0 else f"W{i}", weights_i8="W0" if i == 0 else "NULL",
                                      thresholds=f"T{i}", weight_sums="S0" if i == 0 else "NULL") + ";\n")
        c_header.c_array(f, "uint32_t", "FC_W", fc_rows, " __attribute__((aligned(16)))")
        c_header.c_array(f, "int32_t", "FC_S", net["fc_scale"])
        c_header.c_array(f, "int32_t", "FC_B", net["fc_bias"])
        f.write("".join(f"static uint32_t {m}[{pingpong}] __attribute__((aligned(16)));\n"
                        for m in ("M_A", "M_B")[:n_pingpong])
                + f"static uint32_t M_F[{2 * pw[-1]}] __attribute__((aligned(16)));\n"
                f"static uint32_t B_WINDOW[{2 * max_rw}] __attribute__((aligned(16)));\n"
                f"static int8_t B_WORK[{work}] __attribute__((aligned(16)));\n"
                f"static int32_t B_RAW[{max_out}], B_TMP[{max_out}], B_ACC[{max_out}];\n\n")
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
                f.write(f"    pl_first(&L0, x, {dst}, B_WORK, B_RAW, B_TMP, B_ACC);\n")
            else:
                f.write(f"    pl_conv(&L{i}, {src}, {dst}, B_WINDOW, B_RAW, B_TMP, B_ACC);\n")
            f.write(f"    SYMBEX_MARK({i + 1});\n")
        f.write(f"    const int c = pl_classify(M_F, {pw[-1]}, {levels}, {fc_words}, {fc_bits}, FC_W, {fc_rw}, {n_cls}, "
                f"FC_S, FC_B, B_RAW, B_TMP);\n    SYMBEX_MARK({n + 1});\n    return c;\n}}\n"
                + c_header.epilogue() + f"#endif /* {guard} */\n")
    return flash, ram, steps


def validate_c(header_path, lib_path, x_u8, expected):
    """gcc on this PC: the header + lib_symbex's generic HAL on x_u8 (N, inputs). Returns (equal, total, message)."""
    return c_build.check(header_path, lib_path, x_u8, expected, sources=["hal/generic/symbex_hal_impl.c"],
                         flags=["-Wall", "-Werror"])
