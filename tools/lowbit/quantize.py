"""
quantize.py -- a trained FP32 network as an ordinary low-bit integer network (INT8, INT4, INT2), quantized with
Brevitas, and its exact integer simulation: the arithmetic ESP-NN runs on the chip.

    ops = fold(model, x_sample)                 the network as a list of operations, BatchNorm folded (checked)
    qnet = QNet(ops, bits, weight_init)         the Brevitas model: what QAT trains (fake-quant, straight-through)
    layers = export(qnet)                       its integers: weight codes, int32 biases, multipliers and shifts
    logits = run_integer(layers, x_q)           integer inference, bit-exact with ESP-NN (int8 in, int8 out)

The arithmetic is the standard integer-only scheme of TensorFlow Lite Micro / CMSIS-NN / ESP-NN, with fewer weight bits;
no SYMBEX arithmetic (no bit planes, no popcount): every layer multiplies and accumulates.
  weights      per output channel, symmetric, narrow range: INT8 [-127, 127], INT4 [-7, 7], INT2 [-1, 1]
  activations  int8 per tensor; the input signed (zero point 0), ReLU outputs [0, max] as int8 with zero point -128
  accumulator  int32: sum of w * (x - zero_in) + bias (bias = round(b / (s_in * s_w)))
  requantize   the TFLite fixed-point multiplier (int32 + shift, rounding as esp_nn_multiply_by_quantized_mult),
               then + zero_out, clamped to [-128, 127]
On the chip the 4- and 2-bit codes are stored packed and unpacked to int8 for the ESP-NN kernels.

Supported networks: a sequence of Flatten, Linear, Conv1d, Conv2d (BatchNorm right after a Linear / Conv is folded
into it), ReLU, MaxPool1d / MaxPool2d, Dropout / Identity (skipped). fold() checks the sequence against the model's own
forward on a sample and stops if they differ (e.g. a residual network).
"""
import math

import numpy as np
import torch
import torch.nn as nn

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


# ---------------------------------------------------------------------------------------------------
# 1. The FP32 network as a sequence of operations
# ---------------------------------------------------------------------------------------------------
def _leaves(module):
    kids = list(module.children())
    if not kids:
        return [module]
    return [leaf for k in kids for leaf in _leaves(k)]


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def fold(model, x_sample):
    """model (eval) -> list of ops (float64 numpy), BatchNorm folded. Convolutions are kept 2D: a Conv1d is a
    Conv2d of height 1 (its input N x C x L is handled as N x C x 1 x L)."""
    model = model.eval().cpu()
    ops = []
    for m in _leaves(model):
        if isinstance(m, (nn.Dropout, nn.Identity)):
            continue
        if isinstance(m, nn.Flatten):
            ops.append(dict(op="flatten"))
        elif isinstance(m, nn.Linear):
            ops.append(dict(op="linear", W=m.weight.detach().double().numpy(),
                            b=(m.bias.detach().double().numpy() if m.bias is not None else np.zeros(m.out_features))))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            if m.groups != 1 or any(d != 1 for d in m.dilation):
                raise SystemExit(f"[X] lowbit: {m} (groups / dilation) is not supported")
            W = m.weight.detach().double().numpy()
            one_d = isinstance(m, nn.Conv1d)
            if one_d:
                W = W[:, :, None, :]
            ops.append(dict(op="conv", dim=1 if one_d else 2, W=W,
                            b=(m.bias.detach().double().numpy() if m.bias is not None else np.zeros(m.out_channels)),
                            stride=(1, m.stride[0]) if one_d else _pair(m.stride),
                            padding=(0, m.padding[0]) if one_d else _pair(m.padding)))
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if not ops or ops[-1]["op"] not in ("linear", "conv"):
                raise SystemExit(f"[X] lowbit: a BatchNorm not right after a Linear / Conv cannot be folded ({m})")
            g = m.weight.detach().double().numpy() / np.sqrt(m.running_var.double().numpy() + m.eps)
            last = ops[-1]
            last["b"] = g * (last["b"] - m.running_mean.double().numpy()) + m.bias.detach().double().numpy()
            last["W"] = last["W"] * g.reshape((-1,) + (1,) * (last["W"].ndim - 1))
        elif isinstance(m, nn.ReLU):
            ops.append(dict(op="relu"))
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d)):
            one_d = isinstance(m, nn.MaxPool1d)
            k = m.kernel_size if not one_d else (1, m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0])
            s = m.stride if not one_d else (1, m.stride if isinstance(m.stride, int) else m.stride[0])
            ops.append(dict(op="maxpool", k=_pair(k), s=_pair(s)))
        else:
            raise SystemExit(f"[X] lowbit: layer {type(m).__name__} is not supported (see quantize.py)")
    with torch.no_grad():
        ref = model(x_sample.cpu()).double().numpy()
    got = run_float(ops, x_sample.cpu().double().numpy())
    err = float(np.abs(ref - got).max() / max(np.abs(ref).max(), 1e-12))
    if err > 1e-4:
        raise SystemExit(f"[X] lowbit: the folded sequence does not reproduce the model (relative error {err:.2e}): "
                         "the network is not a plain sequence of supported layers")
    return ops


def _as_4d(x):
    return x[:, :, None, :] if x.ndim == 3 else x


def _im2col(x, kh, kw, sh, sw):
    """x (N, C, H, W) -> (N, Ho, Wo, C * kh * kw), the same order as a weight (out, C, kh, kw) reshaped"""
    n, c, h, w = x.shape
    ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
    cols = np.empty((n, ho, wo, c, kh, kw), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, :, i, j] = x[:, :, i:i + sh * ho:sh, j:j + sw * wo:sw].transpose(0, 2, 3, 1)
    return cols.reshape(n, ho, wo, -1)


def _maxpool(x, k, s):
    n, c, h, w = x.shape
    ho, wo = (h - k[0]) // s[0] + 1, (w - k[1]) // s[1] + 1
    out = np.full((n, c, ho, wo), -np.inf if x.dtype.kind == "f" else np.iinfo(x.dtype).min, dtype=x.dtype)
    for i in range(k[0]):
        for j in range(k[1]):
            out = np.maximum(out, x[:, :, i:i + s[0] * ho:s[0], j:j + s[1] * wo:s[1]])
    return out


def run_float(ops, x):
    """float reference of the folded sequence (to check fold())"""
    for o in ops:
        if o["op"] == "flatten":
            x = x.reshape(len(x), -1)
        elif o["op"] == "linear":
            x = x @ o["W"].T + o["b"]
        elif o["op"] == "conv":
            x = _as_4d(x)
            ph, pw = o["padding"]
            x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
            co, _, kh, kw = o["W"].shape
            x = (_im2col(x, kh, kw, *o["stride"]) @ o["W"].reshape(co, -1).T + o["b"]).transpose(0, 3, 1, 2)
        elif o["op"] == "relu":
            x = np.maximum(x, 0)
        elif o["op"] == "maxpool":
            x = _maxpool(_as_4d(x), o["k"], o["s"])
    return x


# ---------------------------------------------------------------------------------------------------
# 2. The Brevitas model
# ---------------------------------------------------------------------------------------------------
class QNet(nn.Module):
    """The folded network with Brevitas quantizers: an int8 quantizer on the input, `bits`-bit per-channel weights,
    an int8 quantizer after every Linear / Conv (unsigned for ReLU outputs). weight_init: "maxabs" (scale = max|w| /
    qmax, fixed: post-training), "mse" (the scale that minimizes the weight error, fixed: post-training) or "learned"
    (initialized from max|w|, trained with the weights: QAT)."""

    def __init__(self, ops, bits, weight_init="learned"):
        super().__init__()
        import brevitas.nn as qnn
        from brevitas.inject.enum import ScalingImplType
        from brevitas.quant import (Int8ActPerTensorFloat, Int8WeightPerChannelFloat, Int8WeightPerChannelFloatMSE,
                                    Uint8ActPerTensorFloat)
        self.bits, self.weight_init, self.qmax = bits, weight_init, 2 ** (bits - 1) - 1
        wq = Int8WeightPerChannelFloatMSE if weight_init == "mse" else Int8WeightPerChannelFloat
        extra = dict(weight_scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS) if weight_init == "learned" else {}
        self.kinds, mods = [], [qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)]
        self.kinds.append(("input", None))
        i = 0
        while i < len(ops):
            o = ops[i]
            if o["op"] in ("linear", "conv"):
                relu = i + 1 < len(ops) and ops[i + 1]["op"] == "relu"
                W = torch.tensor(o["W"], dtype=torch.float32)
                if o["op"] == "linear":
                    m = qnn.QuantLinear(W.shape[1], W.shape[0], bias=True, weight_quant=wq, weight_bit_width=bits,
                                        **extra)
                else:
                    m = qnn.QuantConv2d(W.shape[1], W.shape[0], tuple(W.shape[2:]), stride=o["stride"],
                                        padding=o["padding"], bias=True, weight_quant=wq, weight_bit_width=bits,
                                        **extra)
                with torch.no_grad():
                    m.weight.copy_(W)
                    m.bias.copy_(torch.tensor(o["b"], dtype=torch.float32))
                mods.append(m)
                self.kinds.append((o["op"], o))
                act = (qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True) if relu else
                       qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True))
                mods.append(act)
                self.kinds.append(("act_relu" if relu else "act", None))
                i += 2 if relu else 1
            elif o["op"] == "relu":
                raise SystemExit("[X] lowbit: a ReLU that does not follow a Linear / Conv is not supported")
            else:
                mods.append(nn.Identity())
                self.kinds.append((o["op"], o))
                i += 1
        self.mods = nn.ModuleList(mods)
        self.scales = {}

    def forward(self, x):
        for k, ((kind, o), m) in enumerate(zip(self.kinds, self.mods)):
            if kind == "conv" and x.dim() == 3:
                x = x[:, :, None, :]
            if kind == "flatten":
                x = x.flatten(1)
            elif kind == "maxpool":
                x = torch.nn.functional.max_pool2d(x if x.dim() == 4 else x[:, :, None, :], o["k"], o["s"])
            elif kind in ("input", "act", "act_relu"):
                q = m(x)
                if hasattr(q, "scale"):                  # a plain tensor while Brevitas calibrates
                    self.scales[k] = q.scale.detach().reshape(-1)[0]
                    q = q.value
                x = q
            else:
                x = m(x)
        return x


# ---------------------------------------------------------------------------------------------------
# 3. The integer model
# ---------------------------------------------------------------------------------------------------
def quantize_multiplier(m):
    """TFLite's QuantizeMultiplier: real m > 0 -> (int32 q, shift) with m = q * 2^(shift - 31)"""
    if m == 0:
        return 0, 0
    frac, shift = math.frexp(m)
    q = int(round(frac * (1 << 31)))
    if q == 1 << 31:
        q //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return q, shift


def export(qnet):
    """the integers of a QNet (after a forward pass in eval mode, which fixes the activation scales):
    {"input": (scale, zero)} then per op a dict; Linear / Conv: W (int codes, out x ...), bias int32, mult and shift
    per channel, zero_in, zero_out (int8 domain)"""
    qnet.eval()
    layers, s_in, z_in = [], None, 0
    pending = None
    for k, ((kind, o), m) in enumerate(zip(qnet.kinds, qnet.mods)):
        if kind == "input":
            s_in, z_in = float(qnet.scales[k]), 0
            layers.append(dict(op="input", scale=s_in, zero=0))
        elif kind in ("linear", "conv"):
            qw = m.quant_weight()
            Wq = qw.int().detach().cpu().numpy().astype(np.int64)
            s_w = qw.scale.detach().cpu().double().numpy().reshape(-1)
            bq = np.round(m.bias.detach().cpu().double().numpy() / (s_in * s_w)).astype(np.int64)
            pending = dict(op=kind, W=Wq, bias=np.clip(bq, INT32_MIN, INT32_MAX), s_w=s_w, s_in=s_in, zero_in=z_in,
                           stride=o.get("stride"), padding=o.get("padding"), dim=o.get("dim"), bits=qnet.bits)
        elif kind in ("act", "act_relu"):
            s_out = float(qnet.scales[k])
            z_out = -128 if kind == "act_relu" else 0
            mults = [quantize_multiplier(s_in * sw / s_out) for sw in pending["s_w"]]
            pending.update(mult=np.array([q for q, _ in mults], dtype=np.int64),
                           shift=np.array([s for _, s in mults], dtype=np.int64), zero_out=z_out, s_out=s_out,
                           relu=kind == "act_relu")
            layers.append(pending)
            s_in, z_in = s_out, z_out
        else:
            layers.append(dict(op=kind, k=o.get("k"), s=o.get("s")))
    return layers


def quantize_input(layers, x):
    """float input -> int8, as the input quantizer of the QNet (round half to even, like torch.round)"""
    s = layers[0]["scale"]
    return np.clip(np.round(np.asarray(x, dtype=np.float64) / s), -128, 127).astype(np.int64)


def _sat_round_doubling_high_mul(a, b):
    """esp_nn_sat_round_doubling_high_mul, vectorized (a: int32 values, b: int32 multipliers, broadcast)"""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    nudge = np.where((a < 0) ^ (b < 0), 1 - (1 << 30), 1 << 30)
    v = a * b + nudge
    to_add = (v >> 63) & ((1 << 31) - 1)
    r = (v + to_add) >> 31
    return np.where((a == INT32_MIN) & (b == INT32_MIN), INT32_MAX, r)


def _div_by_power_of_two(v, e):
    """esp_nn_div_by_power_of_two (rounding half away from zero), vectorized; e >= 0 per channel"""
    mask = (np.int64(1) << e) - 1
    rem = v & mask
    r = v >> e
    return r + (rem > (mask >> 1) + (r < 0))


def _wrap32(v):
    return ((v + (1 << 31)) % (1 << 32)) - (1 << 31)


def requantize(acc, mult, shift):
    """esp_nn_multiply_by_quantized_mult: acc (..., channels) int32, mult / shift per channel"""
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    x = _wrap32(acc * (np.int64(1) << left))
    return _div_by_power_of_two(_sat_round_doubling_high_mul(x, mult), right)


def run_integer(layers, x_q):
    """integer inference: x_q int8 input (N, ...) -> int8 output of the last layer (N, classes) or its feature map"""
    x = np.asarray(x_q, dtype=np.int64)
    for c in layers:
        if c["op"] == "input":
            continue
        if c["op"] == "flatten":
            x = x.reshape(len(x), -1)
        elif c["op"] == "maxpool":
            x = _maxpool(_as_4d(x), c["k"], c["s"])
        elif c["op"] == "linear":
            acc = (x - c["zero_in"]) @ c["W"].T + c["bias"]
            x = np.clip(requantize(acc, c["mult"], c["shift"]) + c["zero_out"], -128, 127)
        elif c["op"] == "conv":
            x = _as_4d(x)
            ph, pw = c["padding"]
            x = np.pad(x - c["zero_in"], ((0, 0), (0, 0), (ph, ph), (pw, pw)))     # padding = float zero
            co, _, kh, kw = c["W"].shape
            acc = _im2col(x, kh, kw, *c["stride"]) @ c["W"].reshape(co, -1).T + c["bias"]
            x = np.clip(requantize(acc, c["mult"], c["shift"]) + c["zero_out"], -128, 127).transpose(0, 3, 1, 2)
    return x


def weight_bytes(layers):
    """packed weights (ceil(n * bits / 8) per layer) and the total with, per output channel, an int32 bias, an int32
    multiplier and an int8 shift"""
    w = sum(math.ceil(c["W"].size * c["bits"] / 8) for c in layers if c["op"] in ("linear", "conv"))
    ch = sum(len(c["bias"]) for c in layers if c["op"] in ("linear", "conv"))
    return dict(weight_bytes=w, total_bytes=w + 9 * ch)
