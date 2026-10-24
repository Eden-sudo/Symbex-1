"""
Binarization methods. A method knows how to turn a regular layer into its
binarized version. (Scan / train / export / validate are separate stages.)

apply_method(model, method)  every Conv2d / Linear with the same method (unchanged).
apply_plan(model, plan)      a different option per layer, for mixed precision:
                             plan = {layer name: "bin" | k}; "bin" -> plain (1 bit), k >= 2 -> kbit.
widen(model, factor)         makes the hidden layers of a plain chain network (MLP or CNN without residual
                             connections) `factor` times wider while computing the same function in FP32 (Net2Net;
                             exact up to a tiny noise): new neurons / channels copy existing ones (plus that noise,
                             to break the symmetry) and the next layer splits their outgoing weights. The binarized
                             model then trains from a wider copy of the original instead of from scratch.
init_thresholds(model, x)    starts every RSign threshold at the per-channel MEDIAN of the real input it
                             receives (instead of 0), moved halfway to the next larger value so that no sample
                             sits exactly on it. After a ReLU every input is >= 0, so a threshold of 0 would
                             turn the whole input into +1 and the layer would start with no information.
"""
import torch
import torch.nn as nn
import sys as _sys
from . import dense as _dense
from .dense import plain as _dense_plain, core as _dense_core, gated as _dense_gated
from .dense.plain import DensePlain
from .dense.ternary import DenseTernary
from .conv.plain import ConvPlain
from .kbit import DenseKBit, ConvKBit
from .base import RSign
from .kbit import ActivationQuantizer

# The folder linear/ is now dense/. Old imports (compiler_core.methods.linear.plain, ...) keep working until
# every caller is translated (see RENAMES.md).
_sys.modules[__name__ + ".linear"] = _dense
_sys.modules[__name__ + ".linear.plain"] = _dense_plain
_sys.modules[__name__ + ".linear.core"] = _dense_core
_sys.modules[__name__ + ".linear.gated"] = _dense_gated
LinearPlain = DensePlain
LinearKBit = DenseKBit

# (kind of layer, method name) -> class.  gated / bitslice get added here when they move in.
METHODS = {
    ("dense", "plain"): DensePlain,
    ("conv", "plain"): ConvPlain,
    ("dense", "ternary"): DenseTernary,
}
KBIT = {"dense": DenseKBit, "conv": ConvKBit}


def apply_method(model, method="plain"):
    """
    Replaces every nn.Conv2d / nn.Linear of `model` (in place) with its `method` version.

    The FIRST layer (in definition order) keeps its input untouched: it receives the
    image / bit-planes, and how those get binarized is decided upstream, not here.
    Call this BEFORE model.to(device).
    """
    targets = [(parent, name, child)
               for parent in model.modules()
               for name, child in parent.named_children()
               if isinstance(child, (nn.Conv2d, nn.Linear))]

    for i, (parent, name, layer) in enumerate(targets):
        kind = "conv" if isinstance(layer, nn.Conv2d) else "dense"
        if (kind, method) not in METHODS:
            available = sorted(m for k, m in METHODS if k == kind)
            raise ValueError(f"method '{method}' is not available for {kind} layers "
                             f"(available: {available})")
        setattr(parent, name, METHODS[(kind, method)].from_float(layer, binarize_input=(i > 0)))
    return model


def apply_plan(model, plan, first=None, keep_first_input=True, primera=None, mantener_entrada_primera=None,
               bin_method="plain"):
    """
    Replaces each nn.Conv2d / nn.Linear named in `plan` (in place): "bin" -> plain (1 bit),
    an integer k >= 2 -> kbit. Layers not in `plan` are left as they are (FP32).
    `first`: name of the first layer the forward pass runs; its input is left untouched
    (same rule as apply_method). If None, the first layer in definition order is used.
    keep_first_input=False quantizes that input too (e.g. an INT8 reference, where the
    deployed model also receives a quantized input).
    bin_method: the METHODS entry a "bin" layer uses ("plain" by default; "ternary" for dense layers).
    Call this BEFORE model.to(device).
    primera / mantener_entrada_primera: old names of first / keep_first_input (kept until every caller is translated).
    """
    if primera is not None:
        first = primera
    if mantener_entrada_primera is not None:
        keep_first_input = mantener_entrada_primera
    names = {m: n for n, m in model.named_modules()}
    targets = [(parent, name, child, names[child])
               for parent in model.modules()
               for name, child in parent.named_children()
               if isinstance(child, (nn.Conv2d, nn.Linear))]
    if first is None and targets:
        first = targets[0][3]
    unknown = set(plan) - {t[3] for t in targets}
    if unknown:
        raise ValueError(f"plan names layers that are not Conv2d / Linear of the model: {sorted(unknown)}")
    for parent, name, layer, full_name in targets:
        if full_name not in plan:
            continue
        op = plan[full_name]
        kind = "conv" if isinstance(layer, nn.Conv2d) else "dense"
        quantize_in = full_name != first or not keep_first_input
        if op == "bin":
            if (kind, bin_method) not in METHODS:
                raise ValueError(f"layer {full_name}: method '{bin_method}' is not available for {kind} layers")
            new_layer = METHODS[(kind, bin_method)].from_float(layer, binarize_input=quantize_in)
        elif isinstance(op, int) and op >= 2:
            new_layer = KBIT[kind].from_float(layer, op, quantize_input=quantize_in)
        else:
            raise ValueError(f"layer {full_name}: unknown option {op!r} ('bin' or an integer >= 2)")
        setattr(parent, name, new_layer)
    return model


@torch.no_grad()
def init_thresholds(model, calibration):
    """Sets each RSign.beta to the per-channel median of the input it receives on `calibration` (real samples),
    moved halfway to the next larger value of that channel (no sample exactly on the threshold).
    Returns the number of thresholds set. Run it right after apply_plan / apply_method."""
    captured, hooks = {}, []

    def store(mod, inputs):
        captured.setdefault(mod, inputs[0].detach())          # returns None: the input is not modified

    for m in model.modules():
        if isinstance(m, RSign):
            hooks.append(m.register_forward_pre_hook(store))
    was_training = model.training
    model.eval()
    model(calibration)
    model.train(was_training)
    for h in hooks:
        h.remove()
    for mod, x in captured.items():
        c = x.shape[1]
        values = x.movedim(1, 0).reshape(c, -1)
        median = values.median(dim=1).values
        # midpoint between the median and the next larger value of that channel: a threshold placed exactly ON
        # a data value creates exact ties, and with discrete inputs (e.g. the +-1 sums of a binary layer) a
        # whole group of samples sits on the boundary, where float rounding decides the bit.
        # If nothing is above the median (e.g. a channel that is always 0 after a ReLU), use the midpoint with the
        # next SMALLER value instead; if the channel is constant, a threshold 1 below it (always +1, no tie).
        larger = torch.where(values > median[:, None], values, torch.full_like(values, float("inf")))
        smaller = torch.where(values < median[:, None], values, torch.full_like(values, float("-inf")))
        next_value = larger.min(dim=1).values
        prev_value = smaller.max(dim=1).values
        beta = torch.where(~torch.isinf(next_value), (median + next_value) / 2,
                           torch.where(~torch.isinf(prev_value), (median + prev_value) / 2, median - 1.0))
        mod.beta.copy_(beta.view_as(mod.beta))
    return len(captured)


@torch.no_grad()
def calibrate_ranges(model, batches):
    """Post-training calibration of the k-bit activation ranges: runs `batches` (an iterable of input
    tensors) with ONLY the activation quantizers in training mode (running min / max), so BatchNorm
    statistics are not touched. Returns the number of quantizers calibrated."""
    was_training = model.training
    model.eval()
    quantizers = [m for m in model.modules() if isinstance(m, ActivationQuantizer)]
    for m in quantizers:
        m.train()
    for x in batches:
        model(x)
    model.train(was_training)
    if not was_training:
        model.eval()
    return len(quantizers)


_SKIP = (nn.Flatten, nn.Dropout, nn.Dropout2d, nn.Identity, nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU,
           nn.Tanh, nn.Sigmoid, nn.Hardtanh, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)
_LAYERS = (nn.Linear, nn.Conv2d)
_NORMS = (nn.BatchNorm1d, nn.BatchNorm2d)


@torch.no_grad()
def widen(model, factor, example, noise=1e-3, seed=0, ruido=None):
    """Widens every hidden layer of a plain chain network (in place, FP32, BEFORE apply_plan) by `factor` (>= 1):
    the outputs of every nn.Linear / nn.Conv2d except the last one. The first layer keeps its inputs and the
    last one its outputs.

    Supported: chains of Linear / Conv2d (groups = 1) -> [BatchNorm1d / BatchNorm2d] -> [element-wise
    activation, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Flatten, Dropout]. A Linear after a Flatten of a
    conv map gets its input columns copied per channel.
    `example` = one input batch: finds the order the layers run in, and CHECKS the result: the widened model
    must compute the same function (up to the symmetry-breaking noise). Anything the forward pass does outside
    those modules (residual additions, concatenations...) breaks that, and raises ValueError. The model is
    modified in place, so pass a copy. Layers are tracked by NAME. Returns the model."""
    if ruido is not None:                    # old name of `noise` (kept until every caller is translated)
        noise = ruido
    if factor < 1:
        raise ValueError("widen: factor must be >= 1")
    if factor == 1:
        return model
    order, hooks = [], []
    for n, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            hooks.append(mod.register_forward_hook(lambda m_, e, s_, n=n: order.append(n)))
    was_training = model.training
    model.eval()
    reference = model(example)
    for h in hooks:
        h.remove()
    chain = []                                         # names of the layers / norms, in forward order
    for n in dict.fromkeys(order):
        mod = model.get_submodule(n)
        if isinstance(mod, _SKIP):
            continue
        if not isinstance(mod, _LAYERS + _NORMS):
            raise ValueError(f"widen: unsupported module {type(mod).__name__} (plain Linear / Conv2d chains only)")
        if isinstance(mod, nn.Conv2d) and mod.groups != 1:
            raise ValueError("widen: grouped / depthwise convolutions are not supported")
        chain.append(n)
    layers = [n for n in chain if isinstance(model.get_submodule(n), _LAYERS)]
    gen = torch.Generator().manual_seed(seed)

    def put(layer_name, new_layer):
        parent_path, _, attr = layer_name.rpartition(".")
        setattr(model.get_submodule(parent_path) if parent_path else model, attr, new_layer)

    for i, layer_name in enumerate(layers[:-1]):
        src_layer = model.get_submodule(layer_name)
        conv = isinstance(src_layer, nn.Conv2d)
        n = src_layer.out_channels if conv else src_layer.out_features
        new_n = int(round(n * factor))
        extra = torch.randint(0, n, (new_n - n,), generator=gen)
        g = torch.cat([torch.arange(n), extra])                     # new unit j copies original unit g[j]
        counts = torch.bincount(g, minlength=n).float()
        # this layer: copy its output units (rows / filters), tiny noise on the copies
        if conv:
            widened = nn.Conv2d(src_layer.in_channels, new_n, src_layer.kernel_size, src_layer.stride, src_layer.padding,
                              src_layer.dilation, 1, src_layer.bias is not None, src_layer.padding_mode)
        else:
            widened = nn.Linear(src_layer.in_features, new_n, bias=src_layer.bias is not None)
        widened.weight.copy_(src_layer.weight[g])
        if new_n > n and noise > 0:
            shape = (new_n - n,) + tuple(src_layer.weight.shape[1:])
            widened.weight[n:] += noise * src_layer.weight.abs().mean() * torch.randn(shape, generator=gen)
        if src_layer.bias is not None:
            widened.bias.copy_(src_layer.bias[g])
        put(layer_name, widened)
        # the norm right after it, if any: copy its per-unit parameters and statistics
        pos = chain.index(layer_name)
        if pos + 1 < len(chain) and isinstance(model.get_submodule(chain[pos + 1]), _NORMS):
            bn = model.get_submodule(chain[pos + 1])
            nb = type(bn)(new_n, eps=bn.eps, momentum=bn.momentum, affine=bn.affine)
            if bn.affine:
                nb.weight.copy_(bn.weight[g])
                nb.bias.copy_(bn.bias[g])
            nb.running_mean.copy_(bn.running_mean[g])
            nb.running_var.copy_(bn.running_var[g])
            put(chain[pos + 1], nb)
        # next layer: split the outgoing weights of each copied unit among its copies
        next_layer = model.get_submodule(layers[i + 1])
        if isinstance(next_layer, nn.Conv2d):
            next_widened = nn.Conv2d(new_n, next_layer.out_channels, next_layer.kernel_size, next_layer.stride, next_layer.padding, next_layer.dilation, 1,
                           next_layer.bias is not None, next_layer.padding_mode)
            next_widened.weight.copy_(next_layer.weight[:, g] / counts[g][None, :, None, None])
        else:
            if next_layer.in_features % n:
                raise ValueError(f"widen: {layers[i + 1]} has {next_layer.in_features} inputs, not a multiple of the "
                                 f"{n} channels before it")
            hw = next_layer.in_features // n                      # 1 for MLPs; H*W after a Flatten of a conv map
            columns = (g[:, None] * hw + torch.arange(hw)[None, :]).reshape(-1)
            scale = (1.0 / counts[g]).repeat_interleave(hw)
            next_widened = nn.Linear(new_n * hw, next_layer.out_features, bias=next_layer.bias is not None)
            next_widened.weight.copy_(next_layer.weight[:, columns] * scale[None, :])
        if next_layer.bias is not None:
            next_widened.bias.copy_(next_layer.bias)
        put(layers[i + 1], next_widened)

    # check: same function (up to the noise); anything outside the chain (e.g. residual additions) breaks it
    model.eval()
    output = model(example)
    diff = (output - reference).abs().max().item()
    tolerance = 1e-3 + (50 * noise) * reference.abs().max().item()
    model.train(was_training)               # the new modules are created in training mode: give them the model's mode
    if not torch.isfinite(output).all() or diff > tolerance:
        raise ValueError(f"widen: the widened model does not compute the same function (max difference {diff:.3g}): "
                         "the forward pass does something outside a plain chain (residual additions, "
                         "concatenations...), which widening does not support")
    return model
