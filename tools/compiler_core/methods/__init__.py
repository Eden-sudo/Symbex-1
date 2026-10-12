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
from .linear.plain import LinearPlain
from .conv.plain import ConvPlain
from .kbit import LinearKBit, ConvKBit
from .base import RSign
from .kbit import CuantizadorActivacion

# (kind of layer, method name) -> class.  gated / bitslice get added here when they move in.
METHODS = {
    ("linear", "plain"): LinearPlain,
    ("conv", "plain"): ConvPlain,
}
KBIT = {"linear": LinearKBit, "conv": ConvKBit}


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
        kind = "conv" if isinstance(layer, nn.Conv2d) else "linear"
        if (kind, method) not in METHODS:
            disponibles = sorted(m for k, m in METHODS if k == kind)
            raise ValueError(f"method '{method}' is not available for {kind} layers "
                             f"(available: {disponibles})")
        setattr(parent, name, METHODS[(kind, method)].from_float(layer, binarize_input=(i > 0)))
    return model


def apply_plan(model, plan, primera=None, mantener_entrada_primera=True):
    """
    Replaces each nn.Conv2d / nn.Linear named in `plan` (in place): "bin" -> plain (1 bit),
    an integer k >= 2 -> kbit. Layers not in `plan` are left as they are (FP32).
    `primera`: name of the first layer the forward pass runs; its input is left untouched
    (same rule as apply_method). If None, the first layer in definition order is used.
    mantener_entrada_primera=False quantizes that input too (e.g. an INT8 reference, where the
    deployed model also receives a quantized input).
    Call this BEFORE model.to(device).
    """
    nombres = {m: n for n, m in model.named_modules()}
    targets = [(parent, name, child, nombres[child])
               for parent in model.modules()
               for name, child in parent.named_children()
               if isinstance(child, (nn.Conv2d, nn.Linear))]
    if primera is None and targets:
        primera = targets[0][3]
    desconocidas = set(plan) - {t[3] for t in targets}
    if desconocidas:
        raise ValueError(f"plan names layers that are not Conv2d / Linear of the model: {sorted(desconocidas)}")
    for parent, name, layer, completo in targets:
        if completo not in plan:
            continue
        op = plan[completo]
        kind = "conv" if isinstance(layer, nn.Conv2d) else "linear"
        entrada = completo != primera or not mantener_entrada_primera
        if op == "bin":
            nuevo = METHODS[(kind, "plain")].from_float(layer, binarize_input=entrada)
        elif isinstance(op, int) and op >= 2:
            nuevo = KBIT[kind].from_float(layer, op, quantize_input=entrada)
        else:
            raise ValueError(f"layer {completo}: unknown option {op!r} ('bin' or an integer >= 2)")
        setattr(parent, name, nuevo)
    return model


@torch.no_grad()
def init_thresholds(model, calibration):
    """Sets each RSign.beta to the per-channel median of the input it receives on `calibration` (real samples),
    moved halfway to the next larger value of that channel (no sample exactly on the threshold).
    Returns the number of thresholds set. Run it right after apply_plan / apply_method."""
    capturas, hooks = {}, []

    def guardar(mod, ent):
        capturas.setdefault(mod, ent[0].detach())          # returns None: the input is not modified

    for m in model.modules():
        if isinstance(m, RSign):
            hooks.append(m.register_forward_pre_hook(guardar))
    estaba = model.training
    model.eval()
    model(calibration)
    model.train(estaba)
    for h in hooks:
        h.remove()
    for mod, x in capturas.items():
        c = x.shape[1]
        valores = x.movedim(1, 0).reshape(c, -1)
        mediana = valores.median(dim=1).values
        # midpoint between the median and the next larger value of that channel: a threshold placed exactly ON
        # a data value creates exact ties, and with discrete inputs (e.g. the +-1 sums of a binary layer) a
        # whole group of samples sits on the boundary, where float rounding decides the bit.
        # If nothing is above the median (e.g. a channel that is always 0 after a ReLU), use the midpoint with the
        # next SMALLER value instead; if the channel is constant, a threshold 1 below it (always +1, no tie).
        mayores = torch.where(valores > mediana[:, None], valores, torch.full_like(valores, float("inf")))
        menores = torch.where(valores < mediana[:, None], valores, torch.full_like(valores, float("-inf")))
        siguiente = mayores.min(dim=1).values
        anterior = menores.max(dim=1).values
        beta = torch.where(~torch.isinf(siguiente), (mediana + siguiente) / 2,
                           torch.where(~torch.isinf(anterior), (mediana + anterior) / 2, mediana - 1.0))
        mod.beta.copy_(beta.view_as(mod.beta))
    return len(capturas)


@torch.no_grad()
def calibrate_ranges(model, batches):
    """Post-training calibration of the k-bit activation ranges: runs `batches` (an iterable of input
    tensors) with ONLY the activation quantizers in training mode (running min / max), so BatchNorm
    statistics are not touched. Returns the number of quantizers calibrated."""
    estaba = model.training
    model.eval()
    cuants = [m for m in model.modules() if isinstance(m, CuantizadorActivacion)]
    for m in cuants:
        m.train()
    for x in batches:
        model(x)
    model.train(estaba)
    if not estaba:
        model.eval()
    return len(cuants)


_SALTAR = (nn.Flatten, nn.Dropout, nn.Dropout2d, nn.Identity, nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU,
           nn.Tanh, nn.Sigmoid, nn.Hardtanh, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)
_CAPAS = (nn.Linear, nn.Conv2d)
_NORMAS = (nn.BatchNorm1d, nn.BatchNorm2d)


@torch.no_grad()
def widen(model, factor, example, ruido=1e-3, seed=0):
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
    if factor < 1:
        raise ValueError("widen: factor must be >= 1")
    if factor == 1:
        return model
    orden, hooks = [], []
    for n, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            hooks.append(mod.register_forward_hook(lambda m_, e, s_, n=n: orden.append(n)))
    estaba = model.training
    model.eval()
    referencia = model(example)
    for h in hooks:
        h.remove()
    cadena = []                                         # names of the layers / norms, in forward order
    for n in dict.fromkeys(orden):
        mod = model.get_submodule(n)
        if isinstance(mod, _SALTAR):
            continue
        if not isinstance(mod, _CAPAS + _NORMAS):
            raise ValueError(f"widen: unsupported module {type(mod).__name__} (plain Linear / Conv2d chains only)")
        if isinstance(mod, nn.Conv2d) and mod.groups != 1:
            raise ValueError("widen: grouped / depthwise convolutions are not supported")
        cadena.append(n)
    capas = [n for n in cadena if isinstance(model.get_submodule(n), _CAPAS)]
    gen = torch.Generator().manual_seed(seed)

    def poner(nombre, nuevo):
        padre, _, hijo = nombre.rpartition(".")
        setattr(model.get_submodule(padre) if padre else model, hijo, nuevo)

    for i, nombre in enumerate(capas[:-1]):
        capa = model.get_submodule(nombre)
        conv = isinstance(capa, nn.Conv2d)
        n = capa.out_channels if conv else capa.out_features
        nuevo_n = int(round(n * factor))
        extra = torch.randint(0, n, (nuevo_n - n,), generator=gen)
        g = torch.cat([torch.arange(n), extra])                     # new unit j copies original unit g[j]
        cuenta = torch.bincount(g, minlength=n).float()
        # this layer: copy its output units (rows / filters), tiny noise on the copies
        if conv:
            nueva = nn.Conv2d(capa.in_channels, nuevo_n, capa.kernel_size, capa.stride, capa.padding,
                              capa.dilation, 1, capa.bias is not None, capa.padding_mode)
        else:
            nueva = nn.Linear(capa.in_features, nuevo_n, bias=capa.bias is not None)
        nueva.weight.copy_(capa.weight[g])
        if nuevo_n > n and ruido > 0:
            forma = (nuevo_n - n,) + tuple(capa.weight.shape[1:])
            nueva.weight[n:] += ruido * capa.weight.abs().mean() * torch.randn(forma, generator=gen)
        if capa.bias is not None:
            nueva.bias.copy_(capa.bias[g])
        poner(nombre, nueva)
        # the norm right after it, if any: copy its per-unit parameters and statistics
        pos = cadena.index(nombre)
        if pos + 1 < len(cadena) and isinstance(model.get_submodule(cadena[pos + 1]), _NORMAS):
            bn = model.get_submodule(cadena[pos + 1])
            nb = type(bn)(nuevo_n, eps=bn.eps, momentum=bn.momentum, affine=bn.affine)
            if bn.affine:
                nb.weight.copy_(bn.weight[g])
                nb.bias.copy_(bn.bias[g])
            nb.running_mean.copy_(bn.running_mean[g])
            nb.running_var.copy_(bn.running_var[g])
            poner(cadena[pos + 1], nb)
        # next layer: split the outgoing weights of each copied unit among its copies
        sig = model.get_submodule(capas[i + 1])
        if isinstance(sig, nn.Conv2d):
            ns = nn.Conv2d(nuevo_n, sig.out_channels, sig.kernel_size, sig.stride, sig.padding, sig.dilation, 1,
                           sig.bias is not None, sig.padding_mode)
            ns.weight.copy_(sig.weight[:, g] / cuenta[g][None, :, None, None])
        else:
            if sig.in_features % n:
                raise ValueError(f"widen: {capas[i + 1]} has {sig.in_features} inputs, not a multiple of the "
                                 f"{n} channels before it")
            hw = sig.in_features // n                      # 1 for MLPs; H*W after a Flatten of a conv map
            columnas = (g[:, None] * hw + torch.arange(hw)[None, :]).reshape(-1)
            escala = (1.0 / cuenta[g]).repeat_interleave(hw)
            ns = nn.Linear(nuevo_n * hw, sig.out_features, bias=sig.bias is not None)
            ns.weight.copy_(sig.weight[:, columnas] * escala[None, :])
        if sig.bias is not None:
            ns.bias.copy_(sig.bias)
        poner(capas[i + 1], ns)

    # check: same function (up to the noise); anything outside the chain (e.g. residual additions) breaks it
    model.eval()
    salida = model(example)
    dif = (salida - referencia).abs().max().item()
    tolerancia = 1e-3 + (50 * ruido) * referencia.abs().max().item()
    model.train(estaba)               # the new modules are created in training mode: give them the model's mode
    if not torch.isfinite(salida).all() or dif > tolerancia:
        raise ValueError(f"widen: the widened model does not compute the same function (max difference {dif:.3g}): "
                         "the forward pass does something outside a plain chain (residual additions, "
                         "concatenations...), which widening does not support")
    return model
