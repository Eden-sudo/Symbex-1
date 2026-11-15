"""
model.py -- the binary 1D networks of the har_ultracompact benchmark: the UniMiB-SHAR architectures of Table 1 of
Daghero et al., "Ultra-compact Binary Neural Networks for Human Activity Recognition on RISC-V Processors" (CF 2021).

    ARCHS[name]                 their four UniMiB-SHAR networks: conv(C_out, K) and pool(4, 4), then one linear layer
    BinaryHAR(arch, first)      the network; input int8 windows N x 151 x 3 (time-major, as har_data.py writes them)
    BinaryHAR(arch, first, layers=..., length=..., channels=..., classes=...)
                                the same layers on another data set (benchmarks/conv1d/ecg_mitbih): layers in the
                                format of ARCHS, the input N x length x channels, `classes` outputs
    evaluate(model, x, y)       accuracy in %

Every layer as in their paper and library:
    first conv  8-bit input (the int8 accelerometer samples), its output binarized (BatchNorm + sign). Its weights
                differ between the two libraries, each as designed:
                  first="binary"  1-bit weights (sign): lib_symbex's uint8 layer (symbex_bool1d_conv_u8)
                  first="int8"    int8 weights (symmetric, one scale per output channel, straight-through rounding):
                                  ultracompactBNN's 8-bit-to-1-bit layer (int8 x int8, then a threshold)
    other convs 1-bit weights on 1-bit inputs, BatchNorm, sign: no padding, stride 1
    pool(4, 4)  max of +-1 over 4 steps = OR of the bits
    linear      1-bit weights on the flattened final bits, BatchNorm; the class is the largest score (not binarized)
On the chip every BatchNorm + sign is one integer threshold per channel.

acts (README section 9, an experiment on the PC: nothing exports it yet): the activations after every conv, 1-bit
weights unchanged. acts="1" sign (the networks above); acts="2" four levels {-1, -1/3, 1/3, 1} (2 bits, thresholds
at -2/3, 0, 2/3 of the BatchNorm output: three integer thresholds per channel on a chip); acts="ternary" three levels
{-1, 0, 1} (thresholds at -1/2 and 1/2). Straight-through gradient where |x| <= 1, as sign. The max pool of levels is
still a max (the quantizers are monotone, so pooling before or after them gives the same outputs).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = {                                  # Table 1 of Daghero et al., UniMiB-SHAR rows
    "min":    [("conv", 4, 7), ("conv", 4, 7), ("pool", 4), ("conv", 4, 7), ("pool", 4)],
    "max10":  [("conv", 8, 15), ("conv", 32, 7), ("pool", 4), ("conv", 32, 7), ("pool", 4)],
    "max5":   [("conv", 32, 7), ("conv", 32, 15), ("pool", 4), ("conv", 32, 15), ("pool", 4)],
    "max":    [("conv", 32, 15), ("conv", 32, 15), ("pool", 4), ("conv", 32, 15), ("pool", 4)],
}
TABLE1 = ("min", "max10", "max5", "max")   # --all trains these; the wider ones below are trained on request
# wider versions (the capacity study, README section 5): every conv's channels x 1.5 / x 2, kernels and pools unchanged
ARCHS.update({f"{a}_w{tag}": [(l[0], round(l[1] * w), l[2]) if l[0] == "conv" else l for l in ARCHS[a]]
              for a in ("min", "max") for tag, w in (("15", 1.5), ("2", 2.0))})
LENGTH, AXES, CLASSES = 151, 3, 17


class _Sign(torch.autograd.Function):
    """sign with 0 -> +1; straight-through gradient where |x| <= 1"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)


sign = _Sign.apply
ACTS = ("1", "2", "ternary")


class _Levels(torch.autograd.Function):
    """uniform levels in [-1, 1] (n = 4: -1, -1/3, 1/3, 1; n = 3: -1, 0, 1); straight-through gradient where |x| <= 1"""
    @staticmethod
    def forward(ctx, x, n):
        ctx.save_for_backward(x)
        step = 2 / (n - 1)
        return torch.clamp(torch.round((x + 1) / step), 0, n - 1) * step - 1

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype), None


def quantize(x, acts):
    """the activation of model.BinaryHAR's convs: sign, 4 levels (2 bits) or 3 levels (ternary)"""
    if acts == "1":
        return sign(x)
    return _Levels.apply(x, 4 if acts == "2" else 3)


def int8_weights(w):
    """symmetric int8, one scale per output channel (the scale folds into the threshold); straight-through rounding.
    Returns the integer values (-127..127) as floats."""
    scale = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min(1e-8) / 127
    q = torch.clamp(torch.round(w / scale), -127, 127)
    return w / scale + (q - w / scale).detach()


class BinaryHAR(nn.Module):
    def __init__(self, arch="max", first="binary", layers=None, length=LENGTH, channels=AXES, classes=CLASSES,
                 pool_first=False, acts="1"):
        super().__init__()
        assert first in ("binary", "int8") and acts in ACTS
        # pool_first: max pool on the BatchNorm output, then sign. sign is monotone, so sign(max(v)) = max(sign(v)):
        # the same outputs as the OR of the bits the chip runs, but the gradient reaches the true maximum instead of
        # one arbitrary element of a window of tied +-1 values (training only; the exported network is unchanged)
        self.arch, self.first, self.pool_first, self.acts = arch, first, pool_first, acts
        self.convs, self.bns, self.plan = nn.ModuleList(), nn.ModuleList(), []
        cin = channels
        for layer in (ARCHS[arch] if layers is None else layers):
            if layer[0] == "conv":
                _, cout, k = layer
                self.convs.append(nn.Conv1d(cin, cout, k, bias=False))
                self.bns.append(nn.BatchNorm1d(cout))
                self.plan.append(("conv", len(self.convs) - 1))
                cin, length = cout, length - k + 1
            else:
                self.plan.append(("pool", layer[1]))
                length //= layer[1]
        self.features = cin * length
        self.fc = nn.Linear(self.features, classes, bias=False)
        self.fc_bn = nn.BatchNorm1d(classes)

    @torch.no_grad()
    def first_weights(self):
        """the integer weights (-127..127) of the first conv when first = "int8" (the exporter folds them)"""
        return torch.round(int8_weights(self.convs[0].weight))

    def forward(self, x):
        x = x.float().transpose(1, 2)                                       # N x channels x length (3 x 151), int8
        for j, (kind, i) in enumerate(self.plan):
            if kind == "pool":
                if not (self.pool_first and j > 0 and self.plan[j - 1][0] == "conv"):
                    x = F.max_pool1d(x, i)                                   # max of +-1 = OR of the bits
                continue
            w = self.convs[i].weight
            if i == 0:
                w = int8_weights(w) if self.first == "int8" else sign(w)
                x = x / 127                                                  # only for the scale of training
            else:
                w = sign(w)
            x = self.bns[i](F.conv1d(x, w))
            nxt = self.plan[j + 1] if j + 1 < len(self.plan) else None
            if self.pool_first and nxt is not None and nxt[0] == "pool":
                x = F.max_pool1d(x, nxt[1])                                  # pooled here, before sign (see above)
            x = quantize(x, self.acts)
        return self.fc_bn(F.linear(x.flatten(1), sign(self.fc.weight)))


@torch.no_grad()
def evaluate(model, x, y, batch=2048):
    model.eval()
    hits = 0
    for i in range(0, len(x), batch):
        hits += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return 100 * hits / len(x)
