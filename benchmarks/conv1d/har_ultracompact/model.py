"""
model.py -- the binary 1D networks of the har_ultracompact benchmark: the UniMiB-SHAR architectures of Table 1 of
Daghero et al., "Ultra-compact Binary Neural Networks for Human Activity Recognition on RISC-V Processors" (CF 2021).

    ARCHS[name]                 their four UniMiB-SHAR networks: conv(C_out, K) and pool(4, 4), then one linear layer
    BinaryHAR(arch, first)      the network; input int8 windows N x 151 x 3 (time-major, as har_data.py writes them)
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


def int8_weights(w):
    """symmetric int8, one scale per output channel (the scale folds into the threshold); straight-through rounding.
    Returns the integer values (-127..127) as floats."""
    scale = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min(1e-8) / 127
    q = torch.clamp(torch.round(w / scale), -127, 127)
    return w / scale + (q - w / scale).detach()


class BinaryHAR(nn.Module):
    def __init__(self, arch="max", first="binary"):
        super().__init__()
        assert first in ("binary", "int8")
        self.arch, self.first = arch, first
        self.convs, self.bns, self.plan = nn.ModuleList(), nn.ModuleList(), []
        cin, length = AXES, LENGTH
        for layer in ARCHS[arch]:
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
        self.fc = nn.Linear(self.features, CLASSES, bias=False)
        self.fc_bn = nn.BatchNorm1d(CLASSES)

    @torch.no_grad()
    def first_weights(self):
        """the integer weights (-127..127) of the first conv when first = "int8" (the exporter folds them)"""
        return torch.round(int8_weights(self.convs[0].weight))

    def forward(self, x):
        x = x.float().transpose(1, 2)                                       # N x 3 x 151, int8 values
        for kind, i in self.plan:
            if kind == "pool":
                x = F.max_pool1d(x, i)                                       # max of +-1 = OR of the bits
                continue
            w = self.convs[i].weight
            if i == 0:
                w = int8_weights(w) if self.first == "int8" else sign(w)
                x = x / 127                                                  # only for the scale of training
            else:
                w = sign(w)
            x = sign(self.bns[i](F.conv1d(x, w)))
        return self.fc_bn(F.linear(x.flatten(1), sign(self.fc.weight)))


@torch.no_grad()
def evaluate(model, x, y, batch=2048):
    model.eval()
    hits = 0
    for i in range(0, len(x), batch):
        hits += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return 100 * hits / len(x)
