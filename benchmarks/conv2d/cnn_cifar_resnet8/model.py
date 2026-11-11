"""
model.py -- the binary network of the cnn_cifar_resnet8 benchmark, and what the scripts here share.

The network has the shape of the MLPerf Tiny ResNet-8 (the INT8 reference, resnet8_tflite.py): a 3x3 stem, three
stages of two 3x3 convolutions (16, 32, 64 channels; stride 1, 2, 2) and a linear classifier. What changes is only
what binarization imposes:
    input       a thermometer of `levels` bits per color (FracBNN): bit k = pixel >= (k + 1) * 256 // (levels + 1)
    every conv  1-bit weights (sign, no scale) on 1-bit inputs, BatchNorm, sign -> 1-bit outputs
    shortcut    an OR of bits (max of +-1); where the shape changes: OR-pool 2x2, 1x1 binary conv, BatchNorm, sign
    classifier  1-bit weights on the final bits, BatchNorm
On the chip there is no multiplication at all: every BatchNorm + sign is one integer threshold per channel.
--width scales every channel count (x1: 16-32-64, the shape of the INT8 reference; x2: 32-64-128).

    BoolResNet8(width, levels)    the network (input: uint8 images N x 3 x 32 x 32, or floats in [0, 1] as the CDT gives)
    BoolResNet8x1 / BoolResNet8x2 the two widths, buildable with no arguments ([model] definition of the CDT)
    VARIANTS                      the stem / shortcut study (resnet8_symbex.py --variant): int8 stem, added shortcuts;
                                  int8stem_pool2: int8 stem + pooled int8 classifier (trained with Bop)
    PoolHead                      the pooled int8 classifier: ones per k x k window of the final map, then int8 weights
    parse_spec / write_definition variants for the sizing study: "16-32-64/2-2-2" = channels / blocks per stage
    initial_weights(cls, path)    the random starting weights the CDT loads ([model] file)
    thermometer(x_u8, levels)     the input encoding, in +-1
    cifar_loaders()               training (crop + flip) and test loaders, uint8 images
    cifar_u8(train)               the images as uint8 NHWC + labels, official order (the INT8 side uses it too)
    evaluate(model, loader)       accuracy in %
    load_model(path, ...)         a trained model; also reads the checkpoints of convul_symbex_v3/purga_symbex.py
Data: ../../data/cifar10 (downloaded the first time).
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cifar10")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
BASE_CHANNELS = (16, 32, 64)
STRIDES = (1, 2, 2)


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


class BinaryConv(nn.Conv2d):
    """1-bit weights (sign), no scale, no bias; the input is already +-1"""
    def forward(self, x):
        return F.conv2d(x, sign(self.weight), None, self.stride, self.padding)


class BinaryLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, sign(self.weight))


def thermometer_thresholds(levels):
    return (np.arange(levels) + 1) * 256 // (levels + 1)


def thermometer(x_u8, levels):
    """uint8 (N, 3, H, W) -> (N, 3 * levels, H, W) in +-1; channel = color * levels + k"""
    t = torch.as_tensor(thermometer_thresholds(levels), device=x_u8.device)
    return (x_u8.unsqueeze(2) >= t.view(1, 1, -1, 1, 1)).flatten(1, 2).float() * 2 - 1


class Int8Conv(nn.Conv2d):
    """8-bit weights (symmetric per output channel, straight-through) on the centered pixels (pixel - 128) / 128: on a
    chip an int8 x int8 dot product per output, then the BatchNorm + sign as one integer threshold (as symbex_bool1d's
    int8 first layer)"""
    def int8_weights(self):
        """the integers the chip uses (what the CDT's boolean export reads): q (O, C, k, k) in -127..127 and the
        per-channel scale s (O,), weight = q * s"""
        w = self.weight.detach()
        s = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8) / 127
        return torch.round(w / s).clamp(-127, 127), s.flatten()

    def forward(self, x):
        w = self.weight
        s = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8) / 127
        wq = w + (torch.round(w / s).clamp(-127, 127) * s - w).detach()
        return F.conv2d(x, wq, None, self.stride, self.padding)


def _int8_ste(w):
    """symmetric int8 weights per output row (scale = max |w| / 127), straight-through"""
    s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
    return w + (torch.round(w / s) * s - w).detach()


class Int8Head(nn.Linear):
    """int8 weights on the counts of the pooled classifier (a sum of int8 weights per class on a chip)"""
    def forward(self, x):
        return F.linear(x, _int8_ste(self.weight))


class PoolHead(nn.Module):
    """the pooled int8 classifier (head="pool2"): the count of ones per window x window block of the final map
    (0..window^2; window 0 = the whole map), then int8 weights per class, no bias (the BatchNorm after it gives the
    offset). Input: the flattened map (N, C*H*W, channel-major, as BoolResNet8.forward gives it). The boolean export
    runs it as lib_symbex's symbex_bool_pool_head (window, side and int8_weights() are what it reads)"""
    def __init__(self, channels, classes, window=0, side=0):
        super().__init__()
        self.channels, self.window, self.side = channels, window, side
        n_in = channels if window == 0 else channels * (side // window) ** 2
        self.fc = Int8Head(n_in, classes, bias=False)

    def forward(self, x):
        bits = (x + 1) / 2
        if self.window == 0:
            return self.fc(bits.view(x.shape[0], self.channels, -1).sum(-1))
        maps = bits.view(x.shape[0], self.channels, self.side, self.side)
        counts = F.avg_pool2d(maps, self.window) * self.window ** 2
        return self.fc(counts.flatten(1))

    def int8_weights(self):
        """integers q (classes, inputs) in -127..127 and per-class scales s: the weights the forward uses = q * s"""
        w = self.fc.weight.detach()
        s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
        return torch.round(w / s), s.flatten()


class Block(nn.Module):
    """one binary conv with its OR shortcut: out = sign(BN(conv(x))) OR shortcut(x), all in +-1.
    shortcut="add": out = sign(BN(conv(x)) + shortcut(x)) instead (the shortcut bit shifts the threshold: two integer
    thresholds per channel on a chip, the maps stay bits)"""
    def __init__(self, cin, cout, stride, shortcut="or"):
        super().__init__()
        self.add = shortcut == "add"
        self.conv = BinaryConv(cin, cout, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.stride = stride
        self.changes = stride != 1 or cin != cout
        if self.changes:                     # shortcut that changes shape: OR-pool (stride 2), 1x1 binary conv, BN, sign
            self.ds_conv = BinaryConv(cin, cout, 1, 1, 0, bias=False)
            self.ds_bn = nn.BatchNorm2d(cout)

    def shortcut(self, x):
        if not self.changes:
            return x
        xp = F.max_pool2d(x, 2) if self.stride == 2 else x          # max of +-1 over 2x2 = OR of the 4 bits
        return sign(self.ds_bn(self.ds_conv(xp)))

    def forward(self, x):
        if self.add:
            return sign(self.bn(self.conv(x)) + self.shortcut(x))
        return torch.maximum(sign(self.bn(self.conv(x))), self.shortcut(x))   # OR of bits


class BoolResNet8(nn.Module):
    """channels / blocks: per stage (default: BASE_CHANNELS times width, two blocks per stage: the ResNet-8).
    stem="int8": an Int8Conv on the pixels instead of the binary conv on the thermometer (the boolean export runs it
    as lib_symbex's optional int8 first layer, symbex_bool_stem_i8); shortcut="add": see Block (the stem / shortcut
    study only; the boolean export takes OR shortcuts); head="pool2": the pooled int8 classifier (PoolHead, 2x2
    windows) instead of the binary one"""
    def __init__(self, width=1.0, levels=8, classes=10, channels=None, blocks=(2, 2, 2), stem="thermometer",
                 shortcut="or", head="binary"):
        super().__init__()
        channels = tuple(channels) if channels else tuple(int(round(c * width)) for c in BASE_CHANNELS)
        self.width, self.levels, self.channels, self.blocks_per_stage = width, levels, channels, tuple(blocks)
        self.stem_kind = stem
        if stem == "int8":
            self.stem = Int8Conv(3, channels[0], 3, 1, 1, bias=False)
        else:
            self.stem = BinaryConv(3 * levels, channels[0], 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        layers, cin = [], channels[0]
        for c, s, n in zip(channels, STRIDES, blocks):
            for j in range(n):                       # the first block of a stage changes the size; the rest keep it
                layers.append(Block(cin, c, s if j == 0 else 1, shortcut))
                cin = c
        blocks = layers
        self.blocks = nn.Sequential(*blocks)
        res = 32
        for s in STRIDES:
            res //= s
        if head == "binary":
            self.fc = BinaryLinear(cin * res * res, classes)
        elif head == "pool2":
            self.fc = PoolHead(cin, classes, 2, res)
        else:
            raise ValueError(f"head {head!r}: binary or pool2")
        self.fc_bn = nn.BatchNorm1d(classes)

    def forward(self, x):
        if x.is_floating_point():                    # the CDT gives images in [0, 1]: back to the 0..255 pixel values
            x = torch.round(x * 255)
        if self.stem_kind == "int8":
            x = sign(self.stem_bn(self.stem((x.float() - 128) / 128)))
        else:
            x = sign(self.stem_bn(self.stem(thermometer(x, self.levels))))
        x = self.blocks(x)
        return self.fc_bn(self.fc(x.flatten(1)))


class BoolResNet8x1(BoolResNet8):
    """16-32-64 channels: the shape of the INT8 reference (built with no arguments, as the CDT needs)"""
    def __init__(self):
        super().__init__(1.0, 8)


class BoolResNet8x2(BoolResNet8):
    """32-64-128 channels"""
    def __init__(self):
        super().__init__(2.0, 8)


class BoolResNet8Int8Stem(BoolResNet8):
    """the x1 network with an int8 stem on the pixels (stem / shortcut study)"""
    def __init__(self):
        super().__init__(1.0, 8, stem="int8")


class BoolResNet8AddShortcut(BoolResNet8):
    """the x1 network with added shortcuts (stem / shortcut study)"""
    def __init__(self):
        super().__init__(1.0, 8, shortcut="add")


class BoolResNet8Int8StemAdd(BoolResNet8):
    """the x1 network with both: int8 stem and added shortcuts (stem / shortcut study)"""
    def __init__(self):
        super().__init__(1.0, 8, stem="int8", shortcut="add")


class BoolResNet8Int8StemPool2(BoolResNet8):
    """the x1 network with an int8 stem and the pooled int8 classifier (2x2 windows): the most accurate x1 of the
    training-gap study when trained with Bop (benchmarks/experiments/training_gap)"""
    def __init__(self):
        super().__init__(1.0, 8, stem="int8", head="pool2")


VARIANTS = {"base": "BoolResNet8x1", "int8stem": "BoolResNet8Int8Stem", "add": "BoolResNet8AddShortcut",
            "int8stem_add": "BoolResNet8Int8StemAdd", "int8stem_pool2": "BoolResNet8Int8StemPool2"}
BOP_VARIANTS = ("int8stem_pool2",)       # trained with Bop for the 1-bit weights ([training] binary_optimizer = "bop")


def parse_spec(spec):
    """"16-32-64/2-2-2" -> ((16, 32, 64), (2, 2, 2)): channels and blocks per stage"""
    ch, bl = spec.split("/")
    return tuple(int(v) for v in ch.split("-")), tuple(int(v) for v in bl.split("-"))


def spec_name(spec):
    ch, bl = parse_spec(spec)
    return "c" + "-".join(map(str, ch)) + "_b" + "-".join(map(str, bl))


def write_definition(spec, folder):
    """a module with this variant as a class buildable with no arguments (what the CDT needs): returns
    'path:BoolResNet8Variant'"""
    parse_spec(spec)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(os.path.abspath(folder), "variant_model.py")
    open(path, "w").write(
        "import sys\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
        "from model import BoolResNet8, parse_spec\n\n\n"
        "class BoolResNet8Variant(BoolResNet8):\n"
        f"    SPEC = {spec!r}\n\n"
        "    def __init__(self):\n"
        "        channels, blocks = parse_spec(self.SPEC)\n"
        "        super().__init__(1.0, 8, 10, channels, blocks)\n")
    return path + ":BoolResNet8Variant"


def initial_weights(definition, path):
    """the starting weights the CDT loads ([model] file): a fresh random initialization (the network is trained from
    scratch), saved once per definition"""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.manual_seed(0)
        torch.save(definition().state_dict(), path)
    return path


def load_model(path, width=1.0, levels=8, device="cpu"):
    """a trained BoolResNet8; the checkpoints of convul_symbex_v3/purga_symbex.py (same network, Spanish names:
    bloques, cambio) load too"""
    sd = torch.load(path, map_location="cpu")
    sd = {k.replace("bloques.", "blocks."): v for k, v in sd.items()}
    model = BoolResNet8(width, levels)
    model.load_state_dict(sd)
    return model.to(device).eval()


class _ToUint8:
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()


def cifar_loaders(batch=128, workers=4):
    import torchvision
    import torchvision.transforms as T
    train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=T.Compose(
        [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), _ToUint8()]))
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=_ToUint8())
    return (torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True,
                                        drop_last=True, persistent_workers=True),
            torch.utils.data.DataLoader(test, batch_size=512, num_workers=workers))


def cifar_u8(train=False):
    """CIFAR-10 as uint8 NHWC images + labels, in the official order (train=False: the 10,000 test images)"""
    import torchvision
    t = torchvision.datasets.CIFAR10(DATA_DIR, train=train, download=True)
    return np.asarray(t.data, dtype=np.uint8), np.asarray(t.targets)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    hits = total = 0
    for x, y in loader:
        hits += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        total += y.numel()
    return 100.0 * hits / total
