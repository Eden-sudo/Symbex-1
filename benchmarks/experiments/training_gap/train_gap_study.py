"""
train_gap_study.py -- how much of the binary ResNet-8's accuracy loss comes from the training, not from the bits?
Step A, the diagnostic: no new method, the benchmark's recipe as it is, three networks of the same graph:

    binary   cnn_cifar_resnet8's BoolResNet8x1 (thermometer input, 1-bit weights and activations, OR shortcuts)
    twin     the same graph with real weights and hardtanh activations: the smooth network the straight-through
             estimator pretends to train (sign forward, gradient 1 where |x| <= 1 is hardtanh's gradient)
    relu     real weights, ReLU activations, added shortcuts (a plain floating-point ResNet-8, same thermometer input)

    python train_gap_study.py --model binary --seed 42
    python train_gap_study.py --model all --seeds 42 43 44
    python train_gap_study.py --model binary --recipe clip bop twostep --seed 42

Same recipe for the three (the benchmark's, boolean.train): Adam lr 5e-3 + cosine, 150 epochs, batch 128, random crop
+ flip, BatchNorm momentum 0.3, labels only, the same starting weights (models/bool_resnet8_x1_init.pth) and the
CDT's split (45,000 training / 5,000 validation, carved out of the training data). The test set is never read
(only --test reads it, once, for a network already chosen on validation).
Every epoch it records:
    train_acc        accuracy on 5,000 fixed training images (no augmentation): train ~ validation and both low means
                     the network underfits (optimization or capacity), not that it overfits
    val_acc, loss    validation accuracy, mean training loss of the epoch
    ste_dead         share of the activations' inputs with |x| > 1 (the estimator passes no gradient there; twin:
                     hardtanh saturated; relu: only reported), on 1,000 fixed validation images
    binary only:
    flips_step       share of the 1-bit weights whose sign changed, summed over every step of the epoch
    flips_epoch      share whose sign differs from the previous epoch; flips_step / flips_epoch >> 1 means the signs
                     oscillate (flip and flip back) instead of settling
    near_zero        share of latent weights with |w| < the current learning rate (one Adam step from flipping)
    frozen           share of latent weights with |w| > 1: model.py's estimator gives them no gradient, they never
                     flip again
Step B, the recipes (binary network only, --recipe, one or several; each is chosen on validation, never on test):
    base             the benchmark's recipe (above)
    clip             latent weights clamped to [-1, 1] after every step (BinaryConnect): none freezes
    bop              Bop for the 1-bit weights (--bop-gamma, --bop-tau; Adam for the BatchNorms), no latent weights
    twostep          half the epochs with binary activations and real weights, then binary weights too, a fresh
                     Adam + cosine (Liu et al. 2021); same total epochs
    twostep_bop      twostep, with Bop for the 1-bit weights in the second half (they start from the signs of the first)
--distill: the loss is 0.9 * T^2 * KL(teacher || student at temperature T = 4) + 0.1 * cross-entropy, the teacher the relu
network of the same seed (trained first: its weights in runs/train_gap/relu_s<seed>.pth). Shown next to the labels-only
recipe, never instead of it. --epochs other than 150 and --distill add _e<epochs> / _kd to the file names.
--distill-loss mse | cosine (with --distill): cross-entropy + MSE between the raw logits (no softmax, Real-to-Bin /
ReActNet style), or cross-entropy + (1 - cosine) between the logit vectors (only their direction); files _kdmse / _kdcos.
--dead-fix (binary network, the dead-channel study: the stride-2 shortcuts' 1x1 convolutions were nearly constant in
most channels; none of these changes the network on a chip):
    or_grad          the OR of a block passes the whole gradient to both branches (torch.maximum gives it only to the
                     larger one)
    wstd             weight standardization (IR-Net): every conv's latent weights centered and scaled per output
                     channel before sign (the bits are sign(w - mean))
    dead_loss        + sum over the binary maps of mean_c relu(0.09 - p_c (1 - p_c)), p_c = mean of sigmoid(x) per
                     channel: only channels with < ~10 % or > ~90 % ones are pushed (no torch.compile with it)
    files _og / _ws / _dl
--density: train nothing; for the runs these flags name (already trained), the share of ones of every binary map's
    channels on 2,000 validation images and how many channels are nearly constant (< 10 % or > 90 % ones), written to
    results/train_gap/density_<run>.json (the binary network's maps in forward order: stem, then per block its 3x3
    convolution and, in a stride-2 block, the 1x1 shortcut convolution); per map also the channels nearly off (< 10 %:
    with the OR they let the shortcut through) and nearly on (> 90 %: they hide it), never / always on over the 2,000
    images, and how many of its BatchNorm scales are negative
--stem int8: the three networks with the int8 first layer on the pixels (cnn_cifar_resnet8's int8stem variant, its
starting weights); files <model>_i8_...
--width 2: twice the channels (32-64-128, BoolResNet8x2): if the binary network's training accuracy rises with width
and not with better recipes, the limit is the capacity of the 1-bit network, not its training. Files <model>_w2_...
--last-stage (binary network, x1): where the layer probes see the small network stall (its last stage), two ways to
spend the same extra binary operations (~2x those of the last stage, +24 % of the network's):
    wide         the last stage with 96 channels instead of 64 (16-32-96): more weights. Files binary_ls96_...
    act2         2-bit activations into the last stage: blocks 3-5 output a thermometer of two bits per channel,
                 a = (sign(z + d) + sign(z - d)) / 2 in {-1, 0, +1} (z the BatchNorm output, d >= 0 learned per channel:
                 two integer thresholds on a chip, two bit planes, every product still between bits); the OR shortcut
                 is the maximum of the two (per plane); no extra weights. Files binary_la2_...
--channels a-b-c[/x-y-z] (binary network, thermometer stem): any channels per stage and blocks per stage (default
2-2-2), from a fresh initialization saved once in runs/train_gap/; the sizing of the stages. Files
binary_c<a>-<b>-<c>[_b<x>-<y>-<z>]_... Every JSON records the network's binary operations (products of bits, counted
layer by layer, 2-bit maps twice), its weights, and its latency PREDICTED on the S3 and the ESP32 (estimates:
resnet8_sizing.py's count_bench cost curves, calibrated with the measured ResNet-8; the avg shortcut's additions are
not counted)
--shortcut2 avg (binary network): the stride-2 blocks' shortcut as in Bi-Real Net: out = sign(BN(conv(x)) + s * avg(x)),
the 2x2 average of the input bits (an integer 0..4 count on a chip), its channels repeated up to the block's outputs
(channel j reads input channel j mod C), s learned per channel; no 1x1 conv. On a chip: five integer thresholds per
channel, one per count (not in lib_symbex). The other blocks keep their OR shortcut. Files ..._sa_...
--shortcut2 pad (binary network): ResNet's option A in the stride-2 blocks: no 1x1 conv; the shortcut is the input
OR-pooled 2x2 (as the 1x1's input now), its channels padded up to the block's outputs with bit 0 (-1 in the +-1
maps: the neutral of OR and XOR), so out = conv OR x for the first C_in channels and the conv alone for the new ones.
Fewer weights and operations than the benchmark's shortcut. Files ..._sp_...
--shortcut2 addpool (binary network): avg's shortcut with the 2x2 OR-pool (one bit) instead of the 2x2 average (a
count): out = sign(BN(conv(x)) + s * orpool(x)); which of avg's two changes (the count, the addition) helps. On a chip:
two integer thresholds per channel. Files ..._sap_...
--shortcut-op xor (binary network): every block combines its conv and its shortcut with XOR instead of OR (in +-1:
-a*b): the conv can switch a shortcut bit off as well as on. PC only (lib_symbex fuses the OR). Files ..._sx_...
--phase2-lr x (twostep, twostep_bop): the learning rate of step 2's fresh Adam (default: --learning-rate). Files
..._p2lr<x>_...
--pad minus|plus: every binary conv pads its border with a -1 / +1 bit instead of leaving it out (0 in +-1, the
benchmark's). On a chip -1 is the window's 0 bits as they are: no per-pixel correction at the border. Files ..._pm_ / _pp_
--head int8|pool: the classifier with int8 weights (one scale per class, straight-through rounding) on the final bits
(int8), or on the count of ones of every channel of the final map (pool: C counts instead of C*H*W bits); the
classifier's BatchNorm stays. A sum of int8 weights per class on a chip. Files ..._hi8_ / _hpool_...
--head pool2|pool4: as pool, the count of ones per 2x2 / 4x4 window of every channel (x1: 64 x 4 x 4 / 64 x 2 x 2
inputs, 10 / 2.5 KB of int8 weights instead of int8's 40 KB). Files ..._hpool2_ / _hpool4_...
--stem int8 combines with --shortcut2 / --shortcut-op (the combined recipe: --stem int8 --recipe bop --shortcut2
addpool --head int8).
--recalibrate-bn: train nothing; for the runs these flags name (already trained), every BatchNorm's running mean and
    variance computed again with the weights frozen: reset, momentum None (a plain average, not the training's 0.3, which
    keeps mostly the last few batches), one pass over the 45,000 training images in train mode, batch 128, no gradient.
    Once on the images as they are (clean) and once with the training's crop + flip (augmented). Only the statistics
    change, so on a chip it is the same network with other integer thresholds. Validation accuracy before / after,
    results/train_gap/bnrecal_<run>.json.
--export: train nothing; the runs these flags name go through the CDT's boolean exporter (export/boolean.py: fold,
    integer simulation, header, the compiled C checked on this PC) into runs/train_gap/export/<run>_s<seed>, the folder
    tools/chip_test.py measures (only what lib_symbex runs: no added shortcuts, binary / pool2 / pool4 heads):
        python train_gap_study.py --model binary --stem int8 --recipe bop --head pool2 --export
--test: train nothing, choose nothing; the network these flags name, already chosen on validation, through the same
    exporter on the 10,000 CIFAR-10 test images (integer simulation, the compiled C checked on every image) into
    runs/train_gap/test/<run>_s<seed>; results/train_gap/test_<run>.json. The test set is used once: it refuses to
    run again if that file exists.
        python train_gap_study.py --model binary --stem int8 --recipe bop --head pool2 --seeds 42 43 44 --test
Results: results/train_gap/<model>_s<seed>.json (<model>_<recipe>_s<seed>.json for step B). Run from anywhere; env_bench (CUDA) recommended.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
BENCH = os.path.join(REPO, "benchmarks", "conv2d", "cnn_cifar_resnet8")
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(REPO, "tools"))
import model as resnet8                                    # noqa: E402
from compiler_core.train.boolean import _augment           # noqa: E402

MODELS = ("binary", "twin", "relu")
RECIPES = ("base", "clip", "bop", "twostep", "twostep_bop")
RUNS = os.path.join(HERE, "runs", "train_gap")              # the trained weights (gitignored): the teachers of --distill
OUT = os.path.join(HERE, "results", "train_gap")

_binary_sign = resnet8.sign
STEM = "thermometer"                                       # --stem: the first layer of the three networks
WIDTH = 1                                                  # --width: channels x1 (16-32-64) or x2 (32-64-128)
LAST = "x1"                                                # --last-stage: x1, wide (16-32-96), act2 (2-bit maps)
WIDE_LAST = 96                                             # wide: last-stage channels with ~act2's binary operations
CHANNELS = None                                            # --channels: (a, b, c) per stage, binary network only
BLOCKS = (2, 2, 2)                                         # --channels .../x-y-z: blocks per stage
SHORTCUT2 = "or"                                           # --shortcut2: or (the benchmark's), avg (Bi-Real), pad
SHORTCUT_OP = "or"                                         # --shortcut-op: or, xor (how a block adds its shortcut)
PHASE2_LR = 0.0                                            # --phase2-lr: step 2 of twostep (0: the learning rate)
PAD = "zero"                                               # --pad: zero (the benchmark's), minus, plus
HEAD = "binary"                                            # --head: binary (the benchmark's), int8, pool
_record = None                                             # list: the activations' inputs, while it is set
DEAD_FIX = ""                                              # --dead-fix: or_grad, wstd, dead_loss
_dead = []                                                 # dead_loss: the binary maps' inputs of the current step


def _sign_for(kind, real_weights=False):
    """what model.py's sign() becomes: the binary network keeps it; the floating-point ones keep real weights
    (parameters pass through) and replace the activation. real_weights: the binary network with real weights (the
    first step of the twostep recipe)"""
    def act(x):
        if isinstance(x, nn.Parameter):
            if kind == "binary" and DEAD_FIX == "wstd" and x.dim() == 4:
                x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-5)
            return _binary_sign(x) if kind == "binary" and not real_weights else x
        if kind == "binary" and DEAD_FIX == "dead_loss" and x.dim() == 4 and torch.is_grad_enabled():
            _dead.append(x)
        if _record is not None:
            _record.append(x.detach())
        if kind == "binary":
            return _binary_sign(x)
        return F.hardtanh(x) if kind == "twin" else F.relu(x)
    return act


class _OrBoth(torch.autograd.Function):
    """OR of two +-1 maps (their maximum) whose backward gives the whole gradient to both (--dead-fix or_grad)"""
    @staticmethod
    def forward(ctx, a, b):
        return torch.maximum(a, b)

    @staticmethod
    def backward(ctx, g):
        return g, g


class OrGradBlock(nn.Module):
    """a Block (OR shortcut) with _OrBoth's backward"""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        b = self.block
        return _OrBoth.apply(resnet8.sign(b.bn(b.conv(x))), b.shortcut(x))


def _dead_loss():
    loss = sum(F.relu(0.09 - (p := torch.sigmoid(x).mean(dim=(0, 2, 3))) * (1 - p)).mean() for x in _dead)
    _dead.clear()
    return loss


class Act2Block(nn.Module):
    """a Block (OR shortcut) whose output is a thermometer of two bits per channel (see --last-stage act2)"""
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.delta = nn.Parameter(torch.full((block.bn.num_features,), 0.5))

    def forward(self, x):
        b = self.block
        z = b.bn(b.conv(x))
        d = self.delta.abs().view(1, -1, 1, 1)
        a = (resnet8.sign(z + d) + resnet8.sign(z - d)) / 2
        return torch.maximum(a, b.shortcut(x))


class AvgShortcutBlock(nn.Module):
    """a stride-2 Block with Bi-Real's shortcut: sign(BN(conv(x)) + s * avg2x2(x)) (see --shortcut2 avg)"""
    def __init__(self, block, pool=F.avg_pool2d):
        super().__init__()
        self.block, self.pool = block, pool
        cin, cout = block.conv.in_channels, block.conv.out_channels
        self.register_buffer("index", torch.arange(cout) % cin)
        self.scale = nn.Parameter(torch.ones(cout))

    def forward(self, x):
        b = self.block
        s = self.pool(x, 2)[:, self.index] * self.scale.view(1, -1, 1, 1)
        return resnet8.sign(b.bn(b.conv(x)) + s)


def _combine(a, b):
    """a block's conv bits with its shortcut bits, both +-1: OR (their maximum) or XOR (-a*b, --shortcut-op)"""
    return -a * b if SHORTCUT_OP == "xor" else torch.maximum(a, b)


class PadShortcutBlock(nn.Module):
    """a stride-2 Block with ResNet's option A shortcut: OR-pool 2x2, channels padded with bit 0 (see --shortcut2 pad)"""
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.extra = block.conv.out_channels - block.conv.in_channels

    def forward(self, x):
        b = self.block
        s = F.max_pool2d(x, 2)
        s = F.pad(s, (0, 0, 0, 0, 0, self.extra), value=-1.0)    # bit 0 in +-1: neutral of OR and XOR
        return _combine(resnet8.sign(b.bn(b.conv(x))), s)


class XorBlock(nn.Module):
    """a Block whose conv and shortcut are combined by XOR (see --shortcut-op xor)"""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        b = self.block
        return _combine(resnet8.sign(b.bn(b.conv(x))), b.shortcut(x))


def build(kind):
    resnet8.sign = _sign_for(kind)                         # model.py reads its global sign() at every forward
    custom = CHANNELS or BLOCKS != (2, 2, 2)
    if LAST != "x1" or custom or SHORTCUT2 != "or" or SHORTCUT_OP != "or":
        if (kind != "binary" or WIDTH != 1 or (custom and LAST != "x1")
                or (STEM != "thermometer" and (LAST != "x1" or custom))):
            raise SystemExit("[X] --last-stage / --channels / --shortcut2: the binary network, width 1 only; "
                             "--last-stage not with --channels; --stem int8 only with --shortcut2 / --shortcut-op")
        if LAST == "wide" or custom:
            channels = CHANNELS or (tuple(resnet8.BASE_CHANNELS[:2]) + (WIDE_LAST,) if LAST == "wide"
                                    else tuple(resnet8.BASE_CHANNELS))
            definition = lambda: resnet8.BoolResNet8(1.0, 8, channels=channels, blocks=BLOCKS)     # noqa: E731
            net = definition()
            name = "c" + "-".join(map(str, channels)) + ("" if BLOCKS == (2, 2, 2) else
                                                         "_b" + "-".join(map(str, BLOCKS)))
            init = os.path.join(RUNS, f"bool_resnet8_{name}_init.pth")
            net.load_state_dict(torch.load(resnet8.initial_weights(definition, init), map_location="cpu"))
            return _with_pad_head(_with_shortcut2(net))
    net = resnet8.BoolResNet8(float(WIDTH), 8, stem=STEM, shortcut="add" if kind == "relu" else "or")
    if STEM == "int8" and WIDTH != 1:
        raise SystemExit("[X] --stem int8 with --width 2: the benchmark has no such variant")
    if STEM == "int8":                                     # the benchmark's starting weights of each variant
        cls, init = resnet8.BoolResNet8Int8Stem, "bool_resnet8_int8stem_init.pth"
    elif WIDTH == 2:
        cls, init = resnet8.BoolResNet8x2, "bool_resnet8_x2_init.pth"
    else:
        cls, init = resnet8.BoolResNet8x1, "bool_resnet8_x1_init.pth"
    init = os.path.join(BENCH, "models", init)
    net.load_state_dict(torch.load(resnet8.initial_weights(cls, init), map_location="cpu"))
    if LAST == "act2":                                     # blocks 3-5: block 3 feeds the last stage
        net.blocks = nn.Sequential(*[Act2Block(b) if i >= 3 else b for i, b in enumerate(net.blocks)])
    return _with_pad_head(_with_shortcut2(net))


def _int8_ste(w):
    """symmetric int8 weights per output row (scale = max |w| / 127), straight-through"""
    s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
    return w + (torch.round(w / s) * s - w).detach()


class Int8Head(nn.Linear):
    """--head int8: int8 weights on the final bits (a sum of int8 weights per class on a chip)"""
    def forward(self, x):
        return F.linear(x, _int8_ste(self.weight))


class PoolHead(nn.Module):
    """--head pool: the count of ones per channel of the final map (0..H*W), then int8 weights per class. Its input
    is the flattened map (N, C*H*W, channel-major, as BoolResNet8.forward gives it).
    window k (--head pool2 / pool4): the count of ones per k x k window instead (0..k*k), C*(H/k)*(W/k) inputs"""
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
        """what the CDT's boolean exporter folds (export/boolean.py, symbex_bool_pool_head): integers q (classes, inputs)
        in -127..127 and per-class scales s, the weights the forward uses = q * s"""
        w = self.fc.weight.detach()
        s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
        return torch.round(w / s), s.flatten()


def _padded_forward(conv, value):
    def forward(x):
        p = conv.padding
        x = F.pad(x, (p[1], p[1], p[0], p[0]), value=value)
        return F.conv2d(x, resnet8.sign(conv.weight), None, conv.stride, 0)
    return forward


def _with_pad_head(net):
    if PAD != "zero":                                  # every binary conv with padding: pad with a bit, -1 or +1
        for m in net.modules():
            if isinstance(m, resnet8.BinaryConv) and m.padding != (0, 0):
                m.forward = _padded_forward(m, -1.0 if PAD == "minus" else 1.0)
    if HEAD == "int8":
        head = Int8Head(net.fc.in_features, net.fc.out_features, bias=False)
        head.weight.data.copy_(net.fc.weight.data)     # the benchmark's starting weights of the classifier
        net.fc = head
    elif HEAD in ("pool", "pool2", "pool4"):
        side = math.isqrt(net.fc.in_features // net.channels[-1])   # the exporter reads it (pool: the whole map)
        net.fc = PoolHead(net.channels[-1], net.fc.out_features, int(HEAD[4:] or 0), side)
    return net


def _with_shortcut2(net):
    if SHORTCUT2 == "avg":
        net.blocks = nn.Sequential(*[AvgShortcutBlock(b) if getattr(b, "stride", 1) == 2 else b for b in net.blocks])
    if SHORTCUT2 == "addpool":                         # avg's block with the OR-pool (max of +-1) instead
        net.blocks = nn.Sequential(*[AvgShortcutBlock(b, F.max_pool2d) if getattr(b, "stride", 1) == 2 else b
                                     for b in net.blocks])
    if SHORTCUT2 == "pad":
        net.blocks = nn.Sequential(*[PadShortcutBlock(b) if getattr(b, "stride", 1) == 2 else b for b in net.blocks])
    if SHORTCUT_OP == "xor":
        if DEAD_FIX == "or_grad" or LAST == "act2":
            raise SystemExit("[X] --shortcut-op xor: not with --dead-fix or_grad or --last-stage act2")
        net.blocks = nn.Sequential(*[XorBlock(b) if isinstance(b, resnet8.Block) and not b.add else b
                                     for b in net.blocks])
    if DEAD_FIX == "or_grad":
        net.blocks = nn.Sequential(*[OrGradBlock(b) if isinstance(b, resnet8.Block) and not b.add else b
                                     for b in net.blocks])
    return net


def tag(kind):
    """the network's name in the file names: <kind> (thermometer stem, x1), <kind>_i8 (--stem int8), <kind>_w2"""
    last = {"x1": "", "wide": f"_ls{WIDE_LAST}", "act2": "_la2"}[LAST]
    if CHANNELS:
        last += "_c" + "-".join(map(str, CHANNELS))
    if BLOCKS != (2, 2, 2):
        last += ("" if CHANNELS else "_c" + "-".join(map(str, resnet8.BASE_CHANNELS))) + "_b" + "-".join(map(str, BLOCKS))
    if SHORTCUT2 == "avg":
        last += "_sa"
    if SHORTCUT2 == "pad":
        last += "_sp"
    if SHORTCUT2 == "addpool":
        last += "_sap"
    if SHORTCUT_OP == "xor":
        last += "_sx"
    last += {"zero": "", "minus": "_pm", "plus": "_pp"}[PAD] + {"binary": "", "int8": "_hi8", "pool": "_hpool", "pool2": "_hpool2", "pool4": "_hpool4"}[HEAD]
    if DEAD_FIX:
        last += {"or_grad": "_og", "wstd": "_ws", "dead_loss": "_dl"}[DEAD_FIX]
    return kind + ("_i8" if STEM == "int8" else "") + ("_w2" if WIDTH == 2 else "") + last


def load_split(seed):
    """the CDT's (train, validation) split of the CIFAR-10 training set, as the benchmark runs it ([data] defaults)"""
    import cdt
    cfg = {"data": dict(cdt.DEFAULTS["data"], format="cifar10", directory=os.path.join(REPO, "benchmarks", "data", "cifar10")),
           "training": {"seed": seed}}
    train, val, _ = cdt.load_data(cfg)
    return train, val


def _to_gpu(ds, dev):
    from torch.utils.data import DataLoader
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=1024, num_workers=4):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(dev), torch.cat(ys).to(dev)


@torch.no_grad()
def _accuracy(net, x, y, bs=1000):
    net.eval()
    hits = sum((net(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs))
    return 100.0 * hits / len(x)


@torch.no_grad()
def cost(net, layer_list=False):
    """(binary operations, weights) of one inference: every conv / linear output costs its fan-in, twice when its
    input is a 2-bit map (act2); the int8 stem counted as binary operations too (only to compare shapes)"""
    ops, handles, prev2 = [0], [], set()
    blocks = list(net.blocks)
    for i, b in enumerate(blocks):                         # a layer reads 2-bit maps when the block before it is Act2
        if i > 0 and isinstance(blocks[i - 1], Act2Block):
            inner = b.block if isinstance(b, Act2Block) else b
            prev2 |= {id(inner.conv)} | ({id(inner.ds_conv)} if inner.changes else set())
    if blocks and isinstance(blocks[-1], Act2Block):
        prev2.add(id(net.fc))

    ran = {}

    def hook(m, i, o):
        per = o[0].numel() if isinstance(m, nn.Conv2d) else o.shape[1]
        planes = 2 if id(m) in prev2 else 1
        ops[0] += per * m.weight[0].numel() * planes
        conv = isinstance(m, nn.Conv2d)
        ran[id(m)] = dict(k=m.kernel_size[0] if conv else 1, cin=(m.in_channels if conv else m.in_features) * planes,
                          cout=m.out_channels if conv else m.out_features, out_h=o.shape[-1] if conv else 1,
                          weights=m.weight.numel())
    layers = [m for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for m in layers:
        handles.append(m.register_forward_hook(hook))
    was = net.training
    net.eval()
    net(torch.zeros(1, 3, 32, 32, dtype=torch.uint8, device=next(net.parameters()).device))
    net.train(was)
    for h in handles:
        h.remove()
    if layer_list:
        return list(ran.values())
    return ops[0], sum(r["weights"] for r in ran.values())


def predicted_ms(net):
    """{chip: ms} PREDICTED for the standard engine (resnet8_sizing.py's method: count_bench cycles per output,
    one factor per chip from the measured ResNet-8); None where a profile or a measurement is missing"""
    try:
        import resnet8_sizing as sizing
        out = {}
        for chip, (_, _, measured) in sizing.CHIPS.items():
            r = json.load(open(os.path.join(BENCH, "results", measured)))
            ms = [s["latency_ms"] for s in r["seeds"] if s.get("latency_ms")]
            curve = sizing.cost_curve(chip)
            base = cost(resnet8.BoolResNet8(1.0, 8), layer_list=True)
            factor = (sum(ms) / len(ms)) / (sizing.raw_cycles(base, curve) / 240e3)
            out[chip] = round(factor * sizing.raw_cycles(cost(net, layer_list=True), curve) / 240e3, 2)
        return out
    except (OSError, SystemExit, KeyError, ImportError):
        return None


@torch.no_grad()
def _ste_dead(net, x):
    global _record
    net.eval()
    _record = []
    net(x)
    inputs, _record = _record, None
    dead = sum((t.abs() > 1).sum().item() for t in inputs)
    return dead / sum(t.numel() for t in inputs)


class Bop:
    """Bop (Helwegen et al. 2019) for the 1-bit weights: no latent magnitude, an exponential moving average m of the
    gradient; a weight flips when |m| > tau and m pushes against its sign. gamma follows the learning rate's cosine"""
    def __init__(self, weights, gamma, tau):
        self.weights, self.gamma, self.tau = weights, gamma, tau
        self.m = [torch.zeros_like(w) for w in weights]
        with torch.no_grad():
            for w in weights:
                w.copy_(torch.where(w >= 0, 1.0, -1.0))

    @torch.no_grad()
    def step(self, scale):
        g = self.gamma * scale
        for w, m in zip(self.weights, self.m):
            if w.grad is None:
                continue
            m.mul_(1 - g).add_(w.grad, alpha=g)
            flip = (m.abs() > self.tau) & ((m >= 0) == (w >= 0))
            w.masked_fill_(flip, 0).add_(torch.where(flip, -torch.sign(m), 0.0))   # flipped: -sign(m) = the opposite sign

    def zero_grad(self):
        for w in self.weights:
            w.grad = None


DISTILL_LOSS = "kl"    # --distill-loss: kl, mse, cosine


def recipe_name(recipe, gamma, tau, epochs=150, distill=False):
    name = recipe + (f"_g{gamma:g}_t{tau:g}" if recipe in ("bop", "twostep_bop") else "")
    name += f"_p2lr{PHASE2_LR:g}" if PHASE2_LR and recipe in ("twostep", "twostep_bop") else ""
    kd = "_kd" + ({"kl": "", "mse": "mse", "cosine": "cos"}[DISTILL_LOSS])
    return name + (f"_e{epochs}" if epochs != 150 else "") + (kd if distill else "")


def load_teacher(seed, dev):
    """the relu network of this seed, frozen, and a forward that runs it with its own activation (model.py's sign()
    is one global: swapped around the teacher's call, put back for the student)"""
    path = os.path.join(RUNS, f"{tag('relu')}_s{seed}.pth")
    if not os.path.exists(path):
        raise SystemExit(f"[X] --distill: {path} not found (train it first: --model relu --seed {seed}, same --stem)")
    student_sign = resnet8.sign
    teacher = build("relu")
    teacher.load_state_dict(torch.load(path, map_location="cpu"))
    teacher_sign = resnet8.sign
    resnet8.sign = student_sign
    teacher = teacher.to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    @torch.no_grad()
    def forward(x):
        saved = resnet8.sign
        resnet8.sign = teacher_sign
        try:
            return teacher(x)
        finally:
            resnet8.sign = saved
    return forward


def run(kind, seed, epochs, lr, bs, compile_mode, recipe="base", bop_gamma=1e-4, bop_tau=1e-8, distill=False):
    if kind != "binary" and (recipe != "base" or distill):
        raise SystemExit(f"[X] --recipe {recipe} / --distill: only for the binary network")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = load_split(seed)
    xs, ys = _to_gpu(train, dev)
    xv, yv = _to_gpu(val, dev)
    torch.manual_seed(seed)
    net = build(kind).to(dev)
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = 0.3
    probe = torch.randperm(len(xs), generator=torch.Generator().manual_seed(0))[:5000].to(dev)
    x_probe, y_probe = xs[probe], ys[probe]
    latent = [m.weight for m in net.modules() if isinstance(m, (resnet8.BinaryConv, resnet8.BinaryLinear))]
    n_latent = sum(w.numel() for w in latent)
    step_model = net
    if DEAD_FIX == "dead_loss":
        compile_mode = ""                                  # the loss reads maps from the forward (a side effect)
    if compile_mode and dev.type == "cuda":
        import torch._inductor.config as inductor_config
        inductor_config.compile_threads = min(4, os.cpu_count() or 1)
        step_model = torch.compile(net, mode=None if compile_mode == "default" else compile_mode)
    bop = Bop(latent, bop_gamma, bop_tau) if recipe == "bop" else None
    ids = {id(w) for w in latent}
    two_steps = recipe in ("twostep", "twostep_bop")
    phase_epochs = epochs // 2 if two_steps else epochs
    teacher = load_teacher(seed, dev) if distill else None

    def new_optimizer(n_epochs, rate=lr):
        params = [p for p in net.parameters() if id(p) not in ids] if bop else net.parameters()
        o = torch.optim.Adam(params, lr=rate)
        return o, torch.optim.lr_scheduler.CosineAnnealingLR(o, n_epochs)

    if two_steps:                                          # step 1: binary activations, real weights
        resnet8.sign = _sign_for(kind, real_weights=True)
    opt, sched = new_optimizer(phase_epochs)
    gen = torch.Generator(device=dev).manual_seed(seed)
    name = recipe_name(recipe, bop_gamma, bop_tau, epochs, distill)
    print(f"[{kind} {name} s{seed}] {epochs} epochs, Adam lr {lr} + cosine, batch {bs}, {len(xs)} train / {len(xv)} "
          f"validation | {dev}" + (f", compile {compile_mode}" if compile_mode and dev.type == "cuda" else ""))
    history, t0 = [], time.time()
    for ep in range(epochs):
        if two_steps and ep == phase_epochs:               # step 2: binary weights too, a fresh Adam + cosine
            resnet8.sign = _sign_for(kind)
            if recipe == "twostep_bop":                    # the 1-bit weights to Bop, from the signs of step 1
                bop = Bop(latent, bop_gamma, bop_tau)
            opt, sched = new_optimizer(epochs - phase_epochs, PHASE2_LR or lr)
        net.train()
        lr_now = opt.param_groups[0]["lr"]
        start = [w.detach() >= 0 for w in latent]
        prev = [s.clone() for s in start]
        flips_step = torch.zeros((), device=dev)
        loss_sum = torch.zeros((), device=dev)
        perm = torch.randperm(len(xs), device=dev, generator=gen)
        steps = 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            xa = _augment(xs[idx])
            out = step_model(xa)
            if teacher is None:
                loss = F.cross_entropy(out, ys[idx])
            elif DISTILL_LOSS == "mse":
                loss = F.cross_entropy(out, ys[idx]) + F.mse_loss(out, teacher(xa))
            elif DISTILL_LOSS == "cosine":
                loss = F.cross_entropy(out, ys[idx]) + (1 - F.cosine_similarity(out, teacher(xa), dim=1)).mean()
            else:
                kl = F.kl_div(F.log_softmax(out / 4, 1), F.log_softmax(teacher(xa) / 4, 1), log_target=True,
                              reduction="batchmean")
                loss = 0.9 * 16 * kl + 0.1 * F.cross_entropy(out, ys[idx])
            if _dead:
                loss = loss + _dead_loss()
            opt.zero_grad(set_to_none=True)
            if bop:
                bop.zero_grad()
            loss.backward()
            opt.step()
            if bop:
                bop.step(opt.param_groups[0]["lr"] / lr)
            elif recipe == "clip":                         # latent weights kept where the estimator passes gradient
                with torch.no_grad():
                    for w in latent:
                        w.clamp_(-1, 1)
            loss_sum += loss.detach()
            steps += 1
            if kind == "binary":
                with torch.no_grad():
                    for j, w in enumerate(latent):
                        now = w >= 0
                        flips_step += (now != prev[j]).sum()
                        prev[j] = now
        sched.step()
        row = dict(epoch=ep + 1, lr=lr_now, loss=loss_sum.item() / steps,
                   train_acc=_accuracy(net, x_probe, y_probe), val_acc=_accuracy(net, xv, yv),
                   ste_dead=_ste_dead(net, xv[:1000]))
        if kind == "binary":
            with torch.no_grad():
                row["flips_step"] = flips_step.item() / n_latent
                row["flips_epoch"] = sum(((w >= 0) != s).sum().item() for w, s in zip(latent, start)) / n_latent
                row["near_zero"] = sum((w.abs() < lr_now).sum().item() for w in latent) / n_latent
                row["frozen"] = sum((w.abs() > 1).sum().item() for w in latent) / n_latent
        history.append(row)
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            extra = (f" | flips/step {row['flips_step']:.4f} epoch {row['flips_epoch']:.4f} near0 {row['near_zero']:.3f}"
                     f" frozen {row['frozen']:.3f}"
                     if kind == "binary" else "")
            print(f"    epoch {ep + 1:3d} | loss {row['loss']:.3f} | train {row['train_acc']:.2f}% | validation "
                  f"{row['val_acc']:.2f}% | ste dead {row['ste_dead']:.3f}{extra} | {(time.time() - t0) / 60:.1f} min")
    last = history[-1]
    ops, weights = cost(net)
    info = dict(model=kind, stem=STEM, width=WIDTH, last_stage=LAST, channels=list(net.channels),
                blocks=list(net.blocks_per_stage), shortcut2=SHORTCUT2, shortcut_op=SHORTCUT_OP, pad=PAD, head=HEAD, dead_fix=DEAD_FIX or None, phase2_lr=PHASE2_LR or None, binary_ops=ops, weights=weights,
                predicted_ms=predicted_ms(net) if kind == "binary" else None, recipe=name, distill=distill, seed=seed, epochs=epochs, learning_rate=lr, batch_size=bs,
                n_train=len(xs),
                n_validation=len(xv), n_latent_binary=n_latent if kind == "binary" else None,
                train_acc=last["train_acc"], val_acc=last["val_acc"], history=history,
                minutes=round((time.time() - t0) / 60, 2))
    os.makedirs(OUT, exist_ok=True)
    stem = f"{tag(kind)}_s{seed}" if name == "base" else f"{tag(kind)}_{name}_s{seed}"
    path = os.path.join(OUT, stem + ".json")
    json.dump(info, open(path, "w"), indent=1)
    os.makedirs(RUNS, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, os.path.join(RUNS, stem + ".pth"))
    print(f"    -> {path} | train {last['train_acc']:.2f}% | validation {last['val_acc']:.2f}%")
    return info


def export(kind, seeds, name, test=False):
    """--export: the trained runs through the CDT's boolean exporter (fold, simulation, header, the C check on this PC)
    into runs/train_gap/export/<run>_s<seed>, the folder tools/chip_test.py measures. Train nothing.
    test=True (--test): the same on the 10,000 test images, into runs/train_gap/test/; results/train_gap/test_<run>.json"""
    from compiler_core.export import boolean as bexport
    unsupported = [what for what, bad in (
        (f"model {kind} (only binary runs on the chip)", kind != "binary"),
        (f"--pad {PAD} (the exporter pads with zeros)", PAD != "zero"),
        (f"--head {HEAD} (an int8 classifier without pooling; use binary, pool, pool2 or pool4)", HEAD == "int8"),
        (f"--shortcut2 {SHORTCUT2} (lib_symbex runs the OR pool + 1x1 shortcut)", SHORTCUT2 != "or"),
        (f"--shortcut-op {SHORTCUT_OP} (lib_symbex ORs the shortcut)", SHORTCUT_OP != "or"),
        (f"--last-stage {LAST} (2-bit maps are not in lib_symbex)", LAST == "act2"),
        (f"--dead-fix {DEAD_FIX} (it changes the weights' signs at every forward)", DEAD_FIX == "wstd"))
        if bad]
    if unsupported:                                        # the exporter would fold a different network: refuse
        raise SystemExit("[X] --export: not what the boolean exporter runs: " + "; ".join(unsupported))
    stem = tag(kind) if name == "base" else f"{tag(kind)}_{name}"
    lib = os.path.join(REPO, "Symbex1_PIO", "lib", "lib_symbex")
    test_json = os.path.join(OUT, f"test_{stem}.json")
    if test and os.path.exists(test_json):                 # the test set is used once
        raise SystemExit(f"[X] --test: {test_json} exists; the test set is used once")
    rows = []
    for seed in seeds:
        net = build(kind)
        net.load_state_dict(torch.load(os.path.join(RUNS, f"{stem}_s{seed}.pth"), map_location="cpu"))
        net.blocks = nn.Sequential(*[b.block if isinstance(b, OrGradBlock) else b    # or_grad: only the backward
                                     for b in net.blocks])
        out = os.path.join(RUNS, "test" if test else "export", f"{stem}_s{seed}")
        os.makedirs(out, exist_ok=True)
        print(f"[*] {'test' if test else 'export'} {stem}_s{seed} -> {out}")
        if test:
            import cdt
            cfg = {"data": dict(cdt.DEFAULTS["data"], format="cifar10",
                                directory=os.path.join(REPO, "benchmarks", "data", "cifar10")), "training": {"seed": seed}}
            images = cdt.load_data(cfg)[2]
            r = bexport.export_and_validate(net.cpu().eval(), images, out, lib, n_validation=len(images))
        else:
            r = bexport.export_and_validate(net.cpu().eval(), load_split(seed)[1], out, lib)
        rows.append((seed, r))
        json.dump({k: v for k, v in r.items() if not k.startswith("_")}, open(os.path.join(out, "export.json"), "w"),
                  indent=1)
    print("\nseed  PyTorch %  simulation %  fidelity %  C exact %  flash KB")
    for seed, r in rows:
        print(f"{seed:4d}  {r['accuracy_torch']:9.2f}  {r['accuracy_sim']:12.2f}  {r['fidelity']:10.2f}  "
              f"{r['exactness'] if r['exactness'] is not None else float('nan'):9.2f}  {r['flash_bytes'] / 1024:8.1f}")
    if test:
        acc = torch.tensor([r["accuracy_sim"] for _, r in rows], dtype=torch.float64)
        summary = dict(mean=round(acc.mean().item(), 3), std=round(acc.std().item(), 3) if len(acc) > 1 else 0.0)
        os.makedirs(OUT, exist_ok=True)
        json.dump(dict(run=stem, split="test", n_test=len(rows[0][1]["_y"]), accuracy_sim=summary,
                       seeds=[dict(seed=s, **{k: v for k, v in r.items() if not k.startswith("_")}) for s, r in rows]),
                  open(test_json, "w"), indent=1)
        print(f"    test (integer simulation = the chip): {summary['mean']:.2f} +- {summary['std']:.2f} %  -> {test_json}")


@torch.no_grad()
def density(kind, seeds, name):
    """--density: per-channel share of ones of the binary maps of the trained runs (see the docstring)"""
    global _record
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stem = tag(kind) if name == "base" else f"{tag(kind)}_{name}"
    per_seed = []
    for seed in seeds:
        net = build(kind)
        net.load_state_dict(torch.load(os.path.join(RUNS, f"{stem}_s{seed}.pth"), map_location="cpu"))
        net.to(dev).eval()
        xv, _ = _to_gpu(load_split(seed)[1], dev)
        _record = []
        net(xv[:2000])
        maps = [a for a in _record if a.dim() == 4]
        _record = None
        bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]   # the same order as the maps
        rows = []
        for i, a in enumerate(maps):
            d = (a >= 0).float().mean(dim=(0, 2, 3))
            r = dict(channels=len(d), mean_ones=round(float(d.mean()), 4),
                     nearly_constant=int(((d < 0.1) | (d > 0.9)).sum()),
                     nearly_off=int((d < 0.1).sum()), nearly_on=int((d > 0.9).sum()),
                     never_on=int((d == 0).sum()), always_on=int((d == 1).sum()))
            if len(bns) == len(maps):
                r["gamma_negative"] = int((bns[i].weight < 0).sum())
            rows.append(r)
        per_seed.append(dict(seed=seed, maps=rows))
        print(f"    {stem} s{seed}: " + "  ".join(f"L{i} {r['mean_ones']:.2f} ({r['nearly_constant']}/{r['channels']})"
                                               for i, r in enumerate(rows)))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"density_{stem}.json")
    json.dump(dict(run=stem, images=2000, seeds=per_seed), open(path, "w"), indent=1)
    print(f"    -> {path}")


@torch.no_grad()
def recalibrate_bn(kind, seeds, name):
    """--recalibrate-bn: the BatchNorm statistics of the trained runs computed again, weights frozen (docstring)"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stem = tag(kind) if name == "base" else f"{tag(kind)}_{name}"
    per_seed = []
    for seed in seeds:
        net = build(kind)
        net.load_state_dict(torch.load(os.path.join(RUNS, f"{stem}_s{seed}.pth"), map_location="cpu"))
        net.to(dev)
        train, val = load_split(seed)
        xs, _ = _to_gpu(train, dev)
        xv, yv = _to_gpu(val, dev)
        trained = copy.deepcopy(net.state_dict())
        row = dict(seed=seed, val_acc=_accuracy(net, xv, yv))
        for mode in ("clean", "augmented"):
            net.load_state_dict(trained)
            for m in net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.reset_running_stats()
                    m.momentum = None                      # a plain average over every batch
            torch.manual_seed(seed)
            order = torch.randperm(len(xs), device=dev)
            net.train()
            for i in range(0, len(xs), 128):
                xb = xs[order[i:i + 128]]
                net(_augment(xb) if mode == "augmented" else xb)
            row[f"val_acc_{mode}"] = _accuracy(net, xv, yv)
        per_seed.append(row)
        print(f"    {stem} s{seed}: validation {row['val_acc']:.2f}% -> clean {row['val_acc_clean']:.2f}% | "
              f"augmented {row['val_acc_augmented']:.2f}%")
    summary = {}
    for k in ("val_acc", "val_acc_clean", "val_acc_augmented"):
        v = torch.tensor([r[k] for r in per_seed], dtype=torch.float64)
        summary[k] = dict(mean=round(v.mean().item(), 3), std=round(v.std().item(), 3) if len(v) > 1 else 0.0)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"bnrecal_{stem}.json")
    json.dump(dict(run=stem, n_train=len(xs), n_validation=len(xv), batch_size=128, seeds=per_seed, summary=summary),
              open(path, "w"), indent=1)
    print("    " + "  ".join(f"{k} {s['mean']:.2f} +- {s['std']:.2f}" for k, s in summary.items()) + f"\n    -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", choices=MODELS + ("all",), default="all")
    ap.add_argument("--seeds", "--seed", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--learning-rate", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--compile", default="reduce-overhead", help="torch.compile mode ('' to disable)")
    ap.add_argument("--recipe", nargs="+", choices=RECIPES, default=["base"],
                    help="step B, binary network only (see the docstring)")
    ap.add_argument("--bop-gamma", type=float, default=1e-4)
    ap.add_argument("--bop-tau", type=float, default=1e-8)
    ap.add_argument("--distill", action="store_true", help="binary network: distill from the relu network of the seed")
    ap.add_argument("--distill-loss", choices=["kl", "mse", "cosine"], default="kl",
                    help="with --distill: KL at T = 4 (default), MSE on the raw logits, or 1 - cosine of the logits")
    ap.add_argument("--stem", choices=("thermometer", "int8"), default="thermometer",
                    help="first layer of the three networks: int8 = the benchmark's int8stem variant (files *_i8_*)")
    ap.add_argument("--width", type=int, choices=(1, 2), default=1,
                    help="2: twice the channels (BoolResNet8x2), the capacity check (files *_w2_*)")
    ap.add_argument("--last-stage", choices=("x1", "wide", "act2"), default="x1",
                    help="binary network: wide = 96 channels in the last stage, act2 = 2-bit maps into it (docstring)")
    ap.add_argument("--channels", default="",
                    help="binary network: channels per stage [/ blocks per stage], e.g. 24-48-96 or 16-48-96/1-2-2")
    ap.add_argument("--shortcut2", choices=("or", "avg", "pad", "addpool"), default="or",
                    help="binary network: avg = Bi-Real's average shortcut in the stride-2 blocks, pad = ResNet's "
                         "option A (no 1x1 conv), addpool = avg with the OR-pool (docstring)")
    ap.add_argument("--shortcut-op", choices=("or", "xor"), default="or",
                    help="binary network: how every block combines conv and shortcut (docstring)")
    ap.add_argument("--phase2-lr", type=float, default=0.0,
                    help="twostep recipes: learning rate of step 2 (default: --learning-rate)")
    ap.add_argument("--pad", choices=("zero", "minus", "plus"), default="zero",
                    help="every binary conv's border: zero = left out (the benchmark's), minus / plus = a -1 / +1 bit")
    ap.add_argument("--head", choices=("binary", "int8", "pool", "pool2", "pool4"), default="binary",
                    help="the classifier: binary weights (the benchmark's), int8 weights on the bits, or int8 weights "
                         "on the count of ones per channel (docstring)")
    ap.add_argument("--dead-fix", choices=("or_grad", "wstd", "dead_loss"), default="",
                    help="binary network: the dead-channel study (docstring)")
    ap.add_argument("--density", action="store_true",
                    help="train nothing: the share of ones per channel of the trained runs (docstring)")
    ap.add_argument("--export", action="store_true",
                    help="train nothing: the trained runs exported for the chip with the CDT's boolean exporter (docstring)")
    ap.add_argument("--test", action="store_true",
                    help="train nothing: the chosen network on the test set, once (docstring)")
    ap.add_argument("--recalibrate-bn", action="store_true",
                    help="train nothing: the trained runs' BatchNorm statistics computed again, weights frozen (docstring)")
    args = ap.parse_args()
    global STEM, WIDTH, LAST, CHANNELS, BLOCKS, SHORTCUT2, SHORTCUT_OP, PHASE2_LR, DISTILL_LOSS, DEAD_FIX, PAD, HEAD
    SHORTCUT_OP, PHASE2_LR, PAD, HEAD = args.shortcut_op, args.phase2_lr, args.pad, args.head
    STEM, WIDTH, LAST, SHORTCUT2 = args.stem, args.width, args.last_stage, args.shortcut2
    DISTILL_LOSS = args.distill_loss
    DEAD_FIX = args.dead_fix
    ch, _, bl = args.channels.partition("/")
    CHANNELS = tuple(int(c) for c in ch.split("-")) if ch else None
    BLOCKS = tuple(int(b) for b in bl.split("-")) if bl else (2, 2, 2)
    kinds = MODELS if args.model == "all" else (args.model,)
    if args.density:
        for kind in kinds:
            for recipe in (args.recipe if kind == "binary" else ["base"]):
                density(kind, args.seeds, recipe_name(recipe, args.bop_gamma, args.bop_tau, args.epochs,
                                                      args.distill and kind == "binary"))
        return
    if args.export or args.test:
        for kind in [k for k in kinds if k == "binary"] or kinds:       # --model all: the binary one; twin / relu refused
            for recipe in (args.recipe if kind == "binary" else ["base"]):
                export(kind, args.seeds, recipe_name(recipe, args.bop_gamma, args.bop_tau, args.epochs,
                                                     args.distill and kind == "binary"), test=args.test)
        return
    if args.recalibrate_bn:
        for kind in kinds:
            for recipe in (args.recipe if kind == "binary" else ["base"]):
                recalibrate_bn(kind, args.seeds, recipe_name(recipe, args.bop_gamma, args.bop_tau, args.epochs,
                                                             args.distill and kind == "binary"))
        return
    done = []
    for seed in args.seeds:
        for kind in kinds:
            for recipe in (args.recipe if kind == "binary" else ["base"]):
                done.append(run(kind, seed, args.epochs, args.learning_rate, args.batch_size, args.compile, recipe,
                                args.bop_gamma, args.bop_tau, args.distill and kind == "binary"))
    print("\nmodel   recipe                       seed  train %  validation %")
    for r in done:
        print(f"{r['model']:7s} {r['recipe']:28s} {r['seed']:4d}  {r['train_acc']:7.2f}  {r['val_acc']:12.2f}")


if __name__ == "__main__":
    main()
