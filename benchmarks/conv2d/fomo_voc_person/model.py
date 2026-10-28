"""
model.py -- the binary FOMO person detector of the fomo_voc_person benchmark, its data and its score.

The network is built from the SAME pieces as the binary ResNet-8 (../cnn_cifar_resnet8/model.py: thermometer input,
binary convolutions, OR shortcuts), shaped for detection:
    96 x 96 RGB -> thermometer -> stem 3x3, stride 2 (48 x 48) -> stages at 48, 24, 12 -> per-cell head
    head        a binary 1x1 convolution + BatchNorm: two numbers per cell (background, person); on the chip it is
                lib_symbex's per-cell head (symbex_bool_cell_head), the integer classifier applied at every cell
so the output is a 12 x 12 grid, one class per 8 x 8 pixel cell: "the center of a person falls here", or not.

    BoolFOMO(channels, blocks, levels)     the network (uint8 or [0, 1] images; output N x 2 x 12 x 12)
    RealFOMO(channels, blocks)             the same architecture in floating point (the reference)
    load(split)                            data/fomo_<split>.npz (fomo_data.py): images, grids, person centers
    split_indices(n, n_val)                (train, validation) indices: one fixed random split (SPLIT_SEED)
    held_out_centers(centers, owner, val)  the person centers of the validation images, re-indexed to them
    detections(grid)                       the person centers a grid of classes predicts (connected cells)
    f1_score(grids, centers, owner)        precision, recall and F1 against the true centers
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
def _shared():
    """the binary pieces of ../cnn_cifar_resnet8/model.py, loaded under their own name (this file is a model.py too)"""
    import importlib.util
    path = os.path.join(HERE, "..", "cnn_cifar_resnet8", "model.py")
    spec = importlib.util.spec_from_file_location("resnet8_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["resnet8_model"] = mod
    spec.loader.exec_module(mod)
    return mod


_r8 = _shared()
BinaryConv, Block, sign, thermometer = _r8.BinaryConv, _r8.Block, _r8.sign, _r8.thermometer

CELL, GRID, SIZE = 8, 12, 96
STRIDES = (1, 2, 2)                       # after the stride-2 stem: 48 -> 48 -> 24 -> 12
MATCH_DISTANCE = 1.5 * CELL               # a detection matches a person whose center is within 12 pixels
SPLIT_SEED = 42                           # one validation split for every model and seed (as [data] split_seed in the CDT)


class RealBlock(nn.Module):
    """the float version of a block (the --fp32 reference): conv, BatchNorm, ReLU, plus the shortcut (sum)"""
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.ds = None
        if stride != 1 or cin != cout:
            self.ds = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, 0, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)) + (self.ds(x) if self.ds is not None else x))


class RealFOMO(nn.Module):
    """the SAME architecture in floating point (8-bit pixels in, real weights and activations): the ceiling the binary
    detector is compared with, and later the INT8 reference"""
    def __init__(self, channels=(16, 32, 64), blocks=(1, 1, 1), classes=2):
        super().__init__()
        self.channels, self.blocks_per_stage = tuple(channels), tuple(blocks)
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, 2, 1, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU())
        layers, cin = [], channels[0]
        for c, s, n in zip(channels, STRIDES, blocks):
            for j in range(n):
                layers.append(RealBlock(cin, c, s if j == 0 else 1))
                cin = c
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Conv2d(cin, classes, 1)

    def forward(self, x):
        x = x.float() / 255.0 if not x.is_floating_point() else x
        return self.head(self.blocks(self.stem(x)))


class BoolFOMO(nn.Module):
    def __init__(self, channels=(16, 32, 64), blocks=(1, 1, 1), levels=8, classes=2):
        super().__init__()
        self.levels, self.channels, self.blocks_per_stage = levels, tuple(channels), tuple(blocks)
        self.width = 1.0
        self.stem = BinaryConv(3 * levels, channels[0], 3, 2, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        layers, cin = [], channels[0]
        for c, s, n in zip(channels, STRIDES, blocks):
            for j in range(n):
                layers.append(Block(cin, c, s if j == 0 else 1))
                cin = c
        self.blocks = nn.Sequential(*layers)
        self.head = BinaryConv(cin, classes, 1, 1, 0, bias=False)
        self.head_bn = nn.BatchNorm2d(classes)

    def forward(self, x):
        if x.is_floating_point():                    # the CDT gives images in [0, 1]
            x = torch.round(x * 255)
        x = sign(self.stem_bn(self.stem(thermometer(x, self.levels))))
        x = self.blocks(x)
        return self.head_bn(self.head(x))            # N x classes x 12 x 12 (logits per cell)


def load(split):
    d = np.load(os.path.join(HERE, "data", f"fomo_{split}.npz"))
    return d["x"], d["y"], d["centers"], d["image_of_center"]


def split_indices(n, n_val=1000):
    """(train, validation) indices of the n training images: random_split with its own generator seeded with
    SPLIT_SEED, as ../../mlp/mlp_fashion/model.py splits Fashion-MNIST. Fixed: the same for every training seed and for
    both columns (binary and INT8), whatever torch.manual_seed was set to."""
    from torch.utils.data import random_split
    train, val = random_split(range(n), [n - n_val, n_val], generator=torch.Generator().manual_seed(SPLIT_SEED))
    return np.array(train.indices), np.array(val.indices)


def held_out_centers(centers, owner, val):
    """the person centers of the images `val` (indices into the training set), with their owner re-indexed to the
    position inside `val`: what f1_score expects for the grids of x[val]"""
    pos = np.full(max(int(owner.max()) + 1 if len(owner) else 0, int(val.max()) + 1), -1)
    pos[val] = np.arange(len(val))
    keep = pos[owner] >= 0
    return centers[keep], pos[owner[keep]]


def detections(grid):
    """centers (x, y) in pixels of the groups of 8-connected positive cells of one grid"""
    pos = np.argwhere(grid > 0)
    seen, out = set(), []
    for start in map(tuple, pos):
        if start in seen:
            continue
        stack, cells = [start], []
        seen.add(start)
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q not in seen and 0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1] and grid[q] > 0:
                        seen.add(q)
                        stack.append(q)
        rc = np.mean(cells, axis=0)
        out.append(((rc[1] + 0.5) * CELL, (rc[0] + 0.5) * CELL))
    return out


def f1_score(grids, centers, owner):
    """greedy matching per image: each detection takes the nearest unmatched person within MATCH_DISTANCE"""
    tp = fp = fn = 0
    by_image = {}
    for (cx, cy), i in zip(centers, owner):
        by_image.setdefault(int(i), []).append((float(cx), float(cy)))
    for i, g in enumerate(grids):
        truth = list(by_image.get(i, []))
        used = [False] * len(truth)
        for dx, dy in detections(g):
            best, bd = -1, MATCH_DISTANCE
            for k, (tx, ty) in enumerate(truth):
                d = ((dx - tx) ** 2 + (dy - ty) ** 2) ** 0.5
                if not used[k] and d <= bd:
                    best, bd = k, d
            if best >= 0:
                used[best] = True
                tp += 1
            else:
                fp += 1
        fn += used.count(False)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=100 * p, recall=100 * r, f1=100 * (2 * p * r / (p + r) if p + r else 0.0), tp=tp, fp=fp, fn=fn)
