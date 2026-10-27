"""
model.py -- the FP32 model of the mlp_fashion benchmark, its training, and the data both sides share.

FashionMLP: 784 -> 256 -> 128 -> 10, hidden layers Linear -> BatchNorm1d -> ReLU, trained on Fashion-MNIST with
the input normalized as (x / 255 - 0.2860) / 0.3530 (the same values symbex.toml declares). It is the ORIGINAL
model both sides start from: SYMBEX (fashion_symbex.py, through the CDT) and the official TFLite INT8
(fashion_tflite.py). Buildable with no arguments, as the CDT requires.

Shared by both sides, so neither depends on the other having run first:
    get_fp32()          models/fashion_mlp.pth, trained here (and saved in models/) if it does not exist, on the
                        training images WITHOUT the validation split
    split_indices(n)    (train, validation) indices of the 60,000 training images: the CDT's split ([data]
                        validation_split, split_seed of symbex.toml), the same for every seed and every method
    validation_u8(n)    the first n Fashion-MNIST test images as raw uint8 (n x 784) + labels: the same images the
                        CDT validates and exports (format "fashionmnist" = the test set, in order)
    calibration_u8(n)   n training images (fixed seed) as raw uint8, for the INT8 calibration
All paths are relative to benchmarks/mlp_fashion/ (the scripts change to that folder first).
"""
import os
import tomllib

import torch
import torch.nn as nn

MEAN, STD = 0.2860, 0.3530
EPOCHS = 30
SEED = 42
MODELS_DIR = "models"                            # every model this benchmark generates goes here
FP32_PATH = os.path.join(MODELS_DIR, "fashion_mlp.pth")
DATA_DIR = os.path.join("..", "data", "fashionmnist")
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbex.toml")


class FashionMLP(nn.Module):
    """784 -> h1 -> h2 -> 10; the defaults (256, 128) are the benchmark model"""
    def __init__(self, h1=256, h2=128):
        super().__init__()
        self.hidden = (h1, h2)
        self.red = nn.Sequential(                # attribute name kept: the saved weights use the keys red.*
            nn.Flatten(),
            nn.Linear(784, h1, bias=False), nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, h2, bias=False), nn.BatchNorm1d(h2), nn.ReLU(),
            nn.Linear(h2, 10),
        )

    def forward(self, x):
        return self.red(x)


def variant(h1, h2):
    """FashionMLP with other hidden widths, as a class buildable with no arguments (what the CDT needs)"""
    return type(f"FashionMLP_{h1}_{h2}", (FashionMLP,), {"__init__": lambda self: FashionMLP.__init__(self, h1, h2)})


def __getattr__(name):
    """model.FashionMLP_<h1>_<h2> -> that variant (the CDT's definition = "model.py:FashionMLP_256_512")"""
    parts = name.split("_")
    if len(parts) == 3 and parts[0] == "FashionMLP" and parts[1].isdigit() and parts[2].isdigit():
        return variant(int(parts[1]), int(parts[2]))
    raise AttributeError(f"module 'model' has no attribute {name!r}")


def fp32_path(h1=256, h2=128):
    return FP32_PATH if (h1, h2) == (256, 128) else os.path.join(MODELS_DIR, f"fashion_mlp_{h1}_{h2}.pth")


def split_indices(n=60000):
    """(train, validation) indices of the n training images, exactly as the CDT splits them (tools/cdt.py, load_data:
    random_split with a generator seeded with [data] split_seed). Fixed: the same for every training seed."""
    from torch.utils.data import random_split
    d = tomllib.load(open(CONFIG, "rb"))["data"]
    seed = d.get("split_seed", -1)
    if seed < 0:
        raise SystemExit("[X] symbex.toml: [data] split_seed must be fixed (>= 0), so every seed shares one validation split")
    n_val = max(1, int(n * d["validation_split"]))
    train, val = random_split(range(n), [n - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    return list(train.indices), list(val.indices)


def train_fp32(path, data_dir, epochs=EPOCHS, h1=256, h2=128):
    """Trains FashionMLP in FP32 on the training images minus the validation split (split_indices) and saves its
    state_dict to `path`. Prints the validation accuracy per epoch; returns (validation, test) accuracy (%) of the
    last epoch, the model it saves (the test set is evaluated once, at the end)."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = T.Compose([T.ToTensor(), T.Normalize([MEAN], [STD])])
    full = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
    keep, held = split_indices(len(full))
    tr, va = Subset(full, keep), Subset(full, held)
    te = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    trl = DataLoader(tr, batch_size=256, shuffle=True)
    val = DataLoader(va, batch_size=1000)
    tel = DataLoader(te, batch_size=1000)

    def accuracy(loader, n):
        with torch.no_grad():
            return 100 * sum((model(x.to(dev)).argmax(1) == y.to(dev)).sum().item() for x, y in loader) / n
    model = FashionMLP(h1, h2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    acc_val = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in trl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
        model.eval()
        acc_val = accuracy(val, len(va))
        print(f"  FP32 epoch {ep + 1}/{epochs} | validation {acc_val:.2f}% ({len(va)} images, not trained on)")
    acc_test = accuracy(tel, len(te))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.cpu().state_dict(), path)
    return acc_val, acc_test


def get_fp32(h1=256, h2=128):
    """Path of the FP32 model of these widths (no arguments: the original); trains it (about a minute) and saves it
    in models/ if missing."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = fp32_path(h1, h2)
    if not os.path.exists(path):
        print(f"[*] {path} not found: training the FP32 model 784-{h1}-{h2}-10 ({EPOCHS} epochs, seed {SEED})")
        acc_val, acc_test = train_fp32(path, DATA_DIR, h1=h1, h2=h2)
        print(f"[*] saved {path} (FP32 validation {acc_val:.2f}%, test {acc_test:.2f}%)")
    return path


def _fashion(train):
    import torchvision
    return torchvision.datasets.FashionMNIST(DATA_DIR, train=train, download=True)


def validation_u8(n=2000):
    """First n test images as raw uint8 (n x 784) and their labels: the CDT's validation images, in order."""
    te = _fashion(False)
    return te.data.numpy()[:n].reshape(-1, 784), te.targets.numpy()[:n].astype("int64")


def calibration_u8(n=1000):
    """n training images (fixed seed, outside the validation split) as raw uint8 (n x 784), for INT8 calibration."""
    import numpy as np
    tr = _fashion(True)
    keep = np.array(split_indices(len(tr))[0])
    idx = keep[np.random.default_rng(SEED).permutation(len(keep))[:n]]
    return tr.data.numpy()[idx].reshape(-1, 784)
