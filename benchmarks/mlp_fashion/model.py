"""
model.py -- the FP32 model of the mlp_fashion benchmark, its training, and the data both sides share.

FashionMLP: 784 -> 256 -> 128 -> 10, hidden layers Linear -> BatchNorm1d -> ReLU, trained on Fashion-MNIST with
the input normalized as (x / 255 - 0.2860) / 0.3530 (the same values symbex.toml declares). It is the ORIGINAL
model both sides start from: SYMBEX (fashion_symbex.py, through the CDT) and the official TFLite INT8
(fashion_tflite.py). Buildable with no arguments, as the CDT requires.

Shared by both sides, so neither depends on the other having run first:
    get_fp32()          models/fashion_mlp.pth, trained here (and saved in models/) if it does not exist
    validation_u8(n)    the first n Fashion-MNIST test images as raw uint8 (n x 784) + labels: the same images the
                        CDT validates and exports (format "fashionmnist" = the test set, in order)
    calibration_u8(n)   n training images (fixed seed) as raw uint8, for the INT8 calibration
All paths are relative to benchmarks/mlp_fashion/ (the scripts change to that folder first).
"""
import os

import torch
import torch.nn as nn

MEAN, STD = 0.2860, 0.3530
EPOCHS = 5
SEED = 42
MODELS_DIR = "models"                            # every model this benchmark generates goes here
FP32_PATH = os.path.join(MODELS_DIR, "fashion_mlp.pth")
DATA_DIR = os.path.join("..", "data", "fashionmnist")


class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(                # attribute name kept: the saved weights use the keys red.*
            nn.Flatten(),
            nn.Linear(784, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.red(x)


def train_fp32(path, data_dir, epochs=EPOCHS):
    """Trains FashionMLP in FP32 and saves its state_dict to `path`. Returns the test accuracy (%)."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = T.Compose([T.ToTensor(), T.Normalize([MEAN], [STD])])
    tr = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
    te = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    trl = DataLoader(tr, batch_size=256, shuffle=True)
    tel = DataLoader(te, batch_size=1000)
    model = FashionMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    acc = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in trl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            correct = sum((model(x.to(dev)).argmax(1) == y.to(dev)).sum().item() for x, y in tel)
        acc = 100 * correct / len(te)
        print(f"  FP32 epoch {ep + 1}/{epochs} | test {acc:.2f}%")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.cpu().state_dict(), path)
    return acc


def get_fp32():
    """Path of the original FP32 model; trains it (about a minute) and saves it in models/ if missing."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(FP32_PATH):
        print(f"[*] {FP32_PATH} not found: training the original FP32 model ({EPOCHS} epochs, seed {SEED})")
        acc = train_fp32(FP32_PATH, DATA_DIR)
        print(f"[*] saved {FP32_PATH} (FP32 test accuracy {acc:.2f}%)")
    return FP32_PATH


def _fashion(train):
    import torchvision
    return torchvision.datasets.FashionMNIST(DATA_DIR, train=train, download=True)


def validation_u8(n=2000):
    """First n test images as raw uint8 (n x 784) and their labels: the CDT's validation images, in order."""
    te = _fashion(False)
    return te.data.numpy()[:n].reshape(-1, 784), te.targets.numpy()[:n].astype("int64")


def calibration_u8(n=1000):
    """n training images (fixed seed) as raw uint8 (n x 784), for INT8 calibration."""
    import numpy as np
    tr = _fashion(True)
    idx = np.random.default_rng(SEED).permutation(len(tr))[:n]
    return tr.data.numpy()[idx].reshape(-1, 784)
