"""
modelo.py -- the FP32 model of the mlp_fashion benchmark and its training.

FashionMLP: 784 -> 256 -> 128 -> 10, hidden layers Linear -> BatchNorm1d -> ReLU, trained on Fashion-MNIST
with the input normalized as (x / 255 - 0.2860) / 0.3530 (the same values symbex.toml declares).
It is the ORIGINAL model both pipelines start from: SYMBEX (fashion_symbex.py, through the CDT) and the
official TFLite INT8 (fashion_tflite.py). Buildable with no arguments, as the CDT requires.

Shared by both sides, so neither depends on the other having run first:
    obtener_fp32()      models/fashion_mlp.pth, trained here (and saved in models/) if it does not exist
    validacion_u8(n)    the first n Fashion-MNIST test images as raw uint8 (N x 784) + labels: the same images
                        the CDT validates and exports (format "fashionmnist" = the test set, in order)
    calibracion_u8(n)   n training images (fixed seed) as raw uint8, for the INT8 calibration
All paths are relative to benchmarks/mlp_fashion/ (the scripts change to that folder first).
"""
import os

import torch
import torch.nn as nn

MEDIA, DESV = 0.2860, 0.3530
EPOCAS = 5
SEMILLA = 42
CARPETA_MODELOS = "models"                       # every model this benchmark generates goes here
RUTA_FP32 = os.path.join(CARPETA_MODELOS, "fashion_mlp.pth")
RUTA_DATOS = os.path.join("..", "data", "fashionmnist")


class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.red(x)


def entrenar_fp32(ruta_pth, ruta_datos, epocas=EPOCAS):
    """Trains FashionMLP in FP32 and saves its state_dict to ruta_pth. Returns the test accuracy (%)."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    torch.manual_seed(SEMILLA)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = T.Compose([T.ToTensor(), T.Normalize([MEDIA], [DESV])])
    tr = torchvision.datasets.FashionMNIST(ruta_datos, train=True, download=True, transform=tf)
    te = torchvision.datasets.FashionMNIST(ruta_datos, train=False, download=True, transform=tf)
    trl = DataLoader(tr, batch_size=256, shuffle=True)
    tel = DataLoader(te, batch_size=1000)
    modelo = FashionMLP().to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    acc = 0.0
    for ep in range(epocas):
        modelo.train()
        for x, y in trl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            nn.functional.cross_entropy(modelo(x), y).backward()
            opt.step()
        modelo.eval()
        with torch.no_grad():
            bien = sum((modelo(x.to(dev)).argmax(1) == y.to(dev)).sum().item() for x, y in tel)
        acc = 100 * bien / len(te)
        print(f"  FP32 epoch {ep + 1}/{epocas} | test {acc:.2f}%")
    os.makedirs(os.path.dirname(ruta_pth) or ".", exist_ok=True)
    torch.save(modelo.cpu().state_dict(), ruta_pth)
    return acc


def obtener_fp32():
    """Path of the original FP32 model; trains it (CPU or GPU, about a minute) and saves it in models/ if missing."""
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    if not os.path.exists(RUTA_FP32):
        print(f"[*] {RUTA_FP32} not found: training the original FP32 model ({EPOCAS} epochs, seed {SEMILLA})")
        acc = entrenar_fp32(RUTA_FP32, RUTA_DATOS)
        print(f"[*] saved {RUTA_FP32} (FP32 test accuracy {acc:.2f}%)")
    return RUTA_FP32


def _fashion(train):
    import torchvision
    return torchvision.datasets.FashionMNIST(RUTA_DATOS, train=train, download=True)


def validacion_u8(n=2000):
    """First n test images as raw uint8 (n x 784) and their labels: the CDT's validation images, in order."""
    te = _fashion(False)
    x = te.data.numpy()[:n].reshape(-1, 784)
    y = te.targets.numpy()[:n].astype("int64")
    return x, y


def calibracion_u8(n=1000):
    """n training images (fixed seed) as raw uint8 (n x 784), for INT8 calibration."""
    import numpy as np
    tr = _fashion(True)
    idx = np.random.default_rng(SEMILLA).permutation(len(tr))[:n]
    return tr.data.numpy()[idx].reshape(-1, 784)
