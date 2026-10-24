"""
model.py -- the ORIGINAL model of the cnn_cifar_resnet20 benchmark, and everything the scripts here share.

The model: a ResNet-20 for CIFAR-10 trained by someone else and published (chenyaofo/pytorch-cifar-models, ~92.6% on
the CIFAR-10 test set). We do not train it: the benchmark binarizes a model as a user would bring it, and measures
how much of it the binary version recovers.

    ResNet20            the architecture, buildable with no arguments (as the CDT requires); the code comes from the
                        published repository through torch.hub (downloaded once, then cached)
    get_pretrained()    models/resnet20_cifar10.pth: the published weights, downloaded and saved once
    CLASSES             the 10 CIFAR-10 class names
    MEAN, STD           the input normalization, read from symbex.toml (the published repository's values)
    cifar_u8(train)     CIFAR-10 as uint8 NHWC images + labels, straight from the official batches (no torchvision)
    cifar_loaders(...)  PyTorch loaders (normalized; flips and crops for training)
    run_dir(seed, distill)   where the CDT leaves each trained seed: runs/symbex_s<seed>[_nd]
Paths are relative to benchmarks/cnn_cifar_resnet20/ (every script changes to it first).
"""
import os
import pickle
import tomllib

import numpy as np
import torch
import torch.nn as nn

REPO = "chenyaofo/pytorch-cifar-models"
ENTRY = "cifar10_resnet20"
MODELS_DIR = "models"
PRETRAINED_PATH = os.path.join(MODELS_DIR, "resnet20_cifar10.pth")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
DATA_DIR = os.path.join("..", "data", "cifar10")

_HERE = os.path.dirname(os.path.abspath(__file__))
_cfg = tomllib.load(open(os.path.join(_HERE, "symbex.toml"), "rb"))
MEAN, STD = tuple(_cfg["data"]["mean"]), tuple(_cfg["data"]["std"])


class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.hub.load(REPO, ENTRY, pretrained=False, trust_repo=True, verbose=False)

    def forward(self, x):
        return self.net(x)


def get_pretrained():
    """Path of the published model; downloads its weights once (needs internet the first time)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(PRETRAINED_PATH):
        print(f"[*] downloading the published {ENTRY} ({REPO}) ...")
        published = torch.hub.load(REPO, ENTRY, pretrained=True, trust_repo=True, verbose=False)
        model = ResNet20()
        model.net.load_state_dict(published.state_dict())
        torch.save(model.state_dict(), PRETRAINED_PATH)
        print(f"[*] saved {PRETRAINED_PATH}")
    return PRETRAINED_PATH


def run_dir(seed, distill=True):
    return os.path.join("runs", f"symbex_s{seed}{'' if distill else '_nd'}")


def cifar_u8(train):
    """(images uint8 N x 32 x 32 x 3, labels) in the official order (the test batch: the 10,000 test images)."""
    batches = os.path.join(DATA_DIR, "cifar-10-batches-py")
    if not os.path.isdir(batches):
        cifar_loaders()                                  # downloads the dataset through torchvision
    files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
    x, y = [], []
    for f in files:
        d = pickle.load(open(os.path.join(batches, f), "rb"), encoding="bytes")
        x.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        y += d[b"labels"]
    return np.concatenate(x), np.array(y)


def cifar_loaders(batch=256, workers=4):
    """(train loader with flips and crops, test loader), normalized with MEAN / STD"""
    import torchvision
    import torchvision.transforms as T
    norm = T.Normalize(MEAN, STD)
    train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=T.Compose(
        [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm]))
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=T.Compose([T.ToTensor(), norm]))
    return (torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=workers, persistent_workers=True),
            torch.utils.data.DataLoader(test, batch_size=1000))
