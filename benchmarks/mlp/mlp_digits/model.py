"""
model.py -- the FP32 model of the mlp_digits benchmark, its data and its training.

The first model SYMBEX ever ran: an MLP on the scikit-learn Digits dataset (1797 images of 8x8 = 64 pixels, values
0-16). Here the pixels are scaled to 0-255 (uint8) so the board receives bytes, as from a sensor.

DigitsMLP: 64 -> 128 -> 10, hidden layer Linear -> BatchNorm1d -> ReLU. Input normalized as (x / 255 - 0.3060) /
0.3762 (mean and std of the pixels of the first training split, kept; the same values symbex.toml declares).
Splits, fixed for every seed and every method: test = 20% stratified (360 images, seed 42), touched once at the end;
validation = 20% stratified of the remaining 1437 (288 images, seed 42), where every choice is made; training = the
other 1149. The FP32 model is trained on the training images only.

    prepare_data()      ../../data/digits/x.npy (1797, 1, 8, 8) uint8, y.npy, val.npy (True = validation image),
                        test.npy (True = test image)
    get_fp32()          models/digits_mlp.pth, trained here (and saved in models/) if it does not exist
    validation_u8()     the 360 test images as raw uint8 (360 x 64) + labels: the images the CDT exports and the
                        board runs (the CDT validates on val.npy and reports test.npy once)
    calibration_u8(n)   n training images (fixed seed) as raw uint8, for an INT8 calibration
All paths are relative to benchmarks/mlp/mlp_digits/ (the scripts change to that folder first).
"""
import os

import numpy as np
import torch
import torch.nn as nn

MEAN, STD = 0.3060, 0.3762
EPOCHS = 100
SEED = 42
MODELS_DIR = "models"
FP32_PATH = os.path.join(MODELS_DIR, "digits_mlp.pth")
DATA_DIR = os.path.join("..", "..", "data", "digits")
DATA_FILES = ("x.npy", "y.npy", "val.npy", "test.npy")
VAL_FRACTION = 0.2                                   # of the images left after the test split


class DigitsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def prepare_threshold_data(threshold):
    """A copy of the dataset with every pixel binarized on the ORIGINAL 0-16 scale: pixel >= threshold -> 255, else 0
    (threshold 1 = any ink, the V2 rule). With [data] input_bits = 1 the network then sees exactly that bit.
    Returns its folder (../../data/digits_t<threshold>/)."""
    folder = DATA_DIR + f"_t{threshold}"
    if all(os.path.exists(os.path.join(folder, f)) for f in DATA_FILES):
        return folder
    base = prepare_data()
    x = np.load(os.path.join(base, "x.npy"))
    original = np.round(x.astype(np.float64) * 16 / 255)            # back to the 0-16 scale (exact)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "x.npy"), np.where(original >= threshold, 255, 0).astype(np.uint8))
    for f in ("y.npy", "val.npy", "test.npy"):
        np.save(os.path.join(folder, f), np.load(os.path.join(base, f)))
    print(f"[*] Digits binarized at pixel >= {threshold} (of 16) in {folder}")
    return folder


def prepare_data():
    """Writes the dataset in the CDT's npy format (once; a folder from before the validation split, without
    test.npy, is written again). Returns the data folder."""
    if all(os.path.exists(os.path.join(DATA_DIR, f)) for f in DATA_FILES):
        return DATA_DIR
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    x, y = load_digits(return_X_y=True)
    x8 = np.round(x * 255 / 16).astype(np.uint8).reshape(-1, 1, 8, 8)
    rest, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    _, val_idx = train_test_split(rest, test_size=VAL_FRACTION, random_state=SEED, stratify=y[rest])
    test, val = np.zeros(len(y), dtype=bool), np.zeros(len(y), dtype=bool)
    test[test_idx], val[val_idx] = True, True
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "x.npy"), x8)
    np.save(os.path.join(DATA_DIR, "y.npy"), y.astype(np.int64))
    np.save(os.path.join(DATA_DIR, "val.npy"), val)
    np.save(os.path.join(DATA_DIR, "test.npy"), test)
    print(f"[*] Digits prepared in {DATA_DIR}: {len(y) - val.sum() - test.sum()} training / {val.sum()} validation / "
          f"{test.sum()} test images (8x8, uint8)")
    return DATA_DIR


def _tensors():
    folder = prepare_data()
    x = np.load(os.path.join(folder, "x.npy"))
    y = np.load(os.path.join(folder, "y.npy"))
    val = np.load(os.path.join(folder, "val.npy"))
    test = np.load(os.path.join(folder, "test.npy"))
    return x, y, val, test


def train_fp32(path, epochs=EPOCHS):
    """Trains DigitsMLP in FP32 on the training images only and saves its state_dict to `path`. Returns the
    (validation, test) accuracy (%) of the last epoch, the model it saves (the test set is evaluated once, at the end)."""
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(SEED)
    x, y, val, test = _tensors()
    xt = (torch.from_numpy(x).float() / 255.0 - MEAN) / STD
    yt = torch.from_numpy(y)
    keep = torch.from_numpy(~val & ~test)
    val, test = torch.from_numpy(val), torch.from_numpy(test)
    trl = DataLoader(TensorDataset(xt[keep], yt[keep]), batch_size=64, shuffle=True)
    model = DigitsMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    acc = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb in trl:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
        if (ep + 1) % 20 == 0 or ep + 1 == epochs:
            model.eval()
            with torch.no_grad():
                acc = 100 * float((model(xt[val]).argmax(1) == yt[val]).float().mean())
            print(f"  FP32 epoch {ep + 1}/{epochs} | validation {acc:.2f}%")
    with torch.no_grad():
        acc_test = 100 * float((model(xt[test]).argmax(1) == yt[test]).float().mean())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    return acc, acc_test


def get_fp32():
    """Path of the original FP32 model; trains it (seconds) and saves it in models/ if missing."""
    prepare_data()
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(FP32_PATH):
        print(f"[*] {FP32_PATH} not found: training the original FP32 model ({EPOCHS} epochs, seed {SEED})")
        acc_val, acc_test = train_fp32(FP32_PATH)
        print(f"[*] saved {FP32_PATH} (FP32 validation {acc_val:.2f}%, test {acc_test:.2f}%)")
    return FP32_PATH


def validation_u8(n=None):
    """The first n (default all 360) test images as raw uint8 (n x 64) and their labels, in the order the CDT exports
    them to the board."""
    x, y, _, test = _tensors()
    return x[test].reshape(-1, 64)[:n], y[test].astype("int64")[:n]


def calibration_u8(n=500):
    """n training images (fixed seed) as raw uint8 (n x 64), for an INT8 calibration."""
    x, _, val, test = _tensors()
    tr = x[~val & ~test].reshape(-1, 64)
    return tr[np.random.default_rng(SEED).permutation(len(tr))[:n]]
