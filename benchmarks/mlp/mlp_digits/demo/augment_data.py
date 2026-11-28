"""
augment_data.py -- the Digits data for the touch demo's model: the same images and splits, plus augmented copies of
the TRAINING images that look like a drawing passed through touch_digits.py's reduction.

Each copy: the 8x8 image (ink 0-16 per block) is enlarged to 128x128, turned, stretched and sheared a little, cut
at a random ink level (thin or thick strokes), then reduced exactly as a drawing is (touch_digits.to_input: crop the
ink, fit a 32x32 box, 4x4 block counts). Validation and test images are the originals, untouched (masks val.npy /
test.npy over the whole array): every choice is still made on validation, the test is read once.

    python augment_data.py                  # -> ../../../data/digits_demo_aug/ (x, y, val, test .npy)
    python augment_data.py --copies 20
Run from any folder.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import touch_digits as T  # noqa: E402

SRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data", "digits"))
DST = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data", "digits_demo_aug"))
BIG = 128


def augment(img8, rng):
    """one 8x8 uint8 image (0-255) -> one augmented 64-byte input, as touch_digits.to_input gives for a drawing"""
    g = Image.fromarray((img8.astype(np.float32) / 255.0).reshape(8, 8), mode="F").resize((BIG, BIG), Image.BILINEAR)
    a = np.deg2rad(rng.uniform(-15, 15))
    sx, sy = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    sh = rng.uniform(-0.2, 0.2)
    m = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[sx, sh], [0, sy]])
    inv = np.linalg.inv(m)                                  # PIL maps output -> input pixels
    c = BIG / 2
    off = np.array([c, c]) - inv @ np.array([c, c])
    g = g.transform((BIG, BIG), Image.AFFINE, (inv[0, 0], inv[0, 1], off[0], inv[1, 0], inv[1, 1], off[1]),
                    resample=Image.BILINEAR)
    ink = np.asarray(g) >= rng.uniform(0.25, 0.65)          # thin or thick strokes
    if not ink.any():
        return img8.reshape(64)
    return T.to_input(Image.fromarray(ink.astype(np.uint8) * 255, mode="L"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--copies", type=int, default=10, help="augmented copies of every training image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.copies < 1:
        ap.error("--copies must be at least 1 (the copies are the point of this script)")
    T.THICKEN = 0                                           # the stroke width comes from the random ink level
    x = np.load(os.path.join(SRC, "x.npy"))
    y, val, test = (np.load(os.path.join(SRC, f)) for f in ("y.npy", "val.npy", "test.npy"))
    rng = np.random.default_rng(args.seed)
    tr = np.flatnonzero(~val & ~test)
    xa = np.stack([augment(x[i], rng) for i in tr for _ in range(args.copies)]).reshape(-1, 1, 8, 8)
    ya = np.repeat(y[tr], args.copies)
    os.makedirs(DST, exist_ok=True)
    np.save(os.path.join(DST, "x.npy"), np.concatenate([x, xa]).astype(np.uint8))
    np.save(os.path.join(DST, "y.npy"), np.concatenate([y, ya]).astype(np.int64))
    np.save(os.path.join(DST, "val.npy"), np.concatenate([val, np.zeros(len(ya), bool)]))
    np.save(os.path.join(DST, "test.npy"), np.concatenate([test, np.zeros(len(ya), bool)]))
    print(f"[*] {DST}: {len(tr)} training images + {len(ya)} augmented copies, {val.sum()} validation, "
          f"{test.sum()} test (originals)")


if __name__ == "__main__":
    main()
