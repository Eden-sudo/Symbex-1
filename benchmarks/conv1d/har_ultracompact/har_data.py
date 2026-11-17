"""
har_data.py -- UniMiB-SHAR (AF-17: 17 activities and falls) as int8 windows, split by SUBJECT.

    python har_data.py                  # reads ../../data/unimib_shar/UniMiB-SHAR/data, writes data/har_unimib.npz

Run from benchmarks/conv1d/har_ultracompact/ (main environment). The data set (Micucci et al. 2017) comes from
http://www.sal.disco.unimib.it/technologies/unimib-shar/ : unzip UniMiB-SHAR.zip into ../../data/unimib_shar/.
    windows     acc_data.mat: 11,771 windows of 151 samples (x, y, z) around a peak above 1.5 g, in m/s^2, clipped
                by the phone at +-2 g (19.58 m/s^2)
    int8        x / 19.58 * 127, rounded: the input both libraries take (ultracompactBNN: int8; lib_symbex: uint8 =
                int8 + 128, the offset folds into the first thresholds)
    split       by subject (30 people), FIXED for every seed: 6 test subjects, 4 validation subjects, 20 training
                subjects, drawn once with numpy seed 0. The test people never appear in training, as a new user would.
                Daghero et al. do not state their split; the data set's own subject-wise folds hold out one person.
Writes data/har_unimib.npz: x (N, 151, 3) int8 time-major, y (N,) 0..16, subject (N,) 1..30, and the subject lists.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "data", "unimib_shar", "UniMiB-SHAR", "data"))
OUT = os.path.join(HERE, "data", "har_unimib.npz")
FULL_SCALE = 19.581081                     # the largest |value| in the data set: the phone's +-2 g range, in m/s^2
CLASSES = ["StandingUpFS", "StandingUpFL", "Walking", "Running", "GoingUpS", "Jumping", "GoingDownS", "LyingDownFS",
           "SittingDown", "FallingForw", "FallingRight", "FallingBack", "HittingObstacle", "FallingWithPS",
           "FallingBackSC", "Syncope", "FallingLeft"]
N_TEST, N_VAL = 6, 4
# the four networks (Table 1 of Daghero et al.) by size, as the other benchmarks name theirs (ECG: s / m / l): the
# README uses these names; the code and the file names keep Daghero's (min / max10 / max5 / max), and every --arch
# takes both (canonical_arch: "xs" -> "min", "l_w2" -> "max_w2")
NAMES = {"min": "xs", "max10": "s", "max5": "m", "max": "l"}


def canonical_arch(arch):
    base, sep, tag = arch.partition("_w")
    old = {v: k for k, v in NAMES.items()}.get(base, base)
    return old + sep + tag


def split_subjects():
    """Return the fixed test, validation and training subject arrays."""
    order = np.random.default_rng(0).permutation(np.arange(1, 31))
    return (
        np.sort(order[:N_TEST]),
        np.sort(order[N_TEST:N_TEST + N_VAL]),
        np.sort(order[N_TEST + N_VAL:]),
    )


def main():
    import scipy.io as sio
    acc = sio.loadmat(os.path.join(RAW, "acc_data.mat"))["acc_data"]           # N x 453: x(151) y(151) z(151)
    labels = sio.loadmat(os.path.join(RAW, "acc_labels.mat"))["acc_labels"]    # N x 3: class, subject, trial
    x = acc.reshape(len(acc), 3, 151).transpose(0, 2, 1)                       # N x 151 x 3, time-major
    x = np.clip(np.round(x / FULL_SCALE * 127), -127, 127).astype(np.int8)
    y = labels[:, 0].astype(np.int64) - 1
    subject = labels[:, 1].astype(np.int64)
    test_s, val_s, train_s = split_subjects()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, x=x, y=y, subject=subject, test_subjects=test_s, val_subjects=val_s, train_subjects=train_s)
    print(f"[*] {len(x)} windows of 151 x 3, int8 (full scale {FULL_SCALE} m/s^2 = 127) -> {OUT}")
    for name, s in (("train", train_s), ("validation", val_s), ("test", test_s)):
        m = np.isin(subject, s)
        print(f"    {name:<10} subjects {s.tolist()} : {m.sum():5d} windows, classes present {len(np.unique(y[m]))}/17")


def load(split):
    """(x int8 N x 151 x 3, y) of one split: 'train', 'validation' or 'test'"""
    d = np.load(OUT)
    key = dict(train="train_subjects", validation="val_subjects", test="test_subjects")[split]
    m = np.isin(d["subject"], d[key])
    return d["x"][m], d["y"][m]


def load_cv(fold, folds=5):
    """cross-validation inside the TRAINING subjects (README section 8): the 20 training subjects in `folds` groups of
    consecutive ids (fold k = the k-th group); returns ((x, y) of the other training subjects, (x, y) of fold k).
    The validation and test subjects are never in either part."""
    if not 0 <= fold < folds:                       # a negative index would keep the held-out group in training
        raise ValueError(f"cross-validation fold {fold}: expected 0..{folds - 1}")
    d = np.load(OUT)
    groups = np.array_split(np.sort(d["train_subjects"]), folds)
    held, rest = groups[fold], np.concatenate([g for i, g in enumerate(groups) if i != fold])
    m_rest, m_held = np.isin(d["subject"], rest), np.isin(d["subject"], held)
    return (d["x"][m_rest], d["y"][m_rest]), (d["x"][m_held], d["y"][m_held])


def augment(x, rng, degrees=20.0, scale=(0.8, 1.2)):
    """training augmentation (README section 10), the same for every method: each window turned by a random rotation
    (uniform axis, angle uniform in +-degrees: the phone sits a little differently in each pocket, not upside down) and
    multiplied by one random factor in `scale` (stronger or weaker movement), then rounded and clipped to int8 as the
    chip would read it. x: int8 N x 151 x 3; rng: numpy Generator. Returns int8 N x 151 x 3."""
    n = len(x)
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = np.radians(rng.uniform(-degrees, degrees, n))
    k = np.zeros((n, 3, 3))                                          # cross-product matrix of each axis
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    k -= k.transpose(0, 2, 1)
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    rot = np.eye(3) + s * k + (1 - c) * (k @ k)                      # Rodrigues
    rot *= rng.uniform(*scale, n)[:, None, None]
    out = np.einsum("ntj,nij->nti", x.astype(np.float64), rot)
    return np.clip(np.round(out), -128, 127).astype(np.int8)


if __name__ == "__main__":
    main()
