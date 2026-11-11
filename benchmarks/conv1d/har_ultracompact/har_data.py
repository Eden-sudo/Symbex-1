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


def main():
    import scipy.io as sio
    acc = sio.loadmat(os.path.join(RAW, "acc_data.mat"))["acc_data"]           # N x 453: x(151) y(151) z(151)
    labels = sio.loadmat(os.path.join(RAW, "acc_labels.mat"))["acc_labels"]    # N x 3: class, subject, trial
    x = acc.reshape(len(acc), 3, 151).transpose(0, 2, 1)                       # N x 151 x 3, time-major
    x = np.clip(np.round(x / FULL_SCALE * 127), -127, 127).astype(np.int8)
    y = labels[:, 0].astype(np.int64) - 1
    subject = labels[:, 1].astype(np.int64)
    order = np.random.default_rng(0).permutation(np.arange(1, 31))
    test_s, val_s, train_s = np.sort(order[:N_TEST]), np.sort(order[N_TEST:N_TEST + N_VAL]), np.sort(order[N_TEST + N_VAL:])
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


if __name__ == "__main__":
    main()
