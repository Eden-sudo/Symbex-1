"""
ecg_data.py -- MIT-BIH Arrhythmia beats as int8 windows, AAMI classes, the inter-patient split of de Chazal et al.

    python ecg_data.py                  # reads ../../data/mitbih, writes data/ecg_mitbih.npz

Run from benchmarks/conv1d/ecg_mitbih/ (main environment). The data set (Moody & Mark 2001, PhysioNet) comes from
https://physionet.org/content/mitdb/1.0.0/ : the 48 records (.hea, .dat, .atr) in ../../data/mitbih/.
    reader      format 212 (two 12-bit samples in 3 bytes) and the MIT annotation format, read here with numpy only;
                every record is checked against the checksum and the length in its header
    lead        MLII (found by its label: record 114 has it on the second channel), 360 Hz averaged by pairs to 180 Hz
    beats       one window per annotated beat, from the reference R position: WINDOW samples (1.42 s), PRE before the
                R (the previous beat is visible: the RR interval is in the window), the window median subtracted
    int8        mV / FULL_SCALE_MV * 127, rounded and clipped (4 mV covers 99.9 % of the training peaks)
    classes     AAMI EC57: N (normal, bundle branch blocks, escapes), S (supraventricular ectopic), V (ventricular
                ectopic), F (fusion); Q (paced, unclassifiable) is dropped, as are the 4 paced records 102 104 107 217
    split       inter-patient (de Chazal et al. 2004): DS1 (22 records) to train, DS2 (22 records) is the TEST set, used
                once. Validation: VAL_RECORDS records of DS1, drawn with numpy seed 0 among the draws that hold at least
                MIN_VAL beats of S, V and F, never records 208 and 209 (most of DS1's fusion and supraventricular
                beats: they stay in training); the rest of DS1 trains.
    folds       patient-wise cross-validation inside DS1 (to choose a recipe with less noise than one validation
                split): CV_FOLDS groups of whole records (cv_folds: numpy seed 0, the draw whose S and V counts are the
                most even, every fold with S, V and F beats); load(split, fold=k) trains on the other folds and
                validates on fold k. The test records are never in a fold.
    augment     augment(x, rng): random amplitude, baseline wander, noise and a small shift of the int8 beats, drawn
                again every epoch, the same function for every method (ecg_train.py / ecg_tflite.py --augment)
    rhythm      rr (N, 3) in seconds: the RR interval before the beat, after it, and the local mean of the RR_LOCAL
                previous intervals (every annotated beat counts, paced and unclassifiable too; only the past, so a
                device has it one beat late). load(..., rr=True) adds them as RR_CHANNELS constant int8 input channels
                (rr_channels: pre / local, post / local, local, RR_SCALE = 1.0): same network, same lib_symbex kernels
Writes data/ecg_mitbih.npz: x (N, WINDOW, 1) int8 time-major, y (N,) 0..3, record (N,), rr (N, 3), and the record
lists.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "data", "mitbih"))
OUT = os.path.join(HERE, "data", "ecg_mitbih.npz")
CLASSES = ["N", "S", "V", "F"]
AAMI = {1: 0, 2: 0, 3: 0, 34: 0, 11: 0,     # N: normal, left / right bundle branch block, nodal / ventricular escape
        8: 1, 4: 1, 7: 1, 9: 1,             # S: aberrated APC, atrial premature, nodal premature, supraventricular premature
        5: 2, 10: 2,                        # V: premature ventricular contraction, ventricular escape
        6: 3}                               # F: fusion of ventricular and normal
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
WINDOW, PRE = 256, 160                      # samples at 180 Hz: R at index PRE
FULL_SCALE_MV = 4.0
VAL_RECORDS, MIN_VAL = 5, (50, 50, 10)      # validation records; at least this many S, V, F beats in them
CV_FOLDS, CV_MIN = 4, (50, 50, 3)           # cross-validation folds of DS1; at least this many S, V, F beats in each
# augment(): amplitude x U(1 - AUG_SCALE, 1 + AUG_SCALE), a baseline sine of AUG_WANDER_HZ with an amplitude up to
# AUG_WANDER LSB, Gaussian noise of AUG_NOISE LSB, a shift of up to AUG_SHIFT samples (1 LSB = 4 mV / 127 = 31 uV)
AUG_SCALE, AUG_WANDER_HZ, AUG_WANDER, AUG_NOISE, AUG_SHIFT = 0.2, (0.1, 0.5), 10.0, 2.0, 3
FS = 180
BEAT_CODES = set(AAMI) | {12, 13, 38}       # every beat annotation (+ paced, unclassifiable, pacemaker fusion): the RR
RR_LOCAL, RR_SCALE, RR_CHANNELS = 10, 64, 3 # intervals in the local mean; int8 per 1.0 (ratio or second); channels
# the networks, in the format of har_ultracompact/model.ARCHS (read by ecg_train.py and ecg_tflite.py): sized for
# binary, three widths of the same shape (lengths 256 -> 248 -> 240 -> pool 60 -> 52 -> pool 13)
ARCHS = {
    "s": [("conv", 8, 9), ("conv", 16, 9), ("pool", 4), ("conv", 16, 9), ("pool", 4)],
    "m": [("conv", 16, 9), ("conv", 32, 9), ("pool", 4), ("conv", 32, 9), ("pool", 4)],
    "l": [("conv", 32, 9), ("conv", 64, 9), ("pool", 4), ("conv", 64, 9), ("pool", 4)],
}


def read_header(path):
    """(samples, [(label, gain, checksum)] per channel)"""
    with open(path + ".hea") as f:
        lines = [l.split() for l in f if l.strip() and not l.startswith("#")]
    chans = [(s[8] if len(s) > 8 else "", float(s[2].split("(")[0].split("/")[0]), int(s[6])) for s in lines[1:]]
    return int(lines[0][3]), chans


def read_signal(path):
    """format 212: (samples, 2) int, checked against the header"""
    n, chans = read_header(path)
    b = np.fromfile(path + ".dat", dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    x = np.stack([b[:, 0] | ((b[:, 1] & 0x0F) << 8), b[:, 2] | ((b[:, 1] & 0xF0) << 4)], 1)
    x[x > 2047] -= 4096
    for c, (_, _, chk) in enumerate(chans):
        if len(x) != n or (int(x[:, c].sum()) + 32768) % 65536 - 32768 != chk:
            raise ValueError(f"{path}: checksum or length does not match the header (channel {c})")
    return x


def read_annotations(path):
    """MIT format: [(sample, code)] of the beat and non-beat annotations, in time order"""
    w = np.fromfile(path + ".atr", dtype="<u2").astype(np.int64)
    t, i, out = 0, 0, []
    while i < len(w):
        a, v = w[i] >> 10, w[i] & 1023
        i += 1
        if a == 0 and v == 0:
            break
        if a == 59:                         # SKIP: a 32-bit interval in the next two words
            t += (w[i] << 16) | w[i + 1]
            i += 2
        elif a == 63:                       # AUX: v bytes of text
            i += (v + 1) // 2
        elif a not in (60, 61, 62):         # NUM, SUB, CHN: no time
            t += v
            out.append((t, a))
    return out


def beats(record):
    """(x int8 (n, WINDOW, 1), y, rr (n, 3) seconds: before, after, local mean) of one record's N / S / V / F beats"""
    path = os.path.join(RAW, str(record))
    _, chans = read_header(path)
    lead = [c[0] for c in chans].index("MLII")
    sig = read_signal(path)[:, lead].astype(np.float64) / chans[lead][1]          # mV, 360 Hz
    sig = sig[:len(sig) // 2 * 2].reshape(-1, 2).mean(1)                          # 180 Hz
    ann = read_annotations(path)
    rt = np.array([t / 360 for t, code in ann if code in BEAT_CODES])              # every beat's R, seconds
    rr_all = np.diff(rt)
    xs, ys, rrs = [], [], []
    for t, code in ann:
        r = t // 2
        if code not in AAMI or r - PRE < 0 or r - PRE + WINDOW > len(sig):
            continue
        k = int(np.searchsorted(rt, t / 360))                                      # this beat in rt
        pre = rr_all[k - 1] if k > 0 else rr_all[0]                                # first / last beat of a record:
        post = rr_all[k] if k < len(rr_all) else rr_all[-1]                        # the nearest interval
        w = sig[r - PRE:r - PRE + WINDOW]
        xs.append(w - np.median(w))
        ys.append(AAMI[code])
        rrs.append((pre, post, rr_all[max(0, k - RR_LOCAL):max(k, 1)].mean()))
    x = np.clip(np.round(np.array(xs) / FULL_SCALE_MV * 127), -127, 127).astype(np.int8)
    return x[:, :, None], np.array(ys, dtype=np.int64), np.array(rrs)


def pick_validation(counts):
    """VAL_RECORDS records of DS1 (not 208, 209) with numpy seed 0: the first draw that holds MIN_VAL beats of S, V, F"""
    rng = np.random.default_rng(0)
    pool = [r for r in DS1 if r not in (208, 209)]
    for _ in range(10000):
        val = sorted(rng.choice(pool, VAL_RECORDS, replace=False).tolist())
        have = np.sum([counts[r] for r in val], axis=0)
        if all(have[c + 1] >= MIN_VAL[c] for c in range(3)):
            return val
    raise RuntimeError("no validation draw holds enough S, V and F beats")


def main():
    data, counts = {}, {}
    for r in DS1 + DS2:
        data[r] = beats(r)
        counts[r] = np.bincount(data[r][1], minlength=4)
    val = pick_validation(counts)
    train = [r for r in DS1 if r not in val]
    x = np.concatenate([data[r][0] for r in DS1 + DS2])
    y = np.concatenate([data[r][1] for r in DS1 + DS2])
    record = np.concatenate([np.full(len(data[r][1]), r) for r in DS1 + DS2])
    rr = np.concatenate([data[r][2] for r in DS1 + DS2])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp.npz"                                  # written aside, then renamed: running jobs read OUT
    np.savez_compressed(tmp, x=x, y=y, record=record, rr=rr, train_records=train, val_records=val, test_records=DS2)
    os.replace(tmp, OUT)
    print(f"[*] {len(x)} beats of {WINDOW} x 1, int8 ({FULL_SCALE_MV} mV = 127, 180 Hz) -> {OUT}")
    print(f"    {'split':<11}{'records':>8}{'N':>8}{'S':>8}{'V':>8}{'F':>8}")
    for name, rs in (("train", train), ("validation", val), ("test", DS2)):
        c = np.sum([counts[r] for r in rs], axis=0)
        print(f"    {name:<11}{len(rs):>8}" + "".join(f"{v:>8}" for v in c))
    print(f"    validation records: {val}")
    for k, f in enumerate(cv_folds(record, y)):
        c = np.sum([counts[r] for r in f], axis=0)
        print(f"    fold {k}: {f}  N S V F = {c.tolist()}")


def class_weights(y):
    """inverse class frequency, mean 1 over the beats: the weights of the cross-entropy (the same for every method)"""
    c = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    return len(y) / (len(CLASSES) * np.maximum(c, 1))


def scores(y, p):
    """accuracy, balanced accuracy (mean sensitivity of the 4 classes: the checkpoint is chosen on it), and per class
    the sensitivity and the positive predictive value, all in %"""
    k = len(CLASSES)
    cm = np.bincount(k * y + p, minlength=k * k).reshape(k, k)
    sens = 100 * np.diag(cm) / np.maximum(cm.sum(1), 1)
    ppv = 100 * np.diag(cm) / np.maximum(cm.sum(0), 1)
    return dict(accuracy=100 * float(np.trace(cm) / cm.sum()), balanced=float(sens.mean()),
                sensitivity=dict(zip(CLASSES, sens.round(2).tolist())), ppv=dict(zip(CLASSES, ppv.round(2).tolist())),
                confusion=cm.tolist())


def cv_folds(record, y):
    """CV_FOLDS lists of DS1 records: numpy seed 0, among 2000 draws the one whose S and V counts per fold vary the
    least (coefficient of variation), every fold holding CV_MIN beats of S, V and F"""
    counts = {r: np.bincount(y[record == r], minlength=len(CLASSES)) for r in DS1}
    rng, best, best_cost = np.random.default_rng(0), None, np.inf
    for _ in range(2000):
        perm = rng.permutation(DS1)
        folds = [sorted(perm[k::CV_FOLDS].tolist()) for k in range(CV_FOLDS)]
        have = np.array([np.sum([counts[r] for r in f], axis=0) for f in folds])
        if not all(have[:, c + 1].min() >= CV_MIN[c] for c in range(3)):
            continue
        cost = sum(have[:, c].std() / have[:, c].mean() for c in (1, 2))
        if cost < best_cost:
            best, best_cost = folds, cost
    return best


def rr_channels(rr):
    """rr (N, 3) seconds -> int8 (N, WINDOW, RR_CHANNELS): pre / local, post / local, local, each constant in time,
    RR_SCALE per 1.0, clipped to 127"""
    f = np.stack([rr[:, 0] / rr[:, 2], rr[:, 1] / rr[:, 2], rr[:, 2]], 1)
    q = np.clip(np.round(f * RR_SCALE), 0, 127).astype(np.int8)
    return np.repeat(q[:, None, :], WINDOW, 1)


def load(split, fold=None, rr=False):
    """(x int8 N x WINDOW x 1, y) of one split: 'train', 'validation' or 'test'. fold=k (0..CV_FOLDS-1): 'train' is
    DS1 without the records of fold k, 'validation' is fold k (cv_folds); 'test' is always DS2. rr: x gets the
    RR_CHANNELS rhythm channels after the ECG (N x WINDOW x (1 + RR_CHANNELS))"""
    d = np.load(OUT)
    if fold is None or split == "test":
        key = dict(train="train_records", validation="val_records", test="test_records")[split]
        m = np.isin(d["record"], d[key])
    else:
        held = cv_folds(d["record"], d["y"])[fold]
        m = np.isin(d["record"], held) if split == "validation" else np.isin(d["record"], [r for r in DS1 if r not in held])
    if rr:
        return np.concatenate([d["x"][m], rr_channels(d["rr"][m])], 2), d["y"][m]
    return d["x"][m], d["y"][m]


def augment(x, rng):
    """a random version of int8 beats (N, WINDOW, 1 [+ RR_CHANNELS]): amplitude, baseline wander, noise, shift (the
    AUG_* constants) of the ECG channel; int8 out, clipped to +-127 as the data set; the rhythm channels unchanged"""
    n = len(x)
    t = np.arange(WINDOW) / FS
    f = rng.uniform(*AUG_WANDER_HZ, (n, 1))
    wander = rng.uniform(0, AUG_WANDER, (n, 1)) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi, (n, 1)))
    a = x[:, :, 0].astype(np.float32) * rng.uniform(1 - AUG_SCALE, 1 + AUG_SCALE, (n, 1)) + wander
    a += rng.normal(0, AUG_NOISE, a.shape)
    shift = rng.integers(-AUG_SHIFT, AUG_SHIFT + 1, n)
    idx = np.clip(np.arange(WINDOW)[None, :] - shift[:, None], 0, WINDOW - 1)      # edge samples repeated
    a = np.take_along_axis(a, idx, 1)
    a = np.clip(np.round(a), -127, 127).astype(np.int8)[:, :, None]
    return np.concatenate([a, x[:, :, 1:]], 2) if x.shape[2] > 1 else a


if __name__ == "__main__":
    main()
