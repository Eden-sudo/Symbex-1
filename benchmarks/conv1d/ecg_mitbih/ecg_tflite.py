"""
ecg_tflite.py -- the INT8 reference of the ecg_mitbih benchmark: the SAME architectures (ecg_data.ARCHS) in INT8 with
TensorFlow Lite, the way anyone deploys a network on a microcontroller.

    python ecg_tflite.py --arch m --seed 42                 # train (once) + INT8 + scores on the PC
    python ecg_tflite.py --all                              # every architecture x seeds 42, 43, 44
    python ecg_tflite.py --arch m --augment --seed 42       # with ecg_data.augment every epoch (= ecg_train.py)
    python ecg_tflite.py --arch m --cv                      # patient-wise cross-validation inside DS1 (no test)
    python ecg_tflite.py --arch m --rr --seed 42            # + the RR-interval input channels (= ecg_train.py --rr)
    python ecg_tflite.py --arch m --seed 42 --board esp32 --port /dev/ttyUSB0          # ... and TFLite Micro on a board
    python ecg_tflite.py --arch m --seed 42 --board s3 --port /dev/ttyACM0 --usb
    python ecg_tflite.py --arch m --seed 42 --board pico2_arm --port /dev/ttyACM1      # Pico 2 M33 (CMSIS-NN)
    python ecg_tflite.py --arch m --seed 42 --board pico2_riscv --port /dev/ttyACM1    # Pico 2 Hazard3 (reference)

Run from benchmarks/conv1d/ecg_mitbih/ with the benchmark environment (env_bench: TensorFlow), after ecg_data.py. The
network is the float version of the binary one, written directly in Keras: every conv (no padding, stride 1, no bias)
+ BatchNorm + ReLU, max pool where the architecture pools, then one dense layer to the 4 classes. The same recipe as
ecg_train.py: the same records, input x / 127, Adam with --lr and cosine decay, --epochs epochs, batches of 128,
cross-entropy weighted by ecg_data.class_weights, no augmentation unless --augment (a new ecg_data.augment draw every
epoch, numpy seed = the seed), the checkpoint of the best validation BALANCED accuracy; the test records are scored
once. --fold k / --cv: the folds of ecg_train.py (validation = fold k, the test set NOT scored), the pooled
out-of-fold INT8 scores in results/tflite/cv_<arch>_s<seed>[_aug].json. Then full-integer post-training quantization (500 representative training
beats) and the TFLite interpreter on the validation and test beats (inputs quantized as on the chip, uint8 = int8 + 128).
board: TensorFlow Lite Micro with the firmware of har_ultracompact/har_tflite.py (ESP-NN on the ESP32 / S3, CMSIS-NN on
the Pico 2 M33, reference kernels on Hazard3) on the same --windows validation beats ecg_export.py gives the SYMBEX
chip test (drawn with numpy seed 0): latency, arena, and the same class as the PC interpreter, beat by beat.
Results: models/ecg_int8_<arch>_s<seed>.tflite (+ .weights.h5), results/tflite/tflite_pc_<arch>_s<seed>.json,
results/tflite/tflite_<board>_<arch>_s<seed>[_ram].json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "har_ultracompact"))
from ecg_data import (ARCHS, CLASSES, CV_FOLDS, RR_CHANNELS, WINDOW, augment, class_weights, load,  # noqa: E402
                      scores)
from har_tflite import measure_board, predict_tflite                                         # noqa: E402

SEEDS = (42, 43, 44)


def build(arch, tf, batch=None, rr=False):
    L = tf.keras.layers
    x_in = L.Input((WINDOW, 1 + (RR_CHANNELS if rr else 0)), batch_size=batch)
    x = x_in
    for layer in ARCHS[arch]:
        if layer[0] == "conv":
            x = L.ReLU()(L.BatchNormalization()(L.Conv1D(layer[1], layer[2], use_bias=False)(x)))
        else:
            x = L.MaxPooling1D(layer[1])(x)
    return tf.keras.Model(x_in, L.Dense(len(CLASSES))(L.Flatten()(x)))


def u8(x):
    return (x.astype(np.int16) + 128).astype(np.uint8).reshape(len(x), -1)


def train_convert(arch, seed, epochs, lr, path, tf, aug=False, fold=None, rr=False):
    (xtr, ytr), (xva, yva) = load("train", fold, rr), load("validation", fold, rr)
    f = lambda x: (x.astype(np.float32) / 127)
    tf.keras.utils.set_random_seed(seed)
    model = build(arch, tf, rr=rr)
    steps = -(-len(xtr) // 128)
    model.compile(optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(lr, steps * epochs)),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    xva_f = f(xva)

    class Best(tf.keras.callbacks.Callback):                         # the checkpoint of the best validation epoch
        best, epoch, weights = -1.0, 0, None

        def on_epoch_end(self, ep, logs=None):
            va = scores(yva, self.model.predict(xva_f, batch_size=4096, verbose=0).argmax(1))["balanced"]
            if va > Best.best:
                Best.best, Best.epoch, Best.weights = va, ep + 1, self.model.get_weights()
            if (ep + 1) % 10 == 0:
                print(f"    {arch}_s{seed} | epoch {ep + 1:3d}/{epochs} | validation balanced {va:5.1f}% "
                      f"(best {Best.best:5.1f}% at {Best.epoch})", flush=True)

    print(f"[*] training the float {arch} network in Keras ({epochs} epochs, seed {seed}) ...")
    w = class_weights(ytr)
    if aug:                                                          # a new draw every epoch, one epoch per fit
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            model.fit(f(augment(xtr, rng)), ytr, batch_size=128, epochs=ep + 1, initial_epoch=ep,
                      class_weight=dict(enumerate(w)), callbacks=[Best()], shuffle=True, verbose=0)
    else:
        model.fit(f(xtr), ytr, batch_size=128, epochs=epochs, class_weight=dict(enumerate(w)), callbacks=[Best()],
                  shuffle=True, verbose=0)
    model.set_weights(Best.weights)
    model.save_weights(path.replace(".tflite", ".weights.h5"))      # to convert again without training
    convert(arch, model.get_weights(), path, tf, fold, rr)
    return dict(best_epoch=Best.epoch, validation_float_balanced=Best.best)


def convert(arch, weights, path, tf, fold=None, rr=False):
    """full-integer INT8, batch fixed to 1 (as on the chip: no dynamic-shape ops around Flatten)"""
    (xtr, _) = load("train", fold, rr)
    model = build(arch, tf, batch=1, rr=rr)
    model.set_weights(weights)
    rep = (xtr[np.random.default_rng(0).choice(len(xtr), 500, replace=False)].astype(np.float32) / 127)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([r[None]] for r in rep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    open(path, "wb").write(conv.convert())


def run_name(arch, seed, aug=False, fold=None, rr=False):
    return f"{arch}_s{seed}" + ("_aug" if aug else "") + ("_rr" if rr else "") + ("" if fold is None else f"_f{fold}")


def run_pc(arch, seed, epochs, lr, tf, aug=False, fold=None, rr=False):
    name = run_name(arch, seed, aug, fold, rr)
    path = os.path.join("models", f"ecg_int8_{name}.tflite")
    out = os.path.join("results", "tflite", ("cv" if fold is not None else ""), f"tflite_pc_{name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    info = json.load(open(out)) if os.path.exists(out) else {}
    if not os.path.exists(path):
        info = train_convert(arch, seed, epochs, lr, path, tf, aug, fold, rr)
    tflite = open(path, "rb").read()
    res = dict(arch=arch, seed=seed, epochs=epochs, augment=aug, fold=fold, rr=rr, model=path, kb=len(tflite) / 1024,
               best_epoch=info.get("best_epoch"), validation_float_balanced=info.get("validation_float_balanced"))
    for split in ("validation",) if fold is not None else ("validation", "test"):    # folds never see the test set
        x, y = load(split, fold, rr)
        pred, ops = predict_tflite(tflite, u8(x))
        res[split] = scores(y, pred)
        if fold is not None:
            res["validation_pred"] = pred.tolist()
    res["ops"] = ops
    json.dump(res, open(out, "w"), indent=1)
    v = res["validation"]
    t = (f"test balanced {res['test']['balanced']:.2f}%, accuracy {res['test']['accuracy']:.2f}%, sensitivity "
         f"{res['test']['sensitivity']}" if "test" in res else f"(fold {fold}) sensitivity {v['sensitivity']}")
    print(f"    {name}: INT8 {res['kb']:.1f} KB | validation balanced {v['balanced']:.2f}% (float "
          f"{res['validation_float_balanced']}) | {t}  (best epoch {res['best_epoch']}) -> {out}", flush=True)
    return tflite, res


def cross_validate(args, tf):
    """every fold (a fold already done is read back), then the pooled out-of-fold INT8 and float scores"""
    folds = range(CV_FOLDS) if args.fold is None else [args.fold]
    rows = [run_pc(args.arch, args.seed, args.epochs, args.lr, tf, args.augment, k, args.rr)[1] for k in folds]
    if args.fold is not None:
        return
    pooled = scores(np.concatenate([load("validation", k)[1] for k in folds]),
                    np.concatenate([np.array(r["validation_pred"]) for r in rows]))
    name = run_name(args.arch, args.seed, args.augment, rr=args.rr)
    out = os.path.join("results", "tflite", f"cv_{name}.json")
    json.dump(dict(name=name, arch=args.arch, seed=args.seed, epochs=args.epochs, augment=args.augment,
                   folds=list(folds), fold_balanced=[r["validation"]["balanced"] for r in rows],
                   fold_float_balanced=[r["validation_float_balanced"] for r in rows], pooled=pooled),
              open(out, "w"), indent=1)
    print(f"[*] CV INT8 {name}: pooled out-of-fold balanced {pooled['balanced']:.1f}%, folds "
          f"{' / '.join(f'{r['validation']['balanced']:.1f}' for r in rows)} -> {out}")


def run_board(args, tflite, ops):
    name = run_name(args.arch, args.seed, args.augment, rr=args.rr)
    x, _ = load("validation", rr=args.rr)
    n = min(args.windows, len(x))
    x_u8 = u8(x[np.sort(np.random.default_rng(0).choice(len(x), n, replace=False))])   # = ecg_export.py
    pc_pred, _ = predict_tflite(tflite, x_u8)
    ram = "_ram" if args.weights == "ram" else ""
    project = os.path.join("runs", "tflite", f"tflm_{args.board}_{name}{ram}")
    chip, times, arena = measure_board(args, tflite, ops, x_u8, len(CLASSES), project)
    equal = 100 * float((chip == pc_pred).mean())
    print(f"\n    {name} INT8, TensorFlow Lite Micro on {args.board} ({n} validation beats, model in {args.weights})")
    print(f"    same class as the PC interpreter : {equal:.2f}%" + ("" if equal == 100 else "  [X]"))
    print(f"    latency                          : median {np.median(times) / 1000:.3f} ms | min {times.min() / 1000:.3f} ms")
    print(f"    flash (model) / RAM (arena)      : {len(tflite) / 1024:.1f} KB / {arena / 1024:.1f} KB")
    out = os.path.join("results", "tflite", f"tflite_{args.board}_{name}{ram}.json")
    json.dump(dict(arch=args.arch, seed=args.seed, board=args.board, weights=args.weights, windows=n, equal_to_pc=equal,
                   median_ms=float(np.median(times)) / 1000, min_ms=float(times.min()) / 1000,
                   model_kb=len(tflite) / 1024, arena_kb=arena / 1024), open(out, "w"), indent=1)
    print(f"    -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(ARCHS), default="m")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="every architecture x seeds 42, 43, 44 (PC only)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--board", choices=["esp32", "s3", "pico2_arm", "pico2_riscv"])
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--usb", action="store_true", help="the ESP32-S3's native USB port (/dev/ttyACM0)")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash",
                    help="where the model lives on the board: flash (default) or RAM (the whole .tflite copied at boot)")
    ap.add_argument("--windows", type=int, default=1000, help="validation beats sent to the board")
    ap.add_argument("--arena-kb", type=int, default=64, help="TFLite Micro arena (default 64 KB)")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--augment", action="store_true", help="a new ecg_data.augment draw of the training beats every epoch")
    ap.add_argument("--cv", action="store_true", help="patient-wise cross-validation inside DS1 (every fold, no test)")
    ap.add_argument("--fold", type=int, choices=range(CV_FOLDS), help="with --cv: only this fold")
    ap.add_argument("--rr", action="store_true", help="+ the RR-interval channels (ecg_data.rr_channels) as input")
    ap.add_argument("--gpu-memory", type=int, help="MB of GPU memory for this process (default: grow as needed); "
                    "a cap lets several trainings share the GPU")
    args = ap.parse_args()
    os.chdir(HERE)
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("results", "tflite"), exist_ok=True)
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):                 # share the GPU with other trainings
        if args.gpu_memory:
            tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(args.gpu_memory)])
        else:
            tf.config.experimental.set_memory_growth(g, True)
    if args.cv:
        return cross_validate(args, tf)
    jobs = [(a, s) for a in ARCHS for s in SEEDS] if args.all else [(args.arch, args.seed)]
    rows = [run_pc(a, s, args.epochs, args.lr, tf, args.augment, rr=args.rr) for a, s in jobs]
    if args.board and not args.all:
        run_board(args, rows[0][0], rows[0][1]["ops"])
    rows = [r for _, r in rows]
    if args.all:
        print(f"\n    INT8 (TFLite), mean +- std over seeds {SEEDS}")
        for a in ARCHS:
            r = [x for x in rows if x["arch"] == a]
            vb, tb = [x["validation"]["balanced"] for x in r], [x["test"]["balanced"] for x in r]
            print(f"    {a:<4} validation balanced {np.mean(vb):5.1f} +- {np.std(vb):.1f} %   "
                  f"test balanced {np.mean(tb):5.1f} +- {np.std(tb):.1f} %")


if __name__ == "__main__":
    main()
