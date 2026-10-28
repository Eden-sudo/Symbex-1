"""
har_tflite.py -- the INT8 reference of the har_ultracompact benchmark: the SAME architectures of Table 1 (model.py),
in INT8 with TensorFlow Lite, the way anyone deploys a network on a microcontroller.

    python har_tflite.py --arch max --seed 42                                # train (once) + INT8 + accuracy on the PC
    python har_tflite.py --all                                               # 4 architectures x seeds 42, 43, 44
    python har_tflite.py --arch max --seed 42 --board esp32 --port /dev/ttyUSB0          # ... and TFLite Micro on the board
    python har_tflite.py --arch max --seed 42 --board s3 --port /dev/ttyACM0 --usb

Run from benchmarks/conv1d/har_ultracompact/ with the benchmark environment (env_bench: TensorFlow), after
har_data.py. The network is the float version of model.BinaryHAR, written directly in Keras: every conv (no padding,
stride 1, no bias) + BatchNorm + ReLU, max pool 4 where the architecture pools, then one dense layer to the 17 classes.
The same recipe as har_train.py: the same subject split, input x / 127, Adam with --lr and cosine decay, --epochs
epochs, batches of 128, cross-entropy, no augmentation, the checkpoint of the best VALIDATION epoch; the test subjects
are scored once. Then:
    INT8    full-integer post-training quantization (TFLiteConverter, 500 representative training windows)
    PC      the TFLite interpreter on the validation and test windows (inputs quantized as on the chip)
    board   TensorFlow Lite Micro (esp-tflite-micro, ESP-NN; the firmware of mlp_fashion/fashion_tflite.py) on the first
            --windows validation windows: latency, arena, and the same class as the PC interpreter, window by window
            (the same windows chip_test.py sends to the SYMBEX network)
Results: models/har_int8_<arch>_s<seed>.tflite (+ .weights.h5, the float Keras weights), results/tflite_pc_<arch>_s<seed>.json,
results/tflite_<board>_<arch>_s<seed>[_ram].json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "..", "mlp", "mlp_fashion"))
from har_data import load                                                                    # noqa: E402

ARCHS = {                                  # the same as model.ARCHS (kept here: model.py imports torch)
    "min":    [("conv", 4, 7), ("conv", 4, 7), ("pool", 4), ("conv", 4, 7), ("pool", 4)],
    "max10":  [("conv", 8, 15), ("conv", 32, 7), ("pool", 4), ("conv", 32, 7), ("pool", 4)],
    "max5":   [("conv", 32, 7), ("conv", 32, 15), ("pool", 4), ("conv", 32, 15), ("pool", 4)],
    "max":    [("conv", 32, 15), ("conv", 32, 15), ("pool", 4), ("conv", 32, 15), ("pool", 4)],
}
LENGTH, AXES, CLASSES = 151, 3, 17
SEEDS = (42, 43, 44)
# TFLite builtin op -> TFLite Micro resolver call (the ops a converted Keras Conv1D network uses)
TFLM_OPS = {"CONV_2D": "AddConv2D", "MAX_POOL_2D": "AddMaxPool2D", "RESHAPE": "AddReshape",
            "EXPAND_DIMS": "AddExpandDims", "SQUEEZE": "AddSqueeze", "FULLY_CONNECTED": "AddFullyConnected",
            "RELU": "AddRelu", "QUANTIZE": "AddQuantize", "DEQUANTIZE": "AddDequantize", "SOFTMAX": "AddSoftmax"}


def build(arch, tf, batch=None):
    L = tf.keras.layers
    x_in = L.Input((LENGTH, AXES), batch_size=batch)
    x = x_in
    for layer in ARCHS[arch]:
        if layer[0] == "conv":
            x = L.ReLU()(L.BatchNormalization()(L.Conv1D(layer[1], layer[2], use_bias=False)(x)))
        else:
            x = L.MaxPooling1D(layer[1])(x)
    return tf.keras.Model(x_in, L.Dense(CLASSES)(L.Flatten()(x)))


def input_table(scale, zero_point):
    """uint8 sent to the chip (int8 + 128) -> the INT8 input tensor: q = round(((p - 128) / 127) / scale) + zero"""
    p = np.arange(256, dtype=np.float32)
    return np.clip(np.round((p - 128) / 127 / scale) + zero_point, -128, 127).astype(np.int8)


def predict_tflite(tflite, x_u8):
    """the classes of the INT8 interpreter (inputs quantized with the chip's table)"""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    try:                                                              # the plain kernels, no XNNPACK delegate
        from ai_edge_litert.interpreter import OpResolverType
    except ImportError:
        from tensorflow.lite.python.interpreter import OpResolverType
    it = Interpreter(model_content=tflite, num_threads=1,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    table = input_table(*inp["quantization"])
    pred = []
    for x in x_u8:
        it.set_tensor(inp["index"], table[x].reshape(inp["shape"]))
        it.invoke()
        pred.append(int(np.argmax(it.get_tensor(out["index"])[0])))
    ops = sorted({d["op_name"] for d in it._get_ops_details()})
    return np.array(pred), ops


def train_convert(arch, seed, epochs, lr, path, tf):
    (xtr, ytr), (xva, yva) = load("train"), load("validation")
    f = lambda x: (x.astype(np.float32) / 127)
    tf.keras.utils.set_random_seed(seed)
    model = build(arch, tf)
    steps = -(-len(xtr) // 128)
    model.compile(optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(lr, steps * epochs)),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    class Best(tf.keras.callbacks.Callback):                         # the checkpoint of the best validation epoch
        best, epoch, weights = -1.0, 0, None

        def on_epoch_end(self, ep, logs=None):
            va = logs["val_accuracy"]
            if va > self.best:
                Best.best, Best.epoch, Best.weights = va, ep + 1, self.model.get_weights()

    print(f"[*] training the float {arch} network in Keras ({epochs} epochs, seed {seed}) ...")
    model.fit(f(xtr), ytr, batch_size=128, epochs=epochs, validation_data=(f(xva), yva), callbacks=[Best()],
              shuffle=True, verbose=2)
    model.set_weights(Best.weights)
    model.save_weights(path.replace(".tflite", ".weights.h5"))      # to convert again without training
    convert(arch, model.get_weights(), path, tf)
    return dict(best_epoch=Best.epoch, validation_float=100 * Best.best)


def convert(arch, weights, path, tf):
    """full-integer INT8, batch fixed to 1 (as on the chip: no dynamic-shape ops around Flatten)"""
    (xtr, _) = load("train")
    model = build(arch, tf, batch=1)
    model.set_weights(weights)
    rep = (xtr[np.random.default_rng(0).choice(len(xtr), 500, replace=False)].astype(np.float32) / 127)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([r[None]] for r in rep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    open(path, "wb").write(conv.convert())


def run_pc(arch, seed, epochs, lr, tf):
    name = f"{arch}_s{seed}"
    path = os.path.join("models", f"har_int8_{name}.tflite")
    out = os.path.join("results", f"tflite_pc_{name}.json")
    info = json.load(open(out)) if os.path.exists(out) else {}
    if not os.path.exists(path):
        info = train_convert(arch, seed, epochs, lr, path, tf)
    tflite = open(path, "rb").read()
    res = dict(arch=arch, seed=seed, epochs=epochs, model=path, kb=len(tflite) / 1024,
               best_epoch=info.get("best_epoch"), validation_float=info.get("validation_float"))
    for split in ("validation", "test"):
        x, y = load(split)
        pred, ops = predict_tflite(tflite, (x.astype(np.int16) + 128).astype(np.uint8).reshape(len(x), -1))
        res[split] = 100 * float((pred == y).mean())
    res["ops"] = ops
    json.dump(res, open(out, "w"), indent=1)
    print(f"    {name}: INT8 {res['kb']:.1f} KB | validation {res['validation']:.2f}% | test {res['test']:.2f}%"
          f"  (best epoch {res['best_epoch']}) -> {out}")
    return tflite, res


def run_board(args, tflite, ops):
    from fashion_tflite import measure_chip, tflm_project
    name = f"{args.arch}_s{args.seed}"
    x, _ = load("validation")
    n = min(args.windows, len(x))
    x_u8 = (x[:n].astype(np.int16) + 128).astype(np.uint8).reshape(n, -1)
    pc_pred, _ = predict_tflite(tflite, x_u8)
    ram = "_ram" if args.weights == "ram" else ""
    project = os.path.join("runs", "tflite", f"tflm_{args.board}_{name}{ram}")
    if not args.no_upload:
        missing = [o for o in ops if o not in TFLM_OPS]
        if missing:
            sys.exit(f"[X] ops without a TFLite Micro resolver call in TFLM_OPS: {missing}")
        tflm_project(project, tflite, args.board, args.port, args.baud, args.arena_kb, LENGTH * AXES, CLASSES,
                     args.weights == "ram", args.usb)
        main_cc = os.path.join(project, "src", "main.cc")
        src = open(main_cc).read()
        old_ops = ("static tflite::MicroMutableOpResolver<6> ops;\n    ops.AddFullyConnected(); ops.AddReshape(); "
                   "ops.AddQuantize(); ops.AddDequantize(); ops.AddSoftmax(); ops.AddRelu();")
        old_in = "roundf((p / 255.0f) / inp->params.scale)"
        if old_ops not in src or old_in not in src:
            sys.exit("[X] the TFLite Micro firmware of mlp_fashion changed: update the replacements in this script")
        src = src.replace(old_ops, f"static tflite::MicroMutableOpResolver<{len(ops)}> ops;\n    "
                          + " ".join(f"ops.{TFLM_OPS[o]}();" for o in ops))
        src = src.replace(old_in, "roundf(((p - 128) / 127.0f) / inp->params.scale)")   # = input_table()
        open(main_cc, "w").write(src)
    chip, times, arena = measure_chip(project, args.board, args.port, args.baud, x_u8, not args.no_upload, args.usb)
    equal = 100 * float((chip == pc_pred).mean())
    print(f"\n    {name} INT8, TensorFlow Lite Micro on {args.board} ({n} validation windows, model in {args.weights})")
    print(f"    same class as the PC interpreter : {equal:.2f}%" + ("" if equal == 100 else "  [X]"))
    print(f"    latency                          : median {np.median(times) / 1000:.3f} ms | min {times.min() / 1000:.3f} ms")
    print(f"    flash (model) / RAM (arena)      : {len(tflite) / 1024:.1f} KB / {arena / 1024:.1f} KB")
    out = os.path.join("results", f"tflite_{args.board}_{name}{ram}.json")
    json.dump(dict(arch=args.arch, seed=args.seed, board=args.board, weights=args.weights, windows=n, equal_to_pc=equal,
                   median_ms=float(np.median(times)) / 1000, min_ms=float(times.min()) / 1000,
                   model_kb=len(tflite) / 1024, arena_kb=arena / 1024), open(out, "w"), indent=1)
    print(f"    -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(ARCHS), default="max")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="every architecture x seeds 42, 43, 44 (PC only)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--board", choices=["esp32", "s3"])
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--usb", action="store_true", help="the ESP32-S3's native USB port (/dev/ttyACM0)")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash",
                    help="where the model lives on the board: flash (default) or RAM (the whole .tflite copied at boot)")
    ap.add_argument("--windows", type=int, default=1000, help="validation windows sent to the board")
    ap.add_argument("--arena-kb", type=int, default=64, help="TFLite Micro arena (default 64 KB)")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    os.chdir(HERE)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):                 # share the GPU with other trainings
        tf.config.experimental.set_memory_growth(g, True)
    if args.all:
        rows = [run_pc(a, s, args.epochs, args.lr, tf)[1] for a in ARCHS for s in SEEDS]
        print(f"\n    INT8 (TFLite), test accuracy, mean +- std over seeds {SEEDS}")
        for a in ARCHS:
            t = [r["test"] for r in rows if r["arch"] == a]
            print(f"    {a:<6} {np.mean(t):5.1f} +- {np.std(t):.1f} %   ({', '.join(f'{v:.1f}' for v in t)})")
        return
    tflite, res = run_pc(args.arch, args.seed, args.epochs, args.lr, tf)
    if args.board:
        run_board(args, tflite, res["ops"])


if __name__ == "__main__":
    main()
