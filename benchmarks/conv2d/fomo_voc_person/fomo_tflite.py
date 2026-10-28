"""
fomo_tflite.py -- the INT8 reference of the fomo_voc_person benchmark: the SAME detector, in INT8 with TensorFlow Lite.

    python fomo_tflite.py                                   # train (once) + INT8 + F1 on the PC
    python fomo_tflite.py --board s3 --port /dev/ttyUSB0    # ... and the latency with TFLite Micro on the board
    python fomo_tflite.py --channels 32-64-128 --blocks 1-2-2 --weight 20 --epochs 60
    python fomo_tflite.py --channels 32-32-128 --blocks 1-2-2 --seed 43     # another training seed

Run from benchmarks/conv2d/fomo_voc_person/ with the benchmark environment (env_bench: TensorFlow). The network is the float
architecture of model.RealFOMO, written directly in Keras (no conversion between frameworks): stem 3x3 stride 2, blocks
of conv + BatchNorm + ReLU with summed shortcuts (1x1 conv where the shape changes), a 1x1 head with two outputs per
cell. The same recipe as fomo_train.py: VOC person grids, cross-entropy per cell with the person cells weighted, a
horizontal flip of image and grid, Adam + cosine, 1,000 random training images kept out to calibrate the decision
threshold (the same fixed split as fomo_train.py, model.SPLIT_SEED; each model gets its own threshold). Then:
    INT8        full-integer post-training quantization (TFLiteConverter, 500 representative training images)
    F1 (PC)     the TFLite interpreter on the 4,952 VOC 2007 test images, with the INT8 model's own calibrated threshold
    board       TensorFlow Lite Micro (esp-tflite-micro, ESP-NN) on the first --images test images: latency, arena, and
                the same output as the PC interpreter (the index of the largest of the 288 outputs, image by image)
Results: models/fomo_int8_<name>.tflite, results/tflite_<pc|board>_<name>.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "..", "mlp", "mlp_fashion"))


def build(channels, blocks, tf):
    L = tf.keras.layers
    x_in = L.Input((96, 96, 3))
    x = L.Rescaling(1 / 255.0)(x_in)
    x = L.ReLU()(L.BatchNormalization()(L.Conv2D(channels[0], 3, 2, "same", use_bias=False)(x)))
    cin = channels[0]
    for c, s, n in zip(channels, (1, 2, 2), blocks):
        for j in range(n):
            st = s if j == 0 else 1
            y = L.BatchNormalization()(L.Conv2D(c, 3, st, "same", use_bias=False)(x))
            sc = x
            if st != 1 or cin != c:
                sc = L.BatchNormalization()(L.Conv2D(c, 1, st, "same", use_bias=False)(x))
            x = L.ReLU()(L.Add()([y, sc]))
            cin = c
    return tf.keras.Model(x_in, L.Conv2D(2, 1)(x))                      # logits per cell: 12 x 12 x 2


def margins_tflite(tflite, x):
    """person - background, per cell, from the INT8 interpreter (inputs quantized as on the chip)"""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_content=tflite, num_threads=1)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, z = inp["quantization"]
    table = np.clip(np.round(np.arange(256) / s) + z, -128, 127).astype(inp["dtype"])
    so, zo = out["quantization"]
    m, flat = [], []
    for img in x:
        it.set_tensor(inp["index"], table[img][None])
        it.invoke()
        q = it.get_tensor(out["index"])[0]
        flat.append(int(np.argmax(q)))                                  # what the chip reports, image by image
        v = (q.astype(np.float32) - zo) * so
        m.append(v[..., 1] - v[..., 0])
    return np.stack(m), np.array(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="32-64-128")
    ap.add_argument("--blocks", default="1-2-2")
    ap.add_argument("--weight", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42, help="training seed (42: the file names of the first runs)")
    ap.add_argument("--board", choices=["esp32", "s3"])
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--arena-kb", type=int, default=230, help="TFLite Micro arena (default 230 KB)")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    os.chdir(HERE)
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):                    # share the GPU with other trainings
        tf.config.experimental.set_memory_growth(g, True)
    from model import SPLIT_SEED, f1_score, held_out_centers, load, split_indices    # the data, the split and the score
    ch = tuple(int(v) for v in args.channels.split("-"))
    bl = tuple(int(v) for v in args.blocks.split("-"))
    name = f"c{args.channels}_b{args.blocks}_w{args.weight:g}_e{args.epochs}" + ("" if args.seed == 42 else f"_s{args.seed}")
    path = os.path.join("models", f"fomo_int8_{name}.tflite")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    xtr, ytr, ctr, otr = load("train")
    xte, _, cte, ote = load("test")
    nv = args.val                                                       # the same fixed random split as fomo_train.py
    tr, va = split_indices(len(xtr), nv)
    xva = xtr[va]
    cva, ova = held_out_centers(ctr, otr, va)
    split = dict(split_seed=SPLIT_SEED, val=nv)
    split_path = path.replace(".tflite", "_split.json")                 # the split the model was trained without
    if os.path.exists(path) and (json.load(open(split_path)) if os.path.exists(split_path) else None) != split:
        sys.exit(f"[X] {path} was trained with another validation split (or before the fixed one): calibrating it on "
                 f"this split would choose its threshold on images it trained on. Delete it to train it again")

    if not os.path.exists(path):
        tf.keras.utils.set_random_seed(args.seed)
        model = build(ch, bl, tf)
        steps = len(tr) // 128
        opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(5e-3, steps * args.epochs))
        w = tf.constant([1.0, args.weight])

        def loss_fn(y, z):
            ce = tf.keras.losses.sparse_categorical_crossentropy(y, z, from_logits=True)
            return tf.reduce_sum(ce * tf.gather(w, tf.cast(y, tf.int32))) / tf.reduce_sum(tf.gather(w, tf.cast(y, tf.int32)))

        def flip(x, y):
            f = tf.random.uniform(()) < 0.5
            return tf.cond(f, lambda: tf.image.flip_left_right(x), lambda: x), tf.cond(f, lambda: y[:, ::-1], lambda: y)

        ds = (tf.data.Dataset.from_tensor_slices((xtr[tr].astype(np.float32), ytr[tr].astype(np.int32)))
              .shuffle(4096, seed=args.seed).map(flip, num_parallel_calls=tf.data.AUTOTUNE).batch(128, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))
        model.compile(optimizer=opt, loss=loss_fn)
        print(f"[*] training the float {args.channels} / {args.blocks} FOMO in Keras ({args.epochs} epochs, person weight "
              f"{args.weight:g}) ...")
        model.fit(ds, epochs=args.epochs, verbose=2)
        rep = xtr[np.random.default_rng(0).choice(tr, 500, replace=False)].astype(np.float32)
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([r[None]] for r in rep)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        open(path, "wb").write(conv.convert())
        json.dump(split, open(split_path, "w"))
    tflite = open(path, "rb").read()

    from fomo_train import calibrate                                    # the same calibration as the binary model
    mva, _ = margins_tflite(tflite, xva)
    th, sva = calibrate(mva, cva, ova)
    mte, flat = margins_tflite(tflite, xte)
    s0, s1 = f1_score(mte > 0, cte, ote), f1_score(mte > th, cte, ote)
    print(f"\n    FOMO INT8 (TFLite), {args.channels} / {args.blocks}: model {len(tflite) / 1024:.1f} KB")
    print(f"    threshold calibrated on {nv} held-out training images: {th:+.3f}  (validation F1 {sva['f1']:.1f}%)")
    print(f"    test F1: default {s0['f1']:.1f}% | calibrated {s1['f1']:.1f}% (precision {s1['precision']:.1f}%, recall "
          f"{s1['recall']:.1f}%)")
    json.dump(dict(model=path, seed=args.seed, split_seed=SPLIT_SEED, val=nv, kb=len(tflite) / 1024, threshold=th,
                   f1_default=s0, f1_calibrated=s1),
              open(os.path.join("results", f"tflite_pc_{name}.json"), "w"), indent=1)
    if not args.board:
        return

    from fashion_tflite import BOARDS, measure_chip, tflm_project
    n = min(args.images, len(xte))
    project = os.path.join("runs", "tflite", f"tflm_{args.board}_{name}")
    if not args.no_upload:
        tflm_project(project, tflite, args.board, args.port, args.baud, args.arena_kb, 96 * 96 * 3, 2 * 12 * 12, False)
        main_cc = os.path.join(project, "src", "main.cc")
        src = open(main_cc).read()
        src = src.replace("MicroMutableOpResolver<6>", "MicroMutableOpResolver<12>")
        ops = ("ops.AddConv2D(); ops.AddAdd(); ops.AddRelu(); ops.AddPad(); ops.AddFullyConnected(); ops.AddReshape(); "
               "ops.AddQuantize(); ops.AddDequantize(); ops.AddSoftmax(); ops.AddMul();")
        src = src.replace("ops.AddFullyConnected(); ops.AddReshape(); ops.AddQuantize(); ops.AddDequantize(); "
                          "ops.AddSoftmax(); ops.AddRelu();", ops)
        src = src.replace("roundf((p / 255.0f) / inp->params.scale)", "roundf(p / inp->params.scale)")
        if ops not in src:
            sys.exit("[X] the TFLite Micro firmware of mlp_fashion changed: update the replacements in this script")
        open(main_cc, "w").write(src)
    chip_pred, times, arena = measure_chip(project, args.board, args.port, args.baud, xte[:n].reshape(n, -1), not args.no_upload)
    equal = 100 * (chip_pred == flat[:n]).mean()
    print(f"\n    FOMO INT8, TensorFlow Lite Micro on {BOARDS[args.board]} ({n} images)")
    print(f"    same output as the PC interpreter : {equal:.2f}%")
    print(f"    latency                           : median {np.median(times) / 1000:.2f} ms | min {times.min() / 1000:.2f} ms")
    print(f"    flash (model) / RAM (arena)       : {len(tflite) / 1024:.1f} KB / {arena / 1024:.1f} KB")
    json.dump(dict(board=args.board, images=n, equal_to_pc=equal, latency_ms=float(np.median(times)) / 1000,
                   model_kb=len(tflite) / 1024, arena_kb=arena / 1024),
              open(os.path.join("results", f"tflite_{args.board}_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
