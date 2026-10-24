"""
resnet20_tflite.py -- the official INT8 side of the cnn_cifar_resnet20 benchmark (TensorFlow Lite), independent of SYMBEX.

    python resnet20_tflite.py                      # convert the published ResNet-20 to INT8, accuracy on the PC
    python resnet20_tflite.py --board s3           # ... and latency on the chip with TensorFlow Lite Micro
    python resnet20_tflite.py --board esp32
    python resnet20_tflite.py --board s3 --model-ram   # the whole model copied to RAM first (if it fits)

Run from benchmarks/cnn_cifar_resnet20/ with the benchmark environment (env_bench: TensorFlow + PyTorch CPU).
Nothing here uses SYMBEX: the INT8 model comes from the official converter and runs with the official TFLite Micro
(esp-tflite-micro, Espressif's ESP-NN kernels).
  1. the SAME published model as the SYMBEX side (model.py), rebuilt in Keras: BatchNorm folded into the
     convolutions, explicit 1-pixel padding (Keras "same" pads differently with stride 2). Checked: the Keras FP32
     model must match the published accuracy before anything is converted.
  2. full-integer INT8 with tf.lite.TFLiteConverter (int8 input and output; 500 training images to calibrate)
  3. accuracy on the PC with the TFLite interpreter, on the 10,000 test images
  4. with --board: the chip classifies the first --images test images streamed over serial; its predictions must
     equal the PC interpreter's
Results: results/tflite_pc.json and results/tflite_<board>.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mlp_fashion"))
from fashion_tflite import BOARDS, MAIN_TFLM, measure_chip, tflm_project     # noqa: E402  (the same TFLite Micro firmware)

OPS = ("ops.AddConv2D(); ops.AddAdd(); ops.AddPad(); ops.AddMean(); ops.AddFullyConnected(); ops.AddReshape(); "
       "ops.AddQuantize(); ops.AddDequantize(); ops.AddRelu(); ops.AddMul(); ops.AddSub(); ops.AddSoftmax();")


def keras_resnet20(sd, mean, std):
    """the published ResNet-20 in Keras, from its PyTorch state dict (input: raw pixels 0..255, NHWC)"""
    import tensorflow as tf
    L = tf.keras.layers
    sd = {k[4:] if k.startswith("net.") else k: v.numpy() for k, v in sd.items()}

    def conv(x, name, bn, stride, k):
        w = sd[f"{name}.weight"]
        s = sd[f"{bn}.weight"] / np.sqrt(sd[f"{bn}.running_var"] + 1e-5)
        layer = L.Conv2D(w.shape[0], k, strides=stride, padding="valid")
        if k == 3:
            x = L.ZeroPadding2D(1)(x)
        y = layer(x)
        layer.set_weights([(w * s[:, None, None, None]).transpose(2, 3, 1, 0),
                           sd[f"{bn}.bias"] - sd[f"{bn}.running_mean"] * s])
        return y

    inp = L.Input((32, 32, 3))
    x = L.Rescaling(1 / 255.0)(inp)
    x = L.Normalization(mean=np.array(mean), variance=np.array(std) ** 2)(x)
    x = L.ReLU()(conv(x, "conv1", "bn1", 1, 3))
    for s in range(1, 4):
        for i in range(3):
            p = f"layer{s}.{i}"
            stride = 2 if (i == 0 and s > 1) else 1
            y = L.ReLU()(conv(x, f"{p}.conv1", f"{p}.bn1", stride, 3))
            y = conv(y, f"{p}.conv2", f"{p}.bn2", 1, 3)
            short = conv(x, f"{p}.downsample.0", f"{p}.downsample.1", stride, 1) if f"{p}.downsample.0.weight" in sd else x
            x = L.ReLU()(L.Add()([y, short]))
    x = L.GlobalAveragePooling2D()(x)
    fc = L.Dense(10)
    out = fc(x)
    fc.set_weights([sd["fc.weight"].T, sd["fc.bias"]])
    return tf.keras.Model(inp, out)


def convert_int8(m, calib):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([calib[i:i + 1].astype(np.float32)] for i in range(len(calib)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def predict_int8(tflite, x_u8):
    """classes from the TFLite interpreter; pixel p enters as round(p / scale) + zero_point (as on the chip)"""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=tflite)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    table = np.clip(np.round(np.arange(256) / scale) + zp, -128, 127).astype(np.int8)
    pred = []
    for x in x_u8:
        it.set_tensor(inp["index"], table[x][None])
        it.invoke()
        pred.append(int(np.argmax(it.get_tensor(out["index"])[0])))
    return np.array(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=sorted(BOARDS), help="measure on this board with TFLite Micro")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--images", type=int, default=1000, help="test images sent to the chip (default 1000)")
    ap.add_argument("--arena-kb", type=int, default=96, help="TFLite Micro arena (default 96 KB)")
    ap.add_argument("--model-ram", action="store_true", help="copy the model to internal RAM before running it")
    ap.add_argument("--no-upload", action="store_true", help="the TFLite firmware is already on the board")
    args = ap.parse_args()
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    import torch
    from model import MEAN, STD, cifar_u8, get_pretrained
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("runs", "tflite"), exist_ok=True)
    x_test, y_test = cifar_u8(train=False)

    keras = keras_resnet20(torch.load(get_pretrained(), map_location="cpu"), MEAN, STD)
    fp32 = 100 * (keras.predict(x_test.astype(np.float32), batch_size=500, verbose=0).argmax(1) == y_test).mean()
    print(f"[*] the published ResNet-20 rebuilt in Keras, FP32: {fp32:.2f}% (the published model: 92.59%)")
    if abs(fp32 - 92.59) > 0.1:
        sys.exit("[X] the Keras model does not match the published one: nothing converted")

    x_train, _ = cifar_u8(train=True)
    calib = x_train[np.random.default_rng(0).choice(len(x_train), 500, replace=False)]
    tflite = convert_int8(keras, calib)
    path = os.path.join("runs", "tflite", "resnet20_int8.tflite")
    open(path, "wb").write(tflite)
    pred = predict_int8(tflite, x_test)
    acc = 100 * (pred == y_test).mean()
    print(f"[*] INT8, official converter: {acc:.2f}% on the 10,000 test images | model {len(tflite) / 1024:.1f} KB -> {path}")
    json.dump(dict(fp32_keras=fp32, int8_accuracy=acc, model_kb=len(tflite) / 1024), open("results/tflite_pc.json", "w"), indent=1)
    if not args.board:
        return

    n = min(args.images, len(x_test))
    project = os.path.join("runs", "tflite", f"tflm_{args.board}" + ("_ram" if args.model_ram else ""))
    if not args.no_upload:
        tflm_project(project, tflite, args.board, args.port, args.baud, args.arena_kb, 32 * 32 * 3, 10, args.model_ram)
        main_cc = os.path.join(project, "src", "main.cc")     # the ops of a CNN, and raw pixels at the input
        src = open(main_cc).read()
        src = src.replace("MicroMutableOpResolver<6>", "MicroMutableOpResolver<14>")
        src = src.replace("ops.AddFullyConnected(); ops.AddReshape(); ops.AddQuantize(); ops.AddDequantize(); "
                          "ops.AddSoftmax(); ops.AddRelu();", OPS)
        src = src.replace("roundf((p / 255.0f) / inp->params.scale)", "roundf(p / inp->params.scale)")
        if OPS not in src or "roundf(p / inp->params.scale)" not in src:
            sys.exit("[X] the TFLite Micro firmware of mlp_fashion changed: update the replacements in this script")
        open(main_cc, "w").write(src)
    chip_pred, times, arena = measure_chip(project, args.board, args.port, args.baud, x_test[:n].reshape(n, -1),
                                           not args.no_upload)
    equal = 100 * (chip_pred == pred[:n]).mean()
    chip_acc = 100 * (chip_pred == y_test[:n]).mean()
    print(f"\n    INT8 ResNet-20, TensorFlow Lite Micro on {BOARDS[args.board]} ({n} images, model in "
          f"{'RAM' if args.model_ram else 'flash'})")
    print(f"    equal to the PC interpreter : {equal:.2f}% {'OK' if equal == 100 else '[X]'}")
    print(f"    accuracy                    : {chip_acc:.2f}%")
    print(f"    latency                     : median {np.median(times) / 1000:.2f} ms | min {times.min() / 1000:.2f} ms")
    print(f"    flash (model) / RAM (arena) : {len(tflite) / 1024:.1f} KB / {arena / 1024:.1f} KB")
    name = f"results/tflite_{args.board}{'_ram' if args.model_ram else ''}.json"
    json.dump(dict(board=args.board, images=n, equal_to_pc=equal, accuracy=chip_acc, latency_ms=float(np.median(times)) / 1000,
                   latency_min_ms=float(times.min()) / 1000, model_kb=len(tflite) / 1024, arena_kb=arena / 1024,
                   model_in_ram=args.model_ram), open(name, "w"), indent=1)
    print(f"    -> {name}")


if __name__ == "__main__":
    main()
