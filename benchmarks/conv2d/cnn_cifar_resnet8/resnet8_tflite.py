"""
resnet8_tflite.py -- the INT8 reference of the cnn_cifar_resnet8 benchmark: the OFFICIAL MLPerf Tiny ResNet-8 (CIFAR-10).

    python resnet8_tflite.py                              # accuracy on the PC (10,000 test images)
    python resnet8_tflite.py --board s3 --port /dev/ttyUSB0
    python resnet8_tflite.py --board esp32 --port /dev/ttyUSB0
    python resnet8_tflite.py --board esp32 --port /dev/ttyUSB0 --model-ram   # the model copied to internal RAM

Run from benchmarks/conv2d/cnn_cifar_resnet8/ with the benchmark environment (env_bench). Nothing here is ours: the model is the
standard image-classification model of MLPerf Tiny (mlcommons/tiny, pinned commit), already quantized to full INT8 for
TensorFlow Lite Micro, and it runs with the official TFLite Micro (esp-tflite-micro, Espressif's ESP-NN kernels).
Nobody can say the reference came out weak because of how we trained or converted it.
  1. downloads the model once (models/pretrainedResnet_quant.tflite)
  2. finds its input convention: raw pixels 0..255 or pixels / 255 (on 2,000 images, the one that is not chance)
  3. accuracy on the PC with the TFLite interpreter, on the 10,000 test images
  4. with --board: the chip classifies the first --images test images; its predictions must equal the PC interpreter's
Results: results/tflite_pc.json and results/tflite_<board>[_ram].json.
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mlp", "mlp_fashion"))
from fashion_tflite import BOARDS, measure_chip, tflm_project            # noqa: E402  (the same TFLite Micro firmware)

URL = ("https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/"
       "benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite")
MODEL = os.path.join("models", "pretrainedResnet_quant.tflite")
OPS = ("ops.AddConv2D(); ops.AddAdd(); ops.AddAveragePool2D(); ops.AddMean(); ops.AddFullyConnected(); "
       "ops.AddReshape(); ops.AddQuantize(); ops.AddDequantize(); ops.AddRelu(); ops.AddSoftmax(); ops.AddPad();")


def interpreter(model):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_content=model, num_threads=1)
    it.allocate_tensors()
    return it


def predict(model, x_u8, divide):
    """classes from the TFLite interpreter; pixel p enters as round(p [/ 255] / scale) + zero_point (as on the chip)"""
    it = interpreter(model)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    lo, hi = np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max
    table = np.clip(np.round((np.arange(256) / (255.0 if divide else 1.0)) / scale) + zp, lo, hi).astype(inp["dtype"])
    pred = []
    for x in x_u8:
        it.set_tensor(inp["index"], table[x][None])
        it.invoke()
        pred.append(int(np.argmax(it.get_tensor(out["index"])[0])))
    return np.array(pred), (float(scale), int(zp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=sorted(BOARDS), help="measure on this board with TFLite Micro")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--images", type=int, default=1000, help="test images sent to the chip (default 1000)")
    ap.add_argument("--arena-kb", type=int, default=100, help="TFLite Micro arena (default 100 KB)")
    ap.add_argument("--model-ram", action="store_true", help="copy the model to internal RAM before running it")
    ap.add_argument("--no-upload", action="store_true", help="the TFLite firmware is already on the board")
    args = ap.parse_args()
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    from model import cifar_u8                                  # CIFAR-10 test images, official order
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("runs", "tflite"), exist_ok=True)
    if not os.path.exists(MODEL):
        print("[*] downloading the official MLPerf Tiny ResNet-8 ...")
        urllib.request.urlretrieve(URL, MODEL)
    tflite = open(MODEL, "rb").read()
    x_test, y_test = cifar_u8(train=False)

    probe = {d: 100 * (predict(tflite, x_test[:2000], d)[0] == y_test[:2000]).mean() for d in (False, True)}
    divide = probe[True] > probe[False]
    print(f"[*] input convention: raw pixels {probe[False]:.1f}% | pixels / 255 {probe[True]:.1f}% (2,000 images) "
          f"-> {'pixels / 255' if divide else 'raw pixels'}")
    pred, (scale, zp) = predict(tflite, x_test, divide)
    acc = 100 * (pred == y_test).mean()
    print(f"[*] MLPerf Tiny ResNet-8, INT8: {acc:.2f}% on the 10,000 test images | model {len(tflite) / 1024:.1f} KB "
          f"| input scale {scale:.6g}, zero point {zp}")
    json.dump(dict(model="MLPerf Tiny pretrainedResnet_quant.tflite", url=URL, int8_accuracy=acc, model_kb=len(tflite) / 1024,
                   input="pixels / 255" if divide else "raw pixels", input_scale=scale, input_zero_point=zp),
              open("results/tflite_pc.json", "w"), indent=1)
    if not args.board:
        return

    n = min(args.images, len(x_test))
    project = os.path.join("runs", "tflite", f"tflm_{args.board}" + ("_ram" if args.model_ram else ""))
    if not args.no_upload:
        tflm_project(project, tflite, args.board, args.port, args.baud, args.arena_kb, 32 * 32 * 3, 10, args.model_ram)
        main_cc = os.path.join(project, "src", "main.cc")      # the ops of this CNN; the input convention found above
        src = open(main_cc).read()
        src = src.replace("MicroMutableOpResolver<6>", "MicroMutableOpResolver<14>")
        src = src.replace("ops.AddFullyConnected(); ops.AddReshape(); ops.AddQuantize(); ops.AddDequantize(); "
                          "ops.AddSoftmax(); ops.AddRelu();", OPS)
        if not divide:
            src = src.replace("roundf((p / 255.0f) / inp->params.scale)", "roundf(p / inp->params.scale)")
        if OPS not in src or ("roundf(p / inp->params.scale)" not in src and not divide):
            sys.exit("[X] the TFLite Micro firmware of mlp_fashion changed: update the replacements in this script")
        open(main_cc, "w").write(src)
    chip_pred, times, arena = measure_chip(project, args.board, args.port, args.baud, x_test[:n].reshape(n, -1),
                                           not args.no_upload)
    equal = 100 * (chip_pred == pred[:n]).mean()
    chip_acc = 100 * (chip_pred == y_test[:n]).mean()
    print(f"\n    MLPerf Tiny ResNet-8 INT8, TensorFlow Lite Micro on {BOARDS[args.board]} ({n} images, model in "
          f"{'RAM' if args.model_ram else 'flash'})")
    print(f"    equal to the PC interpreter : {equal:.2f}% {'OK' if equal == 100 else '[!]'}")
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
