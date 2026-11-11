"""
har_lowbit.py -- har_ultracompact benchmark: the low-bit integer baselines (INT8, INT4, INT2) of the SAME FP32 networks
that har_tflite.py quantizes for TensorFlow Lite, with the shared tool tools/lowbit (Brevitas quantization, ESP-NN
integer arithmetic: multiply-accumulate, no SYMBEX arithmetic).

    python har_lowbit.py --arch max                                 # INT8, INT4, INT2: PTQ + QAT, seeds 42, 43, 44
    python har_lowbit.py --arch max --no-distill                    # QAT on the labels only (the other recipe)
    python har_lowbit.py --arch max --chip-only --board esp32 --port /dev/ttyUSB0       # the seed-42 models on the board
    python har_lowbit.py --arch max --chip-only --board s3 --port /dev/ttyACM0 --usb

Run from benchmarks/conv1d/har_ultracompact/ (environment env_bench) after har_tflite.py: the FP32 network of each seed
is the Keras one har_tflite.py trained (models/har_int8_<arch>_s<seed>.weights.h5), moved to PyTorch layer by layer
(FloatHAR below: conv + BatchNorm + ReLU, max pool, dense) and checked against Keras on the validation windows; the
seed-42 network is the FP32 model lowbit quantizes, as fashion_lowbit.py does with its FP32 model. Data: the subject
split of har_data.py (validation carved out of the training subjects), input x / 127, test scored once per final
model. Results: results/lowbit/lowbit_<arch>[_nodistill].json; integer models in runs/lowbit_<arch>/.
On the board (--board): the seed-42 QAT model of every bit width as C on ESP-NN (tools/lowbit/export_c.py), the first
--windows validation windows, measured with tools/chip_test.py and checked against the integer simulation (must be
100 %); results: results/lowbit/lowbit_esp_nn_<arch>_<board>_<weights>[tag].json.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import brevitas                                      # noqa: F401 -- before TensorFlow: the other order crashes (free())

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools")))
from har_data import load                                                                    # noqa: E402
from har_tflite import ARCHS, AXES, CLASSES, LENGTH, build                                   # noqa: E402

KERAS_BN_EPS = 1e-3                                  # Keras' BatchNormalization default (PyTorch: 1e-5)


class FloatHAR(nn.Sequential):
    """har_tflite.build in PyTorch, input N x 3 x 151 (x / 127): the layers lowbit folds and quantizes"""
    def __init__(self, arch):
        layers, cin, length = [], AXES, LENGTH
        for layer in ARCHS[arch]:
            if layer[0] == "conv":
                _, cout, k = layer
                layers += [nn.Conv1d(cin, cout, k, bias=False), nn.BatchNorm1d(cout, eps=KERAS_BN_EPS), nn.ReLU()]
                cin, length = cout, length - k + 1
            else:
                layers.append(nn.MaxPool1d(layer[1]))
                length //= layer[1]
        super().__init__(*layers, nn.Flatten(), nn.Linear(cin * length, CLASSES))
        self.channels, self.length = cin, length


def from_keras(arch, weights_h5):
    """the Keras network's weights in FloatHAR. Keras flattens time-major (step l, channel c -> l * C + c), PyTorch
    channel-major (c * L + l): the dense columns are reordered."""
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)   # full FP32 on the GPU, as PyTorch computes it
    km = build(arch, tf)
    km.load_weights(weights_h5)
    tm = FloatHAR(arch).eval()
    kl = [l for l in km.layers if l.weights]
    convs = [m for m in tm if isinstance(m, nn.Conv1d)]
    bns = [m for m in tm if isinstance(m, nn.BatchNorm1d)]
    fc = [m for m in tm if isinstance(m, nn.Linear)][0]
    k_convs = [l for l in kl if isinstance(l, tf.keras.layers.Conv1D)]
    k_bns = [l for l in kl if isinstance(l, tf.keras.layers.BatchNormalization)]
    k_fc = [l for l in kl if isinstance(l, tf.keras.layers.Dense)][0]
    with torch.no_grad():
        for c, k in zip(convs, k_convs):
            c.weight.copy_(torch.as_tensor(k.get_weights()[0].transpose(2, 1, 0)))          # (k, in, out) -> (out, in, k)
        for b, k in zip(bns, k_bns):
            g, beta, mean, var = k.get_weights()
            b.weight.copy_(torch.as_tensor(g))
            b.bias.copy_(torch.as_tensor(beta))
            b.running_mean.copy_(torch.as_tensor(mean))
            b.running_var.copy_(torch.as_tensor(var))
        w, bias = k_fc.get_weights()                                                          # (L * C, classes)
        C, L = tm.channels, tm.length
        w = w.reshape(L, C, CLASSES).transpose(1, 0, 2).reshape(C * L, CLASSES)              # -> c * L + l
        fc.weight.copy_(torch.as_tensor(w.T))
        fc.bias.copy_(torch.as_tensor(bias))
    return tm, km


def as_input(x_int8):
    """int8 windows N x 151 x 3 -> the FP32 model's input N x 3 x 151, x / 127"""
    return torch.as_tensor(x_int8.astype(np.float32) / 127).transpose(1, 2).contiguous()


def get_fp32(arch, seed):
    """the FP32 FloatHAR of one seed (from har_tflite.py's Keras weights), checked against Keras; saved as .pth"""
    h5 = os.path.join("models", f"har_int8_{arch}_s{seed}.weights.h5")
    pth = os.path.join("models", f"har_fp32_{arch}_s{seed}.pth")
    if not os.path.exists(h5):
        sys.exit(f"[X] {h5} not found: run python har_tflite.py --arch {arch} --seed {seed} first")
    tm, km = from_keras(arch, h5)
    x, y = load("validation")
    with torch.no_grad():
        pt = tm(as_input(x)).numpy()
    kr = km.predict(x.astype(np.float32) / 127, verbose=0)
    agree = 100 * float((pt.argmax(1) == kr.argmax(1)).mean())
    print(f"    FP32 {arch} s{seed}: PyTorch vs Keras, max |logit difference| {np.abs(pt - kr).max():.2e}, same class "
          f"{agree:.2f}% | validation {100 * float((pt.argmax(1) == y).mean()):.2f}%")
    if agree < 99.9:
        sys.exit("[X] the PyTorch copy does not match the Keras network")
    torch.save(tm.state_dict(), pth)
    return tm, pth


def load_data():
    from lowbit.study import Data
    (xtr, ytr), (xva, yva), (xte, yte) = load("train"), load("validation"), load("test")
    t = lambda y: torch.as_tensor(y.astype(np.int64))
    return Data(as_input(xtr), t(ytr), as_input(xva), t(yva), as_input(xte), t(yte))


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(ARCHS), default="max")
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 2], choices=[2, 3, 4, 8])
    ap.add_argument("--seeds", type=int, default=3, help="QAT seeds: 42, 43, ... (default 3)")
    ap.add_argument("--epochs", type=int, default=15, help="QAT epochs (the SYMBEX CDT uses 15)")
    ap.add_argument("--no-distill", action="store_true", help="QAT on the labels only")
    ap.add_argument("--no-ptq", action="store_true", help="QAT only (skip the post-training methods)")
    ap.add_argument("--chip-only", action="store_true", help="no training: export and measure the saved models")
    ap.add_argument("--board", choices=["esp32", "s3"], help="measure the seed-42 models on this board")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--usb", action="store_true", help="the S3's native USB port (chip_test.py --usb)")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash", help="where the weights live on the board")
    ap.add_argument("--windows", type=int, default=1000, help="validation windows sent to the board")
    args = ap.parse_args()
    os.chdir(HERE)
    from lowbit.export_c import chip_study
    from lowbit.study import run, save, table
    tag = "_nodistill" if args.no_distill else ""
    model_dir = os.path.join("runs", f"lowbit_{args.arch}" + tag)
    data = load_data()
    if not args.chip_only:
        for s in range(43, 42 + args.seeds):                  # every seed's FP32 checked; seed 42 is the one quantized
            get_fp32(args.arch, s)
        model, pth = get_fp32(args.arch, 42)
        res = run(f"har_{args.arch}", model, data, bits=args.bits, seeds=[42 + k for k in range(args.seeds)],
                  epochs=args.epochs, distill=not args.no_distill, out_dir=model_dir, ptq=not args.no_ptq)
        res.update(arch=args.arch, fp32_model=pth, fp32_sha256=sha256(pth))
        table(res)
        save(res, os.path.join("results", "lowbit", f"lowbit_{args.arch}{tag}.json"))
    if args.board or args.chip_only:
        n = args.windows
        rows = chip_study(model_dir, data.x_val[:n].numpy(), data.y_val[:n].numpy(), args.bits, tag,
                          os.path.join("runs", f"lowbit_chip_{args.arch}"), args.board, args.port, args.usb, args.weights)
        if args.board:
            save(rows, os.path.join("results", "lowbit", f"lowbit_esp_nn_{args.arch}_{args.board}_{args.weights}{tag}.json"))


if __name__ == "__main__":
    main()
