"""
fashion_lowbit.py -- mlp_fashion benchmark: the low-bit integer baselines (INT8, INT4, INT2) of the SAME FP32 model,
with the shared tool tools/lowbit (Brevitas quantization, ESP-NN integer arithmetic: multiply-accumulate, no SYMBEX
arithmetic).

    python fashion_lowbit.py                          # INT8, INT4, INT2: PTQ + QAT, 15 epochs, seeds 42, 43, 44
    python fashion_lowbit.py --no-distill             # QAT on the labels only (the other recipe)
    python fashion_lowbit.py --shape 128-512 --bits 8 # another FP32 shape (models/fashion_mlp_128_512.pth)
    python fashion_lowbit.py --chip-only --board s3 --port /dev/ttyACM0 --usb    # the seed-42 models on the board
    python fashion_lowbit.py --chip-only --board s3 --port /dev/ttyACM0 --usb --weights ram
    python fashion_lowbit.py --ablation --no-distill  # weights {1,2,4,8} x activations {binary, int8} bits (PC)

Run from benchmarks/mlp/mlp_fashion/ (environment env_bench). Data: the CDT's split (model.split_indices: the training
images minus the fixed validation split), validation on that split, the 10,000 test images evaluated once per final
model. Results: results/lowbit/lowbit[_h<h1>_<h2>][_nodistill].json; integer models in runs/lowbit/.
On the board (--board): the seed-42 QAT model of every bit width as C on ESP-NN (tools/lowbit/export_c.py), the first
--images test images, measured with tools/chip_test.py and checked against the integer simulation (must be 100 %);
results: results/lowbit/lowbit_esp_nn_<board>_<weights>[tag].json.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools")))


def load_data():
    from lowbit.study import Data
    from model import MEAN, STD, _fashion, split_indices
    tr, te = _fashion(True), _fashion(False)

    def norm(u8):
        return ((torch.as_tensor(u8, dtype=torch.float32) / 255.0 - MEAN) / STD)[:, None]

    keep, held = (np.array(i) for i in split_indices(len(tr.data)))
    x, y = tr.data.numpy(), tr.targets.numpy().astype(np.int64)
    return Data(norm(x[keep]), torch.as_tensor(y[keep]), norm(x[held]), torch.as_tensor(y[held]),
                norm(te.data.numpy()), torch.as_tensor(te.targets.numpy().astype(np.int64)))


def sha256(path):
    """the FP32 model is retrained per machine and not bit-identical: every result records which one it started from"""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 2], choices=[2, 3, 4, 8])
    ap.add_argument("--seeds", type=int, default=3, help="QAT seeds: 42, 43, ... (default 3)")
    ap.add_argument("--epochs", type=int, default=15, help="QAT epochs (the SYMBEX CDT uses 15)")
    ap.add_argument("--no-distill", action="store_true", help="QAT on the labels only")
    ap.add_argument("--no-ptq", action="store_true", help="QAT only (skip the post-training methods)")
    ap.add_argument("--shape", default="256-128", help="hidden widths h1-h2 of the FP32 model (as fashion_sizing.py)")
    ap.add_argument("--fp32-epochs", type=int, default=None,
                    help="epochs of the FP32 model (default model.EPOCHS); others: results and models get _f<N>")
    ap.add_argument("--chip-only", action="store_true", help="no training: export and measure the saved models")
    ap.add_argument("--ablation", action="store_true",
                    help="binary weights or binary activations? weights 1/2/4/8 bits x activations binary/int8, "
                         "fake-quant on the PC; results/lowbit/lowbit_ablation[tag].json")
    ap.add_argument("--board", choices=["esp32", "s3"], help="measure the seed-42 models on this board")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--usb", action="store_true", help="the S3's native USB port (chip_test.py --usb)")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash", help="where the weights live on the board")
    ap.add_argument("--images", type=int, default=1000, help="test images sent to the board")
    args = ap.parse_args()
    os.chdir(HERE)
    from lowbit.export_c import chip_study
    from lowbit.study import ablation, run, save, table
    from model import FashionMLP, get_fp32
    h1, h2 = (int(v) for v in args.shape.split("-"))
    shape_tag = "" if (h1, h2) == (256, 128) else f"_h{h1}_{h2}"
    from model import EPOCHS
    fp32_epochs = args.fp32_epochs or EPOCHS
    shape_tag += "" if fp32_epochs == EPOCHS else f"_f{fp32_epochs}"
    tag = shape_tag + ("_nodistill" if args.no_distill else "")
    model_dir = os.path.join("runs", "lowbit" + tag)
    data = load_data()
    if args.ablation:
        model = FashionMLP(h1, h2)
        model.load_state_dict(torch.load(get_fp32(h1, h2, fp32_epochs), map_location="cpu"))
        res = ablation("mlp_fashion", model, data, seeds=[42 + k for k in range(args.seeds)], epochs=args.epochs,
                       distill=not args.no_distill)
        res.update(shape=f"784-{h1}-{h2}-10", fp32_sha256=sha256(get_fp32(h1, h2, fp32_epochs)))
        save(res, os.path.join("results", "lowbit", f"lowbit_ablation{tag}.json"))
        return
    if not args.chip_only:
        model = FashionMLP(h1, h2)
        model.load_state_dict(torch.load(get_fp32(h1, h2, fp32_epochs), map_location="cpu"))
        res = run("mlp_fashion", model, data, bits=args.bits, seeds=[42 + k for k in range(args.seeds)],
                  epochs=args.epochs, distill=not args.no_distill, out_dir=model_dir, ptq=not args.no_ptq)
        res.update(shape=f"784-{h1}-{h2}-10", fp32_sha256=sha256(get_fp32(h1, h2, fp32_epochs)))
        table(res)
        save(res, os.path.join("results", "lowbit", f"lowbit{tag}.json"))
    if args.board or args.chip_only:
        rows = chip_study(model_dir, data.x_test[:args.images].numpy(), data.y_test[:args.images].numpy(), args.bits,
                          "_nodistill" if args.no_distill else "", os.path.join("runs", "lowbit_chip" + shape_tag),
                          args.board, args.port, args.usb, args.weights)
        if args.board:
            save(rows, os.path.join("results", "lowbit", f"lowbit_esp_nn_{args.board}_{args.weights}{tag}.json"))


if __name__ == "__main__":
    main()
