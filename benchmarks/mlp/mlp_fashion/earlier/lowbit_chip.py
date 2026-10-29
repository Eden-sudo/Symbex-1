"""
lowbit_chip.py -- the low-bit study on the board: the same table as lowbit_study.py, plus the latency on the chip.

    python lowbit_chip.py --board s3 --port /dev/ttyUSB0                  # weights in flash: the usual deployment
    python lowbit_chip.py --board s3 --port /dev/ttyUSB0 --weights ram    # weights in RAM: when the memory allows it
    python lowbit_chip.py --board s3 --show                               # only the table of what is measured

Run from benchmarks/mlp/mlp_fashion/ after lowbit_study.py. Every method runs through the same tool, tools/chip_test.py
(the CDT's chip stage): the same firmware, the same 1,000 test images, the same weights placement (--weights: flash by
default, as a model is usually deployed; ram as the faster variant), and the board's result checked against the method's integer simulation, image by image (must be 100 %):
    SYMBEX 1 bit x1, x1.5, x2 and ternary x1, x1.5   the CDT exports of lowbit_study.py, seed 42
    INT4, INT2                                        int_export.py (bit planes summed by the HAL, no multiplication),
                                                      seed 42; their models are saved by fashion_int4.py
    INT8                                              the official TensorFlow Lite Micro model, as measured by
                                                      fashion_tflite.py (results/tflite/tflite_<board>.json in flash,
                                                      results/tflite/tflite_<board>_ram.json with --model-ram)
Accuracy columns come from results/lowbit/lowbit_study.json (3 seeds); latency from the seed-42 export of each method.
Results: results/lowbit/lowbit_chip_<board>_<weights>.json.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # mlp_fashion/ (this script lives in earlier/)
CHIP_TEST = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "chip_test.py"))
CDT_ROWS = [("SYMBEX 1 bit x1", "runs/lowbit/plain_w1_s42"), ("SYMBEX 1 bit x1.5", "runs/lowbit/plain_w1.5_s42"),
            ("SYMBEX 1 bit x2", "runs/lowbit/plain_w2_s42"), ("ternary x1", "runs/lowbit/ternary_w1_s42"),
            ("ternary x1.5", "runs/lowbit/ternary_w1.5_s42")]
INT_ROWS = [("INT2 QAT x1", 2), ("INT4 QAT x1", 4)]


def find(d, names):
    """the first of the candidate keys present in a result file (the chip files are read, not assumed)"""
    for n in names:
        if isinstance(d, dict) and d.get(n) is not None:
            return d[n]
    return None


def chip_result(folder, board, weights):
    files = sorted(glob.glob(os.path.join(HERE, folder, f"chip_{board}_*_{weights}_*.json")), key=os.path.getmtime)
    if not files:
        return None
    d = json.load(open(files[-1]))
    lat = find(d, ["latency_median_ms", "median_ms", "latency_ms", "ms"])
    eq = find(d, ["equal_simulation", "equal_to_simulation", "equal"])
    if lat is None:
        print(f"    [!] {os.path.basename(files[-1])}: no latency key among {list(d)[:20]}: tell Claude its name")
    return dict(file=os.path.relpath(files[-1], HERE), latency_ms=lat, equal=eq,
                accuracy=find(d, ["accuracy", "acc", "accuracy_pct"]), ram=find(d, ["ram_total", "ram_bytes", "ram"]))


def measure(folder, args):
    cmd = [sys.executable, CHIP_TEST, os.path.join(HERE, folder), "--board", args.board, "--port", args.port,
           "--weights", args.weights, "--no-int8"] + (["--usb"] if args.usb else [])
    if subprocess.run(cmd, cwd=os.path.dirname(CHIP_TEST)).returncode != 0:
        sys.exit(f"[X] chip_test.py failed on {folder} (the output above says why)")


def ensure_int(bits):
    folder = f"runs/lowbit_chip/int{bits}_s42"
    if os.path.exists(os.path.join(HERE, folder, "export.json")):
        return folder
    if not os.path.exists(os.path.join(HERE, "models", f"int{bits}_qat_s42.npz")):
        print(f"\n[*] INT{bits}: no saved model yet: fashion_int4.py again (3 seeds, the same results, now saved)")
        r = subprocess.run([sys.executable, os.path.join("earlier", "fashion_int4.py"), "--bits", str(bits), "--epochs", "15", "--seeds", "3"],
                           cwd=HERE)
        if r.returncode != 0 or not os.path.exists(os.path.join(HERE, "models", f"int{bits}_qat_s42.npz")):
            sys.exit(f"[X] models/int{bits}_qat_s42.npz was not saved: was patch_lowbit_chip.py applied?")
    if subprocess.run([sys.executable, os.path.join("earlier", "int_export.py"), "--bits", str(bits), "--seed", "42"], cwd=HERE).returncode != 0:
        sys.exit(f"[X] int_export.py --bits {bits} failed")
    return folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=["esp32", "s3"], default="s3")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash",
                    help="where every model's weights live on the board (default flash, the usual deployment)")
    ap.add_argument("--show", action="store_true", help="only the table of what is already measured")
    ap.add_argument("--usb", action="store_true", help="the S3's native USB port (chip_test.py --usb)")
    args = ap.parse_args()
    os.chdir(HERE)
    study = {r["model"].split(" (")[0]: r for r in json.load(open("results/lowbit/lowbit_study.json"))["rows"]}
    rows = [(name, folder) for name, folder in CDT_ROWS] + [(name, ensure_int(b) if not args.show else
                                                             f"runs/lowbit_chip/int{b}_s42") for name, b in INT_ROWS]
    table = []
    for name, folder in rows:
        if not args.show and chip_result(folder, args.board, args.weights) is None:
            if not os.path.exists(os.path.join(HERE, folder, "export.json")):
                print(f"[!] {folder}: no export (run lowbit_study.py first): skipped")
                continue
            print(f"\n[*] {name} on the {args.board}: {folder}")
            measure(folder, args)
        c = chip_result(folder, args.board, args.weights) or {}
        s = study.get(name, {})
        table.append(dict(model=name, weight_kb=s.get("weight_bytes", 0) / 1024, test=s.get("test", [None, None]),
                          int2000=s.get("int2000"), latency_ms=c.get("latency_ms"), equal=c.get("equal"), file=c.get("file")))
    # fashion_tflite.py: tflite_<board>.json with the model in flash, tflite_<board>_ram.json with --model-ram
    tfl = os.path.join("results", "tflite", f"tflite_{args.board}{'_ram' if args.weights == 'ram' else ''}.json")
    if os.path.exists(tfl):
        d = json.load(open(tfl))
        s = study.get("INT8 TFLite x1", {})
        placed = "ram" if d.get("model_ram") else "flash"
        same = placed == args.weights
        if not same:
            print(f"\n    [!] {tfl} measured the INT8 model in {placed}, the others ran with the weights in {args.weights}: "
                  f"not compared. Measure it with fashion_tflite.py --board {args.board}"
                  + (" --model-ram" if args.weights == "ram" else "") + ", then run this again with --show")
        table.append(dict(model=f"INT8 TFLite x1 ({placed})", weight_kb=s.get("weight_bytes", 0) / 1024, test=[None, None],
                          int2000=s.get("int2000"), latency_ms=find(d, ["median_ms", "latency_ms"]) if same else None,
                          equal=find(d, ["equal_pc", "equal"]), file=tfl))
    ref = next((r["latency_ms"] for r in table if r["model"].startswith("INT8") and r["latency_ms"]), None)
    print(f"\n    LOW-BIT STUDY ON THE CHIP  ({args.board}, weights in {args.weights}; accuracy: 3 seeds; latency: the seed-42"
          f" export, median over the test images)\n")
    print(f"    {'model':<20}{'weights KB':>11}{'test acc':>16}{'int 2000':>10}{'latency ms':>12}{'vs INT8':>9}{'equal':>8}")
    print("    " + "-" * 86)
    for r in table:
        t, sd = r["test"] if r["test"] else (None, None)
        tt = f"{t:.2f} +- {sd:.2f}" if t is not None else "-"
        lat = f"{r['latency_ms']:.3f}" if r["latency_ms"] else "-"
        vs = f"{ref / r['latency_ms']:.2f}x" if ref and r["latency_ms"] else "-"
        eq = f"{r['equal']:.0f}%" if isinstance(r["equal"], (int, float)) else "-"
        i2 = f"{r['int2000']:.2f}" if r["int2000"] is not None else "-"
        print(f"    {r['model']:<20}{r['weight_kb']:>11.1f}{tt:>16}{i2:>10}{lat:>12}{vs:>9}{eq:>8}")
    print("\n    vs INT8: how many times faster than the official TFLite Micro INT8 model | equal: the board against the "
          "method's integer simulation")
    json.dump(table, open(os.path.join("results", "lowbit", f"lowbit_chip_{args.board}_{args.weights}.json"), "w"), indent=1)
    print(f"    -> results/lowbit/lowbit_chip_{args.board}_{args.weights}.json")


if __name__ == "__main__":
    main()
