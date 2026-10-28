"""
fomo_symbex.py -- the fomo_voc_person benchmark: the binary FOMO detector, trained, exported to lib_symbex and measured.

    python fomo_symbex.py --channels 32-32-128 --seeds 3                                  # train + export (PC)
    python fomo_symbex.py --channels 32-32-128 --seeds 3 --board esp32 --port /dev/ttyUSB0  # ... and measure on the board
    python fomo_symbex.py --channels 32-32-128 --seeds 3 --board esp32 --weights ram
    python fomo_symbex.py --channels 32-64-128 --seeds 1 --board esp32 --popcount swar

Run from benchmarks/conv2d/fomo_voc_person/ (main environment, .venv; a GPU helps the training). For every seed:
fomo_train.py (skipped if runs/<name>_s<seed>/ already holds model.pth and train.json; --redo trains it again),
fomo_export.py (always: the C check on the PC, EXACTNESS must be 100%) and, with --board, tools/chip_test.py on the
same images (equality with the integer simulation must be 100%; latency; RAM). Every number in the summary is read
from the files these scripts write (train.json, export.json, chip_<board>_<popcount>_<weights>_mask.json), never typed.
Summary: results/symbex_seeds_<board>_<popcount>[_ram]_<name>.json (results/symbex_seeds_pc_<name>.json without a
board), <name> = c<channels>_b<blocks>_w<weight>_e<epochs>.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools"))
LIB = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Symbex1_PIO", "lib", "lib_symbex"))


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def run(cmd, cwd=HERE):
    print("\n[*] " + " ".join(os.path.relpath(c) if os.path.isabs(c) else c for c in cmd[1:]))
    return subprocess.run(cmd, cwd=cwd).returncode == 0


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="32-32-128")
    ap.add_argument("--blocks", default="1-2-2")
    ap.add_argument("--weight", type=float, default=40.0, help="weight of the person cells in the loss (default 40)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds: 42, 43, ... (default 1)")
    ap.add_argument("--images", type=int, default=1000, help="test images for the C check and the chip (default 1000)")
    ap.add_argument("--board", choices=["esp32", "s3"], help="measure on this board")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash", help="where the weights live on the board")
    ap.add_argument("--popcount", choices=["swar", "lut16"], default="lut16", help="counting mode (swar: minimum RAM)")
    ap.add_argument("--redo", action="store_true", help="train again seeds that are already trained")
    args = ap.parse_args()
    os.chdir(HERE)
    os.makedirs("results", exist_ok=True)
    name = f"c{args.channels}_b{args.blocks}_w{args.weight:g}_e{args.epochs}"
    rows = []
    for seed in [42 + i for i in range(args.seeds)]:
        out_dir = os.path.join("runs", f"{name}_s{seed}")
        trained = os.path.exists(os.path.join(out_dir, "model.pth")) and os.path.exists(os.path.join(out_dir, "train.json"))
        if trained and not args.redo:
            print(f"\n[*] seed {seed}: already trained in {out_dir} (--redo to train it again): exporting again")
        elif not run([sys.executable, "fomo_train.py", "--channels", args.channels, "--blocks", args.blocks,
                      "--weight", f"{args.weight:g}", "--epochs", str(args.epochs), "--seed", str(seed)]):
            print(f"[X] the training failed on seed {seed}")
            continue
        if not run([sys.executable, "fomo_export.py", out_dir, "--images", str(args.images)]):
            print(f"[X] the export failed on seed {seed}")
            continue
        chip_json = os.path.join(out_dir, f"chip_{args.board}_{args.popcount}_{args.weights}_mask.json")
        if args.board:
            if os.path.exists(chip_json):
                os.remove(chip_json)                 # a failed measurement must not leave an older one in the summary
            if not run([sys.executable, os.path.join(TOOLS, "chip_test.py"), os.path.abspath(out_dir),
                        "--board", args.board, "--port", args.port, "--popcount", args.popcount,
                        "--weights", args.weights, "--samples", str(args.images), "--lib-path", LIB], cwd=TOOLS):
                print(f"[X] the chip measurement failed on seed {seed}")
        tr, ex = load(os.path.join(out_dir, "train.json")), load(os.path.join(out_dir, "export.json"))
        row = dict(seed=seed, f1_val=tr.get("calibrated_validation", {}).get("f1"),
                   f1_test=tr.get("calibrated_test", {}).get("f1"), f1_sim=ex.get("f1_sim", {}).get("f1"),
                   flash_kb=(ex.get("flash_bytes") or 0) / 1024, exactness=ex.get("exactness"), images=ex.get("images"))
        ch = load(chip_json) if args.board else {}
        if ch:
            row.update(equal=ch.get("equal_simulation"), latency_ms=ch.get("median_ms"), min_ms=ch.get("min_ms"),
                       ram_kb=(ch.get("ram") or {}).get("total", 0) / 1024)
        rows.append(row)

    print(f"\n    SEEDS  (binary FOMO {args.channels} / {args.blocks}, person weight {args.weight:g}, {args.epochs} epochs; "
          f"F1 with the calibrated threshold; test = the 4,952 VOC 2007 test images, used for no choice)")
    head = f"    {'seed':>5}{'val F1':>9}{'test F1':>9}{'sim F1':>9}{'flash KB':>10}{'C exact':>9}"
    if args.board:
        head += f"{'ms':>10}{'equal':>8}{'RAM KB':>9}"
    print(head)
    for r in rows:
        line = (f"    {r['seed']:>5}{(r['f1_val'] or 0):>8.2f}%{(r['f1_test'] or 0):>8.2f}%{(r['f1_sim'] or 0):>8.2f}%"
                f"{r['flash_kb']:>10.1f}{(str(round(r['exactness'])) + '%') if r['exactness'] is not None else '-':>9}")
        if args.board and "latency_ms" in r:
            line += f"{(r['latency_ms'] or 0):>10.3f}{(r['equal'] or 0):>7.0f}%{r['ram_kb']:>9.1f}"
        print(line)
    f1, f1_sd = mean_sd([r["f1_test"] for r in rows])
    if f1 is not None:
        print(f"    test F1 over {len(rows)} seed(s): {f1:.2f} +- {f1_sd:.2f} %")
    ms, ms_sd = mean_sd([r.get("latency_ms") for r in rows])
    if ms is not None:
        print(f"    latency over {len(rows)} seed(s): {ms:.3f} +- {ms_sd:.3f} ms ({args.popcount}, weights in {args.weights})")
    tag = f"{args.board}_{args.popcount}{'_ram' if args.weights == 'ram' else ''}" if args.board else "pc"
    path = os.path.join("results", f"symbex_seeds_{tag}_{name}.json")
    json.dump(dict(channels=args.channels, blocks=args.blocks, weight=args.weight, epochs=args.epochs, board=args.board,
                   weights=args.weights, popcount=args.popcount, seeds=rows),
              open(path, "w"), indent=1)
    print(f"    -> {path}")


if __name__ == "__main__":
    main()
