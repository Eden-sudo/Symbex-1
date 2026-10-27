"""
resnet8_symbex.py -- the cnn_cifar_resnet8 benchmark: a fully binary ResNet-8, through the CDT and lib_symbex.

    python resnet8_symbex.py --width 1 --seeds 3                                   # train + export (PC)
    python resnet8_symbex.py --width 1 --seeds 3 --board s3 --port /dev/ttyUSB0   # ... and measure on the board
    python resnet8_symbex.py --width 1 --seeds 3 --board esp32 --weights ram
    python resnet8_symbex.py --width 2 --seeds 3                                   # twice the channels

Run from benchmarks/conv2d/cnn_cifar_resnet8/ (main environment, .venv; a GPU helps). The network (model.py) has the shape of
the MLPerf Tiny ResNet-8 that resnet8_tflite.py measures in INT8 (--width 1: the same 16-32-64 channels); only what
binarization imposes changes: a thermometer input, 1-bit weights and activations, OR shortcuts. The CDT
(symbex.toml, [model] method = "boolean") trains it from scratch on the labels only, folds it into integer thresholds,
exports it to lib_symbex (layers/conv/symbex_bool) and, with --board, measures it with the same firmware and the same
tables as every other benchmark (100% equal to the integer simulation, accuracy, latency per layer, RAM).
Each seed lands in runs/w<width>_s<seed>/; a seed already trained is only exported again (seconds; the export depends
on the board), --redo trains it again. Summary: results/symbex_seeds_<board>_<popcount>[_ram]_w<width>.json
(results/symbex_seeds_pc_w<width>.json without a board).
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CDT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "cdt.py"))


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def config_for(width, out_dir, spec=None):
    """symbex.toml with this network's definition and starting weights, written into the run folder. spec: a variant
    of the sizing study ("16-32-64/2-2-2": channels / blocks per stage) instead of a width"""
    sys.path.insert(0, HERE)
    import importlib.util
    import model as m
    if spec:
        definition = m.write_definition(spec, out_dir)
        path, cls = definition.split(":")
        sp = importlib.util.spec_from_file_location("variant_model", path)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        init = m.initial_weights(getattr(mod, cls), os.path.join("models", f"bool_resnet8_{m.spec_name(spec)}_init.pth"))
    else:
        name = "BoolResNet8x1" if width == 1 else "BoolResNet8x2"
        init = m.initial_weights(getattr(m, name), os.path.join("models", f"bool_resnet8_x{width:g}_init.pth"))
        definition = f"{os.path.join(HERE, 'model.py')}:{name}"
    text = open("symbex.toml").read()
    text = re.sub(r'(?m)^file = .*$', f'file = "{os.path.abspath(init)}"', text)
    text = re.sub(r'(?m)^definition = .*$', f'definition = "{definition}"', text)
    text = re.sub(r'(?m)^directory = "\.\./\.\./data/cifar10"', f'directory = "{os.path.normpath(os.path.join(HERE, "..", "..", "data", "cifar10"))}"', text)
    text = re.sub(r'(?m)^lib_path = .*$', f'lib_path = "{os.path.normpath(os.path.join(HERE, "..", "..", "..", "Symbex1_PIO", "lib", "lib_symbex"))}"', text)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.toml")
    open(path, "w").write(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=1, choices=[1, 2], help="1: 16-32-64 channels (the INT8 reference's)")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds: 42, 43, ... (default 1)")
    ap.add_argument("--board", choices=["esp32", "s3"], help="measure on this board")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash", help="where the weights live on the board")
    ap.add_argument("--popcount", choices=["swar", "lut16"], default="swar", help="counting mode (swar: minimum RAM)")
    ap.add_argument("--epochs", type=int, default=0, help="override [training] epochs (a quick check)")
    ap.add_argument("--redo", action="store_true", help="train again seeds that are already trained")
    ap.add_argument("--spec", default="", help='a variant of the sizing study: "16-32-64/2-2-2" = channels / blocks '
                                               'per stage (replaces --width)')
    args = ap.parse_args()
    os.chdir(HERE)
    os.makedirs("results", exist_ok=True)
    rows = []
    if args.spec:
        sys.path.insert(0, HERE)
        from model import spec_name
        name = spec_name(args.spec) + (f"_e{args.epochs}" if args.epochs else "")
    else:
        name = f"w{args.width:g}" + (f"_e{args.epochs}" if args.epochs else "")
    for seed in [42 + i for i in range(args.seeds)]:
        out_dir = os.path.join("runs", f"{name}_s{seed}")
        trained = os.path.exists(os.path.join(out_dir, "model_binarized.pth")) and os.path.exists(os.path.join(out_dir, "train.json"))
        cmd = [sys.executable, CDT, config_for(args.width, out_dir, args.spec or None), "--seed", str(seed),
               "--output-directory", out_dir]
        if trained and not args.redo:
            print(f"\n[*] seed {seed}: already trained in {out_dir} (--redo to train it again): exporting again")
            cmd += ["--from", "export"]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if args.board:
            cmd += ["--board", args.board, "--port", args.port, "--weights", args.weights, "--popcount", args.popcount]
        else:
            cmd += ["--to", "export"]
        print("\n[*] " + " ".join(os.path.relpath(c) if os.path.isabs(c) else c for c in cmd[1:]))
        if subprocess.run(cmd).returncode != 0:
            print(f"[X] the CDT failed on seed {seed}")
        tr, ex, ch = (load(os.path.join(out_dir, f)) for f in ("train.json", "export.json", "chip.json"))
        # test accuracy: acc_test since the CDT keeps a validation split apart (train.json n_test); in runs made before,
        # the CDT validated on the test set itself, so their acc_last IS the test accuracy
        test_acc = tr.get("acc_test") if "acc_test" in tr else tr.get("acc_last")
        row = dict(seed=seed, width=args.width, val_accuracy=tr.get("acc_last") if "acc_test" in tr else None,
                   test_accuracy=test_acc, flash_kb=(ex.get("flash_bytes") or 0) / 1024,
                   exactness=ex.get("exactness"))
        if args.board and ch:
            row.update(chip_accuracy=ch.get("accuracy"), latency_ms=ch.get("median_ms"), equal=ch.get("equal_simulation"),
                       ram_kb=(ch.get("ram") or {}).get("total", 0) / 1024)
        rows.append(row)

    print(f"\n    SEEDS  (fully binary ResNet-8, width x{args.width:g}; last epoch; test = the 10,000 CIFAR-10 test images, "
          f"used for no choice; val = the validation split carved out of the training data, '-' in older runs)")
    head = f"    {'seed':>5}{'test acc':>10}{'val acc':>10}{'flash KB':>10}{'C exact':>9}"
    if args.board:
        head += f"{'chip acc':>10}{'ms':>10}{'equal':>8}{'RAM KB':>9}"
    print(head)
    for r in rows:
        val = f"{r['val_accuracy']:.2f}%" if r["val_accuracy"] is not None else "-"
        line = (f"    {r['seed']:>5}{(r['test_accuracy'] or 0):>9.2f}%{val:>10}{r['flash_kb']:>10.1f}"
                f"{(str(round(r['exactness'])) + '%') if r['exactness'] is not None else '-':>9}")
        if args.board and "latency_ms" in r:
            line += f"{(r['chip_accuracy'] or 0):>9.2f}%{(r['latency_ms'] or 0):>10.2f}{(r['equal'] or 0):>7.0f}%{r['ram_kb']:>9.2f}"
        print(line)
    accs = [r["test_accuracy"] for r in rows if r["test_accuracy"] is not None]
    if accs:
        import statistics
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"    test accuracy over {len(accs)} seed(s): {statistics.mean(accs):.2f} +- {sd:.2f} %")
    tag = f"{args.board}_{args.popcount}{'_ram' if args.weights == 'ram' else ''}" if args.board else "pc"
    path = os.path.join("results", f"symbex_seeds_{tag}_{name}.json")
    json.dump(dict(width=args.width, spec=args.spec or None, board=args.board, weights=args.weights, popcount=args.popcount,
                   seeds=rows),
              open(path, "w"), indent=1)
    print(f"    -> {path}")


if __name__ == "__main__":
    main()
