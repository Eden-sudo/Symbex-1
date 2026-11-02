"""
pareto.py -- mlp_fashion benchmark: accuracy vs. weight bytes of every result in results/.

    python pareto.py                 # table on screen + results/pareto.json (+ results/pareto.png with matplotlib)
    python pareto.py --by total      # rank by total model bytes instead of packed weight bytes (earlier: --por)
    python pareto.py --symbex-dir runs/symbex_local_s42   # also a CDT output folder (train.json + export.json)

Reads the result files (no model, no PyTorch needed):
  symbex_s<seed>_train.json + symbex_s<seed>_export.json   SYMBEX 1 bit (CDT), one row per seed and the mean
  symbex_s<seed>_export.json ["int8"]                      the CDT's own INT8 reference
  tflite_pc.json                                           official TFLite INT8 (PC interpreter)
  int<bits>[_e<epochs>]_pc.json                            fashion_int4.py: INT8 / INT4 / INT2, PTQ and QAT
  --symbex-dir DIR                                         an extra CDT run (for example on a re-trained FP32)

Columns:
  acc 2000   accuracy on the first 2000 Fashion-MNIST test images (the images every side exports / validates)
  acc 10k    accuracy on the 10,000 test images, when the result has it (SYMBEX: acc_last of the CDT)
  FP32 10k   the 10k accuracy of the FP32 model that row was quantized from. models/fashion_mlp.pth is
             re-created by model.py when missing and is not bit-identical across machines / PyTorch
             versions: compare rows with the same FP32 value first.
  weights B  packed weight bytes only: ceil(n_weights * bits / 8)
  total B    what the result file reports as the model size (SYMBEX: header flash_bytes; TFLite: .tflite bytes;
             fashion_int4.py: weights + float32 channel scales + int32 biases)
Pareto (*): no other row has fewer or equal bytes AND a higher or equal acc 2000 (with one of them strict).
"""
import argparse
import glob
import json
import math
import os
import re

FASHION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # mlp_fashion/ (this script lives in earlier/)
ORIGINAL_SHAPE = [784, 256, 128, 10]                 # FashionMLP (model.py)


def weight_bytes(shape, bits):
    return sum(math.ceil(a * b * bits / 8) for a, b in zip(shape[:-1], shape[1:]))


def width_shape(width):
    return [ORIGINAL_SHAPE[0]] + [int(round(n * width)) for n in ORIGINAL_SHAPE[1:-1]] + [ORIGINAL_SHAPE[-1]]


def row(model, bits, acc, acc_10k, weights, total, fp32, source):
    return dict(modelo=model, bits=bits, acc_2000=acc, acc_10k=acc_10k, pesos_bytes=weights, total_bytes=total,
                fp32_10k=fp32, fuente=source)


def cdt_rows(base_name, export_path, train_path, source):
    """[SYMBEX row, CDT INT8 reference row] of one CDT run (train.json + export.json)."""
    e = json.load(open(export_path))
    t = json.load(open(train_path)) if os.path.exists(train_path) else {}
    width = t.get("width", 2.0)
    shape = width_shape(width)
    binary = set(t.get("plan", {"_": "bin"}).values()) == {"bin"}
    fp32 = t.get("references", {}).get("acc_fp32")
    rows = [row(f"{base_name} x{width:g} ({'-'.join(map(str, shape))})", 1 if binary else None, e.get("acc_integer", e.get("acc_entero")),
                  t.get("acc_last"), weight_bytes(shape, 1) if binary else None, e["flash_bytes"], fp32, source)]
    if "int8" in e:
        rows.append(row(f"INT8 CDT reference (PTQ){base_name.split('1-bit')[-1]}", 8, e["int8"]["acc_int8"], None,
                          weight_bytes(ORIGINAL_SHAPE, 8), e["int8"]["flash_bytes"], fp32, source))
    return rows


def symbex_rows():
    rows, seeds = [], []
    for exp in sorted(glob.glob(os.path.join("results", "earlier", "symbex_s*_export.json"))):
        m = re.fullmatch(r"symbex_s(\d+)_export\.json", os.path.basename(exp))
        if m is None:                                # a variant run (symbex_s42_nd_export.json...): not a base seed
            continue
        seed = m.group(1)
        s, *r8 = cdt_rows(f"SYMBEX 1-bit s{seed}", exp, os.path.join("results", "earlier", f"symbex_s{seed}_train.json"),
                           os.path.basename(exp))
        rows.append(s)
        seeds.append(s)
        if seed == "42":
            rows += r8
    if len(seeds) > 1:
        a10 = [s["acc_10k"] for s in seeds]
        rows.append(row(f"SYMBEX 1-bit mean of {len(seeds)} seeds", 1,
                          sum(s["acc_2000"] for s in seeds) / len(seeds),
                          sum(a10) / len(a10) if None not in a10 else None, seeds[0]["pesos_bytes"],
                          seeds[0]["total_bytes"], seeds[0]["fp32_10k"], "symbex_s*_export.json"))
    return rows


def symbex_dir_rows(folder):
    name = os.path.basename(os.path.normpath(folder))
    return cdt_rows(f"SYMBEX 1-bit [{name}]", os.path.join(folder, "export.json"),
                     os.path.join(folder, "train.json"), folder)


def tflite_rows(fp32_published):
    path = os.path.join("results", "tflite", "tflite_pc.json")
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    return [row("INT8 official TFLite (PTQ)", 8, d.get("accuracy", d.get("precision")), None, weight_bytes(ORIGINAL_SHAPE, 8),
                 d["tflite_bytes"], fp32_published, "tflite_pc.json")]


def int_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join("results", "earlier", "int*_pc.json")) + glob.glob(os.path.join("results", "lowbit", "int*_pc.json"))):
        d = json.load(open(path))
        b = d["bits"]
        for method, r in d["metodos"].items():
            if method == "qat":
                seeds = r.get("semillas", [r])
                label = f"INT{b} QAT {seeds[0]['epocas']} ep" + \
                           (f" mean of {len(seeds)} seeds" if len(seeds) > 1 else f" s{seeds[0]['semilla']}")
            elif "_e" in os.path.basename(path):         # PTQ does not depend on --epocas: listed once
                continue
            else:
                label = f"INT{b} {method.upper().replace('_', ' ')}"
            rows.append(row(label, b, r["precision"], r.get("precision_10k"), r["pesos_bytes"], r["total_bytes"],
                              d.get("acc_fp32_10k"), os.path.basename(path)))
    return rows


def mark_pareto(rows, key):
    for f in rows:
        f["pareto"] = f[key] is not None and not any(
            o is not f and o[key] is not None and o[key] <= f[key] and o["acc_2000"] >= f["acc_2000"]
            and (o[key] < f[key] or o["acc_2000"] > f["acc_2000"]) for o in rows)


def plot(rows, key, png_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not available: no PNG)")
        return False
    colors = {1: "#2563eb", 2: "#d97706", 4: "#059669", 8: "#6b7280"}
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    points = [f for f in rows if f[key] is not None and f["acc_2000"] > 80]
    for f in points:
        ax.scatter(f[key] / 1024, f["acc_2000"], s=50 if f["pareto"] else 26, color=colors.get(f["bits"], "#111"),
                   edgecolor="black" if f["pareto"] else "none", linewidth=0.8, zorder=3)
        ax.annotate(f["modelo"], (f[key] / 1024, f["acc_2000"]), textcoords="offset points", xytext=(5, 3),
                    fontsize=6, color="#333")
    front = sorted((f for f in rows if f["pareto"]), key=lambda f: f[key])
    ax.plot([f[key] / 1024 for f in front], [f["acc_2000"] for f in front], color="#111", linewidth=1,
            alpha=0.5, zorder=2, label="Pareto front")
    for b, c in sorted(colors.items()):
        ax.scatter([], [], color=c, label=f"{b}-bit weights")
    ax.set_xscale("log")
    ax.set_xlabel(("packed weight" if key == "pesos_bytes" else "total model") + " size (KB, log scale)")
    ax.set_ylabel("accuracy, first 2000 Fashion-MNIST test images (%)")
    ax.set_title("mlp_fashion: accuracy vs. size (rows below 80% omitted; see the table)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", "--por", dest="por", choices=["weights", "pesos", "total"], default="weights",
                    help="size used for the Pareto front: packed weight bytes (default) or total model bytes")
    ap.add_argument("--symbex-dir", action="append", default=[],
                    help="extra CDT output folder with train.json and export.json (repeatable)")
    args = ap.parse_args()
    os.chdir(FASHION_DIR)
    key = "total_bytes" if args.por == "total" else "pesos_bytes"   # "pesos": the earlier Spanish name of "weights"
    rows = symbex_rows()
    fp32_published = next((f["fp32_10k"] for f in rows if f["fp32_10k"] is not None), None)
    rows += tflite_rows(fp32_published) + int_rows()
    for folder in args.symbex_dir:
        rows += symbex_dir_rows(folder)
    mark_pareto(rows, key)
    rows.sort(key=lambda f: (f[key] is None, f[key] or 0, -f["acc_2000"]))

    def n(v, fmt):
        return format(v, fmt) if v is not None else "-"
    print(f"\n    {'model':<52}{'bits':>5}{'acc 2000':>10}{'acc 10k':>9}{'FP32 10k':>10}{'weights B':>11}"
          f"{'total B':>10}  P  source")
    print("    " + "-" * 130)
    for f in rows:
        print(f"    {f['modelo']:<52}{n(f['bits'], 'd'):>5}{f['acc_2000']:>9.2f}%{n(f['acc_10k'], '.2f'):>9}"
              f"{n(f['fp32_10k'], '.2f'):>10}{n(f['pesos_bytes'], 'd'):>11}{f['total_bytes']:>10}"
              f"  {'*' if f['pareto'] else ' '}  {f['fuente']}")
    print(f"    (* = on the Pareto front by {'packed weight' if key == 'pesos_bytes' else 'total'} bytes vs acc 2000)")
    suffix = "" if key == "pesos_bytes" else "_total"
    json.dump(dict(por=key, filas=rows), open(os.path.join("results", "earlier", f"pareto{suffix}.json"), "w"), indent=1)
    print(f"    -> results/earlier/pareto{suffix}.json")
    if plot(rows, key, os.path.join("results", "earlier", f"pareto{suffix}.png")):
        print(f"    -> results/earlier/pareto{suffix}.png")


if __name__ == "__main__":
    main()
