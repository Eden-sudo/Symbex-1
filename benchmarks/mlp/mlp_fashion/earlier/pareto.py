"""
pareto.py -- mlp_fashion benchmark: accuracy vs. weight bytes of every result in results/.

    python pareto.py                 # table on screen + results/pareto.json (+ results/pareto.png with matplotlib)
    python pareto.py --por total     # rank by total model bytes instead of packed weight bytes
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

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # mlp_fashion/ (this script lives in earlier/)
FORMA_ORIGINAL = [784, 256, 128, 10]                 # FashionMLP (model.py)


def bytes_pesos(forma, bits):
    return sum(math.ceil(a * b * bits / 8) for a, b in zip(forma[:-1], forma[1:]))


def forma_ancho(ancho):
    return [FORMA_ORIGINAL[0]] + [int(round(n * ancho)) for n in FORMA_ORIGINAL[1:-1]] + [FORMA_ORIGINAL[-1]]


def fila(modelo, bits, acc, acc_10k, pesos, total, fp32, fuente):
    return dict(modelo=modelo, bits=bits, acc_2000=acc, acc_10k=acc_10k, pesos_bytes=pesos, total_bytes=total,
                fp32_10k=fp32, fuente=fuente)


def filas_cdt(nombre_base, ruta_export, ruta_train, fuente):
    """[SYMBEX row, CDT INT8 reference row] of one CDT run (train.json + export.json)."""
    e = json.load(open(ruta_export))
    t = json.load(open(ruta_train)) if os.path.exists(ruta_train) else {}
    ancho = t.get("width", 2.0)
    forma = forma_ancho(ancho)
    binario = set(t.get("plan", {"_": "bin"}).values()) == {"bin"}
    fp32 = t.get("references", {}).get("acc_fp32")
    filas = [fila(f"{nombre_base} x{ancho:g} ({'-'.join(map(str, forma))})", 1 if binario else None, e["acc_entero"],
                  t.get("acc_last"), bytes_pesos(forma, 1) if binario else None, e["flash_bytes"], fp32, fuente)]
    if "int8" in e:
        filas.append(fila(f"INT8 CDT reference (PTQ){nombre_base.split('1-bit')[-1]}", 8, e["int8"]["acc_int8"], None,
                          bytes_pesos(FORMA_ORIGINAL, 8), e["int8"]["flash_bytes"], fp32, fuente))
    return filas


def filas_symbex():
    filas, semillas = [], []
    for exp in sorted(glob.glob(os.path.join("results", "symbex_s*_export.json"))):
        semilla = re.search(r"symbex_s(\d+)_export", exp).group(1)
        s, *r8 = filas_cdt(f"SYMBEX 1-bit s{semilla}", exp, os.path.join("results", f"symbex_s{semilla}_train.json"),
                           os.path.basename(exp))
        filas.append(s)
        semillas.append(s)
        if semilla == "42":
            filas += r8
    if len(semillas) > 1:
        a10 = [s["acc_10k"] for s in semillas]
        filas.append(fila(f"SYMBEX 1-bit mean of {len(semillas)} seeds", 1,
                          sum(s["acc_2000"] for s in semillas) / len(semillas),
                          sum(a10) / len(a10) if None not in a10 else None, semillas[0]["pesos_bytes"],
                          semillas[0]["total_bytes"], semillas[0]["fp32_10k"], "symbex_s*_export.json"))
    return filas


def filas_symbex_dir(carpeta):
    nombre = os.path.basename(os.path.normpath(carpeta))
    return filas_cdt(f"SYMBEX 1-bit [{nombre}]", os.path.join(carpeta, "export.json"),
                     os.path.join(carpeta, "train.json"), carpeta)


def filas_tflite(fp32_publicado):
    ruta = os.path.join("results", "tflite_pc.json")
    if not os.path.exists(ruta):
        return []
    d = json.load(open(ruta))
    return [fila("INT8 official TFLite (PTQ)", 8, d["precision"], None, bytes_pesos(FORMA_ORIGINAL, 8),
                 d["tflite_bytes"], fp32_publicado, "tflite_pc.json")]


def filas_int():
    filas = []
    for ruta in sorted(glob.glob(os.path.join("results", "int*_pc.json"))):
        d = json.load(open(ruta))
        b = d["bits"]
        for metodo, r in d["metodos"].items():
            if metodo == "qat":
                semillas = r.get("semillas", [r])
                etiqueta = f"INT{b} QAT {semillas[0]['epocas']} ep" + \
                           (f" mean of {len(semillas)} seeds" if len(semillas) > 1 else f" s{semillas[0]['semilla']}")
            elif "_e" in os.path.basename(ruta):         # PTQ does not depend on --epocas: listed once
                continue
            else:
                etiqueta = f"INT{b} {metodo.upper().replace('_', ' ')}"
            filas.append(fila(etiqueta, b, r["precision"], r.get("precision_10k"), r["pesos_bytes"], r["total_bytes"],
                              d.get("acc_fp32_10k"), os.path.basename(ruta)))
    return filas


def marcar_pareto(filas, clave):
    for f in filas:
        f["pareto"] = f[clave] is not None and not any(
            o is not f and o[clave] is not None and o[clave] <= f[clave] and o["acc_2000"] >= f["acc_2000"]
            and (o[clave] < f[clave] or o["acc_2000"] > f["acc_2000"]) for o in filas)


def grafico(filas, clave, ruta_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not available: no PNG)")
        return False
    colores = {1: "#2563eb", 2: "#d97706", 4: "#059669", 8: "#6b7280"}
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    puntos = [f for f in filas if f[clave] is not None and f["acc_2000"] > 80]
    for f in puntos:
        ax.scatter(f[clave] / 1024, f["acc_2000"], s=50 if f["pareto"] else 26, color=colores.get(f["bits"], "#111"),
                   edgecolor="black" if f["pareto"] else "none", linewidth=0.8, zorder=3)
        ax.annotate(f["modelo"], (f[clave] / 1024, f["acc_2000"]), textcoords="offset points", xytext=(5, 3),
                    fontsize=6, color="#333")
    frente = sorted((f for f in filas if f["pareto"]), key=lambda f: f[clave])
    ax.plot([f[clave] / 1024 for f in frente], [f["acc_2000"] for f in frente], color="#111", linewidth=1,
            alpha=0.5, zorder=2, label="Pareto front")
    for b, c in sorted(colores.items()):
        ax.scatter([], [], color=c, label=f"{b}-bit weights")
    ax.set_xscale("log")
    ax.set_xlabel(("packed weight" if clave == "pesos_bytes" else "total model") + " size (KB, log scale)")
    ax.set_ylabel("accuracy, first 2000 Fashion-MNIST test images (%)")
    ax.set_title("mlp_fashion: accuracy vs. size (rows below 80% omitted; see the table)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta_png)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por", choices=["pesos", "total"], default="pesos",
                    help="size used for the Pareto front: packed weight bytes (default) or total model bytes")
    ap.add_argument("--symbex-dir", action="append", default=[],
                    help="extra CDT output folder with train.json and export.json (repeatable)")
    args = ap.parse_args()
    os.chdir(AQUI)
    clave = "pesos_bytes" if args.por == "pesos" else "total_bytes"
    filas = filas_symbex()
    fp32_publicado = next((f["fp32_10k"] for f in filas if f["fp32_10k"] is not None), None)
    filas += filas_tflite(fp32_publicado) + filas_int()
    for carpeta in args.symbex_dir:
        filas += filas_symbex_dir(carpeta)
    marcar_pareto(filas, clave)
    filas.sort(key=lambda f: (f[clave] is None, f[clave] or 0, -f["acc_2000"]))

    def n(v, fmt):
        return format(v, fmt) if v is not None else "-"
    print(f"\n    {'model':<52}{'bits':>5}{'acc 2000':>10}{'acc 10k':>9}{'FP32 10k':>10}{'weights B':>11}"
          f"{'total B':>10}  P  source")
    print("    " + "-" * 130)
    for f in filas:
        print(f"    {f['modelo']:<52}{n(f['bits'], 'd'):>5}{f['acc_2000']:>9.2f}%{n(f['acc_10k'], '.2f'):>9}"
              f"{n(f['fp32_10k'], '.2f'):>10}{n(f['pesos_bytes'], 'd'):>11}{f['total_bytes']:>10}"
              f"  {'*' if f['pareto'] else ' '}  {f['fuente']}")
    print(f"    (* = on the Pareto front by {'packed weight' if clave == 'pesos_bytes' else 'total'} bytes vs acc 2000)")
    sufijo = "" if clave == "pesos_bytes" else "_total"
    json.dump(dict(por=clave, filas=filas), open(os.path.join("results", f"pareto{sufijo}.json"), "w"), indent=1)
    print(f"    -> results/pareto{sufijo}.json")
    if grafico(filas, clave, os.path.join("results", f"pareto{sufijo}.png")):
        print(f"    -> results/pareto{sufijo}.png")


if __name__ == "__main__":
    main()
