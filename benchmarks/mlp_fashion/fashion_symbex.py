"""
fashion_symbex.py -- mlp_fashion benchmark, SYMBEX side.

    python fashion_symbex.py                                  # FP32 model (if missing) + CDT, seed 42
    python fashion_symbex.py --semillas 3 --board s3          # 3 seeds (42, 43, 44): CDT + chip test each
    python fashion_symbex.py --board s3 --solo-chip --barrido # every chip configuration on the seed-42 export

Run from benchmarks/mlp_fashion/ with the main environment (.venv, PyTorch).
  1. models/fashion_mlp.pth: the original FP32 model (trained by modelo.py if it does not exist). All seeds
     start from this SAME model: the seed only changes the CDT (search and 1-bit training).
  2. for every seed: the CDT (tools/cdt.py symbex.toml --seed S) -> runs/symbex_sS/ (skipped if that seed is
     already exported; --rehacer trains it again)
  3. with --board: tools/probar_chip.py on every export, with --popcount / --weights / --first-layer
     (SYMBEX and the CDT's own INT8 reference, same chip, same images)
  4. --barrido: on the first seed's export, every combination of first layer (mascara, planos),
     popcount (swar, lut16) and weights (flash, ram), sorted by latency
Summary tables at the end; every .json is copied to results/ (the only outputs that are published).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(AQUI, "..", "..", "tools"))
LIB = "../../Symbex1_PIO/lib/lib_symbex"


def correr_cdt(semilla, salida):
    print(f"\n[*] CDT, seed {semilla} -> {salida}")
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "cdt.py"), "symbex.toml", "--seed", str(semilla),
                        "--output-directory", salida])
    if r.returncode != 0:
        sys.exit(f"[X] the CDT failed (seed {semilla})")


def probar(salida, placa, puerto, popcount, pesos, primera):
    """probar_chip.py on an export; returns the result dict."""
    print(f"\n[*] chip: {salida} | {placa} | popcount {popcount}, weights in {pesos}, first layer {primera}")
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "probar_chip.py"), salida, "--board", placa,
                        "--port", puerto, "--popcount", popcount, "--weights", pesos, "--first-layer", primera,
                        "--lib-path", LIB])
    if r.returncode != 0:
        sys.exit("[X] the chip test failed")
    return json.load(open(os.path.join(salida, f"chip_{placa}_{popcount}_{pesos}_{primera}.json")))


def media(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.2f} +- {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=1, help="number of seeds: 42, 43, ... (default 1)")
    ap.add_argument("--board", choices=["esp32", "s3"], default="", help="measure on this board")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--popcount", choices=["swar", "lut16"], default="swar")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash", help="where the SYMBEX weights live")
    ap.add_argument("--first-layer", choices=["planos", "mascara", "simple"], default="mascara")
    ap.add_argument("--solo-chip", action="store_true", help="skip the CDT: test the existing exports")
    ap.add_argument("--barrido", action="store_true", help="every chip configuration on the first seed's export")
    ap.add_argument("--rehacer", action="store_true", help="run the CDT again even if a seed is already exported")
    args = ap.parse_args()
    if args.port and not args.port.startswith("/") and not args.port.upper().startswith("COM"):
        args.port = "/dev/" + args.port                  # "ttyUSB0" -> "/dev/ttyUSB0"
    os.chdir(AQUI)
    sys.path.insert(0, AQUI)
    from modelo import obtener_fp32
    print("[*] original FP32 model (models/fashion_mlp.pth, trained if missing)")
    obtener_fp32()
    semillas = [42 + i for i in range(args.semillas)]
    os.makedirs("results", exist_ok=True)

    filas = []
    for s in semillas:
        salida = f"runs/symbex_s{s}"
        hecho = os.path.exists(os.path.join(salida, "train.json")) and os.path.exists(os.path.join(salida, "modelo_symbex.h"))
        if args.solo_chip or (hecho and not args.rehacer):
            if not hecho:
                sys.exit(f"[X] {salida} has no export: run without --solo-chip")
            print(f"\n[*] seed {s}: export already in {salida} (use --rehacer to train it again)")
        else:
            correr_cdt(s, salida)
        tr = json.load(open(os.path.join(salida, "train.json")))
        fila = dict(semilla=s, ancho=tr["width"], acc_validacion=tr["acc_last"], flash_kb=tr["flash_total"] / 1024)
        if args.board and not args.barrido:
            c = probar(salida, args.board, args.port, args.popcount, args.weights, args.first_layer)
            fila.update(acc_chip=c["precision"], latencia_ms=c["mediana_ms"], igual=c["igual_simulacion"],
                        ram_kb=c["ram"]["total"] / 1024,
                        int8_ms=c.get("int8", {}).get("mediana_ms"), int8_acc=c.get("int8", {}).get("precision"))
        filas.append(fila)
        for f in glob.glob(os.path.join(salida, "*.json")):
            shutil.copy(f, os.path.join("results", f"symbex_s{s}_" + os.path.basename(f)))

    print(f"\n    SEEDS  (same FP32 original; validation = 10,000 Fashion-MNIST test images, last epoch)")
    print(f"    {'seed':>5}{'width':>7}{'val acc':>10}{'flash KB':>10}" + (f"{'chip acc':>10}{'ms':>9}{'equal':>8}" if args.board and not args.barrido else ""))
    for f in filas:
        print(f"    {f['semilla']:>5}{f['ancho']:>6g}x{f['acc_validacion']:>9.2f}%{f['flash_kb']:>10.1f}"
              + (f"{f['acc_chip']:>9.2f}%{f['latencia_ms']:>9.3f}{f['igual']:>7.0f}%" if 'acc_chip' in f else ""))
    print(f"    validation accuracy over {len(filas)} seed(s): {media([f['acc_validacion'] for f in filas])} %")
    resumen = dict(placa=args.board, popcount=args.popcount, pesos=args.weights, primera=args.first_layer, semillas=filas,
                   acc_validacion=media([f["acc_validacion"] for f in filas]))
    if filas and "latencia_ms" in filas[0]:
        resumen["latencia_ms"] = media([f["latencia_ms"] for f in filas])
        print(f"    chip latency: {resumen['latencia_ms']} ms | chip accuracy: {media([f['acc_chip'] for f in filas])} %")
    nombre = f"results/symbex_semillas_{args.board or 'pc'}.json"
    json.dump(resumen, open(nombre, "w"), indent=1)
    print(f"    -> {nombre}")

    if args.barrido:
        if not args.board:
            sys.exit("[X] --barrido needs --board")
        salida = f"runs/symbex_s{semillas[0]}"
        res = []
        for primera in ("mascara", "planos"):
            for pop in ("swar", "lut16"):
                for pesos in ("flash", "ram"):
                    c = probar(salida, args.board, args.port, pop, pesos, primera)
                    res.append(dict(primera=primera, popcount=pop, pesos=pesos, ms=c["mediana_ms"],
                                    capa0=c["capas"][0]["mediana_ms"], ram_kb=c["ram"]["total"] / 1024,
                                    igual=c["igual_simulacion"], int8_ms=c.get("int8", {}).get("mediana_ms")))
        for f in glob.glob(os.path.join(salida, "chip_*.json")):
            shutil.copy(f, os.path.join("results", f"symbex_s{semillas[0]}_" + os.path.basename(f)))
        res.sort(key=lambda r: r["ms"])
        print(f"\n    CHIP CONFIGURATIONS ({args.board}, seed {semillas[0]}), fastest first")
        print(f"    {'first layer':<12}{'popcount':<10}{'weights':<9}{'total ms':>10}{'layer 0 ms':>12}{'RAM KB':>9}{'equal':>8}")
        for r in res:
            print(f"    {r['primera']:<12}{r['popcount']:<10}{r['pesos']:<9}{r['ms']:>10.3f}{r['capa0']:>12.3f}"
                  f"{r['ram_kb']:>9.2f}{r['igual']:>7.0f}%")
        if res[0]["int8_ms"]:
            print(f"    (the CDT's INT8 reference on this chip: {res[0]['int8_ms']:.3f} ms)")
        json.dump(res, open(f"results/symbex_barrido_{args.board}.json", "w"), indent=1)
        print(f"    -> results/symbex_barrido_{args.board}.json")


if __name__ == "__main__":
    main()
