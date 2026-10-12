"""
probar_chip.py -- tests an exported SYMBEX model on a microcontroller, without retraining anything.

Uses the output folder of the CDT (modelo_symbex.h + the validation images it saved) and lib_symbex:
builds an ESP-IDF project with PlatformIO, flashes the board, streams the validation images over serial
and reports what the chip measured:
    equality with the integer simulation (must be 100%), accuracy,
    latency (median / min) and the time of EACH layer,
    RAM (model buffers, input, popcount table, stack used) and the flash of the model.
If the CDT also exported the INT8 reference of the original model (modelo_int8.h), it runs on the SAME chip
and the SAME images with Espressif's ESP-NN kernel, and the report compares the two (--no-int8 to skip).

Run from tools/:
    python probar_chip.py models/fashion_export --board esp32
    python probar_chip.py models/fashion_export --board s3 --port /dev/ttyACM0
    python probar_chip.py models/fashion_export --board esp32 --popcount swar        # minimum-RAM mode
    python probar_chip.py models/fashion_export --board esp32 --no-upload            # firmware already flashed
    python probar_chip.py models/fashion_export --board esp32 --weights ram          # weights in RAM, not flash
    python probar_chip.py models/fashion_export --board esp32 --first-layer mascara  # first layer with masks
    python probar_chip.py models/fashion_export --board esp32 --first-layer simple   # old first-layer loop
Writes <folder>/chip_<board>_<popcount>_<weights>_<first-layer>.json.
Requires: PlatformIO (pio), pyserial, a connected board.
"""
import argparse
import json
import os
import re
import sys

from compiler_core.export.dense import cargar_validacion, medir_en_chip, PLACAS


def main():
    ap = argparse.ArgumentParser(description="Test an exported SYMBEX model on a microcontroller")
    ap.add_argument("carpeta", help="output folder of the CDT (with modelo_symbex.h and validacion_*.u8)")
    ap.add_argument("--board", choices=sorted(PLACAS), default="esp32")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--samples", type=int, default=1000, help="validation images sent to the board")
    ap.add_argument("--popcount", choices=["lut16", "swar"], default="lut16",
                    help="lut16: 64 KB table, fastest | swar: no table, minimum RAM")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash",
                    help="where the weights live: flash (default) or RAM (faster if the model is bigger than the flash cache)")
    ap.add_argument("--first-layer", choices=["mascara", "planos", "simple"], default="mascara",
                    help="uint8 first layer (same result): masks (default, fastest measured), bit planes or the simple loop")
    ap.add_argument("--lib-path", default="../Symbex1_PIO/lib/lib_symbex")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    ap.add_argument("--no-int8", action="store_true", help="do not run the INT8 reference even if it was exported")
    args = ap.parse_args()
    if args.port and not args.port.startswith("/") and not args.port.upper().startswith("COM"):
        args.port = "/dev/" + args.port                  # "ttyUSB0" -> "/dev/ttyUSB0"

    header = os.path.join(args.carpeta, "modelo_symbex.h")
    for r in (header, os.path.join(args.carpeta, "validacion_x.u8")):
        if not os.path.exists(r):
            sys.exit(f"[X] {r} not found: run the CDT export stage first (it writes the header and the validation images)")
    m = re.search(r"#define SYMBEX_MLP_N_ENTRADAS (\d+)", open(header).read())
    if not m:
        sys.exit(f"[X] {header} is not a dense (MLP) export of the CDT")
    if "SYMBEX_ROM" not in open(header).read():
        sys.exit(f"[X] {header} was exported by an older version: export the model again (cdt.py)")
    x, y, sim = cargar_validacion(args.carpeta, int(m.group(1)))
    flash = None
    ruta_export = os.path.join(args.carpeta, "export.json")
    if os.path.exists(ruta_export):
        flash = json.load(open(ruta_export)).get("flash_bytes")
    int8 = None
    r8h, r8v = os.path.join(args.carpeta, "modelo_int8.h"), os.path.join(args.carpeta, "validacion_int8.u8")
    if not args.no_int8 and os.path.exists(r8h) and os.path.exists(r8v):
        import numpy as np
        f8 = json.load(open(ruta_export)).get("int8", {}).get("flash_bytes") if os.path.exists(ruta_export) else None
        int8 = dict(header=r8h, esperado=np.fromfile(r8v, dtype=np.uint8).astype(np.int64), flash=f8)
    print(f"[*] {args.carpeta}: {len(x)} validation images available, sending {min(args.samples, len(x))}"
          + (" | with the INT8 reference" if int8 else ""))
    try:
        res = medir_en_chip(header, x, y, sim, os.path.join(args.carpeta, f"chip_{args.board}"), args.lib_path,
                            args.board, args.port, args.baud, args.samples, not args.no_upload, args.popcount, flash,
                            pesos=args.weights, primera=args.first_layer, int8=int8)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"[X] {e}")
    ruta = os.path.join(args.carpeta, f"chip_{args.board}_{args.popcount}_{args.weights}_{args.first_layer}.json")
    with open(ruta, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[*] -> {ruta}")


if __name__ == "__main__":
    main()
