"""
chip_test.py -- tests an exported SYMBEX model on a microcontroller, without retraining anything.

Uses the output folder of the CDT (model_symbex.h + the validation images it saved) and lib_symbex:
builds an ESP-IDF project with PlatformIO, flashes the board, streams the validation images over serial
and reports what the chip measured:
    equality with the integer simulation (must be 100%), accuracy,
    latency (median / min) and the time of EACH layer,
    RAM (model buffers, input, popcount table, stack used) and the flash of the model.
If the CDT also exported the INT8 reference of the original model (model_int8.h), it runs on the SAME chip
and the SAME images with Espressif's ESP-NN kernel, and the report compares the two (--no-int8 to skip).

Run from tools/:
    python chip_test.py models/fashion_export --board esp32
    python chip_test.py models/fashion_export --board s3 --port /dev/ttyACM0
    python chip_test.py models/fashion_export --board esp32 --popcount swar        # minimum-RAM mode
    python chip_test.py models/fashion_export --board esp32 --no-upload            # firmware already flashed
    python chip_test.py models/fashion_export --board esp32 --weights ram          # weights in RAM, not flash
    python chip_test.py models/fashion_export --board esp32 --first-layer planes   # first layer with bit planes
    python chip_test.py --kernels --board s3                                        # kernel benchmark only (no model)
    python chip_test.py --kernels --board s3 --weights ram                          # ... with the weights in RAM
--kernels measures which arithmetic (XNOR + popcount or AND + count) is faster on the chip, with the library's own
kernels, and saves it as the chip profile the CDT uses with [output] arithmetic = "auto".
Only for headers exported with the XNOR arithmetic ([output] arithmetic = "xnor"), --first-layer chooses the
uint8 first-layer kernel; the AND arithmetic (default) has a single one.
Writes <folder>/chip_<board>_<popcount>_<weights>_<first-layer>.json.
Requires: PlatformIO (pio), pyserial, a connected board.
"""
import argparse
import json
import os
import re
import sys

from compiler_core.chip import BOARDS
from compiler_core.chip.kernels import run_kernels, save_profile
from compiler_core.chip.measure import measure_on_chip
from compiler_core.export.dense import load_validation


def main():
    ap = argparse.ArgumentParser(description="Test an exported SYMBEX model on a microcontroller")
    ap.add_argument("folder", nargs="?", default="", help="output folder of the CDT (with model_symbex.h and validation_*.u8)")
    ap.add_argument("--kernels", action="store_true", help="kernel benchmark only: the fastest arithmetic on this chip")
    ap.add_argument("--int8", action="store_true", help="with --kernels: also measure INT8 (ESP-NN) as a reference")
    ap.add_argument("--board", choices=sorted(BOARDS), default="esp32")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--samples", type=int, default=1000, help="validation images sent to the board")
    ap.add_argument("--popcount", choices=["lut16", "swar"], default="lut16",
                    help="lut16: 64 KB table, fastest | swar: no table, minimum RAM")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash",
                    help="where the weights live: flash (default) or RAM (faster if the model is bigger than the flash cache)")
    ap.add_argument("--first-layer", choices=["mask", "planes", "simple"], default="mask",
                    help="uint8 first layer of XNOR headers (same result): masks (default), bit planes or the simple loop")
    ap.add_argument("--lib-path", default="../Symbex1_PIO/lib/lib_symbex")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    ap.add_argument("--no-int8", action="store_true", help="do not run the INT8 reference even if it was exported")
    args = ap.parse_args()
    if args.port and not args.port.startswith("/") and not args.port.upper().startswith("COM"):
        args.port = "/dev/" + args.port                  # "ttyUSB0" -> "/dev/ttyUSB0"
    if args.kernels:
        if args.board == "uno":
            sys.exit("[X] the kernel benchmark is for the ESP32 boards (ESP-IDF); on AVR the export always uses XNOR")
        try:
            profile = run_kernels(args.board, args.port, args.lib_path, os.path.join("runs", f"kernels_{args.board}"),
                                  args.popcount, upload=not args.no_upload, with_int8=args.int8,
                                  weights=args.weights or "flash")
        except (RuntimeError, ValueError) as e:
            sys.exit(f"[X] {e}")
        if profile.get("checksums_ok"):
            print(f"\n[*] chip profile -> {save_profile(profile)}")
        return
    if not args.folder:
        sys.exit("[X] give the output folder of the CDT (or --kernels for the kernel benchmark only)")

    header = os.path.join(args.folder, "model_symbex.h")
    for path in (header, os.path.join(args.folder, "validation_x.u8")):
        if not os.path.exists(path):
            sys.exit(f"[X] {path} not found: run the CDT export stage first (it writes the header and the validation "
                     f"images). Exports made before the English translation use other file names: export again.")
    m = re.search(r"#define SYMBEX_MLP_N_INPUTS (\d+)", open(header).read())
    if not m:
        sys.exit(f"[X] {header} is not a dense (MLP) export of the CDT")
    x, y, sim = load_validation(args.folder, int(m.group(1)))
    flash = None
    export_json = os.path.join(args.folder, "export.json")
    if os.path.exists(export_json):
        flash = json.load(open(export_json)).get("flash_bytes")
    int8 = None
    int8_header, int8_expected = os.path.join(args.folder, "model_int8.h"), os.path.join(args.folder, "validation_int8.u8")
    if not args.no_int8 and os.path.exists(int8_header) and os.path.exists(int8_expected):
        import numpy as np
        f8 = json.load(open(export_json)).get("int8", {}).get("flash_bytes") if os.path.exists(export_json) else None
        int8 = dict(header=int8_header, expected=np.fromfile(int8_expected, dtype=np.uint8).astype(np.int64), flash=f8)
    print(f"[*] {args.folder}: {len(x)} validation images available, sending {min(args.samples, len(x))}"
          + (" | with the INT8 reference" if int8 else ""))
    try:
        res = measure_on_chip(header, x, y, sim, os.path.join(args.folder, f"chip_{args.board}"), args.lib_path,
                              args.board, args.port, args.baud, args.samples, not args.no_upload, args.popcount, flash,
                              weights=args.weights, first_layer=args.first_layer, int8=int8)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"[X] {e}")
    path = os.path.join(args.folder, f"chip_{args.board}_{args.popcount}_{args.weights}_{args.first_layer}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[*] -> {path}")


if __name__ == "__main__":
    main()
