"""
touch_digits.py -- draw a digit with the finger (or the mouse) and the Arduino Uno's binary network reads it.

The network is the reference model of this benchmark (../reference/model_symbex_xnor.h: 64-256-10, input, weights and
activations all 1 bit, test 95.56 % on scikit-learn Digits). Its C code (lib_symbex + the header) is compiled here
for the PC with the library's 8-bit paths (the ones the Uno uses), so the answer on the screen is the answer the Uno
gives. At start it is checked against the CDT's integer simulation on the 360 test images (must be 100 %).

From the drawing to the network's 64 bytes, the way scikit-learn Digits was made (UCI optdigits): crop the ink,
thicken the stroke (--thicken: a max filter, so a thin line still fills half of a 4x4 block as optdigits' thick
strokes do; 25 px raised Edén's 35 drawings from 20 to 25 right, Nov 22), scale it to fit a 32x32 box (aspect kept, centered), one bit per pixel, count the ink of each 4x4 block (0-16), scale to
0-255. The network then keeps one bit per pixel (block >= 8 of 16): the 8x8 preview shows exactly that bit.

    python touch_digits.py                  # draw; the answer appears when the finger lifts
    python touch_digits.py --score          # accuracy on the saved drawings (samples/)
    python touch_digits.py --port /dev/ttyUSB1   # the Uno answers each drawing (PC answer shown next to it)

--port: the Uno must hold chip_test.py's AVR firmware with the same model (tools/chip_test.py <folder with the XNOR
header as model_symbex.h> --board uno --port ...): it reads 0xA5 + the 64 bytes and answers "R:class,us,...".

Keys: 0-9 choose the digit you are going to draw (shown on screen), Enter or space saves the drawing with it
(samples/<label>_<n>.png, for --score) and clears, c or Escape clears without saving.
These drawings check the demo, not the network: the network's accuracy is the README's (test set, used once).
Run from any folder; the samples/ and .build/ folders stay next to this script (gitignored).
"""
import argparse
import ctypes
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "reference")
MODELS = {"reference": REF, "demo": os.path.join(HERE, "model")}   # demo: symbex_demo.toml (augmented data)
LIB = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "Symbex1_PIO", "lib", "lib_symbex"))
SOURCES = ["src/ops/symbex_bitops.c", "src/ops/symbex_bitpack.c", "src/layers/dense/symbex_dense.c",
           "src/hal/generic/symbex_hal_impl.c"]
AVR_PATHS = ["-DSYMBEX_U8_BYTEWISE", "-DSYMBEX_POPCOUNT_FORCE_LUT8", "-DSYMBEX_PLANES_PLANE_OUTER",
             "-DSYMBEX_PLANES_PER_NEURON"]   # lib_symbex's 8-bit paths, as on the Uno (tests/run_tests.py)
SAMPLES = os.path.join(HERE, "samples")
CANVAS = 400
THICKEN = 25           # max filter size in canvas pixels (odd; 0 = off), set by --thicken


def build(model=REF):
    """The model's XNOR header + lib_symbex as a shared library for the PC; returns the predict function"""
    out = os.path.join(HERE, ".build")
    os.makedirs(out, exist_ok=True)
    tag = os.path.basename(os.path.normpath(model))
    src, so = os.path.join(out, f"digits_{tag}.c"), os.path.join(out, f"digits_{tag}.so")
    open(src, "w").write('#include "model_symbex_xnor.h"\n'
                         "int predict(const uint8_t* x) { return symbex_mlp_infer(x); }\n")
    cmd = ["gcc", "-std=gnu11", "-O2", "-shared", "-fPIC", "-I", os.path.join(LIB, "include"), "-I", model,
           *AVR_PATHS, "-o", so, src, *[os.path.join(LIB, s) for s in SOURCES]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[X] the C did not compile:\n{r.stderr[-1500:]}")
    fn = ctypes.CDLL(so).predict
    fn.argtypes, fn.restype = [ctypes.POINTER(ctypes.c_uint8)], ctypes.c_int
    return lambda x: fn(np.ascontiguousarray(x, dtype=np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))


def check(predict, model=REF):
    """the compiled C against the CDT's integer simulation on the 360 test images exported with the model"""
    x = np.fromfile(os.path.join(model, "validation_x.u8"), dtype=np.uint8).reshape(-1, 64)
    sim = np.fromfile(os.path.join(model, "validation_sim.u8"), dtype=np.uint8)
    y = np.fromfile(os.path.join(model, "validation_y.u8"), dtype=np.uint8)
    got = np.array([predict(r) for r in x])
    equal, acc = (got == sim).mean() * 100, (got == y).mean() * 100
    print(f"[*] C on the PC vs the integer simulation: {equal:.1f} % equal on {len(x)} test images "
          f"(accuracy {acc:.2f} %)")
    if equal < 100:
        sys.exit("[X] not exact: the demo would not show what the Uno answers")


class Uno:
    """the Uno with chip_test.py's AVR firmware (model_test_avr.cpp): 0xA5 + 64 bytes -> 'R:class,total_us,...'"""
    def __init__(self, port):
        import serial
        import time
        self.s = serial.Serial(port, 115200, timeout=3)     # opening the port resets the Uno (DTR)
        t0 = time.time()
        while b"SYMBEX ready" not in self.s.readline():
            if time.time() - t0 > 8:
                sys.exit(f"[X] no 'SYMBEX ready' from {port}: flash the model with tools/chip_test.py --board uno")
        self.s.reset_input_buffer()
        print(f"[*] Uno ready on {port}")

    def __call__(self, x):
        """(class, microseconds) the Uno answers for one 64-byte input"""
        self.s.write(b"\xa5" + bytes(np.asarray(x, dtype=np.uint8)))
        while True:
            line = self.s.readline().decode(errors="replace").strip()
            if not line:
                sys.exit("[X] the Uno did not answer")
            if line.startswith("R:"):
                f = line[2:].split(",")
                return int(f[0]), int(f[1])


def to_input(img):
    """drawing (PIL 'L', ink = 255) -> the 64 uint8 pixels of a scikit-learn Digits image, or None if empty"""
    box = img.getbbox()
    if box is None:
        return None
    ink = img.crop(box)
    if THICKEN:
        ink = ink.filter(ImageFilter.MaxFilter(THICKEN | 1))
    w, h = ink.size
    s = 32 / max(w, h)
    ink = ink.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    big = Image.new("L", (32, 32), 0)
    big.paste(ink, ((32 - ink.size[0]) // 2, (32 - ink.size[1]) // 2))
    bits = np.asarray(big) >= 128
    counts = bits.reshape(8, 4, 8, 4).sum(axis=(1, 3))                  # 0-16, as optdigits
    return np.round(counts * 255 / 16).astype(np.uint8).reshape(64)    # as model.py prepare_data()


def score(predict):
    files = sorted(glob.glob(os.path.join(SAMPLES, "*.png")))
    if not files:
        sys.exit("[X] no drawings in samples/: draw and press a digit key to save them")
    per = np.zeros((10, 2), dtype=int)
    for f in files:
        label = int(os.path.basename(f)[0])
        x = to_input(Image.open(f).convert("L"))
        per[label] += (x is not None and predict(x) == label, 1)
    print("digit  right / drawn")
    for d in range(10):
        if per[d, 1]:
            print(f"  {d}    {per[d, 0]} / {per[d, 1]}")
    print(f"[*] {per[:, 0].sum()} / {per[:, 1].sum()} right ({per[:, 0].sum() / per[:, 1].sum() * 100:.1f} %)")


class App:
    def __init__(self, predict, pen, uno=None):
        import tkinter as tk
        self.predict, self.pen, self.uno = predict, pen, uno
        self.root = tk.Tk()
        self.root.title("SYMBEX-1: a 1-bit network reads your digit (Arduino Uno model)")
        self.canvas = tk.Canvas(self.root, width=CANVAS, height=CANVAS, bg="white", cursor="pencil")
        self.canvas.grid(row=0, column=0, rowspan=3, padx=8, pady=8)
        self.view = tk.Canvas(self.root, width=160, height=160, bg="black")
        self.view.grid(row=0, column=1, padx=8, pady=8)
        self.answer = tk.Label(self.root, text="draw a digit", font=("Helvetica", 64, "bold"), width=3)
        self.answer.grid(row=1, column=1)
        self.label = None
        self.info = tk.Label(self.root, text="keys 0-9: choose the digit to draw\nEnter: save   c: clear",
                             justify="left")
        self.info.grid(row=2, column=1, padx=8)
        tk.Button(self.root, text="Clear", font=("Helvetica", 20), command=self.clear).grid(
            row=3, column=0, sticky="ew", padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.down)
        self.canvas.bind("<B1-Motion>", self.move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.read())
        self.root.bind("<Key>", self.key)
        self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.view.delete("all")
        self.img = Image.new("L", (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.img)
        self.last = None
        self.answer.config(text="?")

    def down(self, e):
        self.last = (e.x, e.y)
        self.move(e)

    def move(self, e):
        x0, y0 = self.last or (e.x, e.y)
        r = self.pen / 2
        self.canvas.create_line(x0, y0, e.x, e.y, width=self.pen, capstyle="round", fill="black")
        self.draw.line((x0, y0, e.x, e.y), fill=255, width=self.pen)
        self.draw.ellipse((e.x - r, e.y - r, e.x + r, e.y + r), fill=255)
        self.last = (e.x, e.y)

    def read(self):
        self.last = None
        x = to_input(self.img)
        if x is None:
            return
        self.view.delete("all")
        for i, v in enumerate(x):
            if v >= 128:                                                # the bit the network keeps
                r, c = divmod(i, 8)
                self.view.create_rectangle(c * 20, r * 20, c * 20 + 19, r * 20 + 19, fill="white", width=0)
        pc = self.predict(x)
        if self.uno is None:
            self.answer.config(text=str(pc))
            return
        c, us = self.uno(x)
        self.answer.config(text=str(c), fg="black" if c == pc else "red")
        self.root.title(f"SYMBEX-1: the Arduino Uno read {c} in {us / 1000:.2f} ms (PC: {pc}"
                        f"{', equal' if c == pc else ', DIFFERENT'})")

    def key(self, e):
        if e.char.isdigit():                                            # the digit about to be drawn
            self.label = e.char
            self.info.config(text=f"drawing a {self.label}\nEnter: save   c: clear   0-9: other digit")
        elif e.keysym in ("Return", "KP_Enter", "space"):
            if self.label is None:
                self.info.config(text="press 0-9 first: the digit you draw\nEnter: save   c: clear")
                return
            if self.img.getbbox() is None:
                return
            os.makedirs(SAMPLES, exist_ok=True)
            n = len(glob.glob(os.path.join(SAMPLES, f"{self.label}_*.png")))
            self.img.save(os.path.join(SAMPLES, f"{self.label}_{n:03d}.png"))
            total = len(glob.glob(os.path.join(SAMPLES, "*.png")))
            self.info.config(text=f"saved as {self.label} ({total} drawings)\nEnter: save   c: clear   0-9: other digit")
            self.clear()
        elif e.keysym in ("c", "Escape"):
            self.clear()


def main():
    global THICKEN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true", help="accuracy on the saved drawings (samples/), no window")
    ap.add_argument("--pen", type=int, default=32, help="pen width in pixels of the 400x400 canvas (default 32)")
    ap.add_argument("--thicken", type=int, default=THICKEN,
                    help=f"stroke thickening before the reduction, max filter in canvas pixels (default {THICKEN}, 0 = off)")
    ap.add_argument("--model", choices=sorted(MODELS), default="reference",
                    help="reference: the benchmark's model; demo: the same network trained on augmented drawings "
                         "(augment_data.py + symbex_demo.toml)")
    ap.add_argument("--port", help="serial port of the Uno (it answers each drawing; flash it first with chip_test.py)")
    args = ap.parse_args()
    THICKEN = args.thicken
    predict = build(MODELS[args.model])
    check(predict, MODELS[args.model])
    if args.score:
        score(predict)
    else:
        App(predict, args.pen, Uno(args.port) if args.port else None).root.mainloop()


if __name__ == "__main__":
    main()
