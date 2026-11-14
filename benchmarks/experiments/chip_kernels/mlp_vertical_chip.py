"""
mlp_vertical_chip.py -- a whole exported MLP with its wide binary layers vertical (dense_vertical_bench.py's kernel):
is the network faster, and still exact?

Takes a CDT dense export (AND arithmetic, the S3's: model_symbex.h + validation_x.u8 / _y.u8 / _sim.u8) and writes a
second inference function from the header's own: every binary layer with at least --min-neurons neurons (an input a
multiple of 8, below 2048) becomes one dense_vertical call. The other layers (the uint8 first layer, the 10-class
classifier) stay as exported. Its transposed weights and threshold planes are built here from the header's weights,
thresholds and weight counts. The firmware runs both functions on the same validation images (in RAM, as chip_test.py receives them: images
read from flash would evict the weights from the flash cache).
It checks both against the integer simulation, image by image. It times each image (median, a second pass after a
warm-up pass) and reports the share of the vertical layers' inputs that were 1.

    python mlp_vertical_chip.py ../../mlp/mlp_fashion/runs/sizing/h128_512_s42 --port /dev/ttyACM0   # S3, flash
    python mlp_vertical_chip.py <run> --port /dev/ttyACM0 --weights ram
    python mlp_vertical_chip.py <run> --host            # the exactness check on this PC (portable C), every image

Results: results/mlp_vertical_chip_<run name>_<board>[_wram].json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dense_vertical_bench import (EP, RESET, TICKS, VERTICAL_C, c_array, copy_lib, flash,  # noqa: E402
                                  open_port, pack_rows, planes_of, write_project)

LAYER = re.compile(r"    symbex_and_raw\(in_bits, (\d+), symbex_mlp_w(\d+), (\d+), (\d+), symbex_mlp_cw\2, raw\);\n"
                   r"    symbex_dense_threshold\(raw, symbex_mlp_t\2, out_bits, \4\);\n")

FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#if defined(ESP_PLATFORM)
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())
#define OUT printf
#elif defined(SYMBEX_HOST)
#include <time.h>
static uint32_t host_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint32_t)(t.tv_sec * 1000000000ull + t.tv_nsec); }
#define NOW() host_ns()
#define OUT printf
#else
#include <Arduino.h>
#define NOW() ((uint32_t)micros())
#define OUT Serial.printf
#endif
#include "hal/symbex_hal.h"
#include "layers/dense/symbex_dense.h"
#include "ops/symbex_and.h"
#include "vert_sizes.h"
""" + VERTICAL_C + r"""
#include "model_symbex.h"
#include "vert_model.h"

static uint8_t xbuf[SYMBEX_MLP_N_INPUTS] __attribute__((aligned(16)));
static uint32_t t_std[N_IMG], t_vert[N_IMG];
static int cmp_u32(const void* a, const void* b) { const uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b; return (x > y) - (x < y); }

static void bench_once(void) {
    OUT("MLP VERTICAL ready (%s)\n", WHERE);
    list_table(LT);
    int eq_std = 0, eq_vert = 0, same = 0, correct = 0;
    /* one method over all the images, then the other (alternating them per image would make their weights evict
       each other from the flash cache); for each, pass 0 warms the caches and pass 1 is timed */
    static uint8_t pred_std[N_IMG];
    for (int m = 0; m < 2; m++)
        for (int pass = 0; pass < 2; pass++) {
            vert_ones = vert_calls = 0;
            for (int i = 0; i < N_IMG; i++) {
                memcpy(xbuf, IMG + (uint32_t)i * SYMBEX_MLP_N_INPUTS, SYMBEX_MLP_N_INPUTS);
                const uint32_t t0 = NOW();
                const int p = m == 0 ? symbex_mlp_infer(xbuf) : vertical_mlp_infer(xbuf);
                const uint32_t t = NOW() - t0;
                if (pass == 0) continue;
                if (m == 0) { t_std[i] = t; pred_std[i] = (uint8_t)p; eq_std += p == SIM[i]; }
                else { t_vert[i] = t; eq_vert += p == SIM[i]; same += p == pred_std[i]; correct += p == LABEL[i]; }
            }
        }
    qsort(t_std, N_IMG, sizeof t_std[0], cmp_u32);
    qsort(t_vert, N_IMG, sizeof t_vert[0], cmp_u32);
    OUT("M:%d|%d|%d|%d|%d|%lu|%lu|%lu|%lu|%lu|%lu\n", N_IMG, eq_std, eq_vert, same, correct,
        (unsigned long)t_std[N_IMG / 2], (unsigned long)t_std[0], (unsigned long)t_vert[N_IMG / 2], (unsigned long)t_vert[0],
        (unsigned long)vert_ones, (unsigned long)vert_calls);
    OUT("DONE\n");
}

#if defined(ESP_PLATFORM)
void app_main(void) {
    for (;;) {
        bench_once();
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
#elif defined(SYMBEX_HOST)
int main(void) { bench_once(); return 0; }
#else
void setup() { Serial.begin(115200); }
void loop() {
    bench_once();
    delay(2000);
}
#endif
"""


def header_array(text, name):
    m = re.search(r"static SYMBEX_ROM \w+ " + name + r"\[(\d+)\][^=]*= \{(.*?)\};", text, re.S)
    if not m:
        sys.exit(f"[X] {name} not found in the header")
    vals = [-2 ** 31 if v.strip() == "INT32_MIN" else int(v.strip().rstrip("uUlL"), 0) for v in m.group(2).split(",") if v.strip()]
    assert len(vals) == int(m.group(1)), name
    return np.array(vals, np.int64)


def vertical_files(run, n_img, min_neurons, weights):
    """the files of the firmware (beyond lib_symbex): model_symbex.h, vert_sizes.h, vert_model.h; returns
    (writer(src) -> banner, the vertical layers)"""
    text = open(os.path.join(run, "model_symbex.h")).read()
    if "SYMBEX_MLP_ARITHMETIC_AND" not in text:
        sys.exit("[X] only headers exported with the AND arithmetic (the S3's) are supported")
    body = re.search(r"static inline int symbex_mlp_infer\(const uint8_t\* x\) \{\n(.*?)\n\}\n", text, re.S).group(1)
    words = int(re.search(r"static uint32_t bits_a\[(\d+)\]", body).group(1))
    layers, data = [], []

    def swap(m):
        n_in, k, stride, n_out = (int(g) for g in m.groups())
        groups = (n_out + 127) // 128
        if n_out < min_neurons or n_in % 8 or n_in >= 2048 or 4 * groups > words:
            return m.group(0)
        w = header_array(text, f"symbex_mlp_w{k}").reshape(n_out, stride).astype(np.uint64)
        b = ((w[:, :, None] >> np.arange(32, dtype=np.uint64)) & 1).reshape(n_out, stride * 32)[:, :n_in].astype(np.int64)
        thr, cw = header_array(text, f"symbex_mlp_t{k}"), header_array(text, f"symbex_mlp_cw{k}")
        assert (b.sum(1) == cw).all(), f"layer {k}: weight counts differ from the weights"
        lanes = np.zeros((groups * 128, n_in + 1), np.int64)
        lanes[:n_out, :n_in] = b
        wt = np.concatenate([planes_of(lanes[g * 128:(g + 1) * 128]) for g in range(groups)])
        e = np.full(groups * 128, n_in + 1, np.int64)
        e[:n_out] = -((-(thr - n_in + 2 * cw)) // 2)                   # ceil((thr - n_in + 2 count(w)) / 2)
        v = 2048 - np.clip(e, -n_in, n_in + 1)
        ep = np.stack([(v >> i) & 1 for i in range(EP)])
        ev = np.concatenate([planes_of(ep[:, g * 128:(g + 1) * 128].T) for g in range(groups)])
        data.append((f"symbex_vert_wt{k}", wt))
        data.append((f"symbex_vert_e{k}", ev))
        layers.append(dict(layer=k, n_in=n_in, n_out=n_out))
        return (f"    dense_vertical(in_bits, {n_in}, {groups}, (const vplane_t*)symbex_vert_wt{k}, "
                f"(const vplane_t*)symbex_vert_e{k}, out_bits);\n")

    vbody = LAYER.sub(swap, body + "\n")
    if not layers:
        sys.exit(f"[X] no binary layer with {min_neurons}+ neurons in {run}")
    n_inputs = int(re.search(r"#define SYMBEX_MLP_N_INPUTS (\d+)", text).group(1))
    x = np.fromfile(os.path.join(run, "validation_x.u8"), np.uint8).reshape(-1, n_inputs)
    y = np.fromfile(os.path.join(run, "validation_y.u8"), np.uint8)
    sim = np.fromfile(os.path.join(run, "validation_sim.u8"), np.uint8)
    n_img = min(n_img, len(x))
    where = f"{os.path.basename(os.path.normpath(run))}, {n_img} images, weights in {weights}"

    def write(src):
        open(os.path.join(src, "model_symbex.h"), "w").write(text)
        open(os.path.join(src, "vert_sizes.h"), "w").write(
            f"#pragma once\n#define VERT_MAX_IN {max(l['n_in'] for l in layers)}\n#define N_IMG {n_img}\n"
            f"#define WHERE \"{where}\"\n")
        with open(os.path.join(src, "vert_model.h"), "w") as f:
            f.write("#pragma once\n")
            for name, arr in data:
                f.write(c_array("uint32_t", name, arr, False, 8).replace("static uint32_t", "static SYMBEX_ROM uint32_t", 1))
            f.write(c_array("uint8_t", "IMG", x[:n_img], False, 32))     # in RAM: images in flash would evict the weights
            f.write(c_array("uint8_t", "SIM", sim[:n_img], True, 32))
            f.write(c_array("uint8_t", "LABEL", y[:n_img], True, 32))
            f.write("/* symbex_mlp_infer with its wide binary layers vertical (written by mlp_vertical_chip.py) */\n"
                    "static inline int vertical_mlp_infer(const uint8_t* x) {\n" + vbody + "}\n")
        return where
    return write, layers, where


def parse(line):
    n, eq_s, eq_v, same, correct, ms, mins, mv, minv, ones, calls = (int(v) for v in line[2:].split("|"))
    return dict(images=n, equal_sim_standard=eq_s, equal_sim_vertical=eq_v, standard_equals_vertical=same,
                correct_vertical=correct, median_standard=ms, min_standard=mins, median_vertical=mv, min_vertical=minv,
                ones_share=ones / max(calls, 1))


def host(write, layers):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        os.makedirs(src)
        files, get = copy_lib(src, None)
        for name, rel in files.items():
            open(os.path.join(src, name), "wb").write(get(os.path.join("src", *rel)))
        write(src)
        open(os.path.join(src, "main.c"), "w").write(FIRMWARE)
        exe = os.path.join(d, "bench")
        subprocess.run(["gcc", "-O2", "-DSYMBEX_HOST", "-I", src, "-I", os.path.join(src, "include"), "-o", exe,
                        os.path.join(src, "main.c")] + [os.path.join(src, n) for n in files], check=True)
        out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    r = parse(next(l for l in out.splitlines() if l.startswith("M:")))
    return r


def read_results(port, board, where):
    s = open_port(port, 115200, reset=RESET[board])
    ready, t0 = False, time.time()
    while time.time() - t0 < 120:
        line = s.readline().decode(errors="replace").strip()
        if "MLP VERTICAL ready" in line:
            if f"({where})" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected {where}")
            ready = True
        elif ready and line.startswith("M:"):
            s.close()
            return parse(line)
    s.close()
    sys.exit("[X] no results from the board in 120 s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="CDT run folder (model_symbex.h, validation_*.u8)")
    ap.add_argument("--board", choices=sorted(TICKS), default="s3")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--images", type=int, default=200, help="validation images embedded in the firmware (in RAM)")
    ap.add_argument("--min-neurons", type=int, default=128, help="binary layers with this many neurons go vertical")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash", help="ram: -DSYMBEX_ROM= (every layer)")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    ap.add_argument("--host", action="store_true", help="build and run on this PC (portable C): the exactness check")
    a = ap.parse_args()
    write, layers, where = vertical_files(a.run, a.images, a.min_neurons, a.weights)
    print("[*] vertical layers: " + ", ".join(f"{l['layer']} ({l['n_in']} -> {l['n_out']})" for l in layers))
    if a.host:
        r = host(write, layers)
        unit, per_us = "ns", 1000.0
    else:
        tag = f"{os.path.basename(os.path.normpath(a.run))}_{a.board}" + ("_wram" if a.weights == "ram" else "")
        project = os.path.join(HERE, "runs", f"mlp_vertical_chip_{tag}")
        if not a.no_upload:
            write_project(project, a.port, a.board, a.weights, FIRMWARE, write, "mlp_vertical_chip",
                          "-DSYMBEX_ROM=" if a.weights == "ram" else "")
            flash(project, verbose=False, what=where)
        r = read_results(a.port, a.board, where)
        # Arduino's micros() on the Pico 2 (the kernel bench counts ns with its own timer)
        unit, per_us = ("us", 1.0) if a.board.startswith("pico2") else TICKS[a.board]
    n = r["images"]
    print(f"\n{'PC' if a.host else a.board}, {where}")
    print(f"  equal to the simulation: standard {r['equal_sim_standard']}/{n}, vertical {r['equal_sim_vertical']}/{n}; "
          f"accuracy (vertical) {100 * r['correct_vertical'] / n:.1f} %")
    print(f"  median per image: standard {r['median_standard'] / per_us:.1f} us, vertical {r['median_vertical'] / per_us:.1f} us "
          f"({r['median_standard'] / r['median_vertical']:.2f}x); inputs of the vertical layers that were 1: "
          f"{100 * r['ones_share'] / sum(l['n_in'] for l in layers) * len(layers):.1f} % (mean)")
    if r["equal_sim_standard"] != n or r["equal_sim_vertical"] != n:
        sys.exit("[X] not exact: the times are not trusted")
    if a.host:
        print("[*] both equal the simulation on every image (PC)")
        return
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"mlp_vertical_chip_{tag}.json")
    with open(path, "w") as f:
        json.dump(dict(run=os.path.basename(os.path.normpath(a.run)), board=a.board, weights=a.weights, unit=unit,
                       vertical_layers=layers, **r,
                       median_standard_us=round(r["median_standard"] / per_us, 2),
                       median_vertical_us=round(r["median_vertical"] / per_us, 2)), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
