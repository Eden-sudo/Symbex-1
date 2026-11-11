"""
dense_short_bench.py -- how fast is a first dense layer with FEW uint8 inputs (a control network: 19 sensors -> 64)?
Another project measured symbex_dense_input_u8_bits at 19 -> 64 on the ESP32-S3 at ~280 cycles per neuron, the same
as symbex_dense_input_u8_planes. This times the library's first-layer kernels on real shapes, on any board, and can
build the library as it was at an earlier git revision, to compare before / after a change:

    python dense_short_bench.py --port /dev/ttyACM0                       # ESP32-S3, the library in the work tree
    python dense_short_bench.py --port /dev/ttyACM0 --lib-rev master      # the library as committed on master
    python dense_short_bench.py --board esp32 --port /dev/ttyUSB0
    python dense_short_bench.py --board pico2_arm --port /dev/ttyACM1     # Cortex-M33
    python dense_short_bench.py --board pico2_riscv --port /dev/ttyACM1   # Hazard3
    python dense_short_bench.py --port /dev/ttyACM0 --code ram            # -DSYMBEX_CODE_IN_RAM (SYMBEX_FAST_CODE)

Shapes (in -> out): 19 -> 64, 64 -> 64, 64 -> 3, 784 -> 128. Methods, per shape:
    bits        symbex_dense_input_u8_bits, the input 4-byte aligned
    bits_unal   the same, the input at an odd address (a uint8 buffer the caller did not align)
    mask        symbex_dense_input_u8_mask (raw scores)
    planes      symbex_dense_input_u8_planes (raw scores)
Every method is checked against a plain C reference on the chip (bits and raw scores), otherwise the times are not
trusted. Random weights in flash (const; --weights ram: in RAM). Times: best of 7 runs (cycles on the ESP32 boards; on the Pico 2, micros() over 32 calls, in ns
per call).
Results: results/dense_short_bench_<board>[_<rev>].json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
LIB_REL = os.path.join("Symbex1_PIO", "lib", "lib_symbex")
LIB = os.path.join(ROOT, LIB_REL)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from compiler_core.chip.core import BOARDS, flash, open_port  # noqa: E402

SHAPES = [(19, 64), (64, 64), (64, 3), (784, 128)]
METHODS = ["bits", "bits_unal", "mask", "planes"]
TICKS = {"s3": ("cycles", 240.0), "esp32": ("cycles", 240.0), "pico2_arm": ("ns", 1000.0),
         "pico2_riscv": ("ns", 1000.0)}
RESET = {"s3": "usb", "esp32": "rts", "pico2_arm": "pico", "pico2_riscv": "pico"}

FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#if defined(ESP_PLATFORM)
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())
#define OUT printf
#define INNER 1                                           /* cycles of one call */
#define SCALE 1
#else
#include <Arduino.h>
#define NOW() ((uint32_t)micros())
#define OUT Serial.printf
#define INNER 32                                          /* micros() is coarse: 32 calls per sample, in ns per call */
#define SCALE 1000
#endif
#include "layers/dense/symbex_dense.h"
#include "bench_data.h"

#define REPS 7
static uint8_t xa[MAX_IN + 8] __attribute__((aligned(16)));
static uint32_t planes[8 * MAX_WORDS] __attribute__((aligned(16)));
static int32_t raw[MAX_OUT], ref_raw[MAX_OUT];
static uint32_t bits[(MAX_OUT + 31) / 32], ref_bits[(MAX_OUT + 31) / 32];
static const uint8_t* xin;
static const struct SymbexLayerData* cur;
static const int32_t* thr;

static void run_bits(void) { symbex_dense_input_u8_bits(xin, thr, bits, cur); }
static void run_mask(void) { symbex_dense_input_u8_mask(xin, raw, cur); }
static void run_planes(void) { symbex_dense_input_u8_planes(xin, planes, raw, cur); }

static uint32_t best(void (*f)(void)) {
    uint32_t b = 0xFFFFFFFFu;
    for (int r = 0; r < REPS; r++) {
        const uint32_t t0 = NOW();
        for (int k = 0; k < INNER; k++) f();
        const uint32_t t = (NOW() - t0) * SCALE / INNER;
        if (t < b) b = t;
    }
    return b;
}

static void reference(const struct SymbexLayerData* L, const uint8_t* x, const int32_t* t) {
    const int words = (L->in_features + 31) / 32;
    memset(ref_bits, 0, sizeof ref_bits);
    for (int n = 0; n < L->out_features; n++) {
        int32_t s = 0;
        for (int i = 0; i < L->in_features; i++)
            s += ((L->weights[n * words + i / 32] >> (i % 32)) & 1u) ? x[i] : -(int32_t)x[i];
        ref_raw[n] = s;
        if (s >= t[n]) ref_bits[n / 32] |= 1u << (n % 32);
    }
}
static int same_bits(int n_out) { for (int p = 0; p < (n_out + 31) / 32; p++) if (bits[p] != ref_bits[p]) return 0; return 1; }
static int same_raw(int n_out) { for (int n = 0; n < n_out; n++) if (raw[n] != ref_raw[n]) return 0; return 1; }

static void bench_once(void) {
    OUT("DENSE SHORT BENCH ready (%s)\n", LIB_WHERE);
    for (int s = 0; s < N_SHAPES; s++) {
        cur = &LAYERS[s];
        thr = THRS[s];
        const int n_in = cur->in_features, n_out = cur->out_features;
        reference(cur, X, thr);
        uint32_t c;
        memcpy(xa, X, n_in);                                  /* aligned */
        xin = xa;
        memset(bits, 0, sizeof bits); c = best(run_bits);
        OUT("R:%d|%d|bits|%lu|%d\n", n_in, n_out, (unsigned long)c, same_bits(n_out));
        memcpy(xa + 1, X, n_in);                              /* odd address */
        xin = xa + 1;
        memset(bits, 0, sizeof bits); c = best(run_bits);
        OUT("R:%d|%d|bits_unal|%lu|%d\n", n_in, n_out, (unsigned long)c, same_bits(n_out));
        memcpy(xa, X, n_in);
        xin = xa;
        memset(raw, 0, sizeof raw); c = best(run_mask);
        OUT("R:%d|%d|mask|%lu|%d\n", n_in, n_out, (unsigned long)c, same_raw(n_out));
        memset(raw, 0, sizeof raw); c = best(run_planes);
        OUT("R:%d|%d|planes|%lu|%d\n", n_in, n_out, (unsigned long)c, same_raw(n_out));
    }
    OUT("DONE\n");
}

#if defined(ESP_PLATFORM)
void app_main(void) {
    for (;;) {
        bench_once();
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
#else
void setup() { Serial.begin(115200); }
void loop() {
    bench_once();
    delay(2000);
}
#endif
"""


def c_array(ctype, name, values, const, per_line=16):
    vals = [str(int(v)) + ("u" if ctype == "uint32_t" else "") for v in np.asarray(values).ravel()]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return (f"static {'const ' if const else ''}{ctype} {name}[{len(vals)}] __attribute__((aligned(16))) = {{\n  "
            + ",\n  ".join(lines) + "\n};\n")


def write_data(path, where, weights="flash"):
    rng = np.random.default_rng(2026)
    max_in, max_out = max(s[0] for s in SHAPES), max(s[1] for s in SHAPES)
    x = rng.integers(0, 256, max_in)
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define LIB_WHERE \"{where}\"\n#define N_SHAPES {len(SHAPES)}\n"
                f"#define MAX_IN {max_in}\n#define MAX_OUT {max_out}\n#define MAX_WORDS {(max_in + 31) // 32}\n")
        f.write(c_array("uint8_t", "X", x, True))
        for s, (n_in, n_out) in enumerate(SHAPES):
            words = (n_in + 31) // 32
            b = np.zeros((n_out, words * 32), np.uint8)
            b[:, :n_in] = rng.integers(0, 2, (n_out, n_in))
            w = (b.reshape(n_out, words, 32).astype(np.uint64) << np.arange(32, dtype=np.uint64)).sum(-1)
            f.write(c_array("uint32_t", f"W{s}", w, weights == "flash", 8))
            # thresholds around the scores, so both bit values appear
            score = ((b[:, :n_in].astype(np.int64) * 2 - 1) * x[:n_in]).sum(-1)
            f.write(c_array("int32_t", f"T{s}", score + rng.integers(-60, 61, n_out), weights == "flash"))
        f.write("static const struct SymbexLayerData LAYERS[N_SHAPES] = {"
                + ", ".join(f"{{{i}, {o}, 1, W{s}}}" for s, (i, o) in enumerate(SHAPES)) + "};\n")
        f.write("static const int32_t* const THRS[N_SHAPES] = {" + ", ".join(f"T{s}" for s in range(len(SHAPES)))
                + "};\n")


LIB_FILES = {"symbex_dense.c": ("layers", "dense", "symbex_dense.c"),
             "symbex_and.c": ("ops", "symbex_and.c"),
             "hal_generic.c": ("hal", "generic", "symbex_hal_impl.c")}
BOARD_HAL = {"s3": ("hal_s3.c", ("hal", "esp32s3", "symbex_hal_impl.c")),
             "pico2_arm": ("hal_rp2350.c", ("hal", "rp2350", "symbex_hal_impl.c")),
             "pico2_riscv": ("hal_rp2350.c", ("hal", "rp2350", "symbex_hal_impl.c"))}


def copy_lib(src, rev):
    """include/ and the .c files of lib_symbex into src: from the work tree, or as they were at git revision rev"""
    files = dict(LIB_FILES)
    if rev is None:
        shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
        get = lambda rel: open(os.path.join(LIB, rel), "rb").read()
    else:
        tar = subprocess.run(["git", "-C", ROOT, "archive", rev, f"{LIB_REL}/include"], check=True,
                             capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", src, "--strip-components=3"], input=tar, check=True)
        get = lambda rel: subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{LIB_REL}/{rel}"], check=True,
                                         capture_output=True).stdout
    return files, get


def write_project(project, port, board, rev, code="flash", weights="flash"):
    src = os.path.join(project, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)
    files, get = copy_lib(src, rev)
    if board in BOARD_HAL:
        files.update([BOARD_HAL[board]])
    for name, rel in files.items():
        open(os.path.join(src, name), "wb").write(get(os.path.join("src", *rel)))
    write_data(os.path.join(src, "bench_data.h"), f"library {rev or 'work tree'}, code in {code}, weights in {weights}",
               weights)
    ram = " -DSYMBEX_CODE_IN_RAM" if code == "ram" else ""
    if board.startswith("pico2"):                    # Arduino-Pico, as compiler_core/chip/core.py builds SYMBEX
        w("platformio.ini", f"[env:{board}]\nplatform = raspberrypi\nboard = {BOARDS[board]}\nframework = arduino\n"
                            f"board_build.core = earlephilhower\nupload_protocol = picotool\nupload_port = {port}\n"
                            + ("board_build.mcu = rp2350-riscv\n" if board == "pico2_riscv" else "")
                            + f"build_flags = -Isrc/include -O2{ram}\nbuild_unflags = -Os\n")
        w("src/main.cpp", FIRMWARE)
        return
    w("platformio.ini", f"[env:{board}]\nplatform = espressif32\nboard = {BOARDS[board]}\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\n")
    for f in os.listdir(project):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(project, f))
    w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                            "CONFIG_ESP32_DEFAULT_CPU_FREQ_240=y\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\n"
                            + ("CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\n" if board == "s3" else ""))
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(dense_short_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" ' + " ".join(f'"{n}"' for n in files)
                            + ' INCLUDE_DIRS "." "include")\n'
                            + (f"target_compile_options(${{COMPONENT_LIB}} PRIVATE{ram})\n" if ram else ""))
    w("src/main.c", FIRMWARE)


def read_results(port, board, where):
    s = open_port(port, 115200, reset=RESET[board])
    rows, ready, t0 = [], False, time.time()
    while time.time() - t0 < 60:
        line = s.readline().decode(errors="replace").strip()
        if "DENSE SHORT BENCH ready" in line:
            if f"({where})" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected {where}")
            ready, rows = True, []
        elif ready and line.startswith("R:"):
            n_in, n_out, method, t, ok = line[2:].split("|")
            rows.append((int(n_in), int(n_out), method, int(t), ok == "1"))
        elif ready and line == "DONE":
            s.close()
            return rows
    s.close()
    sys.exit("[X] no results from the board in 60 s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=sorted(TICKS), default="s3")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--lib-rev", default=None, help="git revision of lib_symbex to build (default: the work tree)")
    ap.add_argument("--code", choices=["flash", "ram"], default="flash", help="ram: -DSYMBEX_CODE_IN_RAM")
    ap.add_argument("--weights", choices=["flash", "ram"], default="flash", help="where the weights and thresholds are")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    where = f"library {a.lib_rev or 'work tree'}, code in {a.code}, weights in {a.weights}"
    tag = a.board + (f"_{a.lib_rev.replace('/', '-')}" if a.lib_rev else "") + ("_coderam" if a.code == "ram" else "") + ("_wram" if a.weights == "ram" else "")
    project = os.path.join(HERE, "runs", f"dense_short_bench_{tag}")
    if not a.no_upload:
        write_project(project, a.port, a.board, a.lib_rev, a.code, a.weights)
        flash(project, verbose=False, what=where)
    rows = read_results(a.port, a.board, where)
    unit, per_us = TICKS[a.board]
    print(f"\n{a.board}, {where} (best of 7)")
    print(f"  {'shape':<11}{'method':<11}{unit:>9}{'us':>9}{'us/neuron':>11}  equal to the reference")
    for n_in, n_out, m, t, ok in rows:
        us = t / per_us
        print(f"  {f'{n_in}->{n_out}':<11}{m:<11}{t:>9}{us:>9.2f}{us / n_out:>11.3f}  {'OK' if ok else 'NO'}")
    if not all(r[4] for r in rows):
        sys.exit("[X] a method differs from its reference: the times are not trusted")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"dense_short_bench_{tag}.json")
    with open(path, "w") as f:
        json.dump(dict(board=a.board, library=a.lib_rev or "work tree", code=a.code, weights=a.weights, reps="best of 7",
                       rows=[dict(n_in=i, n_out=o, method=m, **({unit: t} if unit != "us" else {}), us=round(t / per_us, 3),
                                  equal_reference=ok) for i, o, m, t, ok in rows]), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
