"""
int8_head_bench.py -- what does the int8 classifier of train_gap_study.py (--head int8) and small_gap_study.py cost on
the chip, against the binary one the networks run today? A separate test: nothing here is in the CDT or lib_symbex.

    python int8_head_bench.py --port /dev/ttyACM0                  # ESP32-S3: weights in flash, then in RAM
        (the S3's /dev/ttyACM number can change after flashing: /dev/serial/by-id/... keeps its name)
    python int8_head_bench.py --board esp32 --port /dev/ttyUSB0    # classic ESP32

The classifier reads the final map's bits (x = +-1) and gives one logit per class. Three shapes, random weights:
    resnet8   4096 bits (64 x 8 x 8) -> 10 classes, 31 % ones (measured: the trained x1 networks, 3 seeds)
    har_max   128 bits -> 17 classes, 50 % ones
    mlp       128 bits -> 10 classes, 50 % ones
Four ways:
    binary      lib_symbex's symbex_bool_classify: 1-bit weights, AND + popcount per class (what runs today)
    i8_dense    int8 weights, every bit: sum of the weights where the bit is 1 (branchless), logit = 2 * sum - sum(w)
    i8_sparse   int8 weights stored per input bit (bits x classes): only the bits that are 1 add their row of weights
    i8_planes   int8 weights as 8 bit planes: 8 AND + popcount passes (lib_symbex's symbex_and_popcount_rows), the
                planes weighted 1, 2, ..., 64 and -128 (two's complement)
Every way is checked on the chip against a plain C reference from the definition (the int8 ways: every logit equal;
binary: the same class), otherwise the times are not trusted. Times: best of 7 runs, CPU cycles at 240 MHz.
Results: results/int8_head_bench_<board>.json (weights in flash and in RAM).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shortcut_bench import BOARD_HAL, LIB, LIB_FILES, c_array, pack  # noqa: E402
from compiler_core.chip.core import BOARDS, flash, open_port         # noqa: E402  (shortcut_bench adds tools/)

RESET = {"s3": "pico", "esp32": "rts"}   # s3: its USB port comes back after flashing; open it again until then
# (name, input bits, classes, share of ones in the input)
SHAPES = (("resnet8", 4096, 10, 0.31), ("har_max", 128, 17, 0.5), ("mlp", 128, 10, 0.5))
METHODS = ("binary", "i8_dense", "i8_sparse", "i8_planes")

FIRMWARE = r"""
#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "layers/conv/symbex_bool.h"
#include "hal/symbex_hal.h"
#include "bench_data.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())

#define REPS 7
#define MAXC 32
struct Head {
    const char* name;
    int n_bits, n_words, classes, row_words;
    const uint32_t* x;                       /* row_words words, zero padded */
    const uint32_t* wb; const int32_t *wb_counts, *scale, *bias;     /* the binary classifier */
    const int8_t *w8, *w8t;                  /* classes x n_bits, n_bits x classes */
    const uint32_t* planes;                  /* 8 x classes x row_words: bit b of every int8 weight */
    const int32_t* w8sum;                    /* sum of row o */
};
HEADS_TABLE
static const struct Head* H;
static int32_t scratch[MAXC], logit[MAXC], cnt[MAXC], ref[MAXC];
static int cls;

static int argmax32(const int32_t* v, int n) {
    int b = 0;
    for (int o = 1; o < n; o++) if (v[o] > v[b]) b = o;
    return b;
}
static void m_binary(void) {
    cls = symbex_bool_classify(H->x, (uint16_t)H->n_words, (uint32_t)H->n_bits, H->wb, (uint16_t)H->row_words,
                               (uint16_t)H->classes, H->wb_counts, H->scale, H->bias, scratch);
}
static void m_i8_dense(void) {
    for (int o = 0; o < H->classes; o++) {
        const int8_t* w = H->w8 + o * H->n_bits;
        int32_t s = 0;
        for (int j = 0; j < H->n_words; j++) {
            const uint32_t v = H->x[j];
            const int8_t* ww = w + 32 * j;
            for (int b = 0; b < 32; b++) s += (int32_t)ww[b] & -(int32_t)((v >> b) & 1u);
        }
        logit[o] = 2 * s - H->w8sum[o];
    }
    cls = argmax32(logit, H->classes);
}
static void m_i8_sparse(void) {
    const int C = H->classes;
    int32_t acc[MAXC] = {0};
    for (int j = 0; j < H->n_words; j++) {
        uint32_t v = H->x[j];
        while (v) {
            const int8_t* r = H->w8t + (32 * j + __builtin_ctz(v)) * C;
            v &= v - 1;
            for (int o = 0; o < C; o++) acc[o] += r[o];
        }
    }
    for (int o = 0; o < C; o++) logit[o] = 2 * acc[o] - H->w8sum[o];
    cls = argmax32(logit, C);
}
static void m_i8_planes(void) {
    const int C = H->classes, plane = C * H->row_words;
    int32_t s[MAXC] = {0};
    for (int b = 0; b < 8; b++) {
        symbex_and_popcount_rows(H->x, H->planes + b * plane, H->row_words, C, cnt);
        for (int o = 0; o < C; o++) s[o] += b == 7 ? -128 * cnt[o] : cnt[o] << b;
    }
    for (int o = 0; o < C; o++) logit[o] = 2 * s[o] - H->w8sum[o];
    cls = argmax32(logit, C);
}
static void (*const FUNCS[4])(void) = {m_binary, m_i8_dense, m_i8_sparse, m_i8_planes};
static const char* const NAMES[4] = {"binary", "i8_dense", "i8_sparse", "i8_planes"};

static int bit(const uint32_t* v, int i) { return (v[i >> 5] >> (i & 31)) & 1u; }
/* from the definitions: int8 logits sum w * x (x = +-1); binary logits scale * (agreements - disagreements) + bias */
static int check(int method) {
    const int C = H->classes;
    if (method == 0) {
        int best = 0;
        int64_t best_l = 0;
        for (int o = 0; o < C; o++) {
            int32_t raw = 0;
            for (int i = 0; i < H->n_bits; i++) raw += bit(H->x, i) == bit(H->wb + o * H->row_words, i) ? 1 : -1;
            const int64_t l = (int64_t)H->scale[o] * raw + H->bias[o];
            if (o == 0 || l > best_l) { best_l = l; best = o; }
        }
        return cls == best;
    }
    for (int o = 0; o < C; o++) {
        ref[o] = 0;
        for (int i = 0; i < H->n_bits; i++) ref[o] += bit(H->x, i) ? H->w8[o * H->n_bits + i] : -H->w8[o * H->n_bits + i];
        if (ref[o] != logit[o]) return 0;
    }
    return cls == argmax32(ref, C);
}

static uint32_t best(void (*f)(void)) {
    uint32_t b = 0xFFFFFFFFu;
    for (int r = 0; r < REPS; r++) {
        const uint32_t t0 = NOW();
        f();
        const uint32_t t = NOW() - t0;
        if (t < b) b = t;
    }
    return b;
}

void app_main(void) {
    for (;;) {
        printf("HEAD BENCH ready (%s)\n", WEIGHTS_WHERE);
        for (int h = 0; h < N_HEADS; h++) {
            H = &HEADS[h];
            for (int m = 0; m < 4; m++) {
                const uint32_t c = best(FUNCS[m]);
                printf("R:%s_%s|%lu|%d\n", H->name, NAMES[m], (unsigned long)c, check(m));
            }
        }
        printf("DONE\n");
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
"""


def head_arrays(rng, name, n_bits, classes, ones):
    """the C arrays of one shape and its line of the HEADS table"""
    n_words = -(-n_bits // 32)
    row_words = -(-n_words // 4) * 4                      # a multiple of 4 words, as the exporter pads rows
    x = (rng.random(n_bits) < ones).astype(np.uint8)[None]
    wb = rng.integers(0, 2, (classes, n_bits), dtype=np.uint8)
    w8 = rng.integers(-127, 128, (classes, n_bits))
    planes = np.concatenate([((w8 & 0xFF) >> b) & 1 for b in range(8)]).astype(np.uint8)
    p = name.upper()
    text = (c_array("uint32_t", f"{p}_X", pack(x, row_words), False)
            + c_array("uint32_t", f"{p}_WB", pack(wb, row_words), True)
            + c_array("int32_t", f"{p}_WBC", wb.sum(1), True)
            + c_array("int32_t", f"{p}_SC", rng.integers(1, 1000, classes), True)
            + c_array("int32_t", f"{p}_BI", rng.integers(-50000, 50000, classes), True)
            + c_array("int8_t", f"{p}_W8", w8, True)
            + c_array("int8_t", f"{p}_W8T", w8.T, True)
            + c_array("uint32_t", f"{p}_PL", pack(planes, row_words), True)
            + c_array("int32_t", f"{p}_W8S", w8.sum(1), True))
    row = (f'    {{"{name}", {n_bits}, {n_words}, {classes}, {row_words}, {p}_X, {p}_WB, {p}_WBC, {p}_SC, {p}_BI, '
           f"{p}_W8, {p}_W8T, {p}_PL, {p}_W8S}},")
    return text, row


def write_data(path, weights):
    rng = np.random.default_rng(2026)
    texts, rows = zip(*(head_arrays(rng, *s) for s in SHAPES))
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define ROM {'const' if weights == 'flash' else ''}\n"
                f"#define WEIGHTS_WHERE \"weights in {weights}\"\n#define N_HEADS {len(SHAPES)}\n" + "".join(texts)
                + "#define HEADS_TABLE static const struct Head HEADS[N_HEADS] = { \\\n"
                + " \\\n".join(rows) + " \\\n};\n")


def write_project(project, port, weights, board):
    src = os.path.join(project, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)   # noqa: E731
    files = dict(LIB_FILES, **dict([BOARD_HAL[board]] if board in BOARD_HAL else []))
    shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
    for name, rel in files.items():
        shutil.copy(os.path.join(LIB, "src", *rel), os.path.join(src, name))
    write_data(os.path.join(src, "bench_data.h"), weights)
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
                        "project(int8_head_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" ' + " ".join(f'"{n}"' for n in files)
                            + ' INCLUDE_DIRS "." "include")\n')
    w("src/main.c", FIRMWARE)


def read_results(port, weights, board):
    s = open_port(port, 115200, reset=RESET[board])
    rows, ready, t0 = {}, False, time.time()
    while time.time() - t0 < 60:
        line = s.readline().decode(errors="replace").strip()
        if "HEAD BENCH ready" in line:
            if f"(weights in {weights})" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected weights in {weights}")
            ready, rows = True, {}
        elif ready and line.startswith("R:"):
            name, cyc, ok = line[2:].split("|")
            rows[name] = (int(cyc), ok == "1")
        elif ready and line == "DONE":
            s.close()
            return rows
    s.close()
    sys.exit("[X] no results from the board in 60 s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=sorted(RESET), default="s3")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--weights", choices=["flash", "ram", "both"], default="both")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    if a.no_upload and a.weights == "both":
        ap.error("--no-upload reads the firmware already on the board: give --weights flash or --weights ram")
    out = {}
    for weights in (["flash", "ram"] if a.weights == "both" else [a.weights]):
        project = os.path.join(HERE, "runs", f"int8_head_bench_{a.board}_{weights}")
        if not a.no_upload:
            write_project(project, a.port, weights, a.board)
            flash(project, verbose=False, what=f"weights in {weights}")
        rows = read_results(a.port, weights, a.board)
        print(f"\n{a.board}, weights in {weights} (best of 7, 240 MHz)")
        print(f"  {'shape / method':<22}{'cycles':>10}{'us':>9}  equal to the reference")
        for name, (cyc, ok) in rows.items():
            print(f"  {name:<22}{cyc:>10}{cyc / 240:>9.1f}  {'OK' if ok else 'NO'}")
        if not all(ok for _, ok in rows.values()):
            sys.exit("[X] a method differs from its reference: the times are not trusted")
        out[weights] = {name: {"cycles": cyc, "us": round(cyc / 240, 2), "equal_reference": ok}
                        for name, (cyc, ok) in rows.items()}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"int8_head_bench_{a.board}.json")
    with open(path, "w") as f:
        json.dump(dict(board=a.board, mhz=240, reps="best of 7",
                       shapes=[dict(name=n, bits=b, classes=c, ones=p) for n, b, c, p in SHAPES], **out), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
