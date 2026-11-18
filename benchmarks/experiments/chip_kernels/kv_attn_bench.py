"""
kv_attn_bench.py -- what does a KV cache below 8 bits cost (or save) in time on each chip?

tiny_llm keeps an int8 KV cache; ../tiny_llm/README.md measures the loss of 4, 3, 2 and 1-bit keys and values on the
PC. This bench times one attention head of one decoding step on the chip, at every width: the scores (the int8
query times the 256 cached keys) and the values (the 256 weights times the cached values), the cache packed at its
width and unpacked here into int8 before the same int8 loops:
    scores      unpack the keys (8 bits: nothing) + the int8 dot products of tinylm.c (plain C)
    scores_pie  ESP32-S3 only: the same with symbex_s8_dot_rows (PIE), as tinylm's S3 firmware
    values      unpack the values + tinylm.c's value loop (16-bit weight halves, 4 columns, int32 sums)
    unpack      the unpacking alone (bits < 8)
Packing per position and head (64 values): 8 bits 64 bytes; 4 bits 2 per byte (two's complement nibbles, -7..7);
3 bits 10 per 32-bit word (code + 3, 7 words = 28 bytes); 2 bits 4 per byte (levels -3, -1, 1, 3); 1 bit 8 per byte
(+-1). Unpacking: 256-entry tables for 4 / 2 / 1 bits, shifts for 3, 32-bit stores. Random codes, query and weights (weights < 2^20);
every scores / values run is checked against numpy (all 256 scores, all 64 sums), otherwise its time is not trusted.
Unpack-then-int8 is the simple way, not the fastest: a 1-bit key could skip the unpacking (masked sums).

Where the cache is: internal RAM on every board (one copy); on the ESP32-S3 also the PSRAM, where tiny_llm keeps it:
48 copies (8 layers x 6 heads, one token's attention) run one after another, so the data cache cannot hold them.

    python kv_attn_bench.py --board s3 --port /dev/ttyACM0
    python kv_attn_bench.py --board esp32 --port /dev/ttyUSB0
    python kv_attn_bench.py --board pico2_arm --port /dev/ttyACM1        # Cortex-M33
    python kv_attn_bench.py --board pico2_riscv --port /dev/ttyACM1      # Hazard3

Times: best of 7, per head (cycles at 240 MHz on the ESP32 boards; on the Pico 2 micros() over 16 calls, ns per call).
Results: results/kv_attn_bench_<board>.json
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
LIB = os.path.join(ROOT, "Symbex1_PIO", "lib", "lib_symbex")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from compiler_core.chip.core import BOARDS, flash, open_port  # noqa: E402

HD, T = 64, 256
WIDTHS = [8, 4, 3, 2, 1]
BYTES = {8: 64, 4: 32, 3: 28, 2: 16, 1: 8}          # per position and head
COPIES = 48                                          # PSRAM copies: 8 layers x 6 heads
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
#include "esp_heap_caps.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())
#define OUT printf
#define INNER 1
#define SCALE 1
#else
#include <Arduino.h>
#define NOW() ((uint32_t)micros())
#define OUT Serial.printf
#define INNER 16                                          /* micros() is coarse: 16 calls per sample, ns per call */
#define SCALE 1000
#endif
#if defined(CONFIG_IDF_TARGET_ESP32S3)
#include "hal/symbex_hal.h"
#define HAS_PIE 1
#define HAS_PSRAM 1
#else
#define HAS_PIE 0
#define HAS_PSRAM 0
#endif
#include "bench_data.h"

#define REPS 7
static int8_t q16[HD] __attribute__((aligned(16)));
/* the unpacked rows: written as 32-bit words, read as int8 (char types may read any object) */
static uint32_t kdec32[T * HD / 4] __attribute__((aligned(16)));
static uint32_t vdec32[T * HD / 4] __attribute__((aligned(16)));
#define kdec ((int8_t*)kdec32)
#define vdec ((int8_t*)vdec32)
static uint8_t ram[NW][T * 64] __attribute__((aligned(16)));    /* the packed caches in internal RAM */
#if HAS_PSRAM
static const uint8_t* psram[NW][COPIES];
#endif
static int32_t d[T], pl[T], ph[T];
static int64_t acc[HD];
static uint32_t lut4[256], lut2[256], lut1[2][256];    /* lut4: 2 values in the low 16 bits */

static void init_luts(void) {
    for (int b = 0; b < 256; b++) {
        int8_t v[8];
        v[0] = (int8_t)((int8_t)(uint8_t)(b << 4) >> 4); v[1] = (int8_t)((int8_t)(uint8_t)b >> 4);
        v[2] = v[3] = 0;
        memcpy(&lut4[b], v, 4);
        for (int k = 0; k < 4; k++) v[k] = (int8_t)(2 * ((b >> (2 * k)) & 3) - 3);
        memcpy(&lut2[b], v, 4);
        for (int k = 0; k < 8; k++) v[k] = ((b >> k) & 1) ? 1 : -1;
        memcpy(&lut1[0][b], v, 4);
        memcpy(&lut1[1][b], v + 4, 4);
    }
}

/* the T positions of width index f (WIDTH[f] bits) into int8 rows of HD, one 32-bit word (4 values) per store
   (little-endian targets: lut4's two halves are joined by a shift) */
static void unpack(int f, const uint8_t* s, uint32_t* o32) {
    int8_t* o = (int8_t*)o32;
    switch (WIDTH[f]) {
    case 4: for (int n = 0; n < T * HD / 4; n++) o32[n] = lut4[s[2 * n]] | lut4[s[2 * n + 1]] << 16; break;
    case 3:
        for (int j = 0; j < T; j++, s += 28)
            for (int w = 0; w < 7; w++) {
                uint32_t x; memcpy(&x, s + 4 * w, 4);
                for (int k = 0; k < (w < 6 ? 10 : 4); k++, x >>= 3) *o++ = (int8_t)((int)(x & 7) - 3);
            }
        break;
    case 2: for (int n = 0; n < T * HD / 4; n++) o32[n] = lut2[s[n]]; break;
    case 1: for (int n = 0; n < T * HD / 8; n++) { o32[2 * n] = lut1[0][s[n]]; o32[2 * n + 1] = lut1[1][s[n]]; } break;
    }
}

static void scores_c(const int8_t* k) {
    for (int j = 0; j < T; j++, k += HD) {
        int32_t s = 0;
        for (int i = 0; i < HD; i++) s += q16[i] * k[i];
        d[j] = s;
    }
}

/* tinylm.c's value loop: weights pv_j in two 16-bit halves, 4 columns at a time over every position */
static void values_c(const int8_t* vq) {
    for (int i = 0; i < HD; i += 4) {
        int32_t l0 = 0, l1 = 0, l2 = 0, l3 = 0, h0 = 0, h1 = 0, h2 = 0, h3 = 0;
        const int8_t* v = vq + i;
        for (int j = 0; j < T; j++, v += HD) {
            const int32_t a = pl[j], b = ph[j];
            const int32_t v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
            l0 += a * v0; l1 += a * v1; l2 += a * v2; l3 += a * v3;
            h0 += b * v0; h1 += b * v1; h2 += b * v2; h3 += b * v3;
        }
        acc[i] = (int64_t)h0 * 65536 + l0; acc[i + 1] = (int64_t)h1 * 65536 + l1;
        acc[i + 2] = (int64_t)h2 * 65536 + l2; acc[i + 3] = (int64_t)h3 * 65536 + l3;
    }
}

enum { OP_SCORES, OP_SCORES_PIE, OP_VALUES, OP_UNPACK };
static const char* OP_NAME[] = {"scores", "scores_pie", "values", "unpack"};

static void run(int op, int f, const uint8_t* s) {
    const int8_t* k = (const int8_t*)s;
    switch (op) {
    case OP_SCORES: if (WIDTH[f] != 8) { unpack(f, s, kdec32); k = kdec; } scores_c(k); break;
#if HAS_PIE
    case OP_SCORES_PIE: if (WIDTH[f] != 8) { unpack(f, s, kdec32); k = kdec; } symbex_s8_dot_rows(q16, HD, k, HD, T, d); break;
#endif
    case OP_VALUES: if (WIDTH[f] != 8) { unpack(f, s, vdec32); k = vdec; } values_c(k); break;
    case OP_UNPACK: unpack(f, s, kdec32); break;
    }
}

static int same(int op, int f) {
    if (op == OP_UNPACK) return 1;
    if (op == OP_VALUES) { for (int i = 0; i < HD; i++) if (acc[i] != EXP_ACC[f][i]) return 0; return 1; }
    for (int j = 0; j < T; j++) if (d[j] != EXP_D[f][j]) return 0;
    return 1;
}

static void bench(int op, int f, const uint8_t* const* srcs, int n, const char* where) {
    uint32_t b = 0xFFFFFFFFu;
    memset(d, 0, sizeof d); memset(acc, 0, sizeof acc);
    for (int r = 0; r < REPS; r++) {
        const uint32_t t0 = NOW();
        for (int k = 0; k < INNER; k++)
            for (int c = 0; c < n; c++) run(op, f, srcs[c]);
        const uint32_t t = (uint32_t)((uint64_t)(NOW() - t0) * SCALE / (INNER * n));
        if (t < b) b = t;
    }
    OUT("R:%s|%d|%s|%lu|%d\n", OP_NAME[op], WIDTH[f], where, (unsigned long)b, same(op, f));
}

static void bench_once(void) {
    OUT("KV ATTN BENCH ready (%s)\n", BOARD_NAME);
    for (int f = 0; f < NW; f++) {
        const uint8_t* one[1] = {ram[f]};
        bench(OP_SCORES, f, one, 1, "ram");
#if HAS_PIE
        bench(OP_SCORES_PIE, f, one, 1, "ram");
#endif
        bench(OP_VALUES, f, one, 1, "ram");
        if (WIDTH[f] != 8) bench(OP_UNPACK, f, one, 1, "ram");
#if HAS_PSRAM
        if (psram[f][0]) {
            bench(OP_SCORES, f, psram[f], COPIES, "psram");
            bench(OP_SCORES_PIE, f, psram[f], COPIES, "psram");
            bench(OP_VALUES, f, psram[f], COPIES, "psram");
            if (WIDTH[f] != 8) bench(OP_UNPACK, f, psram[f], COPIES, "psram");
        }
#endif
    }
    OUT("DONE\n");
}

static void setup_data(void) {
    init_luts();
    memcpy(q16, Q, HD);
    for (int j = 0; j < T; j++) { pl[j] = (int32_t)(PV[j] & 0xFFFF); ph[j] = (int32_t)(PV[j] >> 16); }
    for (int f = 0; f < NW; f++) memcpy(ram[f], PACKED[f], T * BYTES_POS[f]);
#if HAS_PSRAM
    for (int f = 0; f < NW; f++)
        for (int c = 0; c < COPIES; c++) {
            uint8_t* p = (uint8_t*)heap_caps_aligned_alloc(16, T * BYTES_POS[f], MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (!p) { psram[f][0] = 0; break; }
            memcpy(p, PACKED[f], T * BYTES_POS[f]);
            psram[f][c] = p;
        }
#endif
}

#if defined(ESP_PLATFORM)
void app_main(void) {
    setup_data();
    for (;;) {
        bench_once();
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
#else
void setup() { Serial.begin(115200); setup_data(); }
void loop() {
    bench_once();
    delay(2000);
}
#endif
"""


def levels(bits):
    return {8: np.arange(-127, 128), 4: np.arange(-7, 8), 3: np.arange(-3, 4), 2: np.array([-3, -1, 1, 3]),
            1: np.array([-1, 1])}[bits]


def pack(c, bits):
    """codes (T, HD) -> bytes (T, BYTES[bits]) as the firmware reads them"""
    if bits == 8:
        return c.astype(np.int8).view(np.uint8)
    if bits == 4:
        u = (c & 0xF).astype(np.uint8)
        return u[:, 0::2] | (u[:, 1::2] << 4)
    if bits == 3:
        u = np.zeros((c.shape[0], 70), np.uint64)
        u[:, :HD] = c + 3
        w = (u.reshape(-1, 7, 10) << (3 * np.arange(10, dtype=np.uint64))).sum(-1).astype("<u4")
        return w.view(np.uint8).reshape(c.shape[0], 28)
    if bits == 2:
        u = ((c + 3) // 2).astype(np.uint8)
        return sum(u[:, k::4] << (2 * k) for k in range(4)).astype(np.uint8)
    u = (c > 0).astype(np.uint8)
    return sum(u[:, k::8] << k for k in range(8)).astype(np.uint8)


def c_array(ctype, name, values, per_line=24, suffix=""):
    vals = [str(int(v)) + suffix for v in np.asarray(values).ravel()]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f"static const {ctype} {name}[{len(vals)}] __attribute__((aligned(16))) = {{\n  " + ",\n  ".join(lines) + "\n};\n"


def make_data():
    rng = np.random.default_rng(2026)
    q = rng.integers(-127, 128, HD)
    pv = rng.integers(0, 1 << 20, T)
    codes = {b: rng.choice(levels(b), (T, HD)) for b in WIDTHS}
    exp_d = {b: codes[b] @ q for b in WIDTHS}
    exp_acc = {b: pv @ codes[b] for b in WIDTHS}
    for b in WIDTHS:                                     # the packing round-trips (numpy side)
        assert pack(codes[b], b).shape == (T, BYTES[b])
    return q, pv, codes, exp_d, exp_acc


def write_data(path, board):
    q, pv, codes, exp_d, exp_acc = make_data()
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define BOARD_NAME \"{board}\"\n#define HD {HD}\n#define T {T}\n#define NW {len(WIDTHS)}\n"
                f"#define COPIES {COPIES}\n")
        f.write("static const int WIDTH[NW] = {" + ", ".join(map(str, WIDTHS)) + "};\n")
        f.write("static const int BYTES_POS[NW] = {" + ", ".join(str(BYTES[b]) for b in WIDTHS) + "};\n")
        f.write(c_array("int8_t", "Q", q))
        f.write(c_array("uint32_t", "PV", pv, suffix="u"))
        for b in WIDTHS:
            f.write(c_array("uint8_t", f"P{b}", pack(codes[b], b)))
        f.write("static const uint8_t* const PACKED[NW] = {" + ", ".join(f"P{b}" for b in WIDTHS) + "};\n")
        f.write(f"static const int32_t EXP_D[NW][T] = {{" + ", ".join(
            "{" + ", ".join(map(str, exp_d[b])) + "}" for b in WIDTHS) + "};\n")
        f.write(f"static const int64_t EXP_ACC[NW][HD] = {{" + ", ".join(
            "{" + ", ".join(f"{int(v)}LL" for v in exp_acc[b]) + "}" for b in WIDTHS) + "};\n")


def write_project(project, port, board):
    src = os.path.join(project, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)
    write_data(os.path.join(src, "bench_data.h"), board)
    if board.startswith("pico2"):                    # Arduino-Pico, as compiler_core/chip/core.py builds SYMBEX
        w("platformio.ini", f"[env:{board}]\nplatform = raspberrypi\nboard = {BOARDS[board]}\nframework = arduino\n"
                            f"board_build.core = earlephilhower\nupload_protocol = picotool\nupload_port = {port}\n"
                            + ("board_build.mcu = rp2350-riscv\n" if board == "pico2_riscv" else "")
                            + "build_flags = -O2\nbuild_unflags = -Os\n")
        w("src/main.cpp", FIRMWARE)
        return
    files = []
    if board == "s3":
        shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
        for name, rel in {"hal_s3.c": ("hal", "esp32s3", "symbex_hal_impl.c"),
                          "hal_generic.c": ("hal", "generic", "symbex_hal_impl.c")}.items():
            shutil.copy(os.path.join(LIB, "src", *rel), os.path.join(src, name))
            files.append(name)
    w("platformio.ini", f"[env:{board}]\nplatform = espressif32\nboard = {BOARDS[board]}\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\n")
    for f in os.listdir(project):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(project, f))
    cfg = ("CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\nCONFIG_ESP32_DEFAULT_CPU_FREQ_240=y\n"
           "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\n")
    if board == "s3":       # as psram_bw_bench.py: the N16R8's octal PSRAM at 80 MHz
        cfg += ("CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\nCONFIG_ESPTOOLPY_FLASHSIZE_16MB=y\nCONFIG_SPIRAM=y\n"
                "CONFIG_SPIRAM_MODE_OCT=y\nCONFIG_SPIRAM_SPEED_80M=y\nCONFIG_ESPTOOLPY_FLASHFREQ_80M=y\n")
    w("sdkconfig.defaults", cfg)
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(kv_attn_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" ' + " ".join(f'"{n}"' for n in files)
                            + ' INCLUDE_DIRS "." ' + ('"include"' if files else "") + ")\n")
    w("src/main.c", FIRMWARE)


def read_results(port, board):
    s = open_port(port, 115200, reset=RESET[board])
    rows, ready, t0 = [], False, time.time()
    while time.time() - t0 < 120:
        line = s.readline().decode(errors="replace").strip()
        if "KV ATTN BENCH ready" in line:
            if f"({board})" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected {board}")
            ready, rows = True, []
        elif ready and line.startswith("R:"):
            op, bits, where, t, ok = line[2:].split("|")
            rows.append((op, int(bits), where, int(t), ok == "1"))
        elif ready and line == "DONE":
            s.close()
            return rows
        elif not ready and board == "s3" and time.time() - t0 > 20 and not line:
            s.write(b"\x55")
    s.close()
    sys.exit("[X] no complete results from the board in 120 s (a run that lost lines is not saved)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=sorted(TICKS), default="s3")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    project = os.path.join(HERE, "runs", f"kv_attn_bench_{a.board}")
    if not a.no_upload:
        write_project(project, a.port, a.board)
        flash(project, verbose=False, what=a.board)
    rows = read_results(a.port, a.board)
    unit, per_us = TICKS[a.board]
    print(f"\n{a.board}, one head, {T} positions x {HD} (best of 7, per head)")
    print(f"  {'op':<12}{'bits':>5}{'where':>7}{unit:>10}{'us':>9}{'KB':>7}  equal")
    for op, b, wh, t, ok in rows:
        print(f"  {op:<12}{b:>5}{wh:>7}{t:>10}{t / per_us:>9.1f}{T * BYTES[b] / 1024:>7.1f}  {'OK' if ok else 'NO'}")
    if not all(r[4] for r in rows):
        sys.exit("[X] a run differs from numpy: the times are not trusted, nothing saved")
    path = os.path.join(HERE, "results", f"kv_attn_bench_{a.board}.json")
    with open(path, "w") as f:
        json.dump(dict(board=a.board, head_dim=HD, positions=T, psram_copies=COPIES if a.board == "s3" else None,
                       reps="best of 7", unit=unit, date=time.strftime("%Y-%m-%d"),
                       bytes_per_position={str(b): BYTES[b] for b in WIDTHS},
                       rows=[dict(op=op, bits=b, where=wh, **({unit: t}), us=round(t / per_us, 2), equal_numpy=ok)
                             for op, b, wh, t, ok in rows]), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
