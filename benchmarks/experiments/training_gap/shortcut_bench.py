"""
shortcut_bench.py -- what does the stride-2 shortcut of ResNet-8 cost on the chip, and how much would the shortcuts
of train_gap_study.py (--shortcut2 pad | addpool) save? A separate test: nothing here is in the CDT or lib_symbex.

    python shortcut_bench.py --port /dev/ttyACM0                  # ESP32-S3: weights in flash, then in RAM
    python shortcut_bench.py --board esp32 --port /dev/ttyUSB0    # classic ESP32
    python shortcut_bench.py --board pico2_arm --port /dev/ttyACM1

The two stride-2 blocks of ResNet-8 x1, random weights and maps at their real shapes:
    s2  16 -> 32 channels, 32x32 -> 16x16        s3  32 -> 64 channels, 16x16 -> 8x8
Per block, four ways (the block conv is lib_symbex's symbex_bool_conv, 3x3 stride 2, as the network runs it):
    conv      the block conv alone (what every way pays)
    or1x1     today's block: conv + symbex_bool_or_pool of the input + the 1x1 binary conv + symbex_bool_or
    pad       ResNet option A: conv + OR pool, the pooled bits ORed into the first in_ch output channels (the new
              channels get -1 = bit 0, the neutral of OR); no 1x1
    addpool   the pooled bit of channel o % in_ch moves the threshold of output o: two thresholds per channel
              (sign(BN(conv) + s * a), s = +-1); OR pool + a copy of symbex_bool_conv whose threshold step picks one
              of the two (built here from the library's source by exact text edits); no 1x1, no OR
Every way is checked against a plain C reference on the chip (the conv from its definition), otherwise the times are
not trusted. Times: best of 7 runs (cycles on the ESP32 boards, microseconds on the Pico 2).
Results: results/shortcut_bench_<board>.json (weights in flash and in RAM).
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

TICKS = {"s3": ("cycles", 240e3), "esp32": ("cycles", 240e3), "pico2_arm": ("us", 1e3), "pico2_riscv": ("us", 1e3)}
RESET = {"s3": "usb", "esp32": "rts", "pico2_arm": "pico", "pico2_riscv": "pico"}
# (name, in_ch, out_ch, in_hw): the stride-2 blocks of ResNet-8 x1
STAGES = (("s2", 16, 32, 32), ("s3", 32, 64, 16))
METHODS = ("conv", "or1x1", "pad", "addpool")

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
#else
#include <Arduino.h>
#define NOW() ((uint32_t)micros())
#define OUT Serial.printf
#endif
#include "layers/conv/symbex_bool.h"
#include "bench_data.h"

#ifdef __cplusplus
extern "C" {
#endif
void addpool_conv(const struct SymbexBoolConv* L, const uint32_t* in, uint32_t* out, uint32_t* window,
                  int32_t* scratch, const uint32_t* sc, const int32_t* thr1, int sc_ch);
#ifdef __cplusplus
}
#endif

#define REPS 7
#define MAXW (16 * 16 * 2)
static uint32_t pooled[MAXW], scm[MAXW], out[MAXW], refpool[MAXW];
static uint32_t window[16] __attribute__((aligned(16)));
static int32_t scratch[3 * 64];

static const struct SymbexBoolConv C2 = {16, 32, 3, 2, 32, 32, 16, 16, C2_ST, C2W, C2T, C2C};
static const struct SymbexBoolConv K2 = {16, 32, 1, 1, 16, 16, 16, 16, K2_ST, K2W, K2T, K2C};
static const struct SymbexBoolConv C3 = {32, 64, 3, 2, 16, 16, 8, 8, C3_ST, C3W, C3T, C3C};
static const struct SymbexBoolConv K3 = {32, 64, 1, 1, 8, 8, 8, 8, K3_ST, K3W, K3T, K3C};
struct Stage { const char* name; const struct SymbexBoolConv *conv, *sc; const uint32_t* in; const int32_t* thr1; };
static const struct Stage STAGES[2] = {{"s2", &C2, &K2, X2, A2}, {"s3", &C3, &K3, X3, A3}};
static const struct Stage* cur;

static uint32_t n_out_words(void) {
    const struct SymbexBoolConv* L = cur->conv;
    return (uint32_t)L->out_h * L->out_w * ((L->out_ch + 31) / 32);
}
static void m_conv(void) { symbex_bool_conv(cur->conv, cur->in, out, window, scratch); }
static void m_or1x1(void) {
    const struct SymbexBoolConv* L = cur->conv;
    symbex_bool_conv(L, cur->in, out, window, scratch);
    symbex_bool_or_pool(cur->in, pooled, L->in_ch, L->in_h, L->in_w);
    symbex_bool_conv(cur->sc, pooled, scm, window, scratch);
    symbex_bool_or(out, scm, n_out_words());
}
static void m_pad(void) {                            /* in_ch <= 32: one input word per pixel */
    const struct SymbexBoolConv* L = cur->conv;
    symbex_bool_conv(L, cur->in, out, window, scratch);
    symbex_bool_or_pool(cur->in, pooled, L->in_ch, L->in_h, L->in_w);
    const int pout = (L->out_ch + 31) / 32, n = L->out_h * L->out_w;
    for (int p = 0; p < n; p++) out[p * pout] |= pooled[p];
}
static void m_addpool(void) {
    const struct SymbexBoolConv* L = cur->conv;
    symbex_bool_or_pool(cur->in, pooled, L->in_ch, L->in_h, L->in_w);
    addpool_conv(L, cur->in, out, window, scratch, pooled, cur->thr1, L->in_ch);
}
static void (*const FUNCS[4])(void) = {m_conv, m_or1x1, m_pad, m_addpool};
static const char* const NAMES[4] = {"conv", "or1x1", "pad", "addpool"};

static int bit(const uint32_t* v, int i) { return (v[i >> 5] >> (i & 31)) & 1u; }
/* raw = agreements - disagreements over the in-image pixels of the window, from the definition */
static int32_t ref_raw(const struct SymbexBoolConv* L, const uint32_t* in, int oy, int ox, int o) {
    const int k = L->kernel, pad = k / 2, cin = L->in_ch, pin = (cin + 31) / 32;
    int32_t raw = 0;
    for (int q = 0; q < k * k; q++) {
        const int y = oy * L->stride - pad + q / k, x = ox * L->stride - pad + q % k;
        if (y < 0 || y >= L->in_h || x < 0 || x >= L->in_w) continue;
        for (int c = 0; c < cin; c++)
            raw += bit(&in[(y * L->in_w + x) * pin], c) == bit(&L->weights[o * L->row_words], q * cin + c) ? 1 : -1;
    }
    return raw;
}
static int check(int method) {
    const struct SymbexBoolConv* L = cur->conv;
    const int pout = (L->out_ch + 31) / 32, W = L->in_w;
    for (int y = 0; y < L->out_h; y++)
        for (int x = 0; x < L->out_w; x++) {
            const uint32_t* a = &cur->in[(2 * y) * W + 2 * x];
            refpool[y * L->out_w + x] = a[0] | a[1] | a[W] | a[W + 1];
        }
    int ok = 1;
    for (int y = 0; y < L->out_h; y++)
        for (int x = 0; x < L->out_w; x++) {
            const uint32_t pb = refpool[y * L->out_w + x];
            for (int o = 0; o < L->out_ch; o++) {
                int32_t t = L->thresholds[o];
                if (method == 3 && ((pb >> (o % L->in_ch)) & 1u)) t = cur->thr1[o];
                int b = ref_raw(L, cur->in, y, x, o) >= t;
                if (method == 1) b |= ref_raw(cur->sc, refpool, y, x, o) >= cur->sc->thresholds[o];
                if (method == 2 && o < L->in_ch) b |= (int)((pb >> o) & 1u);
                if (bit(&out[(y * L->out_w + x) * pout], o) != b) ok = 0;
            }
        }
    return ok;
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

static void bench_once(void) {
    OUT("SHORTCUT BENCH ready (%s)\n", WEIGHTS_WHERE);
    for (int s = 0; s < 2; s++) {
        cur = &STAGES[s];
        for (int m = 0; m < 4; m++) {
            const uint32_t c = best(FUNCS[m]);
            OUT("R:%s_%s|%lu|%d\n", NAMES[m], cur->name, (unsigned long)c, check(m));
        }
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


def addpool_source():
    """symbex_bool_conv of the library with a two-threshold step: the threshold of output o is thr1[o] where bit
    o % sc_ch of the pooled shortcut pixel is 1, thresholds[o] where it is 0. Exact text edits: stops if the library
    source changed."""
    src = open(os.path.join(LIB, "src", "layers", "conv", "symbex_bool.c")).read()
    start = src.index("static inline void put_bits(")
    body = src[start:src.index("\nvoid symbex_thermometer(")]
    edits = [
        ("void symbex_bool_conv(const struct SymbexBoolConv* L, const uint32_t* in, uint32_t* out,\n"
         "                      uint32_t* window, int32_t* scratch) {",
         "void addpool_conv(const struct SymbexBoolConv* L, const uint32_t* in, uint32_t* out, uint32_t* window,\n"
         "                  int32_t* scratch, const uint32_t* sc, const int32_t* thr1, int sc_ch) {\n"
         "    int32_t* adj1 = scratch + L->out_ch;         /* AND form: the adjusted second thresholds */", 1),
        ("int32_t* cnt = scratch + n_out;", "int32_t* cnt = scratch + 2 * n_out;", 1),
        ("adj[o] = thr[o] + 2 * tot;", "adj[o] = thr[o] + 2 * tot;\n            adj1[o] = thr1[o] + 2 * tot;", 1),
        ("uint32_t* o_px = &out[(oy * L->out_w + ox) * pout];",
         "uint32_t* o_px = &out[(oy * L->out_w + ox) * pout];\n"
         "            const uint32_t sw = sc[oy * L->out_w + ox];   /* the pooled shortcut pixel (sc_ch <= 32) */", 1),
        (">= thr[o]) << b", ">= TSEL(o)) << b", 2),
        ("adj[o] + c0", "ASEL(o) + c0", 2),
    ]
    for old, new, n in edits:
        if body.count(old) != n:
            sys.exit(f"[X] symbex_bool.c changed: {old!r} found {body.count(old)} times, expected {n}")
        body = body.replace(old, new)
    return ("/* built by shortcut_bench.py from lib_symbex's symbex_bool.c: symbex_bool_conv with two thresholds */\n"
            "#include <string.h>\n#include \"layers/conv/symbex_bool.h\"\n#include \"hal/symbex_hal.h\"\n"
            "#define PF_T(v) do { } while (0)\n#define PF_ADD(k, t0) do { } while (0)\n"
            "#define SEL(o) ((sw >> ((o) & (sc_ch - 1))) & 1u)    /* sc_ch: a power of 2 */\n"
            "#define TSEL(o) (SEL(o) ? thr1[o] : thr[o])\n#define ASEL(o) (SEL(o) ? adj1[o] : adj[o])\n" + body)


def c_array(ctype, name, values, rom, per_line=16):
    vals = [str(int(v)) + ("u" if ctype == "uint32_t" else "") for v in np.asarray(values).ravel()]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    q = "ROM " if rom else ""
    return (f"static {q}{ctype} {name}[{len(vals)}] __attribute__((aligned(16))) = {{\n  "
            + ",\n  ".join(lines) + "\n};\n")


def pack(bits, words):
    """rows of bits -> rows of `words` uint32 (bit i of a row = bit i % 32 of word i // 32), zero padded"""
    p = np.zeros((bits.shape[0], words * 32), np.uint64)
    p[:, :bits.shape[1]] = bits
    return (p.reshape(bits.shape[0], words, 32) << np.arange(32, dtype=np.uint64)).sum(-1)


def conv_arrays(rng, prefix, cin, cout, k):
    st = -(-(k * k * cin) // 32)
    st = -(-st // 4) * 4                                  # the exporter's row stride: a multiple of 4 words
    w = rng.integers(0, 2, (cout, k * k * cin), dtype=np.uint8)
    counts = w.reshape(cout, k * k, cin).sum(-1)
    thr = rng.integers(-12, 13, cout)                     # around 0: both bit values occur
    return (f"#define {prefix}_ST {st}\n" + c_array("uint32_t", f"{prefix}W", pack(w, st), True)
            + c_array("int32_t", f"{prefix}T", thr, True) + c_array("int16_t", f"{prefix}C", counts, True)), thr


def write_data(path, weights):
    rng = np.random.default_rng(2026)
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define ROM {'const' if weights == 'flash' else ''}\n"
                f"#define WEIGHTS_WHERE \"weights in {weights}\"\n")
        for name, cin, cout, hw in STAGES:
            i = name[1]
            f.write(c_array("uint32_t", f"X{i}", pack(rng.integers(0, 2, (hw * hw, cin)), 1), False))
            text, thr = conv_arrays(rng, f"C{i}", cin, cout, 3)
            f.write(text)
            f.write(conv_arrays(rng, f"K{i}", cin, cout, 1)[0])
            f.write(c_array("int32_t", f"A{i}", thr - 2 * rng.integers(0, 9, cout), True))   # s = +1: lower


LIB_FILES = {"symbex_and.c": ("ops", "symbex_and.c"), "symbex_bool.c": ("layers", "conv", "symbex_bool.c"),
             "hal_generic.c": ("hal", "generic", "symbex_hal_impl.c")}
BOARD_HAL = {"s3": ("hal_s3.c", ("hal", "esp32s3", "symbex_hal_impl.c")),
             "pico2_arm": ("hal_rp2350.c", ("hal", "rp2350", "symbex_hal_impl.c")),
             "pico2_riscv": ("hal_rp2350.c", ("hal", "rp2350", "symbex_hal_impl.c"))}


def write_project(project, port, weights, board="s3"):
    src = os.path.join(project, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)
    files = dict(LIB_FILES, **dict([BOARD_HAL[board]] if board in BOARD_HAL else []))
    shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
    for name, rel in files.items():
        shutil.copy(os.path.join(LIB, "src", *rel), os.path.join(src, name))
    w("src/addpool_conv.c", addpool_source())
    write_data(os.path.join(src, "bench_data.h"), weights)
    if board.startswith("pico2"):
        w("platformio.ini", f"[env:{board}]\nplatform = raspberrypi\nboard = {BOARDS[board]}\nframework = arduino\n"
                            f"board_build.core = earlephilhower\nupload_protocol = picotool\nupload_port = {port}\n"
                            + ("board_build.mcu = rp2350-riscv\n" if board == "pico2_riscv" else "")
                            + "build_flags = -Isrc/include -O2\nbuild_unflags = -Os\n")
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
                        "project(shortcut_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" "addpool_conv.c" '
                            + " ".join(f'"{n}"' for n in files) + ' INCLUDE_DIRS "." "include")\n')
    w("src/main.c", FIRMWARE)


def read_results(port, weights, board="s3"):
    s = open_port(port, 115200, reset=RESET[board])
    rows, ready, t0 = {}, False, time.time()
    while time.time() - t0 < 60:
        line = s.readline().decode(errors="replace").strip()
        if "SHORTCUT BENCH ready" in line:
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
    ap.add_argument("--board", choices=sorted(TICKS), default="s3")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--weights", choices=["flash", "ram", "both"], default="both")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    if a.no_upload and a.weights == "both":
        ap.error("--no-upload reads the firmware already on the board: give --weights flash or --weights ram")
    out = {}
    unit, per_ms = TICKS[a.board]
    for weights in (["flash", "ram"] if a.weights == "both" else [a.weights]):
        project = os.path.join(HERE, "runs", f"shortcut_bench_{a.board}_{weights}")
        if not a.no_upload:
            write_project(project, a.port, weights, a.board)
            flash(project, verbose=False, what=f"weights in {weights}")
        rows = read_results(a.port, weights, a.board)
        print(f"\n{a.board}, weights in {weights} (best of 7)")
        print(f"  {'method':<12}{unit:>10}{'ms':>9}  equal to the reference")
        for name, (cyc, ok) in rows.items():
            print(f"  {name:<12}{cyc:>10}{cyc / per_ms:>9.3f}  {'OK' if ok else 'NO'}")
        both = {m: sum(rows[f"{m}_{s}"][0] for s, *_ in STAGES) / per_ms for m in METHODS}
        print("  both stride-2 blocks: " + ", ".join(f"{m} {v:.3f} ms" for m, v in both.items()))
        for m in ("pad", "addpool"):
            print(f"  {m} saves {both['or1x1'] - both[m]:.3f} ms per image vs or1x1")
        if not all(ok for _, ok in rows.values()):
            sys.exit("[X] a method differs from its reference: the times are not trusted")
        out[weights] = dict({name: {unit: cyc, "ms": round(cyc / per_ms, 4), "equal_reference": ok}
                             for name, (cyc, ok) in rows.items()},
                            both_blocks_ms={m: round(v, 4) for m, v in both.items()})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"shortcut_bench_{a.board}.json")
    with open(path, "w") as f:
        json.dump(dict(board=a.board, mhz=240 if a.board in ("s3", "esp32") else 150, reps="best of 7",
                       stages=[dict(name=n, in_ch=c, out_ch=o, in_hw=h) for n, c, o, h in STAGES], **out), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
