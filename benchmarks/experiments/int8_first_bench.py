"""
int8_first_bench.py -- would an int8 first layer also speed up the MLP and conv2d benchmarks on the ESP32-S3? A quick,
separate test (the conv1d networks got 1.25-2.2x from it, see conv1d/har_ultracompact).

    python int8_first_bench.py --port /dev/ttyACM0                  # weights in flash, then in RAM
    python int8_first_bench.py --port /dev/ttyACM0 --weights flash

The real lib_symbex kernels (copied from Symbex1_PIO/lib/lib_symbex), random weights at the real shapes:
    mlp_bits     mlp_fashion's first layer today: 784 uint8 pixels x 128 rows of 1-bit weights (symbex_and_input_u8:
                 prepare + the PIE mask kernel symbex_u8_sum_bits_rows + the +-1 correction)
    mlp_int8     the same layer with int8 weights: pixels - 128 copied to an aligned buffer + symbex_s8_dot_rows
    stem_bits    cnn_cifar_resnet8's stem today, its counts only: 32x32 positions x 16 channels over a 216-bit window
                 (3x3 pixels x 24 thermometer bits; symbex_xor_popcount_rows_n). A LOWER bound: the real stem also
                 builds the thermometer windows (fine_profile: the whole stem is 7.84 ms of the 39.2 ms)
    stem_int8    an int8 stem, whole: 32x32 positions, gather the 3x3x3 window (27 bytes, zero padding), 16 int8 dot
                 products (symbex_s8_dot_rows), threshold, pack the 16 output bits
*_kernel rows time the counting call alone. Every method is checked against a plain C reference on the chip (and
mlp_bits against mlp_int8: the same +-1 weights), otherwise the times are not trusted. Times: best of 7 runs.
Results: results/int8_first_bench_s3.json (weights in flash and in RAM).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LIB = os.path.join(ROOT, "Symbex1_PIO", "lib", "lib_symbex")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from compiler_core.chip.core import flash, open_port  # noqa: E402

MLP_IN, MLP_OUT, MLP_STRIDE = 784, 128, 28          # stride: 25 words rounded up to a multiple of 4 (the exporter's)
STEM_HW, STEM_CH, STEM_BITS, STEM_STRIDE = 32, 16, 216, 8

FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "hal/symbex_hal.h"
#include "bench_data.h"

void symbex_and_input_u8(const uint8_t* x, uint16_t n_in, const uint32_t* w, uint16_t stride, uint16_t n_out,
                         uint8_t* work, int32_t* raw);

#define REPS 7
static uint8_t work[MLP_STRIDE * 32] __attribute__((aligned(16)));
static int8_t xs[MLP_IN + 16] __attribute__((aligned(16)));
static int8_t win[32] __attribute__((aligned(16)));
static uint32_t wbits[STEM_STRIDE] __attribute__((aligned(16)));
static int32_t raw[MLP_OUT], dot[MLP_OUT], ref[MLP_OUT];
static uint16_t stem_out[STEM_HW * STEM_HW], stem_ref[STEM_HW * STEM_HW];
static int32_t stem_cnt[STEM_CH];

static void mlp_bits(void) { symbex_and_input_u8(X, MLP_IN, W1B, MLP_STRIDE, MLP_OUT, work, raw); }
static void mlp_bits_kernel(void) { symbex_u8_sum_bits_rows(work, MLP_IN, W1B, MLP_STRIDE, MLP_OUT, raw); }
static void mlp_int8(void) {
    for (int i = 0; i < MLP_IN; i++) xs[i] = (int8_t)(X[i] ^ 0x80);
    symbex_s8_dot_rows(xs, MLP_IN, W1I, MLP_IN, MLP_OUT, dot);
}
static void mlp_int8_kernel(void) { symbex_s8_dot_rows(xs, MLP_IN, W1I, MLP_IN, MLP_OUT, dot); }

static void stem_bits(void) {
    for (int p = 0; p < STEM_HW * STEM_HW; p++) symbex_xor_popcount_rows_n(wbits, SB, 7, STEM_STRIDE, STEM_CH, stem_cnt);
}
static inline void gather(int y, int x) {
    int k = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            const int yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= STEM_HW || xx < 0 || xx >= STEM_HW) { win[k] = win[k + 1] = win[k + 2] = 0; }
            else { const uint8_t* p = IMG + (yy * STEM_HW + xx) * 3;
                   win[k] = (int8_t)(p[0] ^ 0x80); win[k + 1] = (int8_t)(p[1] ^ 0x80); win[k + 2] = (int8_t)(p[2] ^ 0x80); }
            k += 3;
        }
}
static void stem_int8(void) {
    int32_t d[STEM_CH];
    for (int y = 0; y < STEM_HW; y++)
        for (int x = 0; x < STEM_HW; x++) {
            gather(y, x);
            symbex_s8_dot_rows(win, 27, SI, 27, STEM_CH, d);
            uint16_t b = 0;
            for (int o = 0; o < STEM_CH; o++) b |= (uint16_t)(d[o] >= STHR[o]) << o;
            stem_out[y * STEM_HW + x] = b;
        }
}

static uint32_t best(void (*f)(void)) {
    uint32_t b = 0xFFFFFFFFu;
    for (int r = 0; r < REPS; r++) {
        const uint32_t t0 = esp_cpu_get_cycle_count();
        f();
        const uint32_t t = esp_cpu_get_cycle_count() - t0;
        if (t < b) b = t;
    }
    return b;
}

static int check_mlp(void) {
    int ok = 1;
    for (int j = 0; j < MLP_OUT; j++) {
        int32_t s = 0, sw = 0;
        for (int i = 0; i < MLP_IN; i++) {
            const int w = ((W1B[j * MLP_STRIDE + (i >> 5)] >> (i & 31)) & 1u) ? 1 : -1;
            if (W1I[j * MLP_IN + i] != w) ok = 0;
            s += ((int32_t)X[i] - 128) * w;
            sw += w;
        }
        ref[j] = s;
        if (dot[j] != s) ok = 0;
        if (raw[j] - 128 * sw != s) ok = 0;          /* uint8 x +-1 = (int8 x +-1) + 128 * sum(w) */
    }
    return ok;
}
static int check_stem(void) {
    int ok = 1;
    for (int y = 0; y < STEM_HW; y++)
        for (int x = 0; x < STEM_HW; x++) {
            uint16_t b = 0;
            for (int o = 0; o < STEM_CH; o++) {
                int32_t s = 0, k = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++, k += 3) {
                        const int yy = y + dy, xx = x + dx;
                        if (yy < 0 || yy >= STEM_HW || xx < 0 || xx >= STEM_HW) continue;
                        for (int c = 0; c < 3; c++) s += ((int32_t)IMG[(yy * STEM_HW + xx) * 3 + c] - 128) * SI[o * 27 + k + c];
                    }
                b |= (uint16_t)(s >= STHR[o]) << o;
            }
            stem_ref[y * STEM_HW + x] = b;
            if (stem_out[y * STEM_HW + x] != b) ok = 0;
        }
    for (int o = 0; o < STEM_CH; o++) {
        int32_t c = 0;
        for (int i = 0; i < STEM_STRIDE; i++) c += __builtin_popcount(wbits[i] ^ SB[o * STEM_STRIDE + i]);
        if (stem_cnt[o] != c) ok = 0;
    }
    return ok;
}

void app_main(void) {
    for (int i = 0; i < 7; i++) wbits[i] = 0x9E3779B9u * (i + 1);
    wbits[6] &= (1u << (STEM_BITS - 192)) - 1u;     /* 216 bits: the padding stays 0 on both sides */
    for (;;) {
        printf("INT8 FIRST BENCH ready (%s)\n", WEIGHTS_WHERE);
        uint32_t c[6];
        c[0] = best(mlp_bits); c[1] = best(mlp_bits_kernel); c[2] = best(mlp_int8); c[3] = best(mlp_int8_kernel);
        mlp_bits();                                     /* the kernel alone left raw without the +-1 correction */
        const int ok_mlp = check_mlp();
        c[4] = best(stem_bits); c[5] = best(stem_int8);
        const int ok_stem = check_stem();
        const char* names[6] = {"mlp_bits", "mlp_bits_kernel", "mlp_int8", "mlp_int8_kernel", "stem_bits", "stem_int8"};
        for (int i = 0; i < 6; i++) printf("R:%s|%lu|%d\n", names[i], (unsigned long)c[i], i < 4 ? ok_mlp : ok_stem);
        printf("DONE\n");
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
"""


def c_array(ctype, name, values, rom, per_line=24):
    vals = [str(int(v)) + ("u" if ctype == "uint32_t" else "") for v in np.asarray(values).ravel()]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    q = "ROM " if rom else ""
    return (f"static {q}{ctype} {name}[{len(vals)}] __attribute__((aligned(16))) = {{\n  "
            + ",\n  ".join(lines) + "\n};\n")


def write_data(path, weights):
    rng = np.random.default_rng(12345)
    bits = rng.integers(0, 2, (MLP_OUT, MLP_IN), dtype=np.uint8)
    padded = np.zeros((MLP_OUT, MLP_STRIDE * 32), np.uint8)
    padded[:, :MLP_IN] = bits
    w1b = (padded.reshape(MLP_OUT, MLP_STRIDE, 32).astype(np.uint64) << np.arange(32, dtype=np.uint64)).sum(-1)
    sbits = np.zeros((STEM_CH, STEM_STRIDE * 32), np.uint8)
    sbits[:, :STEM_BITS] = rng.integers(0, 2, (STEM_CH, STEM_BITS))
    sb = (sbits.reshape(STEM_CH, STEM_STRIDE, 32).astype(np.uint64) << np.arange(32, dtype=np.uint64)).sum(-1)
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define ROM {'const' if weights == 'flash' else ''}\n"
                f"#define WEIGHTS_WHERE \"weights in {weights}\"\n"
                f"#define MLP_IN {MLP_IN}\n#define MLP_OUT {MLP_OUT}\n#define MLP_STRIDE {MLP_STRIDE}\n"
                f"#define STEM_HW {STEM_HW}\n#define STEM_CH {STEM_CH}\n#define STEM_BITS {STEM_BITS}\n"
                f"#define STEM_STRIDE {STEM_STRIDE}\n")
        f.write(c_array("uint8_t", "X", rng.integers(0, 256, MLP_IN), False))
        f.write(c_array("uint32_t", "W1B", w1b, True, 8))
        f.write(c_array("int8_t", "W1I", bits.astype(np.int16) * 2 - 1, True))
        f.write(c_array("uint32_t", "SB", sb, True, 8))
        f.write(c_array("int8_t", "SI", rng.integers(-128, 128, STEM_CH * 27), True))
        f.write(c_array("int32_t", "STHR", rng.integers(-3000, 3000, STEM_CH), True))
        f.write(c_array("uint8_t", "IMG", rng.integers(0, 256, STEM_HW * STEM_HW * 3), True))


def write_project(project, port, weights):
    src = os.path.join(project, "src")
    os.makedirs(src, exist_ok=True)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)
    w("platformio.ini", f"[env:s3]\nplatform = espressif32\nboard = esp32-s3-devkitc-1\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\n")
    for f in os.listdir(project):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(project, f))
    w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\n"
                            "CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\n")
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(int8_first_bench)\n")
    import shutil
    if os.path.isdir(os.path.join(src, "include")):
        shutil.rmtree(os.path.join(src, "include"))
    shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
    shutil.copy(os.path.join(LIB, "src", "hal", "esp32s3", "symbex_hal_impl.c"), os.path.join(src, "hal_s3.c"))
    shutil.copy(os.path.join(LIB, "src", "hal", "generic", "symbex_hal_impl.c"), os.path.join(src, "hal_generic.c"))
    shutil.copy(os.path.join(LIB, "src", "ops", "symbex_and.c"), os.path.join(src, "symbex_and.c"))
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" "hal_s3.c" "hal_generic.c" "symbex_and.c" '
                            'INCLUDE_DIRS "." "include")\n')
    w("src/main.c", FIRMWARE)
    write_data(os.path.join(src, "bench_data.h"), weights)


def read_results(port):
    s = open_port(port, 115200, reset="usb")
    rows, ready, t0 = {}, False, time.time()
    while time.time() - t0 < 60:
        line = s.readline().decode(errors="replace").strip()
        if "INT8 FIRST BENCH ready" in line:
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
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--weights", choices=["flash", "ram", "both"], default="both")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    out = {}
    for weights in (["flash", "ram"] if a.weights == "both" else [a.weights]):
        project = os.path.join(HERE, "runs", f"int8_first_bench_s3_{weights}")
        if not a.no_upload:
            write_project(project, a.port, weights)
            flash(project, verbose=False, what=f"weights in {weights}")
        rows = read_results(a.port)
        print(f"\nESP32-S3, weights in {weights} (240 MHz, best of 7)")
        print(f"  {'method':<17}{'cycles':>10}{'ms':>9}  equal to the reference")
        for name, (cyc, ok) in rows.items():
            print(f"  {name:<17}{cyc:>10}{cyc / 240e3:>9.3f}  {'OK' if ok else 'NO'}")
        for a_, b_ in (("mlp_bits", "mlp_int8"), ("mlp_bits_kernel", "mlp_int8_kernel"), ("stem_bits", "stem_int8")):
            print(f"  {a_} / {b_}: {rows[a_][0] / rows[b_][0]:.2f}x")
        if not all(ok for _, ok in rows.values()):
            sys.exit("[X] a method differs from its reference: the times are not trusted")
        out[weights] = {name: dict(cycles=cyc, ms=round(cyc / 240e3, 4), equal_reference=ok)
                        for name, (cyc, ok) in rows.items()}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "int8_first_bench_s3.json")
    with open(path, "w") as f:
        json.dump(dict(board="esp32-s3", mhz=240, reps="best of 7", **out), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
