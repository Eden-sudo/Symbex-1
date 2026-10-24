"""
pie_swar_bench.py -- a faster way to count bits with the ESP32-S3's multiplier? A quick, separate test.

    python pie_swar_bench.py --port /dev/ttyUSB0
    python pie_swar_bench.py --port /dev/ttyUSB0 --no-upload      # the firmware is already on the board

The S3 has no popcount instruction; its SIMD multiplier (ee.vmulas.u8.accx: 16 bytes times 16 bytes, summed into the
accumulator) is used as a fast adder. Per 128-bit block of (window XOR weights):
    pie         (what lib_symbex does today) 8 steps, one per bit position in the byte: shift, keep that bit of every
                byte, multiply by 1 and accumulate -- the multiplier runs 8 times (~33 instructions per block)
    pie_swar    first count INSIDE every byte with vector SWAR on the 16 bytes at once (ee.vsr.32 + ee.andq +
                ee.vadds.s8: the bits 4 at a time with the mask 0x11, then the two nibbles), so every byte holds its own
                count (0..8); then the multiplier runs ONCE to add the 16 counts (~22 instructions per block)
    lut16       the C 64 KB table in AND form: the fastest method on the S3 for small windows (count_bench.py)
on windows of 5..144 words (64 positions x 64 outputs per layer, rows padded with zeros to 16 bytes). All must give
the same raw values (checksums), otherwise the times are not trusted.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HERE, "runs", "pie_swar_bench_s3")
WIDTHS = [5, 9, 18, 36, 72, 144]

FIRMWARE = r"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"

#define P 64
#define O 64
static uint32_t *X, *Wt;
static int32_t PCW[O];
static uint8_t* LUT16;
static int nw, st;
static volatile uint32_t sink;

static const uint8_t ONES[16] __attribute__((aligned(16))) = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static const uint8_t M11[16] __attribute__((aligned(16))) = {0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11};
static const uint8_t M0F[16] __attribute__((aligned(16))) = {0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F};

static inline void ld(const uint8_t* m, int reg);
#define LOADQ(REG, PTR) { const uint8_t* p_ = (PTR); __asm__ __volatile__("ee.vld.128.ip " REG ", %0, 0" : "+r"(p_) : : "memory"); }

/* ---- pie: today's method. q6 = q7 = ones ---- */
#define STEP(k) "ssai " #k " \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q6 \n\t ee.vmulas.u8.accx q3, q7 \n\t"
static inline int cnt_pie(const uint32_t* a, const uint32_t* b, int words) {
    __asm__ __volatile__("ee.zero.accx");
    for (int g = 0; g < words / 4; g++)
        __asm__ __volatile__("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.xorq q2, q0, q1 \n\t"
                             "ee.andq q3, q2, q6 \n\t ee.vmulas.u8.accx q3, q7 \n\t"
                             STEP(1) STEP(2) STEP(3) STEP(4) STEP(5) STEP(6) STEP(7)
                             : "+r"(a), "+r"(b) : : "memory");
    uint32_t v; __asm__ __volatile__("rur.accx_0 %0" : "=r"(v)); return (int)v;
}

/* ---- pie_swar: vector SWAR per byte, then ONE multiply-accumulate. q5 = 0x11, q6 = 0x0F, q7 = ones.
   PIE's byte add (ee.vadds.s8) SATURATES at 127, so the classic SWAR (bytes up to 170) would break: the bits are
   summed 4 at a time with the mask 0x11 instead (every nibble collects its count, at most 4: a byte never passes 68),
   then the two nibbles are added (at most 8). Checked in numpy against the exact count, with no saturation. ---- */
static inline int cnt_pie_swar(const uint32_t* a, const uint32_t* b, int words) {
    __asm__ __volatile__("ee.zero.accx");
    for (int g = 0; g < words / 4; g++)
        __asm__ __volatile__("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.xorq q2, q0, q1 \n\t"
                             "ee.andq q4, q2, q5 \n\t"
                             "ssai 1 \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q5 \n\t ee.vadds.s8 q4, q4, q3 \n\t"
                             "ssai 2 \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q5 \n\t ee.vadds.s8 q4, q4, q3 \n\t"
                             "ssai 3 \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q5 \n\t ee.vadds.s8 q4, q4, q3 \n\t"
                             "ssai 4 \n\t ee.vsr.32 q3, q4 \n\t ee.andq q3, q3, q6 \n\t ee.andq q4, q4, q6 \n\t ee.vadds.s8 q4, q4, q3 \n\t"
                             "ee.vmulas.u8.accx q4, q7 \n\t"
                             : "+r"(a), "+r"(b) : : "memory");
    uint32_t v; __asm__ __volatile__("rur.accx_0 %0" : "=r"(v)); return (int)v;
}

/* ---- lut16 in AND form (C) ---- */
static inline int cnt_lut16_and(const uint32_t* a, const uint32_t* b, int words) {
    int c = 0;
    for (int i = 0; i < words; i++) { uint32_t v = a[i] & b[i]; c += LUT16[v & 0xFFFF] + LUT16[v >> 16]; }
    return c;
}

/* raw = bits - 2 * differences for every (position, output); XOR forms count the differences directly, the AND
   form corrects with the popcounts of the window and of the weights */
static uint32_t layer(int method) {
    uint32_t ck = 0; const int32_t nbits = nw * 32;
    if (method == 0) { LOADQ("q6", ONES); LOADQ("q7", ONES); }
    if (method == 1) { LOADQ("q5", M11); LOADQ("q6", M0F); LOADQ("q7", ONES); }
    for (int p = 0; p < P; p++) {
        const uint32_t* x = &X[p * st];
        int32_t pcx = 0;
        if (method == 2) for (int i = 0; i < nw; i++) pcx += LUT16[x[i] & 0xFFFF] + LUT16[x[i] >> 16];
        for (int o = 0; o < O; o++) {
            const uint32_t* w = &Wt[o * st];
            int32_t d;
            if (method == 0) d = cnt_pie(x, w, st);
            else if (method == 1) d = cnt_pie_swar(x, w, st);
            else d = pcx + PCW[o] - 2 * cnt_lut16_and(x, w, nw);
            ck = ck * 31u + (uint32_t)(nbits - 2 * d);
        }
    }
    return ck;
}

static void measure(const char* name, int method, int reps) {
    uint32_t ck = layer(method);
    double best = 1e30;
    for (int r = 0; r < 7; r++) {
        uint32_t t0 = (uint32_t)esp_cpu_get_cycle_count();
        for (int k = 0; k < reps; k++) sink = layer(method);
        double t = (double)(uint32_t)((uint32_t)esp_cpu_get_cycle_count() - t0) / reps;
        if (t < best) best = t;
    }
    printf("R:%s|%d|%.2f|%u\n", name, nw, best / (P * O), (unsigned)ck);
    fflush(stdout);
}

void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(500));
    LUT16 = (uint8_t*)heap_caps_malloc(65536, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!LUT16) { printf("ERROR: no RAM for the table\n"); return; }
    for (uint32_t i = 0; i < 65536; i++) LUT16[i] = (uint8_t)__builtin_popcount(i);
    printf("\nPIE SWAR BENCH ready\n");
    fflush(stdout);
    static const int widths[] = {WIDTHS_C};
    uint32_t r = 12345;
    for (unsigned wi = 0; wi < sizeof widths / sizeof widths[0]; wi++) {
        nw = widths[wi]; st = (nw + 3) / 4 * 4;
        X = (uint32_t*)heap_caps_aligned_alloc(16, (size_t)P * st * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        Wt = (uint32_t*)heap_caps_aligned_alloc(16, (size_t)O * st * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!X || !Wt) { printf("ERROR: no RAM for %d words\n", nw); return; }
        memset(X, 0, (size_t)P * st * 4); memset(Wt, 0, (size_t)O * st * 4);
        for (int i = 0; i < P; i++) for (int j = 0; j < nw; j++) X[i * st + j] = (r = r * 1103515245u + 12345u);
        for (int o = 0; o < O; o++) { PCW[o] = 0; for (int j = 0; j < nw; j++) { Wt[o * st + j] = (r = r * 1103515245u + 12345u); PCW[o] += __builtin_popcount(Wt[o * st + j]); } }
        int reps = 36 / nw + 1;
        measure("pie", 0, reps);
        measure("pie_swar", 1, reps);
        measure("lut16_and", 2, reps);
        heap_caps_free(X); heap_caps_free(Wt);
    }
    printf("DONE\n");
    fflush(stdout);
    while (1) vTaskDelay(portMAX_DELAY);
}
"""


def write_project(port):
    os.makedirs(os.path.join(PROJECT, "src"), exist_ok=True)
    w = lambda name, text: open(os.path.join(PROJECT, name), "w").write(text)
    w("platformio.ini", f"[env:s3]\nplatform = espressif32\nboard = esp32-s3-devkitc-1\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\n")
    w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\n")
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(pie_swar_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" INCLUDE_DIRS ".")\n')
    src = FIRMWARE.replace("WIDTHS_C", ", ".join(map(str, WIDTHS)))
    src = src.replace("static inline void ld(const uint8_t* m, int reg);\n", "")
    w("src/main.c", src)


def read_results(port):
    import serial
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, 115200, 1
    s.dtr = s.rts = False
    s.open()
    s.reset_input_buffer()
    s.rts = True
    time.sleep(0.15)
    s.rts = False
    rows, ready, t0 = [], False, time.time()
    while True:
        line = s.readline().decode(errors="replace").strip()
        if "ERROR" in line:
            sys.exit(f"[X] {line}")
        if "PIE SWAR BENCH ready" in line:
            ready = True
        elif ready and line.startswith("R:"):
            m, w, c, ck = line[2:].split("|")
            rows.append(dict(method=m, words=int(w), cycles=float(c), checksum=int(ck)))
            if m == "lut16_and":
                print(f"      measured windows of {w} words")
        elif ready and line == "DONE":
            return rows
        if not ready and time.time() - t0 > 10:
            print("  >>> no answer to the software reset: press EN/RST once")
            t0 = time.time() + 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    write_project(args.port)
    if not args.no_upload:
        print("[*] building and flashing the ESP32-S3 ...")
        if subprocess.run(["pio", "run", "-d", PROJECT, "-t", "upload"]).returncode != 0:
            sys.exit(f"[X] build or upload failed (pio run -d {PROJECT} -t upload)")
    rows = read_results(args.port)
    print("\n    COUNTING WITH THE MULTIPLIER ON THE ESP32-S3  (cycles per output; 64 positions x 64 outputs; best of 7)\n")
    print(f"    {'words':>6}{'bits':>7}  {'pie (today)':>12}{'pie_swar':>11}{'lut16 AND':>11}   {'pie_swar vs pie':>16}"
          f"{'bit-ops/cyc':>13}   checksums")
    for w in WIDTHS:
        r = {x["method"]: x for x in rows if x["words"] == w}
        same = len({x["checksum"] for x in r.values()}) == 1
        print(f"    {w:>6}{w * 32:>7}  {r['pie']['cycles']:>12.1f}{r['pie_swar']['cycles']:>11.1f}{r['lut16_and']['cycles']:>11.1f}"
              f"   {r['pie']['cycles'] / r['pie_swar']['cycles']:>15.2f}x{w * 32 / r['pie_swar']['cycles']:>13.2f}"
              f"   {'agree' if same else 'DIFFER [X]'}")


if __name__ == "__main__":
    main()
