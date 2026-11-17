"""
psram_bw_bench.py -- is tiny_llm on the ESP32-S3 limited by the PSRAM or by the kernels?

The 15.7 M-parameter model of ../tiny_llm keeps its 1.77 MB of binary weights, the 1.5 MB int8 output head and
the KV cache in the octal PSRAM of the S3 N16R8. A token costs 138 ms (lib_symbex's 8-bit x 1-bit kernel, seed 42):
linear layers 55.6, attention 43.9, head 28.0 ms. This bench runs the same kernels on the same shapes with the data
in PSRAM and in internal RAM, so the memory part of each time is measured rather than guessed:
    seq_c      plain C sum of 32-bit words, 4 at a time (the raw read bandwidth)
    memcpy     PSRAM -> internal RAM in 192 KB blocks
    u8_rows    symbex_u8_sum_bits_rows on the model's three linear shapes (384 -> 384, 384 -> 1024, 1024 -> 384)
    s8_rows    symbex_s8_dot_rows, 384 inputs: 512 rows (fits internal RAM) and the whole 4096-row head
    kv_scores  the attention scores at position 255: symbex_s8_dot_rows over 256 keys of 64 bytes, 6 heads
    kv_values  the attention values at position 255: tinylm.c's int32 loop (16-bit halves, 4 columns), 6 heads
Random data; the PSRAM and internal copies are identical and every kernel's outputs are compared between the two
(the kernels themselves are checked against plain C in their own tests). Best of 7, cycles at 240 MHz. In PSRAM
every weight block is read from a fresh copy (32 copies of a linear layer, 8 KV caches), as the model reads each
layer once per token: a single small block would stay in the data cache and measure only the kernel.

    python psram_bw_bench.py --port /dev/ttyACM0                  # PSRAM at 80 MHz (the default of every build)
    python psram_bw_bench.py --port /dev/ttyACM0 --speed 120      # octal PSRAM at 120 MHz: experimental in ESP-IDF
                                                                  # (needs a stable temperature), flash at 120 too

Results: results/psram_bw_bench_s3_<speed>.json. Built with ESP-IDF through PlatformIO (framework = espidf), not
the Arduino core tiny_llm's firmware uses: the kernels are the same, the PSRAM clock is set by sdkconfig.
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
LIB = os.path.join(ROOT, "Symbex1_PIO", "lib", "lib_symbex")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from compiler_core.chip.core import BOARDS, flash, open_port  # noqa: E402

MHZ = 240
FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "sdkconfig.h"
#include "hal/symbex_hal.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())
#define REPS 7

#define BIG_BYTES (4096 * 384)      /* the int8 head: 1.5 MB */
#define INT_BYTES (512 * 384)       /* 192 KB: the internal copy (512 head rows, or 6 heads of K + V); 8 of them = BIG */
#define DIM 384
#define HID 1024
#define HD 64
#define SEQ 256
#define HEADS 6

static uint8_t *psram, *internal;
static uint32_t lcg = 2026;
static uint32_t rnd(void) { lcg = lcg * 1664525u + 1013904223u; return lcg >> 8; }

static uint32_t best; /* best cycles of the last TIME */
#define TIME(stmt) do { best = 0xFFFFFFFFu; for (int r_ = 0; r_ < REPS; r_++) { uint32_t t0_ = NOW(); stmt; \
    uint32_t c_ = NOW() - t0_; if (c_ < best) best = c_; } } while (0)

static void report(const char *name, const char *where, long bytes, int ok) {
    printf("R:%s|%s|%ld|%lu|%d\n", name, where, bytes, (unsigned long)best, ok);
}

/* ---- seq_c ----------------------------------------------------------------------------------------------------- */
static uint32_t sum32(const uint32_t *p, long words) {
    uint32_t a = 0, b = 0, c = 0, d = 0;
    for (long i = 0; i + 4 <= words; i += 4) { a += p[i]; b += p[i + 1]; c += p[i + 2]; d += p[i + 3]; }
    return a + b + c + d;
}
static volatile uint32_t sink;

/* ---- kv_values: tinylm.c's loop at position t (keys 0..t) ---------------------------------------------------- */
static int32_t pl[SEQ], ph[SEQ];
static void values(const int32_t *p, const int32_t *vm, const int8_t *vqh, int t, int64_t *acc) {
    for (int j = 0; j <= t; j++) {
        const int64_t pv = (int64_t)p[j] * vm[j];
        pl[j] = (int32_t)(pv & 0xFFFF); ph[j] = (int32_t)(pv >> 16);
    }
    for (int i = 0; i < HD; i += 4) {
        int32_t l0 = 0, l1 = 0, l2 = 0, l3 = 0, h0 = 0, h1 = 0, h2 = 0, h3 = 0;
        const int8_t *v = vqh + i;
        for (int j = 0; j <= t; j++, v += HD) {
            const int32_t a = pl[j], b = ph[j];
            const int32_t v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
            l0 += a * v0; l1 += a * v1; l2 += a * v2; l3 += a * v3;
            h0 += b * v0; h1 += b * v1; h2 += b * v2; h3 += b * v3;
        }
        /* the high sums can be negative (vq is signed): multiplied, not shifted (C99 leaves the shift undefined) */
        acc[i] = (int64_t)h0 * 65536 + l0; acc[i + 1] = (int64_t)h1 * 65536 + l1;
        acc[i + 2] = (int64_t)h2 * 65536 + l2; acc[i + 3] = (int64_t)h3 * 65536 + l3;
    }
}

static void bench_once(void) {
    printf("PSRAM BW BENCH ready (psram %d MHz, flash %s, cpu %d MHz) | internal free %u | psram free %u\n",
           CONFIG_SPIRAM_SPEED, CONFIG_ESPTOOLPY_FLASHFREQ, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    lcg = 2026;
    for (long i = 0; i < BIG_BYTES; i++) psram[i] = (uint8_t)rnd();
    memcpy(internal, psram, INT_BYTES);
    static int32_t out_a[4096], out_b[4096];
    static uint8_t ws[HID] __attribute__((aligned(16)));
    static int8_t x8[HID] __attribute__((aligned(16)));
    static int64_t acc_a[HEADS * HD], acc_b[HEADS * HD];
    int ok;

    /* raw reads */
    TIME(sink = sum32((const uint32_t *)psram, BIG_BYTES / 4));
    report("seq_c", "psram", BIG_BYTES, 1);
    TIME(sink = sum32((const uint32_t *)internal, INT_BYTES / 4));
    report("seq_c", "internal", INT_BYTES, 1);
    TIME(for (long o = 0; o < BIG_BYTES; o += INT_BYTES) memcpy(internal, psram + o, INT_BYTES));
    report("memcpy", "psram", BIG_BYTES, 1);
    memcpy(internal, psram, INT_BYTES);

    /* the linear layers: n_in -> n_out, 1-bit weights (n_out rows of n_in / 32 words) on prepared 8-bit inputs */
    const int shapes[3][2] = {{DIM, DIM}, {DIM, HID}, {HID, DIM}};
    const char *shape_names[3] = {"u8_rows_384x384", "u8_rows_384x1024", "u8_rows_1024x384"};
    for (int s = 0; s < 3; s++) {
        const int n_in = shapes[s][0], n_out = shapes[s][1], words = n_in / 32;
        for (int i = 0; i < n_in; i++) ws[i] = (uint8_t)rnd();
        static uint8_t prepared[HID] __attribute__((aligned(16)));
        symbex_u8_prepare(ws, n_in, prepared, words);
        const long bytes = (long)n_out * words * 4;
        /* PSRAM: 32 copies one after another, as the 8 layers' weights pass through the cache once per token
           (a single 18 KB block stays in the data cache across the repetitions and measures nothing) */
        const int copies = 32;
        TIME(for (int c = 0; c < copies; c++)
                 symbex_u8_sum_bits_rows(prepared, n_in, (const uint32_t *)(psram + (long)c * bytes), words, n_out, out_a));
        best /= copies;
        symbex_u8_sum_bits_rows(prepared, n_in, (const uint32_t *)psram, words, n_out, out_a);
        report(shape_names[s], "psram", bytes, 1);
        TIME(symbex_u8_sum_bits_rows(prepared, n_in, (const uint32_t *)internal, words, n_out, out_b));
        ok = memcmp(out_a, out_b, n_out * 4) == 0;
        report(shape_names[s], "internal", bytes, ok);
    }

    /* the int8 head: 384 inputs, rows of 384 int8 */
    for (int i = 0; i < DIM; i++) x8[i] = (int8_t)rnd();
    TIME(for (int c = 0; c < 8; c++) symbex_s8_dot_rows(x8, DIM, (const int8_t *)psram + c * 512L * DIM, DIM, 512, out_a));
    best /= 8;
    symbex_s8_dot_rows(x8, DIM, (const int8_t *)psram, DIM, 512, out_a);
    report("s8_rows_512", "psram", 512L * DIM, 1);
    TIME(symbex_s8_dot_rows(x8, DIM, (const int8_t *)internal, DIM, 512, out_b));
    ok = memcmp(out_a, out_b, 512 * 4) == 0;
    report("s8_rows_512", "internal", 512L * DIM, ok);
    TIME(symbex_s8_dot_rows(x8, DIM, (const int8_t *)psram, DIM, 4096, out_a));
    report("s8_rows_4096", "psram", 4096L * DIM, 1);

    /* the attention at position t: per head K (SEQ x HD) then V (SEQ x HD), 6 heads = 192 KB */
    static int32_t p[SEQ], vm[SEQ];
    for (int j = 0; j < SEQ; j++) { p[j] = (int32_t)(rnd() & 0xFFFF); vm[j] = 1 + (int32_t)(rnd() % 60); }
    static int8_t q8[HD] __attribute__((aligned(16)));
    for (int i = 0; i < HD; i++) q8[i] = (int8_t)rnd();
    const int positions[2] = {255, 127};
    for (int pi = 0; pi < 2; pi++) {
        const int t = positions[pi];
        char name[32];
        const long bytes = (long)HEADS * (t + 1) * HD;
        snprintf(name, sizeof name, "kv_scores_t%d", t);
        /* 8 layers' caches one after another (192 KB each, the whole buffer): read once per token, like the model */
        TIME(for (int l = 0; l < 8; l++) for (int h = 0; h < HEADS; h++)
                 symbex_s8_dot_rows(q8, HD, (const int8_t *)psram + l * (long)INT_BYTES + (long)h * 2 * SEQ * HD, HD,
                                    t + 1, out_a + h * SEQ));
        best /= 8;
        for (int h = 0; h < HEADS; h++)
            symbex_s8_dot_rows(q8, HD, (const int8_t *)psram + (long)h * 2 * SEQ * HD, HD, t + 1, out_a + h * SEQ);
        report(name, "psram", bytes, 1);
        TIME(for (int h = 0; h < HEADS; h++)
                 symbex_s8_dot_rows(q8, HD, (const int8_t *)internal + (long)h * 2 * SEQ * HD, HD, t + 1, out_b + h * SEQ));
        ok = memcmp(out_a, out_b, HEADS * SEQ * 4) == 0;
        report(name, "internal", bytes, ok);
        snprintf(name, sizeof name, "kv_values_t%d", t);
        TIME(for (int l = 0; l < 8; l++) for (int h = 0; h < HEADS; h++)
                 values(p, vm, (const int8_t *)psram + l * (long)INT_BYTES + ((long)h * 2 + 1) * SEQ * HD, t, acc_a + h * HD));
        best /= 8;
        for (int h = 0; h < HEADS; h++) values(p, vm, (const int8_t *)psram + ((long)h * 2 + 1) * SEQ * HD, t, acc_a + h * HD);
        report(name, "psram", bytes, 1);
        TIME(for (int h = 0; h < HEADS; h++)
                 values(p, vm, (const int8_t *)internal + ((long)h * 2 + 1) * SEQ * HD, t, acc_b + h * HD));
        ok = memcmp(acc_a, acc_b, sizeof acc_a) == 0;
        report(name, "internal", bytes, ok);
    }
    printf("DONE\n");
}

void app_main(void) {
    /* 16-byte aligned like tiny_llm's model buffer: symbex_u8_sum_bits_rows takes its PIE path only on aligned rows
       (otherwise its C fallback, 40x slower, which a first run of this bench measured by mistake) */
    psram = heap_caps_aligned_alloc(16, BIG_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    internal = heap_caps_aligned_alloc(16, INT_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!psram || !internal) { for (;;) { printf("FAIL: no memory (psram %p, internal %p)\n", psram, internal);
                                          vTaskDelay(pdMS_TO_TICKS(2000)); } }
    for (;;) {
        bench_once();
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
"""

LIB_FILES = {"hal_s3.c": ("hal", "esp32s3", "symbex_hal_impl.c"), "hal_generic.c": ("hal", "generic", "symbex_hal_impl.c")}

SDKCONFIG = ("CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
             "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_ESP_TASK_WDT_EN=n\nCONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\n"
             "CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y\nCONFIG_SPIRAM=y\nCONFIG_SPIRAM_MODE_OCT=y\n")
SDKCONFIG_SPEED = {80: "CONFIG_SPIRAM_SPEED_80M=y\nCONFIG_ESPTOOLPY_FLASHFREQ_80M=y\n",
                   120: "CONFIG_IDF_EXPERIMENTAL_FEATURES=y\nCONFIG_SPIRAM_SPEED_120M=y\n"
                        "CONFIG_ESPTOOLPY_FLASHFREQ_120M=y\nCONFIG_SPI_FLASH_HPM_ENA=y\n"
                        "CONFIG_SPI_FLASH_HPM_DC_DISABLE=y\n"}
# 120 MHz PSRAM: ESP-IDF has no timing table for PSRAM 120 + flash 80, so the flash runs at 120 too (its High
# Performance Mode, which depends on the flash chip: the N16R8's accepted it)


def write_project(project, port, speed):
    src = os.path.join(project, "src")
    if os.path.isdir(src):
        shutil.rmtree(src)
    os.makedirs(src)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)
    shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
    for name, rel in LIB_FILES.items():
        shutil.copy(os.path.join(LIB, "src", *rel), os.path.join(src, name))
    w("platformio.ini", f"[env:s3]\nplatform = espressif32\nboard = {BOARDS['s3']}\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\n")
    for f in os.listdir(project):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(project, f))
    w("sdkconfig.defaults", SDKCONFIG + SDKCONFIG_SPEED[speed])
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(psram_bw_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" ' + " ".join(f'"{n}"' for n in LIB_FILES)
                            + ' INCLUDE_DIRS "." "include")\n')
    w("src/main.c", FIRMWARE)


def read_results(port, speed):
    s = open_port(port, 115200, reset="usb")
    rows, ready, header, t0 = [], False, "", time.time()
    while time.time() - t0 < 90:
        line = s.readline().decode(errors="replace").strip()
        if "PSRAM BW BENCH ready" in line:
            if f"(psram {speed} MHz" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected psram {speed} MHz")
            ready, rows, header = True, [], line
        elif line.startswith("FAIL"):
            s.close()
            sys.exit("[X] " + line)
        elif ready and line.startswith("R:"):
            name, where, nbytes, cycles, ok = line[2:].split("|")
            rows.append(dict(test=name, where=where, bytes=int(nbytes), cycles=int(cycles), ms=int(cycles) / MHZ / 1e3,
                             mb_per_s=int(nbytes) / (int(cycles) / MHZ / 1e6) / 1e6, equal=ok == "1"))
        elif ready and line == "DONE":
            s.close()
            return header, rows
        elif not ready and time.time() - t0 > 20 and not line:
            s.write(b"\x55")
    s.close()
    sys.exit("[X] no results from the board (the firmware did not print DONE: at 120 MHz the PSRAM may not boot)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--speed", type=int, default=80, choices=(80, 120), help="PSRAM clock in MHz (80 is the default)")
    a = ap.parse_args()
    project = os.path.join(HERE, "runs", f"psram_bw_bench_{a.speed}")
    write_project(project, a.port, a.speed)
    flash(project, verbose=False, what=f"PSRAM at {a.speed} MHz")
    header, rows = read_results(a.port, a.speed)
    print(header)
    print(f"{'test':<20}{'where':<10}{'KB':>8}{'ms':>10}{'MB/s':>10}  equal")
    for r in rows:
        print(f"{r['test']:<20}{r['where']:<10}{r['bytes'] / 1024:>8.0f}{r['ms']:>10.3f}{r['mb_per_s']:>10.1f}  "
              f"{'yes' if r['equal'] else 'NO'}")
    bad = [r for r in rows if not r["equal"]]
    if bad:
        print(f"[X] {len(bad)} tests differ between PSRAM and internal RAM")
    out = os.path.join(HERE, "results", f"psram_bw_bench_s3_{a.speed}.json")
    with open(out, "w") as f:
        json.dump(dict(board="s3", psram_mhz=a.speed, cpu_mhz=MHZ, header=header, reps=7, best_of="cycles",
                       rows=rows, date=time.strftime("%Y-%m-%d")), f, indent=1)
    print("saved", os.path.relpath(out, ROOT))
    if bad:
        sys.exit(1)                         # results kept for the diagnosis, but a mismatch is a failed run


if __name__ == "__main__":
    main()
