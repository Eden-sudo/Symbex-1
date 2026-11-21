"""
psram_prefetch_bench.py -- can tiny_llm's 1-bit linears on the ESP32-S3 read their PSRAM weights while computing?

psram_bw_bench.py measured symbex_u8_sum_bits_rows at ~40 MB/s on weights in PSRAM, against 61 MB/s for a plain
sequential read and ~70 MB/s for the same kernel on internal RAM: the kernel waits for each cache miss and then
computes, one after the other. This bench tries to overlap the two, on the model's three linear shapes:
    plain       the kernel straight on PSRAM (the reference, as tiny_llm runs it today)
    internal    the kernel on an internal-RAM copy (the ceiling: no PSRAM at all)
    copy_then   memcpy a block of rows to internal RAM, then compute it (no overlap: the cost of the copy)
    preload     the data cache's manual preload (ROM Cache_Start_DCache_Preload) fetches the next block of rows
                while the kernel computes the current one
    autoload    the data cache's automatic next-line fetch on a miss (ROM Cache_Config_DCache_Autoload), size 0..3
    dma         GDMA (esp_async_memcpy) copies the next block to internal RAM while the kernel computes the
                current one from the other buffer (double buffering)
Blocks of 4, 8 and 16 KB. As in psram_bw_bench, every PSRAM run walks 32 fresh copies of the layer (1.5 MB) so that
nothing stays in the 32 KB data cache. Exact: every variant's outputs over the 32 copies are hashed and compared with
the plain kernel's (a run that differs prints ok 0). Best of 7, cycles at 240 MHz, PSRAM at 80 MHz.

    python psram_prefetch_bench.py --port /dev/ttyACM0

Results: results/psram_prefetch_bench_s3.json. ESP-IDF build (framework = espidf), like psram_bw_bench.py.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import psram_bw_bench as bw  # noqa: E402
from kv_attn_bench import check_rows  # noqa: E402
from compiler_core.chip.core import flash, open_port  # noqa: E402

FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_cache.h"
#include "esp_async_memcpy.h"
#include "esp32s3/rom/cache.h"
#include "sdkconfig.h"
#include "hal/symbex_hal.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())
#define REPS 7
#define COPIES 32
#define BIG_BYTES (32 * 1024 * 48)  /* 32 copies of the largest layer (1024 rows of 48 bytes, or 384 of 128) */
#define MAX_BLK (16 * 1024)

static uint8_t *psram, *internal, *buf[2];
static uint32_t lcg = 2026;
static uint32_t rnd(void) { lcg = lcg * 1664525u + 1013904223u; return lcg >> 8; }
static int n_lines;
static uint32_t best;
#define TIME(stmt) do { best = 0xFFFFFFFFu; for (int r_ = 0; r_ < REPS; r_++) { uint32_t t0_ = NOW(); stmt; \
    uint32_t c_ = NOW() - t0_; if (c_ < best) best = c_; } } while (0)

typedef void (*layer_fn)(const uint8_t *prep, int n_in, const uint8_t *w, int words, int n_out, int32_t *out);
static int blk;                       /* block size in bytes of the variant being run */
static int blk_rows(int row) { int r = blk / row; return r & ~1; }   /* even: every DMA size a multiple of 32 */

static void v_plain(const uint8_t *prep, int n_in, const uint8_t *w, int words, int n_out, int32_t *out) {
    symbex_u8_sum_bits_rows(prep, n_in, (const uint32_t *)w, words, n_out, out);
}
static void v_copy_then(const uint8_t *prep, int n_in, const uint8_t *w, int words, int n_out, int32_t *out) {
    const int row = words * 4, rows = blk_rows(row);
    for (int r = 0; r < n_out; r += rows) {
        const int n = n_out - r < rows ? n_out - r : rows;
        memcpy(buf[0], w + (long)r * row, (size_t)n * row);
        symbex_u8_sum_bits_rows(prep, n_in, (const uint32_t *)buf[0], words, n, out + r);
    }
}
static void v_preload(const uint8_t *prep, int n_in, const uint8_t *w, int words, int n_out, int32_t *out) {
    const int row = words * 4, rows = blk_rows(row);
    uint32_t al = Cache_Start_DCache_Preload((uint32_t)w, (uint32_t)((n_out < rows ? n_out : rows) * row), 0);
    for (int r = 0; r < n_out; r += rows) {
        const int n = n_out - r < rows ? n_out - r : rows, nx = r + rows;
        while (!Cache_DCache_Preload_Done()) {}
        if (nx < n_out) {
            const int nn = n_out - nx < rows ? n_out - nx : rows;
            Cache_Start_DCache_Preload((uint32_t)(w + (long)nx * row), (uint32_t)(nn * row), 0);
        }
        symbex_u8_sum_bits_rows(prep, n_in, (const uint32_t *)(w + (long)r * row), words, n, out + r);
    }
    while (!Cache_DCache_Preload_Done()) {}
    Cache_End_DCache_Preload(al);
}
static async_memcpy_handle_t mcp;
static volatile int dma_done;
static int dma_err;
static bool IRAM_ATTR on_done(async_memcpy_handle_t h, async_memcpy_event_t *e, void *a) { dma_done = 1; return false; }
static void dma_start(void *dst, const void *src, size_t n) {
    dma_done = 0;
    esp_err_t e = esp_async_memcpy(mcp, dst, (void *)src, n, on_done, NULL);
    if (e != ESP_OK) { dma_err = e; dma_done = 1; }
}
static void dma_wait(void) { for (long i = 0; !dma_done && i < 100000000L; i++) {} if (!dma_done) dma_err = -1; }
static void v_dma(const uint8_t *prep, int n_in, const uint8_t *w, int words, int n_out, int32_t *out) {
    const int row = words * 4, rows = blk_rows(row);
    dma_start(buf[0], w, (size_t)(n_out < rows ? n_out : rows) * row);
    dma_wait();
    for (int r = 0, b = 0; r < n_out; r += rows, b ^= 1) {
        const int n = n_out - r < rows ? n_out - r : rows, nx = r + rows;
        if (nx < n_out) dma_start(buf[b ^ 1], w + (long)nx * row, (size_t)(n_out - nx < rows ? n_out - nx : rows) * row);
        symbex_u8_sum_bits_rows(prep, n_in, (const uint32_t *)buf[b], words, n, out + r);
        if (nx < n_out) dma_wait();
    }
}

static uint32_t hash_all(layer_fn f, const uint8_t *prep, int n_in, int words, int n_out, long bytes, int32_t *out) {
    uint32_t h = 2166136261u;
    for (int c = 0; c < COPIES; c++) {
        memset(out, 0x5A, (size_t)n_out * 4);
        f(prep, n_in, psram + (long)c * bytes, words, n_out, out);
        const uint8_t *o = (const uint8_t *)out;
        for (int i = 0; i < n_out * 4; i++) h = (h ^ o[i]) * 16777619u;
    }
    return h;
}

static void report(const char *shape, const char *var, int b, long bytes, int ok) {
    printf("R:%s|%s|%d|%ld|%lu|%d\n", shape, var, b, bytes, (unsigned long)best, ok);
    n_lines++;
}

static void bench_once(void) {
    printf("PREFETCH BENCH ready (psram %d MHz, cpu %d MHz, dcache %d KB line %d) | internal free %u\n",
           CONFIG_SPIRAM_SPEED, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, CONFIG_ESP32S3_DATA_CACHE_SIZE / 1024,
           CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    n_lines = 0;
    lcg = 2026;
    for (long i = 0; i < BIG_BYTES; i++) psram[i] = (uint8_t)rnd();
    esp_cache_msync(psram, BIG_BYTES, ESP_CACHE_MSYNC_FLAG_DIR_C2M);   /* the GDMA reads PSRAM, not the cache */
    static int32_t out[1024];
    static uint8_t ws[1024], prep[1024] __attribute__((aligned(16)));
    const int shapes[3][2] = {{384, 384}, {384, 1024}, {1024, 384}};
    const char *names[3] = {"384x384", "384x1024", "1024x384"};
    const int blks[3] = {4096, 8192, MAX_BLK};
    for (int s = 0; s < 3; s++) {
        const int n_in = shapes[s][0], n_out = shapes[s][1], words = n_in / 32;
        const long bytes = (long)n_out * words * 4;
        for (int i = 0; i < n_in; i++) ws[i] = (uint8_t)rnd();
        symbex_u8_prepare(ws, n_in, prep, words);
        const uint32_t ref = hash_all(v_plain, prep, n_in, words, n_out, bytes, out);
        TIME(for (int c = 0; c < COPIES; c++) v_plain(prep, n_in, psram + (long)c * bytes, words, n_out, out));
        best /= COPIES; report(names[s], "plain", 0, bytes, 1);
        memcpy(internal, psram, bytes);
        static int32_t out_a[1024];
        v_plain(prep, n_in, psram, words, n_out, out_a);
        TIME(v_plain(prep, n_in, internal, words, n_out, out));
        report(names[s], "internal", 0, bytes, memcmp(out, out_a, n_out * 4) == 0);
        for (int bi = 0; bi < 3; bi++) {
            blk = blks[bi];
            const layer_fn fs[3] = {v_copy_then, v_preload, v_dma};
            const char *vn[3] = {"copy_then", "preload", "dma"};
            for (int v = 0; v < 3; v++) {
                dma_err = 0;
                const uint32_t h = hash_all(fs[v], prep, n_in, words, n_out, bytes, out);
                TIME(for (int c = 0; c < COPIES; c++) fs[v](prep, n_in, psram + (long)c * bytes, words, n_out, out));
                best /= COPIES;
                if (dma_err) printf("E:%s|%s|%d|dma error %d\n", names[s], vn[v], blk, dma_err);
                report(names[s], vn[v], blk, bytes, h == ref && !dma_err);
            }
        }
        for (int size = 0; size < 4; size++) {
            struct autoload_config cfg = {.ena = 1, .order = 0, .trigger = 0, .size = (uint8_t)size};
            struct autoload_region_config rc = {.region = 0, .ena = 1, .addr = (uint32_t)psram, .size = BIG_BYTES};
            Cache_Config_DCache_Autoload(&cfg);
            Cache_Config_DCache_Region_Autoload(&rc);
            Cache_Enable_DCache_Autoload();
            const uint32_t h = hash_all(v_plain, prep, n_in, words, n_out, bytes, out);
            TIME(for (int c = 0; c < COPIES; c++) v_plain(prep, n_in, psram + (long)c * bytes, words, n_out, out));
            best /= COPIES;
            Cache_Disable_DCache_Autoload();
            report(names[s], "autoload", size, bytes, h == ref);
        }
    }
    printf("DONE:%d\n", n_lines);
}

void app_main(void) {
    psram = heap_caps_aligned_alloc(64, BIG_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    internal = heap_caps_aligned_alloc(64, 1024 * 48, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    buf[0] = heap_caps_aligned_alloc(64, MAX_BLK, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    buf[1] = heap_caps_aligned_alloc(64, MAX_BLK, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    async_memcpy_config_t cfg = ASYNC_MEMCPY_DEFAULT_CONFIG();
    esp_err_t e = esp_async_memcpy_install_gdma_ahb(&cfg, &mcp);
    if (!psram || !internal || !buf[0] || !buf[1] || e != ESP_OK) {
        for (;;) { printf("FAIL: no memory or no GDMA (err %d)\n", e); vTaskDelay(pdMS_TO_TICKS(2000)); }
    }
    for (;;) {
        bench_once();
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
"""


def read_results(port):
    s = open_port(port, 115200, reset="usb")
    rows, ready, header, t0 = [], False, "", time.time()
    while time.time() - t0 < 180:
        line = s.readline().decode(errors="replace").strip()
        if "PREFETCH BENCH ready" in line:
            ready, rows, header = True, [], line
        elif line.startswith("FAIL") or line.startswith("E:"):
            s.close()
            sys.exit("[X] " + line)
        elif ready and line.startswith("R:"):
            shape, var, blk, nbytes, cycles, ok = line[2:].split("|")
            rows.append(dict(shape=shape, variant=var, block=int(blk), bytes=int(nbytes), cycles=int(cycles),
                             ms=int(cycles) / bw.MHZ / 1e3, mb_per_s=int(nbytes) / (int(cycles) / bw.MHZ / 1e6) / 1e6,
                             equal=ok == "1"))
        elif ready and line.startswith("DONE"):
            s.close()
            return header, check_rows(rows, line, lambda r: (r["shape"], r["variant"], r["block"]))
        elif not ready and time.time() - t0 > 20 and not line:
            s.write(b"\x55")
    s.close()
    sys.exit("[X] no results from the board (no DONE)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", default="/dev/ttyACM0")
    a = ap.parse_args()
    project = os.path.join(HERE, "runs", "psram_prefetch_bench")
    bw.FIRMWARE = FIRMWARE
    bw.write_project(project, a.port, 80)
    flash(project, verbose=False, what="PSRAM prefetch bench")
    header, rows = read_results(a.port)
    print(header)
    print(f"{'shape':<10}{'variant':<11}{'block':>7}{'ms':>9}{'MB/s':>8}  equal")
    for r in rows:
        print(f"{r['shape']:<10}{r['variant']:<11}{r['block']:>7}{r['ms']:>9.3f}{r['mb_per_s']:>8.1f}  "
              f"{'yes' if r['equal'] else 'NO'}")
    bad = [r for r in rows if not r["equal"]]
    out = os.path.join(HERE, "results", "psram_prefetch_bench_s3.json")
    with open(out, "w") as f:
        json.dump(dict(board="s3", psram_mhz=80, cpu_mhz=bw.MHZ, header=header, reps=7, best_of="cycles",
                       copies=32, rows=rows, date=time.strftime("%Y-%m-%d")), f, indent=1)
    print("saved", os.path.relpath(out, bw.ROOT))
    if bad:
        print(f"[X] {len(bad)} variants differ from the plain kernel")
        sys.exit(1)


if __name__ == "__main__":
    main()
