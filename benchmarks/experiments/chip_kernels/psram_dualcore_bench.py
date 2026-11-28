"""
psram_dualcore_bench.py -- do tiny_llm's 1-bit linears on the ESP32-S3 get faster when both cores share the rows?

tiny_llm runs on one core: its linears read the weights from PSRAM at ~40 MB/s (plain) or ~56 MB/s (data cache
autoload, psram_prefetch_bench.py). Both cores sit behind the same data cache and the same PSRAM bus, so a second core
helps only if one core alone does not fill the bus. This bench splits the rows of each linear in two contiguous halves:
core 0 computes the first, a task pinned to core 1 the second. cores 1 = core 0 alone (core 1 parked, blocked in the
RTOS), 2 = core 1 spins on a shared counter between jobs, 3 = core 1 blocked and woken by a task notification per job.
Variants, each with cores 1 / 2 / 3:
    plain       the kernel straight on PSRAM (tiny_llm today without --autoload)
    autoload    the data cache's automatic next-line fetch on a miss, size 0..3 (tiny_llm --autoload 3 today)
    internal    the kernel on an internal-RAM copy (no PSRAM: the compute ceiling)
The model's three linear shapes (384x384, 384x1024, 1024x384). As in psram_prefetch_bench, every PSRAM run walks 32
fresh copies of the layer (1.5 MB) so that nothing stays in the cache. Exact: every variant's outputs over the 32
copies are hashed and compared with the 1-core plain kernel's (a run that differs prints ok 0). Best of 7, cycles of
core 0 at 240 MHz (they include waiting for core 1), PSRAM at 80 MHz.

Rule (Nov 26 2026, written before measuring): splitting the linears over two cores goes into tiny_llm's firmware
only if, on all three shapes, 2-core autoload (best size) is at least 1.25x faster than 1-core autoload 3, exact.
Otherwise the second core is reported and left out.

    python psram_dualcore_bench.py --port /dev/ttyACM0

Results: results/psram_dualcore_bench_s3.json. ESP-IDF build (framework = espidf), like psram_bw_bench.py.
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
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp32s3/rom/cache.h"
#include "sdkconfig.h"
#include "hal/symbex_hal.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())
#define REPS 7
#define COPIES 32
#define BIG_BYTES (32 * 1024 * 48)  /* 32 copies of the largest layer (1024 rows of 48 bytes, or 384 of 128) */

static uint8_t *psram, *internal;
static uint32_t lcg = 2026;
static uint32_t rnd(void) { lcg = lcg * 1664525u + 1013904223u; return lcg >> 8; }
static int n_lines;
static uint32_t best;
#define TIME(stmt) do { best = 0xFFFFFFFFu; for (int r_ = 0; r_ < REPS; r_++) { uint32_t t0_ = NOW(); stmt; \
    uint32_t c_ = NOW() - t0_; if (c_ < best) best = c_; } } while (0)

typedef void (*layer_fn)(const uint8_t *prep, int n_in, const uint8_t *w, int words, int n_out, int32_t *out);

/* the job core 1 runs: the second half of the rows */
static struct { const uint8_t *prep; int n_in; const uint8_t *w; int words; int n; int32_t *out; } job;
static volatile uint32_t go, done;
static volatile int mode;            /* 0 parked (blocked: the 1-core runs), 1 spin on go, 2 woken by a notification */
static TaskHandle_t worker_h;

static void set_mode(int m) { mode = m; __sync_synchronize(); xTaskNotifyGive(worker_h); }

static void worker(void *arg) {
    uint32_t seen = 0;
    for (;;) {
        if (mode != 1) ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (mode == 0 || go == seen) continue;
        seen = go;
        __sync_synchronize();
        symbex_u8_sum_bits_rows(job.prep, job.n_in, (const uint32_t *)job.w, job.words, job.n, job.out);
        __sync_synchronize();
        done = seen;
    }
}

static void v_one(const uint8_t *prep, int n_in, const uint8_t *w, int words, int n_out, int32_t *out) {
    symbex_u8_sum_bits_rows(prep, n_in, (const uint32_t *)w, words, n_out, out);
}
static void v_two(const uint8_t *prep, int n_in, const uint8_t *w, int words, int n_out, int32_t *out) {
    const int h = (n_out / 2) & ~1;
    job.prep = prep; job.n_in = n_in; job.w = w + (long)h * words * 4; job.words = words; job.n = n_out - h;
    job.out = out + h;
    __sync_synchronize();
    const uint32_t g = go + 1;
    go = g;
    if (mode == 2) xTaskNotifyGive(worker_h);
    symbex_u8_sum_bits_rows(prep, n_in, (const uint32_t *)w, words, h, out);
    while (done != g) {}
    __sync_synchronize();
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

/* cores: 1 = core 0 alone (core 1 parked), 2 = core 1 spinning between jobs, 3 = core 1 woken per job */
static void cores_mode(int c) { set_mode(c == 1 ? 0 : c == 2 ? 1 : 2); vTaskDelay(pdMS_TO_TICKS(5)); }
static void report(const char *shape, const char *var, int cores, int size, long bytes, int ok) {
    printf("R:%s|%s|%d|%d|%ld|%lu|%d\n", shape, var, cores, size, bytes, (unsigned long)best, ok);
    n_lines++;
}

static void autoload(int on, int size) {
    if (!on) { Cache_Disable_DCache_Autoload(); return; }
    struct autoload_config cfg = {.ena = 1, .order = 0, .trigger = 0, .size = (uint8_t)size};
    struct autoload_region_config rc = {.region = 0, .ena = 1, .addr = (uint32_t)psram, .size = BIG_BYTES};
    Cache_Config_DCache_Autoload(&cfg);
    Cache_Config_DCache_Region_Autoload(&rc);
    Cache_Enable_DCache_Autoload();
}

static void bench_once(void) {
    printf("DUALCORE BENCH ready (psram %d MHz, cpu %d MHz, dcache %d KB line %d) | internal free %u\n",
           CONFIG_SPIRAM_SPEED, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, CONFIG_ESP32S3_DATA_CACHE_SIZE / 1024,
           CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    n_lines = 0;
    lcg = 2026;
    for (long i = 0; i < BIG_BYTES; i++) psram[i] = (uint8_t)rnd();
    static int32_t out[1024], out_a[1024];
    static uint8_t ws[1024], prep[1024] __attribute__((aligned(16)));
    const int shapes[3][2] = {{384, 384}, {384, 1024}, {1024, 384}};
    const char *names[3] = {"384x384", "384x1024", "1024x384"};
    const layer_fn fs[3] = {v_one, v_two, v_two};
    for (int s = 0; s < 3; s++) {
        const int n_in = shapes[s][0], n_out = shapes[s][1], words = n_in / 32;
        const long bytes = (long)n_out * words * 4;
        for (int i = 0; i < n_in; i++) ws[i] = (uint8_t)rnd();
        symbex_u8_prepare(ws, n_in, prep, words);
        const uint32_t ref = hash_all(v_one, prep, n_in, words, n_out, bytes, out);
        cores_mode(1);
        for (int c = 0; c < 3; c++) {
            cores_mode(c + 1);
            const uint32_t h = hash_all(fs[c], prep, n_in, words, n_out, bytes, out);
            TIME(for (int k = 0; k < COPIES; k++) fs[c](prep, n_in, psram + (long)k * bytes, words, n_out, out));
            best /= COPIES; report(names[s], "plain", c + 1, 0, bytes, h == ref);
        }
        memcpy(internal, psram, bytes);
        v_one(prep, n_in, psram, words, n_out, out_a);
        for (int c = 0; c < 3; c++) {
            cores_mode(c + 1);
            memset(out, 0x5A, sizeof out);
            TIME(fs[c](prep, n_in, internal, words, n_out, out));
            report(names[s], "internal", c + 1, 0, bytes, memcmp(out, out_a, n_out * 4) == 0);
        }
        for (int size = 0; size < 4; size++) for (int c = 0; c < 3; c++) {
            cores_mode(c + 1);
            autoload(1, size);
            const uint32_t h = hash_all(fs[c], prep, n_in, words, n_out, bytes, out);
            TIME(for (int k = 0; k < COPIES; k++) fs[c](prep, n_in, psram + (long)k * bytes, words, n_out, out));
            best /= COPIES;
            autoload(0, 0);
            report(names[s], "autoload", c + 1, size, bytes, h == ref);
        }
    }
    cores_mode(1);
    printf("DONE:%d\n", n_lines);
}

void app_main(void) {
    psram = heap_caps_aligned_alloc(64, BIG_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    internal = heap_caps_aligned_alloc(64, 1024 * 48, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!psram || !internal || xTaskCreatePinnedToCore(worker, "core1", 4096, NULL, configMAX_PRIORITIES - 1, &worker_h, 1)
                                   != pdPASS) {
        for (;;) { printf("FAIL: no memory or no core-1 task\n"); vTaskDelay(pdMS_TO_TICKS(2000)); }
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
        if "DUALCORE BENCH ready" in line:
            ready, rows, header = True, [], line
        elif line.startswith("FAIL"):
            s.close()
            sys.exit("[X] " + line)
        elif ready and line.startswith("R:"):
            shape, var, cores, size, nbytes, cycles, ok = line[2:].split("|")
            ms = int(cycles) / bw.MHZ / 1e3
            rows.append(dict(shape=shape, variant=var, cores=int(cores), autoload_size=int(size), bytes=int(nbytes),
                             cycles=int(cycles), ms=ms, mb_per_s=int(nbytes) / (ms / 1e3) / 1e6, equal=ok == "1"))
        elif ready and line.startswith("DONE"):
            s.close()
            return header, check_rows(rows, line, lambda r: (r["shape"], r["variant"], r["cores"], r["autoload_size"]))
        elif not ready and time.time() - t0 > 20 and not line:
            s.write(b"\x55")
    s.close()
    sys.exit("[X] no results from the board (no DONE)")


def rule(rows):
    """The rule of the docstring: 2-core autoload (best size) >= 1.25x 1-core autoload 3 on every shape, exact."""
    verdict = {}
    for shape in sorted({r["shape"] for r in rows}):
        one = next(r["ms"] for r in rows if r["shape"] == shape and r["variant"] == "autoload" and r["cores"] == 1
                   and r["autoload_size"] == 3)
        two = min((r for r in rows if r["shape"] == shape and r["variant"] == "autoload" and r["cores"] > 1),
                  key=lambda r: r["ms"])
        verdict[shape] = dict(one_core_autoload3_ms=one, two_core_best_ms=two["ms"],
                              two_core_best_size=two["autoload_size"], two_core_best_mode=two["cores"],
                              speedup=one / two["ms"])
    ok = all(v["speedup"] >= 1.25 for v in verdict.values()) and all(r["equal"] for r in rows)
    return verdict, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", default="/dev/ttyACM0")
    a = ap.parse_args()
    project = os.path.join(HERE, "runs", "psram_dualcore_bench")
    bw.FIRMWARE = FIRMWARE
    bw.write_project(project, a.port, 80)
    flash(project, verbose=False, what="PSRAM dual-core bench")
    header, rows = read_results(a.port)
    print(header)
    print(f"{'shape':<10}{'variant':<10}{'cores':>6}{'size':>5}{'ms':>9}{'MB/s':>8}  equal")
    for r in rows:
        print(f"{r['shape']:<10}{r['variant']:<10}{r['cores']:>6}{r['autoload_size']:>5}{r['ms']:>9.3f}"
              f"{r['mb_per_s']:>8.1f}  {'yes' if r['equal'] else 'NO'}")
    verdict, ok = rule(rows)
    for shape, v in verdict.items():
        print(f"{shape}: 1 core autoload 3 {v['one_core_autoload3_ms']:.3f} ms, 2 cores best (mode "
              f"{v['two_core_best_mode']}, size {v['two_core_best_size']}) {v['two_core_best_ms']:.3f} ms, "
              f"{v['speedup']:.2f}x")
    print("rule (>= 1.25x on every shape, exact):", "PASSES" if ok else "fails")
    out = os.path.join(HERE, "results", "psram_dualcore_bench_s3.json")
    with open(out, "w") as f:
        json.dump(dict(board="s3", psram_mhz=80, cpu_mhz=bw.MHZ, header=header, reps=7, best_of="cycles of core 0",
                       copies=32, rows=rows, rule=dict(threshold=1.25, per_shape=verdict, passes=ok),
                       date=time.strftime("%Y-%m-%d")), f, indent=1)
    print("saved", os.path.relpath(out, bw.ROOT))
    bad = [r for r in rows if not r["equal"]]
    if bad:
        print(f"[X] {len(bad)} variants differ from the 1-core plain kernel")
        sys.exit(1)


if __name__ == "__main__":
    main()
