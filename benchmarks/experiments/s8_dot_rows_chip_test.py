"""
s8_dot_rows_chip_test.py -- checks the ESP32-S3 PIE kernel symbex_s8_dot_rows on the chip against a plain C reference,
over every alignment. The PC tests build the S3 HAL in its C mode (SYMBEX_HAL_TEST_NO_PIE), so the PIE assembly is
only checked here (and by the benchmarks' whole networks).

    python s8_dot_rows_chip_test.py --port /dev/ttyACM0

The sweep (weights in RAM and in flash, rows read where they lie, as the exporter leaves them):
    n        1..96 and 127, 128, 129, 255, 256, 784 (every n % 16, short rows, the mlp_fashion layer)
    offset   0..15: the first row starts at every alignment
    stride   n, n + 1, n + 5: the following rows move through the alignments
    rows     17 (one more than 16 alignments)
    values   random int8; every 4th case all -128 (the largest products: checks the 32-bit sum)
x keeps its contract: 16-byte aligned, 16 * ceil(n / 16) bytes, zero past n. Every case is compared row by row.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LIB = os.path.join(ROOT, "Symbex1_PIO", "lib", "lib_symbex")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from compiler_core.chip.core import flash, open_port  # noqa: E402

ROWS, MAX_N, MAX_STRIDE_EXTRA = 17, 784, 5
POOL = 16 + (MAX_N + MAX_STRIDE_EXTRA) * ROWS + 32      # offset + every row + the last row's block

FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hal/symbex_hal.h"
#include "pool.h"

static int8_t wram[POOL] __attribute__((aligned(16)));
static int8_t x[MAX_N + 16] __attribute__((aligned(16)));
static int32_t out[ROWS];
static uint32_t rng = 0x12345678u;
static uint32_t next(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static const int extra_n[] = {127, 128, 129, 255, 256, 784};
static const int extra_stride[] = {0, 1, 5};

/* one case: returns the number of rows that differ from the reference (prints the first ones) */
static int one(const char* where, const int8_t* pool, int n, int off, int stride, int printed) {
    const int8_t* w = pool + off;
    symbex_s8_dot_rows(x, n, w, stride, ROWS, out);
    int bad = 0;
    for (int r = 0; r < ROWS; r++) {
        int32_t s = 0;
        for (int i = 0; i < n; i++) s += (int32_t)x[i] * w[r * stride + i];
        if (s != out[r]) {
            if (printed + bad < 10) printf("F:%s n=%d off=%d stride=%d row=%d got=%ld want=%ld\n", where, n, off,
                                           stride, r, (long)out[r], (long)s);
            bad++;
        }
    }
    return bad;
}

void app_main(void) {
    for (;;) {
        printf("S8 DOT ROWS TEST ready\n");
        long cases = 0, bad_cases = 0, bad_rows = 0;
        for (int k = 0; k < 96 + 6; k++) {
            const int n = k < 96 ? k + 1 : extra_n[k - 96];
            for (int si = 0; si < 3; si++) {
                const int stride = n + extra_stride[si];
                for (int off = 0; off < 16; off++) {
                    const int extreme = (cases & 3) == 3;
                    memset(x, 0, sizeof x);
                    for (int i = 0; i < n; i++) x[i] = extreme ? -128 : (int8_t)next();
                    for (int i = 0; i < POOL; i++) wram[i] = extreme ? -128 : (int8_t)next();
                    for (int p = 0; p < 2; p++) {
                        const int b = one(p ? "flash" : "ram", p ? WFLASH : wram, n, off, stride, (int)bad_rows);
                        cases++;
                        bad_rows += b;
                        bad_cases += b != 0;
                    }
                }
            }
            vTaskDelay(1);
        }
        printf("RESULT cases=%ld bad_cases=%ld bad_rows=%ld\n", cases, bad_cases, bad_rows);
        printf("DONE\n");
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}
"""


def write_project(project, port):
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
                        "project(s8_dot_rows_chip_test)\n")
    if os.path.isdir(os.path.join(src, "include")):
        shutil.rmtree(os.path.join(src, "include"))
    shutil.copytree(os.path.join(LIB, "include"), os.path.join(src, "include"))
    shutil.copy(os.path.join(LIB, "src", "hal", "esp32s3", "symbex_hal_impl.c"), os.path.join(src, "hal_s3.c"))
    shutil.copy(os.path.join(LIB, "src", "hal", "generic", "symbex_hal_impl.c"), os.path.join(src, "hal_generic.c"))
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" "hal_s3.c" "hal_generic.c" INCLUDE_DIRS "." "include")\n')
    w("src/main.c", FIRMWARE)
    pool = np.random.default_rng(42).integers(-128, 128, POOL)
    with open(os.path.join(src, "pool.h"), "w") as f:
        f.write(f"#define ROWS {ROWS}\n#define MAX_N {MAX_N}\n#define POOL {POOL}\n")
        f.write("static const int8_t WFLASH[POOL] __attribute__((aligned(16))) = {\n")
        for i in range(0, POOL, 32):
            f.write("    " + ", ".join(str(int(v)) for v in pool[i:i + 32]) + ",\n")
        f.write("};\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    project = os.path.join(HERE, "runs", "s8_dot_rows_chip_test")
    if not a.no_upload:
        write_project(project, a.port)
        flash(project, verbose=False, what="s8_dot_rows test")
    s = open_port(a.port, 115200, reset="usb")
    ready, t0 = False, time.time()
    while time.time() - t0 < 120:
        line = s.readline().decode(errors="replace").strip()
        if "S8 DOT ROWS TEST ready" in line:
            ready = True
        elif ready and line.startswith("F:"):
            print("  " + line[2:])
        elif ready and line.startswith("RESULT"):
            print(line)
            s.close()
            sys.exit(0 if " bad_cases=0 " in line + " " else "[X] symbex_s8_dot_rows differs from the C reference")
    s.close()
    sys.exit("[X] no result from the board in 120 s")


if __name__ == "__main__":
    main()
