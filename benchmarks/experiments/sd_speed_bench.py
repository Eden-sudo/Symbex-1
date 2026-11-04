"""
sd_speed_bench.py -- how fast can an ESP32-S3 read its microSD card, and who limits it: the card or the bus? The
question behind storing model weights (an LLM, a large layer) on the card.

    python sd_speed_bench.py --port /dev/ttyACM0 --card "SanDisk Ultra microSDHC A1 32 GB"
    python sd_speed_bench.py --port /dev/ttyACM0 --khz 400,5000,10000 --clk 39 --cmd 38 --d0 40

The firmware reads raw sectors (no file system, nothing written) over SDMMC with one data line, one bus clock per
power-up: sequential reads of 512 B, 4 KB, 32 KB and 128 KB, then random 4 KB reads. Each row gives MB/s against the
bus limit (clock x 1 bit / 8): near 100 % means the bus (the slot wiring) is the bottleneck, not the card.

Card hangs: a chip reset in the middle of a transfer leaves the card stuck until it loses power (card init fails with
a timeout). So the firmware waits for the host before touching the card, runs one clock, and restarts with the card
idle. If init still fails, unplug the board's USB and plug it back, then rerun from that clock.
Results: results/sd_speed_s3.json.
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from compiler_core.chip.core import flash  # noqa: E402

FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_random.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include "esp_log.h"
#include "esp_system.h"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "pins.h"

#define BIG 256                                       /* sectors per read, at most (128 KB) */
#define START 2048                                    /* first sector read: past the partition table */
static const int chunks[] = {1, 8, 64, 256};

static int run(int khz) {
    uint8_t* buf = heap_caps_malloc(BIG * 512, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!buf) { printf("no DMA buffer\n"); return 1; }
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = khz;                          /* one clock per boot: the card is initialized once, at it */
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1; slot.clk = PIN_CLK; slot.cmd = PIN_CMD; slot.d0 = PIN_D0;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
    sdmmc_host_init();
    sdmmc_host_init_slot(host.slot, &slot);
    sdmmc_card_t card;
    esp_err_t e = sdmmc_card_init(&host, &card);
    if (e != ESP_OK) { printf("card init: %s\n", esp_err_to_name(e)); return 1; }
    const uint32_t sectors = (uint32_t)card.csd.capacity;
    printf("I:%s|%lu|%d\n", card.cid.name, (unsigned long)sectors, card.real_freq_khz);
    const double busmax = card.real_freq_khz * 1e3 / 8 / 1e6;   /* MB/s with 1 data line */
    int err = 0;
    for (int ci = 0; ci < (int)(sizeof(chunks) / sizeof(chunks[0])) && !err; ci++) {
        const int n = chunks[ci];
        int reads = khz * 64 / (n * 512);             /* ~0.5 s at the bus limit */
        if (reads < 4) reads = 4;
        const int64_t t0 = esp_timer_get_time();
        for (int r = 0; r < reads && !err; r++)
            err = sdmmc_read_sectors(&card, buf, START + (uint32_t)r * n, n) != ESP_OK;
        const double s = (esp_timer_get_time() - t0) / 1e6;
        const double mbs = (double)reads * n * 512 / s / 1e6;
        if (!err) printf("R:%d|seq|%d|%.3f|%.3f|%.0f|%.2f\n", khz, n * 512, mbs, busmax, 100 * mbs / busmax,
                         s * 1e3 / reads);
    }
    if (!err) {                                       /* random 4 KB reads over the whole card */
        const int nr = khz < 5000 ? 10 : 200;
        const int64_t t0 = esp_timer_get_time();
        for (int r = 0; r < nr && !err; r++)
            err = sdmmc_read_sectors(&card, buf, START + (esp_random() % (sectors - START - 8)) / 8 * 8, 8) != ESP_OK;
        const double s = (esp_timer_get_time() - t0) / 1e6;
        if (!err) printf("R:%d|rand4k|4096|%.3f|%.3f|%.0f|%.2f\n", khz, nr * 4096 / s / 1e6, busmax,
                         100 * nr * 4096 / s / 1e6 / busmax, s * 1e3 / nr);
    }
    if (err) printf("read error at %d kHz: stopped\n", khz);
    heap_caps_free(buf);
    return err;
}

void app_main(void) {
    usb_serial_jtag_driver_config_t usb = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_serial_jtag_driver_install(&usb);
    usb_serial_jtag_vfs_use_driver();
    /* wait for the host's 0x55 before touching the card: opening the port resets the chip, and a reset in the middle
       of a transfer leaves the card stuck until it loses power */
    uint8_t hello = 0;
    while (hello != 0x55) usb_serial_jtag_read_bytes(&hello, 1, portMAX_DELAY);
    uint8_t code = 0;                                 /* then the clock: kHz / 200, as one byte (2 = 400 kHz) */
    usb_serial_jtag_read_bytes(&code, 1, portMAX_DELAY);
    esp_log_level_set("*", ESP_LOG_NONE);
    printf("SD SPEED live\n");
    run(code * 200);
    printf("SD SPEED done\n");
    fflush(stdout);
    vTaskDelay(pdMS_TO_TICKS(500));
    esp_restart();                                    /* the card is idle now: the next clock starts from a clean boot */
}
"""


def write_project(project, port, clk, cmd, d0):
    src = os.path.join(project, "src")
    os.makedirs(src, exist_ok=True)
    w = lambda name, text: open(os.path.join(project, name), "w").write(text)
    w("platformio.ini", f"[env:s3]\nplatform = espressif32\nboard = esp32-s3-devkitc-1\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\n")
    for f in os.listdir(project):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(project, f))
    w("sdkconfig.defaults", "CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\nCONFIG_ESP_TASK_WDT_EN=n\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192\n")
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        "project(sd_speed_bench)\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.c" INCLUDE_DIRS "." REQUIRES sdmmc esp_driver_sdmmc '
                            'esp_driver_usb_serial_jtag esp_timer)\n')
    w("src/pins.h", f"#define PIN_CLK {clk}\n#define PIN_CMD {cmd}\n#define PIN_D0 {d0}\n")
    w("src/main.c", FIRMWARE)


def open_again(port):
    """The port disappears while the chip restarts: open it again until it is back."""
    import serial
    t0 = time.time()
    while time.time() - t0 < 20:
        try:
            return serial.Serial(port, 115200, timeout=1)
        except (serial.SerialException, OSError):
            time.sleep(0.5)
    sys.exit(f"[X] {port} did not come back")


def run_clock(port, khz):
    """One bus clock: send 0x55 + kHz/200 until the firmware answers, then read its rows. Returns (card, rows, ok)."""
    import serial
    s = open_again(port)
    live, last, t0, card, rows, failed = False, 0.0, time.time(), None, [], False
    while time.time() - t0 < 90:
        if not live and time.time() - last > 2:
            try:
                s.write(b"\x55" + bytes([khz // 200]))
                last = time.time()
            except (serial.SerialException, OSError):
                s.close()
                s = open_again(port)
        try:
            line = s.readline().decode(errors="replace").strip()
        except (serial.SerialException, OSError):
            s.close()
            s = open_again(port)
            continue
        if "SD SPEED live" in line:
            live = True
        elif not live or not line:
            continue
        elif line.startswith("I:"):
            name, sectors, real = line[2:].split("|")
            card = dict(cid_name=name, sectors=int(sectors), real_khz=int(real))
        elif line.startswith("R:"):
            f = line[2:].split("|")
            rows.append(dict(khz=int(f[0]), kind=f[1], bytes=int(f[2]), mbs=float(f[3]), bus_max=float(f[4]),
                             pct=float(f[5]), ms_per_read=float(f[6])))
        elif "SD SPEED done" in line:
            s.close()
            return card, rows, card is not None and not failed
        else:
            print(f"    {line}")                    # card init or read errors
            failed = True
    s.close()
    return card, rows, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--khz", default="400,5000,10000,20000,40000", help="bus clocks, comma-separated (multiples of 200)")
    ap.add_argument("--clk", type=int, default=39)
    ap.add_argument("--cmd", type=int, default=38)
    ap.add_argument("--d0", type=int, default=40)
    ap.add_argument("--card", default="", help="the card's label (brand, class, size), stored with the results")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    clocks = [int(k) for k in a.khz.split(",")]
    if any(k % 200 or not 200 <= k <= 51000 for k in clocks):
        ap.error("--khz: every clock must be a multiple of 200 between 200 and 51000")
    if not a.no_upload:
        project = os.path.join(HERE, "runs", "sd_speed_bench_s3")
        if os.path.isdir(os.path.join(project, "src")):
            shutil.rmtree(os.path.join(project, "src"))
        write_project(project, a.port, a.clk, a.cmd, a.d0)
        flash(project, verbose=False, what="SD speed firmware")
    card, results = None, []
    print(f"  {'clock':>8} {'read':>8} {'MB/s':>8} {'bus limit':>10} {'%':>5} {'ms/read':>9}")
    for khz in clocks:
        c, rows, ok = run_clock(a.port, khz)
        card = card or c
        for r in rows:
            print(f"  {r['khz']:>8} {r['kind'] + ' ' + str(r['bytes']):>8} {r['mbs']:>8.3f} {r['bus_max']:>10.3f} "
                  f"{r['pct']:>5.0f} {r['ms_per_read']:>9.2f}")
        results += rows
        if not ok:
            print(f"[X] stopped at {khz} kHz: unplug the board's USB, plug it back and rerun with --no-upload "
                  f"--khz starting at {khz}")
            break
        time.sleep(3)
    path = os.path.join(HERE, "results", "sd_speed_s3.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    measured = {r["khz"] for r in results}
    old = json.load(open(path)) if os.path.exists(path) else {}
    if old and old.get("bus") == f"SDMMC 1-bit, CLK {a.clk}, CMD {a.cmd}, D0 {a.d0}":
        results = [r for r in old.get("results", []) if r["khz"] not in measured] + results   # a rerun from a clock
        card, a.card = card or old.get("card_id"), a.card or old.get("card", "")
    results.sort(key=lambda r: (r["khz"], r["kind"], r["bytes"]))
    with open(path, "w") as f:
        json.dump(dict(board="esp32-s3", bus=f"SDMMC 1-bit, CLK {a.clk}, CMD {a.cmd}, D0 {a.d0}", card=a.card,
                       card_id=card, results=results), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
