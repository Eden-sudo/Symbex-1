"""
fashion_tflite.py -- mlp_fashion benchmark, official INT8 side (TensorFlow Lite), independent from SYMBEX.

    python fashion_tflite.py                    # convert to INT8 with TFLite and evaluate it on the PC
    python fashion_tflite.py --board esp32      # ... and measure it on the board with TensorFlow Lite Micro
    python fashion_tflite.py --table-only       # only the comparison table, from results/ (no board)
    python fashion_tflite.py --shape 128-512 --board s3 --port /dev/ttyACM0 --usb   # INT8 of the binary-sized MLP
--no-upload assumes the TFLite firmware is still on the board: after running the SYMBEX side (which flashes
its own firmware), measure without --no-upload, or just print the table with --table-only.

Run from benchmarks/mlp_fashion/ with the benchmark environment (env_bench: TensorFlow + PyTorch CPU,
see ../README.md). Nothing here uses the SYMBEX library or the CDT: the INT8 model is produced by the
official TFLite converter and runs on the chip with the official TFLite Micro (esp-tflite-micro, which
uses Espressif's ESP-NN kernels).

Steps:
  1. load the SAME original FP32 model as the SYMBEX side (models/fashion_mlp.pth) and rebuild it in Keras:
     BatchNorm and the input normalization folded into the Dense layers, input = pixels / 255
  2. full-integer INT8 conversion with tf.lite.TFLiteConverter (int8 input and output; 1000 training images
     as the representative dataset) -> runs/tflite/fashion_int8.tflite
  3. accuracy on the PC with the TFLite interpreter, on the SAME validation images the SYMBEX side uses
     (the first 2000 Fashion-MNIST test images, from model.validation_u8)
It does not need the SYMBEX side to have run: the FP32 model is trained here (models/) if it is missing.
  4. with --board: ESP-IDF project with TFLite Micro, flashed with PlatformIO; the images are streamed over
     serial and the chip reports the class and the latency it measured; the predictions must equal the
     PC interpreter's
Results: results/tflite_pc.json and results/tflite_<board>.json; with both sides done, a comparison table.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN, STD = 0.2860, 0.3530
BOARDS = {"esp32": "esp32dev", "s3": "esp32-s3-devkitc-1"}


# ---------------------------------------------------------------------------------------------------
# 1-2. Keras rebuild of the original model and official INT8 conversion
# ---------------------------------------------------------------------------------------------------
def folded_weights(pth_path):
    """state_dict of FashionMLP -> [(W (out x in), b, relu)] with BatchNorm and the input normalization folded,
    so the network takes pixels / 255 directly."""
    import torch
    sd = torch.load(pth_path, map_location="cpu")
    layers = []
    for lin, bn, relu in (("red.1", "red.2", True), ("red.4", "red.5", True), ("red.7", None, False)):
        W = sd[f"{lin}.weight"].double().numpy()
        b = sd[f"{lin}.bias"].double().numpy() if f"{lin}.bias" in sd else np.zeros(W.shape[0])
        if bn:
            g = sd[f"{bn}.weight"].double().numpy() / np.sqrt(sd[f"{bn}.running_var"].double().numpy() + 1e-5)
            b = g * (b - sd[f"{bn}.running_mean"].double().numpy()) + sd[f"{bn}.bias"].double().numpy()
            W = W * g[:, None]
        layers.append([W, b, relu])
    W0, b0, r0 = layers[0]                                # x = (p/255 - m) / s  ->  weights on p/255
    layers[0] = [W0 / STD, b0 - W0.sum(1) * MEAN / STD, r0]
    return layers


def keras_model(layers):
    import tensorflow as tf
    inp_layer = tf.keras.Input(shape=(layers[0][0].shape[1],), name="pixels_div_255")
    x = inp_layer
    for i, (W, b, relu) in enumerate(layers):
        x = tf.keras.layers.Dense(W.shape[0], activation="relu" if relu else None, name=f"dense{i}")(x)
    m = tf.keras.Model(inp_layer, x)
    for i, (W, b, _) in enumerate(layers):
        m.get_layer(f"dense{i}").set_weights([W.T.astype(np.float32), b.astype(np.float32)])
    return m


def convert_int8(m, calib):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative():
        for i in range(len(calib)):
            yield [calib[i:i + 1].astype(np.float32)]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def evaluate_pc(tflite, x_u8):
    """Classes predicted by the TFLite interpreter (int8 in / int8 out), input quantized from pixels / 255."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=tflite)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zero_point = inp["quantization"]
    table = np.clip(np.round(np.arange(256) / 255.0 / scale) + zero_point, -128, 127).astype(np.int8)
    pred = []
    for x in x_u8:
        it.set_tensor(inp["index"], table[x][None, :])
        it.invoke()
        pred.append(int(np.argmax(it.get_tensor(out["index"])[0])))
    return np.array(pred), (float(scale), int(zero_point))


# ---------------------------------------------------------------------------------------------------
# 4. TFLite Micro on the chip
# ---------------------------------------------------------------------------------------------------
MAIN_TFLM = r"""
#include <cstdio>
#include <cstdint>
#include <cmath>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <cstring>
#ifdef USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#endif
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_tflite.h"

static uint8_t arena[ARENA_KB * 1024] __attribute__((aligned(16)));
static uint8_t x[N_INPUTS];
static int8_t q8[256];

static void read_exact(uint8_t* d, size_t n) {
    size_t r = 0;
#ifdef USB_SERIAL_JTAG
    while (r < n) { int k = usb_serial_jtag_read_bytes(d + r, n - r, portMAX_DELAY); if (k > 0) r += (size_t)k; }
#else
    while (r < n) { int k = uart_read_bytes(UART_NUM_0, d + r, n - r, portMAX_DELAY); if (k > 0) r += (size_t)k; }
#endif
}

extern "C" void app_main(void) {
    esp_log_level_set("*", ESP_LOG_WARN);
#ifdef USB_SERIAL_JTAG
    /* the S3's native USB port (--usb): the same protocol, the host sends 0x55 when it listens */
    usb_serial_jtag_driver_config_t usb = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb.rx_buffer_size = 8192;
    usb.tx_buffer_size = 4096;
    usb_serial_jtag_driver_install(&usb);
    usb_serial_jtag_vfs_use_driver();
#else
    uart_driver_install(UART_NUM_0, 8192, 0, 0, nullptr, 0);
    uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(100));
    uart_set_baudrate(UART_NUM_0, BAUD_RATE);
#endif
    vTaskDelay(pdMS_TO_TICKS(50));
#ifdef MODEL_IN_RAM
    /* copy the whole .tflite (weights included) to internal RAM: no flash cache misses */
    uint8_t* copy_buf = (uint8_t*)heap_caps_aligned_alloc(16, g_model_len, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!copy_buf) { printf("\nERROR:no RAM for the model copy (%u bytes)\n", (unsigned)g_model_len); fflush(stdout); while (true) vTaskDelay(portMAX_DELAY); }
    memcpy(copy_buf, g_model, g_model_len);
    const tflite::Model* model = tflite::GetModel(copy_buf);
#else
    const tflite::Model* model = tflite::GetModel(g_model);
#endif
    static tflite::MicroMutableOpResolver<6> ops;
    ops.AddFullyConnected(); ops.AddReshape(); ops.AddQuantize(); ops.AddDequantize(); ops.AddSoftmax(); ops.AddRelu();
    static tflite::MicroInterpreter it(model, ops, arena, sizeof arena);
    if (it.AllocateTensors() != kTfLiteOk) { printf("\nERROR:AllocateTensors failed\n"); fflush(stdout); while (true) vTaskDelay(portMAX_DELAY); }
    TfLiteTensor* inp = it.input(0);
    TfLiteTensor* out = it.output(0);
    for (int p = 0; p < 256; p++) {
        float q = roundf((p / 255.0f) / inp->params.scale) + inp->params.zero_point;
        q8[p] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
#ifdef USB_SERIAL_JTAG
    uint8_t hello = 0;
    while (hello != 0x55) read_exact(&hello, 1);
#endif
    printf("\nTFLITE ready | arena used %u bytes\n", (unsigned)it.arena_used_bytes());
    fflush(stdout);
    while (true) {
        uint8_t mark = 0;
        read_exact(&mark, 1);
        if (mark == 0x55) {                           /* the host asks for the ready line again */
            printf("\nTFLITE ready | arena used %u bytes\n", (unsigned)it.arena_used_bytes());
            fflush(stdout);
            continue;
        }
        if (mark != 0xA5) continue;
        read_exact(x, sizeof x);
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < N_INPUTS; i++) inp->data.int8[i] = q8[x[i]];
        it.Invoke();
        int c = 0;
        for (int j = 1; j < N_CLASSES; j++) if (out->data.int8[j] > out->data.int8[c]) c = j;
        int64_t t1 = esp_timer_get_time();
        printf("T:%d,%lld\n", c, (long long)(t1 - t0));
        fflush(stdout);
    }
}
"""


def tflm_project(project_dir, tflite, board, port, baud, arena_kb, n_inputs, n_classes, model_ram=False, usb=False):
    src = os.path.join(project_dir, "src")
    os.makedirs(src, exist_ok=True)
    for f in os.listdir(project_dir):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(project_dir, f))
    w = lambda r, t: open(os.path.join(project_dir, r), "w").write(t)
    w("platformio.ini", f"[env:{BOARDS[board]}]\nplatform = espressif32\nboard = {BOARDS[board]}\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {port}\n")
    w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                            "CONFIG_ESP32_DEFAULT_CPU_FREQ_240=y\nCONFIG_ESP32S3_DEFAULT_CPU_FREQ_240=y\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_LOG_DEFAULT_LEVEL_WARN=y\n"
                            "CONFIG_ESP_TASK_WDT_EN=n\nCONFIG_NN_OPTIMIZED=y\n"
                            + ("CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\n" if usb else ""))
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        f"project({os.path.basename(os.path.abspath(project_dir))})\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.cc" "model_tflite.cc" INCLUDE_DIRS ".")\n')
    w("src/idf_component.yml", 'dependencies:\n  espressif/esp-tflite-micro: "*"\n')
    w("src/model_tflite.h", "#pragma once\n#include <cstddef>\nextern const unsigned char g_model[];\n"
                             "extern const size_t g_model_len;\n")
    lines = ['#include "model_tflite.h"', "alignas(16) const unsigned char g_model[] = {"]
    lines += ["  " + ", ".join(f"0x{b:02x}" for b in tflite[i:i + 16]) + "," for i in range(0, len(tflite), 16)]
    lines += ["};", f"const size_t g_model_len = {len(tflite)};"]
    w("src/model_tflite.cc", "\n".join(lines) + "\n")
    w("src/main.cc", f"#define BAUD_RATE {baud}\n#define ARENA_KB {arena_kb}\n#define N_INPUTS {n_inputs}\n"
                     f"#define N_CLASSES {n_classes}\n" + ("#define MODEL_IN_RAM 1\n" if model_ram else "")
                     + ("#define USB_SERIAL_JTAG 1\n" if usb else "") + MAIN_TFLM)


def measure_chip(project_dir, board, port, baud, x_u8, upload, usb=False):
    import serial
    if upload:
        if shutil.which("pio") is None:
            sys.exit("[X] PlatformIO (pio) not found in this environment: pip install platformio")
        print(f"[*] building and flashing {BOARDS[board]} (TFLite Micro; the first build downloads it)...")
        if subprocess.run(["pio", "run", "-d", project_dir, "--target", "upload"]).returncode != 0:
            sys.exit("[X] build or upload failed")
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, 1
    s.dtr = False
    s.rts = False
    s.open()
    s.reset_input_buffer()
    if usb:                                              # native USB: opening the port resets the chip
        time.sleep(1.5)
        s.reset_input_buffer()
        s.write(b"\x55")
    else:
        s.rts = True
        time.sleep(0.15)
        s.rts = False
    arena, t0, warned = None, time.time(), False
    while True:
        l = s.readline().decode(errors="replace").strip()
        if usb and not l:
            s.write(b"\x55")                             # ask again for the ready line
        if "ERROR:" in l:
            sys.exit(f"[X] {l[l.index('ERROR:'):]} (raise --arena-kb)")
        if "TFLITE ready" in l:
            arena = int(l.split("arena used")[1].split()[0])
            break
        if not warned and time.time() - t0 > 8:
            print("    >>> the board did not answer the reset: press EN/RST once")
            warned = True
        if time.time() - t0 > 60:
            sys.exit("[X] the board did not start")
    pred, times = [], []
    for i in range(len(x_u8)):
        s.write(bytes([0xA5]) + np.ascontiguousarray(x_u8[i], dtype=np.uint8).tobytes())
        deadline = time.time() + 10
        while True:
            l = s.readline().decode(errors="replace").strip()
            if l.startswith("T:"):
                c, us = l[2:].split(",")
                pred.append(int(c))
                times.append(int(us))
                break
            if time.time() > deadline:
                sys.exit(f"[X] no answer for image {i}")
        if (i + 1) % max(1, len(x_u8) // 5) == 0:
            print(f"      {i + 1}/{len(x_u8)} | median {np.median(times) / 1000:.3f} ms")
    s.close()
    return np.array(pred), np.array(times), arena


def final_table(r_tfl, board):
    """Comparison with the SYMBEX side on the same board, in TWO tables, one per weight placement:
      flash  the usual deployment: the weights stay in flash (TFLite Micro's model array, SYMBEX's const header)
      RAM    best case of each one when it fits in RAM (TFLite: the whole .tflite copied at boot)
    Each SYMBEX row is compared against TFLite with the SAME placement."""
    import glob
    tflite = {}
    for suffix, place in (("", "flash"), ("_ram", "ram")):
        path = os.path.join("results", f"tflite_{board}{suffix}.json")
        if os.path.exists(path):
            tflite[place] = json.load(open(path))
    symbex = {"flash": [], "ram": []}
    for f in sorted(glob.glob(os.path.join("results", f"symbex_s*_chip_{board}_*.json"))):
        d = json.load(open(f))
        if d.get("weights") in symbex and "accuracy" in d:
            seed = os.path.basename(f).split("_")[1]
            symbex[d["weights"]].append((f"SYMBEX {seed} ({str(d.get('arithmetic', '?')).upper()}, {d.get('popcount', '?')})",
                                         d["accuracy"], d["median_ms"], (d.get("model_flash") or 0) / 1024,
                                         d["ram"]["total"] / 1024))
    titles = {"flash": "WEIGHTS IN FLASH (the usual deployment)", "ram": "WEIGHTS IN RAM (best case, when the model fits)"}
    for place in ("flash", "ram"):
        if place not in tflite and not symbex[place]:
            continue
        print(f"\n    {titles[place]}  --  {board}, 240 MHz, one core")
        print(f"    {'model':<44}{'accuracy':>9}{'latency ms':>12}{'vs TFLite':>11}{'flash KB':>10}{'RAM KB':>9}")
        print("    " + "-" * 95)
        ref = tflite.get(place)
        rows = []
        if ref:
            ram = ref["arena_kb"] + (ref["flash_kb"] if place == "ram" else 0)
            rows.append(("INT8, official TFLite Micro", ref["accuracy"], ref["median_ms"], ref["flash_kb"], ram))
        rows += symbex[place]
        for name, acc, ms, fl, ram in rows:
            vs = f"{ref['median_ms'] / ms:>10.2f}x" if ref else f"{'-':>11}"
            print(f"    {name:<44}{acc:>8.2f}%{ms:>12.3f}{vs}{fl:>10.1f}{ram:>9.2f}")
        if not ref:
            print(f"    (no TFLite result with the model in {place} on {board} yet)")
        if not symbex[place]:
            print(f"    (no SYMBEX result with the weights in {place} on {board} yet: fashion_symbex.py --weights {place})")
    print("\n    vs TFLite > 1: faster than the official TFLite Micro INT8 with the same weight placement.")
    print("    RAM: SYMBEX = buffers + input + popcount table + stack (+ weights when in RAM); TFLite = tensor arena"
          " (+ the .tflite copy when in RAM).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=sorted(BOARDS), default="")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--arena-kb", type=int, default=16)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--validation", type=int, default=2000, help="validation images (the first N of the test set)")
    ap.add_argument("--model-ram", action="store_true",
                    help="copy the .tflite to internal RAM at boot (fair against SYMBEX --weights ram; the ESP32-S3 "
                         "has room for it, the classic ESP32 usually does not)")
    ap.add_argument("--table-only", action="store_true",
                    help="only print the comparison table from results/ (no conversion, no board needed)")
    ap.add_argument("--shape", default="256-128",
                    help="hidden widths h1-h2 of the FP32 model (models/fashion_mlp_<h1>_<h2>.pth, as in "
                         "fashion_sizing.py): INT8 of a network sized for binary; other than 256-128, results get "
                         "_h<h1>_<h2>")
    ap.add_argument("--usb", action="store_true", help="the ESP32-S3's native USB port (/dev/ttyACM0), not its UART")
    args = ap.parse_args()
    if args.port and not args.port.startswith("/") and not args.port.upper().startswith("COM"):
        args.port = "/dev/" + args.port                  # "ttyUSB0" -> "/dev/ttyUSB0"
    if args.usb and args.board != "s3":
        sys.exit("[X] --usb: only the ESP32-S3 (--board s3) has a native USB-Serial-JTAG port")
    h1, h2 = (int(v) for v in args.shape.split("-"))
    tag = "" if (h1, h2) == (256, 128) else f"_h{h1}_{h2}"
    os.chdir(HERE)
    if args.table_only:
        board = args.board or "esp32"
        path = os.path.join("results", f"tflite_{board}{tag}{'_ram' if args.model_ram else ''}.json")
        if not os.path.exists(path):
            sys.exit(f"[X] {path} not found: measure the TFLite model on the board first (--board {board})")
        final_table(json.load(open(path)), board)
        return
    sys.path.insert(0, HERE)
    from model import get_fp32, validation_u8, calibration_u8
    fp32_path = get_fp32(h1, h2)                     # trained here if missing (models/)
    x_val, y_val = validation_u8(args.validation)        # the same images the SYMBEX side validates

    print("[*] 1-2. Keras rebuild of the original FP32 model and official INT8 conversion (TFLiteConverter)")
    calib = calibration_u8(1000).astype(np.float32) / 255.0
    layers = folded_weights(fp32_path)
    m = keras_model(layers)
    os.makedirs("runs/tflite", exist_ok=True)
    tflite = convert_int8(m, calib)
    open(f"runs/tflite/fashion_int8{tag}.tflite", "wb").write(tflite)
    pred_k = np.argmax(m.predict(x_val.astype(np.float32) / 255.0, verbose=0), axis=1)
    print(f"    Keras FP32 rebuild: {100 * (pred_k == y_val).mean():.2f}% on {len(y_val)} validation images "
          f"(must match the original FP32 model)")

    print("[*] 3. INT8 on the PC (TFLite interpreter)")
    pred_pc, (in_scale, zero_point) = evaluate_pc(tflite, x_val)
    r_pc = dict(accuracy=100 * float((pred_pc == y_val).mean()), acuerdo_fp32=100 * float((pred_pc == pred_k).mean()),
                tflite_bytes=len(tflite), input_scale=in_scale, input_zero_point=zero_point, n=len(y_val),
                shape=f"784-{h1}-{h2}-10")
    print(f"    accuracy {r_pc['accuracy']:.2f}% | same class as FP32 {r_pc['acuerdo_fp32']:.2f}% | "
          f".tflite {len(tflite) / 1024:.1f} KB | input scale {in_scale:.6f}, zero point {zero_point}")
    os.makedirs("results", exist_ok=True)
    json.dump(r_pc, open(f"results/tflite_pc{tag}.json", "w"), indent=1)

    if args.board:
        print(f"[*] 4. INT8 on the chip ({args.board}, TensorFlow Lite Micro)")
        n = min(args.samples, len(x_val))
        project_dir = f"runs/tflite/chip_{args.board}{tag}"
        tflm_project(project_dir, tflite, args.board, args.port, args.baud, args.arena_kb, 784, 10, args.model_ram,
                     args.usb)
        pred, times, arena = measure_chip(project_dir, args.board, args.port, args.baud, x_val[:n], not args.no_upload,
                                          args.usb)
        r = dict(board=args.board, model_ram=args.model_ram, n=n, shape=f"784-{h1}-{h2}-10",
                 accuracy=100 * float((pred == y_val[:n]).mean()),
                 equal_pc=100 * float((pred == pred_pc[:n]).mean()), median_ms=float(np.median(times)) / 1000,
                 min_ms=float(np.min(times)) / 1000, arena_kb=arena / 1024, flash_kb=len(tflite) / 1024)
        status = "OK" if r["equal_pc"] == 100 else "[!] differs from the PC interpreter"
        print(f"\n    ON THE CHIP: equal to the PC interpreter {r['equal_pc']:.2f}% {status} | accuracy {r['accuracy']:.2f}%"
              f" | latency median {r['median_ms']:.3f} ms | arena {r['arena_kb']:.1f} KB | model {r['flash_kb']:.1f} KB")
        suffix = tag + ("_ram" if args.model_ram else "")
        json.dump(r, open(f"results/tflite_{args.board}{suffix}.json", "w"), indent=1)
        if not tag:                                      # the table compares with the SYMBEX side of 256-128
            final_table(r, args.board)
        else:
            print(f"    -> results/tflite_{args.board}{suffix}.json")


if __name__ == "__main__":
    main()
