"""
fashion_tflite.py -- mlp_fashion benchmark, official INT8 side (TensorFlow Lite), independent from SYMBEX.

    python fashion_tflite.py                    # convert to INT8 with TFLite and evaluate it on the PC
    python fashion_tflite.py --board esp32      # ... and measure it on the board with TensorFlow Lite Micro
    python fashion_tflite.py --solo-tabla       # only the comparison table, from results/ (no board)
--no-upload assumes the TFLite firmware is still on the board: after running the SYMBEX side (which flashes
its own firmware), measure without --no-upload, or just print the table with --solo-tabla.

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
     (the first 2000 Fashion-MNIST test images, from modelo.validacion_u8)
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

AQUI = os.path.dirname(os.path.abspath(__file__))
MEDIA, DESV = 0.2860, 0.3530
PLACAS = {"esp32": "esp32dev", "s3": "esp32-s3-devkitc-1"}


# ---------------------------------------------------------------------------------------------------
# 1-2. Keras rebuild of the original model and official INT8 conversion
# ---------------------------------------------------------------------------------------------------
def pesos_plegados(ruta_pth):
    """state_dict of FashionMLP -> [(W (out x in), b, relu)] with BatchNorm and the input normalization folded,
    so the network takes pixels / 255 directly."""
    import torch
    sd = torch.load(ruta_pth, map_location="cpu")
    capas = []
    for lin, bn, relu in (("red.1", "red.2", True), ("red.4", "red.5", True), ("red.7", None, False)):
        W = sd[f"{lin}.weight"].double().numpy()
        b = sd[f"{lin}.bias"].double().numpy() if f"{lin}.bias" in sd else np.zeros(W.shape[0])
        if bn:
            g = sd[f"{bn}.weight"].double().numpy() / np.sqrt(sd[f"{bn}.running_var"].double().numpy() + 1e-5)
            b = g * (b - sd[f"{bn}.running_mean"].double().numpy()) + sd[f"{bn}.bias"].double().numpy()
            W = W * g[:, None]
        capas.append([W, b, relu])
    W0, b0, r0 = capas[0]                                # x = (p/255 - m) / s  ->  weights on p/255
    capas[0] = [W0 / DESV, b0 - W0.sum(1) * MEDIA / DESV, r0]
    return capas


def modelo_keras(capas):
    import tensorflow as tf
    entrada = tf.keras.Input(shape=(capas[0][0].shape[1],), name="pixels_div_255")
    x = entrada
    for i, (W, b, relu) in enumerate(capas):
        x = tf.keras.layers.Dense(W.shape[0], activation="relu" if relu else None, name=f"dense{i}")(x)
    m = tf.keras.Model(entrada, x)
    for i, (W, b, _) in enumerate(capas):
        m.get_layer(f"dense{i}").set_weights([W.T.astype(np.float32), b.astype(np.float32)])
    return m


def convertir_int8(m, calib):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def representativo():
        for i in range(len(calib)):
            yield [calib[i:i + 1].astype(np.float32)]
    conv.representative_dataset = representativo
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def evaluar_pc(tflite, x_u8):
    """Classes predicted by the TFLite interpreter (int8 in / int8 out), input quantized from pixels / 255."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=tflite)
    it.allocate_tensors()
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    escala, cero = ent["quantization"]
    tabla = np.clip(np.round(np.arange(256) / 255.0 / escala) + cero, -128, 127).astype(np.int8)
    pred = []
    for x in x_u8:
        it.set_tensor(ent["index"], tabla[x][None, :])
        it.invoke()
        pred.append(int(np.argmax(it.get_tensor(sal["index"])[0])))
    return np.array(pred), (float(escala), int(cero))


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
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "modelo_tflite.h"

static uint8_t arena[ARENA_KB * 1024] __attribute__((aligned(16)));
static uint8_t x[N_ENTRADAS];
static int8_t q8[256];

static void leer_exacto(uint8_t* d, size_t n) {
    size_t r = 0;
    while (r < n) { int k = uart_read_bytes(UART_NUM_0, d + r, n - r, portMAX_DELAY); if (k > 0) r += (size_t)k; }
}

extern "C" void app_main(void) {
    esp_log_level_set("*", ESP_LOG_WARN);
    uart_driver_install(UART_NUM_0, 8192, 0, 0, nullptr, 0);
    uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(100));
    uart_set_baudrate(UART_NUM_0, BAUDIOS);
    vTaskDelay(pdMS_TO_TICKS(50));
#ifdef MODELO_EN_RAM
    /* copy the whole .tflite (weights included) to internal RAM: no flash cache misses */
    uint8_t* copia = (uint8_t*)heap_caps_aligned_alloc(16, g_modelo_len, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!copia) { printf("\nERROR:no RAM for the model copy (%u bytes)\n", (unsigned)g_modelo_len); fflush(stdout); while (true) vTaskDelay(portMAX_DELAY); }
    memcpy(copia, g_modelo, g_modelo_len);
    const tflite::Model* modelo = tflite::GetModel(copia);
#else
    const tflite::Model* modelo = tflite::GetModel(g_modelo);
#endif
    static tflite::MicroMutableOpResolver<6> ops;
    ops.AddFullyConnected(); ops.AddReshape(); ops.AddQuantize(); ops.AddDequantize(); ops.AddSoftmax(); ops.AddRelu();
    static tflite::MicroInterpreter it(modelo, ops, arena, sizeof arena);
    if (it.AllocateTensors() != kTfLiteOk) { printf("\nERROR:AllocateTensors failed\n"); fflush(stdout); while (true) vTaskDelay(portMAX_DELAY); }
    TfLiteTensor* ent = it.input(0);
    TfLiteTensor* sal = it.output(0);
    for (int p = 0; p < 256; p++) {
        float q = roundf((p / 255.0f) / ent->params.scale) + ent->params.zero_point;
        q8[p] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    printf("\nTFLITE listo | arena used %u bytes\n", (unsigned)it.arena_used_bytes());
    fflush(stdout);
    while (true) {
        uint8_t marca = 0;
        leer_exacto(&marca, 1);
        if (marca != 0xA5) continue;
        leer_exacto(x, sizeof x);
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < N_ENTRADAS; i++) ent->data.int8[i] = q8[x[i]];
        it.Invoke();
        int c = 0;
        for (int j = 1; j < N_CLASES; j++) if (sal->data.int8[j] > sal->data.int8[c]) c = j;
        int64_t t1 = esp_timer_get_time();
        printf("T:%d,%lld\n", c, (long long)(t1 - t0));
        fflush(stdout);
    }
}
"""


def proyecto_tflm(dir_p, tflite, placa, puerto, baudios, arena_kb, n_entradas, n_clases, modelo_ram=False):
    src = os.path.join(dir_p, "src")
    os.makedirs(src, exist_ok=True)
    for f in os.listdir(dir_p):
        if f.startswith("sdkconfig") and f != "sdkconfig.defaults":
            os.remove(os.path.join(dir_p, f))
    w = lambda r, t: open(os.path.join(dir_p, r), "w").write(t)
    w("platformio.ini", f"[env:{PLACAS[placa]}]\nplatform = espressif32\nboard = {PLACAS[placa]}\nframework = espidf\n"
                        f"monitor_speed = 115200\nupload_port = {puerto}\n")
    w("sdkconfig.defaults", "CONFIG_COMPILER_OPTIMIZATION_PERF=y\nCONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y\n"
                            "CONFIG_ESP32_DEFAULT_CPU_FREQ_240=y\nCONFIG_ESP32S3_DEFAULT_CPU_FREQ_240=y\n"
                            "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\nCONFIG_LOG_DEFAULT_LEVEL_WARN=y\n"
                            "CONFIG_ESP_TASK_WDT_EN=n\nCONFIG_NN_OPTIMIZED=y\n")
    w("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16.0)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
                        f"project({os.path.basename(os.path.abspath(dir_p))})\n")
    w("src/CMakeLists.txt", 'idf_component_register(SRCS "main.cc" "modelo_tflite.cc" INCLUDE_DIRS ".")\n')
    w("src/idf_component.yml", 'dependencies:\n  espressif/esp-tflite-micro: "*"\n')
    w("src/modelo_tflite.h", "#pragma once\n#include <cstddef>\nextern const unsigned char g_modelo[];\n"
                             "extern const size_t g_modelo_len;\n")
    lineas = ['#include "modelo_tflite.h"', "alignas(16) const unsigned char g_modelo[] = {"]
    lineas += ["  " + ", ".join(f"0x{b:02x}" for b in tflite[i:i + 16]) + "," for i in range(0, len(tflite), 16)]
    lineas += ["};", f"const size_t g_modelo_len = {len(tflite)};"]
    w("src/modelo_tflite.cc", "\n".join(lineas) + "\n")
    w("src/main.cc", f"#define BAUDIOS {baudios}\n#define ARENA_KB {arena_kb}\n#define N_ENTRADAS {n_entradas}\n"
                     f"#define N_CLASES {n_clases}\n" + ("#define MODELO_EN_RAM 1\n" if modelo_ram else "") + MAIN_TFLM)


def medir_chip(dir_p, placa, puerto, baudios, x_u8, subir):
    import serial
    if subir:
        if shutil.which("pio") is None:
            sys.exit("[X] PlatformIO (pio) not found in this environment: pip install platformio")
        print(f"[*] building and flashing {PLACAS[placa]} (TFLite Micro; the first build downloads it)...")
        if subprocess.run(["pio", "run", "-d", dir_p, "--target", "upload"]).returncode != 0:
            sys.exit("[X] build or upload failed")
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = puerto, baudios, 1
    s.dtr = False
    s.rts = False
    s.open()
    s.reset_input_buffer()
    s.rts = True
    time.sleep(0.15)
    s.rts = False
    arena, t0, avisado = None, time.time(), False
    while True:
        l = s.readline().decode(errors="replace").strip()
        if "ERROR:" in l:
            sys.exit(f"[X] {l[l.index('ERROR:'):]} (raise --arena-kb)")
        if "TFLITE listo" in l:
            arena = int(l.split("arena used")[1].split()[0])
            break
        if not avisado and time.time() - t0 > 8:
            print("    >>> the board did not answer the reset: press EN/RST once")
            avisado = True
        if time.time() - t0 > 60:
            sys.exit("[X] the board did not start")
    pred, tiempos = [], []
    for i in range(len(x_u8)):
        s.write(bytes([0xA5]) + np.ascontiguousarray(x_u8[i], dtype=np.uint8).tobytes())
        limite = time.time() + 10
        while True:
            l = s.readline().decode(errors="replace").strip()
            if l.startswith("T:"):
                c, us = l[2:].split(",")
                pred.append(int(c))
                tiempos.append(int(us))
                break
            if time.time() > limite:
                sys.exit(f"[X] no answer for image {i}")
        if (i + 1) % max(1, len(x_u8) // 5) == 0:
            print(f"      {i + 1}/{len(x_u8)} | median {np.median(tiempos) / 1000:.3f} ms")
    s.close()
    return np.array(pred), np.array(tiempos), arena


def tabla_final(r_tfl, placa):
    """Comparison with the SYMBEX side, if fashion_symbex.py already measured the same board."""
    import glob
    filas = [("INT8, official TFLite Micro" + (" (model in RAM)" if r_tfl.get("modelo_ram") else ""),
              r_tfl["precision"], r_tfl["mediana_ms"], r_tfl["flash_kb"], r_tfl["arena_kb"])]
    otra = os.path.join("results", f"tflite_{placa}{'' if r_tfl.get('modelo_ram') else '_ram'}.json")
    if os.path.exists(otra):
        o = json.load(open(otra))
        filas.append(("INT8, official TFLite Micro" + (" (model in RAM)" if o.get("modelo_ram") else ""),
                      o["precision"], o["mediana_ms"], o["flash_kb"], o["arena_kb"] + (o["flash_kb"] if o.get("modelo_ram") else 0)))
    for f in sorted(glob.glob(os.path.join("results", f"symbex_*chip_{placa}_*.json"))):
        d = json.load(open(f))
        semilla = os.path.basename(f).split("_")[1] if os.path.basename(f).startswith("symbex_s") else ""
        etiqueta = (f"SYMBEX {semilla} " if semilla else "SYMBEX ") + \
                   f"({d.get('pesos', '?')}, {d.get('popcount', '?')}, {d.get('primera', '?')})"
        filas.append((etiqueta, d["precision"], d["mediana_ms"], (d.get("flash_modelo") or 0) / 1024, d["ram"]["total"] / 1024))
        if "int8" in d and not any(r[0].startswith("INT8, CDT") for r in filas):
            filas.insert(1, ("INT8, CDT reference (ESP-NN)", d["int8"]["precision"], d["int8"]["mediana_ms"],
                             (d["int8"].get("flash") or 0) / 1024, None))
    print(f"\n    {'model (' + placa + ')':<40}{'accuracy':>9}{'latency ms':>12}{'vs TFLite':>11}{'flash KB':>10}{'RAM KB':>9}")
    print("    " + "-" * 91)
    for nombre, acc, ms, fl, ram in filas:
        print(f"    {nombre:<40}{acc:>8.2f}%{ms:>12.3f}{r_tfl['mediana_ms'] / ms:>10.2f}x{fl:>10.1f}"
              + (f"{ram:>9.2f}" if ram is not None else f"{'-':>9}"))
    print("    (vs TFLite > 1: faster than the official TFLite Micro INT8. RAM for TFLite = tensor arena used.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=sorted(PLACAS), default="")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--arena-kb", type=int, default=16)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--validacion", type=int, default=2000, help="validation images (the first N of the test set)")
    ap.add_argument("--modelo-ram", action="store_true",
                    help="copy the .tflite to internal RAM at boot (fair against SYMBEX --weights ram; the ESP32-S3 "
                         "has room for it, the classic ESP32 usually does not)")
    ap.add_argument("--solo-tabla", action="store_true",
                    help="only print the comparison table from results/ (no conversion, no board needed)")
    args = ap.parse_args()
    if args.port and not args.port.startswith("/") and not args.port.upper().startswith("COM"):
        args.port = "/dev/" + args.port                  # "ttyUSB0" -> "/dev/ttyUSB0"
    os.chdir(AQUI)
    if args.solo_tabla:
        placa = args.board or "esp32"
        ruta = os.path.join("results", f"tflite_{placa}{'_ram' if args.modelo_ram else ''}.json")
        if not os.path.exists(ruta):
            sys.exit(f"[X] {ruta} not found: measure the TFLite model on the board first (--board {placa})")
        tabla_final(json.load(open(ruta)), placa)
        return
    sys.path.insert(0, AQUI)
    from modelo import obtener_fp32, validacion_u8, calibracion_u8
    ruta_fp32 = obtener_fp32()                           # trained here if missing (models/)
    x_val, y_val = validacion_u8(args.validacion)        # the same images the SYMBEX side validates

    print("[*] 1-2. Keras rebuild of the original FP32 model and official INT8 conversion (TFLiteConverter)")
    calib = calibracion_u8(1000).astype(np.float32) / 255.0
    capas = pesos_plegados(ruta_fp32)
    m = modelo_keras(capas)
    os.makedirs("runs/tflite", exist_ok=True)
    tflite = convertir_int8(m, calib)
    open("runs/tflite/fashion_int8.tflite", "wb").write(tflite)
    pred_k = np.argmax(m.predict(x_val.astype(np.float32) / 255.0, verbose=0), axis=1)
    print(f"    Keras FP32 rebuild: {100 * (pred_k == y_val).mean():.2f}% on {len(y_val)} validation images "
          f"(must match the original FP32 model)")

    print("[*] 3. INT8 on the PC (TFLite interpreter)")
    pred_pc, (esc, cero) = evaluar_pc(tflite, x_val)
    r_pc = dict(precision=100 * float((pred_pc == y_val).mean()), acuerdo_fp32=100 * float((pred_pc == pred_k).mean()),
                tflite_bytes=len(tflite), input_scale=esc, input_zero_point=cero, n=len(y_val))
    print(f"    accuracy {r_pc['precision']:.2f}% | same class as FP32 {r_pc['acuerdo_fp32']:.2f}% | "
          f".tflite {len(tflite) / 1024:.1f} KB | input scale {esc:.6f}, zero point {cero}")
    os.makedirs("results", exist_ok=True)
    json.dump(r_pc, open("results/tflite_pc.json", "w"), indent=1)

    if args.board:
        print(f"[*] 4. INT8 on the chip ({args.board}, TensorFlow Lite Micro)")
        n = min(args.samples, len(x_val))
        dir_p = f"runs/tflite/chip_{args.board}"
        proyecto_tflm(dir_p, tflite, args.board, args.port, args.baud, args.arena_kb, 784, 10, args.modelo_ram)
        pred, tiempos, arena = medir_chip(dir_p, args.board, args.port, args.baud, x_val[:n], not args.no_upload)
        r = dict(placa=args.board, modelo_ram=args.modelo_ram, n=n, precision=100 * float((pred == y_val[:n]).mean()),
                 igual_pc=100 * float((pred == pred_pc[:n]).mean()), mediana_ms=float(np.median(tiempos)) / 1000,
                 min_ms=float(np.min(tiempos)) / 1000, arena_kb=arena / 1024, flash_kb=len(tflite) / 1024)
        estado = "OK" if r["igual_pc"] == 100 else "[!] differs from the PC interpreter"
        print(f"\n    ON THE CHIP: equal to the PC interpreter {r['igual_pc']:.2f}% {estado} | accuracy {r['precision']:.2f}%"
              f" | latency median {r['mediana_ms']:.3f} ms | arena {r['arena_kb']:.1f} KB | model {r['flash_kb']:.1f} KB")
        sufijo = "_ram" if args.modelo_ram else ""
        json.dump(r, open(f"results/tflite_{args.board}{sufijo}.json", "w"), indent=1)
        tabla_final(r, args.board)


if __name__ == "__main__":
    main()
