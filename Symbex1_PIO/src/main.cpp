/*
 * main.cpp -- how to run a SYMBEX model on a board with PlatformIO: the "hello world" of lib_symbex.
 *
 * As it is, it runs the small DEMO network of model_symbex.h (8 x 8 RGB input, binary end to end, random weights:
 * not a trained model) on the test images of demo_input.h, and checks every result against the class the CDT's
 * integer simulation gave on the PC. "8/8 equal" means the library computes on this chip exactly what was computed on
 * the PC -- the guarantee every SYMBEX benchmark measures.
 *
 * To run a real model:
 *   1. export it with the CDT or a benchmark (for example benchmarks/conv2d/cnn_cifar_resnet8: resnet8_symbex.py); the run
 *      folder holds its model_symbex.h
 *   2. copy that model_symbex.h over src/model_symbex.h, and delete src/demo_input.h
 *   3. give the input to symbex_mlp_infer(): SYMBEX_MLP_N_INPUTS bytes (an RGB image in HWC order, uint8, for the
 *      convolutional models; the input vector for the dense ones) -- here, from your camera or sensor
 * Every model exported for lib_symbex offers this same interface: symbex_mlp_infer(input) -> class.
 *
 * The benchmarks measure accuracy, latency and memory on the board with their own firmware (they send the test images
 * over the serial port); this file is the minimal starting point for an application.
 */
#include <Arduino.h>
#include <stdint.h>

#include "model_symbex.h"       // the network: weights, thresholds, buffers, symbex_mlp_infer()
#include "hal/symbex_hal.h"     // only for the name of the bit counting method in use

#if __has_include("demo_input.h")
#include "demo_input.h"
#define SYMBEX_DEMO 1
#endif

static void run_demo(void) {
#ifdef SYMBEX_DEMO
    int equal = 0;
    uint32_t best_us = 0xFFFFFFFFUL;
    for (int i = 0; i < DEMO_N_IMAGES; i++) {
        const uint32_t t0 = micros();
        const int cls = symbex_mlp_infer(demo_images[i]);
        const uint32_t us = micros() - t0;
        if (us < best_us) best_us = us;
        equal += (cls == demo_expected[i]);
        Serial.print("image ");
        Serial.print(i);
        Serial.print(": class ");
        Serial.print(cls);
        Serial.print(" (PC simulation: ");
        Serial.print(demo_expected[i]);
        Serial.print(")  ");
        Serial.print(us);
        Serial.println(" us");
    }
    Serial.print(equal);
    Serial.print("/");
    Serial.print(DEMO_N_IMAGES);
    Serial.println(equal == DEMO_N_IMAGES ? " equal to the PC simulation: OK" : " equal to the PC simulation: [X] MISMATCH");
    Serial.print("fastest inference: ");
    Serial.print(best_us);
    Serial.println(" us");
#else
    Serial.println("your model is loaded: call symbex_mlp_infer(input) with SYMBEX_MLP_N_INPUTS bytes");
#endif
}

void setup() {
    Serial.begin(115200);
    delay(1500);
    Serial.println();
    Serial.println("SYMBEX-1 | lib_symbex example");
    Serial.print("bit counting: ");
    Serial.println(symbex_hal_name());
    Serial.print("model: ");
    Serial.print(SYMBEX_MLP_N_INPUTS);
    Serial.print(" inputs, ");
    Serial.print(SYMBEX_MLP_N_CLASSES);
    Serial.println(" classes");
    run_demo();
}

void loop() {
    delay(10000);
}
