// SYMBEX-1 | lib_symbex example for the ESP32-S3.
// Runs symbex_mlp_infer on every demo image in setup() and prints over Serial
// (115200) how many results equal the PC simulation and the microseconds per
// inference. Both figures are measured on the board; none is written here.
// The demo model has random weights: it is not a trained model.
// model_symbex.h and demo_input.h are byte-identical copies of
// Symbex1_PIO/src/: pio ci and the Arduino IDE build the example folder alone,
// so they cannot reach src/ by include path.
// This is an example that only has to compile; no board is needed to build it.
#include <Arduino.h>
#include <stdint.h>
#include <symbex.h>
#include "model_symbex.h"
#include "demo_input.h"

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

    int equal = 0;
    uint32_t total_us = 0;
    for (int i = 0; i < DEMO_N_IMAGES; i++) {
        const uint32_t t0 = micros();
        const int cls = symbex_mlp_infer(demo_images[i]);
        const uint32_t us = micros() - t0;
        total_us += us;
        equal += (cls == demo_expected[i]);
    }
    Serial.print(equal);
    Serial.print("/");
    Serial.print(DEMO_N_IMAGES);
    Serial.println(equal == DEMO_N_IMAGES ? " equal to the PC simulation: OK" : " equal to the PC simulation: [X] MISMATCH");
    Serial.print(total_us / DEMO_N_IMAGES);
    Serial.println(" us per inference (micros())");
}

void loop() {
    delay(10000);
}
