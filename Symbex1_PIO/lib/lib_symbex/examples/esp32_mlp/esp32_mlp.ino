/*
 * Demo network with random weights, not a trained model.
 * pio ci copies this example to a temporary project, where the original
 * headers in Symbex1_PIO/src are not on the include path, so copies live here.
 */
#include <Arduino.h>
#include <symbex.h>
#include "model_symbex.h"
#include "demo_input.h"

static void run_demo(void) {
    int equal = 0;
    uint32_t best_us = 0xFFFFFFFFUL;
    for (int i = 0; i < DEMO_N_IMAGES; ++i) {
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
}

void setup() {
    Serial.begin(115200);
    delay(1500);
    Serial.print("bit counting: ");
    Serial.println(symbex_hal_name());
    run_demo();
}

void loop() {
    delay(10000);
}
