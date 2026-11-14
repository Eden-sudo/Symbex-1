#include <Arduino.h>
#include <symbex.h>

// pio ci stages this example separately, so these are copies of
// Symbex1_PIO/src/model_symbex.h and Symbex1_PIO/src/demo_input.h.
#include "model_symbex.h"
#include "demo_input.h"

static void run_demo(void) {
    int equal = 0;
    for (int i = 0; i < DEMO_N_IMAGES; i++) {
        const uint32_t start = micros();
        const int cls = symbex_mlp_infer(demo_images[i]);
        const uint32_t elapsed_us = uint32_t(micros() - start);
        equal += (cls == demo_expected[i]);
        Serial.print("image ");
        Serial.print(i);
        Serial.print(": class ");
        Serial.print(cls);
        Serial.print(" (PC simulation: ");
        Serial.print(demo_expected[i]);
        Serial.print(")  ");
        Serial.print(elapsed_us);
        Serial.println(" us");
    }
    Serial.print(equal);
    Serial.print("/");
    Serial.print(DEMO_N_IMAGES);
    Serial.println(equal == DEMO_N_IMAGES ? " equal to the PC simulation: OK" : " equal to the PC simulation: [X] MISMATCH");
}

void setup() {
    Serial.begin(115200);
    run_demo();
}

void loop() {
    delay(10000);
}
