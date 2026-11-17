// Demo: pio ci copies this example into a temporary project, where the original
// headers cannot be reached by include path, so both headers are copied here.
#include <Arduino.h>
#include <symbex.h>
#include "model_symbex.h"
#include "demo_input.h"

void setup() {
    Serial.begin(115200);
    delay(1500);
    int equal = 0;
    for (int i = 0; i < DEMO_N_IMAGES; i++) {
        const uint32_t t0 = micros();
        const int cls = symbex_mlp_infer(demo_images[i]);
        const uint32_t us = micros() - t0;
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
    Serial.println(" equal to the PC simulation");
}

void loop() {
    delay(10000);
}
