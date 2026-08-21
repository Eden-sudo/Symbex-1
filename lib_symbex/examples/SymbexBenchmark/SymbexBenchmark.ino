#include "SymbexLayer.h"
#include "symbex_weights.h"   // trae symbex_net y symbex_init() ya armados

void setup() {
    Serial.begin(115200);
    symbex_init();
    while (!Serial) { ; }
    Serial.println("READY");
}

void loop() {
    static uint8_t input_buffer[8];
    static uint8_t bytes_received = 0;

    while (Serial.available() > 0 && bytes_received < 8) {
        input_buffer[bytes_received++] = Serial.read();
    }

    if (bytes_received == 8) {
        unsigned long start = micros();
        int prediction = symbex_net.classify(input_buffer);
        unsigned long latency = micros() - start;

        Serial.print(prediction);
        Serial.print(",");
        Serial.println(latency);

        bytes_received = 0;
    }
}
