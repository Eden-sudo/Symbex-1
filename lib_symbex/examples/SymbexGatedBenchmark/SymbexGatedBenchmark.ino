#include "SymbexLayer.h"
// Este header será el nuevo que generaremos, sin las magnitudes pesadas
#include "symbex_gated_weights.h" 

void setup() {
    Serial.begin(115200);
    symbex_init();
    while (!Serial) { ; }
    
    Serial.println("=========================================");
    Serial.println(" SYMBEX-1: GATED ARCHITECTURE BENCHMARK");
    Serial.println("=========================================");
    Serial.println("Esperando vector de entrada (8 bytes)...");
}

void loop() {
    static uint8_t input_buffer[8];
    static uint8_t bytes_received = 0;

    // Leer 8 bytes exactos (64 bits, la entrada de UCI Digits)
    while (Serial.available() > 0 && bytes_received < 8) {
        input_buffer[bytes_received++] = Serial.read();
    }

    if (bytes_received == 8) {
        // --- INICIO DE PROFILING ---
        unsigned long start_time = micros();
        
        // La inferencia (que por dentro apagará circuitos dinámicamente)
        int prediction = symbex_net.classify(input_buffer);
        
        unsigned long end_time = micros();
        unsigned long latency = end_time - start_time;
        // --- FIN DE PROFILING ---

        // Reporte serial para tu script de Python
        Serial.print("Prediccion: ");
        Serial.print(prediction);
        Serial.print(" | Latencia: ");
        Serial.print(latency);
        Serial.println(" us");

        bytes_received = 0;
    }
}
