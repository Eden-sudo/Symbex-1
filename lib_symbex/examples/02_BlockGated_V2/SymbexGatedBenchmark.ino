#include <Arduino.h>
#include "SymbexBlockGatedLayer.h"
#include "symbex_gated_weights.h" 

SymbexBlockGatedLayer* hidden_layer;
SymbexBlockGatedLayer* out_layer;

// Buffers globales para evitar cualquier colisión en el Stack durante las interrupciones Seriales
uint8_t input_buffer[8];
uint8_t temp_buffer[(OUT_FEATURES_BITS + 7) / 8];
uint8_t bytes_received = 0;

void setup() {
    Serial.begin(115200);
    
    hidden_layer = new SymbexBlockGatedLayer(
        IN_FEATURES_BITS, OUT_FEATURES_BITS, BLOCK_SIZE_BITS, GATE_K_ACTIVE, 
        (const uint8_t*)gate_weights_bin, 
        (const uint8_t*)core_weights_bin
    );
    
    out_layer = new SymbexBlockGatedLayer(
        OUT_FEATURES_BITS, FINAL_CLASSES, 0, 0, 
        nullptr, 
        (const uint8_t*)out_weights_bin
    );
    
    while (!Serial) { ; }
    Serial.println("READY");
}

void loop() {
    // Lectura blindada estilo V1 (Inmune a desincronizaciones de Python)
    while (Serial.available() > 0 && bytes_received < 8) {
        input_buffer[bytes_received++] = Serial.read();
    }

    if (bytes_received == 8) {
        unsigned long start = micros();
        
        hidden_layer->forward(input_buffer, temp_buffer);
        int prediction = out_layer->argmax(temp_buffer);
        
        unsigned long latencia = micros() - start;

        Serial.print("Prediccion: ");
        Serial.print(prediction);
        Serial.print(" | Latencia: ");
        Serial.print(latencia);
        Serial.println(" us");

        bytes_received = 0; // Reset para la siguiente imagen
    }
}
