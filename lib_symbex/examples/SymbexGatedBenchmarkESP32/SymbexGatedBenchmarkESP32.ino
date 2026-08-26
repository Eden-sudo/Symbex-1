#include <Arduino.h>
#include "SymbexNetwork.h"
#include "SymbexBlockGatedLayer.h"
#include "symbex_gated_weights.h"

// 1. Instanciamos el Orquestador oficial
SymbexNetwork symbex_net;

// 2. Declaramos las dos capas usando tu nueva clase rápida
SymbexBlockGatedLayer* hidden_layer;
SymbexBlockGatedLayer* out_layer;

void setup() {
    Serial.begin(115200);

    // Oculta (Gate Activo)
    hidden_layer = new SymbexBlockGatedLayer(
        IN_FEATURES_BITS, OUT_FEATURES_BITS, BLOCK_SIZE_BITS, GATE_K_ACTIVE,
        (const uint8_t*)gate_weights_bin, (const uint8_t*)core_weights_bin
    );

    // Salida (Densa, K=0, Gate nulo)
    out_layer = new SymbexBlockGatedLayer(
        OUT_FEATURES_BITS, FINAL_CLASSES, 0, 0,
        nullptr, (const uint8_t*)out_weights_bin
    );
    
    // (Aviso: Si SymbexNetwork espera punteros a 'SymbexLayer', 
    // y no 'SymbexBlockGatedLayer', avísame y aplicamos herencia de clases).
}

void loop() {
    if (Serial.available() >= 8) {
        uint8_t input_buffer[8];
        Serial.readBytes(input_buffer, 8);
        
        unsigned long start = micros();
        
        // El Ping-Pong
        uint8_t temp_buffer[(OUT_FEATURES_BITS + 7) / 8] = {0};
        
        // [!] AQUI ESTÁ EL CAMBIO: Usamos las variantes MAC
        hidden_layer->forward_mac(input_buffer, temp_buffer);
        int prediction = out_layer->argmax_mac(temp_buffer);
        
        unsigned long latencia = micros() - start;

        Serial.print("Prediccion: ");
        Serial.print(prediction);
        Serial.print(" | Latencia: ");
        Serial.print(latencia);
        Serial.println(" us");
    }
}
