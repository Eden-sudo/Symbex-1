#include <Arduino.h>
#include "SymbexNetwork.h"
#include "symbex_weights.h"

// =====================================================================
// 1. MAPEANDO LA MEMORIA FLASH (Generada por Python)
// =====================================================================

// Capa Oculta: 64 -> 128 (M=4) | Sub-capas 0 a 3
const SymbexSubLayer hidden_subs[4] = {
    {layer0_bit0, layer0_bit1, layer0_bit2, layer0_outliers, layer0_outlier_mag},
    {layer1_bit0, layer1_bit1, layer1_bit2, layer1_outliers, layer1_outlier_mag},
    {layer2_bit0, layer2_bit1, layer2_bit2, layer2_outliers, layer2_outlier_mag},
    {layer3_bit0, layer3_bit1, layer3_bit2, layer3_outliers, layer3_outlier_mag}
};
SymbexLayer hidden_layer(64, 128, 4, hidden_subs);

// Capa de Salida: 128 -> 8 (M=4) | Sub-capas 4 a 7
const SymbexSubLayer output_subs[4] = {
    {layer4_bit0, layer4_bit1, layer4_bit2, layer4_outliers, layer4_outlier_mag},
    {layer5_bit0, layer5_bit1, layer5_bit2, layer5_outliers, layer5_outlier_mag},
    {layer6_bit0, layer6_bit1, layer6_bit2, layer6_outliers, layer6_outlier_mag},
    {layer7_bit0, layer7_bit1, layer7_bit2, layer7_outliers, layer7_outlier_mag}
};
SymbexLayer output_layer(128, 8, 4, output_subs);

// =====================================================================
// 2. ORQUESTADOR DE RED
// =====================================================================
SymbexNetwork net;

void setup() {
    // IMPORTANTE: El baud rate debe coincidir con hardware_tester.py
    Serial.begin(115200);
    
    // Construimos el grafo de inferencia
    net.add_layer(&hidden_layer);
    net.add_layer(&output_layer);
    
    // Esperar a que el puerto serial se inicialice (necesario en algunas placas nativas USB)
    while (!Serial) {
        ; 
    }
    
    // Señal de sincronización para el script de Python
    Serial.println("READY");
}

void loop() {
    // 1. Leer sin bloqueos (Byte por byte hasta llenar 8)
    static uint8_t input_buffer[8];
    static uint8_t bytes_received = 0;
    
    while (Serial.available() > 0 && bytes_received < 8) {
        input_buffer[bytes_received] = Serial.read();
        bytes_received++;
    }
    
    // 2. Cuando tenemos la imagen completa, disparamos la red
    if (bytes_received == 8) {
        // BENCHMARK PURO DE INFERENCIA
        unsigned long start_time = micros();
        int prediction = net.classify(input_buffer);
        unsigned long end_time = micros();
        
        unsigned long latency = end_time - start_time;
        
        // Responder y resetear
        Serial.print(prediction);
        Serial.print(",");
        Serial.println(latency);
        
        bytes_received = 0; // Limpiar para la siguiente imagen
    }
}
