#include <iostream>
#include <chrono>
#include <iomanip>
#include "../include/SymbexNetwork.h"

// ¡El cerebro exportado por Python!
#include "../include/symbex_weights.h" 

int main() {
    std::cout << "==========================================\n";
    std::cout << "  BENCHMARK CIENTIFICO: SYMBEX-1 ENGINE\n";
    std::cout << "==========================================\n\n";

    SymbexNetwork network;
    
    // Conectamos la capa usando los nombres de los arreglos del .h
    SymbexLayer hidden_layer(64, 8, weights_msb, weights_mid, weights_lsb, thresholds);
    network.add_layer(&hidden_layer);

    // Entrada simulada de 64 bits (8 bytes)
    const uint8_t sensor_input[8] = {0xAA, 0xFF, 0x00, 0x55, 0xAA, 0xFF, 0x00, 0x55};
    uint8_t output = 0;

    // 3. WARM-UP (Calentamiento del procesador y caché)
    for (int i = 0; i < 1000; i++) {
        output = network.predict(sensor_input);
    }

    // 4. PRUEBA DE ESTRÉS (Medición de Tiempo)
    const int ITERATIONS = 100000;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < ITERATIONS; i++) {
        output = network.predict(sensor_input); // ¡Inferencia real!
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> total_us = end_time - start_time;

    // 5. CÁLCULOS MATEMÁTICOS PARA EL PAPER
    double avg_latency_us = total_us.count() / ITERATIONS;
    double throughput_ips = 1000000.0 / avg_latency_us;

    // Calculo de Memoria Estática (ROM) -> Pesos y Umbrales
    size_t flash_usage = sizeof(weights_msb) + sizeof(weights_mid) + sizeof(weights_lsb) + sizeof(thresholds);
    
    // Calculo de Memoria Dinámica (SRAM) -> El chasis de la red + Ping-Pong Buffers (2 x 32 bytes)
    size_t sram_usage = sizeof(SymbexNetwork) + sizeof(SymbexLayer) + 64; 

    // 6. IMPRESIÓN DEL REPORTE DE RESULTADOS
    std::cout << "[+] PRUEBA DE ESTRÉS COMPLETADA (" << ITERATIONS << " iteraciones)\n\n";
    
    std::cout << "--- MÉTRICAS DE RENDIMIENTO ---\n";
    std::cout << std::left << std::setw(30) << "Latencia Promedio:" 
              << avg_latency_us << " microsegundos/inferencia\n";
    std::cout << std::left << std::setw(30) << "Throughput Máximo:" 
              << (int)throughput_ips << " inferencias/segundo\n";
              
    std::cout << "\n--- MÉTRICAS DE MEMORIA ---\n";
    std::cout << std::left << std::setw(30) << "Consumo en Flash (ROM):" 
              << flash_usage << " bytes\n";
    std::cout << std::left << std::setw(30) << "Consumo en SRAM (RAM):" 
              << sram_usage << " bytes estaticos (No Malloc)\n";
              
    std::cout << "\n==========================================\n";
    
    return 0;
}
