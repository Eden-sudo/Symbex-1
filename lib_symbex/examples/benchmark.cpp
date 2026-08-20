#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include "../include/SymbexLayer.h"
#include "../include/symbex_weights.h"

using namespace std::chrono;

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  SYMBEX-1: BENCHMARK DE VELOCIDAD NATIVA (C++)\n";
    std::cout << "=========================================================\n\n";

    // Usaremos la capa 0 exportada (64 -> 256 neuronas)
    // Nota: Asegurate de que los nombres coincidan con los generados en symbex_weights.h
    SymbexLayer hidden_layer(64, 256, 
                             weights_msb_0, weights_mid_0, weights_lsb_0, 
                             weights_outlier_0, outlier_magnitudes_0, thresholds_0);

    const int NUM_INFERENCIAS = 10000;
    std::cout << "[*] Generando " << NUM_INFERENCIAS << " muestras de sensores aleatorios...\n";
    
    // Generar datos aleatorios
    std::vector<std::vector<uint8_t>> inputs(NUM_INFERENCIAS, std::vector<uint8_t>(8));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    for (int i = 0; i < NUM_INFERENCIAS; i++) {
        for (int j = 0; j < 8; j++) {
            inputs[i][j] = dis(gen);
        }
    }
    
    uint8_t output_buffer[32]; // 256 / 8
    
    std::cout << "[*] Ejecutando Benchmark de Velocidad Extrema...\n";
    
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < NUM_INFERENCIAS; i++) {
        hidden_layer.process_layer(inputs[i].data(), output_buffer);
    }
    
    auto end = high_resolution_clock::now();
    auto duration_us = duration_cast<microseconds>(end - start).count();
    
    double tiempo_promedio_us = (double)duration_us / NUM_INFERENCIAS;
    double inferencias_por_segundo = 1000000.0 / tiempo_promedio_us;
    
    std::cout << "\n--- REPORTE DE VELOCIDAD ---\n";
    std::cout << "[-] Inferencia Total (" << NUM_INFERENCIAS << " ciclos) : " << duration_us / 1000.0 << " ms\n";
    std::cout << "[+] Tiempo Promedio por Inferencia : " << tiempo_promedio_us << " microsegundos\n";
    std::cout << "[+] Tasa de Procesamiento (FPS/Hz) : " << inferencias_por_segundo << " Hz\n";
    std::cout << "=========================================================\n";

    return 0;
}
