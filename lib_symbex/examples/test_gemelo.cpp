#include <iostream>
#include <iomanip>
#include "../include/SymbexLayer.h"
#include "../include/symbex_weights.h"

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  SYMBEX-1: PRUEBA DEL GEMELO DIGITAL (C++ NATIVO)\n";
    std::cout << "=========================================================\n\n";

    // Instanciamos la capa usando las dimensiones que arrojo el traductor
    // 64 entradas, 256 neuronas binarizadas (M=2)
    SymbexLayer hidden_layer(64, 256, 
                             weights_msb, weights_mid, weights_lsb, 
                             weights_outlier, outlier_magnitudes, thresholds);

    // Simulamos un dato de entrada del dataset UCI (8 bytes = 64 bits)
    uint8_t dummy_input[8] = {0x00, 0xFF, 0x55, 0xAA, 0x0F, 0xF0, 0x33, 0xCC};
    
    // El buffer de salida necesita (256 neuronas / 8) = 32 bytes
    uint8_t output_buffer[32];

    std::cout << "[*] Ejecutando inferencia a nivel de bits (Vias K=3 + Outliers)...\n";
    
    // Ejecutamos la capa
    hidden_layer.process_layer(dummy_input, output_buffer);

    std::cout << "[+] Inferencia completada sin errores de segmentacion.\n";
    std::cout << "[*] Muestra del buffer de salida (Primeros 4 bytes / 32 neuronas):\n    ";
    
    for(int i = 0; i < 4; i++) {
        std::cout << "0x" << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << (int)output_buffer[i] << " ";
    }
    std::cout << "\n\n=========================================================\n";

    return 0;
}
