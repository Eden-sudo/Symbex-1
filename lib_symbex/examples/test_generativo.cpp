#include <iostream>
#include <iomanip>
#include "../include/SymbexNetwork.h"
#include "../include/symbex_weights.h"  

int main() {
    SymbexNetwork network;
    
    // CIRUGÍA APLICADA: Actualizamos al nuevo constructor Dual-Path
    // Usamos los sufijos _0 exportados por el CDT de Python.
    // Nota: El compilador ensanchó la capa de 64->128 a 64->256 (M=2)
    SymbexLayer hidden_layer(64, 256, 
                             weights_msb_0, weights_mid_0, weights_lsb_0, 
                             weights_outlier_0, outlier_magnitudes_0, thresholds_0);
                             
    network.add_layer(&hidden_layer);

    // Estado inicial de los sensores
    uint8_t current_state[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    
    // MÁSCARA DE SEGURIDAD (Regla física)
    // 0b01111111 (0x7F) -> Usaremos AND lógico.  
    uint8_t safety_mask = 0x7F;  

    std::cout << "=========================================================================\n";
    std::cout << "  INSPECCIÓN PROFUNDA: BUCLE AUTOREGRESIVO Y EXPANSIÓN SIMBÓLICA\n";
    std::cout << "=========================================================================\n\n";
    
    for(int step = 1; step <= 20; step++) {
        // 1. Mostrar la cinta de correr (Sliding Window)
        std::cout << "Ciclo " << std::dec << std::setfill('0') << std::setw(2) << step << " | Estado: [ ";
        for(int i = 0; i < 8; i++) {
            std::cout << std::hex << std::uppercase << std::setw(2) << (int)current_state[i] << " ";
        }
        std::cout << "] ";

        // 2. La red genera el símbolo puro
        uint8_t raw_symbol = network.predict(current_state);
        
        // 3. Expansión Simbólica (Filtro de seguridad AND)
        uint8_t action = raw_symbol & safety_mask;

        // 4. Mostrar la transformación
        std::cout << "-> Red cruda: 0x" << std::hex << std::setw(2) << (int)raw_symbol  
                  << " -> Filtro AND: 0x" << std::setw(2) << (int)action << "\n";

        // 5. Retroalimentación manual
        for(int j = 0; j < 7; j++) {
            current_state[j] = current_state[j + 1];
        }
        current_state[7] = action; 
    }

    std::cout << "\n=========================================================================\n";
    return 0;
}
