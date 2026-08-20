#include <iostream>
#include <bitset> // Para imprimir los bits de forma bonita en la terminal

//verbose
#define SYMBEX_VERBOSE 
#include "../include/Symbex1.h"

int main() {
    std::cout << "==========================================\n";
    std::cout << "   INICIANDO PRUEBA DE MOTOR SYMBEX-1\n";
    std::cout << "==========================================\n\n";

    // 1. MATRICES FALSAS (Simulando la salida de tu herramienta en Python)
    // Red pequeñita de prueba: 8 entradas, 8 neuronas de salida.
    // Como son 8 entradas, cada neurona necesita solo 1 byte por capa.
    
    const uint8_t dummy_msb[8] = {0xFF, 0x00, 0xAA, 0x55, 0xFF, 0x00, 0xAA, 0x55};
    const uint8_t dummy_mid[8] = {0x00, 0xFF, 0x55, 0xAA, 0x00, 0xFF, 0x55, 0xAA};
    const uint8_t dummy_lsb[8] = {0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0};
    
    // Umbrales arbitrarios para forzar a algunas neuronas a disparar y a otras no
    const int16_t dummy_thresholds[8] = {10, -5, 0, 5, 20, -10, 2, 8};

    // 2. CONFIGURACIÓN DE LA CAPA
    SymbexLayer layer_config;
    layer_config.num_inputs = 8;
    layer_config.num_neurons = 8;
    layer_config.weights_msb = dummy_msb;
    layer_config.weights_mid = dummy_mid;
    layer_config.weights_lsb = dummy_lsb;
    layer_config.thresholds = dummy_thresholds;

    // 3. INICIALIZACIÓN DEL MOTOR
    SymbexEngine engine;
    engine.load_network(&layer_config);

    // 4. LECTURA DE SENSOR SIMULADA
    // Imagina que esto es un sensor de proximidad ya binarizado
    const uint8_t sensor_input[1] = { 0b10101010 }; // En hexadecimal es 0xAA

    std::cout << "[>] Estado de sensores crudos : " << std::bitset<8>(sensor_input[0]) << "\n";
    std::cout << "[*] Procesando inferencia con Bit-Slicing (K=3)...\n";

    // 5. ¡LA INFERENCIA! (Esto es lo que ocurrirá en un ciclo de reloj del micro)
    uint8_t decision_final = engine.predict(sensor_input);

    std::cout << "[<] Decision de la Red Neuronal: " << std::bitset<8>(decision_final) << "\n\n";

    // 6. EL DICCIONARIO LUT (La integración de hardware de Bryan)
    std::cout << "--- TRADUCCION A ACCIONES FISICAS (LUT) ---\n";
    
    // Definimos unas máscaras de bits de prueba
    const uint8_t MASK_MOTOR_AVANCE = 0b10000000; // Revisa el 8vo bit
    const uint8_t MASK_ALARMA       = 0b01000000; // Revisa el 7mo bit
    const uint8_t MASK_REVERSA      = 0b00000001; // Revisa el 1er bit

    if (decision_final & MASK_MOTOR_AVANCE) {
        std::cout << " [ACCION] -> Motor principal: AVANCE\n";
    }
    if (decision_final & MASK_ALARMA) {
        std::cout << " [ACCION] -> LED Seguridad: ENCENDIDO\n";
    }
    if (decision_final & MASK_REVERSA) {
        std::cout << " [ACCION] -> Motor principal: REVERSA\n";
    }
    
    if (decision_final == 0) {
         std::cout << " [ACCION] -> Sistema en modo reposo (Ningun bit encendido)\n";
    }

    std::cout << "\n==========================================\n";
    
    return 0;
}
