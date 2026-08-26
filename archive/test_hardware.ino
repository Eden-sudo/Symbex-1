#include <Arduino.h>

#define ITERACIONES 10000
#define NUM_WORDS 100  // 100 palabras de 32 bits = 400 bytes

uint32_t dataA_32[NUM_WORDS];
uint32_t dataB_32[NUM_WORDS];

int8_t dataA_8[NUM_WORDS * 4];
int8_t dataB_8[NUM_WORDS * 4];

void setup() {
    Serial.begin(115200);
    
    // Llenar datos una sola vez
    for(int i=0; i < NUM_WORDS; i++) {
        dataA_32[i] = random(0xFFFFFFFF);
        dataB_32[i] = random(0xFFFFFFFF);
    }
    for(int i=0; i < NUM_WORDS * 4; i++) {
        dataA_8[i] = random(1, 127);
        dataB_8[i] = random(1, 127);
    }
}

void loop() {
    Serial.println("\n========================================");
    Serial.println(" BENCHMARK XTENSA: POPCOUNT vs MAC INT8 ");
    Serial.println("========================================");

    // ----------------------------------------------------
    // TEST 1: XNOR + POPCOUNT (32 bits)
    // ----------------------------------------------------
    uint32_t start_pop = ESP.getCycleCount();
    volatile int acumulador_pop = 0; 
    
    for (int iter = 0; iter < ITERACIONES; iter++) {
        for(int i = 0; i < NUM_WORDS; i++) {
            acumulador_pop += __builtin_popcount(~(dataA_32[i] ^ dataB_32[i]));
        }
    }
    uint32_t end_pop = ESP.getCycleCount();
    uint32_t ciclos_pop_total = (end_pop - start_pop) / ITERACIONES;


    // ----------------------------------------------------
    // TEST 2: Multiplicación Entera (8 bits)
    // ----------------------------------------------------
    uint32_t start_mac = ESP.getCycleCount();
    volatile int acumulador_mac = 0;
    
    for (int iter = 0; iter < ITERACIONES; iter++) {
        for(int i = 0; i < NUM_WORDS * 4; i++) { // 4x iteraciones
            acumulador_mac += dataA_8[i] * dataB_8[i];
        }
    }
    uint32_t end_mac = ESP.getCycleCount();
    uint32_t ciclos_mac_total = (end_mac - start_mac) / ITERACIONES;

    // ----------------------------------------------------
    // RESULTADOS
    // ----------------------------------------------------
    Serial.print("Ciclos de Reloj - Bloque POPCOUNT (32-bit) : "); 
    Serial.println(ciclos_pop_total);
    Serial.print("Ciclos de Reloj - Bloque INT8 MAC (8-bit)  : "); 
    Serial.println(ciclos_mac_total);
    
    float ratio = (float)ciclos_mac_total / (float)ciclos_pop_total;
    Serial.print("\nAceleracion POPCOUNT vs MAC: ");
    Serial.print(ratio);
    Serial.println("x");

    Serial.println("Esperando 3 segundos para repetir...");
    delay(3000);
}
