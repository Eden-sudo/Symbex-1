// Compilar con: gcc -O2 -march=native -o bench bench.c
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define ITERATIONS 100000000 

// El sumidero ahora es de 64 bits
volatile int64_t dummy_sink = 0;

int32_t dot_int8(const int8_t* a, const int8_t* b, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

int32_t dot_bitpacked_k1(uint64_t input_bits, uint64_t weight_bits) {
    uint64_t xnor = ~(input_bits ^ weight_bits);
    return 2 * __builtin_popcountll(xnor) - 64;
}

int32_t dot_bitpacked_k3(uint64_t input_bits, uint64_t w0, uint64_t w1, uint64_t w2) {
    int32_t acc = 0;
    acc += 4 * ((2 * __builtin_popcountll(~(input_bits ^ w0))) - 64);
    acc += 2 * ((2 * __builtin_popcountll(~(input_bits ^ w1))) - 64);
    acc += 1 * ((2 * __builtin_popcountll(~(input_bits ^ w2))) - 64);
    return acc;
}

double get_time_ns(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
}

int main() {
    int8_t a[64], b[64];
    for(int i = 0; i < 64; i++) { a[i] = 1; b[i] = -1; }
    
    uint64_t bits_a  = 0xFFFFFFFFFFFFFFFF;
    uint64_t bits_w0 = 0x0000000000000000;
    uint64_t bits_w1 = 0x0F0F0F0F0F0F0F0F;
    uint64_t bits_w2 = 0x3333333333333333;

    struct timespec start, end;
    
    // FIX: Ahora es int64_t. Adiós al Undefined Behavior.
    int64_t acc = 0; 

    printf("==================================================\n");
    printf(" BENCHMARK DE SILICIO REAL: SIMD vs POPCNT\n");
    printf(" Iteraciones: %d\n", ITERATIONS);
    printf("==================================================\n");

    acc = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < ITERATIONS; i++) {
        // Barrera de memoria para evitar que GCC asuma que el array es constante
        __asm__ volatile("" ::: "memory"); 
        acc += dot_int8(a, b, 64);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    dummy_sink = acc;
    printf("[*] 1. SIMD int8 estándar (64 bytes) : %.3f ns / op\n", get_time_ns(start, end) / ITERATIONS);

    acc = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < ITERATIONS; i++) {
        // FIX: Engaña a GCC forzando a mantener bits_a en un registro de CPU real
        __asm__ volatile("" : "+r"(bits_a)); 
        acc += dot_bitpacked_k1(bits_a, bits_w0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    dummy_sink = acc;
    printf("[*] 2. XNOR + POPCNT puro (K=1)      : %.3f ns / op\n", get_time_ns(start, end) / ITERATIONS);

    acc = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < ITERATIONS; i++) {
        __asm__ volatile("" : "+r"(bits_a)); 
        acc += dot_bitpacked_k3(bits_a, bits_w0, bits_w1, bits_w2);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    dummy_sink = acc;
    printf("[*] 3. XNOR + POPCNT escalado (K=3)  : %.3f ns / op\n", get_time_ns(start, end) / ITERATIONS);
    printf("==================================================\n");

    return 0;
}
