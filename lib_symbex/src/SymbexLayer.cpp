#include "SymbexLayer.h"
#include <string.h>

// Conteo rápido de bits en 1 (Operación matemática a nivel hardware)
static inline uint8_t symbex_popcount(uint8_t val) {
    uint8_t count = 0;
    for (uint8_t i = 0; i < 8; i++) {
        if ((val >> i) & 1) count++;
    }
    return count;
}

// Constructor
SymbexLayer::SymbexLayer(uint16_t in_f, uint16_t out_f, uint8_t m, uint8_t k, const SymbexSubLayer* subs, uint8_t blk_size, uint8_t k_act) {
    in_features = in_f;
    out_features = out_f;
    M = m;
    k_bits = k;
    sub_layers = subs;
    block_size = blk_size;
    k_active = k_act;
}

// ==========================================
// EL DIRECTOR: EVALUACIÓN Y RANKING TOP-K
// ==========================================
void SymbexLayer::evaluate_gates(const uint8_t* input_buffer, bool* active_blocks) {
    uint8_t num_blocks = out_features / block_size;
    uint16_t bytes_per_input = (in_features + 7) / 8;
    
    int16_t gate_scores[MAX_BLOCKS];
    uint8_t indices[MAX_BLOCKS];

    // 1. Puntuar cada bloque usando el plano 0
    for (uint8_t b = 0; b < num_blocks; b++) {
        int16_t score = 0;
        uint16_t base_idx = b * bytes_per_input;
        
        for (uint16_t i = 0; i < bytes_per_input; i++) {
            uint8_t in_byte = input_buffer[i];
            uint8_t w_gate = SYMBEX_READ_BYTE(&sub_layers[0].bit_planes[0][base_idx + i]);
            uint8_t xnor_val = ~(in_byte ^ w_gate);
            score += ((2 * symbex_popcount(xnor_val)) - 8);
        }
        gate_scores[b] = score;
        indices[b] = b;
        active_blocks[b] = false; // Apagamos todos por defecto
    }

    // 2. Ordenamiento Top-K Rápido (Insertion Sort)
    for (uint8_t i = 1; i < num_blocks; i++) {
        uint8_t key_idx = indices[i];
        int16_t key_val = gate_scores[key_idx];
        int8_t j = i - 1;
        
        while (j >= 0 && gate_scores[indices[j]] < key_val) {
            indices[j + 1] = indices[j];
            j = j - 1;
        }
        indices[j + 1] = key_idx;
    }

    // 3. Encender solo los k_active mejores bloques
    for (uint8_t i = 0; i < k_active; i++) {
        active_blocks[indices[i]] = true;
    }
}

// ======================
// EL CORAZÓN MATEMÁTICO  
// ======================
inline int32_t SymbexLayer::compute_neuron_votes(uint16_t n, const uint8_t* input_buffer) {
    int32_t total_votes = 0;
    
    // Pre-calculamos el salto de memoria una sola vez por neurona (Cero multiplicaciones en el bucle)
    uint16_t bytes_per_neuron = (in_features + 7) / 8;
    uint16_t base_idx = n * bytes_per_neuron;

    // Si hay compuerta (block_size > 0), el músculo está en el plano 1. Si no, en el 0.
    uint8_t plane_idx = (block_size > 0) ? 1 : 0;

    for (uint8_t m = 0; m < M; m++) {
        const SymbexSubLayer& sub = sub_layers[m];

        // 2. NO Le pogas nombres raros  
        int32_t core_score = 0;
        for (uint16_t b = 0; b < bytes_per_neuron; b++) {
            uint8_t in_byte = input_buffer[b];
            uint8_t w_core = SYMBEX_READ_BYTE(&sub.bit_planes[plane_idx][base_idx + b]);
            uint8_t xnor_val = ~(in_byte ^ w_core);
            
            // Usamos la función nativa del compilador C++ (Mucho más rápido que el for loop)
            core_score += ((2 * symbex_popcount(xnor_val)) - 8);
        }
        
        total_votes += core_score;
    }
    return total_votes;
}

// ===================
// FUNCIONES PÚBLICAS   
// ===================

void SymbexLayer::forward(const uint8_t* input_buffer, uint8_t* output_buffer) {
    memset(output_buffer, 0, (out_features + 7) / 8);

    // Evaluamos el Director ANTES de iterar neuronas
    bool active_blocks[MAX_BLOCKS];
    if (block_size > 0) {
        evaluate_gates(input_buffer, active_blocks);
    }

    for (uint16_t n = 0; n < out_features; n++) {
        // [!] EARLY EXIT REAL FÍSICO (Nos saltamos la neurona)
        if (block_size > 0) {
            uint8_t block_id = n / block_size;
            if (!active_blocks[block_id]) continue;
        }

        int32_t total_votes = compute_neuron_votes(n, input_buffer);
        if (total_votes > 0) {
            uint8_t byte_idx = n / 8;
            uint8_t bit_idx = 7 - (n % 8);
            output_buffer[byte_idx] |= (1 << bit_idx);
        }
    }
}

int SymbexLayer::argmax(const uint8_t* input_buffer) {
    int best_class = 0;
    int32_t max_votes = -2147483647 - 1;  
    
    // Evaluamos el Director ANTES de iterar neuronas
    bool active_blocks[MAX_BLOCKS];
    if (block_size > 0) {
        evaluate_gates(input_buffer, active_blocks);
    }

    for (uint16_t n = 0; n < out_features; n++) {
        // [!] EARLY EXIT REAL FÍSICO (Nos saltamos la neurona)
        if (block_size > 0) {
            uint8_t block_id = n / block_size;
            if (!active_blocks[block_id]) continue;
        }

        int32_t total_votes = compute_neuron_votes(n, input_buffer);
        if (total_votes > max_votes) {
            max_votes = total_votes;
            best_class = n;
        }
    }
    return best_class;
}
