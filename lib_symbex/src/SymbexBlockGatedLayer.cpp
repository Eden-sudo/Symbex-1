#include "SymbexBlockGatedLayer.h"
#include <Arduino.h>
#define SYMBEX_MAX_BLOCKS 64

SymbexBlockGatedLayer::SymbexBlockGatedLayer(uint16_t in_f, uint16_t out_f, uint16_t b_size, uint16_t k_act, const uint8_t* g_w, const uint8_t* c_w) {
    in_features = in_f;
    out_features = out_f;
    block_size = b_size;
    num_blocks = (b_size > 0) ? (out_features / block_size) : 0;
    k_active = k_act;
    gate_weights = g_w;
    core_weights = c_w;
}

// ==========================================
// VARIANTE MAC: RECONSTRUCCIÓN A INT8 Y MULTIPLICACIÓN NATIVA
// ==========================================
void SymbexBlockGatedLayer::forward_mac(const uint8_t* __restrict input_buffer, uint8_t* __restrict output_buffer) {
    const int loc_out_features = this->out_features;
    const int loc_in_bytes = this->in_features / 8;
    const int loc_num_blocks = this->num_blocks;
    const int loc_k_active = this->k_active;
    const int loc_block_size = this->block_size;
    
    memset(output_buffer, 0, (loc_out_features + 7) / 8);
    if (loc_k_active == 0 || loc_num_blocks == 0) return;

    int16_t gate_scores[SYMBEX_MAX_BLOCKS];
    uint8_t block_indices[SYMBEX_MAX_BLOCKS];
    bool active_blocks[SYMBEX_MAX_BLOCKS];
    memset(active_blocks, 0, SYMBEX_MAX_BLOCKS);

    // 1. DIRECTOR CON MAC NATIVA
    for(int b = 0; b < loc_num_blocks; b++) {
        int score = 0;
        for(int i = 0; i < loc_in_bytes; i++) {
            uint8_t in_byte = input_buffer[i];
            uint8_t w_byte = gate_weights[b * loc_in_bytes + i];
            
            // Reconstrucción y multiplicación bit a bit
            for(int bit = 7; bit >= 0; bit--) {
                int8_t val_in = ((in_byte >> bit) & 1) ? 1 : -1;
                int8_t val_w  = ((w_byte >> bit) & 1) ? 1 : -1;
                score += (val_in * val_w);
            }
        }
        gate_scores[b] = score;
        block_indices[b] = b; 
    }
    
    // 2. INSERTION SORT (Intacto)
    for (int i = 1; i < loc_num_blocks; i++) {
        uint8_t key_idx = block_indices[i];
        int16_t key_val = gate_scores[key_idx];
        int j = i - 1;
        while (j >= 0 && gate_scores[block_indices[j]] < key_val) {
            block_indices[j + 1] = block_indices[j];
            j--;
        }
        block_indices[j + 1] = key_idx;
    }
    for(int i = 0; i < loc_k_active; i++) {
        active_blocks[block_indices[i]] = true;
    }

    // 3. MÚSCULO CON MAC NATIVA
    for(int n = 0; n < loc_out_features; n++) {
        if(!active_blocks[n / loc_block_size]) continue;
        
        int score = 0;
        for(int i = 0; i < loc_in_bytes; i++) {
            uint8_t in_byte = input_buffer[i];
            uint8_t w_byte = core_weights[n * loc_in_bytes + i];
            
            for(int bit = 7; bit >= 0; bit--) {
                int8_t val_in = ((in_byte >> bit) & 1) ? 1 : -1;
                int8_t val_w  = ((w_byte >> bit) & 1) ? 1 : -1;
                score += (val_in * val_w);
            }
        }
        
        if(score > 0) {
            output_buffer[n / 8] |= (1 << (7 - (n % 8)));
        }
    }
}

int SymbexBlockGatedLayer::argmax_mac(const uint8_t* __restrict input_buffer) {
    int best_class = 0;
    int max_score = -99999;
    
    const int loc_in_bytes = this->in_features / 8;
    const int loc_out_features = this->out_features;
    
    for(int c = 0; c < loc_out_features; c++) {
        int score = 0;
        for(int i = 0; i < loc_in_bytes; i++) {
            uint8_t in_byte = input_buffer[i];
            uint8_t w_byte = core_weights[c * loc_in_bytes + i];
            
            for(int bit = 7; bit >= 0; bit--) {
                int8_t val_in = ((in_byte >> bit) & 1) ? 1 : -1;
                int8_t val_w  = ((w_byte >> bit) & 1) ? 1 : -1;
                score += (val_in * val_w);
            }
        }
        
        if(score > max_score) {
            max_score = score;
            best_class = c;
        }
    }
    return best_class;
}
