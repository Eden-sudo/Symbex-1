/*
 * symbex_network_conv.c -- FORMATO HWC EN BITS (ver symbex_conv2d.c).
 *
 * Cambios contra la version anterior:
 *   - binarizar_mapa_q_act escribe bits HWC (canales de cada pixel juntos).
 *   - Ya no se usa line_buffer: extraer_ventana lee directo del mapa de bits.
 *     El parametro line_buffer se conserva en las firmas para no romper a
 *     quien llama, pero se ignora.
 * Los mapas int32 (x, identity, salidas) siguen en CHW, igual que antes.
 */

#include "core/network/symbex_network_conv.h"
#include "layers/spatial/symbex_pool_binario.h"
#include "hal/symbex_hal.h"
#include "ops/symbex_bitops.h"
#include <stddef.h>
#include <string.h>

void symbex_network_conv_init(struct SymbexNetworkConv* net) {
    net->bloque_count = 0;
}

bool symbex_network_conv_add_bloque(struct SymbexNetworkConv* net, const struct SymbexResidualBlock* bloque) {
    if (net->bloque_count >= SYMBEX_NETWORK_CONV_MAX_BLOQUES) return false;
    net->bloques[net->bloque_count++] = bloque;
    return true;
}

static uint16_t out_dim(uint16_t in_dim, uint16_t kernel, uint16_t stride, uint16_t padding) {
    return (in_dim + 2 * padding - kernel) / stride + 1;
}

// int32 CHW -> bits HWC. Se recorre canal por canal para LEER en orden
// (la entrada es CHW); las escrituras caen en un mapa chico (<= 4 KB) que
// vive en cache, asi que no importa que salten.
static void binarizar_mapa_q_act(const int32_t* valores, uint32_t* bits_out,
                                  const int32_t* beta, uint16_t ch, uint16_t h, uint16_t w) {
    uint16_t pw = (ch + 31) / 32;
    uint32_t hw = (uint32_t)h * w;
    memset(bits_out, 0, hw * pw * sizeof(uint32_t));
    for (uint16_t c = 0; c < ch; c++) {
        const int32_t* fila = &valores[(uint32_t)c * hw];
        const int32_t umbral = beta[c];
        uint32_t* destino = &bits_out[c / 32];
        const uint32_t bit = 1UL << (c % 32);
        for (uint32_t p = 0; p < hw; p++) {
            if (symbex_dsp_binarizar(fila[p], umbral)) destino[p * pw] |= bit;
        }
    }
}

static void procesar_capa_completa(
    const uint32_t* bits_entrada, const int32_t* identity_q_act,
    int32_t* out_q_act, const struct SymbexConv2DLayer* layer,
    uint32_t* line_buffer, uint32_t* patch_buffer)
{
    (void)line_buffer;
    uint16_t out_h = out_dim(layer->in_h, layer->kernel, layer->stride, layer->padding);
    uint16_t out_w = out_dim(layer->in_w, layer->kernel, layer->stride, layer->padding);

    for (uint16_t oh = 0; oh < out_h; oh++) {
        for (uint16_t ow = 0; ow < out_w; ow++) {
            uint32_t mascara_padding[64];
            symbex_conv2d_extraer_ventana(bits_entrada, patch_buffer, mascara_padding,
                                           layer->in_ch, layer->in_h, layer->in_w, layer->kernel,
                                           layer->stride, layer->padding, oh, ow);

            int32_t identity_pos[256];
            int32_t salida_pos[256];
            for (uint16_t oc = 0; oc < layer->out_ch; oc++) {
                identity_pos[oc] = identity_q_act[(oc * out_h + oh) * out_w + ow];
            }

            symbex_conv2d_procesar_posicion(patch_buffer, mascara_padding, identity_pos, salida_pos, layer);

            for (uint16_t oc = 0; oc < layer->out_ch; oc++) {
                out_q_act[(oc * out_h + oh) * out_w + ow] = salida_pos[oc];
            }
        }
    }
}

static void procesar_downsample(
    const int32_t* x_q_act, const struct SymbexResidualBlock* bloque,
    uint16_t in_ch, uint16_t entrada_h, uint16_t entrada_w,
    uint32_t* bits_entrada_full, uint32_t* bits_pooled,
    uint32_t* patch_buffer,
    int32_t* out_identity_q_act)
{
    binarizar_mapa_q_act(x_q_act, bits_entrada_full, bloque->beta_rsign_downsample,
                          in_ch, entrada_h, entrada_w);

    symbex_pool_binario_2x2(bits_entrada_full, bits_pooled, in_ch, entrada_h, entrada_w);

    uint16_t pooled_h = entrada_h / 2;
    uint16_t pooled_w = entrada_w / 2;
    uint16_t out_ch = bloque->conv1.out_ch;
    uint16_t kernel = bloque->downsample_kernel;
    uint16_t padding = bloque->downsample_padding;

    uint16_t window_elems = in_ch * kernel * kernel;
    uint16_t patch_words = (((window_elems + 31) / 32) + 3) & ~3;

    uint16_t out_h = out_dim(pooled_h, kernel, 1, padding);
    uint16_t out_w = out_dim(pooled_w, kernel, 1, padding);

    for (uint16_t oh = 0; oh < out_h; oh++) {
        for (uint16_t ow = 0; ow < out_w; ow++) {
            uint32_t mascara_padding[64];
            symbex_conv2d_extraer_ventana(bits_pooled, patch_buffer, mascara_padding,
                                           in_ch, pooled_h, pooled_w, kernel, 1, padding, oh, ow);

            int32_t n_padding = 0;
            for (uint16_t i = 0; i < patch_words; i++) n_padding += symbex_hal_popcount32(mascara_padding[i]);

            const uint32_t* w_ptr = bloque->downsample_pesos;
            for (uint16_t oc = 0; oc < out_ch; oc++) {
                int32_t raw_medido = symbex_xnor_popcount(patch_buffer, w_ptr, patch_words, window_elems);
                int32_t raw = raw_medido;
                if (n_padding > 0) {
                    int32_t coincidencias_padding = symbex_popcount_and(mascara_padding, w_ptr, patch_words);
                    raw = raw_medido - n_padding + 2 * coincidencias_padding;
                }
                w_ptr += patch_words;
                out_identity_q_act[(oc * out_h + oh) * out_w + ow] =
                    symbex_dsp_aplicar_downsample(raw, oc, &bloque->downsample_dsp);
            }
        }
    }
}

void symbex_network_conv_procesar_bloque(
    const struct SymbexResidualBlock* bloque,
    const int32_t* x_actual_q_act,
    int32_t* destino_q_act,
    int32_t* buffer_temporal,
    uint32_t* bits_buffer,
    int32_t* buffer_identity_downsample,
    uint32_t* bits_buffer_full_downsample,
    uint32_t* bits_buffer_pooled_downsample,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* debug_identity1, int32_t* debug_conv1_out
) {
    const int32_t* identity_conv1 = x_actual_q_act;
    if (bloque->tiene_downsample) {
        static uint32_t patch_buffer_ds[64];
        procesar_downsample(x_actual_q_act, bloque,
                             bloque->conv1.in_ch, bloque->conv1.in_h, bloque->conv1.in_w,
                             bits_buffer_full_downsample, bits_buffer_pooled_downsample,
                             patch_buffer_ds, buffer_identity_downsample);
        identity_conv1 = buffer_identity_downsample;
    }

    uint16_t h1 = out_dim(bloque->conv1.in_h, bloque->conv1.kernel, bloque->conv1.stride, bloque->conv1.padding);
    uint16_t w1 = out_dim(bloque->conv1.in_w, bloque->conv1.kernel, bloque->conv1.stride, bloque->conv1.padding);

    // Esta copia debe pasar ANTES de procesar_capa_completa (buffer_temporal
    // e identity_conv1 pueden ser el mismo puntero).
    if (debug_identity1 != NULL) {
        uint32_t n = (uint32_t)bloque->conv1.out_ch * h1 * w1;
        for (uint32_t k = 0; k < n; k++) debug_identity1[k] = identity_conv1[k];
    }

    binarizar_mapa_q_act(x_actual_q_act, bits_buffer, bloque->beta_rsign1,
                          bloque->conv1.in_ch, bloque->conv1.in_h, bloque->conv1.in_w);

    procesar_capa_completa(bits_buffer, identity_conv1, buffer_temporal, &bloque->conv1, line_buffer, patch_buffer);

    if (debug_conv1_out != NULL) {
        uint32_t n = (uint32_t)bloque->conv1.out_ch * h1 * w1;
        for (uint32_t k = 0; k < n; k++) debug_conv1_out[k] = buffer_temporal[k];
    }

    binarizar_mapa_q_act(buffer_temporal, bits_buffer, bloque->beta_rsign2,
                          bloque->conv1.out_ch, h1, w1);

    procesar_capa_completa(bits_buffer, buffer_temporal, destino_q_act, &bloque->conv2, line_buffer, patch_buffer);
}

bool symbex_network_conv_run(
    const struct SymbexNetworkConv* net,
    const int32_t* x_inicial_q_act,
    int32_t* buffer_a, int32_t* buffer_b,
    uint32_t* bits_buffer,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* out_final_q_act
) {
    return symbex_network_conv_run_ext(net, x_inicial_q_act, buffer_a, buffer_b,
                                        bits_buffer, NULL, NULL, NULL,
                                        line_buffer, patch_buffer, out_final_q_act);
}

bool symbex_network_conv_run_ext(
    const struct SymbexNetworkConv* net,
    const int32_t* x_inicial_q_act,
    int32_t* buffer_a, int32_t* buffer_b,
    uint32_t* bits_buffer,
    int32_t* buffer_identity_downsample,
    uint32_t* bits_buffer_full_downsample,
    uint32_t* bits_buffer_pooled_downsample,
    uint32_t* line_buffer, uint32_t* patch_buffer,
    int32_t* out_final_q_act
) {
    if (net->bloque_count == 0) return false;

    for (int i = 0; i < net->bloque_count; i++) {
        if (net->bloques[i]->tiene_downsample &&
            (buffer_identity_downsample == NULL || bits_buffer_full_downsample == NULL || bits_buffer_pooled_downsample == NULL)) {
            return false;
        }
    }

    int32_t* x_actual = buffer_a;
    if (x_inicial_q_act != buffer_a) {
        const struct SymbexConv2DLayer* primera_conv1 = &net->bloques[0]->conv1;
        uint32_t tam = (uint32_t)primera_conv1->in_ch * primera_conv1->in_h * primera_conv1->in_w;
        for (uint32_t k = 0; k < tam; k++) buffer_a[k] = x_inicial_q_act[k];
    }

    int32_t* destino_conv1 = buffer_b;

    for (int i = 0; i < net->bloque_count; i++) {
        const struct SymbexResidualBlock* bloque = net->bloques[i];
        bool es_ultimo_bloque = (i == net->bloque_count - 1);
        int32_t* destino = es_ultimo_bloque ? out_final_q_act : x_actual;

        symbex_network_conv_procesar_bloque(
            bloque, x_actual, destino, destino_conv1,
            bits_buffer, buffer_identity_downsample,
            bits_buffer_full_downsample, bits_buffer_pooled_downsample,
            line_buffer, patch_buffer, NULL, NULL
        );
    }

    return true;
}
