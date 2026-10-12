/*
 * symbex_conv2d.c -- FORMATO HWC EN BITS.
 *
 * Los mapas binarios guardan los canales de cada pixel juntos:
 *   pixel (y, x) empieza en la palabra (y * W + x) * PW,  PW = ceil(C / 32)
 *   canal c es el bit (c % 32) de la palabra (c / 32) de ese pixel
 *
 * La ventana (patch) queda en orden [kh][kw][c]:  bit local = (kh*K + kw)*C + c
 * -> los pesos tienen que exportarse en ese mismo orden (plain.py, transpose).
 *
 * Armar la ventana ahora son K*K copias de C bits (una por pixel), leidas
 * directo del mapa -- sin line_buffer, sin extract/insert bit a bit.
 *
 * El padding es el mismo de antes: bit=0 en el patch y bit=1 en la mascara,
 * y la misma correccion exacta en symbex_conv2d_procesar_posicion.
 *
 * DEBUG: compilar con -DSYMBEX_DEBUG_STEM_POS para imprimir el raw del stem
 * (oc=0, oh=5, ow=5). Ya no esta activo por defecto.
 */

#include <stdio.h>
#include <string.h>
#include "layers/spatial/symbex_conv2d.h"
#include "hal/symbex_hal.h"
#include "ops/symbex_bitops.h"
#include <stdbool.h>

// OR de `valor` (hasta 32 bits, ya enmascarado) en `dst` a partir del bit `off`.
static inline void or_bits(uint32_t* dst, uint32_t off, uint32_t valor) {
    uint32_t w = off >> 5;
    uint32_t s = off & 31;
    dst[w] |= valor << s;
    if (s != 0) {
        uint32_t alto = valor >> (32 - s);
        if (alto != 0) dst[w + 1] |= alto;
    }
}

static inline uint32_t mascara_n(uint32_t n) {
    return (n >= 32) ? 0xFFFFFFFFUL : ((1UL << n) - 1);
}

// Copia los primeros n_bits de src (alineado a palabra) a dst desde el bit off.
static inline void poner_bits(uint32_t* dst, uint32_t off, const uint32_t* src, uint32_t n_bits) {
    for (uint32_t i = 0; n_bits > 0; i++) {
        uint32_t n = (n_bits > 32) ? 32 : n_bits;
        or_bits(dst, off, src[i] & mascara_n(n));
        off += n;
        n_bits -= n;
    }
}

// Pone n_bits unos en dst desde el bit off.
static inline void poner_unos(uint32_t* dst, uint32_t off, uint32_t n_bits) {
    while (n_bits > 0) {
        uint32_t n = (n_bits > 32) ? 32 : n_bits;
        or_bits(dst, off, mascara_n(n));
        off += n;
        n_bits -= n;
    }
}

void symbex_conv2d_procesar_posicion(
    const uint32_t* patch_buffer,
    const uint32_t* mascara_padding,
    const int32_t* identity_q_act,
    int32_t* out_valores_q_act,
    const struct SymbexConv2DLayer* layer)
{
    uint16_t window_elems = layer->in_ch * layer->kernel * layer->kernel;
    uint16_t patch_words = (((window_elems + 31) / 32) + 3) & ~3;

    int32_t n_padding = 0;
    for (uint16_t i = 0; i < patch_words; i++) {
        n_padding += symbex_hal_popcount32(mascara_padding[i]);
    }

    #ifdef SYMBEX_DEBUG_STEM_POS
    static int contador_pos = 0;
    bool es_la_posicion = (layer->in_ch == 24 && contador_pos == 165);  // oh=5, ow=5 (5*32+5)
    if (layer->in_ch == 24) contador_pos++;
    #endif

    const uint32_t* w_ptr = layer->weights;

    for (uint16_t oc = 0; oc < layer->out_ch; oc++) {
        int32_t raw_medido = symbex_xnor_popcount(patch_buffer, w_ptr, patch_words, window_elems);

        int32_t raw;
        if (n_padding == 0) {
            raw = raw_medido;
        } else {
            int32_t coincidencias_padding = symbex_popcount_and(mascara_padding, w_ptr, patch_words);
            raw = raw_medido - n_padding + 2 * coincidencias_padding;
        }

        w_ptr += patch_words;

        #ifdef SYMBEX_DEBUG_STEM_POS
        if (es_la_posicion && oc == 0) {
            printf("    [debug] stem oc=0 oh=5 ow=5 raw=%ld\n", (long)raw);
        }
        #endif

        out_valores_q_act[oc] = symbex_dsp_aplicar_capa(raw, identity_q_act[oc], oc, &layer->dsp);
    }
}

// int32 CHW -> bits HWC. Lee en orden (canal por canal); las escrituras caen
// en un mapa chico que vive en cache.
void symbex_dsp_binarizar_mapa(
    const int32_t* valores_q_act, uint32_t* out_bits,
    const int32_t* beta_rsign_q_act,
    uint16_t ch, uint16_t h, uint16_t w)
{
    uint16_t pw = (ch + 31) / 32;
    uint32_t hw = (uint32_t)h * w;
    memset(out_bits, 0, hw * pw * sizeof(uint32_t));
    for (uint16_t c = 0; c < ch; c++) {
        const int32_t* fila = &valores_q_act[(uint32_t)c * hw];
        const int32_t umbral = beta_rsign_q_act[c];
        uint32_t* destino = &out_bits[c / 32];
        const uint32_t bit = 1UL << (c % 32);
        for (uint32_t p = 0; p < hw; p++) {
            if (symbex_dsp_binarizar(fila[p], umbral)) destino[p * pw] |= bit;
        }
    }
}

// float CHW -> bits HWC (umbral 0).
void symbex_binarizar_imagen(const float* input_map, uint32_t* out_bits,
                              uint16_t ch, uint16_t h, uint16_t w) {
    uint16_t pw = (ch + 31) / 32;
    uint32_t hw = (uint32_t)h * w;

    for (uint32_t p = 0; p < hw; p++) {
        for (uint16_t i = 0; i < pw; i++) {
            uint16_t c0 = i * 32;
            uint16_t c1 = (uint16_t)((c0 + 32 < ch) ? c0 + 32 : ch);
            uint32_t palabra = 0;
            for (uint16_t c = c0; c < c1; c++) {
                if (input_map[c * hw + p] >= 0.0f) palabra |= (1UL << (c - c0));
            }
            out_bits[p * pw + i] = palabra;
        }
    }
}

// OBSOLETA en formato HWC: la ventana se lee directo del mapa. Se deja vacia
// para que el codigo viejo que la llama siga compilando; no hace nada.
void symbex_conv2d_preparar_fila(const uint32_t* img_bits, uint32_t* line_buffer,
                                  uint16_t in_ch, uint16_t in_h, uint16_t in_w,
                                  uint16_t kernel, uint16_t stride, uint16_t padding, uint16_t oh) {
    (void)img_bits; (void)line_buffer; (void)in_ch; (void)in_h; (void)in_w;
    (void)kernel; (void)stride; (void)padding; (void)oh;
}

// OJO: el primer argumento ahora es el MAPA HWC completo (antes: line_buffer).
void symbex_conv2d_extraer_ventana(const uint32_t* mapa_hwc, uint32_t* patch_buffer, uint32_t* mascara_padding,
                                    uint16_t in_ch, uint16_t in_h, uint16_t in_w, uint16_t kernel,
                                    uint16_t stride, uint16_t padding, uint16_t oh, uint16_t ow) {
    uint16_t window_elems = in_ch * kernel * kernel;
    uint16_t patch_words = (((window_elems + 31) / 32) + 3) & ~3;
    uint16_t pw = (in_ch + 31) / 32;
    int16_t ih0 = (int16_t)(oh * stride) - (int16_t)padding;
    int16_t iw0 = (int16_t)(ow * stride) - (int16_t)padding;

    memset(patch_buffer, 0, patch_words * sizeof(uint32_t));
    memset(mascara_padding, 0, patch_words * sizeof(uint32_t));

    uint32_t off = 0;
    for (uint16_t kh = 0; kh < kernel; kh++) {
        int16_t ih = ih0 + kh;
        bool fila_valida = (ih >= 0 && ih < in_h);
        for (uint16_t kw = 0; kw < kernel; kw++) {
            int16_t iw = iw0 + kw;
            if (fila_valida && iw >= 0 && iw < in_w) {
                poner_bits(patch_buffer, off, &mapa_hwc[((uint32_t)ih * in_w + iw) * pw], in_ch);
            } else {
                poner_unos(mascara_padding, off, in_ch);   // padding: bit 0 en patch, 1 en mascara
            }
            off += in_ch;
        }
    }

    // Cola del patch (despues de window_elems, hasta el multiplo de 4 palabras):
    // en 1, igual que antes (los pesos tienen 0 ahi -> XNOR da 0, no suma).
    poner_unos(patch_buffer, window_elems, (uint32_t)patch_words * 32 - window_elems);
}
