/*
 * main.cpp -- arma la red CIFAR-10 completamente binaria (stem +
 * N bloques residuales + clasificador), usando la libreria symbex
 * y los datos exportados en datos_cifar10_dsp.h.
 *
 * GENERICO EN CANTIDAD DE BLOQUES: las declaraciones de bloques,
 * symbex_preparar_bloques() y el arreglo bloques[]/n_bloques ya NO
 * estan escritos a mano aca -- vienen de bloques_generados.h,
 * generado por compiler_core/export/plain.py:generar_wiring_cpp().
 *
 * IMAGENES POR SERIE: ya no hay imagen_test.h horneado en tiempo de
 * compilacion. Python manda las imagenes en vivo por USB/serie
 * (ver enviar_imagenes_serie.py) -- elimina la clase entera de bugs
 * de sincronizacion que costo perseguir ayer (exportar a un .h,
 * copiarlo al lugar correcto, no olvidarse, etc).
 *
 * FORMATO HWC EN BITS (libreria v4): los mapas binarios guardan los
 * canales de cada pixel juntos, y extraer_ventana lee DIRECTO del mapa
 * (ya no hay preparar_fila / line_buffer en el stem). Por eso los
 * buffers de bits son de 32*32 palabras (un pixel = una palabra para
 * hasta 32 canales). Mismo codigo que symbex_x86/main_x86.cpp.
 *
 * Protocolo, simple a proposito:
 *   Python manda 1 byte marca (0xAA = silencioso, 0xAB = verboso,
 *   con las estadisticas por bloque) + 3072 bytes crudos de imagen
 *   RGB (32x32x3, orden HWC -- fila, columna, canal).
 *   El chip responde una linea: "RESULT:<clase>,<ms>\n"
 *   Python ya tiene la etiqueta real y la prediccion de PyTorch --
 *   el chip no necesita saber nada de eso, solo clasifica.
 */

#include <Arduino.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "layers/spatial/symbex_bpr.h"
#include "layers/spatial/symbex_conv2d.h"
#include "layers/vector/symbex_dense.h"
#include "core/network/symbex_network_conv.h"
#include "ops/symbex_mac_1bit.h"
#include "hal/symbex_hal.h"
#include "ops/symbex_bitops.h"
#include "layers/spatial/symbex_pool_binario.h"

#include "datos_cifar10_dsp.h"
#ifndef SYMBEX_LAYOUT_HWC
#error "Pesos en formato viejo (CHW): reexporta con layout=hwc"
#endif

// ======================================================================
// Macros para construir las structs desde los arrays exportados --
// genericas, no dependen de cuantos bloques haya. bloques_generados.h
// las usa.
// ======================================================================

#define HACER_DSP(nombre) { \
    nombre##_escala, nombre##_shift_bn, nombre##_gamma_rprelu, nombre##_zeta, \
    (bool)(nombre##_USA_SHIFT), nombre##_log2_beta, nombre##_signo_beta, nombre##_slope, \
    SYMBEX_Q_ACT, SYMBEX_Q_MULT \
}

#define HACER_CONV(nombre, ih, iw) { \
    HACER_DSP(nombre), nombre##_pesos, \
    nombre##_IN_CH, (ih), (iw), \
    nombre##_OUT_CH, nombre##_KERNEL, nombre##_STRIDE, nombre##_PADDING \
}

// ======================================================================
// Bloques residuales, symbex_preparar_bloques(), bloques[]/n_bloques/
// nombres_bloques[] -- TODO esto viene generado, para N bloques.
// Define RES_STEM tambien (resolucion de entrada de la imagen).
// ======================================================================
#include "bloques_generados.h"

struct SymbexConv2DLayer capa_stem = HACER_CONV(STEM_CONV1, RES_STEM, RES_STEM);

// ======================================================================
// Buffers de trabajo -- dimensionados al mapa mas grande REAL de la
// arquitectura. Si la arquitectura cambia (mas canales, mas stages),
// estos numeros hay que revisarlos -- no se generan automaticamente
// todavia (queda como el proximo ajuste, no bloquea esta version).
// ======================================================================

#define BUFFER_MAX_ELEMENTOS (16 * 32 * 32)
// HWC: un mapa de bits ocupa H*W*ceil(C/32) palabras. Peor caso: 32x32 con
// hasta 32 canales (BPR de 24, stage 1 de 16) = 1024 palabras.
#define BITS_MAX_PALABRAS (32 * 32 * 1)

static int32_t buffer_a[BUFFER_MAX_ELEMENTOS];
static int32_t buffer_b[BUFFER_MAX_ELEMENTOS];
static int32_t buffer_identity_downsample[BUFFER_MAX_ELEMENTOS];
static int32_t buffer_final[SYMBEX_ULTIMA_SALIDA];   // definido en bloques_generados.h, no fijo

static uint32_t bits_buffer[BITS_MAX_PALABRAS];
static uint32_t bits_buffer_full_downsample[BITS_MAX_PALABRAS];
static uint32_t bits_buffer_pooled_downsample[BITS_MAX_PALABRAS];
static uint32_t line_buffer[BITS_MAX_PALABRAS];   // ya no se usa (HWC); se conserva por la firma de procesar_bloque
static uint32_t patch_buffer[64];

static uint32_t bpr_bits[BITS_MAX_PALABRAS];
static uint32_t bits_finales[(SYMBEX_ULTIMA_SALIDA) / 32 + 1];

// Buffer de la imagen recibida por serie -- RGB crudo, HWC (fila, columna, canal).
static uint8_t imagen_rgb[RES_STEM * RES_STEM * 3];

void imprimir_estadisticas(const char* nombre, const int32_t* buffer, uint32_t n) {
    int32_t minimo = buffer[0], maximo = buffer[0];
    int64_t suma = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (buffer[i] < minimo) minimo = buffer[i];
        if (buffer[i] > maximo) maximo = buffer[i];
        suma += buffer[i];
    }
    Serial.print("  "); Serial.print(nombre);
    Serial.print(" | min="); Serial.print(minimo);
    Serial.print(" max="); Serial.print(maximo);
    Serial.print(" suma="); Serial.println((long)suma);
}

int symbex_clasificar_imagen(const uint8_t* imagen_rgb_in, bool verboso) {
    // 1. BPR: imagen -> 24 planos de bits (HWC: una palabra por pixel)
    symbex_rgb_a_24bits(imagen_rgb_in, bpr_bits, RES_STEM, RES_STEM);

    // 2. Stem: conv -> Q_ACT (identity=0, no hay suma residual).
    //    La ventana se lee directo de bpr_bits (sin preparar_fila).
    uint16_t out_h_stem = RES_STEM, out_w_stem = RES_STEM;

    for (uint16_t oh = 0; oh < out_h_stem; oh++) {
        for (uint16_t ow = 0; ow < out_w_stem; ow++) {
            uint32_t mascara_padding[64];
            symbex_conv2d_extraer_ventana(bpr_bits, patch_buffer, mascara_padding,
                                           capa_stem.in_ch, capa_stem.in_h, capa_stem.in_w, capa_stem.kernel,
                                           capa_stem.stride, capa_stem.padding, oh, ow);
            int32_t identity_cero[16] = {0};
            int32_t salida_pos[16];
            symbex_conv2d_procesar_posicion(patch_buffer, mascara_padding, identity_cero, salida_pos, &capa_stem);
            for (uint16_t oc = 0; oc < capa_stem.out_ch; oc++) {
                buffer_a[(oc * out_h_stem + oh) * out_w_stem + ow] = salida_pos[oc];
            }
        }
    }
    if (verboso) imprimir_estadisticas("stem", buffer_a, capa_stem.out_ch * out_h_stem * out_w_stem);

    // 3. Los N bloques residuales -- bloques[]/n_bloques/nombres_bloques[]
    // vienen de bloques_generados.h, no se declaran mas aca.
    int32_t* x_actual = buffer_a;
    int32_t* otro_buffer = buffer_b;

    for (int i = 0; i < n_bloques; i++) {
        int32_t* destino = (i == n_bloques - 1) ? buffer_final : otro_buffer;

        symbex_network_conv_procesar_bloque(
            bloques[i].bloque, x_actual, destino, buffer_identity_downsample,
            bits_buffer, buffer_identity_downsample, bits_buffer_full_downsample, bits_buffer_pooled_downsample,
            line_buffer, patch_buffer, NULL, NULL
        );

        if (verboso) imprimir_estadisticas(nombres_bloques[i], destino, bloques[i].n_salida);

        if (i < n_bloques - 1) {
            for (uint32_t k = 0; k < bloques[i].n_salida; k++) x_actual[k] = destino[k];
        }
    }

    // 4. Clasificador
    struct SymbexLayerData capa_clasificador;
    capa_clasificador.weights = CLASIFICADOR_pesos;
    capa_clasificador.in_features = CLASIFICADOR_IN_FEATURES;
    capa_clasificador.out_features = CLASIFICADOR_OUT_FEATURES;
    capa_clasificador.k_bits = 1;

    uint16_t total_features = CLASIFICADOR_IN_FEATURES;
    uint16_t words_needed = (total_features + 31) / 32;
    for (uint16_t wi = 0; wi < words_needed; wi++) bits_finales[wi] = 0;
    for (uint16_t i = 0; i < total_features; i++) {
        if (symbex_dsp_binarizar(buffer_final[i], CLASIFICADOR_beta_rsign_entrada[i])) {
            bits_finales[i / 32] |= (1UL << (i % 32));
        }
    }

    int clase = symbex_clasificador_binario_argmax(
        bits_finales, &capa_clasificador,
        CLASIFICADOR_escala, CLASIFICADOR_shift_bn,
        SYMBEX_Q_ACT, SYMBEX_Q_MULT
    );

    if (verboso) {
        int32_t min_f = buffer_final[0], max_f = buffer_final[0];
        int64_t suma_f = 0;
        for (int i = 0; i < SYMBEX_ULTIMA_SALIDA; i++) {
            if (buffer_final[i] < min_f) min_f = buffer_final[i];
            if (buffer_final[i] > max_f) max_f = buffer_final[i];
            suma_f += buffer_final[i];
        }
        Serial.print("  buffer_final: min="); Serial.print(min_f);
        Serial.print(" max="); Serial.print(max_f);
        Serial.print(" suma="); Serial.println((long)suma_f);
    }

    return clase;
}

// Lee exactamente `n` bytes del serial, esperando lo que haga falta
// (hasta el timeout configurado con Serial.setTimeout). Devuelve
// false si se corto antes de completar los n bytes.
static bool leer_bytes_exacto(uint8_t* destino, size_t n) {
    size_t leidos = Serial.readBytes(destino, n);
    return leidos == n;
}

void setup() {
    Serial.begin(115200);
    Serial.setTimeout(5000);   // una imagen (3072 bytes) a 115200 baud tarda ~270ms -- 5s da margen de sobra
    delay(1000);
    symbex_preparar_bloques();
    Serial.println("SYMBEX-1 listo -- esperando imagenes por serie");
    Serial.println(symbex_hal_nombre());
}

void loop() {
    // Espera el byte de marca (0xAA silencioso, 0xAB verboso). Cualquier
    // otro byte se descarta -- asi el protocolo se auto-resincroniza si
    // algo quedo a mitad de camino (por ejemplo un reset a mitad de envio).
    while (!Serial.available()) { }
    uint8_t marca = Serial.read();
    if (marca != 0xAA && marca != 0xAB) return;
    bool verboso = (marca == 0xAB);

    if (!leer_bytes_exacto(imagen_rgb, sizeof(imagen_rgb))) {
        Serial.println("ERROR:timeout esperando los 3072 bytes de la imagen");
        return;
    }

    unsigned long t_inicio = millis();
    int clase = symbex_clasificar_imagen(imagen_rgb, verboso);
    unsigned long t_fin = millis();

    Serial.print("RESULT:");
    Serial.print(clase);
    Serial.print(",");
    Serial.println(t_fin - t_inicio);
}
