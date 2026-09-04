/*
 * ============================================================================
 * SYMBEX-1 — Ejemplo manual: red XOR construida 100% a mano, sin Python.
 * ============================================================================
 *
 * Objetivo: mostrar cómo declarar pesos, outliers y armar una red SYMBEX
 * directamente en C++, sin el compilador de entrenamiento.
 *
 * XOR no es linealmente separable, así que hacen falta 2 capas:
 *   Capa oculta: 2 neuronas -> N1 = OR(a,b),  N2 = NAND(a,b)
 *   Capa salida: 1 neurona  -> AND(N1, N2) = XOR(a,b)
 *
 * -------------------------------------------------------------------------
 * EL TRUCO DEL "BIAS" (importante, léelo antes de tocar los pesos)
 * -------------------------------------------------------------------------
 * SymbexLayer no tiene término de bias. Un perceptrón bipolar sin bias no
 * puede resolver OR de forma limpia: con pesos (+1,+1), la suma da 0 quijar
 * en (a=-1,b=+1) y (a=+1,b=-1) — y el hardware activa con ">0" estricto,
 * así que ese empate queda mal clasificado.
 *
 * La solución: cada capa procesa un BYTE COMPLETO (8 posiciones) sin
 * importar cuántas entradas "reales" declares (in_features). Las posiciones
 * sobrantes (bits de relleno) también se comparan por XNOR+popcount, así
 * que podemos usarlas a propósito como un OFFSET CONSTANTE — el equivalente
 * a un bias — eligiendo cuántas de ellas "coinciden" entre el peso fijo que
 * horneamos y el input fijo (0) que siempre mandamos ahí.
 *
 * Con 6 posiciones de relleno (in_features=2 -> 8-2=6 libres), el offset
 * lograble es 2*k-6 para k = cantidad de posiciones que coinciden (peso=0
 * cuando el input de relleno es 0). Usamos:
 *   - Capa oculta (OR y NAND): offset = +2  ->  k=4 coincidencias, 2 no
 *   - Capa salida (AND):        offset =  0  ->  k=3 coincidencias, 3 no
 *
 * Verificado a mano contra las 4 combinaciones de a,b (ver tabla al final).
 * ============================================================================
 */

#include "SymbexLayer.h"

// ----------------------------------------------------------------------
// CAPA OCULTA: in_features=2 (a=bit7, b=bit6), out_features=2, M=1, k_bits=1
// ----------------------------------------------------------------------
// Convención: bit=1 -> peso/entrada +1 (bipolar) | bit=0 -> peso/entrada -1
//
// Neurona 0 (N1 = OR):    wa=+1(bit7=1), wb=+1(bit6=1), relleno: 4 ceros + 2 unos
//                         byte = 1 1 0 0 0 0 1 1 = 0xC3
// Neurona 1 (N2 = NAND):  wa=-1(bit7=0), wb=-1(bit6=0), mismo patrón de relleno
//                         byte = 0 0 0 0 0 0 1 1 = 0x03
const uint8_t hidden_bit0[2] PROGMEM = { 0xC3, 0x03 };

const uint8_t hidden_outliers[2] PROGMEM = { 0x00, 0x00 }; // sin outliers
const int8_t  hidden_outlier_mag[2] = { 0, 0 };            // sin magnitud

const SymbexSubLayer hidden_subs[1] = {
    { { hidden_bit0 }, hidden_outliers, hidden_outlier_mag } // solo 1 plano (k_bits=1)
};

SymbexLayer hidden_layer(2, 2, 1, 1, hidden_subs); // in=2, out=2, M=1, k=1

// ----------------------------------------------------------------------
// CAPA SALIDA: in_features=2 (N1=bit7, N2=bit6), out_features=1, M=1, k_bits=1
// ----------------------------------------------------------------------
// Neurona 0 (AND):  wx=+1(bit7=1), wy=+1(bit6=1), relleno: 3 ceros + 3 unos
//                   byte = 1 1 0 0 0 1 1 1 = 0xC7
const uint8_t output_bit0[1] PROGMEM = { 0xC7 };

const uint8_t output_outliers[1] PROGMEM = { 0x00 };
const int8_t  output_outlier_mag[1] = { 0 };

const SymbexSubLayer output_subs[1] = {
    { { output_bit0 }, output_outliers, output_outlier_mag }
};

SymbexLayer output_layer(2, 1, 1, 1, output_subs); // in=2, out=1, M=1, k=1

// ----------------------------------------------------------------------
// PRUEBA: las 4 combinaciones de entrada, verificadas a mano
// ----------------------------------------------------------------------
void run_xor(uint8_t a, uint8_t b, uint8_t expected) {
    // Empaquetamos: bit7=a, bit6=b, resto en 0 (el "relleno" que ya
    // horneamos en los pesos de arriba).
    uint8_t input_byte = (a << 7) | (b << 6);

    uint8_t hidden_out = 0;
    hidden_layer.forward(&input_byte, &hidden_out);
    // hidden_out: bit7 = N1 (OR), bit6 = N2 (NAND), resto en 0 (memset interno)

    uint8_t final_out = 0;
    output_layer.forward(&hidden_out, &final_out);
    // Nota: usamos forward(), NO argmax() -- argmax busca la mejor clase
    // entre out_features neuronas, y acá out_features=1, así que siempre
    // devolvería "clase 0" sin sentido. forward() sí nos da el bit real.

    uint8_t result = (final_out >> 7) & 1;

    Serial.print(a); Serial.print(" XOR "); Serial.print(b);
    Serial.print(" = "); Serial.print(result);
    Serial.print(" (esperado: "); Serial.print(expected);
    Serial.println(result == expected ? ")  OK" : ")  ERROR");
}

void setup() {
    Serial.begin(115200);
    while (!Serial) { ; }

    Serial.println("--- XOR construido a mano, sin Python ---");
    run_xor(0, 0, 0);
    run_xor(0, 1, 1);
    run_xor(1, 0, 1);
    run_xor(1, 1, 0);
}

void loop() {}

