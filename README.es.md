# SYMBEX-1 (V2 - New Block-Gated Architecture)

SYMBEX-1 es una librería optimizada para microcontroladores que permite la inferencia de redes neuronales binarizadas (BNN). 

Incorpora herramientas de entrenamiento y conversión directa para transformar modelos grandes (FP16 o INT) en topologías binarizadas altamente optimizadas. Esto permite que incluso microcontroladores limitados de 8 bits sin FPU (como el ATmega328P) o procesadores embebidos (ESP32) puedan ejecutar redes neuronales complejas de forma precisa y rápida.

## La Actualización (De V1 a V2): Outliers Estáticos vs. Saltos Dinámicos

La primera versión de SYMBEX mitigaba el error de cuantización guardando los *outliers* (los pesos más críticos) en un canal paralelo de matrices numéricas estáticas enteras. Aunque efectivo, estas matrices eran pesadas computacionalmente y consumían memoria vital.

**La actualización V2 incorpora la conversión de esos pesos estáticos y pesados hacia una arquitectura binarizada dinámica (Block-Gating) mucho más ligera.** 
En lugar de procesar toda la red de forma estática, la arquitectura evalúa los datos periféricos en tiempo real y realiza **saltos dinámicos** (*Early-Exit* físico). Apaga físicamente las ramas y bloques de la red que no son relevantes para la entrada actual. Esto permite mantener la misma precisión global de la red original, pero maximizando la eficiencia y reduciendo drásticamente la latencia y la memoria en uso.

## Arquitectura

```text
┌─────────────────────────────────────────────────────────────┐
│ Compilador Python (Transformación FP16/INT → 1-Bit Dinámico)│
│ Teacher FP32 → QAT Estudiante Binarizado (Block-Gating)     │
│ ↓                                                           │
│ Exportador → symbex_gated_weights.h (Planos de bits)        │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Inferencia Optimizada (C++ / ESP32 / AVR)                   │
│ [Input Empaquetado]                                         │
│    ├─> 1. Gate: Evalúa K bloques y ordena Top-K             │
│    └─> 2. Core: Salto dinámico de bloques apagados          │
│ [Salida Densa]                                              │
│    └─> 3. Argmax: Decisión final directa sin FP             │
└─────────────────────────────────────────────────────────────┘
```

### 1. Motor de Empaquetamiento (XNOR + Popcount)
Tanto los pesos como las entradas se empaquetan en bits puros. El procesador nunca multiplica aritméticamente; aplica una operación binaria `XNOR` seguida de un conteo de bits (`popcount`). 
- En hardware avanzado (PC, Xtensa), aprovecha el paralelismo de registros.
- En microcontroladores sencillos, utiliza el algoritmo **SWAR** (SIMD Within A Register) inyectado mediante una capa HAL (`symbex_config.h`), asegurando un conteo veloz (~15 ciclos) puramente en software.
- Toda la matemática está simplificada para ser **estrictamente creciente** (se eliminan desplazamientos y restas en inferencia), evaluando directamente la acumulación bruta de aciertos para maximizar la velocidad de reloj.

### 2. Block-Gating y Early Exit (Saltos Dinámicos)
Una capa oculta masiva (ej. 512 neuronas) se subdivide en bloques aislados. El Gate binarizado hace una revisión periférica rápida de la entrada, puntúa la relevancia topológica de cada bloque y activa solo los mejores (Top-K). El bucle principal de inferencia lee estas banderas y ejecuta un salto físico (`continue;`) si el bloque no es necesario, evadiendo ciclos de cálculo muertos.

## Estructura Modular del SDK

```text
Symbex1/
├── lib_symbex/           # Motor C++ optimizado, HAL y ejemplos (ESP32/AVR)
└── tools/                
    ├── symbex_compiler.py # Compilador maestro universal (CLI para FP32 -> Binarizado V1/V2)
    └── compiler_core/     # Núcleo modular en Python (Modelos, Entrenador, Validador, Exportador)
```

## Resultados de Validación en Hardware (SYMBEX-1 V2)

El framework SYMBEX-1 está diseñado para escalar de manera determinista desde microcontroladores de 32 bits hasta arquitecturas de 8 bits sin FPU (Floating-Point Unit). Las validaciones garantizan equivalencia matemática estricta independientemente de la arquitectura subyacente.

### 1. Línea Base del Modelo (Referencia PyTorch FP32)
*Modelo Feed-Forward (MLP) entrenado con el conjunto de datos Digits de Scikit-Learn (64 características), utilizado como "Profesor" para la destilación del modelo binario.*
* **Topología FP32:** 64 → 128 → 10
* **Parámetros Totales:** 37,888
* **Tamaño en RAM/Disco:** ~148 KB
* **Precisión Base (Test Set, 20% split):** 94.17%

---

### 2. Perfil de Rendimiento en 32-bits (ESP32 - Xtensa, 240 MHz)
*Red binarizada mediante destilación de conocimiento (Knowledge Distillation) y Block-Gating Dinámico.*
* **Topología Binaria ("Estudiante"):** 64 → 512 → 10 (Expansión de capacidad 4x respecto a la línea base)

| Métrica | Valor |
|---|---|
| Precisión de Inferencia | **95.28%** (343/360 aciertos en Test Set)* |
| Latencia de Inferencia | **~645 µs** (Media sobre 360 ejecuciones continuas) |
| Tamaño de Pesos (ROM/Flash) | **~4.8 KB** (Tasa de compresión de ~31x vs FP32) |
| Fidelidad Algorítmica | **100% de acuerdo con simulador binario** en Python (ver Notas) |
| Motor Base | XNOR + Popcount SWAR (32-bit nativo) |

---

### 3. Perfil de Restricción Extrema en 8-bits (Arduino Uno - ATmega328P, 16 MHz)
*Misma topología binaria operando dentro de las limitaciones de 2KB de SRAM, probando la portabilidad del motor.*

| Métrica | Valor |
|---|---|
| Precisión de Inferencia | **95.28%** (343/360 aciertos en Test Set) |
| Latencia de Inferencia | **~18.3 ms** (Media sobre 360 ejecuciones continuas) |
| Consumo de ROM (Flash) | **~4.8 KB** de pesos (El binario final ocupa 8.7 KB / 27% del total) |
| Consumo de SRAM (Dinámica)| **311 bytes globales** + picos locales de 72 bytes (buffers) |
| Fidelidad Algorítmica | **100% de acuerdo con simulador binario** en Python |
| Motor Base | XNOR seguro (255-XOR) + LUT precalculada (256 bytes en PROGMEM) |

---

### 4. Metodología de Pruebas y Notas Técnicas

*   **Superación de la Línea Base:** El modelo binarizado alcanza un 95.28% frente al 94.17% de la línea base en FP32. Esto no es inherente a la binarización, sino a la arquitectura: el modelo binario "Estudiante" posee una capa oculta de 512 neuronas (frente a las 128 del "Profesor" FP32). La mejora es resultado de esta expansión de capacidad (para compensar la pérdida de precisión numérica) combinada con el proceso de destilación (Knowledge Distillation) que regula el entrenamiento. Ambos modelos fueron evaluados sobre la misma partición estratificada de 360 muestras estáticas.
*   **Fidelidad Bit-a-Bit:** El término "100% de fidelidad" no compara el modelo FP32 original contra el hardware. Define que la ejecución en C++ produce exactamente los mismos acumuladores internos, los mismos logits y la misma clase de salida que el simulador del modelo binario ejecutado en Python utilizando matrices NumPy. 
*   **Determinismo Multi-Arquitectura:** Las salidas y acumulaciones intermedias generadas por el ESP32 (aritmética de 32 bits, popcount) y el Arduino Uno (aritmética de 8 bits, LUT de 256 bytes) son bit a bit idénticas. El framework abstrae las diferencias de *endianness* y promoción de enteros (`Integer Promotion`) para garantizar la equivalencia.
*   **Consumo de SRAM:** En el perfil del ATmega328P, la memoria ROM almacena los pesos. En la SRAM (2 KB disponibles), las variables globales consumen 311 bytes, dejando 1737 bytes libres. Durante la inferencia, la huella dinámica local es estrictamente de 72 bytes adicionales (8 bytes para el buffer de entrada empaquetado + 64 bytes para el buffer de activaciones ocultas de 512 bits).
*   **Medición de Latencia:** La latencia reportada abarca exclusivamente la ejecución computacional (`forward` y `argmax`) cronometrada en hardware mediante temporizadores nativos (`micros()`) durante un ciclo ininterrumpido de 360 inferencias. Se excluye del tiempo reportado el overhead de transmisión del puerto Serial y la lectura de datos inicial.

> **Nota de Rendimiento (AVR):** A diferencia de las arquitecturas de 32 bits, el ATmega328P carece de instrucciones nativas para contar bits. Para superar esta limitación física y evitar los bugs catastróficos de promoción de enteros de GCC (que corrompen la matemática binaria), SYMBEX-1 inyecta un motor de hardware virtual basado en una tabla precalculada de 256 bytes en la memoria Flash. Esto permite procesar 512 neuronas binarizadas en tan solo 18 milisegundos de forma invulnerable.

## Créditos
Carlos Duarte

Inspirado en la transición de redes densas hacia enfoques dispersos (como *Mixture of Experts*) y QAT, llevado al extremo del silicio embebido.
