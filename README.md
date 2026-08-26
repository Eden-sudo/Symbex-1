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
│    ├─> 1. Director (Gate): Evalúa K bloques y ordena Top-K  │
│    └─> 2. Músculo (Core): Salto dinámico de bloques apagados│
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
Una capa oculta masiva (ej. 512 neuronas) se subdivide en bloques aislados. El *Gate* binarizado hace una revisión periférica rápida de la entrada, puntúa la relevancia topológica de cada bloque y activa solo los mejores (Top-K). El bucle principal de inferencia lee estas banderas y ejecuta un salto físico (`continue;`) si el bloque no es necesario, evadiendo ciclos de cálculo muertos.

## Estructura Modular del SDK

```text
Symbex1/
├── lib_symbex/           # Motor C++ optimizado, HAL y ejemplos (ESP32/AVR)
├── symbex_v2/            # SDK Python para conversión, destilación y QAT
│   ├── nn.py             
│   ├── distiller.py      
│   └── exporter.py       
├── tools/                
│   ├── train_digits.py   # Compilador principal (FP32 -> Binarizado)
│   └── send_digit.py     # Validador de hardware end-to-end por puerto Serial
└── archive/              # Motores densos, scripts viejos y experimentos de la V1
```

## Resultados de Validación (ESP32 Xtensa)

Prueba empírica realizada sobre el dataset UCI Optical Digits, procesando una expansión dinámica de `64 -> 512 -> 10` (Block Size = 32, K_Active = 8).

| Métrica | Valor |
|---|---|
| Precisión Universal (Hardware real) | **95.28%** (343/360 aciertos en Test Set) |
| Latencia Promedio (Pipeline completo)| **496 µs** (~0.5 milisegundos) |
| Fidelidad bit-a-bit (Python ↔ C++) | 100% (Comportamiento matemáticamente idéntico) |
| Motor Base | XNOR + Popcount SWAR (sin desempaquetado de memoria) |

## Créditos

- **Arquitectura y desarrollo C++/Python:** Carlos Duarte
- **Documentación y diseño conceptual:** Bryan Mendez

Inspirado en la transición de redes densas hacia enfoques dispersos (como *Mixture of Experts*) y QAT, llevado al extremo del silicio embebido.
