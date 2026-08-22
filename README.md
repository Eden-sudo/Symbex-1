# SYMBEX-1

Motor de inferencia para redes neuronales cuantizadas a 3 bits, diseñado para correr en microcontroladores de 8 bits sin FPU (ej. Arduino Uno / ATmega328P). Usa un esquema de **bit-slicing con XNOR-popcount** para el grueso de los pesos y un **canal paralelo de outliers** para preservar los pesos de magnitud crítica que la cuantización agresiva normalmente destruye.

## Por qué existe esto

Cuantizar una red a pocos bits con un simple min-max destruye la señal de dos formas opuestas, dependiendo de cómo se maneje la dispersión de los pesos:

- **Min-max puro:** un solo peso de magnitud grande fuerza la escala de cuantización, aplastando al resto de los pesos "normales" hacia el nivel cero.
- **Clipping estadístico ciego:** recortar los pesos grandes salva a la mayoría, pero amputa los outliers — y en redes más grandes, esos outliers suelen ser las "características maestras" que la red usa para decisiones críticas (fenómeno documentado en [LLM.int8()](https://arxiv.org/abs/2208.07339), Dettmers et al.).

SYMBEX-1 resuelve esto separando cada capa en dos caminos: un núcleo cuantizado a 3 bits (la "masa aburrida", ~95-99% de los pesos) y un canal disperso de alta precisión para los outliers, sumados al final — la misma idea de fondo que LLM.int8(), adaptada a un microcontrolador de 8 bits sin punto flotante.

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│  Entrenamiento (Python / PyTorch)                        │
│  Teacher FP32  →  QAT Estudiante (3-bit + outliers)      │
│  ↓                                                       │
│  Validación bit-a-bit (simulador NumPy = firmware C++)   │
│  ↓                                                       │
│  Exportador → symbex_weights.h (bit-planes en PROGMEM)   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Inferencia (C++ / Arduino)                             │
│  SymbexNetwork → SymbexLayer → XNOR-popcount + outliers │
└─────────────────────────────────────────────────────────┘
```

### Bit-slicing con XNOR-popcount

Cada peso se cuantiza a un entero de 3 bits (`k_bits=3`, niveles 0-7), almacenado como **3 planos de bits binarios** (`weight_bit0/1/2`, de MSB a LSB) en vez de un array de bytes. La inferencia reconstruye la contribución de cada plano correlacionando el plano contra el input vía **XNOR + popcount** — una operación bit-a-bit nativa en cualquier CPU, sin multiplicaciones:

```
bit_i_contrib = (2 * popcount(input XNOR peso_bit_i)) - 8   // mapeo a bipolar [-8, 8]
acumulador += 4 * bit0_contrib + 2 * bit1_contrib + 1 * bit2_contrib
```

Esto reconstruye exactamente `2*W_quant - 7`, una transformación afín del valor cuantizado — matemáticamente equivalente a una multiplicación entera, pero implementada solo con lógica de bits.

### Canal de outliers

Los pesos que exceden `2·std` de la media se excluyen del núcleo cuantizado y se manejan aparte:
- Un **bitmask** (`outliers`) marca qué posiciones son outliers.
- Una **magnitud por neurona** (`outlier_magnitudes`, `int8_t`) cuantifica su tamaño promedio.
- El **signo** se deriva del MSB del núcleo cuantizado (`weight_bit0`), sin necesidad de guardarlo aparte.

En inferencia, la contribución de outlier se calcula solo en las posiciones marcadas por el bitmask, correlacionando el signo (vía XNOR contra `weight_bit0`) con el input:
```
match = (input XNOR weight_bit0) AND outlier_mask
outl_contrib = (2 * popcount(match)) - popcount(outlier_mask)
acumulador += outl_contrib * outlier_magnitude
```

### Expansión por voting pool (factor M)

Cada capa puede tener `M` sub-capas cuantizadas independientes ("voting pool") cuyos votos se suman. Esto añade redundancia que compensa el ruido de cuantización, a costa de `M`x el tamaño en flash.

## Entrenamiento (QAT)

El compilador (`tools/symbex_compiler.py`) entrena vía **destilación de conocimiento** desde un teacher FP32, con el forward pass del estudiante simulando exactamente el pipeline de cuantización que ejecutará el hardware (clip → normalizar → cuantizar con STE → outliers con STE), para que la red aprenda ya adaptada al error de cuantización real, no a una aproximación.

Componentes clave del entrenamiento estable:
- **STE (Straight-Through Estimator)** en la cuantización del núcleo y en la magnitud de outliers, para que el gradiente fluya a través de operaciones no diferenciables (`round`).
- **EMA (media móvil exponencial)** para `mean`/`std`/`W_max` de cada capa — sin esto, la grilla de cuantización se recalcula desde cero en cada forward pass y se mueve junto con el gradiente que la persigue, produciendo un loss que oscila sin converger.
- **Warm start**: los pesos del estudiante se inicializan copiando los del teacher (+ ruido pequeño), evitando arrancar desde una configuración aleatoria que la cuantización agresiva podría no recuperar.
- **Escalar de salida aprendible** (`output_scale`, entrenable, no cuantizado): la suma de `M` sub-capas de pesos en `±7` produce logits en un rango de magnitud muy distinto al del teacher (crece con `√(in_features · M)`), lo que satura el softmax de la loss de destilación. Este escalar corrige la magnitud durante el entrenamiento sin afectar el hardware, porque `argmax` es invariante a un escalar positivo constante — por eso el exportador lo ignora.
- **Loss híbrida** (`KLDivLoss` con temperatura + `CrossEntropyLoss` directa contra las etiquetas), para anclar el estudiante a las etiquetas reales además de imitar al teacher.

### Validación antes de exportar

`validate_before_export` compara la salida de PyTorch contra un simulador en NumPy que replica bit a bit la misma cuantización, outliers y reconstrucción que ejecutará el C++. Si el acuerdo cae debajo de 98%, el compilador **aborta la exportación** en vez de generar un `.h` que no coincida con lo que se validó.

> Esta validación confirma que Python es consistente consigo mismo — no reemplaza probar en hardware real. Ver la sección de resultados: hubo bugs en el C++ que el validador de Python no podía detectar por diseño.

## Estructura del repositorio

```
tools/
  symbex_compiler.py     # Entrenamiento QAT + validación + exportador a .h
  hardware_tester.py     # Benchmark de precisión/latencia sobre Arduino real vía serial

lib_symbex/
  src/
    SymbexLayer.h/.cpp    # Núcleo de inferencia: XNOR-popcount + outliers
    SymbexNetwork.h/.cpp  # Orquestador multi-capa (ping-pong buffers)
    symbex_config.h       # Macros de lectura de memoria portables (AVR/ESP32/ARM/PC)
    symbex_weights.h      # Pesos exportados (generado, no editar a mano)
  examples/
    SymbexBenchmark/      # Sketch de Arduino para el benchmark serial

archive/                  # Prototipos y experimentos anteriores, fuera de la rama activa
```

## Uso

**Entrenar y exportar:**
```bash
python tools/symbex_compiler.py --epochs 150 --expansion 4 --k_bits 3 --out_dir lib_symbex/src
```

**Compilar y flashear el benchmark (Arduino Uno):**
```bash
arduino-cli compile --fqbn arduino:avr:uno lib_symbex/examples/SymbexBenchmark
arduino-cli upload --fqbn arduino:avr:uno -p /dev/ttyUSB0 lib_symbex/examples/SymbexBenchmark
```

**Correr el benchmark de precisión/latencia en hardware real:**
```bash
python tools/hardware_tester.py
```

## Resultados de referencia

Red de ejemplo: `64 → 128 → 8` (clasificación de dígitos, UCI Optical Digits), `k_bits=3`, `M=4`.

| Métrica | Valor |
|---|---|
| Precisión Teacher (FP32, test set) | 96.89% |
| Precisión Estudiante (3-bit, validado en Python, test set) | 96.19% |
| Precisión en hardware (Arduino Uno) | 99.5% (199/200)* |
| Fidelidad bit-a-bit Python↔NumPy | 100% |
| Tamaño original (FP32) | ~36.9 KB |
| Tamaño exportado (3-bit + outliers, M=4) | ~19.5 KB |
| Compresión | ~1.9x |
| Latencia por inferencia | ~102.3 ms (~9.8 Hz) |
| Placa | Arduino Uno (ATmega328P, 16 MHz, sin FPU) |

\* Corrido sobre el dataset completo, no aislado al split de test — no es directamente comparable al 96.19%, que sí es sobre datos no vistos.

**Nota sobre compresión:** con `M=4`, cada capa guarda 4 copias completas del núcleo cuantizado — la ganancia teórica de 3 bits (~10.7x) se ve reducida a ~1.9x por esa redundancia. Con `M=1`, la compresión estimada sube a ~7.6x, a costa de precisión. El factor `M` es actualmente fijo por argumento de línea de comandos; una estimación automática por capa (basada en error de cuantización medido, no en varianza de pesos) es un trabajo pendiente.

## Estado del proyecto

Funcional y validado en hardware real. Pendientes conocidos:
- Estimación automática de `M` por capa basada en error de cuantización real (la heurística de varianza actual subestima sistemáticamente).
- Exploración de `k_bits=2` para mayor compresión.
- Guard explícito para dimensiones de entrada no múltiplos de 8 (bits de padding).
- `hardware_tester.py`: filtrar por el split de test real para reportar una precisión directamente comparable a la de entrenamiento.

## Créditos

Arquitectura y desarrollo: Carlos Duarte
Documentacion y Redaccion: Bryan Mendez
Inspirado conceptualmente en el manejo de outliers de [LLM.int8()](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022), adaptado a un esquema de cuantización entera fija para microcontroladores sin FPU.  
