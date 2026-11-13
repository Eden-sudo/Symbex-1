<p align="center">
  <img src="docs/assets/symbex-1-banner.jpg" alt="SYMBEX-1, redes neuronales binarias para microcontroladores" width="100%">
</p>

<p align="center"><b>Redes neuronales binarias para microcontroladores.</b></p>

<p align="center">
  <a href="https://github.com/Eden-sudo/Symbex-1/releases"><img src="https://img.shields.io/github/v/release/Eden-sudo/Symbex-1?color=14b8a6&label=release" alt="release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Eden-sudo/Symbex-1?color=0d9488&label=license" alt="license"></a>
  <a href="https://symbex-1.pages.dev/"><img src="https://img.shields.io/badge/sitio-symbex--1.pages.dev-0f766e" alt="sitio"></a>
  <a href="benchmarks/"><img src="https://img.shields.io/badge/chips-ESP32%20%7C%20ESP32--S3%20%7C%20RP2350%20%7C%20AVR-115e59" alt="chips"></a>
  <a href="https://github.com/Eden-sudo/Symbex-1/pulls"><img src="https://img.shields.io/badge/PRs-welcome-f97316" alt="PRs"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/read-in%20English-334155" alt="read"></a>
</p>

SYMBEX-1 compila redes neuronales en **redes completamente binarias que corren en microcontroladores pequeños**, y las
mide en chips reales contra la herramienta oficial de INT8 (TensorFlow Lite Micro).

Tiene tres partes:

- **`lib_symbex`**: un motor de inferencia en C, portable, para redes binarias: conteo de bits en vez de
  multiplicaciones, umbrales enteros, sin punto flotante. Una capa de hardware por familia de chips (genérico, ESP32,
  ESP32-S3, AVR, RP2350).
- **El CDT** (`tools/cdt.py`): toma un modelo, tus datos y los límites de tu hardware, entrena la red binaria, la
  pliega a enteros, escribe su header en C y la verifica bit a bit, en la PC y en el chip.
- **`benchmarks/`**: comparaciones reproducibles contra INT8 de TensorFlow Lite Micro en los mismos chips.

## Resultados (medidos en los chips, un núcleo)

Aquí, INT8 es la ruta de despliegue que usa casi todo el mundo: TensorFlow Lite Micro con los kernels optimizados
ESP-NN de Espressif. No es el INT8 más rápido posible: en el ESP32-S3, la librería ESP-DL de Espressif ejecuta
ResNet-8 en 14.8 ms, 2.6× más rápido que el motor estándar de SYMBEX. El [motor vertical](docs/vertical.es.md) de
SYMBEX lo ejecuta en 10.4 ms, 1.43× más rápido que ESP-DL con los pesos en flash; con su modo compacto opcional, 7.70 ms
con 151 KB de RAM, también es 1.27× más rápido que ESP-DL con sus pesos en RAM (ahí usa 147 KB), y 6.44 ms con el modo
preparado (238 KB), mientras que en el ESP32 clásico SYMBEX sigue siendo 8-11× más rápido
que ESP-DL también. [docs/results.es.md](docs/results.es.md#qué-tan-rápido-podría-ser-int8) tiene los números y lo que falta medir.

### Clasificación de imágenes: una ResNet-8 completamente binaria ([`benchmarks/conv2d/cnn_cifar_resnet8`](benchmarks/conv2d/cnn_cifar_resnet8/))

CIFAR-10, contra la ResNet-8 INT8 oficial de MLPerf Tiny. Todas las capas de la red de SYMBEX son binarias, la primera
incluida (la imagen entra como un termómetro de bits); 3 semillas de entrenamiento, solo etiquetas.

| | **SYMBEX, binaria** | MLPerf Tiny INT8 |
|---|---|---|
| Precisión | 71.28 ± 0.81 % | 87.0 % |
| ESP32-S3 | **39.2 ms (1.33× más rápida)** | 52.0 ms |
| ESP32 | **59.7 ms (12.0× más rápida)**, pesos en RAM; 63.3 ms en flash | 717.8 ms |
| Raspberry Pi Pico 2, Cortex-M33 | **74.5 ms (6.1× más rápido)** | 453.0 ms (CMSIS-NN) |
| Raspberry Pi Pico 2, Hazard3 RISC-V | **56.1 ms** | 5702 ms (kernels de referencia, los únicos que TFLite Micro tiene para él) |
| Flash | **24.3 KB** | 96.2 KB |
| RAM | **15 KB** | 53-80 KB |
| Mismo resultado que su referencia en la PC | **100 %** | 98.1 % |

**La receta más precisa.** Con una primera capa int8 sobre los píxeles y un clasificador int8 con pooling (todas las
demás capas binarias, Bop para los pesos binarios; `--variant int8stem_pool2` en el CDT), la misma ResNet-8 llega a
**76.7 ± 0.3 %** y corre más rápido: ESP32-S3 32.3 ms (1.61×), 6.0-9.7 ms con el motor vertical; ESP32 62.5 ms
(11.5×); Pico 2 M33 67.8 ms (6.7×), Hazard3 57.9 ms; 29 KB de flash, exacta en todas las placas. La semilla 42 está
lista para usar en [`reference/`](benchmarks/conv2d/cnn_cifar_resnet8/reference/README.md).

**Dimensionada para binario.** La mejor forma de una red binaria no es la de INT8: un estudio de dimensionamiento
(bloques y canales por etapa, latencia predicha con perfiles medidos en el chip) encontró que las primeras etapas pueden
ser más delgadas y la última más ancha. Medido en el chip, 3 semillas (en el ESP32 la ganancia es precisión a la misma
velocidad; la mejor forma depende del chip):

| Red | Precisión | ESP32-S3 | ESP32 | RAM |
|---|---|---|---|---|
| Forma de la ResNet-8 (16-32-64, bloques 2-2-2) | 71.28 ± 0.81 % | 39.2 ms | 59.7 ms (pesos en RAM) | 15 KB |
| **16-32-128, bloques 1-1-2** | **73.10 ± 0.23 %** | **39.6 ms** | 60.9 ms (pesos en RAM) | 15 KB |
| **16-64-128, bloques 1-1-2** | **74.45 ± 0.23 %** | 46.6 ms | | 16 KB |

### Detección de personas: FOMO, binario contra el mismo detector en INT8 ([`benchmarks/conv2d/fomo_voc_person`](benchmarks/conv2d/fomo_voc_person/))

Personas de Pascal VOC, imágenes de 96 × 96, una cuadrícula de 12 × 12 de centros de personas. La referencia INT8 es
**la misma arquitectura** entrenada con la misma receta.

| | **SYMBEX, binario** | TFLite INT8 |
|---|---|---|
| F1 (3 semillas) | **31.2 ± 0.7 %** | 30.1 ± 0.2 % |
| ESP32-S3 | **126.3 ms (2.0× más rápido)** | 256.9 ms |
| ESP32 | **corre** | no cabe (su arena desborda la RAM) |
| Flash | **41 KB** | 236 KB |
| RAM | **51 KB** | 223 KB |
| Mismo resultado que su referencia en la PC | **100 %** | 88 % |

La detección se binariza bien: cada celda responde una pregunta de sí o no, y el detector binario es tan preciso como
el INT8.

### Redes densas ([`benchmarks/mlp/mlp_fashion`](benchmarks/mlp/mlp_fashion/), [`benchmarks/mlp/mlp_digits`](benchmarks/mlp/mlp_digits/))

Fashion-MNIST, dimensionada para binario (784-128-512-10: una primera capa angosta y capas binarias anchas), contra
INT8 de la misma red; 3 semillas, solo etiquetas, pesos en flash salvo que se indique:

| | **SYMBEX, binaria** | INT8 (misma red) |
|---|---|---|
| Precisión | 87.47 ± 0.09 % | 89.81 ± 0.11 % (QAT) |
| ESP32-S3 | **0.42 ms (28× más rápida)** | 11.7 ms (TFLite Micro) |
| ESP32-S3, pesos en RAM | **0.41 ms (2.1× más rápida)** | 0.89 ms |
| ESP32 | **3.5 ms (8.8× más rápida)** | 31.1 ms |
| Raspberry Pi Pico 2, Cortex-M33 | **4.6 ms (6.9× más rápido)**; 1.5 ms con los pesos en RAM (1.9×) | 31.4 ms (CMSIS-NN); 2.9 ms en RAM |
| Raspberry Pi Pico 2, Hazard3 | **4.4 ms (8.5× más rápido)**; 1.4 ms con los pesos en RAM (6.8×) | 37.5 ms; 9.6 ms en RAM (kernels de referencia) |
| Flash | **27 KB** | 187 KB |
| RAM | **4.6 KB** | 6.7 KB |

INT8 es 2.3 puntos más precisa; SYMBEX gana en velocidad y memoria. En el MLP más chico (Digits) los dos empatan en el
ESP32-S3 (0.109 contra 0.110 ms), SYMBEX es 2-3.3× más rápido en el ESP32, y es el único que corre en un Arduino Uno.

### Reconocimiento de actividad: convoluciones 1D binarias ([`benchmarks/conv1d/har_ultracompact`](benchmarks/conv1d/har_ultracompact/))

UniMiB-SHAR (acelerómetro, 17 actividades), las cuatro redes del trabajo previo más cercano (Daghero et al. 2021,
ultracompactBNN), 3 semillas:

- **Contra su librería binaria, en su propio entorno RISC-V simulado (GVSoC):** SYMBEX necesita **1.07-1.37× menos
  instrucciones** en las cuatro redes, también con sus propios pesos entrenados.
- **Contra INT8 de TFLite Micro, mismas arquitecturas, pesos en flash:** **5.3-10.4× más rápido** en el ESP32 y
  **1.47-1.90× más rápido** en el ESP32-S3, con 2-10× menos RAM y 6.5-10× menos flash. INT8 es más preciso en estas
  redes pequeñas (unos 9 puntos en las grandes).
- **Dónde deja de convenir lo completamente binario (una teoría de trabajo):** con una entrada de 8 bits como la del
  acelerómetro, una primera capa con pesos int8 es más rápida y más precisa que una binaria en todas las placas medidas.

### Por qué es más rápido: contar bits contra multiplicar

`tools/dev/count_bench.py` mide la aritmética pura de cada chip con el mismo patrón de convolución: binaria (conteo de
bits) contra int8 (multiplicación-suma).

| Chip | Binario contra int8, aritmética pura |
|---|---|
| ESP32 (sin SIMD) | **11-17× más rápido** |
| ESP32-S3 (SIMD de 8 bits, sin instrucción de popcount) | 1.05-2.1× más rápido (el multiplicador SIMD cuenta los bits: `pie_swar`) |
| x86 (AVX2 / VNNI) | 1.5-3.8× |

## Cómo funciona

1. **Todas las capas son binarias.** Pesos y activaciones son de un bit. Una convolución es un conteo de bits (AND o
   XOR seguido de popcount); un atajo residual es un OR de bits; el pooling es un OR.
2. **La entrada también se vuelve bits.** Cada píxel entra como un termómetro (8 niveles por color), así que la primera
   capa es binaria como las demás.
3. **Todo lo demás se pliega en enteros.** La escala, la BatchNorm y la activación se reducen a un umbral entero por
   salida: `bit = (count >= T)`. La inferencia no usa punto flotante.
4. **El método de conteo se elige por chip.** Una instrucción nativa de popcount donde la hay, una tabla de 64 KB o SWAR
   en el ESP32, el multiplicador SIMD en el ESP32-S3. `count_bench.py` los mide; la librería elige el más rápido.
5. **Exacto bit a bit, siempre.** El C compilado, en la PC y en el chip, tiene que reproducir el 100 % de la simulación
   entera del modelo: se verifica en cada modelo exportado, siempre.

## Metodología

El estándar completo, las referencias y cómo repetir un benchmark: [docs/methodology.es.md](docs/methodology.es.md). En resumen:

- **La misma arquitectura** siempre que SYMBEX se compara con INT8 (el modelo oficial de MLPerf, o la misma red
  entrenada con la misma receta); una red binaria redimensionada se compara con INT8 redimensionado igual.
- **Una referencia independiente:** INT8 con el convertidor oficial de TensorFlow Lite y TensorFlow Lite Micro, en un
  entorno aparte que no importa nada de SYMBEX.
- **Medido en el chip:** medianas de 1000 inferencias, un núcleo, a la frecuencia de fábrica (240 MHz en los ESP32,
  150 MHz en el Pico 2); la RAM se mide en el chip.
- **Varias semillas** para SYMBEX: la precisión, como media ± desviación estándar.
- **Split de validación:** cada decisión (checkpoint, umbral, variante, forma) se toma con un split de validación sacado
  de los datos de entrenamiento; el test se evalúa una sola vez. Una revisión encontró lugares donde se había usado el
  test para elegir; se corrigieron y las cifras afectadas se volvieron a medir (octubre de 2026).

## Estado actual y próximos pasos

Hecho en octubre de 2026: convoluciones 1D (contra Daghero et al. 2021, en GVSoC y en placas reales), la Raspberry Pi
Pico 2 (los dos núcleos, Arm Cortex-M33 y Hazard3 RISC-V, exactos), y todos los benchmarks vueltos a medir con el motor
actual y, todos menos FOMO, con el split de validación corregido. Lo siguiente:

- **La aritmética de la primera capa elegida por chip** (bits o bytes), a partir de mediciones, por el CDT.
- **La energía** todavía no se ha medido; la latencia y la memoria sí.
- **FOMO reentrenado** con el nuevo split de validación fijo.

Los dos objetivos principales del proyecto, con sus responsables (el detalle y cada tarea abierta están en
[docs/roadmap.md](docs/roadmap.md), en inglés; lo hecho, PR por PR, en [UPDATES.md](UPDATES.md)):

- **Un framework fácil** (Edén, Obed): una librería de PlatformIO que lleve un modelo entrenado a un ESP32, una placa
  tipo Arduino o un RP2350 en pocos pasos.
- **Visión con FOMO** (Abrahan): más clases y una cámara en la placa con el detector corriendo de punta a punta.

**Línea de investigación: modelos de lenguaje en microcontroladores** (Obed, Edén): si un modelo de 1 bit puede seguir
instrucciones cortas de forma confiable en un solo chip. Aún no se presentan resultados propios; ver la [hoja de ruta](docs/roadmap.md).

## Estructura del repositorio

```
Symbex1_PIO/               proyecto de PlatformIO
  lib/lib_symbex/          el motor en C (include/, src/, hal/) y tests/run_tests.py (pruebas de exactitud, todas las HAL)
  src/main.cpp             ejemplo mínimo: una red de demostración verificada contra su simulación en la PC, en tu placa
tools/
  cdt.py                   el CDT: configuración (TOML) -> entrenar -> exportar -> validar -> chip
  chip_test.py             corre un modelo exportado en una placa: exactitud, precisión, latencia por capa, RAM
  compiler_core/           la librería detrás del CDT
  configs/                 example.toml: todas las opciones, comentadas
  lowbit/                  referencias INT8 / INT4 / INT2 (Brevitas, kernels de ESP-NN) para cualquier benchmark
  dev/                     herramientas de desarrollo: fine_profile.py (tiempo por función de la librería),
                           count_bench.py (aritmética pura de cada chip), train_bench.py
benchmarks/                comparaciones reproducibles contra TensorFlow Lite Micro: mlp/, conv1d/, conv2d/, experiments/
tests/                     pruebas de Python (pytest -q desde la raíz)
docs/                      documentación (inglés y español)
```

## Cómo empezar

```
# la librería en tu placa (PlatformIO): el ejemplo tiene que imprimir "8/8 equal to the PC simulation"
cd Symbex1_PIO && pio run -t upload && pio device monitor -b 115200

# las pruebas de la librería (gcc)
python Symbex1_PIO/lib/lib_symbex/tests/run_tests.py

# un modelo con el CDT (desde tools/), y después en una placa
python cdt.py mi_modelo.toml
python chip_test.py <carpeta de salida> --board esp32 --port /dev/ttyUSB0
```

Requisitos: Python con PyTorch; gcc; PlatformIO y pyserial para el chip; TensorFlow para las referencias INT8.

## Versiones anteriores

La versión 2 reportaba resultados en el conjunto Digits de scikit-learn con los métodos `gated` y `bitslice`. Durante la
revisión de la versión 3 se encontraron errores en esos métodos; esos resultados no se reportan hasta que los métodos se
reconstruyan y se vuelvan a medir. El primer motor convolucional de la versión 3 (con una primera capa int8 aprendida)
fue reemplazado por el motor completamente binario de arriba.

## Cómo citarlo

Hay un preprint de SYMBEX-1 en preparación; la referencia se agregará aquí.

## Colaboradores

Carlos Duarte (Edén) -- Panamá -- GitHub: [Eden-sudo](https://github.com/Eden-sudo)
Obed -- GitHub: [Obed0101](https://github.com/Obed0101) -- suite de pruebas, referencias de bajo número de bits, revisión del split de validación
Abrahan -- GitHub: [abrahandejesusaparicio-code](https://github.com/abrahandejesusaparicio-code) -- detección de personas con FOMO en el ESP32 clásico

Versión en inglés: [README.md](README.md) -- Licencia: ver [LICENSE](LICENSE).
