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

**SYMBEX-1 convierte redes neuronales en redes completamente binarias, un bit por peso y por activación, que corren
en microcontroladores pequeños.** Las mide en chips reales contra la herramienta oficial de INT8, TensorFlow Lite Micro,
con la misma red en el mismo chip, y cada modelo compilado da la misma respuesta que su simulación en el 100 % de las
entradas.

## ¿Por qué binario?

Un microcontrolador tiene unos cientos de KB de RAM, unos pocos MB de flash y muchas veces ni SIMD ni unidad de punto
flotante. Una red INT8 multiplica y suma un byte por cada peso. Una red binaria guarda **un bit** por peso y convierte
cada multiplicación en una operación de bits y un conteo:

- **8× menos memoria para los pesos que INT8** (32× menos que FP32): ResNet-8 necesita 24 KB de flash en vez de 96 KB.
- **Contar en vez de multiplicar.** De 32 a 128 pesos a la vez: XOR o AND, y después un conteo de bits. En un chip sin
  SIMD, como el ESP32 clásico, esa aritmética es 11-17× más barata que la multiplicación-suma int8 (medido con
  `tools/dev/count_bench.py`).
- **Nada de punto flotante.** La escala, la BatchNorm y la activación se pliegan en un umbral entero por salida:
  `bit = (conteo >= T)`.

> [!IMPORTANT]
> **Las redes binarias no reemplazan a INT8.** Un bit por peso cuesta precisión, y más cuanto más fina es la tarea.
> SYMBEX cambia esa precisión por **velocidad y recursos**: sirve cuando la latencia, la memoria o un chip muy pequeño
> importan más que los últimos puntos de precisión. Las gráficas de abajo muestran exactamente cuánto cede cada lado.

| Lo binario conviene | INT8 es la mejor opción |
|---|---|
| El chip no tiene SIMD (ESP32 clásico, RISC-V pequeños, AVR) | El chip tiene una unidad SIMD int8 potente y RAM de sobra |
| El modelo INT8 no cabe en el chip | Cada punto de precisión importa |
| Preguntas de sí o no: detección, presencia, activación | Muchas clases que solo difieren en detalles finos (como CIFAR-10) |
| Un presupuesto de latencia estricto, o muchas inferencias por segundo | |

## Lo que medimos

### ResNet-8: velocidad a cambio de precisión

Cada punto es una red: más arriba es más precisa, más a la izquierda es más rápida. La latencia se mide en el chip; la
precisión es la del modelo en las 10,000 imágenes de prueba, en la PC (en el chip, SYMBEX da la clase de su simulación
en el 100 % de las imágenes, TFLite Micro la del intérprete de la PC en el 98.1 %). La línea une los puntos que ningún
otro supera en los dos ejes a la vez; "vertical" es el motor de SYMBEX para la unidad SIMD del ESP32-S3.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pareto_resnet8_dark.svg">
  <img alt="ResNet-8 en CIFAR-10, precisión contra latencia en el ESP32-S3 y el ESP32: SYMBEX contra INT8" src="docs/assets/pareto_resnet8.svg">
</picture>

- **INT8 es la más precisa:** el modelo oficial de MLPerf Tiny llega a 87.0 % en CIFAR-10. La ResNet-8 completamente
  binaria llega a 71.3 %, y a 76.7 % con la primera capa y el clasificador en 8 bits: de 10 a 16 puntos menos.
- **La binaria es mucho más rápida:** en el ESP32-S3, 9.70 ms contra 52.0 ms (5.4×) con el [motor vertical](docs/es/vertical.md);
  en el ESP32 clásico, 62.5 ms contra 717.8 ms (11.5×); en la Raspberry Pi Pico 2 (Cortex-M33), 67.8 ms contra 453 ms
  con CMSIS-NN (6.7×).
- **Y mucho más pequeña:** 24-29 KB de flash contra 96 KB.

TFLite Micro es la ruta INT8 que usa casi todo el mundo, no el INT8 más rápido posible: ESP-DL de Espressif ejecuta
ResNet-8 en 14.8 ms en el ESP32-S3 (una prueba rápida, no está en la gráfica). El motor vertical sigue siendo más
rápido ahí con los pesos en flash; [Resultados](docs/es/results.md#qué-tan-rápido-podría-ser-int8) tiene la comparación y
lo que falta medir.

### El MLP: un bit contra INT8, INT4 e INT2

La misma receta para todos los métodos (split de validación, 3 semillas), Fashion-MNIST en el ESP32-S3:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pareto_mlp_dark.svg">
  <img alt="MLP en Fashion-MNIST, precisión contra latencia en el ESP32-S3: SYMBEX contra INT8 / INT4 / INT2" src="docs/assets/pareto_mlp.svg">
</picture>

Los enteros de pocos bits son 1-2 puntos más precisos; un bit se queda con la esquina más rápida. Con los pesos en
flash, donde leerlos es el costo, lo binario es 24-37× más rápido (0.42 ms contra 10.3-15.6 ms); con los pesos en RAM
la diferencia baja a 1.4× (0.41 contra 0.56 ms).

### Todos los benchmarks, lado a lado

La misma red en el mismo chip, binaria contra INT8 de TFLite Micro:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/speedup_dark.svg">
  <img alt="Aceleración de SYMBEX sobre INT8 de TensorFlow Lite Micro por benchmark y chip" src="docs/assets/speedup.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/flash_dark.svg">
  <img alt="Flash que usa cada modelo, SYMBEX contra INT8" src="docs/assets/flash.svg">
</picture>

| Benchmark | Precisión, SYMBEX / INT8 | Velocidad | Memoria |
|---|---|---|---|
| [ResNet-8](benchmarks/conv2d/cnn_cifar_resnet8/), imágenes CIFAR-10 | 76.7 / 87.0 % | de 5.4× (ESP32-S3) a 11.5× (ESP32) más rápida | 29 / 96 KB de flash |
| [FOMO](benchmarks/conv2d/fomo_voc_person/), detección de personas | **F1 31.2 / 30.1 %** | 2.0× más rápido (ESP32-S3); solo SYMBEX cabe en el ESP32 | 41 / 236 KB de flash |
| [MLP](benchmarks/mlp/mlp_fashion/), Fashion-MNIST | 87.5 / 89.8 % | 28× (flash) y 2.1× (RAM) en el ESP32-S3, 8.8× en el ESP32 | 27 / 187 KB de flash |
| [Convoluciones 1D](benchmarks/conv1d/har_ultracompact/), reconocimiento de actividad | INT8 unos 9 puntos mejor | 5.3-10.4× (ESP32), 1.47-1.90× (ESP32-S3) | 6.5-10× menos flash |
| [Dígitos](benchmarks/mlp/mlp_digits/), en un Arduino Uno | 94.5 % (FP32 con la misma entrada 94.4 %) | 5.3 ms en un AVR de 16 MHz | 3.4 KB de flash, 0.21 KB de RAM |

Contra otra librería binaria, ultracompactBNN (Daghero et al. 2021), en su propio procesador RISC-V simulado, SYMBEX
necesita 1.07-1.37× menos instrucciones en sus cuatro redes. Todas las tablas, chips y semillas:
[docs/es/results.md](docs/es/results.md).

## Lo que aprendimos

1. **Lo binario rinde más donde el chip no tiene SIMD.** En el ESP32 clásico, contar bits es un orden de magnitud más
   barato que multiplicar bytes (5.3-11.5× en redes completas). En el ESP32-S3, cuya unidad SIMD acelera INT8, el
   motor estándar gana por 1.3-2×, y superar a la propia librería INT8 de Espressif necesitó un motor dedicado (vertical).
2. **Lo binario no reemplaza a INT8 en clasificación fina.** En CIFAR-10 incluso la mejor receta binaria queda 10
   puntos atrás. Un [estudio](benchmarks/experiments/training_gap/) encontró que la causa es sobre todo la capacidad, no
   el entrenamiento: la ResNet-8 binaria ajusta el 72 % de sus imágenes de entrenamiento, su gemela de precisión completa
   el 92 %.
3. **La detección y las tareas de sí o no se binarizan bien.** El detector de personas binario es tan preciso como el
   INT8, 2× más rápido en el ESP32-S3, 6× más pequeño y el único que cabe en el ESP32 clásico.
4. **Dimensiona la red para binario.** Su mejor forma no es la de INT8: una primera capa angosta y capas binarias
   anchas (MLP 784-128-512-10), etapas iniciales más delgadas y una última más ancha (ResNet-8 16-32-128: +1.8 puntos a
   la misma velocidad).
5. **Una entrada de 8 bits merece una primera capa de 8 bits.** Para la cámara (ResNet-8) y el acelerómetro (1D), una
   primera capa int8 es más precisa, y más rápida donde el chip tiene SIMD.
6. **Por byte, los enteros de pocos bits son más precisos.** INT2 e INT4 superan a un bit con el mismo tamaño de
   modelo; un bit gana en velocidad y en los chips más pequeños.
7. **Exacto, o no cuenta.** Cada modelo se verifica bit a bit: el C en la PC y en el chip tiene que ser igual a la
   simulación entera, imagen por imagen.

## Cómo funciona

SYMBEX-1 tiene tres partes:

- **`lib_symbex`**: un motor de inferencia portable en C para redes binarias: conteo de bits en vez de multiplicación,
  umbrales enteros, sin punto flotante. Una capa de hardware por familia de chips (genérica, ESP32, ESP32-S3, AVR,
  RP2350).
- **El CDT** (`tools/cdt.py`): toma un modelo, tus datos y los límites de tu hardware, entrena la red binaria, la
  pliega en enteros, escribe su header de C y lo verifica bit a bit, en la PC y en el chip.
- **`benchmarks/`**: comparaciones reproducibles contra INT8 de TensorFlow Lite Micro en los mismos chips.

El motor, paso a paso:

1. **Todas las capas son binarias.** Pesos y activaciones son de un bit. Una convolución es un conteo de bits (AND o
   XOR seguido de popcount); un atajo residual es un OR de bits; el pooling es un OR.
2. **La entrada también se convierte en bits.** Cada píxel entra como un termómetro (8 niveles por color), así que la
   primera capa es binaria como las demás. Como opción, la primera capa y el clasificador quedan en int8 (más precisos,
   ver arriba).
3. **Todo lo demás se pliega en enteros.** La escala, la BatchNorm y la activación se reducen a un umbral entero por
   salida: `bit = (conteo >= T)`. Nada de punto flotante en la inferencia.
4. **El método de conteo se elige por chip.** Una instrucción nativa de popcount donde la hay, una tabla de 64 KB o
   SWAR en el ESP32, el multiplicador SIMD en el ESP32-S3. `count_bench.py` los mide; la librería elige el más rápido.
5. **Exacto bit a bit, siempre.** El C compilado, en la PC y en el chip, tiene que reproducir la simulación entera del
   modelo al 100 %: cada modelo exportado se verifica, siempre.

[docs/es/how-it-works.md](docs/es/how-it-works.md) lo explica con diagramas.

## Metodología

El estándar completo, las referencias y cómo repetir un benchmark: [docs/es/methodology.md](docs/es/methodology.md). En resumen:

- **Misma arquitectura** siempre que SYMBEX se compara con INT8 (el modelo oficial de MLPerf, o la misma red entrenada
  con la misma receta); una red binaria redimensionada se compara con INT8 redimensionado igual.
- **Referencia independiente:** INT8 con el convertidor oficial de TensorFlow Lite y TensorFlow Lite Micro, en un
  entorno aparte que no importa nada de SYMBEX.
- **Medido en el chip:** medianas de 1000 inferencias, un núcleo, a la frecuencia de fábrica (240 MHz en los chips
  ESP32, 150 MHz en la Pico 2); la RAM medida en el chip.
- **Varias semillas** para SYMBEX: la precisión como media ± desviación estándar de 3 semillas.
- **Split de validación:** cada decisión (checkpoint, umbral, variante, forma) se toma en un split de validación sacado
  de los datos de entrenamiento; el conjunto de prueba se evalúa una sola vez. Una revisión encontró lugares donde el
  conjunto de prueba se había usado para elegir; se corrigieron y las cifras afectadas se volvieron a medir (octubre de
  2026).

## Estado actual y próximos pasos

Hecho: convoluciones 2D y 1D completamente binarias, redes densas, cinco chips (ESP32, ESP32-S3, los dos núcleos de la
Raspberry Pi Pico 2, AVR), el motor vertical para el ESP32-S3, y la primera capa y el clasificador int8 como opciones
(noviembre de 2026). Lo siguiente:

- **La energía** por inferencia, con un medidor de corriente: la latencia y la memoria están medidas, la energía todavía no.
- **ESP-DL** como referencia INT8 completa (3 semillas, todo el conjunto de prueba, todos los benchmarks).
- **FOMO reentrenado** con el nuevo split de validación fijo.

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
docs/                      documentación: en/ (inglés), es/ (español), assets/ (gráficas), preprint/
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
