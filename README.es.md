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

SYMBEX-1 convierte una red neuronal entrenada en una **red binarizada que corre en microcontroladores pequeños**,
y la mide en chips reales contra la herramienta oficial de INT8.

**La versión 3 logra redes convolucionales binarizadas corriendo en el ESP32**, más rápidas que los modelos INT8
oficiales del benchmark MLPerf Tiny, y un compilador (el CDT) que lleva cualquier modelo FP32 desde el
entrenamiento hasta un modelo binario validado en el chip.

Tiene tres partes:

- **`lib_symbex`**: un motor de inferencia en C, portable, para capas binarias (XNOR + popcount, umbrales enteros).
- **El CDT** (`tools/cdt.py`): toma un modelo FP32, tus datos y los límites de tu hardware, y produce un modelo
  binarizado, su header en C y una validación exacta bit a bit, listo para el chip.
- **`benchmarks/`**: comparaciones reproducibles contra INT8 de TensorFlow Lite / TensorFlow Lite Micro, partiendo
  del mismo modelo FP32, con las mismas imágenes y en el mismo chip.

## Por qué redes binarias

Un peso de 1 bit ocupa **8 veces menos** que uno INT8 y 32 veces menos que uno FP32, y una multiplicación-suma de
1 bit es un XNOR seguido de un conteo de bits. En un microcontrolador, con kilobytes de RAM y una caché de flash
pequeña, eso cambia qué cabe y qué tan rápido corre:

- **El modelo cabe donde el INT8 no.** Un modelo binario puede vivir en la RAM, o caber en la caché de la flash,
  donde el modelo INT8 de la misma red no cabe.
- **La precisión se recupera con ancho.** Una capa binaria es tan barata que la red se puede ensanchar para
  recuperar la precisión que se pierde al binarizar, y aun así ser más chica y más rápida que INT8. El INT8 no
  puede hacer lo mismo sin pagar el precio completo en memoria.
- **La aritmética es barata incluso en hardware que no fue diseñado para ella.** Los microcontroladores actuales
  están hechos alrededor de la multiplicación (SIMD int8) y no tienen instrucción de popcount; aun así, una prueba
  de concepto en el ESP32-S3 muestra que su propio multiplicador SIMD puede hacer el conteo de bits, y entonces un
  peso binario cuesta **menos que un peso INT8** en el mismo hardware (ver Resultados).

## Resultados

### Redes convolucionales binarizadas (motor booleano de SYMBEX v3)

Medidas en el chip contra los modelos INT8 oficiales de MLPerf Tiny (TensorFlow Lite Micro), con varias semillas
de entrenamiento. Estos resultados vienen del motor de investigación v3; su carpeta de benchmark se está
preparando para publicarla en `benchmarks/`.

| Tarea | Modelo SYMBEX | Precisión | ESP32 (sin SIMD) | ESP32-S3 |
|---|---|---|---|---|
| CIFAR-10 | x1 | 71.44 ± 0.22% | **17.5 veces más rápido que INT8** (40.5 ms contra 710 ms) | 1.52 veces más rápido (34.2 ms contra 52 ms) |
| CIFAR-10 | x2 | 82.90 ± 0.29% | 3.7 veces más rápido que INT8 (189 ms) | -- |
| Visual Wake Words | x2 | 84.19 ± 0.47% (INT8: 86.71%) | 3.5 veces más rápido que INT8 (147 ms contra 521 ms) | -- |

El motor convolucional v3 conserva como primera capa un pequeño **codificador int8 aprendido** (por ahí entra la
imagen); todas las capas que siguen son binarias. Su RAM intermedia baja hasta 16 KB con el popcount en software.

### MLP en Fashion-MNIST, de punta a punta con el CDT ([`benchmarks/mlp/mlp_fashion`](benchmarks/mlp/mlp_fashion/))

Mismo modelo FP32 (88.17%), INT8 oficial de TFLite Micro, las mismas 1000 imágenes, todas las capas de SYMBEX en
1 bit:

| | Precisión | Pesos | ESP32 (sin SIMD) | ESP32-S3 |
|---|---|---|---|---|
| INT8, TensorFlow Lite Micro | 87.20% | 242 KB | 41.97 ms (no cabe en la RAM) | 0.88 ms (modelo en RAM) |
| **SYMBEX, 1 bit, ancho x2 (3 semillas)** | 86.80 ± 0.36% | **69 KB** | **16.70 ms (2.5 veces más rápido)**, 25.54 ms con los pesos en flash (1.6 veces más rápido) | 14.78 ms |

El CDT duplicó el ancho de la red binaria para mantener su precisión a 0.4 puntos del INT8, y aun así es 3.5 veces
más chica y más rápida en el ESP32. En el ESP32-S3, el motor actual todavía no usa el multiplicador SIMD
(siguiente resultado).

### El costo de un peso en el ESP32-S3 (prueba de concepto, mismo hardware para los dos)

Una capa de 256 x 512 pesos en RAM interna, los dos lados usando la instrucción nativa de multiplicación SIMD de
PIE (`ee.vmulas`): el INT8 con el kernel ESP-NN de Espressif; el binario como un XOR de 128 bits seguido del
conteo de bits que hace el mismo multiplicador contra un vector de unos.

| Kernel | Ciclos por peso | Velocidad | Memoria de pesos |
|---|---|---|---|
| INT8, ESP-NN (SIMD de PIE) | 0.531 | 1.00x | 128 KB |
| **Binario, XOR de PIE + multiplicador de PIE** | **0.322** | **1.65x** | **16 KB** |

## Cómo funciona

1. **Binarización.** Los pesos y las activaciones pasan a ser -1 / +1 (un bit). El producto punto de dos vectores
   de 1 bit de largo n es `n - 2 * popcount(a XOR b)`: sin multiplicaciones.
2. **Todo lo demás se pliega en enteros.** La escala de los pesos, la BatchNorm, la activación y el umbral de
   binarización de la capa siguiente se reducen a **un umbral entero por neurona**: `bit = (raw >= T)`.
   La inferencia no usa punto flotante.
3. **La entrada queda como llega.** Los datos (píxeles, valores de un sensor) entran a la primera capa como enteros
   de 8 bits; los pesos de esa capa también son de 1 bit, y se calcula sin multiplicaciones (máscaras).
4. **El CDT decide hasta dónde llegar.** Mide la sensibilidad de cada capa, empieza con todas en 1 bit, y solo
   ensancha la red o sube los bits de la capa más sensible si la caída de precisión supera lo que aceptas, dentro
   del presupuesto de tu hardware. Reentrena con destilación desde el modelo original, pliega el resultado a
   enteros, escribe el header en C y lo verifica.
5. **Validación exacta bit a bit.** El C compilado (en la PC y en el chip) tiene que reproducir el **100%** de una
   simulación entera del modelo: se verifica en cada modelo exportado, siempre.

## Metodología

- **El mismo punto de partida:** un modelo FP32, entrenado una sola vez; SYMBEX e INT8 parten de ese archivo.
- **Una referencia independiente:** el modelo INT8 lo produce el convertidor oficial de TensorFlow Lite y corre con
  TensorFlow Lite Micro oficial, en un entorno aparte que no importa nada de SYMBEX.
- **Las mismas imágenes, el mismo chip:** medianas de 1000 inferencias medidas en el chip (`esp_timer`), un
  núcleo, 240 MHz.
- **Resultados verificados:** SYMBEX contra su simulación entera, TFLite Micro contra el intérprete de TFLite en
  la PC; los dos tienen que coincidir al 100%.
- **Varias semillas** en el entrenamiento de SYMBEX; la precisión, como media ± desviación estándar.

## Estado actual y próximos pasos

- **El motor todavía no usa el multiplicador SIMD del S3.** La prueba de concepto de arriba muestra la ganancia;
  es lo siguiente en entrar a la capa de hardware del S3. Mientras tanto, en el ESP32-S3 con el modelo en RAM, el
  INT8 es más rápido.
- **El CDT todavía no exporta redes convolucionales.** El CDT las binariza y entrena, y corren en el chip con el
  motor v3; unir las dos cosas (que el CDT exporte al motor v3) está en curso.
- **La primera capa recibe datos de 8 bits**, y en el MLP concentra la mayor parte de las operaciones (96% del
  tiempo). Separar la entrada en planos de bits es el paso siguiente.
- **La precisión depende de la tarea:** a 0.4 puntos del INT8 en Fashion-MNIST; en CIFAR-10 la diferencia se cierra
  con ancho (x2), con un costo todavía menor que el del INT8 en el ESP32.
- **La energía todavía no se ha medido**; la latencia y la memoria sí.

## Estructura del repositorio

```
lib_symbex/        motor de inferencia en C y su capa de abstracción de hardware (genérico, ESP32, ESP32-S3, AVR)
                   (por ahora en Symbex1_PIO/lib/lib_symbex)
tools/
  cdt.py           el CDT: configuración (TOML) -> inspección -> estimación -> entrenamiento -> exportación -> validación -> chip
  probar_chip.py   prueba un modelo exportado en una placa: exactitud, precisión, latencia por capa, RAM
  compiler_core/   la librería detrás del CDT (sensibilidad, métodos, entrenamiento, exportación)
  configs/         example.toml: todas las opciones, comentadas
  tests/           pruebas de las piezas del CDT (desde tools/: python -m tests.test_mezclas)
benchmarks/        comparaciones reproducibles contra TensorFlow Lite Micro (ver benchmarks/README.md)
docs/              documentación (inglés y español)
```

## Cómo empezar

```
# 1. configurar: copiar tools/configs/example.toml y completar el modelo, los datos y el hardware
# 2. correr el CDT (desde tools/)
python cdt.py mi_modelo.toml
# 3. probar el resultado en una placa
python probar_chip.py <carpeta de salida> --board esp32 --port /dev/ttyUSB0
```

Requisitos: Python con PyTorch; gcc (verificación de exactitud en la PC); PlatformIO y pyserial para el chip.

## Versiones anteriores

La versión 2 reportaba resultados en el conjunto Digits de scikit-learn con los métodos `gated` y `bitslice`.
Durante la revisión de la versión 3 se encontraron errores en esos métodos (las compuertas de `gated` no recibían
señal de entrenamiento; el simulador de `bitslice` no correspondía al motor en C). Esos resultados no se reportan
aquí hasta que los métodos se reconstruyan y se vuelvan a medir.

## Cómo citarlo

Hay un preprint de SYMBEX-1 en preparación; la referencia se agregará aquí.

## Autor

Carlos Duarte (Edén) -- Panamá -- GitHub: [Eden-sudo](https://github.com/Eden-sudo)

Versión en inglés: [README.md](README.md) -- Licencia: ver [LICENSE](LICENSE).
