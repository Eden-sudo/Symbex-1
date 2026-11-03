# Resultados: todos los benchmarks en un solo lugar

*[Read in English](results.md)*

Cada benchmark compara una red SYMBEX completamente binaria con INT8 en el mismo chip y con la misma arquitectura. El
lado INT8 es el modelo oficial de TensorFlow Lite Micro o, donde no existe un modelo oficial, la misma red entrenada con
la misma receta. Todas las reglas vienen de la [metodología](../README.es.md#metodología):

- los resultados son medias sobre 3 semillas (42, 43, 44);
- toda elección se hace sobre un split de validación, y el conjunto de test se usa una sola vez;
- los pesos están en flash salvo que la tabla diga otra cosa;
- en cada placa, el chip da **la misma salida que la simulación entera en el 100 % de las entradas**.

**Qué significa INT8 aquí.** El lado INT8 es la ruta de despliegue que usa casi todo el mundo: TensorFlow Lite Micro
con los kernels optimizados ESP-NN de Espressif. No es el INT8 más rápido posible: en el ESP32-S3, la librería ESP-DL de
Espressif ejecuta ResNet-8 2.6× más rápido que el motor estándar de SYMBEX. El motor vertical de SYMBEX
([El motor vertical](vertical.es.md)) lo ejecuta en 10.4 ms, 1.43× más rápido que ESP-DL con los pesos en flash; con los
pesos en RAM, ESP-DL sigue 1.06× más rápido, salvo con uno de los modos opcionales del motor: compacto (7.70 ms, 1.27×
más rápido que ESP-DL en RAM, con 151 KB de RAM frente a los 147 KB de ESP-DL) o preparado (6.44 ms, 1.52× más rápido,
238 KB de RAM). [¿Qué tan rápido podría ser INT8?](#qué-tan-rápido-podría-ser-int8)
muestra qué está medido y qué no.

Esta página da lo principal de cada benchmark. Los detalles, los comandos y los archivos crudos están en el README de
cada benchmark.

## Velocidad, flash y RAM

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speedup_dark.svg">
  <img alt="Aceleración de SYMBEX sobre TensorFlow Lite Micro INT8 por benchmark y chip" src="assets/speedup.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/flash_dark.svg">
  <img alt="Flash que usa cada modelo, SYMBEX frente a INT8" src="assets/flash.svg">
</picture>

Todas las corridas en un núcleo, a 240 MHz en los dos ESP32. SYMBEX e INT8 siempre tienen sus pesos en el mismo lugar:
flash en todas las filas menos una.

| Benchmark | Chip | Pesos en | Latencia, SYMBEX / INT8 | Aceleración | Flash, SYMBEX / INT8 | RAM, SYMBEX / INT8 |
|---|---|---|---|---|---|---|
| MLP (Fashion-MNIST, 784-128-512-10) | ESP32-S3 | flash | 0.42 / 11.67 ms | **27.8×** | 27.3 / 187.0 KB | 4.6 / 6.7 KB |
| MLP (Fashion-MNIST, 784-128-512-10) | ESP32-S3 | **RAM** | 0.414 / 0.888 ms | **2.14×** | 27.3 / 187.0 KB | 4.6 / 6.7 KB + los pesos (21 / 187 KB) |
| MLP (Fashion-MNIST, 784-128-512-10) | ESP32 | flash | 3.64 / 31.11 ms | **8.56×** | 27.3 / 187.0 KB | 4.4 / 6.7 KB |
| ResNet-8 (CIFAR-10) | ESP32-S3 | flash | 39.2 / 52.0 ms | **1.33×** | 24.3 / 96.2 KB | 14.8 / 80.5 KB |
| ResNet-8 (CIFAR-10) | ESP32 | flash | 63.3 / 717.8 ms | **11.3×** | 24.3 / 96.2 KB | 78.6 / 53.1 KB ¹ |
| Detección de personas FOMO (32-32-128) | ESP32-S3 | flash | 126.3 / 256.9 ms | **2.03×** | 41.2 / 236.1 KB | 79.0 / 222.6 KB |
| Detección de personas FOMO (32-32-128) | ESP32 | flash | 226.3 ms / no cabe | **solo SYMBEX corre** | 41.2 / 236.1 KB | 142.8 KB ¹ / no cabe |
| Reconocimiento de actividad (1D), 4 redes | ESP32-S3 | flash | 0.49-3.83 / 0.72-7.11 ms | **1.47-1.90×** | 0.7-6.5 / 7.3-42.5 KB | 1.3-2.7 / 2.7-25.5 KB |
| Reconocimiento de actividad (1D), 4 redes | ESP32 | flash | 0.78-14.08 / 4.17-115.62 ms | **5.3-10.4×** | 0.7-6.5 / 7.3-42.5 KB | 1.3-2.7 / 2.7-10.3 KB |

- **Flash** es el modelo completo: pesos, umbrales y escalas en SYMBEX, el archivo `.tflite` en INT8.
- **RAM** es lo que usa el modelo mientras corre. En SYMBEX son los buffers, la entrada y la pila, medidos en el chip.
  En INT8 es la arena de TFLite Micro, que también guarda la entrada.
- ¹ En el ESP32 clásico, SYMBEX cuenta bits con una tabla de 64 KB en RAM, el método más rápido ahí. Sin ella, ResNet-8
  necesita 15 KB de RAM y corre en 71.0 ms (10.1×), y FOMO necesita 79 KB y corre en 274.5 ms.
- Las filas 1D son los rangos sobre las cuatro redes de Daghero, de la más pequeña a la más grande.

Las gráficas se dibujan a partir de los archivos de resultados con `python tools/dev/docs_charts.py`, así que una nueva
medición las actualiza.

La diferencia la marca el chip. El ESP32 clásico no tiene SIMD, y ahí contar bits es un orden de magnitud más barato que
multiplicar y acumular en 8 bits. El ESP32-S3 tiene una unidad SIMD (PIE) que acelera INT8, así que la diferencia es
menor. Donde INT8 puede usar esa unidad por completo, la diferencia es la más estrecha: ResNet-8 a 1.33× y el MLP con
sus pesos en RAM a 2.14×. Ahí, la librería ESP-DL de Espressif es más rápida que el motor estándar de SYMBEX; en
ResNet-8, [el motor vertical](vertical.es.md) le gana con los pesos en flash, y en RAM con sus modos opcionales compacto y preparado (ver
[¿Qué tan rápido podría ser INT8?](#qué-tan-rápido-podría-ser-int8)). [Cómo funciona](how-it-works.es.md) explica por
qué contar bits es más barato.

## Cómo se mide

- **Reloj.** El firmware fija el reloj en su configuración de compilación: 240 MHz en el ESP32 y el ESP32-S3
  (`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240`), 150 MHz en la Pico 2, 16 MHz en el Arduino Uno. Nadie mide los MHz; son el
  ajuste con el que se compilan los dos lados.
- **Tiempo.** El firmware lee el temporizador de microsegundos del chip (`esp_timer_get_time()`, `micros()` en el Uno)
  justo antes y justo después de una inferencia. Cada entrada se mide por separado, y se reporta la mediana sobre todas
  las entradas (1,000 entradas en la mayoría de las corridas). Un núcleo, nada más corriendo.
- **La misma compilación para los dos lados.** Los firmwares de SYMBEX y de TFLite usan la misma configuración de
  ESP-IDF: optimizado para velocidad (`CONFIG_COMPILER_OPTIMIZATION_PERF`), el mismo reloj y el mismo protocolo serie.
- **Corrección.** Cada salida de SYMBEX se compara con la simulación entera en la PC. Cada salida de TFLite se compara
  con el intérprete de TFLite en la PC.

## ¿Qué tan rápido podría ser INT8?

El lado INT8 es la ruta oficial de despliegue, ya optimizada: TensorFlow Lite Micro de Espressif (`esp-tflite-micro`)
con los kernels ESP-NN de Espressif activados (`CONFIG_NN_OPTIMIZED`). En el ESP32-S3, esos kernels usan la unidad SIMD
PIE. La arena se dimensiona a lo que necesita cada modelo.

Dónde podría INT8 ir todavía más rápido:

| Opción | Lo que sabemos |
|---|---|
| Llamar a ESP-NN directamente, sin el intérprete de TFLite | **Medido** en el MLP x1 (784-256-128-10) en el ESP32-S3, pesos en RAM: 0.556 ms en vez de 0.876 ms (1.58× más rápido). SYMBEX x1 corre en 0.525 ms, así que en RAM los dos están cerca (`tools/lowbit`, [mlp_fashion](../benchmarks/mlp/mlp_fashion/README.md)). |
| Pesos en RAM | **Medido** donde el modelo cabe: el MLP y las redes 1D. En flash, INT8 es lento porque lee 8× más bytes a través de la caché de la flash. La RAM elimina eso, si el chip tiene espacio. |
| ESP-DL (la otra librería de Espressif, optimizada para el S3) | **Medido** en una comprobación rápida, tabla abajo. En el ESP32-S3 es el INT8 más rápido: más rápido que el motor estándar de SYMBEX en ResNet-8 (el motor vertical le gana con los pesos en flash, y en RAM con su modo compacto con casi la misma RAM), y en el MLP cuando los pesos están en RAM. |
| CMSIS-NN en el Cortex-M33 de la Pico 2 | **Medido** (TFLite Micro con los kernels CMSIS-NN de Arm, DSP activo). No alcanza a SYMBEX: ResNet-8 453.0 ms en flash y 211.4 ms con el modelo en RAM, frente a 74.5 ms de SYMBEX (6.1× / 2.8×); MLP 784-128-512-10 31.35 / 2.89 ms frente a 4.57 / 1.53 ms (6.9× / 1.9×). |

**ESP-DL, comprobación rápida.** ESP-DL v3 (`espressif/esp-dl`, ESP-IDF 6.1), modelos cuantizados a INT8 con `esp-ppq`
(post-entrenamiento, 512 imágenes de calibración; `target="esp32s3"` en el S3, `"c"` en el ESP32). ResNet-8 usa los
pesos en punto flotante de MLPerf Tiny; el MLP es el mismo FP32 784-128-512-10 del que parten las filas INT8. El mismo
reloj y la misma configuración de compilación que todas las demás corridas, mediana de 100 (ResNet-8) o 200 (MLP)
entradas de test, una compilación por fila. "Flash" deja los pesos en flash (`param_copy = false`); "RAM" los copia a
RAM cuando el modelo se carga.

| Modelo | Chip | Pesos en | ESP-DL | SYMBEX | TFLite Micro | SYMBEX frente a ESP-DL |
|---|---|---|---|---|---|---|
| ResNet-8 | ESP32-S3 | flash | **14.8 ms** | 39.2 ms | 52.0 ms | ESP-DL **2.6× más rápido** |
| ResNet-8, motor vertical de SYMBEX | ESP32-S3 | flash | 14.8 ms | **10.4 ms** (63 KB de RAM) | 52.0 ms | SYMBEX **1.43× más rápido** |
| ResNet-8, motor vertical de SYMBEX | ESP32-S3 | RAM | **9.8 ms** (147 KB de RAM) | 10.4 ms (87 KB de RAM) | — | ESP-DL **1.06× más rápido** |
| ResNet-8, motor vertical de SYMBEX, compacto (opcional) | ESP32-S3 | flash | 14.8 ms | **7.70 ms** (151 KB de RAM) | 52.0 ms | SYMBEX **1.92× más rápido** |
| ResNet-8, motor vertical de SYMBEX, compacto (opcional) | ESP32-S3 | RAM | 9.8 ms (147 KB de RAM) | **7.70 ms** (175 KB de RAM) | — | SYMBEX **1.27× más rápido** |
| ResNet-8, motor vertical de SYMBEX, preparado (opcional) | ESP32-S3 | flash | 14.8 ms | **6.44 ms** (238 KB de RAM) | 52.0 ms | SYMBEX **2.3× más rápido** |
| ResNet-8, motor vertical de SYMBEX, preparado (opcional) | ESP32-S3 | RAM | 9.8 ms (147 KB de RAM) | **6.44 ms** (262 KB de RAM) | — | SYMBEX **1.52× más rápido**, con más RAM |
| ResNet-8 | ESP32 | flash | 712.4 ms | **63.3 ms** | 717.8 ms | SYMBEX **11.3× más rápido** |
| ResNet-8 | ESP32 | RAM | 603.4 ms | **59.7 ms** | — | SYMBEX **10.1× más rápido** |
| MLP 784-128-512-10 | ESP32-S3 | flash | 11.38 ms | **0.42 ms** | 11.67 ms | SYMBEX **27× más rápido** |
| MLP 784-128-512-10 | ESP32-S3 | RAM | **0.317 ms** (177 KB de RAM) | 0.414 ms | 0.888 ms | ESP-DL **1.31× más rápido** |
| MLP 784-128-512-10 | ESP32 | flash | 28.49 ms | **3.64 ms** | 31.11 ms | SYMBEX **7.8× más rápido** |
| MLP 784-128-512-10 | ESP32 | RAM | 4.84 ms (177 KB de RAM) | **3.64 ms** (en flash) | — | SYMBEX 1.33× más rápido, desde flash |

Precisión en esas entradas: ResNet-8 86 / 100, el MLP 179 / 200, la misma en los dos chips. Esta comprobación es una
corrida sobre una parte del conjunto de test, todavía no un benchmark completo con 3 semillas; basta para mostrar dónde
está ESP-DL.

Así que los números principales comparan SYMBEX con la ruta INT8 que despliega casi todo el mundo, no con el INT8 más
rápido posible. En el ESP32-S3, ESP-DL es más rápido que el motor estándar de SYMBEX en ResNet-8 (2.6×) y en el MLP
con sus pesos en RAM (1.31×). [El motor vertical](vertical.es.md) le da la vuelta en ResNet-8 con los pesos en flash
(10.4 ms, 1.43× más rápido que ESP-DL); con los pesos en RAM, ESP-DL sigue 1.06× más rápido, salvo con un modo
opcional: compacto (7.70 ms, 1.27× más rápido que ESP-DL con 151 KB de RAM frente a 147 KB) o preparado (6.44 ms,
1.52× más rápido, 238 KB). Las ventajas claras de SYMBEX son los chips sin SIMD (el ESP32 clásico: 7.8-11.3× más rápido que ESP-DL
también), los pesos que se quedan en flash en capas densas, el tamaño en flash (4-10× menor) y la RAM.

## Precisión: dónde cuesta lo binario

Los pesos y las activaciones binarias cuestan precisión. Cuánto depende de la tarea:

| Benchmark | SYMBEX | INT8 (la misma red) | Diferencia |
|---|---|---|---|
| Detección de personas FOMO, F1 | **31.2 ± 0.7 %** | 30.1 ± 0.2 % | binaria tan buena como INT8 |
| Dígitos en un Arduino Uno (entrada de 1 bit) | 94.5 ± 1.3 % | FP32 con la misma entrada: 94.4 % | binarizar la red no cuesta nada |
| MLP, Fashion-MNIST (784-128-512-10) | 87.5 % | 89.8 % (QAT) | −2.3 puntos |
| ResNet-8, CIFAR-10 | 71.3 ± 0.8 % (73.1 % dimensionada para binario) | 87.0 % (MLPerf Tiny) | −14 a −16 puntos |
| Reconocimiento de actividad, redes grandes | 56.8-58.8 % | 66.2-67.9 % (TFLite) | unos −9 puntos |

La detección, y las tareas pequeñas con una entrada binaria, se binarizan bien. Las clases que solo se distinguen por
detalles finos, como las diez de CIFAR-10, son las que más pierden. Con una entrada de un byte, como un acelerómetro, una
primera capa int8 es más rápida y más precisa que una binaria (ver el benchmark 1D).

## Los benchmarks

### ResNet-8 en CIFAR-10: contra el modelo oficial de MLPerf Tiny

[`benchmarks/conv2d/cnn_cifar_resnet8`](../benchmarks/conv2d/cnn_cifar_resnet8/README.md)

- La arquitectura es la misma que la ResNet-8 INT8 oficial de MLPerf Tiny, medida con TFLite Micro en las mismas
  placas.
- **ESP32-S3:** 39.2 ms frente a 52.0 ms (1.33×). ESP-DL de Espressif ejecuta la misma red en 14.8 ms, 2.6× más rápido
  que el motor estándar de SYMBEX (ver [¿Qué tan rápido podría ser INT8?](#qué-tan-rápido-podría-ser-int8)).
- **ESP32-S3, motor vertical:** la misma red y los mismos bits en 10.36 ± 0.05 ms (3 semillas), 5.0× más rápido que
  TFLite Micro y 1.43× más rápido que ESP-DL con los pesos en flash, 63 KB de RAM ([El motor vertical](vertical.es.md)).
  Sus modos opcionales construyen listas y umbrales una vez en RAM: compacto 7.70 ± 0.05 ms con 151 KB de RAM (1.27×
  más rápido que ESP-DL en RAM), preparado 6.44 ± 0.02 ms con 238 KB (8.1× más rápido que TFLite Micro, 1.52× más
  rápido que ESP-DL en RAM).
- **ESP32:** 63.3 ms con los pesos en flash frente a 717.8 ms (11.3×); 59.7 ms con los pesos en RAM.
- **Pico 2:** 56.1 ms en Hazard3 y 74.5 ms en el M33; en el M33 el modo preparado del motor vertical la ejecuta en
  57.4 ms (semilla 42, un ranking; [El motor vertical](vertical.es.md#otros-chips-pico-2)). TFLite Micro INT8 con
  CMSIS-NN en el M33: 453.0 ms (6.1×), 211.4 ms con el modelo en RAM; en Hazard3, con los kernels de referencia: 5702 ms.
- **Memoria:** 24.3 KB de flash frente a 96.2 KB, y 15 KB de RAM frente a una arena de 53-80 KB.
- **Precisión:** 71.3 ± 0.8 % frente a 87.0 %. Con la red dimensionada para binario (16-32-128 / 1-1-2) llega a 73.1 %
  a la misma velocidad.

### Detección de personas FOMO en Pascal VOC

[`benchmarks/conv2d/fomo_voc_person`](../benchmarks/conv2d/fomo_voc_person/README.md)

- La comparación es el mismo detector, entrenado de la misma forma, contra su versión INT8.
- **Detección:** F1 31.2 % frente a 30.1 %, así que los dos detectan igual.
- **ESP32-S3:** la binaria es 2.03× más rápida.
- **Flash:** 41 KB frente a 236 KB.
- **ESP32 clásico:** el detector binario corre en 226.3 ms; el INT8 no cabe, porque su arena excede la RAM por 88 KB.
- **Pico 2:** 338.5 ms en Hazard3 y 416.4 ms en el M33.

### MLP en Fashion-MNIST: binario contra INT8, INT4 e INT2

[`benchmarks/mlp/mlp_fashion`](../benchmarks/mlp/mlp_fashion/README.md)

- **Dimensionado para binario** (784-128-512-10): 0.42 ms en el ESP32-S3 con los pesos en flash frente a 11.7 ms de
  TFLite INT8 con la misma red (28×), 27 KB de flash y 4.6 KB de RAM.
- **Precisión por byte:** ganan los enteros de pocos bits. INT2 QAT es 1.0-1.6 puntos más preciso que un modelo binario
  de más o menos el mismo tamaño.
- **Velocidad y memoria:** gana SYMBEX. El ancho funciona mejor cuando la primera capa es estrecha y las capas binarias
  son anchas.

### Dígitos en un Arduino Uno: binario de principio a fin

[`benchmarks/mlp/mlp_digits`](../benchmarks/mlp/mlp_digits/README.md)

- La entrada, los pesos y las activaciones son todos de 1 bit.
- **Arduino Uno:** 94.5 % en dígitos escritos a mano en 5.3 ms, con 0.21 KB de RAM y 3.4 KB de flash.
- **La misma red** corre en 0.11 ms en el ESP32-S3, 0.23 ms en el ESP32 y 0.38-0.40 ms en la Pico 2.

### Reconocimiento de actividad con convoluciones 1D: contra ultracompactBNN e INT8

[`benchmarks/conv1d/har_ultracompact`](../benchmarks/conv1d/har_ultracompact/README.md)

- **Contra ultracompactBNN** (Daghero et al. 2021), en su propio entorno: su procesador (RI5CY en GVSoC), sus cuatro
  redes y su código sin cambios. SYMBEX necesita **1.07-1.37× menos instrucciones**, y los dos son exactos.
- **Contra TFLite INT8** en las placas: 5.3-10.4× más rápido en el ESP32 y 1.47-1.90× en el ESP32-S3, con 6.5-10× menos
  flash.
- **Dónde deja de convenir lo completamente binario (una teoría de trabajo):** con una entrada de 8 bits, una primera
  capa int8 es más rápida que la binaria y más precisa.

## Trabajo futuro

- Energía por inferencia, medida con un medidor de corriente.
- ESP-DL como referencia INT8 completa (3 semillas, todo el conjunto de test, todos los benchmarks), con sus scripts en
  el repositorio.
- Ganarle a ESP-DL en el ESP32-S3 con los pesos en RAM usando claramente menos RAM (motor vertical por defecto: 10.4 ms
  frente a 9.8 ms con 63 KB; el modo compacto gana con 7.70 ms y casi la misma RAM, 151 KB frente a 147 KB), y el
  motor vertical para el MLP y las redes 1D.
- Volver a calibrar el umbral de decisión de FOMO sobre el nuevo split de validación.
