# SYMBEX-1 -- Redes neuronales binarias para microcontroladores

SYMBEX-1 convierte una red neuronal entrenada en una **red de 1 bit que corre en microcontroladores pequeños**, y mide,
en chips reales, cuánto se gana y cuánto se pierde frente a la herramienta oficial de INT8.

Tiene tres partes:

- **`lib_symbex`**: un motor de inferencia en C, portable, para capas binarias (XNOR + popcount, umbrales enteros).
- **El CDT** (`tools/cdt.py`): toma un modelo FP32, tus datos y los límites de tu hardware, y produce un modelo
  binarizado, su header en C y una validación exacta bit a bit, listo para el chip.
- **`benchmarks/`**: comparaciones reproducibles contra INT8 de TensorFlow Lite / TensorFlow Lite Micro, partiendo
  del mismo modelo FP32, con las mismas imágenes y en el mismo chip.

## La afirmación, dicha con precisión

En el hardware que más lo necesita (**microcontroladores sin SIMD y con poca memoria**), una red binaria alcanza
una precisión cercana a la de INT8 con **varias veces menos memoria** e **inferencia más rápida**.
En hardware diseñado alrededor de unidades de multiplicación-suma (SIMD para int8), la red binaria sigue ahorrando
memoria, pero pierde en velocidad: no porque su aritmética sea menos eficiente, sino porque ese hardware tiene
instrucciones para multiplicar y **ninguna para la operación que necesitan las redes binarias (popcount)**.

## Por qué

Los microcontroladores tienen kilobytes, no gigabytes: decenas o cientos de KB de RAM, unos pocos MB de flash, y a
menudo no tienen FPU ni unidad vectorial. Un modelo normal no cabe, y un modelo INT8 muchas veces solo cabe en la
flash, donde cada lectura de pesos pasa por una caché pequeña. Un peso de 1 bit ocupa 8 veces menos que uno INT8 y
32 veces menos que uno FP32, y una multiplicación-suma de 1 bit es un XNOR seguido de un conteo de bits.

## Cómo funciona

1. **Binarización.** Los pesos y las activaciones pasan a ser -1 / +1 (un bit). El producto punto de dos vectores de
   1 bit de largo n es `n - 2 * popcount(a XOR b)`: sin multiplicaciones.
2. **Todo lo demás se pliega en enteros.** La escala de los pesos, la BatchNorm, la activación y el umbral de
   binarización de la capa siguiente se reducen a **un umbral entero por neurona**: `bit = (raw >= T)`.
   La inferencia no usa punto flotante ni multiplicaciones en las capas binarias.
3. **La primera capa conserva la entrada de 8 bits.** Los píxeles (o los valores de un sensor) entran como uint8
   contra pesos de 1 bit, calculados de forma exacta con enteros.
4. **El CDT decide hasta dónde llegar.** Mide la sensibilidad de cada capa, empieza con todas las capas en 1 bit,
   y solo ensancha la red o sube los bits de la capa más sensible cuando la caída de precisión supera lo que
   aceptas, con el presupuesto de tu hardware como techo. Reentrena con destilación desde el modelo original,
   pliega el resultado a enteros, escribe el header en C y lo verifica.
5. **Validación exacta bit a bit.** El C compilado (en la PC y en el chip) tiene que reproducir el **100%** de una
   simulación entera del modelo: se verifica en cada modelo exportado, siempre.

## Metodología

Todas las comparaciones de `benchmarks/` siguen las mismas reglas:

- **El mismo punto de partida:** un modelo FP32, entrenado una sola vez; SYMBEX e INT8 parten de ese archivo.
- **Una referencia independiente:** el modelo INT8 lo produce el convertidor oficial de TensorFlow Lite y corre con
  TensorFlow Lite Micro oficial, en un entorno aparte que no importa nada de SYMBEX.
- **Las mismas imágenes, el mismo chip:** los dos modelos reciben las mismas imágenes de validación por puerto
  serie; los tiempos son medianas de 1000 inferencias medidas en el chip (`esp_timer`), un núcleo, 240 MHz.
- **Resultados verificados:** SYMBEX contra su simulación entera, TensorFlow Lite Micro contra el intérprete de
  TFLite en la PC. Los dos tienen que coincidir al 100%.
- **Varias semillas** en el entrenamiento de SYMBEX; la precisión se reporta como media ± desviación estándar.

## Resultados

**MLP en Fashion-MNIST** ([`benchmarks/mlp_fashion`](benchmarks/mlp_fashion/)). FP32 original: 88.17%.

| | Precisión | Flash | ESP32 (sin SIMD) | ESP32-S3 (SIMD int8) |
|---|---|---|---|---|
| INT8, TensorFlow Lite Micro | 87.20% | 242 KB | 41.97 ms (modelo en flash; no cabe en la RAM) | **0.88 ms** (modelo en RAM) |
| SYMBEX, 1 bit (3 semillas) | 86.80 ± 0.36% | **69 KB** | **16.70 ms** (pesos en RAM) / 25.54 ms (flash) | 14.78 ms (pesos en RAM) |

Los resultados del motor booleano de SYMBEX v3 en **CIFAR-10** y **Visual Wake Words** (convolucionales, ESP32 y
ESP32-S3, contra los modelos INT8 oficiales de MLPerf Tiny) se están pasando a `benchmarks/`.

## Limitaciones

- **El hardware diseñado para multiplicar gana en velocidad.** En el ESP32-S3, el modelo INT8 en RAM es 17 veces
  más rápido que SYMBEX: su SIMD hace 16 multiplicaciones-suma int8 por instrucción, y el chip no tiene
  instrucción de popcount, así que SYMBEX cuenta los bits en software.
- **En el ESP32 clásico, parte de la ventaja en velocidad viene de la memoria**, no de la aritmética: los pesos
  INT8 no caben en la caché de la flash ni en la RAM; los binarios sí.
- **El costo en precisión depende de la tarea.** Pequeño en Fashion-MNIST (unos 0.4 puntos contra INT8); de
  varios puntos en CIFAR-10.
- **La primera capa es el cuello de botella** del motor actual (entrada de 8 bits: entre el 96 y el 98% del tiempo
  en el MLP).
- **Todavía no soportado por el CDT:** exportar redes convolucionales (se entrenan, pero aún no llegan al chip),
  capas de k bits en el chip (la precisión mixta solo se entrena en Python), y ensanchar redes residuales.
- **La energía no se ha medido.** Solo se reportan latencia y memoria.

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

## Estado

En desarrollo activo (versión 3). Funciona de punta a punta: MLP de 1 bit, desde el modelo FP32 hasta la medición
en el chip. En curso: exportación de convolucionales, integrar el motor booleano v3 a `lib_symbex`, y capas de
k bits en el chip.

## Versiones anteriores

La versión 2 reportaba resultados en el conjunto Digits de scikit-learn con los métodos `gated` y `bitslice`.
Durante la revisión de la versión 3 se encontraron errores en esos métodos (las compuertas de `gated` no recibían
señal de entrenamiento; el simulador de `bitslice` no correspondía al motor en C). Esos resultados no se reportan
aquí hasta que los métodos se reconstruyan y se vuelvan a medir.

## Cómo citarlo

Hay un preprint de SYMBEX-1 en preparación; la referencia se agregará aquí.

## Autor

Carlos Duarte (Edén) -- Panamá -- GitHub: [Eden-sudo](https://github.com/Eden-sudo)

Versión en inglés: [README.md](README.md)

Licencia: ver [LICENSE](LICENSE).
