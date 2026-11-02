# Documentación de SYMBEX-1

*[Read in English](README.md)*

Cómo funciona SYMBEX-1, cómo se usa y qué midió. El [README principal](../README.es.md) tiene el resumen; estas páginas
entran en los detalles.

| Página | Qué explica |
|---|---|
| [Cómo funciona](how-it-works.es.md) | Redes binarias, contar bits en vez de multiplicar, plegar todo en umbrales enteros, exactitud bit a bit |
| [CDT](cdt.es.md) | El compilador: de un modelo de PyTorch a un header de C comprobado, en la PC y en el chip |
| [Librería](library.es.md) | lib_symbex: las capas, la capa de hardware de cada chip, cómo añadirla a un proyecto |
| [Resultados](results.es.md) | Todos los benchmarks en un solo lugar: velocidad, flash y precisión frente a INT8, con las gráficas |

Los diagramas están escritos en [Mermaid](https://mermaid.js.org/) dentro de las páginas (GitHub los dibuja). Las
gráficas viven en [`assets/`](assets/) y se generan a partir de los archivos de resultados de los benchmarks, así que
siempre se pueden volver a crear.

## De un vistazo

Redes completamente binarias frente a las INT8 oficiales (TensorFlow Lite Micro con los kernels ESP-NN de Espressif, la
ruta de despliegue que usa casi todo el mundo, no el INT8 más rápido posible: ver
[Resultados](results.es.md#qué-tan-rápido-podría-ser-int8)), la misma red en el mismo chip:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speedup_dark.svg">
  <img alt="Aceleración de SYMBEX sobre TensorFlow Lite Micro INT8 por benchmark y chip" src="assets/speedup.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/flash_dark.svg">
  <img alt="Flash que usa cada modelo, SYMBEX frente a INT8" src="assets/flash.svg">
</picture>

La diferencia depende del chip y de dónde están los pesos. El ESP32-S3 tiene una unidad SIMD que acelera INT8, así que
ahí la diferencia es menor cuando INT8 puede usarla por completo (ResNet-8: 1.33×, el MLP con sus pesos en RAM: 2.14×),
y la librería ESP-DL de Espressif es más rápida que SYMBEX ahí (ResNet-8 en 14.8 ms frente a 39.2 ms); en el ESP32
clásico, sin SIMD, SYMBEX es 8.6-11.3× más rápido, también frente a ESP-DL, y el detector INT8 ni siquiera cabe. La
precisión está en el README de cada benchmark. Para volver a crear las gráficas después de una nueva medición:
`python tools/dev/docs_charts.py`.
