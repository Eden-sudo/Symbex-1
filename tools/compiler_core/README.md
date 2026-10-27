# compiler_core — the CDT inside

`cdt.py` runs the stages; this package does the work of each one. How a model goes through it:

```
your model ─► inspect ─► sensitivity.py (estimate) ─► methods/ + train/ (train) ─► export/ (export + check) ─► chip/ (board)
```

**Current** marks what the CDT and the benchmarks use today; **earlier versions** marks code kept from SYMBEX v2 and the
first v3 engine, still runnable but not behind any published result.

## chip/ — everything that runs on a real board

| File | What it does |
|---|---|
| `core.py` | what every chip tool shares: the boards, the PlatformIO / ESP-IDF project, flashing, the serial port |
| `measure.py` | the chip stage and `chip_test.py`: flashes an exported model, streams the test images, and reports equality with the integer simulation (must be 100 %), accuracy, latency per layer and RAM |
| `kernels.py` | which arithmetic is faster on THIS chip for each layer size (dense layers); the result is a chip profile |
| `firmware/` | the C programs that run on the board: `model_test.c` (the model), `fine_profile.c` (`dev/fine_profile.py`), `kernels.c` |

## export/ — from the trained network to integers and C

Every exporter does the same three things, with the SAME integers: **fold** the trained model into integers, **simulate**
it exactly (the ground truth the C must reproduce bit for bit), **write** `model_symbex.h`; then the C is compiled on
this PC and checked against the simulation.

| File | Method | |
|---|---|---|
| `boolean.py` | `boolean`: fully binary convolutional networks (thermometer input, OR shortcuts, classifier or per-cell head); also its training loop | current |
| `dense.py` | `plain` dense networks (MLPs at 1 bit) | current |
| `ternary.py` | `ternary` dense networks (weights in {-1, 0, +1}) | current |
| `int8_reference.py` | the INT8 version of the original model, with TFLite's integer arithmetic (an optional reference) | current |
| `report.py` | the before / after table every exporter prints | current |
| `plain.py`, `conv.py` | binarized ResNets for the first v3 engine (int8 first layer, fixed-point BatchNorm) | earlier versions |
| `gated.py`, `bitslice.py` | the block-gated and multi-bit dense methods of v2 | earlier versions |

## methods/ — turning a layer into its binary version

| File | What it does | |
|---|---|---|
| `base.py` | what every method shares: the sign functions and their gradients (straight-through, Bi-Real) | current |
| `__init__.py` | `apply_plan` (a different option per layer: 1 bit or k bits), `widen` (wider hidden layers, starting from copies of the original neurons) | current |
| `dense/plain.py`, `conv/plain.py` | the 1-bit layers: binary input and weights | current |
| `kbit.py` | k-bit layers (2, 4, 8) for mixed precision, quantization simulated during training | current |
| `dense/ternary.py` | the ternary layer, with one learnable zeroing threshold per layer | current |
| `dense/gated.py`, `dense/bitslice.py`, `dense/core.py` | the v2 methods and their sign convention (0 → −1) | earlier versions |

The boolean networks do not go through `methods/`: they are binary as defined (their layers live in the benchmark's
`model.py`).

## sensitivity.py — the estimate stage

On the ORIGINAL model, layer by layer: how much each one suffers when its weights, activations and normalization go down
to k bits (cosine similarity on real samples), and the memory of each choice. It orders which layer the search raises
first; the final decision is made by accuracy, in the train stage.

## train/ — the training loops

| File | What it does | |
|---|---|---|
| `standard.py` | cross-entropy with the straight-through gradient of the binary layers, the BatchNorm fix, checkpoints, the input bit quantizer | current |
| `distillation.py` | optional: the original model teaches the binary one (soft outputs + labels) | current |
| `attention.py` | reserved: binary attention needs something else (not implemented) | — |

## validate/ — independent checks

| File | What it does | |
|---|---|---|
| `ternary.py` | the ternary engine's packed-word simulator against a plain numpy integer reference | current |
| `plain.py` | the bit-exact simulator of the first v3 convolutional engine | earlier versions |

The current exporters check their C themselves (`validate_c` in each one); the library itself is tested by
`Symbex1_PIO/lib/lib_symbex/tests/run_tests.py`, and the Python side by `pytest` from the repository root.
