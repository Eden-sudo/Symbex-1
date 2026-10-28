# lowbit: the integer baselines below INT8, for every benchmark

`tools/lowbit` quantizes the **same FP32 network** a benchmark compares SYMBEX against to INT8, INT4 and INT2, and runs
it on the board with **ordinary integer arithmetic**: multiply-accumulate on Espressif's ESP-NN kernels, the same
arithmetic as TensorFlow Lite Micro. No SYMBEX arithmetic is involved, so the comparison is binary logic against the
standard integer method at the same bit budget.

| Step | Module | What it does |
|---|---|---|
| fold | `quantize.py` | the network as a sequence of layers, BatchNorm folded; checked against the model's own forward |
| quantize | `quantize.py` | [Brevitas](https://github.com/Xilinx/brevitas): weights per output channel, symmetric, narrow range (INT8 [-127, 127], INT4 [-7, 7], INT2 [-1, 1]); activations int8 per tensor |
| train | `study.py` | post-training (`ptq_maxabs`, `ptq_mse`) and quantization-aware training (`qat`) with the CDT's recipe, 3 seeds, distillation or labels only |
| simulate | `quantize.py` | exact integer inference: int32 accumulators, the fixed-point requantization of ESP-NN (`esp_nn_multiply_by_quantized_mult`, checked equal on 200,000 random cases) |
| export | `export_c.py` | C for the board: packed weights (2 or 4 codes per byte), unpacked to int8 for `esp_nn_fully_connected_per_ch_s8`, `esp_nn_conv_s8`, `esp_nn_max_pool_s8`; checked on the PC with ESP-NN's reference kernels against the simulation (**100 %**) |
| measure | `export_c.py` | `tools/chip_test.py` on the board: latency per layer, RAM, and the board against the simulation (**100 %**) |

Supported layers: Linear, Conv1d, Conv2d (BatchNorm right after them), ReLU, MaxPool1d / MaxPool2d, Flatten, Dropout.
Networks with skip connections (ResNet-8) are not supported yet: `fold()` stops with a message.

## In a benchmark

Each benchmark has a short `<bench>_lowbit.py` that loads its FP32 model and its data (validation carved out of the
training data, the test set evaluated once per final model) and calls the tool:

```python
from lowbit.study import Data, run, save, table
from lowbit.export_c import chip_study
res = run("mlp_fashion", model_fp32, Data(x_train, y_train, x_val, y_val, x_test, y_test))   # INT8, INT4, INT2
table(res); save(res, "results/lowbit.json")
chip_study("runs/lowbit", x_test[:1000], y_test[:1000], [8, 4, 2], board="s3", port="/dev/ttyACM0", usb=True)
```

`benchmarks/mlp/mlp_fashion/fashion_lowbit.py` is the first one.

## Two placements of the weights on the board

- **flash** (default): INT8 weights are read in place; INT4 / INT2 codes stay packed in flash and are unpacked to
  int8, a group of output channels at a time (`SCRATCH` = 8 KB), right before each kernel call.
- **ram** (`--weights ram`): the weights are unpacked to int8 in RAM once, at the first call, as TensorFlow Lite Micro
  does with its INT4 weights.

The weight bytes reported are the packed codes plus, per output channel, an int32 bias, an int32 multiplier and its
shift.
