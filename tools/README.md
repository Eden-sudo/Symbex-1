# The CDT: from your model to your board

The CDT (`cdt.py`) takes a trained model, your data and the limits of your microcontroller, and gives back a **binary
network that runs on the chip with lib_symbex**: trained, folded into integers, written as a C header
(`model_symbex.h`), and checked bit by bit against its integer simulation, on this PC and on the board.

```
python cdt.py my_model.toml                                    # everything the file says
python cdt.py my_model.toml --board esp32 --port /dev/ttyUSB0  # ... and measured on a board
python chip_test.py <output folder> --board s3 --port /dev/ttyUSB0   # an export, on a board, again
```

The header runs in any project with lib_symbex: `Symbex1_PIO/src/main.cpp` is the minimal example.

## The stages

| Stage | What it does |
|---|---|
| **inspect** | reads the model: its layers, parameters and precision; chooses the conv or dense path |
| **estimate** | dense networks: how much each layer suffers at 1, 2, 4, 8 bits, and the memory of each choice |
| **train** | trains the binary network (dense: searches width and bits per layer within your budget; boolean: the network as defined) |
| **export** | folds it into integers, writes `model_symbex.h`, and checks the compiled C against the integer simulation (**must be 100 %**) |
| **chip** | flashes a board, sends the test images over the serial port, measures accuracy, latency per layer and RAM, and checks the chip against the simulation (**100 %** again) |

`[run] from / to` (or `--from` / `--to`) runs a part: `--from export` exports a model trained before; `--from chip`
only measures an existing export.

## Writing a symbex.toml

`configs/example.toml` has every option, commented: copy it and change what you need. Only what differs from the
defaults has to be written. The main ones:

| Section | Key | What it is |
|---|---|---|
| `[model]` | `file` | the trained model (`.pth`) |
| | `definition` | `"model.py:ClassName"` when the file holds only the weights (the usual `state_dict`) |
| | `method` | `"plain"` (dense networks, 1 bit), `"boolean"` (fully binary convolutional networks), `"ternary"` (dense, weights in {-1, 0, +1}) |
| `[data]` | `directory`, `format` | your data: `"npy"`, `"imagefolder"`, or a torchvision dataset (`"cifar10"`, `"mnist"`, `"fashionmnist"`...) |
| | `validation_split` | the part of the training data kept for validation: every choice is made on it; the test set is evaluated once, at the end |
| | `split_seed` | the seed of that split: fixed (e.g. 42), every training seed is validated on the same images; `-1` uses the training seed |
| `[hardware]` | `ram_kb`, `flash_kb` | your microcontroller |
| `[target]` | `budget`, `max_accuracy_drop` | how much of the hardware the model may use, and how many accuracy points it may lose (dense search) |
| `[training]` | `epochs`, `learning_rate`, `batch_size`, `seed` | the training |
| | `data_on_gpu`, `compile` | faster training (below) |
| `[output]` | `directory`, `lib_path` | where the results go; where lib_symbex is |
| `[chip]` | `board`, `port`, `popcount`, `weights` | the board (`"esp32"`, `"s3"`), how it counts bits, where the weights live (`"flash"` or `"ram"`) |

Examples that work as they are: `benchmarks/cnn_cifar_resnet8/symbex.toml` (boolean),
`benchmarks/mlp_fashion/symbex.toml` (plain), `configs/ternary_fashion.toml` (ternary).

## Command-line options

Any key of the file can be changed with a flag, which always wins over the file: `--epochs 10`,
`--learning-rate 0.001`. A key that exists in two sections carries the section's name: `--output-directory`,
`--data-directory`. Yes/no keys are written alone: `--data-on-gpu` turns one on, `--no-data-on-gpu` turns it off.
`python cdt.py --help` lists them all; `--show-config` prints the configuration that would run, and where every value
came from (default, file, or flag).

## What the output folder holds

| File | What it is |
|---|---|
| `config_used.toml` | the exact configuration of this run (every result can be reproduced from it) |
| `train.json` | the training: its history, the validation accuracy, and the test accuracy, measured once |
| `model_binarized.pth` | the trained binary network |
| `model_symbex.h` | the C header for lib_symbex |
| `export.json` | flash, RAM, layers, fidelity (PyTorch against the integers) and exactness (C against the integers) |
| `chip.json` | what the board measured |

## Faster training

For the boolean method, in `[training]`:

```
data_on_gpu = true
compile = "reduce-overhead"
```

The training set goes to the GPU once, and the training step is recorded once and replayed (CUDA graphs). On the binary
ResNet-8: **8.6 → 3.0 minutes** for 150 epochs, the same test accuracy, the same exactness. The CDT detects the
accelerator itself (an NVIDIA or AMD GPU through CUDA / ROCm; otherwise the CPU).

## What is in this folder

| | What it is |
|---|---|
| `cdt.py` | the CDT |
| `chip_test.py` | an exported model, on a board |
| `configs/` | `example.toml`: every option, commented; `ternary_fashion.toml`: an example of the ternary method |
| `chip_profiles/` | what each chip measured (`dev/count_bench.py`, the chip stage): the fastest way to count bits per layer size. The HAL's choice of arithmetic and the latency predictor of the sizing studies in `benchmarks/` read them. Published: they are measurements |
| `compiler_core/` | how the CDT is built inside: see [`compiler_core/`](compiler_core/) |
| `dev/` | development tools: measuring the chips, the library and the training speed. See [`dev/`](dev/) |
| `runs/` | what the tools generate (firmware projects, kernel measurements): rebuilt on every run, not published |
