# har_ultracompact — binary 1D convolutions for sensor data (in preparation)

Human activity recognition from accelerometer signals, with binary **1D** convolutional networks, compared against the
closest prior work: the binary inference library of **Daghero et al., "Ultra-compact Binary Neural Networks for Human
Activity Recognition on RISC-V Processors"** (ACM Computing Frontiers 2021), whose code is public
(`github.com/francescodaghero/ultracompactBNN`).

The plan is a comparison in their own environment, so their numbers can be reproduced first:

| | Their library | SYMBEX |
|---|---|---|
| Processor | RI5CY (PULP), simulated with GVSoC | the same |
| Compiler | PULP RISC-V GCC (`__builtin_pulp_cnt`: hardware bit count) | the same, the same instruction |
| Layers | their kernels, unchanged | lib_symbex with 1D support |
| Data | UniMiB-SHAR, their architectures | the same |

Plus TensorFlow Lite Micro INT8 with the same architectures, and the same networks on the boards of the other benchmarks.

## Done so far

- **Their data layout is understood and checked:** `ucbnn_check.py` writes their layer data (bit streams, most
  significant bit first, time-major) and runs their 1D convolution, compiled with plain gcc, against the expected
  output: **60/60 random layers equal, bit by bit**. Their library supports power-of-2 output channel counts, as their
  paper states.
- **The simulator runs:** `pulp/Dockerfile` builds the PULP compiler, the PULP SDK and GVSoC (pulp-open target, whose
  fabric controller is a RI5CY core) on Ubuntu 22.04.

```
python ucbnn_check.py --repo <path to the ultracompactBNN clone>
cd pulp && docker build -t symbex-pulp .
docker run --rm -it -v ~/Proyectos:/work symbex-pulp        # then, inside: cd /opt/pulp-sdk/tests/hello && make clean all run
```

## Results so far (working notes: conclusions come at the end)

### 1. One convolution layer, the same layer through both libraries (GVSoC, RI5CY)

Same bits, weights and thresholds; the same core, compiler and flags (PULP GCC, `-O3 -flto`); both count bits with
the `p.cnt` instruction (SYMBEX: the generic HAL with `-DSYMBEX_POPCOUNT_FORCE_NATIVE`; checked in its binary). Both
outputs exact on every layer. `ucbnn_gvsoc.py --symbex`:

| Layer (L_in × C_in × C_out × K) | MAC | ultracompactBNN instr / cycles | SYMBEX v1 instr / cycles | Instr ratio | Cycle ratio |
|---|---|---|---|---|---|
| 151 × 3 × 8 × 15 | 49 320 | 43 616 / 94 461 | 22 347 / 51 944 | **1.95×** | **1.82×** |
| 137 × 8 × 8 × 15 | 118 080 | 53 514 / 114 286 | 36 001 / 78 015 | **1.49×** | **1.46×** |
| 64 × 16 × 16 × 7 | 103 936 | 32 003 / 68 320 | 27 507 / 64 268 | 1.16× | 1.06× |
| 128 × 32 × 32 × 15 | 1 751 040 | 317 789 / 703 884 | 286 547 / 641 450 | 1.11× | 1.10× |
| 64 × 32 × 32 × 7 | 415 744 | 76 895 / 164 871 | 85 412 / 206 541 | 0.90× | 0.80× |
| 32 × 64 × 64 × 3 | 368 640 | 69 015 / 149 593 | 78 652 / 202 383 | 0.88× | 0.74× |

Ratio = ultracompactBNN / SYMBEX (> 1: SYMBEX faster).

Notes:
- **Few channels** (their target, "ultra-compact"): SYMBEX v1 needs up to 1.95× fewer instructions. It cuts each window
  once and counts every output channel against it; their kernel re-aligns the bits with shifts and masks word by word.
- **Many channels, short kernels:** their kernel needs up to 1.14× fewer instructions (1.35× fewer cycles). Its hand-made
  2 × 2 unrolling reads each weight once for two time steps and keeps the sums in registers; SYMBEX v1 reads each weight
  row once per output and keeps the counts in memory before the thresholds.
- **The environment:** the pulp-open platform of the current GVSoC (PULPissimo, their chip, is no longer a buildable
  target); its memory is slower for the core than PULPissimo's: about 2.2 cycles per instruction for both libraries, so
  instructions are the cleaner comparison and both are reported.
- **Channel counts:** their library gives wrong outputs with output channel counts that are not powers of 2 (5, 7:
  checked on the PC); SYMBEX takes any count.

### 2. SYMBEX v2: two windows per pass over the weights

Where the compiler counts bits with one instruction, the 1D convolution now counts two time steps per pass: each weight
word is read once for two windows, the sums stay in registers and meet their thresholds at once (the HAL row functions
stay in use on the ESP32 and the ESP32-S3). Same layers, same environment:

| Layer (L_in × C_in × C_out × K) | ultracompactBNN instr / cycles | SYMBEX v2 instr / cycles | Instr ratio | Cycle ratio |
|---|---|---|---|---|
| 151 × 3 × 8 × 15 | 43 616 / 94 461 | 22 101 / 51 366 | **1.97×** | **1.84×** |
| 137 × 8 × 8 × 15 | 53 514 / 114 286 | 28 450 / 62 387 | **1.88×** | **1.83×** |
| 64 × 16 × 16 × 7 | 32 003 / 68 320 | 23 420 / 55 039 | **1.37×** | **1.24×** |
| 64 × 32 × 32 × 7 | 76 895 / 164 871 | 71 032 / 154 210 | **1.08×** | **1.07×** |
| 128 × 32 × 32 × 15 | 317 789 / 703 884 | 267 729 / 561 362 | **1.19×** | **1.25×** |
| 32 × 64 × 64 × 3 | 69 015 / 149 593 | 61 040 / 137 426 | **1.13×** | **1.09×** |

SYMBEX v2 needs fewer instructions and fewer cycles on every layer, both outputs exact: 1.07-1.25× with many channels,
1.8× with few, the ultra-compact networks their library was designed for.

## Next

1. Their benchmark program on the current SDK (the cycle counters), their kernels unchanged; their published cycles
   reproduced.
2. 1D layers in lib_symbex (1 × k kernels, no padding, OR pooling of 4) and a HAL using the PULP bit count instruction.
3. The comparison layer by layer and network by network, then INT8 and the boards.
