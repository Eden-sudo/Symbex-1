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

## Next

1. Their benchmark program on the current SDK (the cycle counters), their kernels unchanged; their published cycles
   reproduced.
2. 1D layers in lib_symbex (1 × k kernels, no padding, OR pooling of 4) and a HAL using the PULP bit count instruction.
3. The comparison layer by layer and network by network, then INT8 and the boards.
