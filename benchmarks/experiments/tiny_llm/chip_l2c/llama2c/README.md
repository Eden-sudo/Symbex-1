`run.c` and `LICENSE` copied unchanged from [karpathy/llama2.c](https://github.com/karpathy/llama2.c) (MIT), commit
350e04fe35433e6d2941dce5a1f53308f87058eb. `sys/mman.h` is a stub for the chips (no files, no mmap): `run.c` builds
with `-DTESTING` (no `main`) and the firmware maps the weights from a const array instead (`../src/l2c_mcu.c`).
