/* sys/mman.h stub for the chips (no mmap): run.c's read_checkpoint compiles but the firmware never calls it, the
   weights come from a const array in flash (../../src/l2c_mcu.c). The PC build uses the system header instead. */
#pragma once
#include <stddef.h>
#define PROT_READ 1
#define MAP_PRIVATE 2
#define MAP_FAILED ((void *)-1)
static inline void *mmap(void *a, size_t n, int prot, int flags, int fd, long off) {
    (void)a; (void)n; (void)prot; (void)flags; (void)fd; (void)off;
    return MAP_FAILED;
}
static inline int munmap(void *a, size_t n) { (void)a; (void)n; return 0; }
