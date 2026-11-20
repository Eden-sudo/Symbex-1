#ifndef LITTLEBIT_BLOB_H
#define LITTLEBIT_BLOB_H
#include <stddef.h>
#include <stdint.h>
/* LBT1 layout is specified in littlebit_blob.py. No allocation, float math,
 * alignment or native-endian assumptions. x and y must not overlap blob or
 * each other. Lengths are element counts. Returns -1 for invalid input.
 * Every header/payload/output bound is validated before writing y. */
int littlebit_blob_matvec(const uint8_t *blob, size_t size,
                         const int8_t *x, size_t nx, int32_t *y, size_t ny);
#endif
