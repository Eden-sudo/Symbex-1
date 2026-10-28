/*
 * onebit_gemv.c -- see onebit_gemv.h. Portable C99; the same file is compiled by the host tests
 * (tests/test_edge_llm_kernels.py) and by the PlatformIO firmware (src/main.cpp).
 */
#include "onebit_gemv.h"

/* 32-bit popcount, SWAR. The same idea as popcount_swar in Symbex1_PIO/lib/lib_symbex/src/hal/generic. */
static inline uint32_t popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

/* ---- activation preparation ---- */

void lut4_build(const int8_t* x, int n, int16_t* lut) {
    for (int g = 0; g < n / 4; g++) {
        const int a = x[4 * g], b = x[4 * g + 1], c = x[4 * g + 2], d = x[4 * g + 3];
        /* v = (bits of c, d) << 2 | (bits of a, b): entry = low pair sum + high pair sum */
        const int lo[4] = {-a - b, a - b, -a + b, a + b};
        const int hi[4] = {-c - d, c - d, -c + d, c + d};
        int16_t* t = lut + 16 * g;
        for (int v = 0; v < 16; v++) t[v] = (int16_t)(lo[v & 3] + hi[v >> 2]);
    }
}

int32_t bs4_pack(const uint8_t* a, int n, uint32_t* planes) {
    const int nw = n / 32;
    int32_t sum = 0;
    for (int b = 0; b < 4; b++)
        for (int w = 0; w < nw; w++) planes[b * nw + w] = 0u;
    for (int i = 0; i < n; i++) {
        const uint32_t v = a[i] & 15u;
        sum += (int32_t)v;
        for (int b = 0; b < 4; b++) planes[b * nw + i / 32] |= ((v >> b) & 1u) << (i % 32);
    }
    return sum;
}

void onebit_rowsums(const uint32_t* w, int m, int n, int32_t* rowsum) {
    const int nw = n / 32;
    for (int r = 0; r < m; r++) {
        int32_t pc = 0;
        for (int k = 0; k < nw; k++) pc += (int32_t)popcount32(w[r * nw + k]);
        rowsum[r] = 2 * pc - n;
    }
}

void xnor_pack(const int8_t* x, int n, uint32_t* bits) {
    for (int k = 0; k < n / 32; k++) bits[k] = 0u;
    for (int i = 0; i < n; i++)
        if (x[i] > 0) bits[i / 32] |= 1u << (i % 32);
}

/* ---- products ---- */

void gemv_ref(const uint32_t* w, const int8_t* x, int32_t* y, int m, int n) {
    const int nw = n / 32;
    for (int r = 0; r < m; r++) {
        int32_t acc = 0;
        for (int i = 0; i < n; i++) {
            const uint32_t bit = (w[r * nw + i / 32] >> (i % 32)) & 1u;
            acc += (bit ? 1 : -1) * (int32_t)x[i];
        }
        y[r] = acc;
    }
}

void gemv_mask(const uint32_t* w, const int8_t* x, int32_t* y, int m, int n) {
    const int nw = n / 32;
    int32_t sum_x = 0;
    for (int i = 0; i < n; i++) sum_x += x[i];
    for (int r = 0; r < m; r++) {
        const uint32_t* row = w + r * nw;
        int32_t s = 0;
        for (int k = 0; k < nw; k++) {
            const uint32_t word = row[k];
            const int8_t* xk = x + 32 * k;
            for (int b = 0; b < 32; b++) s += (int32_t)xk[b] & -(int32_t)((word >> b) & 1u);
        }
        y[r] = 2 * s - sum_x;
    }
}

void gemv_lut4_rows(const uint32_t* w, const int16_t* lut, int32_t* y, int n, int r0, int r1) {
    const int nw = n / 32;
    for (int r = r0; r < r1; r++) {
        const uint32_t* row = w + r * nw;
        const int16_t* t = lut;
        int32_t acc = 0;
        for (int k = 0; k < nw; k++) {
            const uint32_t word = row[k];
            acc += t[0 * 16 + (word & 15u)];
            acc += t[1 * 16 + ((word >> 4) & 15u)];
            acc += t[2 * 16 + ((word >> 8) & 15u)];
            acc += t[3 * 16 + ((word >> 12) & 15u)];
            acc += t[4 * 16 + ((word >> 16) & 15u)];
            acc += t[5 * 16 + ((word >> 20) & 15u)];
            acc += t[6 * 16 + ((word >> 24) & 15u)];
            acc += t[7 * 16 + (word >> 28)];
            t += 8 * 16;
        }
        y[r] = acc;
    }
}

void gemv_lut4(const uint32_t* w, const int16_t* lut, int32_t* y, int m, int n) {
    gemv_lut4_rows(w, lut, y, n, 0, m);
}

void gemv_bitserial4(const uint32_t* w, const uint32_t* planes, int32_t sum_a, const int32_t* rowsum,
                     int32_t* y, int m, int n) {
    const int nw = n / 32;
    const uint32_t* p0 = planes;
    const uint32_t* p1 = planes + nw;
    const uint32_t* p2 = planes + 2 * nw;
    const uint32_t* p3 = planes + 3 * nw;
    for (int r = 0; r < m; r++) {
        const uint32_t* row = w + r * nw;
        int32_t acc = 0; /* sum over planes b of 2^b * popcount(A_b & W) */
        for (int k = 0; k < nw; k++) {
            const uint32_t word = row[k];
            acc += (int32_t)popcount32(p0[k] & word);
            acc += (int32_t)(popcount32(p1[k] & word) << 1);
            acc += (int32_t)(popcount32(p2[k] & word) << 2);
            acc += (int32_t)(popcount32(p3[k] & word) << 3);
        }
        /* sum_i a_i^b w_i = 2 popcount(A_b & W) - popcount(A_b); the planes' popcounts add up to sum(a);
           the -8 offset of the activations contributes -8 * sum(w) */
        y[r] = 2 * acc - sum_a - 8 * rowsum[r];
    }
}

void gemv_xnor(const uint32_t* w, const uint32_t* xbits, int32_t* y, int m, int n) {
    const int nw = n / 32;
    for (int r = 0; r < m; r++) {
        const uint32_t* row = w + r * nw;
        int32_t pc = 0;
        for (int k = 0; k < nw; k++) pc += (int32_t)popcount32(xbits[k] ^ row[k]);
        y[r] = n - 2 * pc;
    }
}
