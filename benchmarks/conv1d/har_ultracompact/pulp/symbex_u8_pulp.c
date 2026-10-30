/*
 * symbex_u8_pulp.c -- lib_symbex's first layer (uint8 input x 1-bit weights) for PULP's RI5CY core, with its 4-way
 * dot products of bytes (four products added in one instruction). Two versions, chosen by the library flag:
 *   -DSYMBEX_U8_EXTERNAL         the HAL row functions symbex_u8_* (p.sdotup4, one time step per call); the library's
 *                                layer loop is kept
 *   -DSYMBEX_BOOL1D_U8_EXTERNAL  the whole layer symbex_bool1d_conv_u8 (p.sdotusp4): see the second half of the file
 *
 * The network and its bits do not change: the weight bits of the layer are expanded ONCE into bytes (0 or 1) in a RAM
 * cache, the first time the layer runs; then every output is the dot product of the input bytes with those bytes:
 * sum of x where the weight bit is 1, exactly what the HAL contract asks. Four rows share every input load.
 *   cache       n_rows * stride * 32 bytes (the HAR first layer of Table 1, 32 x 4 words: 4 KB); larger layers fall
 *               back to the plain bit loop
 *   -DSYMBEX_U8_PULP_DOTP   use the PULP builtin (PULP GCC); without it a C stand-in (this PC), for the exactness check
 */
#include <stdint.h>
#include "hal/symbex_hal.h"

#if defined(SYMBEX_U8_EXTERNAL)

#ifndef SYMBEX_U8_PULP_CACHE_BYTES
#define SYMBEX_U8_PULP_CACHE_BYTES 8192
#endif

typedef unsigned char u8x4 __attribute__((vector_size(4)));

#if defined(SYMBEX_U8_PULP_DOTP)
#define DOTP(a, b, acc) __builtin_pulp_sdotup4((a), (b), (acc))
#else
static inline uint32_t dotp_c(u8x4 a, u8x4 b, uint32_t acc) {
    return acc + a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
#define DOTP(a, b, acc) dotp_c((a), (b), (acc))
#endif

static u8x4 cache[SYMBEX_U8_PULP_CACHE_BYTES / 4];
static const uint32_t* cached_w;
static int cached_rows, cached_stride;

/* the weight bits -> bytes 0 / 1 (byte i of row r = bit i of the row), once per layer */
static int expand(const uint32_t* w, int stride, int n_rows) {
    if (w == cached_w && stride == cached_stride && n_rows == cached_rows) return 1;
    if (n_rows * stride * 32 > SYMBEX_U8_PULP_CACHE_BYTES) return 0;
    uint8_t* b = (uint8_t*)cache;
    for (int r = 0; r < n_rows; r++)
        for (int j = 0; j < stride; j++)
            for (int k = 0; k < 32; k++) *b++ = (uint8_t)((w[r * stride + j] >> k) & 1u);
    cached_w = w;
    cached_stride = stride;
    cached_rows = n_rows;
    return 1;
}

void symbex_u8_prepare(const uint8_t* x, int n_inputs, uint8_t* workspace, int n_words) {
    const int n = (n_inputs + 3) & ~3;                   /* whole groups of 4, the last one zero padded */
    (void)n_words;
    for (int i = 0; i < n; i++) workspace[i] = (i < n_inputs) ? x[i] : 0;
}

static int32_t bit_loop(const uint8_t* x, const uint32_t* w, int n_inputs) {
    int32_t s = 0;
    for (int i = 0; i < n_inputs; i++) s += ((w[i >> 5] >> (i & 31)) & 1u) ? x[i] : 0;
    return s;
}

int32_t symbex_u8_sum_bits(const uint8_t* workspace, const uint32_t* w, int n_words) {
    return bit_loop(workspace, w, n_words * 32);
}

void symbex_u8_sum_bits_rows(const uint8_t* workspace, int n_inputs, const uint32_t* w, int stride, int n_rows,
                             int32_t* out) {
    if (n_inputs > stride * 32) n_inputs = stride * 32;
    if (!expand(w, stride, n_rows)) {
        for (int r = 0; r < n_rows; r++) out[r] = bit_loop(workspace, w + r * stride, n_inputs);
        return;
    }
    const u8x4* xv = (const u8x4*)workspace;
    const int nv = (n_inputs + 3) >> 2;
    const int row_v = stride * 8;                        /* u8x4 per expanded row */
    int r = 0;
    for (; r + 3 < n_rows; r += 4) {
        const u8x4* b0 = cache + r * row_v;
        const u8x4* b1 = b0 + row_v;
        const u8x4* b2 = b1 + row_v;
        const u8x4* b3 = b2 + row_v;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int j = 0; j < nv; j++) {
            const u8x4 v = xv[j];
            a0 = DOTP(v, b0[j], a0);
            a1 = DOTP(v, b1[j], a1);
            a2 = DOTP(v, b2[j], a2);
            a3 = DOTP(v, b3[j], a3);
        }
        out[r] = (int32_t)a0;
        out[r + 1] = (int32_t)a1;
        out[r + 2] = (int32_t)a2;
        out[r + 3] = (int32_t)a3;
    }
    for (; r < n_rows; r++) {
        const u8x4* b0 = cache + r * row_v;
        uint32_t a0 = 0;
        for (int j = 0; j < nv; j++) a0 = DOTP(xv[j], b0[j], a0);
        out[r] = (int32_t)a0;
    }
}
#endif // SYMBEX_U8_EXTERNAL

#if defined(SYMBEX_BOOL1D_U8_EXTERNAL)
/*
 * The whole layer (1-bit or int8 weights: SymbexBool1D.weights_i8), S time steps x four output channels per pass (each input load serves 4 rows, each weight load S
 * steps: S = 3, 7 loads for 12 dot products). The weight bits are expanded ONCE into signed bytes, +1 where the bit
 * is 1 and -1 where it is 0, so one unsigned x signed dot product (p.sdotusp4) gives raw = sum of x * (+1 / -1) directly: no
 * input sum, no copy of the window (RI5CY loads 4 unaligned bytes in one instruction). When kernel * in_ch is not a
 * multiple of 4, the last group of 4 bytes is read ENDING at the window's last byte (nothing is read past the window)
 * and the weights of the bytes it repeats are 0: one more dot product instead of a byte loop.
 *   cache       out_ch rows of kernel * in_ch bytes rounded up to 4, plus 4 when there is a tail (HAR, Table 1 max:
 *               32 x 48 = 1.5 KB of RAM);
 *               out_ch not a multiple of 4, or a larger layer: a plain loop (exact, slow)
 *               The cache is found again by the weights' address: the weights must not change between calls (they
 *               are const, as har_export.py writes them)
 */
#include <string.h>
#include "layers/conv/symbex_bool1d.h"

#ifndef SYMBEX_BOOL1D_U8_PULP_CACHE_BYTES
#define SYMBEX_BOOL1D_U8_PULP_CACHE_BYTES 4096
#endif
#ifndef SYMBEX_BOOL1D_U8_PULP_STEPS
#define SYMBEX_BOOL1D_U8_PULP_STEPS 3                       /* time steps per pass: 3 measured best on GVSoC (2-4) */
#endif
#define S SYMBEX_BOOL1D_U8_PULP_STEPS

typedef unsigned char v4u __attribute__((vector_size(4)));
typedef signed char v4s __attribute__((vector_size(4)));

#if defined(SYMBEX_U8_PULP_DOTP)
#define DOTUS(a, b, acc) __builtin_pulp_sdotusp4((a), (b), (acc))
#define LOAD_U4(p) (*(const v4u*)(p))                      /* RI5CY: misaligned loads in hardware */
#else
static inline int32_t dotus_c(v4u a, v4s b, int32_t acc) {
    return acc + a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
static inline v4u load_u4(const uint8_t* p) { v4u v; memcpy(&v, p, 4); return v; }
#define DOTUS(a, b, acc) dotus_c((a), (b), (acc))
#define LOAD_U4(p) load_u4(p)
#endif

static v4s pm_cache[SYMBEX_BOOL1D_U8_PULP_CACHE_BYTES / 4];
static const void* pm_w;                             /* the weights the cache holds (1-bit rows or int8) */
static int pm_rows, pm_n;

/* row o, input i -> +1 / -1. Rows of nb bytes: the 4 * nv bytes of the whole groups, then (n % 4 != 0) the tail
   group, which covers inputs n - 4 .. n - 1: 0 for the ones a whole group already has */
static int8_t pm(const struct SymbexBool1D* L, int o, int i) {
    if (L->weights_i8) return L->weights_i8[o * L->kernel * L->in_ch + i];   /* int8 weights: copied as they are */
    return ((L->weights[o * L->row_words + (i >> 5)] >> (i & 31)) & 1u) ? 1 : -1;
}

static int expand_pm(const struct SymbexBool1D* L, int n, int nb) {
    const void* key = L->weights_i8 ? (const void*)L->weights_i8 : (const void*)L->weights;
    if (key == pm_w && L->out_ch == pm_rows && n == pm_n) return 1;
    if (L->out_ch * nb > SYMBEX_BOOL1D_U8_PULP_CACHE_BYTES) return 0;
    const int whole = n & ~3;
    int8_t* b = (int8_t*)pm_cache;
    for (int o = 0; o < L->out_ch; o++, b += nb) {
        for (int i = 0; i < whole; i++) b[i] = pm(L, o, i);
        for (int i = whole; i < nb; i++) {                       /* the tail group: input n - 4 + (i - whole) */
            const int x = n - 4 + (i - whole);
            b[i] = (x >= whole) ? pm(L, o, x) : 0;
        }
    }
    pm_w = key;
    pm_rows = L->out_ch;
    pm_n = n;
    return 1;
}

static void plain(const struct SymbexBool1D* L, const uint8_t* in, uint32_t* out, int n, uint32_t steps) {
    for (uint32_t t = 0; t < steps; t++) {
        const uint8_t* x = in + t * L->stride * L->in_ch;
        const uint32_t base = (t / L->pool) * L->out_ch;
        for (int o = 0; o < L->out_ch; o++) {
            int32_t raw = 0;
            for (int i = 0; i < n; i++) raw += x[i] * (int32_t)pm(L, o, i);
            if (raw >= L->thresholds[o]) out[(base + o) >> 5] |= 1UL << ((base + o) & 31);
        }
    }
}

void symbex_bool1d_conv_u8(const struct SymbexBool1D* L, const uint8_t* in, uint32_t* out, uint8_t* workspace,
                           int32_t* scratch) {
    (void)workspace;
    (void)scratch;
    const int n = L->kernel * L->in_ch;
    int nv = n >> 2;
    const int tail = (n & 3) != 0, nb = 4 * (nv + tail);
#if defined(SYMBEX_U8_PULP_DOTP)
    /* with -flto the layer sizes are constants and GCC unrolls the dot-product loop completely; it then runs out of
       registers and loads every weight twice (measured). The empty asm hides the count: the loop stays a hardware
       loop with its 6 loads for 8 dot products. */
    __asm__("" : "+r"(nv));
#endif
    const uint32_t steps = (uint32_t)L->out_len * L->pool;
    const uint32_t hop = (uint32_t)L->stride * L->in_ch;
    const uint32_t n_out = ((uint32_t)L->out_len * L->out_ch + 31) / 32 + 1;
    for (uint32_t i = 0; i < n_out; i++) out[i] = 0;
    if ((L->out_ch & 3) || n < 4 || !expand_pm(L, n, nb)) {
        plain(L, in, out, n, steps);
        return;
    }
    const int8_t* cache = (const int8_t*)pm_cache;
    for (uint32_t t = 0; t < steps; t += S) {
        const uint8_t* x[S];
        uint32_t b[S];
        for (int k = 0; k < S; k++) {                           /* past the last step: the last one again, same bits */
            const uint32_t tk = (t + k < steps) ? t + k : steps - 1;
            x[k] = in + tk * hop;
            b[k] = (tk / L->pool) * L->out_ch;
        }
        const int32_t* thr = L->thresholds;
        const int8_t* r0 = cache;
        for (int o = 0; o < L->out_ch; o += 4, r0 += 4 * nb, thr += 4) {
            int32_t a[S][4] = {{0}};
            /* pointers that advance (post-increment loads) */
            const uint8_t* p[S];
            for (int k = 0; k < S; k++) p[k] = x[k];
            const v4s *q0 = (const v4s*)r0, *q1 = (const v4s*)(r0 + nb), *q2 = (const v4s*)(r0 + 2 * nb),
                      *q3 = (const v4s*)(r0 + 3 * nb);
            for (int j = 0; j < nv; j++) {
                const v4s v0 = *q0++, v1 = *q1++, v2 = *q2++, v3 = *q3++;
                for (int k = 0; k < S; k++) {
                    const v4u u = LOAD_U4(p[k]);
                    p[k] += 4;
                    a[k][0] = DOTUS(u, v0, a[k][0]);
                    a[k][1] = DOTUS(u, v1, a[k][1]);
                    a[k][2] = DOTUS(u, v2, a[k][2]);
                    a[k][3] = DOTUS(u, v3, a[k][3]);
                }
            }
            if (tail) {
                const v4s v0 = *q0, v1 = *q1, v2 = *q2, v3 = *q3;
                for (int k = 0; k < S; k++) {
                    const v4u u = LOAD_U4(x[k] + n - 4);
                    a[k][0] = DOTUS(u, v0, a[k][0]);
                    a[k][1] = DOTUS(u, v1, a[k][1]);
                    a[k][2] = DOTUS(u, v2, a[k][2]);
                    a[k][3] = DOTUS(u, v3, a[k][3]);
                }
            }
            for (int k = 0; k < S; k++) {
                const uint32_t m = (uint32_t)(a[k][0] >= thr[0]) | (uint32_t)(a[k][1] >= thr[1]) << 1 |
                                   (uint32_t)(a[k][2] >= thr[2]) << 2 | (uint32_t)(a[k][3] >= thr[3]) << 3;
                out[(b[k] + o) >> 5] |= m << ((b[k] + o) & 31);
            }
        }
    }
}
#endif // SYMBEX_BOOL1D_U8_EXTERNAL
