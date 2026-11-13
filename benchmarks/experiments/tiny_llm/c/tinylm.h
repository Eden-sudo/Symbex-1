/*
 * tinylm.h -- the integer tiny language model of intsim.py, one token at a time with a KV cache (plain C99).
 * Weights: 1-bit, packed 32 per uint32 (bit i of word j = input 32*j + i, 1 = +1).
 */
#ifndef TINYLM_H
#define TINYLM_H

#include <stdint.h>

#define TINYLM_MAGIC 0x4D4C5954
/* Allocation hooks: FAST for the small per-token buffers (internal RAM on the chip), BIG for the KV cache (PSRAM). */
#ifndef TINYLM_MALLOC_FAST
#define TINYLM_MALLOC_FAST(n) malloc(n)
#endif
#ifndef TINYLM_MALLOC_BIG
#define TINYLM_MALLOC_BIG(n) malloc(n)
#endif
#define TINYLM_EXP_N 2048
#define TINYLM_SIG_N 1024

typedef struct {
    const uint32_t *w;  /* out rows x words_in */
    const int32_t *m;   /* out */
    const int32_t *ones;/* out: weight bits set per row (only with TINYLM_U8_SUM_ROWS, else NULL) */
    int n_in, n_out, words_in;
    int acts, had;      /* input bits (8, or 1: XOR + popcount) and Hadamard block (0: none); model.bin version 2 */
} TinyLinear;

typedef struct {
    const int32_t *n1, *n2;
    TinyLinear q, k, v, o, w1, w3, w2;
} TinyBlock;

typedef struct {
    int vocab, dim, hidden, layers, heads, seq;
    const int8_t *emb_q;     /* vocab x dim */
    const int32_t *emb_m;    /* vocab */
    const uint8_t *emb_p;    /* model.bin version 3: the int4 table packed (vocab x dim / 2, see intsim.pack_nibbles);
                                emb_q is then NULL */
    const uint8_t *emb_c;    /* model.bin version 4: the int2 table packed (vocab x dim / 4, see intsim.pack_crumbs);
                                emb_q is then NULL */
    const int32_t *norm_f;   /* dim */
    const int16_t *cos, *sin;/* seq x (dim / heads / 2) */
    const uint16_t *exp_t;   /* TINYLM_EXP_N */
    const uint16_t *sig_t;   /* TINYLM_SIG_N */
    TinyBlock *blocks;       /* layers */
} TinyModel;

typedef struct {
    int8_t *kq, *vq;         /* layers x heads x seq x head size (vq with TINYLM_DOT_ROWS_U8: ... x head size x seq,
                                value + 128) */
    int32_t *km, *vm;        /* layers x heads x seq */
    int32_t *x, *t1, *t2, *t3; /* dim, dim, hidden, hidden scratch */
    int32_t *q, *k, *v, *a;  /* dim */
    int8_t *iq;              /* max(dim, hidden) */
    uint16_t *p;             /* seq */
    int16_t *lut;            /* max(dim, hidden) / 8 x 256: sums of every subset of 8 inputs (lut8 kernel) */
    int64_t *r;              /* max(dim, hidden): the input rotated by the Hadamard block */
    uint32_t *xb;            /* max(dim, hidden) / 32: the 1-bit input, bit i of word j = input 32*j + i, 1 = +1 */
    int pos;
} TinyState;

/* Linear layers: the 8-bit table kernel (one lookup per 8 weights); -DTINYLM_REFERENCE = one bit at a time;
   TINYLM_U8_PREPARE / TINYLM_U8_SUM_ROWS defined (c/tinylm_symbex.h) = lib_symbex's 8-bit x 1-bit kernel.
   All give the same integers. */
/* the int4 table (model.bin version 3): value i of a row of dim / 2 bytes; per 32 columns, byte j holds column j in
   its low nibble and column 16 + j in its high one */
static inline int emb4(const uint8_t *row, int i) {
    const uint8_t b = row[(i >> 5) * 16 + (i & 15)];
    return (i & 16) ? (int8_t)b >> 4 : (int8_t)(uint8_t)(b << 4) >> 4;
}
/* the int2 table (model.bin version 4): value i (2c + 1, -3..3) of a row of dim / 4 bytes; per 64 columns, byte j
   holds c of columns j, 16 + j, 32 + j, 48 + j in bits 1:0, 3:2, 5:4, 7:6 */
static inline int emb2(const uint8_t *row, int i) {
    const uint8_t b = row[(i >> 6) * 16 + (i & 15)];
    return 2 * ((int8_t)(uint8_t)(b << (6 - 2 * ((i >> 4) & 3))) >> 6) + 1;
}

/* Parse model.bin (kept in memory by the caller). Returns 0 if the header is valid. */
int tinylm_load(TinyModel *m, const uint8_t *buf, long size);
/* Allocate the cache and the scratch buffers (malloc). Returns 0 on success. */
int tinylm_state_init(TinyState *s, const TinyModel *m);
void tinylm_state_free(TinyState *s);
/* One token: z (vocab, int64) gets the output, *hmax the final state's max (Q12). Returns argmax(z), or -1 when
   the context is full (pos == seq: the C has no sliding window) or the token is out of range. */
int tinylm_step(const TinyModel *m, TinyState *s, int token, int64_t *z, int64_t *hmax);

#endif
