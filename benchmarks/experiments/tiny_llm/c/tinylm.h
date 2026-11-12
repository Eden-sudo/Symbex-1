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
    int n_in, n_out, words_in;
} TinyLinear;

typedef struct {
    const int32_t *n1, *n2;
    TinyLinear q, k, v, o, w1, w3, w2;
} TinyBlock;

typedef struct {
    int vocab, dim, hidden, layers, heads, seq;
    const int8_t *emb_q;     /* vocab x dim */
    const int32_t *emb_m;    /* vocab */
    const int32_t *norm_f;   /* dim */
    const int16_t *cos, *sin;/* seq x (dim / heads / 2) */
    const uint16_t *exp_t;   /* TINYLM_EXP_N */
    const uint16_t *sig_t;   /* TINYLM_SIG_N */
    TinyBlock *blocks;       /* layers */
} TinyModel;

typedef struct {
    int8_t *kq, *vq;         /* layers x seq x dim */
    int32_t *km, *vm;        /* layers x seq x heads */
    int32_t *x, *t1, *t2, *t3; /* dim, dim, hidden, hidden scratch */
    int32_t *q, *k, *v, *a;  /* dim */
    int8_t *iq;              /* max(dim, hidden) */
    uint16_t *p;             /* seq */
    int16_t *lut;            /* max(dim, hidden) / 8 x 256: sums of every subset of 8 inputs (lut8 kernel) */
    int pos;
} TinyState;

/* Linear layers: the 8-bit table kernel (one lookup per 8 weights); -DTINYLM_REFERENCE = one bit at a time.
   Both give the same integers. */
/* Parse model.bin (kept in memory by the caller). Returns 0 if the header is valid. */
int tinylm_load(TinyModel *m, const uint8_t *buf, long size);
/* Allocate the cache and the scratch buffers (malloc). Returns 0 on success. */
int tinylm_state_init(TinyState *s, const TinyModel *m);
void tinylm_state_free(TinyState *s);
/* One token: z (vocab, int64) gets the output, *hmax the final state's max (Q12). Returns argmax(z), or -1 when
   the context is full (pos == seq: the C has no sliding window) or the token is out of range. */
int tinylm_step(const TinyModel *m, TinyState *s, int token, int64_t *z, int64_t *hmax);

#endif
