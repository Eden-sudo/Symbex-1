/*
 * tinylm.c -- the integer tiny language model, the same integers as intsim.py (see there for the formats).
 */
#include "tinylm.h"

#include <stdlib.h>
#include <string.h>

/* Time per part of a token (optional): TINYLM_PROF(k) adds the time since the previous mark to part k
   (0 embedding / norms, 1 q k v, 2 attention values, 3 o, 4 w1 w3, 5 SwiGLU, 6 w2, 7 output head, 8 attention rope
   and int8 cache, 9 attention scores, 10 attention softmax, 11 attention division). Empty by default. */
#ifndef TINYLM_PROF
#define TINYLM_PROF(k)
#endif

/* TINYLM_DOT_ROWS(x, n, w, stride, n_rows, out): int8 row products out[r] = sum_i x[i] * w[r * stride + i] (the
   board's PIE kernel): the output head and the attention scores run on it. TINYLM_DOT_ROWS_U8: the same with
   unsigned bytes: the attention values run on it and the V cache is kept transposed (see tinylm_step).
   TINYLM_DOT_ROWS_C: both hooks in plain C, to run that path on the PC (check_pc). */
#ifdef TINYLM_DOT_ROWS_C
static void tinylm_dot_rows_c(const int8_t *x, int n, const int8_t *w, int stride, int n_rows, int32_t *out) {
    for (int r = 0; r < n_rows; r++, w += stride) {
        int32_t d = 0;
        for (int i = 0; i < n; i++) d += x[i] * w[i];
        out[r] = d;
    }
}
static void tinylm_dot_rows_u8_c(const uint8_t *x, int n, const uint8_t *w, int stride, int n_rows, int32_t *out) {
    for (int r = 0; r < n_rows; r++, w += stride) {
        int32_t d = 0;
        for (int i = 0; i < n; i++) d += x[i] * w[i];
        out[r] = d;
    }
}
/* the int4 rows as the S3 does them: per 16-byte group, high nibbles = byte & 0xF0 (16 x value), low nibbles =
   (byte << 4) as int8 (16 x value), int8 products with x[32g + 16 ..] and x[32g ..], the sum / 16 */
static void tinylm_dot_rows4_c(const int8_t *x, int n, const uint8_t *w, int n_rows, int32_t *out) {
    for (int r = 0; r < n_rows; r++, w += n / 2) {
        int32_t d = 0;
        for (int g = 0; g < n / 32; g++)
            for (int j = 0; j < 16; j++) {
                const uint8_t b = w[g * 16 + j];
                d += x[32 * g + j] * (int8_t)(uint8_t)(b << 4) + x[32 * g + 16 + j] * (int8_t)(b & 0xF0);
            }
        out[r] = d >> 4;
    }
}
#define TINYLM_DOT_ROWS4(x, n, w, n_rows, out) tinylm_dot_rows4_c((x), (n), (w), (n_rows), (out))
/* the int2 rows as the S3 does them: per 16-byte group, crumb k moved to the top bits (byte << (6 - 2k)) & 0xC0 = 64 c
   as int8, products with x[64g + 16k ..], the sum / 64 = sum x c; the dot is 2 sum x c + sum x */
static void tinylm_dot_rows2_c(const int8_t *x, int n, const uint8_t *w, int n_rows, int32_t *out) {
    int32_t sx = 0;
    for (int i = 0; i < n; i++) sx += x[i];
    for (int r = 0; r < n_rows; r++, w += n / 4) {
        int32_t d = 0;
        for (int g = 0; g < n / 64; g++)
            for (int k = 0; k < 4; k++)
                for (int j = 0; j < 16; j++)
                    d += x[64 * g + 16 * k + j] * (int8_t)((uint8_t)(w[g * 16 + j] << (6 - 2 * k)) & 0xC0);
        out[r] = 2 * (d >> 6) + sx;
    }
}
#define TINYLM_DOT_ROWS2(x, n, w, n_rows, out) tinylm_dot_rows2_c((x), (n), (w), (n_rows), (out))
#define TINYLM_DOT_ROWS(x, n, w, stride, n_rows, out) tinylm_dot_rows_c((x), (n), (w), (stride), (n_rows), (out))
#define TINYLM_DOT_ROWS_U8(x, n, w, stride, n_rows, out) tinylm_dot_rows_u8_c((x), (n), (w), (stride), (n_rows), (out))
#endif

#define QF 12
#define MSH 30

/* TINYLM_DOT_ROWS4(x, n, w, n_rows, out): out[r] = sum_i x[i] * w4[r][i] over int4 rows of n / 2 bytes (the board's
   kernel for the int4 output head); plain C (emb4) otherwise. TINYLM_DOT_ROWS2: the same over int2 rows of n / 4
   bytes (values 2c + 1, emb2). */
#define SIG_Q 6

static int64_t rdiv(int64_t n, int64_t d) {  /* rounded half away from zero, d > 0 */
    int64_t a = n < 0 ? -n : n;
    int64_t r = (2 * a + d) / (2 * d);
    return n < 0 ? -r : r;
}

static int64_t isqrt64(int64_t n) {          /* floor(sqrt(n)), n >= 0 */
    uint64_t x = (uint64_t)n, r = 0, bit = (uint64_t)1 << 62;
    while (bit > x) bit >>= 2;
    while (bit) {
        if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; }
        else r >>= 1;
        bit >>= 2;
    }
    return (int64_t)r;
}

/* int8 quantization, absmax: q = rdiv(127 x, max), returns max (>= 1) */
static int32_t quant8(const int32_t *x, int n, int8_t *q) {
    int32_t mx = 1;
    for (int i = 0; i < n; i++) { int32_t a = x[i] < 0 ? -x[i] : x[i]; if (a > mx) mx = a; }
    for (int i = 0; i < n; i++) q[i] = (int8_t)rdiv(127 * (int64_t)x[i], mx);
    return mx;
}

/* RMSNorm + int8: q, returns the normalized max (Q12, >= 1) */
static int32_t norm_quant(const int32_t *x, const int32_t *g, int n, int32_t *h, int8_t *q) {
    int64_t ss = 0;
    for (int i = 0; i < n; i++) ss += (int64_t)x[i] * x[i];
    int64_t rms = isqrt64(ss / n);
    if (rms < 1) rms = 1;
    for (int i = 0; i < n; i++) h[i] = (int32_t)(((int64_t)x[i] * g[i] + (1 << (QF - 1))) >> QF);
    int32_t hm = quant8(h, n, q);
    int64_t hmax = (((int64_t)hm << QF) + rms / 2) / rms;
    return (int32_t)(hmax < 1 ? 1 : hmax);
}

/* RMSNorm without the division: h = x g (Q12), returns the integer rms (>= 1); normalized input = h / rms */
static int64_t norm_h(const int32_t *x, const int32_t *g, int n, int32_t *h) {
    int64_t ss = 0;
    for (int i = 0; i < n; i++) ss += (int64_t)x[i] * x[i];
    int64_t rms = isqrt64(ss / n);
    if (rms < 1) rms = 1;
    for (int i = 0; i < n; i++) h[i] = (int32_t)(((int64_t)x[i] * g[i] + (1 << (QF - 1))) >> QF);
    return rms;
}

/* r times the unnormalized Sylvester Hadamard matrix, block by block (butterflies: additions only) */
static void fwht(int64_t *r, int n, int blk) {
    for (int b = 0; b < n; b += blk)
        for (int len = 1; len < blk; len <<= 1)
            for (int i = b; i < b + blk; i += 2 * len)
                for (int j = i; j < i + len; j++) {
                    const int64_t u = r[j], v = r[j + len];
                    r[j] = u + v;
                    r[j + len] = u - v;
                }
}

/* A linear layer's input, intsim.py quant_in: rotated when L->had, then int8 in s->iq (returns max, or max / rms in
   Q12 when rms > 0) or 1 bit (s->iq = +-1, s->xb packed; returns 127 mean|input|, divided by rms the same way) */
static int64_t in_quant(const TinyLinear *L, TinyState *s, const int32_t *h, int n, int64_t rms) {
    int64_t v;
    if (!L->had && L->acts == 8) {
        v = quant8(h, n, s->iq);
        if (rms) v = ((v << QF) + rms / 2) / rms;
        return v < 1 ? 1 : v;
    }
    int64_t *r = s->r;
    for (int i = 0; i < n; i++) r[i] = h[i];
    if (L->had) fwht(r, n, L->had);
    if (L->acts == 8) {
        int64_t mx = 1;
        for (int i = 0; i < n; i++) { const int64_t a = r[i] < 0 ? -r[i] : r[i]; if (a > mx) mx = a; }
        for (int i = 0; i < n; i++) s->iq[i] = (int8_t)rdiv(127 * r[i], mx);
        v = rms ? ((mx << QF) + rms / 2) / rms : mx;
    } else {
        int64_t sa = 0;
        memset(s->xb, 0, 4L * (n / 32));
        for (int i = 0; i < n; i++) {
            const int pos = r[i] >= 0;
            sa += pos ? r[i] : -r[i];
            s->iq[i] = (int8_t)(pos ? 1 : -1);
            if (pos) s->xb[i >> 5] |= 1u << (i & 31);
        }
        sa *= 127;
        v = rms ? rdiv(sa << QF, (int64_t)n * rms) : rdiv(sa, n);
    }
    return v < 1 ? 1 : v;
}

/* 1-bit inputs x 1-bit weights: acc = n_in - 2 popcount(input XOR weight) (n_in a multiple of 32: tinylm_load).
   TINYLM_XOR_ROWS(a, w, words, n_rows, out): out[r] = popcount(a XOR row r), rows of `words` words (c/tinylm_symbex.h:
   lib_symbex's symbex_xor_popcount_rows, PIE on the S3); plain C by default. */
static void linear_bits(const TinyLinear *L, const uint32_t *xb, int64_t scale, int32_t *out) {
#ifdef TINYLM_XOR_ROWS
    TINYLM_XOR_ROWS(xb, L->w, L->words_in, L->n_out, out);
#endif
    for (int r = 0; r < L->n_out; r++) {
#ifdef TINYLM_XOR_ROWS
        const int32_t d = out[r];
#else
        const uint32_t *w = L->w + (long)r * L->words_in;
        int32_t d = 0;
        for (int j = 0; j < L->words_in; j++) d += __builtin_popcount(w[j] ^ xb[j]);
#endif
        const int64_t acc = L->n_in - 2 * (int64_t)d;
        out[r] = (int32_t)((acc * scale * L->m[r] + ((int64_t)1 << (MSH - 1))) >> MSH);
    }
}

#if defined(TINYLM_U8_SUM_ROWS)
/* lib_symbex's 8-bit x 1-bit kernel (c/tinylm_symbex.h): sums of the unsigned inputs x + 128 over the weight bits
   that are 1; pos = that sum - 128 * (ones of the row), the same integer as the other kernels */
static void prepare(TinyState *s, const int8_t *q, int n_in) { (void)s; TINYLM_U8_PREPARE(q, n_in); }

static void linear(const TinyLinear *L, const TinyState *s, const int8_t *q, int64_t hmax, int32_t *out) {
    (void)s;
    int32_t total = 0;
    for (int i = 0; i < L->n_in; i++) total += q[i];
    TINYLM_U8_SUM_ROWS(L->n_in, L->w, L->words_in, L->n_out, out);
    for (int r = 0; r < L->n_out; r++) {
        int64_t acc = 2 * (int64_t)(out[r] - 128 * L->ones[r]) - total;
        out[r] = (int32_t)((acc * hmax * L->m[r] + ((int64_t)1 << (MSH - 1))) >> MSH);
    }
}
#elif defined(TINYLM_REFERENCE)
static void prepare(TinyState *s, const int8_t *q, int n_in) { (void)s; (void)q; (void)n_in; }

static void linear(const TinyLinear *L, const TinyState *s, const int8_t *q, int64_t hmax, int32_t *out) {
    (void)s;
    int32_t total = 0;
    for (int i = 0; i < L->n_in; i++) total += q[i];
    for (int r = 0; r < L->n_out; r++) {
        const uint32_t *w = L->w + (long)r * L->words_in;
        int32_t pos = 0;
        for (int j = 0; j < L->words_in; j++) {
            uint32_t bits = w[j];
            const int8_t *qq = q + 32 * j;
            int lim = L->n_in - 32 * j < 32 ? L->n_in - 32 * j : 32;
            for (int i = 0; i < lim; i++) if (bits >> i & 1u) pos += qq[i];
        }
        int64_t acc = 2 * (int64_t)pos - total;
        out[r] = (int32_t)((acc * hmax * L->m[r] + ((int64_t)1 << (MSH - 1))) >> MSH);
    }
}
#else
/* lut[g][b] = sum of q[8g + k] over the bits k of b (b = one weight byte: inputs 8g..8g+7, bit k = input 8g+k) */
static void prepare(TinyState *s, const int8_t *q, int n_in) {
    for (int g = 0; g < n_in / 8; g++) {
        int16_t *t = s->lut + 256 * g;
        const int8_t *qg = q + 8 * g;
        t[0] = 0;
        for (int k = 0; k < 8; k++) {
            int16_t v = qg[k];
            for (int b = 0; b < (1 << k); b++) t[(1 << k) + b] = (int16_t)(t[b] + v);
        }
    }
}

static void linear(const TinyLinear *L, const TinyState *s, const int8_t *q, int64_t hmax, int32_t *out) {
    int32_t total = 0;
    for (int i = 0; i < L->n_in; i++) total += q[i];
    const int groups = L->n_in / 8;
    for (int r = 0; r < L->n_out; r++) {
        const uint32_t *w = L->w + (long)r * L->words_in;
        const int16_t *t = s->lut;
        int32_t pos = 0;
        int g = 0;
        for (; g + 4 <= groups; g += 4, t += 1024) {     /* one 32-bit load per 4 groups (bytes little-endian) */
            uint32_t b = w[g >> 2];
            pos += t[b & 255] + t[256 + (b >> 8 & 255)] + t[512 + (b >> 16 & 255)] + t[768 + (b >> 24)];
        }
        for (; g < groups; g++, t += 256) pos += t[((const uint8_t *)w)[g]];
        int64_t acc = 2 * (int64_t)pos - total;
        out[r] = (int32_t)((acc * hmax * L->m[r] + ((int64_t)1 << (MSH - 1))) >> MSH);
    }
}
#endif

/* one linear layer on the input in_quant left in s: XOR + popcount for 1-bit inputs, else the int8 kernel (its
   prepare already run on s->iq) */
static void run_linear(const TinyLinear *L, TinyState *s, int64_t scale, int32_t *out) {
    if (L->acts == 1) linear_bits(L, s->xb, scale, out);
    else linear(L, s, s->iq, scale, out);
}

static void rope(int32_t *x, const int16_t *c, const int16_t *s, int hd) {
    for (int i = 0; i < hd / 2; i++) {
        int64_t x1 = x[2 * i], x2 = x[2 * i + 1];
        x[2 * i] = (int32_t)((x1 * c[i] - x2 * s[i] + (1 << 13)) >> 14);
        x[2 * i + 1] = (int32_t)((x1 * s[i] + x2 * c[i] + (1 << 13)) >> 14);
    }
}

int tinylm_step(const TinyModel *m, TinyState *s, int token, int64_t *z, int64_t *hmax_out) {
    const int dim = m->dim, hd = dim / m->heads, t = s->pos;
    if (t >= m->seq || token < 0 || token >= m->vocab) return -1;  /* context full (no sliding window) or bad token */
    int32_t *x = s->x;
    for (int i = 0; i < dim; i++) {
        const int e = m->emb_p ? emb4(m->emb_p + (long)token * (dim / 2), i)
                    : m->emb_c ? emb2(m->emb_c + (long)token * (dim / 4), i) : m->emb_q[(long)token * dim + i];
        x[i] = (int32_t)(((int64_t)e * m->emb_m[token] + 128) >> 8);
    }
    const int16_t *C = m->cos + (long)t * (hd / 2), *S = m->sin + (long)t * (hd / 2);

    for (int l = 0; l < m->layers; l++) {
        const TinyBlock *B = &m->blocks[l];
        int8_t *kq = s->kq + (long)l * m->seq * dim, *vq = s->vq + (long)l * m->seq * dim;
        int32_t *km = s->km + (long)l * m->seq * m->heads, *vm = s->vm + (long)l * m->seq * m->heads;
        int64_t hm = in_quant(&B->q, s, s->t1, dim, norm_h(x, B->n1, dim, s->t1));   /* q, k, v: one input */
        TINYLM_PROF(0);
        if (B->q.acts == 8) prepare(s, s->iq, dim);
        run_linear(&B->q, s, hm, s->q);
        run_linear(&B->k, s, hm, s->k);
        run_linear(&B->v, s, hm, s->v);
        TINYLM_PROF(1);
        for (int h = 0; h < m->heads; h++) {
            int8_t qq[256];
            int32_t *qh = s->q + h * hd, *kh = s->k + h * hd;
            rope(qh, C, S, hd);
            rope(kh, C, S, hd);
            int32_t qm = quant8(qh, hd, qq);
            /* the cache of head h: keys and values of positions 0..seq-1 one after another (read in order) */
            int8_t *kqh = kq + (long)h * m->seq * hd, *vqh = vq + (long)h * m->seq * hd;
            int32_t *kmh = km + (long)h * m->seq, *vmh = vm + (long)h * m->seq;
            kmh[t] = quant8(kh, hd, kqh + (long)t * hd);
#ifdef TINYLM_DOT_ROWS_U8
            {   /* the V cache transposed and unsigned: row i of head h = value i + 128 of positions 0..seq-1 */
                int8_t vt[256];
                vmh[t] = quant8(s->v + h * hd, hd, vt);
                for (int i = 0; i < hd; i++) ((uint8_t *)vqh)[(long)i * m->seq + t] = (uint8_t)(vt[i] + 128);
            }
#else
            vmh[t] = quant8(s->v + h * hd, hd, vqh + (long)t * hd);
#endif
            TINYLM_PROF(8);
            /* scores u_j = (qq . kq_j) * km_j, softmax through the exp table */
            int64_t *u = (int64_t *)s->t2;  /* hidden int32 >= 2 * seq int32: checked in init */
            int64_t umax = INT64_MIN;
#ifdef TINYLM_DOT_ROWS
            TINYLM_DOT_ROWS(qq, hd, kqh, hd, t + 1, s->t3);  /* t3: free until the feed-forward */
#endif
            for (int j = 0; j <= t; j++) {
#ifdef TINYLM_DOT_ROWS
                int32_t d = s->t3[j];
#else
                const int8_t *kj = kqh + (long)j * hd;
                int32_t d = 0;
                for (int i = 0; i < hd; i++) d += qq[i] * kj[i];
#endif
                u[j] = (int64_t)d * kmh[j];
                if (u[j] > umax) umax = u[j];
            }
            TINYLM_PROF(9);
            int64_t P = 0;
            /* floor(n / (16129 << 20)) = floor((n >> 20) / 16129) for n >= 0; past the table p = 0, so only
               n >> 20 < 2048 * 16129 (32 bits) needs the division */
            for (int j = 0; j <= t; j++) {
                const uint64_t n20 = (uint64_t)((umax - u[j]) * qm) >> 20;
                const uint32_t xi = n20 < (uint64_t)TINYLM_EXP_N * 16129 ? (uint32_t)n20 / 16129u : TINYLM_EXP_N;
                s->p[j] = xi < TINYLM_EXP_N ? m->exp_t[xi] : 0;
                P += s->p[j];
            }
            TINYLM_PROF(10);
            int64_t acc[256];
#ifdef TINYLM_DOT_ROWS_U8
            /* acc_i = sum_j pv_j vq_ji, pv_j = p_j vm_j < 2^32, as row products of the transposed V cache (row i =
               value i + 128 of every position, unsigned) with the weights byte by byte: pv_j = sum_c 256^c b_cj,
               sum_j pv_j (vq_ji + 128) = sum_c 256^c dot_c[i], dot_c[i] <= 255 * 255 * seq < 2^31; then the 128 of
               every position comes off: 128 * sum_j pv_j. As many bytes as the largest weight needs (at most 4,
               pv < 2^32: checked). The same integers as the loop below. */
            {
                uint32_t *pv = (uint32_t *)s->t3;              /* t3: free until the feed-forward */
                uint32_t pvmax = 0;
                int64_t pvsum = 0;
                int wide = 0;
                for (int j = 0; j <= t; j++) {
                    const int64_t w = (int64_t)s->p[j] * vmh[j];
                    if (w >> 32) wide = 1;
                    pv[j] = (uint32_t)w; pvsum += w;
                    if (pv[j] > pvmax) pvmax = pv[j];
                }
                int nb = 1;
                while (nb < 4 && (pvmax >> (8 * nb))) nb++;
                for (int i = 0; i < hd; i++) acc[i] = -128 * pvsum;
                uint8_t *b = (uint8_t *)s->iq;                 /* iq: free here; nb chunk vectors of t + 1 bytes */
                for (int j = 0; j <= t; j++)
                    for (int c = 0; c < nb; c++) b[(long)c * m->seq + j] = (uint8_t)(pv[j] >> (8 * c));
                int32_t dot[256];
                for (int c = 0; c < nb && !wide; c++) {
                    TINYLM_DOT_ROWS_U8(b + (long)c * m->seq, t + 1, (const uint8_t *)vqh, m->seq, hd, dot);
                    for (int i = 0; i < hd; i++) acc[i] += (int64_t)dot[i] << (8 * c);
                }
                if (wide) {                                    /* a weight past 32 bits: the plain sums */
                    for (int i = 0; i < hd; i++) acc[i] = 0;
                    for (int j = 0; j <= t; j++) {
                        const int64_t w = (int64_t)s->p[j] * vmh[j];
                        const uint8_t *vj = (const uint8_t *)vqh + j;
                        for (int i = 0; i < hd; i++) acc[i] += w * ((int32_t)vj[(long)i * m->seq] - 128);
                    }
                }
            }
#else
            /* key by key (rows of the V cache read in order): sum_j p_j vm_j vq_ji. Weights pv < 2^32 (p < 2^16,
               vm >= 1) go in two 16-bit halves into int32 sums: |half * vq| < 2^16 * 127, at most seq <= 258 keys
               (tinylm_load), so no overflow; a larger weight sends the head to the int64 sums. The same integers. */
            int wide = 0;
            int32_t *pl = s->t3, *ph = s->t3 + m->seq;  /* t3: hidden >= 2 * seq (tinylm_state_init) */
            for (int j = 0; j <= t; j++) {
                const int64_t pv = (int64_t)s->p[j] * vmh[j];
                if (pv >> 32) wide = 1;
                pl[j] = (int32_t)(pv & 0xFFFF); ph[j] = (int32_t)(pv >> 16);
            }
            /* 4 columns at a time over every key: 8 sums in registers, the head's cache (t * hd bytes) stays in the
               data cache across the 16 passes */
            for (int i = 0; i < hd && !wide; i += 4) {
                int32_t l0 = 0, l1 = 0, l2 = 0, l3 = 0, h0 = 0, h1 = 0, h2 = 0, h3 = 0;
                const int8_t *v = vqh + i;
                for (int j = 0; j <= t; j++, v += hd) {
                    const int32_t a = pl[j], b = ph[j];
                    const int32_t v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
                    l0 += a * v0; l1 += a * v1; l2 += a * v2; l3 += a * v3;
                    h0 += b * v0; h1 += b * v1; h2 += b * v2; h3 += b * v3;
                }
                /* the high sums can be negative (vq is signed): multiplied, not shifted (C99 leaves the shift undefined) */
                acc[i] = (int64_t)h0 * 65536 + l0; acc[i + 1] = (int64_t)h1 * 65536 + l1;
                acc[i + 2] = (int64_t)h2 * 65536 + l2; acc[i + 3] = (int64_t)h3 * 65536 + l3;
            }
            if (wide) {
                for (int i = 0; i < hd; i++) acc[i] = 0;
                for (int j = 0; j <= t; j++) {
                    const int64_t pv = (int64_t)s->p[j] * vmh[j];
                    const int8_t *vj = vqh + (long)j * hd;
                    for (int i = 0; i < hd; i++) acc[i] += pv * vj[i];
                }
            }
#endif
            TINYLM_PROF(2);
            for (int i = 0; i < hd; i++) s->a[h * hd + i] = (int32_t)rdiv(acc[i], P * 127);
            TINYLM_PROF(11);
        }
        int64_t am = in_quant(&B->o, s, s->a, dim, 0);
        TINYLM_PROF(8);
        if (B->o.acts == 8) prepare(s, s->iq, dim);
        run_linear(&B->o, s, am, s->t1);
        for (int i = 0; i < dim; i++) x[i] += s->t1[i];
        TINYLM_PROF(3);

        hm = in_quant(&B->w1, s, s->t1, dim, norm_h(x, B->n2, dim, s->t1));          /* w1, w3: one input */
        TINYLM_PROF(0);
        if (B->w1.acts == 8) prepare(s, s->iq, dim);
        run_linear(&B->w1, s, hm, s->t2);
        run_linear(&B->w3, s, hm, s->t3);
        TINYLM_PROF(4);
        for (int i = 0; i < m->hidden; i++) {
            int32_t g1 = s->t2[i];
            int32_t idx = (g1 >> (QF - SIG_Q)) + TINYLM_SIG_N / 2;
            idx = idx < 0 ? 0 : idx > TINYLM_SIG_N - 1 ? TINYLM_SIG_N - 1 : idx;
            int64_t silu = ((int64_t)g1 * m->sig_t[idx] + (1 << 14)) >> 15;
            s->t2[i] = (int32_t)((silu * s->t3[i] + (1 << (QF - 1))) >> QF);
        }
        int64_t fm = in_quant(&B->w2, s, s->t2, m->hidden, 0);
        TINYLM_PROF(5);
        if (B->w2.acts == 8) prepare(s, s->iq, m->hidden);
        run_linear(&B->w2, s, fm, s->t1);
        for (int i = 0; i < dim; i++) x[i] += s->t1[i];
        TINYLM_PROF(6);
    }

    int32_t hm = norm_quant(x, m->norm_f, dim, s->t1, s->iq);
    TINYLM_PROF(0);
    int best = 0;
    if (m->emb_p) {   /* int4 table: m->hidden rows at a time into t2 */
        for (int r0 = 0; r0 < m->vocab; r0 += m->hidden) {
            const int n = m->vocab - r0 < m->hidden ? m->vocab - r0 : m->hidden;
#ifdef TINYLM_DOT_ROWS4
            TINYLM_DOT_ROWS4(s->iq, dim, m->emb_p + (long)r0 * (dim / 2), n, s->t2);
#else
            for (int r = 0; r < n; r++) {
                const uint8_t *e = m->emb_p + (long)(r0 + r) * (dim / 2);
                int32_t d = 0;
                for (int i = 0; i < dim; i++) d += emb4(e, i) * s->iq[i];
                s->t2[r] = d;
            }
#endif
            for (int r = r0; r < r0 + n; r++) {
                z[r] = (int64_t)s->t2[r - r0] * m->emb_m[r];
                if (z[r] > z[best]) best = r;
            }
        }
        TINYLM_PROF(7);
        *hmax_out = hm;
        s->pos++;
        return best;
    }
    if (m->emb_c) {   /* int2 table: m->hidden rows at a time into t2 */
        for (int r0 = 0; r0 < m->vocab; r0 += m->hidden) {
            const int n = m->vocab - r0 < m->hidden ? m->vocab - r0 : m->hidden;
#ifdef TINYLM_DOT_ROWS2
            TINYLM_DOT_ROWS2(s->iq, dim, m->emb_c + (long)r0 * (dim / 4), n, s->t2);
#else
            for (int r = 0; r < n; r++) {
                const uint8_t *e = m->emb_c + (long)(r0 + r) * (dim / 4);
                int32_t d = 0;
                for (int i = 0; i < dim; i++) d += emb2(e, i) * s->iq[i];
                s->t2[r] = d;
            }
#endif
            for (int r = r0; r < r0 + n; r++) {
                z[r] = (int64_t)s->t2[r - r0] * m->emb_m[r];
                if (z[r] > z[best]) best = r;
            }
        }
        TINYLM_PROF(7);
        *hmax_out = hm;
        s->pos++;
        return best;
    }
#ifdef TINYLM_DOT_ROWS
    /* the board's int8 row products (out[r] = sum_i x[i] * w[r * stride + i]), m->hidden rows at a time into t2 */
    for (int r0 = 0; r0 < m->vocab; r0 += m->hidden) {
        const int n = m->vocab - r0 < m->hidden ? m->vocab - r0 : m->hidden;
        TINYLM_DOT_ROWS(s->iq, dim, m->emb_q + (long)r0 * dim, dim, n, s->t2);
        for (int r = r0; r < r0 + n; r++) {
            z[r] = (int64_t)s->t2[r - r0] * m->emb_m[r];
            if (z[r] > z[best]) best = r;
        }
    }
#else
    for (int r = 0; r < m->vocab; r++) {
        const int8_t *e = m->emb_q + (long)r * dim;
        int32_t d = 0;
        for (int i = 0; i < dim; i++) d += e[i] * s->iq[i];
        z[r] = (int64_t)d * m->emb_m[r];
        if (z[r] > z[best]) best = r;
    }
#endif
    TINYLM_PROF(7);
    *hmax_out = hm;
    s->pos++;
    return best;
}

/* ---- loading ---------------------------------------------------------------------------------------------------- */

static const uint8_t *take(const uint8_t **p, const uint8_t *end, long n) {
    const uint8_t *r = *p;
    if (n < 0 || end - r < n) return NULL;
    *p += n;
    return r;
}

static int take_linear(TinyLinear *L, int n_in, int n_out, const uint8_t **p, const uint8_t *end, int version) {
    L->n_in = n_in; L->n_out = n_out; L->words_in = (n_in + 31) / 32;
    L->acts = 8; L->had = 0;
    if (version >= 2) {   /* int32 acts, had, 0, 0 before the weights (16 bytes: they stay 16-byte aligned) */
        const int32_t *f = (const int32_t *)take(p, end, 16);
        if (!f) return -1;
        L->acts = f[0]; L->had = f[1];
        if (L->acts != 8 && (L->acts != 1 || n_in % 32)) return -1;
        if (L->had && (L->had < 2 || (L->had & (L->had - 1)) || n_in % L->had)) return -1;
    }
    L->w = (const uint32_t *)take(p, end, 4L * n_out * L->words_in);
    L->m = (const int32_t *)take(p, end, 4L * n_out);
    if (!L->w || !L->m) return -1;
#if defined(TINYLM_U8_SUM_ROWS)
    int32_t *ones = TINYLM_MALLOC_BIG(4L * n_out);  /* weight bits set per row (padding bits are 0) */
    if (!ones) return -1;
    for (int r = 0; r < n_out; r++) {
        int32_t c = 0;
        for (int j = 0; j < L->words_in; j++) c += __builtin_popcount(L->w[(long)r * L->words_in + j]);
        ones[r] = c;
    }
    L->ones = ones;
#endif
    return 0;
}

int tinylm_load(TinyModel *m, const uint8_t *buf, long size) {
    const int32_t *h = (const int32_t *)buf;
    if (size < 32 || h[0] != TINYLM_MAGIC || h[1] < 1 || h[1] > 4) return -1;
    const int version = h[1];  /* 2: per linear layer, its input bits and Hadamard block; 3 / 4: also the int4 /
                                  int2 table */
    m->vocab = h[2]; m->dim = h[3]; m->hidden = h[4]; m->layers = h[5]; m->heads = h[6]; m->seq = h[7];
    if (m->vocab <= 0 || m->dim <= 0 || m->hidden <= 0 || m->layers <= 0 || m->heads <= 0 || m->seq <= 0) return -1;
    /* the attention scale 16129 << 20 = 127^2 * 2^24 / 128 * sqrt(64) assumes 64-wide heads (as intsim.py) */
    if (m->dim % m->heads || m->dim / m->heads != 64 || m->dim % 8 || m->hidden % 8) return -1;
    if (m->seq > 258) return -1;  /* the int32 attention sums (tinylm_step) */
#if defined(TINYLM_U8_SUM_ROWS)
    if (m->dim > TINYLM_U8_MAX_IN || m->hidden > TINYLM_U8_MAX_IN) return -1;
#endif
    const uint8_t *p = buf + 32, *end = buf + size;
    int hd2 = m->dim / m->heads / 2;
    m->emb_q = NULL; m->emb_p = NULL; m->emb_c = NULL;
    if (version == 3) {
        if (m->dim % 32) return -1;
        m->emb_p = take(&p, end, (long)m->vocab * m->dim / 2);
    } else if (version == 4) {
        if (m->dim % 64) return -1;
        m->emb_c = take(&p, end, (long)m->vocab * m->dim / 4);
    } else {
        m->emb_q = (const int8_t *)take(&p, end, (long)m->vocab * m->dim);
    }
    m->emb_m = (const int32_t *)take(&p, end, 4L * m->vocab);
    m->norm_f = (const int32_t *)take(&p, end, 4L * m->dim);
    m->cos = (const int16_t *)take(&p, end, 2L * m->seq * hd2);
    m->sin = (const int16_t *)take(&p, end, 2L * m->seq * hd2);
    m->exp_t = (const uint16_t *)take(&p, end, 2L * TINYLM_EXP_N);
    m->sig_t = (const uint16_t *)take(&p, end, 2L * TINYLM_SIG_N);
    if ((!m->emb_q && !m->emb_p && !m->emb_c) || !m->emb_m || !m->norm_f || !m->cos || !m->sin || !m->exp_t || !m->sig_t) return -1;
    m->blocks = (TinyBlock *)calloc(m->layers, sizeof(TinyBlock));
    if (!m->blocks) return -1;
    for (int l = 0; l < m->layers; l++) {
        TinyBlock *B = &m->blocks[l];
        B->n1 = (const int32_t *)take(&p, end, 4L * m->dim);
        B->n2 = (const int32_t *)take(&p, end, 4L * m->dim);
        if (!B->n1 || !B->n2) return -1;
        if (take_linear(&B->q, m->dim, m->dim, &p, end, version) ||
            take_linear(&B->k, m->dim, m->dim, &p, end, version) ||
            take_linear(&B->v, m->dim, m->dim, &p, end, version) ||
            take_linear(&B->o, m->dim, m->dim, &p, end, version) ||
            take_linear(&B->w1, m->dim, m->hidden, &p, end, version) ||
            take_linear(&B->w3, m->dim, m->hidden, &p, end, version) ||
            take_linear(&B->w2, m->hidden, m->dim, &p, end, version))
            return -1;
        /* layers that share an input share its format */
        if (B->k.acts != B->q.acts || B->v.acts != B->q.acts || B->k.had != B->q.had || B->v.had != B->q.had ||
            B->w3.acts != B->w1.acts || B->w3.had != B->w1.had)
            return -1;
    }
    return p == end ? 0 : -1;
}

int tinylm_state_init(TinyState *s, const TinyModel *m) {
    long kv = (long)m->layers * m->seq * m->dim, kvm = (long)m->layers * m->seq * m->heads;
    int big = m->hidden > m->dim ? m->hidden : m->dim;
    if (m->hidden < 2 * m->seq) return -1;  /* t2 doubles as the int64 score buffer */
#ifdef TINYLM_DOT_ROWS_U8
    if (big < 4 * m->seq) return -1;        /* iq holds the 4 byte vectors of the attention weights */
#endif
    memset(s, 0, sizeof(*s));
    s->kq = TINYLM_MALLOC_BIG(kv); s->vq = TINYLM_MALLOC_BIG(kv + 16);  /* +16: the row kernel reads whole blocks */
    s->km = TINYLM_MALLOC_BIG(4 * kvm); s->vm = TINYLM_MALLOC_BIG(4 * kvm);
    s->x = TINYLM_MALLOC_FAST(4L * m->dim);
    s->t1 = TINYLM_MALLOC_FAST(4L * big); s->t2 = TINYLM_MALLOC_FAST(4L * big); s->t3 = TINYLM_MALLOC_FAST(4L * big);
    s->q = TINYLM_MALLOC_FAST(4L * m->dim); s->k = TINYLM_MALLOC_FAST(4L * m->dim);
    s->v = TINYLM_MALLOC_FAST(4L * m->dim); s->a = TINYLM_MALLOC_FAST(4L * m->dim);
    s->iq = TINYLM_MALLOC_FAST(big); s->p = TINYLM_MALLOC_FAST(2L * m->seq);
    s->lut = TINYLM_MALLOC_FAST(2L * 256 * (big / 8));
    s->r = TINYLM_MALLOC_FAST(8L * big); s->xb = TINYLM_MALLOC_FAST(4L * (big / 32 + 1));
    return s->kq && s->vq && s->km && s->vm && s->x && s->t1 && s->t2 && s->t3 && s->q && s->k && s->v && s->a &&
                   s->iq && s->p && s->lut && s->r && s->xb ? 0 : -1;
}

void tinylm_state_free(TinyState *s) {
    void *all[] = {s->kq, s->vq, s->km, s->vm, s->x, s->t1, s->t2, s->t3, s->q, s->k, s->v, s->a, s->iq, s->p, s->lut,
                   s->r, s->xb};
    for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) free(all[i]);
    memset(s, 0, sizeof(*s));
}
