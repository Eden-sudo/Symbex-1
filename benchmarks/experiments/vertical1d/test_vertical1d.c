/*
 * test_vertical1d.c -- v1d_conv (vertical1d.h) against symbex_bool1d_conv on random layers: every output word equal.
 * Built and run by vertical1d.py test (portable C on the PC; the S3's PIE path is checked on the chip, whole networks).
 * Random shapes (channels 1-70, kernels 1-17, pools 1-8, lengths up to 400: several tiles), random weights and
 * streams, thresholds random or at the extremes (INT32_MIN / INT32_MAX, +-n, n + 1).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layers/conv/symbex_bool1d.h"
#include "vertical1d.h"

static uint32_t rng = 12345u;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

int main(int argc, char** argv) {
    const int cases = argc > 1 ? atoi(argv[1]) : 2000;
    int bad = 0;
    static uint32_t shared[1 << 18] __attribute__((aligned(16)));   /* one block for every layer, as in a network */
    for (int it = 0; it < cases; it++) {
        const int cin = 1 + (int)(rnd() % 70), kq = 1 + (int)(rnd() % 17);
        if (kq * cin > 1016) { it--; continue; }
        const int pools[] = {1, 2, 3, 4, 8, 5};
        const int pool = pools[rnd() % 6], cout = 1 + (int)(rnd() % 40);
        const int in_len = kq + pool - 1 + (int)(rnd() % 400);
        const int conv_len = in_len - kq + 1, out_len = conv_len / pool;
        const int n = kq * cin, row_words = ((n + 31) / 32 + 3) / 4 * 4;
        uint32_t* W = calloc((size_t)cout * row_words, 4);
        int32_t* T = malloc((size_t)cout * 4);
        int32_t* C = malloc((size_t)cout * 4);
        for (int o = 0; o < cout; o++) {
            const int dens = (int)(rnd() % 5);                     /* 0: all zero ... 4: all one */
            int c = 0;
            for (int j = 0; j < n; j++) {
                const int b = dens == 0 ? 0 : dens == 4 ? 1 : (int)(rnd() % 4) < dens;
                if (b) { W[o * row_words + j / 32] |= 1u << (j % 32); c++; }
            }
            C[o] = c;
            const int kind = (int)(rnd() % 8);
            T[o] = kind == 0 ? INT32_MIN : kind == 1 ? INT32_MAX : kind == 2 ? -n : kind == 3 ? n + 1 : kind == 4 ? n
                 : (int32_t)(rnd() % (2 * n + 3)) - n - 1;
        }
        const size_t in_words = ((size_t)in_len * cin + 31) / 32 + 1, out_words = ((size_t)out_len * cout + 31) / 32 + 1;
        uint32_t* in = calloc(in_words, 4);
        const int dens = (int)(rnd() % 3);
        for (size_t i = 0; i + 1 < in_words; i++) in[i] = dens == 0 ? rnd() & rnd() : dens == 1 ? rnd() : rnd() | rnd();
        if ((in_len * cin) & 31) in[in_words - 2] &= (1u << ((in_len * cin) & 31)) - 1u;
        struct SymbexBool1D L = {(uint16_t)cin, (uint16_t)cout, (uint16_t)kq, 1, (uint16_t)pool, (uint16_t)in_len,
                                 (uint16_t)out_len, (uint16_t)row_words, W, T, C, NULL};
        uint32_t* ref = calloc(out_words, 4);
        uint32_t* got = malloc(out_words * 4);
        memset(got, 0xA5, out_words * 4);
        uint32_t* window = aligned_alloc(16, (size_t)2 * row_words * 4 + 16);
        int32_t* scratch = malloc((size_t)cout * 4);
        const size_t ww = V1D_WORDS(cin, cout, kq, pool, in_len, out_len);
        uint32_t* work = aligned_alloc(16, (ww * 4 + 15) / 16 * 16);
        memset(work, 0, ww * 4);
        symbex_bool1d_conv(&L, in, ref, window, scratch);
        v1d_conv(&L, in, got, work);
        v1d_conv(&L, in, got, work);                               /* twice: the work block is reused */
        if (ww > sizeof shared / 4) { fprintf(stderr, "shared block too small\n"); return 2; }
        uint32_t* got2 = malloc(out_words * 4);
        v1d_conv(&L, in, got2, shared);                            /* the block the previous layers used */
        const size_t used = ((size_t)out_len * cout + 31) / 32;
        uint32_t lastmask = ((out_len * cout) & 31) ? (1u << ((out_len * cout) & 31)) - 1u : ~0u;
        for (size_t i = 0; i < used; i++) {
            const uint32_t m = i + 1 == used ? lastmask : ~0u;
            if ((ref[i] & m) != (got[i] & m) || (ref[i] & m) != (got2[i] & m)) {
                if (bad < 10) printf("[X] case %d: cin %d cout %d k %d pool %d len %d: word %zu ref %08x got %08x\n",
                                     it, cin, cout, kq, pool, in_len, i, ref[i] & m, got[i] & m);
                bad++;
                break;
            }
        }
        free(W); free(T); free(C); free(in); free(ref); free(got); free(window); free(scratch); free(work); free(got2);
    }
    printf("%d random layers, %d different\n", cases, bad);
    return bad != 0;
}
