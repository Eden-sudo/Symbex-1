/*
 * check_pc.c -- the C model against intsim.py on the PC: every output z (int64, every token of the vocabulary) and
 * every final max equal, token by token, on the validation windows of check.bin; then the greedy story.
 *
 *     gcc -O2 -Wall -Wextra -o check_pc check_pc.c tinylm.c && ./check_pc ../runs/binary_d384l8_s42
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tinylm.h"

static uint8_t *read_file(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*size);
    if (b && fread(b, 1, *size, f) != (size_t)*size) { free(b); b = NULL; }
    fclose(f);
    return b;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <run folder>\n", argv[0]); return 2; }
    char path[1024];
    long msize, csize;
    snprintf(path, sizeof path, "%s/model.bin", argv[1]);
    uint8_t *mb = read_file(path, &msize);
    snprintf(path, sizeof path, "%s/check.bin", argv[1]);
    uint8_t *cb = read_file(path, &csize);
    TinyModel m;
    TinyState s;
    if (!mb || !cb || tinylm_load(&m, mb, msize) || tinylm_state_init(&s, &m)) {
        fprintf(stderr, "cannot load the model or check.bin\n");
        return 2;
    }
    const int32_t *hdr = (const int32_t *)cb;
    int n = hdr[0], T = hdr[1];
    const int32_t *tok = hdr + 2;
    const int64_t *zref = (const int64_t *)(tok + (long)n * T);
    const int64_t *href = zref + (long)n * T * m.vocab;
    const int32_t *g = (const int32_t *)(href + (long)n * T);
    int n_ids = g[0], n_prompt = g[1];
    const int32_t *ids = g + 2;

    int64_t *z = malloc(8L * m.vocab), hm;
    long tokens = 0, bad = 0;
    clock_t c0 = clock();
    for (int w = 0; w < n; w++) {
        s.pos = 0;
        for (int t = 0; t < T; t++, tokens++) {
            tinylm_step(&m, &s, tok[(long)w * T + t], z, &hm);
            const int64_t *zr = zref + ((long)w * T + t) * m.vocab;
            if (memcmp(z, zr, 8L * m.vocab) || hm != href[(long)w * T + t]) {
                if (bad < 5) fprintf(stderr, "  differs: window %d token %d\n", w, t);
                bad++;
            }
        }
    }
    double sec = (double)(clock() - c0) / CLOCKS_PER_SEC;
    printf("validation: %ld / %ld tokens with every output equal (%.2f ms per token on this PC)\n",
           tokens - bad, tokens, 1000 * sec / tokens);

    s.pos = 0;
    int same = 1, next = 0;
    for (int i = 0; i < n_ids - 1; i++) {
        next = tinylm_step(&m, &s, i < n_prompt ? ids[i] : next, z, &hm);
        if (i + 1 >= n_prompt && next != ids[i + 1]) same = 0;
        if (i + 1 < n_prompt) next = ids[i + 1];
    }
    printf("greedy story: %d tokens, %s\n", n_ids - n_prompt, same ? "the same as intsim.py" : "DIFFERENT");
    tinylm_state_free(&s);
    return bad || !same;
}
