/* pc_check.c -- the same l2c_mcu.c on the PC: reads a llama2.c checkpoint and the story's prompt ids, prints the
   greedy ids (one per line) and the time per token. Built by chip_fp32.py:
   cc -O2 -o pc_check pc_check.c src/l2c_mcu.c -lm ; ./pc_check model.bin n_new id0 id1 ... */
#include <stdio.h>
#include <stdlib.h>
#include "src/l2c_mcu.h"

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: pc_check model.bin n_new id0 [id1 ...]\n"); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(n);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    int n_new = atoi(argv[2]), n_prompt = argc - 3;
    l2c_init(buf, n_prompt + n_new);
    int next = 0;
    for (int i = 0; i < n_prompt + n_new - 1; i++) {
        next = l2c_step(i < n_prompt ? atoi(argv[3 + i]) : next, i);
        if (i + 1 >= n_prompt) printf("%d\n", next);
    }
    return 0;
}
