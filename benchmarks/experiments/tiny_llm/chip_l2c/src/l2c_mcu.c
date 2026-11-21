/* l2c_mcu.c -- llama2.c's run.c (unchanged, ../llama2c/run.c) with the weights mapped from memory instead of a file:
   the model is a llama2.c version-0 checkpoint (7 int32 header + fp32 weights, chip_fp32.py writes it), a const
   array in flash on the chips. cap: the context the KV cache is allocated for (run.c allocates seq_len positions;
   the fp32 cache of the 260K model at 512 positions fits neither the classic ESP32 nor the Pico 2). */
#define TESTING
#include "../llama2c/run.c"
#include "l2c_mcu.h"

static Transformer l2c;

int l2c_init(const void *model, int cap) {
    const int *h = (const int *)model;
    Config *p = &l2c.config;
    memcpy(p, h, sizeof(Config));
    int shared = p->vocab_size > 0;
    p->vocab_size = abs(p->vocab_size);
    memory_map_weights(&l2c.weights, p, (float *)((const char *)model + sizeof(Config)), shared);
    if (cap > 0 && cap < p->seq_len) p->seq_len = cap;      /* after the mapping: it skips seq_len RoPE rows */
    malloc_run_state(&l2c.state, p);
    return p->seq_len;
}

int l2c_step(int token, int pos) {
    float *logits = forward(&l2c, token, pos);
    return sample_argmax(logits, l2c.config.vocab_size);
}

float l2c_logit(int i) { return l2c.state.logits[i]; }
