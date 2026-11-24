#ifndef SENSOR_TEXT_ENCODER_H
#define SENSOR_TEXT_ENCODER_H
#include <stddef.h>
typedef struct {
    const char *text;
    int id;
    float score;
    int kind; /* 0: normal, 1: atomic user symbol, 2: byte fallback */
} sensor_piece;
typedef struct { const sensor_piece *pieces; size_t count; } sensor_vocab;
/* Greedy highest-score BPE merges, leftmost on ties; no BOS/EOS.
 * Identity-normalized printable ASCII input, at most 4096 bytes.
 * Returns ID count, or -1 on unsupported input/capacity. No output on error.
 * Vocabulary must come from export_encoder.py; input/output must not overlap.
 * Fixed stack workspace, no heap; not a general Unicode SentencePiece encoder. */
int sensor_encode(const sensor_vocab *vocab, const char *text, int *ids, size_t capacity);
#endif
