#include "encoder.h"
#include <string.h>

#define LIMIT 4096
#define SLOTS 4096
typedef struct { size_t start, length; int piece; } symbol;

static unsigned hash(const char *text, size_t length, int kind) {
    unsigned h = (unsigned)kind + 2166136261u;
    size_t i;
    for (i = 0; i < length; ++i) h = (h ^ (unsigned char)text[i]) * 16777619u;
    return h % SLOTS;
}

static int lookup(const sensor_vocab *v, const int *table, const char *text, size_t length, int kind) {
    unsigned slot = hash(text, length, kind);
    while (table[slot] >= 0) {
        const sensor_piece *p = &v->pieces[table[slot]];
        if (p->kind == kind && strlen(p->text) == length && !memcmp(p->text, text, length))
            return table[slot];
        slot = (slot + 1) % SLOTS;
    }
    return -1;
}

int sensor_encode(const sensor_vocab *v, const char *text, int *ids, size_t capacity) {
    symbol symbols[LIMIT];
    int table[SLOTS];
    size_t length = 0, pos = 0, count = 0, i;
    if (!v || !v->pieces || !text || !ids || v->count >= SLOTS / 2) return -1;
    for (i = 0; i < SLOTS; ++i) table[i] = -1;
    for (i = 0; i < v->count; ++i) {
        unsigned slot = hash(v->pieces[i].text, strlen(v->pieces[i].text), v->pieces[i].kind);
        while (table[slot] >= 0) slot = (slot + 1) % SLOTS;
        table[slot] = (int)i;
    }
    while (length <= LIMIT && text[length]) {
        if ((unsigned char)text[length] < 32 || (unsigned char)text[length] > 126) return -1;
        ++length;
    }
    if (length > LIMIT) return -1;
    while (pos < length) {
        int found = -1;
        size_t width = 1;
        /* User-defined symbols are indivisible, selected longest first. */
        for (i = 0; i < v->count; ++i) {
            size_t n = strlen(v->pieces[i].text);
            if (v->pieces[i].kind == 1 && n >= width && n <= length - pos &&
                !memcmp(text + pos, v->pieces[i].text, n)) {
                found = (int)i;
                width = n;
            }
        }
        if (found < 0) found = lookup(v, table, text + pos, 1, 0);
        if (found < 0) found = lookup(v, table, text + pos, 1, 2);
        if (found < 0) return -1;
        symbols[count].start = pos;
        symbols[count].length = width;
        symbols[count++].piece = found;
        pos += width;
    }
    for (;;) {
        int best = -1;
        size_t at = 0;
        for (i = 0; i + 1 < count; ++i) {
            int candidate;
            if (v->pieces[symbols[i].piece].kind || v->pieces[symbols[i + 1].piece].kind) continue;
            candidate = lookup(v, table, text + symbols[i].start, symbols[i].length + symbols[i + 1].length, 0);
            if (candidate >= 0 && (best < 0 || v->pieces[candidate].score > v->pieces[best].score)) {
                best = candidate;
                at = i;
            }
        }
        if (best < 0) break;
        symbols[at].length += symbols[at + 1].length;
        symbols[at].piece = best;
        memmove(symbols + at + 1, symbols + at + 2, (count - at - 2) * sizeof(symbol));
        --count;
    }
    if (count > capacity) return -1;
    for (i = 0; i < count; ++i) ids[i] = v->pieces[symbols[i].piece].id;
    return (int)count;
}
