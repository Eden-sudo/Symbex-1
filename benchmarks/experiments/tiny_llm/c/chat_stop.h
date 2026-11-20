#ifndef CHAT_STOP_H
#define CHAT_STOP_H
#include <stddef.h>
#include <stdint.h>
typedef struct {
    int eos, reason;
    uint32_t count, limit;
    unsigned n, used, lengths[8];
    unsigned char strings[8][64], tail[64];
} ChatStop;
/* Copies 0..8 stops of 1..64 bytes. Invalid init leaves destination untouched. */
int chat_stop_init(ChatStop *s, int eos, uint32_t limit,
                   const unsigned char *const *strings, const unsigned *lengths, unsigned n);
/* 0 running, 1 EOS (not counted), 2 byte match, 3 max tokens; -1 invalid.
 * Terminal state sticky. EOS > byte match > token limit. Includes stop bytes;
 * caller handles output holdback. Empty non-EOS pieces count as one token. */
int chat_stop_feed(ChatStop *s, int token, const unsigned char *piece, size_t size);
#endif
