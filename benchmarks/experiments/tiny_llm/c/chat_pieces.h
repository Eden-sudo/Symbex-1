#ifndef CHAT_PIECES_H
#define CHAT_PIECES_H
#include <stddef.h>
#include <stdint.h>
typedef struct {
    const unsigned char *offsets, *payload;
    uint32_t vocab;
    int started;
} ChatDecoder;
/* Validate TCP1 table, borrow immutable blob for decoder lifetime; no allocations.
 * On failure destination unchanged. Re-init for a new independently encoded text. */
int chat_decoder_init(ChatDecoder *d, const unsigned char *blob, size_t size);
/* Return borrowed piece bytes (possibly empty); strip one initial dummy space.
 * UTF-8 may span tokens: emit raw bytes, decode text only after accumulation.
 * Invalid ID leaves state/output unchanged. */
int chat_decode(ChatDecoder *d, int token, const unsigned char **bytes, size_t *size);
#endif
