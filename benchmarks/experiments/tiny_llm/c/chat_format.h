#ifndef CHAT_FORMAT_H
#define CHAT_FORMAT_H
#include <stddef.h>
#include <stdint.h>
typedef struct { const int *ids; size_t count; } ChatIds;
typedef struct { unsigned role; ChatIds content; } ChatTurn;
/* Roles 0 system, 1 user, 2 assistant. Markers are sp.encode("System:"),
 * sp.encode("User:"), sp.encode("Assistant:"). BOS=1, EOS=2; vocab>=3.
 * Output: BOS + (marker,content,EOS)* + open assistant marker.
 * Caller owns disjoint input/output. Returns count, -1 invalid/capacity.
 * Checks all spans/IDs/capacity before writing output. */
int chat_build(const ChatIds markers[3], const ChatTurn *turns, size_t count,
               int vocab, int *out, size_t capacity);
#endif
