#ifndef CHAT_MEMORY_H
#define CHAT_MEMORY_H
#include "chat_format.h"
/* Keep system + contiguous newest whole turns + suffix within context-reserve.
 * context 1..256. Fail if fixed spans or newest turn cannot fit. Return length
 * or -1; invalid inputs leave out/dropped unchanged. IDs nonnegative int.
 * All buffers disjoint. Reset/replay KV cache after dropping turns. */
int chat_retain(ChatIds system, const ChatIds *turns, size_t count, ChatIds suffix,
                unsigned reserve, unsigned context, int *out, size_t capacity, size_t *dropped);
#endif
