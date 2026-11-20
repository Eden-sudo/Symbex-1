#ifndef BOARD_CHAT_H
#define BOARD_CHAT_H
#include "chat_format.h"
#include "chat_loop.h"
typedef int (*BoardChatWrite)(void *ctx, const char *bytes, size_t size);
typedef int (*BoardChatReset)(void *ctx);
/* No heap, tokenizer, hardware or model dependency. Caller provides disjoint
 * logits[vocab], scratch[vocab]. At most 32 turns, each formatted span <=256.
 * Optional system turn must be first; preserve it and BOS during eviction.
 * Each turn is an eviction unit (pair exchanges before calling if required).
 * Reserve limit positions, then reset and replay even if nothing was dropped.
 * Protocol: CHAT1 BEGIN <reference_id>\n, CHAT1 TOKEN <index> <id>\n,
 * CHAT1 END <reason> <count>\n. On callback error the stream is incomplete.
 * Return stop reason (1 EOS, 3 limit, 4 context) or -1. No score verification
 * is implied by a reference ID; adapter must supply the matching model/prompt.
 * options.seed advances. Fresh stop state is created for every call. */
int board_chat_turn(const ChatIds markers[3], const ChatTurn *turns, size_t count,
                    ChatOptions *options, unsigned limit, uint32_t reference_id,
                    BoardChatReset reset, ChatForward forward, BoardChatWrite write,
                    void *ctx, int64_t *logits, int *scratch, size_t *dropped);
#endif
