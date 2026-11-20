/* Standalone integration entry: link existing runtime modules, not tinylm.c. */
#include "board_chat.h"
#include "chat_memory.h"
#include <stdio.h>
#include <string.h>

typedef struct {
    ChatForward forward;
    BoardChatWrite write;
    void *ctx;
    unsigned emitted;
} Stream;

static int forward_token(void *opaque, int token, int64_t *logits)
{
    Stream *s = opaque;
    return s->forward(s->ctx, token, logits);
}

static int emit_token(void *opaque, int token)
{
    Stream *s = opaque;
    char line[80];
    int n = snprintf(line, sizeof line, "CHAT1 TOKEN %u %d\n", s->emitted, token);
    if (n < 0 || (size_t)n >= sizeof line || s->write(s->ctx, line, (size_t)n)) return -1;
    ++s->emitted;
    return 0;
}

int board_chat_turn(const ChatIds markers[3], const ChatTurn *turns, size_t count,
                    ChatOptions *o, unsigned limit, uint32_t reference_id,
                    BoardChatReset reset, ChatForward forward, BoardChatWrite write,
                    void *ctx, int64_t *logits, int *scratch, size_t *dropped)
{
    int storage[32][256], system[256] = {1}, prompt[256];
    ChatIds spans[32];
    size_t nspans = 0, system_count = 1, removed;
    ChatStop stop;
    char line[80];
    if (!markers || !o || !reset || !forward || !write || !logits || !scratch ||
        !dropped || (count && !turns) || count > 32 || o->vocab < 3 ||
        o->context < 1 || o->context > 256 || limit > (unsigned)o->context ||
        o->k < 0 || !o->top_p || o->top_p > 65536 || !o->seed || o->penalty < 0) return -1;
    /* chat_build validates all markers and IDs before any reset or stream write. */
    if (chat_build(markers, NULL, 0, o->vocab, prompt, 256) < 0) return -1;
    for (size_t i = 0; i < count; ++i) {
        if (turns[i].role == 0 && i != 0) return -1;
        int n = chat_build(markers, turns+i, 1, o->vocab, storage[i], 256);
        if (n < 0) return -1;
        size_t span_count = (size_t)n - 1 - markers[2].count;
        if (turns[i].role == 0) {
            memcpy(system+1, storage[i]+1, span_count*sizeof(int));
            system_count += span_count;
        } else {
            spans[nspans++] = (ChatIds){storage[i]+1, span_count};
        }
    }
    int used = chat_retain((ChatIds){system, system_count}, spans, nspans, markers[2],
                           limit, (unsigned)o->context, prompt, 256, &removed);
    if (used < 0 || chat_stop_init(&stop, 2, limit, NULL, NULL, 0)) return -1;
    if (reset(ctx)) return -1;
    Stream stream = {forward, write, ctx, 0};
    int n = snprintf(line, sizeof line, "CHAT1 BEGIN %u\n", (unsigned)reference_id);
    if (n < 0 || (size_t)n >= sizeof line || write(ctx, line, (size_t)n)) return -1;
    *dropped = removed;
    int reason = chat_generate(o, prompt, used, &stop, forward_token, NULL,
                               emit_token, &stream, logits, scratch);
    if (reason < 0) return -1;
    n = snprintf(line, sizeof line, "CHAT1 END %d %u\n", reason, stream.emitted);
    if (n < 0 || (size_t)n >= sizeof line || write(ctx, line, (size_t)n)) return -1;
    return reason;
}
