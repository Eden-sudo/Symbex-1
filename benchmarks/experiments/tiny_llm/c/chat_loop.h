#ifndef CHAT_LOOP_H
#define CHAT_LOOP_H
#include "chat_stop.h"
typedef int (*ChatForward)(void *ctx, int token, int64_t *logits);
typedef int (*ChatEmit)(void *ctx, int token);
typedef int (*ChatPiece)(void *ctx, int token, const unsigned char **bytes, size_t *size);
typedef struct {
    int vocab, context, k;
    uint32_t temperature, top_p, seed;
    int64_t penalty;
} ChatOptions;
/* Fresh/reset forward state required. Prompt IDs are already framed/retained.
 * Context bounds prompt+generated IDs. No global state or heap allocation.
 * Piece callback optional iff no string stops; EOS never emitted. Other stop
 * bytes are included (emit callback can hold them back). Returns stop reason,
 * 4 context full, -1 error. Forward/emit failures may leave partial output.
 * Caller supplies logits[vocab], scratch[vocab], nonoverlapping buffers.
 * stop and options.seed advance; stop must be freshly initialized. */
int chat_generate(ChatOptions *options, const int *prompt, int count, ChatStop *stop,
                  ChatForward forward, ChatPiece piece, ChatEmit emit, void *ctx,
                  int64_t *logits, int *scratch);
#endif
