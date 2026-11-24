#ifndef CHAT_SERIAL_H
#define CHAT_SERIAL_H
#include <stddef.h>
#include <stdint.h>
#define CHAT_SERIAL_TOKEN 1
#define CHAT_SERIAL_TEXT 2
#define CHAT_SERIAL_END 3
#define CHAT_SERIAL_MAX_PAYLOAD 512
#define CHAT_SERIAL_MAX_FRAME 1038
/* TCS1 framing, caller-owned output. Returns bytes written or 0 on error.
 * Invalid input/insufficient capacity leaves output untouched. Input/output
 * must not overlap. See ../chat_serial.md for payloads and stream lifecycle. */
size_t chat_serial_encode(unsigned kind, const unsigned char *payload, size_t size,
                          unsigned char *output, size_t capacity);
#endif
