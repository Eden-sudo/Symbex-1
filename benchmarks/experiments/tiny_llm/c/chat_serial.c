#include "chat_serial.h"
#include <string.h>

size_t chat_serial_encode(unsigned kind, const unsigned char *payload, size_t size,
                          unsigned char *output, size_t capacity)
{
    unsigned char body[CHAT_SERIAL_MAX_PAYLOAD + 6];
    uint16_t crc = 0xffff;
    size_t required = 2, at = 0;
    if (!payload || !output || size > CHAT_SERIAL_MAX_PAYLOAD) return 0;
    if (kind == CHAT_SERIAL_TOKEN) {
        if (size != 4 || (payload[3] & 0x80)) return 0;
    } else if (kind == CHAT_SERIAL_TEXT) {
        if (!size) return 0;
    } else if (kind == CHAT_SERIAL_END) {
        if (size != 1 || payload[0] < 1 || payload[0] > 5) return 0;
    } else return 0;
    body[0] = 1; body[1] = (unsigned char)kind;
    body[2] = (unsigned char)size; body[3] = (unsigned char)(size >> 8);
    memcpy(body + 4, payload, size);
    for (size_t i = 0; i < size + 4; ++i) {
        crc ^= (uint16_t)((uint16_t)body[i] << 8);
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (uint16_t)((crc << 1) ^ ((crc & 0x8000) ? 0x1021 : 0));
    }
    body[size + 4] = (unsigned char)crc;
    body[size + 5] = (unsigned char)(crc >> 8);
    for (size_t i = 0; i < size + 6; ++i)
        required += (body[i] == 0x7e || body[i] == 0x7d) ? 2 : 1;
    if (capacity < required) return 0;
    output[at++] = 0x7e;
    for (size_t i = 0; i < size + 6; ++i) {
        if (body[i] == 0x7e || body[i] == 0x7d) {
            output[at++] = 0x7d; output[at++] = body[i] ^ 0x20;
        } else output[at++] = body[i];
    }
    output[at++] = 0x7e;
    return at;
}
