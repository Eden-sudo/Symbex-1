#include "prompt.h"
#include <stdio.h>

static int letter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static int valid_class(const char *s) {
    size_t n;
    if (!s || !letter((unsigned char)s[0])) return 0;
    for (n = 1; n <= 64; ++n) {
        unsigned char c = (unsigned char)s[n];
        if (!c) return 1;
        if (!letter(c) && !(c >= '0' && c <= '9') && c != '_') return 0;
    }
    return 0;
}

int sensor_prompt(char *dst, size_t capacity, const sensor_record *r) {
    int n;
    if (!dst || !capacity) return -1;
    dst[0] = '\0';
    if (!r || !valid_class(r->class_name)) return -1;
    switch (r->source) {
    case SENSOR_HAR:
        if (r->duration_seconds < 1 || r->duration_seconds > 120) return -1;
        n = snprintf(dst, capacity,
                     "<sensor_v1>|source=har_ultracompact|class=%s|duration_seconds=%d s|<text>",
                     r->class_name, r->duration_seconds);
        break;
    case SENSOR_ECG:
        if (r->heart_rate_before < 40 || r->heart_rate_before > 180 ||
            r->heart_rate_after < 40 || r->heart_rate_after > 180) return -1;
        n = snprintf(dst, capacity,
                     "<sensor_v1>|source=ecg_mitbih|class=%s|heart_rate_before=%d bpm|heart_rate_after=%d bpm|<text>",
                     r->class_name, r->heart_rate_before, r->heart_rate_after);
        break;
    case SENSOR_FOMO:
        if (r->count < 0 || r->count > 5) return -1;
        n = snprintf(dst, capacity,
                     "<sensor_v1>|source=fomo_voc_person|class=%s|count=%d objects|<text>",
                     r->class_name, r->count);
        break;
    default:
        return -1;
    }
    if (n < 0 || (size_t)n >= capacity) {
        dst[0] = '\0';
        return -1;
    }
    return n;
}
