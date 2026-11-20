#ifndef SENSOR_TEXT_PROMPT_H
#define SENSOR_TEXT_PROMPT_H
#include <stddef.h>

enum sensor_source { SENSOR_HAR = 0, SENSOR_ECG = 1, SENSOR_FOMO = 2 };
typedef struct {
    enum sensor_source source;
    const char *class_name; /* NUL-terminated ASCII identifier, 1..64 characters. */
    int duration_seconds;
    int heart_rate_before;
    int heart_rate_after;
    int count;
} sensor_record;

/* Returns bytes excluding NUL, or -1 for invalid input/insufficient capacity.
 * On error, clears dst[0] when dst != NULL and capacity > 0.
 * Only fields belonging to source are used. No float, heap, or locale dependence.
 * The struct/class storage must not overlap dst. */
int sensor_prompt(char *dst, size_t capacity, const sensor_record *record);
#endif
