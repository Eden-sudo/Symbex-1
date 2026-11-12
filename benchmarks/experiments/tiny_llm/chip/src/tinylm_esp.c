/* tinylm.c for the ESP32-S3: per-token buffers and the lut8 table in internal RAM, the KV cache in PSRAM. */
#include "esp_heap_caps.h"
#define TINYLM_MALLOC_FAST(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define TINYLM_MALLOC_BIG(n) heap_caps_malloc((n), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#include "../../c/tinylm.c"
