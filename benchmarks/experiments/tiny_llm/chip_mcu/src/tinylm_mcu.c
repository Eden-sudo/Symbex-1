/* tinylm.c on chips without PSRAM (classic ESP32, Pico 2): every buffer and the KV cache in internal RAM, the
   weights read in place from flash (model.bin is a const array, src/model_data.h). platformio.ini defines
   TINYLM_KV_SPLIT on every board (the cache allocated per layer: the classic ESP32's heap is split in regions). */
#ifdef ESP_PLATFORM
/* the cache's int32 scales in the ESP32's 32-bit-only IRAM heap (~72 KB the byte heap lacks), else the byte heap */
#include <stdlib.h>
#include "esp_heap_caps.h"
static void *tinylm_malloc_words(size_t n) {
    void *p = heap_caps_malloc(n, MALLOC_CAP_32BIT | MALLOC_CAP_EXEC);
    return p ? p : malloc(n);
}
#define TINYLM_MALLOC_WORDS(n) tinylm_malloc_words(n)
#endif
#include "../../c/tinylm.c"
