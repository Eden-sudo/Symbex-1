/*
 * main.cpp -- tiny_llm on the ESP32-S3: model.bin from the "model" flash partition into PSRAM, then
 *   1. the validation windows of check_data.h, token by token: a hash of every output (4096 int64 + the final max)
 *      against the one intsim.py computed ("H:window,token,ok,us");
 *   2. the greedy story from the prompt: every token against intsim.py's ("G:token,ok,us").
 * Waits for the host's 0x55 first (native USB: opening the port resets the chip).
 */
#include <Arduino.h>
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

extern "C" {
#include "../../c/tinylm.h"
}
#include "check_data.h"

static uint64_t out_hash(const int64_t *z, int vocab, int64_t hm) {
    uint64_t h = 0;
    for (int r = 0; r < vocab; r++) h += (uint64_t)z[r] * ((uint64_t)(r + 1) * 0x9E3779B97F4A7C15ull);
    return h + (uint64_t)hm * 0xD6E8FEB86659FD93ull;
}

static void fail(const char *why) {
    for (;;) { Serial.printf("ERROR: %s\n", why); delay(2000); }
}

void setup() {
    Serial.begin(115200);
    while (Serial.read() != 0x55) delay(10);
    Serial.printf("TINYLM ready | cpu %d MHz | psram %u bytes free\n", getCpuFrequencyMhz(),
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40,
                                                           "model");
    if (!part) fail("no model partition");
    uint8_t *buf = (uint8_t *)heap_caps_malloc(MODEL_BYTES, MALLOC_CAP_SPIRAM);
    if (!buf) fail("no PSRAM for the model");
    int64_t t0 = esp_timer_get_time();
    if (esp_partition_read(part, 0, buf, MODEL_BYTES) != ESP_OK) fail("partition read");
    uint32_t sum = 0;
    for (long i = 0; i < MODEL_BYTES; i++) sum = sum * 31u + buf[i];
    if (sum != MODEL_SUM) fail("model.bin in flash differs from the one of this run (write it again)");
    static TinyModel m;
    static TinyState s;
    if (tinylm_load(&m, buf, MODEL_BYTES) || tinylm_state_init(&s, &m)) fail("load / state");
    int64_t *z = (int64_t *)heap_caps_malloc(8L * m.vocab, MALLOC_CAP_INTERNAL);
    if (!z) fail("no RAM for the outputs");
    Serial.printf("LOADED %d bytes in %lld ms | internal free %u | psram free %u\n", (int)MODEL_BYTES,
                  (esp_timer_get_time() - t0) / 1000, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    int64_t hm;
    for (int w = 0; w < CHECK_WINDOWS; w++) {
        s.pos = 0;
        for (int t = 0; t < CHECK_TOKENS; t++) {
            int64_t a = esp_timer_get_time();
            tinylm_step(&m, &s, check_tokens[w][t], z, &hm);
            int64_t us = esp_timer_get_time() - a;
            Serial.printf("H:%d,%d,%d,%lld\n", w, t, out_hash(z, m.vocab, hm) == check_hash[w][t], us);
        }
    }

    s.pos = 0;
    int next = 0;
    for (int i = 0; i < GEN_IDS - 1; i++) {
        int64_t a = esp_timer_get_time();
        int tok = tinylm_step(&m, &s, i < GEN_PROMPT ? gen_ids[i] : next, z, &hm);
        int64_t us = esp_timer_get_time() - a;
        next = tok;
        if (i + 1 >= GEN_PROMPT) Serial.printf("G:%d,%d,%lld\n", tok, tok == gen_ids[i + 1], us);
    }
    Serial.println("DONE");
}

void loop() { delay(1000); }
