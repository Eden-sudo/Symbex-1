/*
 * main.cpp -- tiny_llm on chips without PSRAM (classic ESP32, Raspberry Pi Pico 2): model.bin stays in flash (a
 * const array of model_data.h: the ESP32 maps it through its flash cache, the RP2350 through XIP), the KV cache
 * and the per-token buffers in internal RAM. Then, as chip/src/main.cpp on the S3:
 *   1. the validation windows of check_data.h, token by token: a hash of every output (vocab int64 + the final max)
 *      against the one intsim.py computed ("H:window,token,ok,us");
 *   2. the greedy story from the prompt: every token against intsim.py's ("G:token,ok,us").
 * Waits for the host's 0x55 first.
 */
#include <Arduino.h>

extern "C" {
#include "../../c/tinylm.h"
}
#include "check_data.h"
#include "model_data.h"

#if defined(ARDUINO_ARCH_RP2040)
#ifdef __riscv
#define BOARD_NAME "pico2_riscv"
#else
#define BOARD_NAME "pico2_arm"
#endif
static unsigned free_ram(void) { return (unsigned)rp2040.getFreeHeap(); }
#else
#define BOARD_NAME "esp32"
static unsigned free_ram(void) { return (unsigned)ESP.getFreeHeap(); }
#endif

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
    Serial.printf("ID:%u\n", (unsigned)CHECK_ID);       /* which run and reference set this firmware holds */
    Serial.printf("BOARD:%s\n", BOARD_NAME);
    Serial.println("LINEAR:lut8");
    Serial.printf("TINYLM ready | cpu %u MHz | ram %u bytes free\n", (unsigned)(F_CPU / 1000000), free_ram());

    const uint8_t *buf = model_bin;
    uint32_t t0 = micros();
    uint32_t sum = 0;
    for (long i = 0; i < MODEL_BYTES; i++) sum = sum * 31u + buf[i];
    if (sum != MODEL_SUM) fail("model_data.h differs from the run's model.bin");
    static TinyModel m;
    static TinyState s;
    if (tinylm_load(&m, buf, MODEL_BYTES)) fail("load");
#ifndef TINYLM_KV_SPLIT
#error "platformio.ini defines TINYLM_KV_SPLIT (TinyState must match tinylm_mcu.c)"
#endif
    if (tinylm_state_init(&s, &m)) fail("state: not enough RAM");
    int64_t *z = (int64_t *)malloc(8L * m.vocab);
    if (!z) fail("no RAM for the outputs");
    Serial.printf("LOADED %d bytes in %lu ms | ram free %u\n", (int)MODEL_BYTES, (unsigned long)(micros() - t0) / 1000,
                  free_ram());

    int64_t hm;
    for (int w = 0; w < CHECK_WINDOWS; w++) {
        s.pos = 0;
        for (int t = 0; t < CHECK_TOKENS; t++) {
            uint32_t a = micros();
            tinylm_step(&m, &s, check_tokens[w][t], z, &hm);
            uint32_t us = micros() - a;
            Serial.printf("H:%d,%d,%d,%lu\n", w, t, out_hash(z, m.vocab, hm) == check_hash[w][t], (unsigned long)us);
        }
    }

    s.pos = 0;
    int next = 0;
    for (int i = 0; i < GEN_IDS - 1; i++) {
        uint32_t a = micros();
        int tok = tinylm_step(&m, &s, i < GEN_PROMPT ? gen_ids[i] : next, z, &hm);
        uint32_t us = micros() - a;
        next = tok;
        if (i + 1 >= GEN_PROMPT) Serial.printf("G:%d,%d,%lu\n", tok, tok == gen_ids[i + 1], (unsigned long)us);
    }
    Serial.println("DONE");
}

void loop() { delay(1000); }
