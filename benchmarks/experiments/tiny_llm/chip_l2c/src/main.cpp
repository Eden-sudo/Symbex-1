/*
 * main.cpp -- llama2.c's run.c (fp32, unchanged) on chips without PSRAM, the comparison point of the binary 260K
 * (README "The comparison point on the chip"): the checkpoint is a const array in flash (model_data.h), the KV cache
 * and the buffers in internal RAM, for GEN_IDS positions. The greedy story from the prompt: every token against run.c
 * on the PC ("G:token,ok,us", as ../chip_mcu/src/main.cpp). Waits for the host's 0x55 first.
 */
#include <Arduino.h>

#include "l2c_mcu.h"
#include "gen_data.h"
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

static void fail(const char *why) {
    for (;;) { Serial.printf("ERROR: %s\n", why); delay(2000); }
}

void setup() {
    Serial.begin(115200);
    while (Serial.read() != 0x55) delay(10);
    Serial.printf("ID:%u\n", (unsigned)GEN_ID);
    Serial.printf("BOARD:%s\n", BOARD_NAME);
    Serial.printf("L2C ready | cpu %u MHz | ram %u bytes free\n", (unsigned)(F_CPU / 1000000), free_ram());
    uint32_t sum = 0;
    for (long i = 0; i < MODEL_BYTES; i++) sum = sum * 31u + model_bin[i];
    if (sum != MODEL_SUM) fail("model_data.h differs from the exported checkpoint");
    /* run.c exits on a failed calloc (the board restarts with no line): check the total first. A fragmented heap
       can still refuse one large block (the KV cache) with enough bytes free in total, and the count leaves out the
       malloc headers (ESP32: 160 bytes on Arduino 2.0.17, 1640 on 3.3, #210). */
    long need = l2c_state_bytes(model_bin, GEN_IDS);
    if (need > (long)free_ram()) {
        static char why[96];
        snprintf(why, sizeof why, "run.c needs %ld bytes of RAM, %u free", need, free_ram());
        fail(why);
    }
    uint32_t t0 = micros();
    int ctx = l2c_init(model_bin, GEN_IDS);
    Serial.printf("LOADED %d bytes, context %d, in %lu ms | ram free %u\n", (int)MODEL_BYTES, ctx,
                  (unsigned long)(micros() - t0) / 1000, free_ram());
    int next = 0;
    for (int i = 0; i < GEN_IDS - 1; i++) {
        uint32_t a = micros();
        int tok = l2c_step(i < GEN_PROMPT ? gen_ids[i] : next, i);
        uint32_t us = micros() - a;
        next = tok;
        if (i + 1 >= GEN_PROMPT) Serial.printf("G:%d,%d,%lu\n", tok, tok == gen_ids[i + 1], (unsigned long)us);
    }
    Serial.println("DONE");
}

void loop() { delay(1000); }
