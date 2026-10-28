/*
 * model_test_pico.cpp -- firmware of the model test for the Raspberry Pi Pico 2 (RP2350, Arduino-Pico core), on
 * either of its core types: Cortex-M33 (pico2_arm) or Hazard3 RISC-V (pico2_riscv), 150 MHz, one core.
 * Same serial protocol as model_test.c, so chip/measure.py reads every board the same way.
 * chip/core.py writes BAUD_RATE before this file (USB CDC: the baud rate is ignored by the port).
 *
 * Protocol: the board waits for the host's 0x55 (the USB port re-enumerates after flashing, so the host cannot catch
 * lines printed at boot), then prints "SYMBEX ready | <hal> | ...", STEP / RAM lines; for every 0xA5 + image it
 * answers "R:class,total_us,us_layer0,..."; 0x5A asks for "STACK:bytes never touched by the stack"; 0x55 prints
 * the ready lines again.
 *
 * Times come from the RP2350's 1 MHz system timer (time_us_32, 1 us like esp_timer on the ESP32s).
 * RAM: the lut16 table is measured as the heap it took on the first call; the stack of core 0 (from __StackBottom
 * to __StackTop in the linker script) is painted at boot and the never-touched bytes are counted on request.
 * The input image is 16-byte aligned: the first layer reads 4 inputs per 32-bit load only when it is 4-byte aligned.
 */
#include <Arduino.h>
#include "pico/time.h"

#define SYMBEX_PROFILE 1
static uint32_t marks[32];
#define SYMBEX_MARK(k) (marks[(k)] = time_us_32())
#include "model_symbex.h"
#include "hal/symbex_hal.h"

extern "C" {
extern char __StackBottom;
extern char __StackTop;
}

static uint8_t* paint_start;
static uint32_t paint_size;
static uint32_t ram_table;                           /* bytes the HAL allocated on the first call (its table) */
static uint8_t x[SYMBEX_MLP_N_INPUTS] __attribute__((aligned(16)));

static void paint_stack(void) {                    /* fill the free stack with 0xAA, to find its peak later */
    char top;
    paint_start = (uint8_t*)&__StackBottom;
    uint8_t* end = (uint8_t*)&top - 64;
    paint_size = (uint32_t)(end - paint_start);
    for (uint8_t* p = paint_start; p < end; p++) *p = 0xAA;
}

static uint32_t never_touched(void) {              /* painted bytes the stack never reached */
    uint32_t n = 0;
    while (n < paint_size && paint_start[n] == 0xAA) n++;
    return n;
}

static void read_exact(uint8_t* d, size_t n) {
    size_t r = 0;
    while (r < n) {
        int c = Serial.read();
        if (c >= 0) d[r++] = (uint8_t)c;
    }
}

static void print_ready(void) {
    Serial.printf("\nSYMBEX ready | %s | %d inputs, %d classes\n", symbex_hal_name(),
                  SYMBEX_MLP_N_INPUTS, SYMBEX_MLP_N_CLASSES);
    Serial.printf("STEPS:%d\n", SYMBEX_MLP_N_STEPS);
    for (int i = 0; i < SYMBEX_MLP_N_STEPS; i++) Serial.printf("STEP:%d|%s\n", i, symbex_mlp_steps[i]);
    Serial.printf("RAM:%u,%u,%u,%u\n", (unsigned)SYMBEX_MLP_BYTES_BUFFERS, (unsigned)SYMBEX_MLP_N_INPUTS,
                  (unsigned)ram_table, (unsigned)(&__StackTop - &__StackBottom));
    Serial.flush();
}

void setup() {
    Serial.begin(BAUD_RATE);
    paint_stack();
    uint32_t heap_before = rp2040.getFreeHeap();
    symbex_mlp_infer(x);                             /* first call: the HAL builds its table here, not timed */
    ram_table = heap_before - rp2040.getFreeHeap();
    uint8_t hello = 0;
    while (hello != 0x55) read_exact(&hello, 1);    /* the host is listening: only now the ready lines */
    print_ready();
}

void loop() {
    int m = Serial.read();
    if (m < 0) return;
    if (m == 0x55) { print_ready(); return; }
    if (m == 0x5A) { Serial.printf("STACK:%u\n", (unsigned)never_touched()); Serial.flush(); return; }
    if (m != 0xA5) return;
    read_exact(x, sizeof x);
    int c = symbex_mlp_infer(x);
    Serial.printf("R:%d,%lu", c, (unsigned long)(marks[SYMBEX_MLP_N_STEPS] - marks[0]));
    for (int i = 0; i < SYMBEX_MLP_N_STEPS; i++) Serial.printf(",%lu", (unsigned long)(marks[i + 1] - marks[i]));
    Serial.printf("\n");
    Serial.flush();
}
