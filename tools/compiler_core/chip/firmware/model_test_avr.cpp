/*
 * model_test_avr.cpp -- firmware of the model test for AVR boards (Arduino Uno: ATmega328P, 8 bits, 16 MHz,
 * 2 KB of RAM, 32 KB of flash). Same serial protocol as model_test.c, so chip/measure.py reads both the same way.
 * The exported header (XNOR arithmetic) keeps its constants in flash (PROGMEM, read by lib_symbex with
 * pgm_read_dword) and writes each neuron's bit directly (no raw-score arrays), so it fits in 2 KB of RAM.
 * chip/core.py writes BAUD_RATE before this file.
 *
 * Protocol: "SYMBEX ready | ...", STEPS / STEP / RAM lines, then for every 0xA5 + image it answers
 * "R:class,total_us,us_layer0,..."; 0x5A asks for "STACK:bytes never touched by the stack".
 * Times come from micros() (4 us resolution on a 16 MHz AVR).
 */
#include <Arduino.h>

#define SYMBEX_PROFILE 1
static unsigned long marks[8];
#define SYMBEX_MARK(k) (marks[(k)] = micros())
#include "model_symbex.h"
#include "hal/symbex_hal.h"

extern char __heap_start;
extern char* __brkval;
static uint8_t* paint_start;
static uint16_t paint_size;

static void paint_stack(void) {                    /* fill the free RAM with 0xAA, to find the stack peak later */
    char top;
    paint_start = (uint8_t*)(__brkval ? __brkval : &__heap_start);
    uint8_t* end = (uint8_t*)&top - 48;
    paint_size = (uint16_t)(end - paint_start);
    for (uint8_t* p = paint_start; p < end; p++) *p = 0xAA;
}

static uint16_t never_touched(void) {              /* painted bytes the stack never reached */
    uint16_t n = 0;
    while (n < paint_size && paint_start[n] == 0xAA) n++;
    return n;
}

static uint8_t x[SYMBEX_MLP_N_INPUTS];

static void read_exact(uint8_t* d, size_t n) {
    size_t r = 0;
    while (r < n) { int c = Serial.read(); if (c >= 0) d[r++] = (uint8_t)c; }
}

void setup() {
    Serial.begin(BAUD_RATE);
    paint_stack();
    symbex_mlp_infer(x);
    Serial.print(F("\nSYMBEX ready | avr: ATmega328P, "));
    Serial.print(symbex_hal_name());
    Serial.print(F(" | "));
    Serial.print(SYMBEX_MLP_N_INPUTS); Serial.print(F(" inputs, ")); Serial.print(SYMBEX_MLP_N_CLASSES); Serial.println(F(" classes"));
    Serial.print(F("STEPS:")); Serial.println(SYMBEX_MLP_N_STEPS);
    for (int i = 0; i < SYMBEX_MLP_N_STEPS; i++) {
        Serial.print(F("STEP:")); Serial.print(i); Serial.print('|'); Serial.println(symbex_mlp_steps[i]);
    }
    Serial.print(F("RAM:")); Serial.print((unsigned)SYMBEX_MLP_BYTES_BUFFERS); Serial.print(',');
    Serial.print((unsigned)sizeof x); Serial.print(F(",0,")); Serial.println(paint_size);
}

void loop() {
    int m = Serial.read();
    if (m < 0) return;
    if (m == 0x5A) { Serial.print(F("STACK:")); Serial.println(never_touched()); return; }
    if (m != 0xA5) return;
    read_exact(x, sizeof x);
    int c = symbex_mlp_infer(x);
    Serial.print(F("R:")); Serial.print(c); Serial.print(','); Serial.print(marks[SYMBEX_MLP_N_STEPS] - marks[0]);
    for (int i = 0; i < SYMBEX_MLP_N_STEPS; i++) { Serial.print(','); Serial.print(marks[i + 1] - marks[i]); }
    Serial.println();
}
