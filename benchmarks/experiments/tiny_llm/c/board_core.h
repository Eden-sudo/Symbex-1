/* board_core.h -- the Arduino core / SDK this firmware was built with, as one line "CORE:<name>" for the result JSON.
 * chip_run.py / chip_fp32.py also record the platform pinned in platformio.ini; this line is the board's own build,
 * so a stale firmware read with --no-upload cannot name a core it was not built with (Abraham, #217). */
#pragma once
#include <Arduino.h>

#if defined(ARDUINO_ARCH_RP2040)
#include "RP2040Version.h"
static inline void print_board_core(void) { Serial.printf("CORE:arduino-pico %s\n", ARDUINO_PICO_VERSION_STR); }
#else
#include "esp_arduino_version.h"
#include "esp_idf_version.h"
static inline void print_board_core(void) {
    Serial.printf("CORE:arduino-esp32 %d.%d.%d / esp-idf %d.%d.%d\n", ESP_ARDUINO_VERSION_MAJOR,
                  ESP_ARDUINO_VERSION_MINOR, ESP_ARDUINO_VERSION_PATCH, ESP_IDF_VERSION_MAJOR, ESP_IDF_VERSION_MINOR,
                  ESP_IDF_VERSION_PATCH);
}
#endif
