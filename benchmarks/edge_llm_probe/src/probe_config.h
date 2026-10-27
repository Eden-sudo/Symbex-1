/*
 * probe_config.h -- every board-specific or tunable value of the probe, with its default. Override any PROBE_*
 * macro with -D in platformio.ini (build_flags). All magic numbers of main.cpp live here.
 */
#ifndef PROBE_CONFIG_H
#define PROBE_CONFIG_H

#if defined(__has_include)
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"     /* CONFIG_IDF_TARGET_* */
#  endif
#endif

/* ---- energy schedule ---- */
#ifndef PROBE_PHASE_SECONDS
#define PROBE_PHASE_SECONDS 20          /* duration of every phase; a USB meter needs >= 20 s to settle */
#endif
#ifndef PROBE_MARKER_GPIO
#define PROBE_MARKER_GPIO -1            /* high during every non-idle phase; -1 = disabled */
#endif
#ifndef PROBE_CPU_MHZ
#define PROBE_CPU_MHZ 240               /* CPU clock set at boot */
#endif

/* ---- SD card ----
   ESP32-S3: any GPIOs, set with SD_MMC.setPins (1-bit mode). Defaults: Freenove ESP32-S3-WROOM CAM / ESP32-S3-EYE
   style boards (CLK 39, CMD 38, D0 40). Classic ESP32 (ESP32-CAM): fixed SDMMC slot (CLK 14, CMD 15, D0 2, D1 4,
   D2 12, D3 13), no setPins. Set PROBE_SD_CLK to -1 to disable the SD test on an S3 board without a card slot. */
#ifndef PROBE_SD_CLK
#define PROBE_SD_CLK 39
#endif
#ifndef PROBE_SD_CMD
#define PROBE_SD_CMD 38
#endif
#ifndef PROBE_SD_D0
#define PROBE_SD_D0 40
#endif
#ifndef PROBE_SD_FREQ_KHZ
#define PROBE_SD_FREQ_KHZ 40000         /* SDMMC_FREQ_HIGHSPEED */
#endif
#ifndef PROBE_SD_FILE_MB
#define PROBE_SD_FILE_MB 8              /* size of the test file, written once if missing */
#endif
/* 4-bit mode is tested on the classic ESP32 only. It drives GPIO4, which is the flash LED of the ESP32-CAM. */
#if !defined(PROBE_SD_TEST_4BIT) && !defined(CONFIG_IDF_TARGET_ESP32S3)
#define PROBE_SD_TEST_4BIT 1
#endif
#ifndef PROBE_SD_TEST_4BIT
#define PROBE_SD_TEST_4BIT 0
#endif

/* ---- camera: define exactly one of PROBE_CAMERA_AI_THINKER, PROBE_CAMERA_ESP32S3_EYE, PROBE_CAMERA_NONE ---- */
#if !defined(PROBE_CAMERA_AI_THINKER) && !defined(PROBE_CAMERA_ESP32S3_EYE) && !defined(PROBE_CAMERA_NONE)
#define PROBE_CAMERA_NONE
#endif

/* ---- memory-bandwidth test sizes (bytes) ---- */
#define PROBE_BW_TARGET_BYTES   (4u * 1024u * 1024u)   /* bytes read per repetition (passes over small buffers) */
#define PROBE_BW_REPS           7                      /* repetitions of every bandwidth test (best and median) */
#define PROBE_SD_REPS           3                      /* the SD tests are slow: fewer repetitions */
#define PROBE_SRAM_SMALL        (16u * 1024u)
#define PROBE_SRAM_LARGE        (128u * 1024u)
#define PROBE_PSRAM_CACHED      (16u * 1024u)          /* fits the data cache: shows the cache-hit rate */
#define PROBE_PSRAM_MEDIUM      (1u * 1024u * 1024u)
#define PROBE_PSRAM_LARGE       (4u * 1024u * 1024u)
#define PROBE_FLASH_BYTES       (1u * 1024u * 1024u)   /* mapped part of the probe partition */
#define PROBE_COPY_CHUNK        (16u * 1024u)          /* SRAM destination of the memcpy tests */

/* ---- probe partition (partitions_probe.csv) ---- */
#define PROBE_PART_NAME         "probe"
#define PROBE_PART_SUBTYPE      0x40
#define PROBE_PART_TAG_BYTES    4096u                  /* tag sector at the start; the payload follows it */
#define PROBE_PART_TAG          "SYMBEXPROBE-v1\0"     /* 16 bytes with the terminator */

/* ---- kernel benchmark ---- */
#define PROBE_KERNEL_REPS       3                      /* calls per kernel and location; best (minimum) is reported */

#endif
