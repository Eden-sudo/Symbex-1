/*
 * main.cpp -- edge-LLM probe for the ESP32-S3 and the classic ESP32 (AI-Thinker ESP32-CAM).
 *
 * Measures, on the chip, the two numbers that decide whether a natively trained 1-bit LLM can run usefully:
 *   1. memory read bandwidth per tier (internal SRAM, PSRAM, memory-mapped flash, SD card),
 *   2. cycles per weight of the 1-bit matrix-vector kernels in kernels/onebit_gemv.c, per weight location.
 * Then it loops forever through an energy schedule (one steady workload per phase) for a USB power meter.
 *
 * Output: machine-readable lines (INFO / BW / CAM / KERN / PHASE, comma separated, see README.md) and human
 * comments starting with '#'. parse_log.py reads the lines and ignores the comments.
 * Board-specific values and tunables: probe_config.h (override with -D in platformio.ini).
 */
#include <Arduino.h>
#include <SD_MMC.h>
#include <string.h>
#include <esp_heap_caps.h>
#include <esp_partition.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include "probe_config.h"
#include "onebit_gemv.h"
#include "hal/symbex_hal.h"
#include "layers/dense/symbex_dense.h"
#include "ops/symbex_and.h"           /* symbex_and_input_u8: the first layer of the exported SYMBEX MLPs */   /* lib_symbex: the first-layer kernels a SYMBEX model runs, through this chip's HAL */

#if !defined(PROBE_CAMERA_NONE)
#include <esp_camera.h>
#include "camera_pins.h"
#endif

#define PROBE_VERSION 1
#define START_DELAY_MS 3000

/* ============================================================================================================ */
/* small helpers                                                                                                */
/* ============================================================================================================ */

static volatile uint32_t g_sink;   /* every measured loop stores its result here: the compiler cannot drop the reads */

static inline uint32_t cycles_now() { return ESP.getCycleCount(); }
static inline int64_t micros64() { return esp_timer_get_time(); }

/* deterministic xorshift32: weights and activations are reproducible, flash payload included */
static inline uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}
static const uint32_t PAYLOAD_SEED = 0x5EDA0001u;

static void fill_pattern(uint32_t* p, size_t words, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < words; i++) p[i] = xorshift32(&s);
}

static void* alloc_internal(size_t bytes) { return heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT); }
static void* alloc_psram(size_t bytes) { return heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }

/* sequential 32-bit read-sum, 4 independent accumulators. noipa: without it GCC infers the function is pure and
 * folds repeated calls on the same buffer into one, which inflated read32_seq by the number of passes */
__attribute__((noinline, noipa)) static uint32_t read_sum(const uint32_t* p, size_t words) {
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (size_t i = 0; i + 4 <= words; i += 4) {
        s0 += p[i];
        s1 += p[i + 1];
        s2 += p[i + 2];
        s3 += p[i + 3];
    }
    return s0 + s1 + s2 + s3;
}

struct BwStat {
    double best;
    double median;
};

static BwStat summarise(double* v, int n) {
    for (int i = 1; i < n; i++) {   /* insertion sort, n is tiny */
        double key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
    BwStat s;
    s.best = v[n - 1];
    s.median = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return s;
}

/* One bandwidth test: `work` reads `bytes` bytes per call. MB/s = 10^6 bytes per second = bytes / microseconds. */
template <typename Work>
static void bw_report(const char* tier, const char* test, size_t bytes, int reps, Work&& work) {
    double mbps[16];
    if (reps > 16) reps = 16;
    work();   /* warm-up: first-touch of the buffer, file open */
    for (int r = 0; r < reps; r++) {
        const int64_t t0 = micros64();
        work();
        const int64_t dt = micros64() - t0;
        mbps[r] = dt > 0 ? (double)bytes / (double)dt : 0.0;
    }
    const BwStat s = summarise(mbps, reps);
    Serial.printf("BW,%s,%s,%u,%.2f,%.2f\n", tier, test, (unsigned)bytes, s.best, s.median);
}

static uint32_t passes_for(size_t bytes) {
    const uint32_t p = PROBE_BW_TARGET_BYTES / (uint32_t)bytes;
    return p ? p : 1;
}

/* ============================================================================================================ */
/* probe partition: 1 MB of reproducible pseudo-random words, memory-mapped                                     */
/* ============================================================================================================ */

struct FlashProbe {
    const uint32_t* payload;   /* first word after the tag sector; PROBE_FLASH_BYTES readable */
    bool ok;
};
static FlashProbe g_flash = {nullptr, false};

static bool partition_has_tag(const esp_partition_t* part) {
    uint8_t tag[16];
    static const char expect[16] = PROBE_PART_TAG;
    if (esp_partition_read(part, 0, tag, sizeof(tag)) != ESP_OK) return false;
    return memcmp(tag, expect, sizeof(tag)) == 0;
}

/* erase, write the payload, write the tag LAST: an interrupted fill is redone at the next boot */
static bool partition_fill(const esp_partition_t* part) {
    Serial.println("# probe partition has no tag: filling it once (erase + write ~1 MB, a few seconds)");
    if (esp_partition_erase_range(part, 0, part->size) != ESP_OK) return false;
    static uint32_t chunk[PROBE_PART_TAG_BYTES / 4];   /* static: keep the 4 KB off the 8 KB loop-task stack */
    uint32_t s = PAYLOAD_SEED;
    for (size_t off = PROBE_PART_TAG_BYTES; off + sizeof(chunk) <= part->size; off += sizeof(chunk)) {
        for (size_t i = 0; i < sizeof(chunk) / 4; i++) chunk[i] = xorshift32(&s);
        if (esp_partition_write(part, off, chunk, sizeof(chunk)) != ESP_OK) return false;
    }
    static const char tag[16] = PROBE_PART_TAG;
    return esp_partition_write(part, 0, tag, sizeof(tag)) == ESP_OK;
}

static void flash_probe_init() {
    const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                                           (esp_partition_subtype_t)PROBE_PART_SUBTYPE, PROBE_PART_NAME);
    if (!part) {
        Serial.println("# partition '" PROBE_PART_NAME "' not found: flash tests skipped (use partitions_probe.csv)");
        Serial.println("INFO,flash_probe,missing");
        return;
    }
    if (part->size < PROBE_PART_TAG_BYTES + PROBE_FLASH_BYTES) {
        Serial.println("# probe partition too small: flash tests skipped");
        Serial.println("INFO,flash_probe,too_small");
        return;
    }
    if (!partition_has_tag(part) && !partition_fill(part)) {
        Serial.println("INFO,flash_probe,fill_failed");
        return;
    }
    const void* ptr = nullptr;
    spi_flash_mmap_handle_t handle;   /* never unmapped: the mapping lives for the whole run */
    if (esp_partition_mmap(part, 0, part->size, SPI_FLASH_MMAP_DATA, &ptr, &handle) != ESP_OK) {
        Serial.println("INFO,flash_probe,mmap_failed");
        return;
    }
    g_flash.payload = (const uint32_t*)((const uint8_t*)ptr + PROBE_PART_TAG_BYTES);
    g_flash.ok = true;
    Serial.printf("INFO,flash_probe,ok_%u_KB\n", (unsigned)(part->size / 1024));
}

/* evicts the data cache (shared by flash and PSRAM) so a timed call starts with cold weights */
static void flush_cache() {
    if (g_flash.ok) g_sink = g_sink + read_sum(g_flash.payload, 256u * 1024u / 4u);
}

/* ============================================================================================================ */
/* SD card                                                                                                      */
/* ============================================================================================================ */

static bool g_sd_ok = false;
static char g_sd_path[] = "/probe_data.bin";

static bool sd_mount(bool mode1bit) {
    SD_MMC.end();
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    if (PROBE_SD_CLK < 0) return false;
    if (mode1bit) SD_MMC.setPins(PROBE_SD_CLK, PROBE_SD_CMD, PROBE_SD_D0);
#endif
    return SD_MMC.begin("/sdcard", mode1bit, false, PROBE_SD_FREQ_KHZ);
}

static const char* sd_type_name() {
    switch (SD_MMC.cardType()) {
        case CARD_MMC: return "MMC";
        case CARD_SD: return "SDSC";
        case CARD_SDHC: return "SDHC";
        default: return "unknown";
    }
}

static void sd_info() {
    g_sd_ok = sd_mount(true);
    if (!g_sd_ok || SD_MMC.cardType() == CARD_NONE) {
        g_sd_ok = false;
        Serial.println("INFO,sd,none");
        return;
    }
    Serial.printf("INFO,sd,%s_%lluMB\n", sd_type_name(), (unsigned long long)(SD_MMC.cardSize() / (1024ull * 1024ull)));
}

static size_t sd_file_size() {
    File f = SD_MMC.open(g_sd_path, FILE_READ);
    if (!f) return 0;
    const size_t n = f.size();
    f.close();
    return n;
}

/* the test file is written once (only if missing or short); the write speed of that one pass is reported */
static void sd_prepare_file() {
    const size_t want = (size_t)PROBE_SD_FILE_MB * 1024u * 1024u;
    if (sd_file_size() >= want) return;
    Serial.printf("# SD: writing %d MB test file %s (once)\n", PROBE_SD_FILE_MB, g_sd_path);
    const size_t chunk = 32u * 1024u;
    uint32_t* buf = (uint32_t*)alloc_internal(chunk);
    if (!buf) return;
    fill_pattern(buf, chunk / 4, PAYLOAD_SEED);
    File f = SD_MMC.open(g_sd_path, FILE_WRITE);
    if (f) {
        const int64_t t0 = micros64();
        size_t done = 0;
        while (done < want && f.write((const uint8_t*)buf, chunk) == chunk) done += chunk;
        f.close();
        const double mbps = (double)done / (double)(micros64() - t0);
        Serial.printf("BW,sd_1bit,file_write_once_buf32k,%u,%.2f,%.2f\n", (unsigned)done, mbps, mbps);
    }
    free(buf);
}

static void sd_read_tests(const char* tier) {
    const size_t file_bytes = sd_file_size();
    if (file_bytes == 0) return;
    static const size_t BUF_SIZES[2] = {4u * 1024u, 32u * 1024u};
    static const char* NAMES[2] = {"file_read_buf4k", "file_read_buf32k"};
    for (int b = 0; b < 2; b++) {
        uint8_t* buf = (uint8_t*)alloc_internal(BUF_SIZES[b]);
        if (!buf) continue;
        bw_report(tier, NAMES[b], file_bytes, PROBE_SD_REPS, [&]() {
            File f = SD_MMC.open(g_sd_path, FILE_READ);
            int n;
            while ((n = f.read(buf, BUF_SIZES[b])) > 0) g_sink = g_sink + buf[0];
            f.close();
        });
        free(buf);
    }
}

static void sd_bandwidth() {
    if (!g_sd_ok) return;
    sd_prepare_file();
    sd_read_tests("sd_1bit");
#if PROBE_SD_TEST_4BIT
    Serial.println("# SD 4-bit mode drives GPIO4 = the flash LED of the ESP32-CAM: it will light up");
    if (sd_mount(false)) {
        sd_read_tests("sd_4bit");
    } else {
        Serial.println("# SD 4-bit mount failed");
        Serial.println("INFO,sd_4bit,fail");
    }
    sd_mount(true);   /* back to 1-bit for the energy schedule */
#endif
}

/* ============================================================================================================ */
/* camera                                                                                                       */
/* ============================================================================================================ */

#if !defined(PROBE_CAMERA_NONE)
static bool g_camera_inited = false;

static esp_err_t camera_start() {
    camera_config_t cfg = {};
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer = LEDC_TIMER_0;
    cfg.pin_d0 = Y2_GPIO_NUM;
    cfg.pin_d1 = Y3_GPIO_NUM;
    cfg.pin_d2 = Y4_GPIO_NUM;
    cfg.pin_d3 = Y5_GPIO_NUM;
    cfg.pin_d4 = Y6_GPIO_NUM;
    cfg.pin_d5 = Y7_GPIO_NUM;
    cfg.pin_d6 = Y8_GPIO_NUM;
    cfg.pin_d7 = Y9_GPIO_NUM;
    cfg.pin_xclk = XCLK_GPIO_NUM;
    cfg.pin_pclk = PCLK_GPIO_NUM;
    cfg.pin_vsync = VSYNC_GPIO_NUM;
    cfg.pin_href = HREF_GPIO_NUM;
    cfg.pin_sccb_sda = SIOD_GPIO_NUM;
    cfg.pin_sccb_scl = SIOC_GPIO_NUM;
    cfg.pin_pwdn = PWDN_GPIO_NUM;
    cfg.pin_reset = RESET_GPIO_NUM;
    cfg.xclk_freq_hz = 20000000;
    cfg.pixel_format = PIXFORMAT_JPEG;
    cfg.frame_size = FRAMESIZE_QVGA;
    cfg.jpeg_quality = 12;
    cfg.fb_count = 2;
    cfg.fb_location = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode = CAMERA_GRAB_LATEST;
    const esp_err_t err = esp_camera_init(&cfg);
    g_camera_inited = (err == ESP_OK);
    return err;
}

static void camera_stop() {
    if (g_camera_inited) esp_camera_deinit();
    g_camera_inited = false;
}

/* init, grab 30 JPEG frames, deinit (the sensor must not run during the other measurements) */
static bool camera_smoke_test() {
    const esp_err_t err = camera_start();
    if (err != ESP_OK) {
        Serial.printf("INFO,camera,fail:0x%x\n", (unsigned)err);
        return false;
    }
    sensor_t* sensor = esp_camera_sensor_get();
    if (sensor) Serial.printf("INFO,camera_pid,0x%x\n", (unsigned)sensor->id.PID);
    const int frames = 30;
    size_t bytes = 0;
    uint16_t w = 0, h = 0;
    int got = 0;
    const int64_t t0 = micros64();
    for (int i = 0; i < frames; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) continue;
        got++;
        bytes += fb->len;
        w = fb->width;
        h = fb->height;
        esp_camera_fb_return(fb);
    }
    const double secs = (double)(micros64() - t0) / 1e6;
    camera_stop();
    if (got == 0) {
        Serial.println("INFO,camera,fail:no_frame");
        return false;
    }
    Serial.println("INFO,camera,ok");
    Serial.printf("CAM,%u,%u,%u,%.1f\n", (unsigned)w, (unsigned)h, (unsigned)(bytes / got), got / secs);
    return true;
}
#endif

/* ============================================================================================================ */
/* boot information                                                                                             */
/* ============================================================================================================ */

static bool g_camera_ok = false;

static void print_info() {
    Serial.printf("INFO,probe_version,%d\n", PROBE_VERSION);
    /* the detected module; ARDUINO_BOARD is only the PlatformIO build target and can name a different variant */
    Serial.printf("INFO,board,%s_%uMB_flash_%uMB_psram\n", ESP.getChipModel(),
                  (unsigned)(ESP.getFlashChipSize() >> 20), (unsigned)(ESP.getPsramSize() >> 20));
    Serial.printf("INFO,pio_board,%s\n", ARDUINO_BOARD);
    Serial.printf("INFO,build,%s_%s\n", __DATE__, __TIME__);
    Serial.printf("INFO,sdk,%s\n", ESP.getSdkVersion());
    Serial.printf("INFO,chip,%s\n", ESP.getChipModel());
    Serial.printf("INFO,chip_revision,%d\n", (int)ESP.getChipRevision());
    Serial.printf("INFO,cores,%d\n", (int)ESP.getChipCores());
    Serial.printf("INFO,cpu_mhz,%u\n", (unsigned)getCpuFrequencyMhz());
    Serial.printf("INFO,phase_seconds,%d\n", PROBE_PHASE_SECONDS);
    Serial.printf("INFO,heap_internal_free,%u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    Serial.printf("INFO,heap_internal_largest,%u\n", (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    Serial.printf("INFO,psram_size,%u\n", (unsigned)ESP.getPsramSize());
    Serial.printf("INFO,symbex_hal,%s\n", symbex_hal_name());   /* the lib_symbex HAL behind the symbex_* kernels */
    Serial.printf("INFO,psram_free,%u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    Serial.printf("INFO,flash_size,%u\n", (unsigned)ESP.getFlashChipSize());
    Serial.printf("INFO,flash_speed_hz,%u\n", (unsigned)ESP.getFlashChipSpeed());
    flash_probe_init();
    sd_info();
#if defined(PROBE_CAMERA_NONE)
    Serial.println("INFO,camera,skipped");
#else
    g_camera_ok = camera_smoke_test();
#endif
}

/* ============================================================================================================ */
/* memory bandwidth                                                                                             */
/* ============================================================================================================ */

static void bw_read_buffer(const char* tier, const char* test, const uint32_t* buf, size_t bytes) {
    const uint32_t passes = passes_for(bytes);
    bw_report(tier, test, bytes * passes, PROBE_BW_REPS, [&]() {
        uint32_t s = 0;
        for (uint32_t p = 0; p < passes; p++) {
            s += read_sum(buf, bytes / 4);
            /* without this barrier GCC sees read_sum as pure (same buffer, same result) and calls it ONCE for all
               the passes: the small buffers reported 100+ GB/s and the 1 MB ones 4x their real bandwidth */
            __asm__ __volatile__("" ::: "memory");
        }
        g_sink = g_sink + s;
    });
}

static void bw_copy_to_sram(const char* tier, const uint8_t* src, size_t bytes, uint8_t* dst) {
    const uint32_t passes = passes_for(bytes);
    bw_report(tier, "memcpy_to_sram", bytes * passes, PROBE_BW_REPS, [&]() {
        for (uint32_t p = 0; p < passes; p++)
            for (size_t off = 0; off + PROBE_COPY_CHUNK <= bytes; off += PROBE_COPY_CHUNK)
                memcpy(dst, src + off, PROBE_COPY_CHUNK);
        g_sink = g_sink + dst[0];
    });
}

static void bw_buffer_tier(const char* tier, size_t bytes, bool internal, uint8_t* copy_dst) {
    uint32_t* buf = (uint32_t*)(internal ? alloc_internal(bytes) : alloc_psram(bytes));
    if (!buf) {
        Serial.printf("# skip %s %u KB: allocation failed\n", tier, (unsigned)(bytes / 1024));
        return;
    }
    fill_pattern(buf, bytes / 4, PAYLOAD_SEED);
    bw_read_buffer(tier, "read32_seq", buf, bytes);
    if (bytes >= PROBE_COPY_CHUNK) bw_copy_to_sram(tier, (const uint8_t*)buf, bytes, copy_dst);
    free(buf);
}

static void memory_bandwidth() {
    Serial.println("# --- memory bandwidth: BW,<tier>,<test>,<bytes per repetition>,<MB/s best>,<MB/s median> (MB = 10^6 B)");
    uint8_t* copy_dst = (uint8_t*)alloc_internal(PROBE_COPY_CHUNK);
    if (!copy_dst) return;
    bw_buffer_tier("sram_16KB", PROBE_SRAM_SMALL, true, copy_dst);
    bw_buffer_tier("sram_128KB", PROBE_SRAM_LARGE, true, copy_dst);
    if (ESP.getPsramSize() > 0) {
        bw_buffer_tier("psram_16KB", PROBE_PSRAM_CACHED, false, copy_dst);
        bw_buffer_tier("psram_1MB", PROBE_PSRAM_MEDIUM, false, copy_dst);
        bw_buffer_tier("psram_4MB", PROBE_PSRAM_LARGE, false, copy_dst);
    } else {
        Serial.println("# no PSRAM: PSRAM tests skipped");
    }
    if (g_flash.ok) {
        bw_read_buffer("flash_mmap_1MB", "read32_seq", g_flash.payload, PROBE_FLASH_BYTES);
        bw_copy_to_sram("flash_mmap_1MB", (const uint8_t*)g_flash.payload, PROBE_FLASH_BYTES, copy_dst);
    }
    free(copy_dst);
    sd_bandwidth();
}

/* ============================================================================================================ */
/* 1-bit kernels                                                                                                */
/* ============================================================================================================ */

/* second core: runs the lut4 rows [r0, r1) of a job, signalled by a semaphore */
struct LutJob {
    const uint32_t* w;
    const int16_t* lut;
    int32_t* y;
    int n, r0, r1;
};
static LutJob g_job;
static SemaphoreHandle_t g_job_start, g_job_done;
static bool g_worker_ready = false;

static void worker_task(void*) {
    for (;;) {
        xSemaphoreTake(g_job_start, portMAX_DELAY);
        gemv_lut4_rows(g_job.w, g_job.lut, g_job.y, g_job.n, g_job.r0, g_job.r1);
        xSemaphoreGive(g_job_done);
    }
}

static void worker_start() {
    g_job_start = xSemaphoreCreateBinary();
    g_job_done = xSemaphoreCreateBinary();
    const BaseType_t other_core = 1 - xPortGetCoreID();
    g_worker_ready = xTaskCreatePinnedToCore(worker_task, "probe_w", 4096, nullptr, 2, nullptr, other_core) == pdPASS;
}

/* lut4 on two cores: rows [0, m/2) here, [m/2, m) on the other core. Returns after both halves are done. */
static void gemv_lut4_2core(const uint32_t* w, const int16_t* lut, int32_t* y, int m, int n) {
    g_job = {w, lut, y, n, m / 2, m};
    xSemaphoreGive(g_job_start);
    gemv_lut4_rows(w, lut, y, n, 0, m / 2);
    xSemaphoreTake(g_job_done, portMAX_DELAY);
}

/* activations of one input vector, in the representation each kernel needs */
struct KernelInputs {
    int n;
    int8_t* x8;        /* int8 in [-128, 127] */
    uint8_t* a4;       /* 4-bit unsigned a in [0, 15] */
    int8_t* x4;        /* a4 - 8 as int8: what gemv_ref sees for the bit-serial kernel */
    int8_t* x1;        /* +-1 as int8: what gemv_ref sees for the xnor kernel */
    int16_t* lut;
    uint32_t* planes;
    uint32_t* xbits;
};

static bool inputs_alloc(KernelInputs* in, int n) {
    in->n = n;
    in->x8 = (int8_t*)alloc_internal(n);
    in->a4 = (uint8_t*)alloc_internal(n);
    in->x4 = (int8_t*)alloc_internal(n);
    in->x1 = (int8_t*)alloc_internal(n);
    in->lut = (int16_t*)alloc_internal(ONEBIT_LUT4_ENTRIES(n) * sizeof(int16_t));
    in->planes = (uint32_t*)alloc_internal(ONEBIT_BS4_WORDS(n) * sizeof(uint32_t));
    in->xbits = (uint32_t*)alloc_internal((n / 32) * sizeof(uint32_t));
    return in->x8 && in->a4 && in->x4 && in->x1 && in->lut && in->planes && in->xbits;
}

static void inputs_free(KernelInputs* in) {
    free(in->x8);
    free(in->a4);
    free(in->x4);
    free(in->x1);
    free(in->lut);
    free(in->planes);
    free(in->xbits);
}

static void inputs_fill(KernelInputs* in) {
    uint32_t s = 0xA11CE001u;
    for (int i = 0; i < in->n; i++) {
        const uint32_t r = xorshift32(&s);
        in->x8[i] = (int8_t)(r & 0xFF);
        in->a4[i] = (uint8_t)((r >> 8) & 15u);
        in->x4[i] = (int8_t)((int)in->a4[i] - 8);
        in->x1[i] = ((r >> 16) & 1u) ? 1 : -1;
    }
}

/* cold: the data cache is flushed before every call, so PSRAM and flash weights stream in like a layer that was last
   touched a whole token ago. Returns the best (minimum) cycle count of `reps` calls. */
template <typename Fn>
static uint32_t best_cycles(bool cold, Fn&& fn) {
    uint32_t best = 0xFFFFFFFFu;
    for (int r = 0; r < PROBE_KERNEL_REPS; r++) {
        if (cold) flush_cache();
        const uint32_t c0 = cycles_now();
        fn();
        const uint32_t dt = cycles_now() - c0;
        if (dt < best) best = dt;
    }
    return best;
}

static void kern_report(const char* kernel, const char* mem, int m, int n, uint32_t cycles, bool ok) {
    const double cpw = (double)cycles / ((double)m * (double)n);
    const double us = (double)cycles / (double)getCpuFrequencyMhz();
    Serial.printf("KERN,%s,%s,%d,%d,%.3f,%.1f,%s\n", kernel, mem, m, n, cpw, us, ok ? "OK" : "FAIL");
}

/* ---- lib_symbex's own first-layer kernels (uint8 inputs x 1-bit weights), through this chip's HAL ---- */

/* the uint8 reference of lib_symbex's convention: raw = 2 * sum(u where bit = 1) - sum(u), bit 1 = weight +1, rows of
   ceil(n / 32) words, bit i of word i / 32 = input i (the probe's packing when n is a multiple of 32) */
static void u8_reference(const uint32_t* w, int rw, const uint8_t* u, int32_t* raw, int m, int n) {
    for (int r = 0; r < m; r++) {
        int32_t pos = 0, sum = 0;
        for (int i = 0; i < n; i++) {
            sum += u[i];
            if ((w[(size_t)r * rw + i / 32] >> (i % 32)) & 1u) pos += u[i];
        }
        raw[r] = 2 * pos - sum;
    }
}

static bool bits_match(const uint32_t* bits, const int32_t* raw, const int32_t* thr, int m) {
    for (int r = 0; r < m; r++)
        if ((int)((bits[r / 32] >> (r % 32)) & 1u) != (int)(raw[r] >= thr[r])) return false;
    return true;
}

/* symbex_dense_input_u8_mask (int32 raw scores) and symbex_dense_input_u8_planes_bits with k = 4 and 8 (bits after a
   threshold; the k bit planes of the input are built inside each call). Every result is checked on the chip against
   u8_reference; mask also against the probe's gemv_ref (u = x8 + 128: raw_u8 = raw_int8 + 128 * rowsum). */
static void symbex_kernels_at(const char* mem, const uint32_t* w, int m, int n, KernelInputs* in, const int32_t* rowsum,
                              const int32_t* y_ref, int32_t* y) {
    const bool cold = strcmp(mem, "sram") != 0;
    const int rw = (n + 31) / 32;
    uint8_t* u = (uint8_t*)alloc_internal(n);
    uint8_t* uq = (uint8_t*)alloc_internal(n);
    int32_t* ref = (int32_t*)alloc_internal((size_t)m * sizeof(int32_t));
    int32_t* thr = (int32_t*)alloc_internal((size_t)m * sizeof(int32_t));
    uint32_t* planes = (uint32_t*)alloc_internal((size_t)8 * rw * sizeof(uint32_t));
    uint32_t* bits = (uint32_t*)alloc_internal((size_t)((m + 31) / 32) * sizeof(uint32_t));
    if (!u || !uq || !ref || !thr || !planes || !bits) {
        Serial.printf("# skip symbex kernels %dx%d in %s: no SRAM for their buffers\n", m, n, mem);
    } else {
        const struct SymbexLayerData layer = {(uint16_t)n, (uint16_t)m, 1, w};
        for (int i = 0; i < n; i++) u[i] = (uint8_t)((int)in->x8[i] + 128);
        u8_reference(w, rw, u, ref, m, n);
        bool ok = true;
        if (y_ref && rowsum)
            for (int r = 0; r < m; r++) ok = ok && ref[r] == y_ref[r] + 128 * rowsum[r];
        const uint32_t c_mask = best_cycles(cold, [&]() { symbex_dense_input_u8_mask(u, y, &layer); });
        kern_report("symbex_u8_mask", mem, m, n, c_mask, ok && memcmp(y, ref, (size_t)m * sizeof(int32_t)) == 0);
        /* the call the exported MLP headers make (AND arithmetic): the HAL prepares the input once, then counts every
           row; on the S3 with PIE when rows are a multiple of 4 words and the buffers 16-byte aligned, else in C */
        if (rw % 4 == 0) {
            uint8_t* work = (uint8_t*)heap_caps_aligned_alloc(16, (size_t)rw * 32, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            /* the exporter aligns its weights to 16 bytes; a PSRAM block from the probe's allocator may not be: an
               aligned copy in the same memory, so the HAL takes its PIE path as it would in a real model */
            const uint32_t* wa = w;
            uint32_t* copy = nullptr;
            if (((uintptr_t)w & 15u) != 0 && strcmp(mem, "psram") == 0) {
                copy = (uint32_t*)heap_caps_aligned_alloc(16, (size_t)m * rw * 4, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
                if (copy) {
                    memcpy(copy, w, (size_t)m * rw * 4);
                    wa = copy;
                }
            }
            if (work) {
                Serial.printf("# symbex_and_input_u8 %s %dx%d: weights %s16-byte aligned%s\n", mem, m, n,
                              ((uintptr_t)wa & 15u) == 0 ? "" : "NOT ", copy ? " (aligned copy)" : "");
                const uint32_t c_and = best_cycles(cold, [&]() {
                    symbex_and_input_u8(u, (uint16_t)n, wa, (uint16_t)rw, (uint16_t)m, work, y); });
                kern_report("symbex_and_input_u8", mem, m, n, c_and, memcmp(y, ref, (size_t)m * sizeof(int32_t)) == 0);
                heap_caps_free(work);
            }
            heap_caps_free(copy);
        }
        for (int k = 4; k <= 8; k += 4) {
            const uint8_t top = (uint8_t)(0xFFu << (8 - k));
            for (int i = 0; i < n; i++) uq[i] = u[i] & top;
            u8_reference(w, rw, uq, ref, m, n);
            for (int r = 0; r < m; r++) thr[r] = ref[(r * 7 + 3) % m];   /* thresholds from the scores: both bit values */
            const uint32_t c = best_cycles(cold, [&]() { symbex_dense_input_u8_planes_bits(u, (uint8_t)k, planes, thr, bits, &layer); });
            kern_report(k == 4 ? "symbex_u8_planes_bits_k4" : "symbex_u8_planes_bits_k8", mem, m, n, c,
                        bits_match(bits, ref, thr, m));
        }
    }
    free(u);
    free(uq);
    free(ref);
    free(thr);
    free(planes);
    free(bits);
}

/* the mlp_fashion first layer (784 x 128, weights in SRAM): cross-checks lib_symbex's 0.270 ms there. 784 is not a
   multiple of 32, so only the lib_symbex kernels run (the probe's kernels need N % 32 == 0) */
static void symbex_mlp_fashion_layer() {
    const int m = 128, n = 784, rw = (n + 31) / 32;
    uint32_t* w = (uint32_t*)alloc_internal((size_t)m * rw * sizeof(uint32_t));
    const int stride = (rw + 3) / 4 * 4;                        /* 28 words: the mlp_fashion export's row stride */
    uint32_t* w4 = (uint32_t*)heap_caps_aligned_alloc(16, (size_t)m * stride * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    uint8_t* work = (uint8_t*)heap_caps_aligned_alloc(16, (size_t)stride * 32, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int32_t* y = (int32_t*)alloc_internal((size_t)m * sizeof(int32_t));
    KernelInputs in = {};                                       /* all null: inputs_free is safe on every path */
    if (!w || !y || !inputs_alloc(&in, n)) {
        Serial.println("# skip symbex 128x784 in sram: no SRAM");
    } else {
        fill_pattern(w, (size_t)m * rw, PAYLOAD_SEED);
        inputs_fill(&in);
        symbex_kernels_at("sram", w, m, n, &in, nullptr, nullptr, y);
        if (w4 && work) {
            memset(w4, 0, (size_t)m * stride * 4);
            for (int r = 0; r < m; r++) {
                memcpy(w4 + (size_t)r * stride, w + (size_t)r * rw, (size_t)rw * 4);
                w4[(size_t)r * stride + rw - 1] &= (1u << (n % 32)) - 1u;   /* padding bits 0, as exported */
            }
            uint8_t* u = (uint8_t*)alloc_internal(n);
            int32_t* ref = (int32_t*)alloc_internal((size_t)m * sizeof(int32_t));
            if (u && ref) {
                for (int i = 0; i < n; i++) u[i] = (uint8_t)((int)in.x8[i] + 128);
                u8_reference(w4, stride, u, ref, m, n);
                const uint32_t c_and = best_cycles(false, [&]() {
                    symbex_and_input_u8(u, (uint16_t)n, w4, (uint16_t)stride, (uint16_t)m, work, y); });
                kern_report("symbex_and_input_u8", "sram", m, n, c_and, memcmp(y, ref, (size_t)m * sizeof(int32_t)) == 0);
            }
            free(u);
            free(ref);
        }
    }
    inputs_free(&in);                                           /* also a partial allocation on the skip path */
    heap_caps_free(w4);
    heap_caps_free(work);
    free(w);
    free(y);
}

/* all kernels for one weight matrix at one location */
static void kernels_at(const char* mem, const uint32_t* w, int m, int n, KernelInputs* in, const int32_t* rowsum,
                       int32_t* y, int32_t* y_mask, int32_t* y_ref, int32_t* y_ref4, int32_t* y_ref1) {
    const bool cold = strcmp(mem, "sram") != 0;
    const size_t yb = (size_t)m * sizeof(int32_t);

    /* mask first: gemv_ref is validated by agreement with this independent algorithm (and on the host against numpy) */
    const uint32_t c_mask = best_cycles(cold, [&]() { gemv_mask(w, in->x8, y_mask, m, n); });
    const uint32_t c_ref = best_cycles(cold, [&]() { gemv_ref(w, in->x8, y_ref, m, n); });
    const bool ref_ok = memcmp(y_ref, y_mask, yb) == 0;
    kern_report("ref", mem, m, n, c_ref, ref_ok);
    kern_report("mask", mem, m, n, c_mask, ref_ok);

    /* lut4: build once per vector (always in SRAM), lookup over all rows */
    const uint32_t c_build = best_cycles(false, [&]() { lut4_build(in->x8, n, in->lut); });
    const uint32_t c_lookup = best_cycles(cold, [&]() { gemv_lut4(w, in->lut, y, m, n); });
    const bool lut_ok = memcmp(y, y_ref, yb) == 0;
    kern_report("lut4_build", mem, m, n, c_build, lut_ok);
    kern_report("lut4_lookup", mem, m, n, c_lookup, lut_ok);
    kern_report("lut4", mem, m, n, c_build + c_lookup, lut_ok);

    if (g_worker_ready) {
        memset(y, 0, yb);
        const uint32_t c_two = best_cycles(cold, [&]() { gemv_lut4_2core(w, in->lut, y, m, n); });
        kern_report("lut4_2core_lookup", mem, m, n, c_two, memcmp(y, y_ref, yb) == 0);
    }

    /* bit-serial 4-bit */
    gemv_ref(w, in->x4, y_ref4, m, n);
    int32_t sum_a = 0;
    const uint32_t c_pack = best_cycles(false, [&]() { sum_a = bs4_pack(in->a4, n, in->planes); });
    const uint32_t c_bs = best_cycles(cold, [&]() { gemv_bitserial4(w, in->planes, sum_a, rowsum, y, m, n); });
    const bool bs_ok = memcmp(y, y_ref4, yb) == 0;
    kern_report("bitserial4_pack", mem, m, n, c_pack, bs_ok);
    kern_report("bitserial4", mem, m, n, c_bs, bs_ok);

    /* xnor */
    gemv_ref(w, in->x1, y_ref1, m, n);
    xnor_pack(in->x1, n, in->xbits);
    const uint32_t c_xnor = best_cycles(cold, [&]() { gemv_xnor(w, in->xbits, y, m, n); });
    kern_report("xnor", mem, m, n, c_xnor, memcmp(y, y_ref1, yb) == 0);

    symbex_kernels_at(mem, w, m, n, in, rowsum, y_ref, y);
}

static void kernels_for_shape(int m, int n) {
    KernelInputs in;
    const size_t y_bytes = (size_t)m * sizeof(int32_t);
    int32_t* ybuf = (int32_t*)alloc_internal(5 * y_bytes);   /* y, y_mask, y_ref, y_ref4, y_ref1 */
    int32_t* rowsum = (int32_t*)alloc_internal(y_bytes);
    if (!inputs_alloc(&in, n) || !ybuf || !rowsum) {
        Serial.printf("# skip %dx%d: no SRAM for activations and outputs\n", m, n);
        inputs_free(&in);
        free(ybuf);
        free(rowsum);
        return;
    }
    inputs_fill(&in);
    const size_t words = (size_t)m * (size_t)(n / 32);
    const size_t bytes = words * 4;
    int32_t* y = ybuf;
    int32_t* y_mask = ybuf + m;
    int32_t* y_ref = ybuf + 2 * m;
    int32_t* y_ref4 = ybuf + 3 * m;
    int32_t* y_ref1 = ybuf + 4 * m;

    /* weights: the first `words` words of the payload pattern, at every location (identical bits everywhere) */
    uint32_t* w_sram = (uint32_t*)alloc_internal(bytes);
    if (w_sram) {
        fill_pattern(w_sram, words, PAYLOAD_SEED);
        onebit_rowsums(w_sram, m, n, rowsum);   /* offline in a real deployment */
        kernels_at("sram", w_sram, m, n, &in, rowsum, y, y_mask, y_ref, y_ref4, y_ref1);
    } else {
        Serial.printf("# skip %dx%d in sram: %u KB does not fit\n", m, n, (unsigned)(bytes / 1024));
    }
    /* flush_cache evicts by reading the flash payload: without it the PSRAM runs would be warm, not cold */
    uint32_t* w_psram = ESP.getPsramSize() > 0 && g_flash.ok ? (uint32_t*)alloc_psram(bytes) : nullptr;
    if (w_psram) {
        fill_pattern(w_psram, words, PAYLOAD_SEED);
        if (!w_sram) onebit_rowsums(w_psram, m, n, rowsum);
        kernels_at("psram", w_psram, m, n, &in, rowsum, y, y_mask, y_ref, y_ref4, y_ref1);
    } else if (ESP.getPsramSize() > 0 && !g_flash.ok) {
        Serial.printf("# skip %dx%d in psram: no flash probe partition to evict the cache with\n", m, n);
    } else {
        Serial.printf("# skip %dx%d in psram: not available\n", m, n);
    }
    if (g_flash.ok && bytes <= PROBE_FLASH_BYTES) {
        if (!w_sram && !w_psram) onebit_rowsums(g_flash.payload, m, n, rowsum);
        kernels_at("flash", g_flash.payload, m, n, &in, rowsum, y, y_mask, y_ref, y_ref4, y_ref1);
    } else {
        Serial.printf("# skip %dx%d in flash: not available\n", m, n);
    }
    free(w_sram);
    free(w_psram);
    inputs_free(&in);
    free(ybuf);
    free(rowsum);
}

static void kernel_benchmark() {
    Serial.println("# --- kernels: KERN,<kernel>,<mem>,<M>,<N>,<cycles per weight>,<us per call>,<OK|FAIL>");
    Serial.println("# weights are cache-cold for psram and flash (both need the flash probe partition, which evicts the "
                   "cache); *_build / *_pack are per input vector, amortised over M rows");
    static const int SHAPES[][2] = {{256, 1024}, {1024, 1024}, {2048, 2048}};
    for (const auto& shape : SHAPES) kernels_for_shape(shape[0], shape[1]);
    symbex_mlp_fashion_layer();
}

/* ============================================================================================================ */
/* energy schedule                                                                                              */
/* ============================================================================================================ */

/* Buffers and matrices shared by the phases, allocated once. */
struct PhaseState {
    uint32_t* sram_buf;
    size_t sram_bytes;
    uint32_t* psram_buf;
    size_t psram_bytes;
    KernelInputs in;
    uint32_t* w;       /* 1024 x 1024 weights, in PSRAM when present */
    int32_t* y;
    int32_t* rowsum;
    int m, n;
    uint8_t* sd_buf;
    bool kernels_ok;
};
static PhaseState P;

static const size_t PHASE_SRAM_BYTES = 64u * 1024u;
static const size_t PHASE_SD_BUF = 32u * 1024u;

static void phase_state_init() {
    P.sram_bytes = PHASE_SRAM_BYTES;
    P.sram_buf = (uint32_t*)alloc_internal(P.sram_bytes);
    if (P.sram_buf) fill_pattern(P.sram_buf, P.sram_bytes / 4, PAYLOAD_SEED);
    P.psram_bytes = PROBE_PSRAM_MEDIUM;
    P.psram_buf = ESP.getPsramSize() > 0 ? (uint32_t*)alloc_psram(P.psram_bytes) : nullptr;
    if (P.psram_buf) fill_pattern(P.psram_buf, P.psram_bytes / 4, PAYLOAD_SEED);
    P.m = 1024;
    P.n = 1024;
    P.w = (uint32_t*)(P.psram_buf ? alloc_psram((size_t)P.m * P.n / 8) : alloc_internal((size_t)P.m * P.n / 8));
    P.y = (int32_t*)alloc_internal(P.m * sizeof(int32_t));
    P.rowsum = (int32_t*)alloc_internal(P.m * sizeof(int32_t));
    P.sd_buf = (uint8_t*)alloc_internal(PHASE_SD_BUF);
    P.kernels_ok = P.w && P.y && P.rowsum && inputs_alloc(&P.in, P.n);
    if (P.kernels_ok) {
        fill_pattern(P.w, (size_t)P.m * P.n / 32, PAYLOAD_SEED);
        inputs_fill(&P.in);
        lut4_build(P.in.x8, P.n, P.in.lut);
    }
}

typedef uint32_t (*PhaseStep)();   /* one iteration of the steady workload; returns iterations done (1) */

static uint32_t step_idle() {
    delay(10);
    return 1;
}
static uint32_t step_sram() {
    g_sink = g_sink + read_sum(P.sram_buf, P.sram_bytes / 4);
    return 1;
}
static uint32_t step_psram() {
    g_sink = g_sink + read_sum(P.psram_buf, P.psram_bytes / 4);
    return 1;
}
static uint32_t step_flash() {
    g_sink = g_sink + read_sum(g_flash.payload, PROBE_FLASH_BYTES / 4);
    return 1;
}

static File g_phase_file;
static uint32_t step_sd() {
    if (!g_phase_file) g_phase_file = SD_MMC.open(g_sd_path, FILE_READ);
    const int n = g_phase_file ? g_phase_file.read(P.sd_buf, PHASE_SD_BUF) : 0;
    if (n <= 0) g_phase_file.close();   /* end of file: reopen on the next step */
    else g_sink = g_sink + P.sd_buf[0];
    return 1;
}
static uint32_t step_lut4() {
    lut4_build(P.in.x8, P.n, P.in.lut);
    gemv_lut4(P.w, P.in.lut, P.y, P.m, P.n);
    return 1;
}
static uint32_t step_lut4_2core() {
    lut4_build(P.in.x8, P.n, P.in.lut);
    gemv_lut4_2core(P.w, P.in.lut, P.y, P.m, P.n);
    return 1;
}
static uint32_t step_mask() {
    gemv_mask(P.w, P.in.x8, P.y, P.m, P.n);
    return 1;
}
#if !defined(PROBE_CAMERA_NONE)
static uint32_t step_camera() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) {
        g_sink = g_sink + fb->len;
        esp_camera_fb_return(fb);
    }
    return 1;
}
#endif

struct Phase {
    const char* name;
    bool active;            /* marker GPIO high */
    bool available;
    PhaseStep step;
    bool (*prepare)();      /* before the PHASE start line (not metered), may be null */
    void (*cleanup)();      /* after the PHASE end line, may be null */
};

static bool prepare_sd() {
    g_phase_file = File();
    return true;
}
static void cleanup_sd() {
    if (g_phase_file) g_phase_file.close();
}
#if !defined(PROBE_CAMERA_NONE)
static bool prepare_camera() { return camera_start() == ESP_OK; }
static void cleanup_camera() { camera_stop(); }
#endif

static void marker_set(bool high) {
#if PROBE_MARKER_GPIO >= 0
    digitalWrite(PROBE_MARKER_GPIO, high ? HIGH : LOW);
#else
    (void)high;
#endif
}

static void run_phase(int number, const Phase& ph) {
    if (!ph.available) {
        Serial.printf("# phase %d %s skipped (not available on this board)\n", number, ph.name);
        return;
    }
    if (ph.prepare && !ph.prepare()) {
        Serial.printf("# phase %d %s skipped (prepare failed)\n", number, ph.name);
        return;
    }
    Serial.printf("PHASE,%d,%s,start\n", number, ph.name);
    marker_set(ph.active);
    uint32_t iterations = 0;
    const int64_t t_start = micros64();
    const int64_t t_end = t_start + (int64_t)PROBE_PHASE_SECONDS * 1000000;
    int64_t next_print = t_start + 5000000;
    while (micros64() < t_end) {
        iterations += ph.step();
        const int64_t now = micros64();
        if (now >= next_print) {
            Serial.printf("# phase %d %s: %d s left, read the meter now\n", number, ph.name,
                          (int)((t_end - now) / 1000000));
            next_print += 5000000;
        }
    }
    marker_set(false);
    Serial.printf("PHASE,%d,%s,end,%u\n", number, ph.name, (unsigned)iterations);
    if (ph.cleanup) ph.cleanup();
}

static void energy_schedule() {
    const bool have_psram = P.psram_buf != nullptr;
    const bool have_kernels = P.kernels_ok;
    const bool two_cores = g_worker_ready && have_kernels;
    const Phase phases[] = {
        {"idle", false, true, step_idle, nullptr, nullptr},
        {"sram-read", true, P.sram_buf != nullptr, step_sram, nullptr, nullptr},
        {"psram-read", true, have_psram, step_psram, nullptr, nullptr},
        {"flash-read", true, g_flash.ok, step_flash, nullptr, nullptr},
        {"sd-read", true, g_sd_ok, step_sd, prepare_sd, cleanup_sd},
        {"lut4-gemv", true, have_kernels, step_lut4, nullptr, nullptr},
        {"lut4-2core", true, two_cores, step_lut4_2core, nullptr, nullptr},
        {"mask-gemv", true, have_kernels, step_mask, nullptr, nullptr},
#if !defined(PROBE_CAMERA_NONE)
        {"camera", true, g_camera_ok, step_camera, prepare_camera, cleanup_camera},
#endif
    };
    int cycle = 0;
    for (;;) {
        cycle++;
        Serial.printf("# --- energy schedule, round %d: %d s per phase. Write down watts per phase (total and above idle).\n",
                      cycle, PROBE_PHASE_SECONDS);
        Serial.printf("# weights for the gemv phases: %dx%d from %s\n", P.m, P.n, have_psram ? "PSRAM" : "SRAM");
        int number = 0;
        for (const Phase& ph : phases) run_phase(++number, ph);
        Serial.println("# --- round complete");
    }
}

/* ============================================================================================================ */

void setup() {
    Serial.begin(115200);
    delay(START_DELAY_MS);   /* time to open the monitor / start the log */
    setCpuFrequencyMhz(PROBE_CPU_MHZ);
    disableCore0WDT();       /* the busy loops and the second-core worker would starve the idle tasks */
    disableCore1WDT();
    disableLoopWDT();
    btStop();                /* no radio is ever started; WiFi stays off */
#if PROBE_MARKER_GPIO >= 0
    pinMode(PROBE_MARKER_GPIO, OUTPUT);
    digitalWrite(PROBE_MARKER_GPIO, LOW);
#endif
    Serial.println();
    Serial.println("# SYMBEX edge-LLM probe: nothing below is an estimate; every number is measured on this chip");
    print_info();
    worker_start();
    memory_bandwidth();
    kernel_benchmark();
    phase_state_init();
    energy_schedule();   /* never returns */
}

void loop() {}
