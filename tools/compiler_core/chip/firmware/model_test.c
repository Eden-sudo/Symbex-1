/*
 * model_test.c -- firmware of the model test (chip/measure.py): runs the exported model (model_symbex.h)
 * on every image received over serial and prints the class, the total time and the time of every layer.
 * With SYMBEX_WITH_INT8, it also runs the INT8 reference (model_int8.h, Espressif's ESP-NN) on the same image.
 * chip/core.py writes BAUD_RATE (and SYMBEX_WITH_INT8 / SYMBEX_WITH_ESP_NN) before this file.
 *
 * Protocol: the board prints "SYMBEX ready | <hal> | ...", STEP / RAM (/ INT8) lines, then for every
 * 0xA5 + image it answers "R:class,total_us,us_layer0,..." (and "Q:..." for INT8); 0x5A asks for "STACK:bytes".
 *
 * With SYMBEX_USB_SERIAL_JTAG (optional, ESP32-S3 / C3 native USB port), the same protocol runs over the chip's
 * USB-Serial-JTAG instead of UART0: the console must be USB-Serial-JTAG (sdkconfig), and the board waits for the
 * host's 0x55 before its ready lines (a reset re-enumerates the USB port, so the host cannot catch them at boot).
 * In both modes, 0x55 in the main loop prints the ready lines again.
 *
 * The input image is 16-byte aligned: the first layer of the library reads 4 inputs per 32-bit load (SWAR) only
 * when the input is 4-byte aligned (otherwise it falls back to one input at a time, ~2x slower on the ESP32/S3).
 */
#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "hal/symbex_hal.h"
#ifdef SYMBEX_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#endif

#define SYMBEX_PROFILE 1
static int64_t marks[32];
#define SYMBEX_MARK(k) (marks[(k)] = esp_timer_get_time())
#include "model_symbex.h"
#ifdef SYMBEX_WITH_INT8
static int64_t marks8[32];
#define INT8_MARK(k) (marks8[(k)] = esp_timer_get_time())
#include "model_int8.h"
#if INT8_N_INPUTS != SYMBEX_MLP_N_INPUTS
#error "model_int8.h and model_symbex.h do not have the same number of inputs: export them together"
#endif
#endif

static void read_exact(uint8_t* d, size_t n) {
    size_t r = 0;
    while (r < n) {
#ifdef SYMBEX_USB_SERIAL_JTAG
        int k = usb_serial_jtag_read_bytes(d + r, n - r, portMAX_DELAY);
#else
        int k = uart_read_bytes(UART_NUM_0, d + r, n - r, portMAX_DELAY);
#endif
        if (k > 0) r += (size_t)k;
    }
}

static size_t ram_table;                             /* bytes the HAL allocated on the first call (its table) */

static void print_ready(void) {
    printf("\nSYMBEX ready | %s | %d inputs, %d classes\n", symbex_hal_name(),
           SYMBEX_MLP_N_INPUTS, SYMBEX_MLP_N_CLASSES);
    printf("STEPS:%d\n", SYMBEX_MLP_N_STEPS);
    for (int i = 0; i < SYMBEX_MLP_N_STEPS; i++) printf("STEP:%d|%s\n", i, symbex_mlp_steps[i]);
    printf("RAM:%u,%u,%u,%u\n", (unsigned)SYMBEX_MLP_BYTES_BUFFERS, (unsigned)SYMBEX_MLP_N_INPUTS,
           (unsigned)ram_table, (unsigned)CONFIG_ESP_MAIN_TASK_STACK_SIZE);
#ifdef SYMBEX_WITH_INT8
    printf("INT8:%d,%u\n", INT8_N_LAYERS, (unsigned)INT8_BYTES_BUFFERS);
#endif
    fflush(stdout);
}

void app_main(void) {
    esp_log_level_set("*", ESP_LOG_WARN);
#ifdef SYMBEX_USB_SERIAL_JTAG
    usb_serial_jtag_driver_config_t usb = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb.rx_buffer_size = 8192;
    usb.tx_buffer_size = 4096;
    usb_serial_jtag_driver_install(&usb);
    usb_serial_jtag_vfs_use_driver();                /* printf goes through the driver, not the ROM routine */
#else
    uart_driver_install(UART_NUM_0, 8192, 0, 0, NULL, 0);
    uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(100));
    uart_set_baudrate(UART_NUM_0, BAUD_RATE);
#endif
    vTaskDelay(pdMS_TO_TICKS(50));
    static uint8_t x[SYMBEX_MLP_N_INPUTS] __attribute__((aligned(16)));    /* aligned: SWAR first layer */
    size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    symbex_mlp_infer(x);                              /* first call: the HAL builds its table here, not timed */
    size_t heap_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    ram_table = heap_before - heap_after;
#ifdef SYMBEX_WITH_INT8
    symbex_int8_infer(x);
#endif
#ifdef SYMBEX_USB_SERIAL_JTAG
    uint8_t hello = 0;
    while (hello != 0x55) read_exact(&hello, 1);     /* the host is listening: only now the ready lines */
#endif
    print_ready();
    while (1) {
        uint8_t mark = 0;
        read_exact(&mark, 1);
        if (mark == 0x55) {                            /* the host asks for the ready lines again */
            print_ready();
            continue;
        }
        if (mark == 0x5A) {                            /* memory request: free stack (bytes) of this task */
            printf("STACK:%u\n", (unsigned)uxTaskGetStackHighWaterMark(NULL));
            fflush(stdout);
            continue;
        }
        if (mark != 0xA5) continue;
        read_exact(x, sizeof x);
        int c = symbex_mlp_infer(x);
        printf("R:%d,%lld", c, (long long)(marks[SYMBEX_MLP_N_STEPS] - marks[0]));
        for (int i = 0; i < SYMBEX_MLP_N_STEPS; i++) printf(",%lld", (long long)(marks[i + 1] - marks[i]));
        printf("\n");
#ifdef SYMBEX_WITH_INT8
        int c8 = symbex_int8_infer(x);              /* the INT8 reference, on the SAME image */
        printf("Q:%d,%lld", c8, (long long)(marks8[INT8_N_LAYERS] - marks8[0]));
        for (int i = 0; i < INT8_N_LAYERS; i++) printf(",%lld", (long long)(marks8[i + 1] - marks8[i]));
        printf("\n");
#endif
        fflush(stdout);
    }
}
