/*
 * fine_profile.c -- firmware of tools/fine_profile.py: where the time goes INSIDE lib_symbex, function by function.
 * Like model_test.c (the same serial protocol for the images), but the library is built with -DSYMBEX_FINE_PROFILE:
 * every part of the binary layers adds its CPU cycles and calls (layers/conv/symbex_bool.h, SYMBEX_PF_*), and the
 * time of every step of the network is summed over all the images.
 *
 * Protocol: the board prints "PROFILE ready | <hal> | ...", STEP lines; for every 0xA5 + image it answers
 * "R:class,total_us"; 0x77 asks for the totals: PF_IMG, PF_COST, PF:<part>|cycles|calls, ST:<step>|us, PF_END.
 * chip/core.py writes BAUD_RATE before this file.
 */
#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "hal/symbex_hal.h"

/* without -DSYMBEX_FINE_PROFILE (fine_profile.py --plain): the normal library, only the time of every step */

#define SYMBEX_PROFILE 1
static int64_t marks[32];
#define SYMBEX_MARK(k) (marks[(k)] = esp_timer_get_time())
#include "model_symbex.h"
#include "layers/conv/symbex_bool.h"
#ifndef SYMBEX_FINE_PROFILE
#define SYMBEX_PF_N 0
static void symbex_pf_reset(void) {}
#endif

static int64_t step_us[32];
static uint32_t n_images;

static void read_exact(uint8_t* d, size_t n) {
    size_t r = 0;
    while (r < n) {
        int k = uart_read_bytes(UART_NUM_0, d + r, n - r, portMAX_DELAY);
        if (k > 0) r += (size_t)k;
    }
}

void app_main(void) {
    esp_log_level_set("*", ESP_LOG_WARN);
    uart_driver_install(UART_NUM_0, 8192, 0, 0, NULL, 0);
    uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(100));
    uart_set_baudrate(UART_NUM_0, BAUD_RATE);
    vTaskDelay(pdMS_TO_TICKS(50));
    static uint8_t x[SYMBEX_MLP_N_INPUTS] __attribute__((aligned(16)));
    symbex_mlp_infer(x);                                /* first call (tables, caches): not counted */
    symbex_pf_reset();
    printf("\nPROFILE ready | %s | %d inputs\n", symbex_hal_name(), SYMBEX_MLP_N_INPUTS);
    printf("STEPS:%d\n", SYMBEX_MLP_N_STEPS);
    for (int i = 0; i < SYMBEX_MLP_N_STEPS; i++) printf("STEP:%d|%s\n", i, symbex_mlp_steps[i]);
    fflush(stdout);
    while (1) {
        uint8_t mark = 0;
        read_exact(&mark, 1);
        if (mark == 0x77) {                             /* the totals */
#ifdef SYMBEX_FINE_PROFILE
            printf("PF_IMG:%u\nPF_COST:%u\n", (unsigned)n_images, (unsigned)symbex_pf_counter_cost);
            for (int k = 0; k < SYMBEX_PF_N; k++)
                printf("PF:%s|%llu|%lu\n", symbex_pf_names[k], (unsigned long long)symbex_pf_cycles[k],
                       (unsigned long)symbex_pf_calls[k]);
#else
            printf("PF_IMG:%u\nPF_COST:0\n", (unsigned)n_images);
#endif
            for (int i = 0; i < SYMBEX_MLP_N_STEPS; i++) printf("ST:%d|%lld\n", i, (long long)step_us[i]);
            printf("PF_END\n");
            fflush(stdout);
            continue;
        }
        if (mark != 0xA5) continue;
        read_exact(x, sizeof x);
        int c = symbex_mlp_infer(x);
        for (int i = 0; i < SYMBEX_MLP_N_STEPS; i++) step_us[i] += marks[i + 1] - marks[i];
        n_images++;
        printf("R:%d,%lld\n", c, (long long)(marks[SYMBEX_MLP_N_STEPS] - marks[0]));
        fflush(stdout);
    }
}
