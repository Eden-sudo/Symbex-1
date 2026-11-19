"""
bit_attn_bench.py -- attention in bits on the chip: what do sign scores and bucketed values cost per head?

../tiny_llm/README.md "Attention in bits" trains a model whose query and key are signs (64 bits per head): a score
is then h = popcount(q XOR k) (q.k = 64 - 2h), a softmax weight one entry of a 65-entry table, w = TAB[h - hmin], and
the 2-bit values can be summed per h first (buckets) and multiplied by the table only 65 times. This bench times one
head of one decoding step (256 cached positions x 64), plain C on every board, internal RAM, against the int8
attention of tinylm.c:
    i8_scores       int8 query x 256 int8 keys (int32 dot products)
    i8_values       256 int32 weights (< 2^15) x int8 values
    bit_scores      XOR + __builtin_popcount over 2 words per key, hmin, w = TAB[h - hmin]
    bit_scores_swar the same with a SWAR popcount (no popcount instruction on Xtensa)
    bit_values      2-bit values (16 bytes per position, levels -3 -1 1 3), unpacked by a 256-entry table, x w
    bit_values_bkt  the same by buckets: 2-bit codes summed per h (int16), then sum over the used h of TAB x bucket
Both value forms also give z = sum of w (the softmax denominator). Not timed: tinylm's int8 softmax (exp table +
division; measured in the whole model: 2.0 + 1.6 ms per token on the S3 = ~75 us per head) and the division by z.
Random bits, codes and weights (seed 2028); every result is checked against numpy (h, w, the 64 sums, z), otherwise
its time is not trusted. Cache per position and head: int8 K + V 128 bytes, bits K + 2-bit V 24 bytes.

    python bit_attn_bench.py --board s3 --port /dev/ttyACM0
    python bit_attn_bench.py --board esp32 --port /dev/ttyUSB0
    python bit_attn_bench.py --board pico2_arm --port /dev/ttyACM1        # Cortex-M33
    python bit_attn_bench.py --board pico2_riscv --port /dev/ttyACM1      # Hazard3

Times: best of 7, per head (cycles at 240 MHz on the ESP32 boards; on the Pico 2 micros() over 16 calls, ns).
Results: results/bit_attn_bench_<board>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import kv_attn_bench as kv
from kv_attn_bench import HERE, RESET, TICKS, c_array, flash, open_port

HD, T, TAU = 64, 256, 0.25            # TAU: the table's temperature (any value; the trained one is learned per head)

FIRMWARE = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#if defined(ESP_PLATFORM)
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#define NOW() ((uint32_t)esp_cpu_get_cycle_count())
#define OUT printf
#define INNER 1
#define SCALE 1
#else
#include <Arduino.h>
#define NOW() ((uint32_t)micros())
#define OUT Serial.printf
#define INNER 16
#define SCALE 1000
#endif
#include "bench_data.h"
#define REPS 7

static int8_t q8[HD], k8[T * HD], v8[T * HD];
static int32_t pw[T], d[T], acc[HD];
static uint32_t qb[2], kb[T * 2];
static uint8_t v2[T * 16];
static uint8_t hh[T];
static uint16_t w[T];
static int hmin;
static uint32_t z;
static int16_t bucket[65][HD];
static uint16_t cnt[65];
static uint32_t lut2[256];

static void i8_scores(void) {
    const int8_t* k = k8;
    for (int j = 0; j < T; j++, k += HD) {
        int32_t s = 0;
        for (int i = 0; i < HD; i++) s += q8[i] * k[i];
        d[j] = s;
    }
}
static void i8_values(void) {
    for (int i = 0; i < HD; i++) acc[i] = 0;
    const int8_t* v = v8;
    for (int j = 0; j < T; j++, v += HD) {
        const int32_t a = pw[j];
        for (int i = 0; i < HD; i++) acc[i] += a * v[i];
    }
}
static inline uint32_t popc_swar(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}
static void bit_weights(void) {
    for (int j = 0; j < T; j++) w[j] = TAB[hh[j] - hmin];
}
static void bit_scores(void) {
    int m = 64;
    for (int j = 0; j < T; j++) {
        const int h = __builtin_popcount(qb[0] ^ kb[2 * j]) + __builtin_popcount(qb[1] ^ kb[2 * j + 1]);
        hh[j] = (uint8_t)h;
        if (h < m) m = h;
    }
    hmin = m;
    bit_weights();
}
static void bit_scores_swar(void) {
    int m = 64;
    for (int j = 0; j < T; j++) {
        const int h = (int)(popc_swar(qb[0] ^ kb[2 * j]) + popc_swar(qb[1] ^ kb[2 * j + 1]));
        hh[j] = (uint8_t)h;
        if (h < m) m = h;
    }
    hmin = m;
    bit_weights();
}
static void bit_values(void) {
    int32_t a[HD];
    uint32_t row32[HD / 4];
    const int8_t* row = (const int8_t*)row32;
    uint32_t s = 0;
    for (int i = 0; i < HD; i++) a[i] = 0;
    for (int j = 0; j < T; j++) {
        const uint8_t* p = v2 + 16 * j;
        for (int b = 0; b < 16; b++) row32[b] = lut2[p[b]];
        const int32_t x = w[j];
        s += (uint32_t)x;
        for (int i = 0; i < HD; i++) a[i] += x * row[i];
    }
    for (int i = 0; i < HD; i++) acc[i] = a[i];
    z = s;
}
static void bit_values_bkt(void) {
    uint32_t row32[HD / 4];
    const int8_t* row = (const int8_t*)row32;
    for (int j = 0; j < T; j++) {
        const int h = hh[j];
        int16_t* bk = bucket[h];
        if (!cnt[h]++) for (int i = 0; i < HD; i++) bk[i] = 0;
        const uint8_t* p = v2 + 16 * j;
        for (int b = 0; b < 16; b++) row32[b] = lut2[p[b]];
        for (int i = 0; i < HD; i++) bk[i] += row[i];
    }
    int32_t a[HD];
    uint32_t s = 0;
    for (int i = 0; i < HD; i++) a[i] = 0;
    for (int h = hmin; h <= 64; h++) {
        if (!cnt[h]) continue;
        const int32_t x = TAB[h - hmin];
        s += (uint32_t)x * cnt[h];
        cnt[h] = 0;
        const int16_t* bk = bucket[h];
        for (int i = 0; i < HD; i++) a[i] += x * bk[i];
    }
    for (int i = 0; i < HD; i++) acc[i] = a[i];
    z = s;
}

enum { OP_I8_SCORES, OP_I8_VALUES, OP_BIT_SCORES, OP_BIT_SCORES_SWAR, OP_BIT_VALUES, OP_BIT_VALUES_BKT, N_OPS };
static const char* OP_NAME[] = {"i8_scores", "i8_values", "bit_scores", "bit_scores_swar", "bit_values",
                                "bit_values_bkt"};
static void (*const OP_FN[])(void) = {i8_scores, i8_values, bit_scores, bit_scores_swar, bit_values, bit_values_bkt};

static int same(int op) {
    switch (op) {
    case OP_I8_SCORES: for (int j = 0; j < T; j++) if (d[j] != EXP_D8[j]) return 0; return 1;
    case OP_I8_VALUES: for (int i = 0; i < HD; i++) if (acc[i] != EXP_ACC8[i]) return 0; return 1;
    case OP_BIT_SCORES: case OP_BIT_SCORES_SWAR:
        if (hmin != EXP_HMIN) return 0;
        for (int j = 0; j < T; j++) if (hh[j] != EXP_H[j] || w[j] != EXP_W[j]) return 0;
        return 1;
    default:
        if (z != EXP_Z) return 0;
        for (int i = 0; i < HD; i++) if (acc[i] != EXP_ACCB[i]) return 0;
        return 1;
    }
}

static void bench_once(void) {
    OUT("BIT ATTN BENCH ready (%s)\n", BOARD_NAME);
    for (int op = 0; op < N_OPS; op++) {
        uint32_t b = 0xFFFFFFFFu;
        memset(d, 0, sizeof d); memset(acc, 0, sizeof acc); z = 0;
        if (op == OP_BIT_SCORES || op == OP_BIT_SCORES_SWAR) { memset(hh, 0, sizeof hh); memset(w, 0, sizeof w); hmin = -1; }
        for (int r = 0; r < REPS; r++) {
            const uint32_t t0 = NOW();
            for (int k = 0; k < INNER; k++) OP_FN[op]();
            const uint32_t t = (uint32_t)((uint64_t)(NOW() - t0) * SCALE / INNER);
            if (t < b) b = t;
        }
        OUT("R:%s|%lu|%d\n", OP_NAME[op], (unsigned long)b, same(op));
    }
    OUT("DONE\n");
}

static void setup_data(void) {
    for (int b = 0; b < 256; b++) {
        int8_t v[4];
        for (int k = 0; k < 4; k++) v[k] = (int8_t)(2 * ((b >> (2 * k)) & 3) - 3);
        memcpy(&lut2[b], v, 4);
    }
    memcpy(q8, Q8, sizeof q8); memcpy(k8, K8, sizeof k8); memcpy(v8, V8, sizeof v8); memcpy(pw, PW, sizeof pw);
    memcpy(qb, QB, sizeof qb); memcpy(kb, KB, sizeof kb); memcpy(v2, V2, sizeof v2);
    bit_scores();                                    /* the values ops read h and w */
}

#if defined(ESP_PLATFORM)
void app_main(void) {
    setup_data();
    for (;;) { bench_once(); vTaskDelay(pdMS_TO_TICKS(2000)); }
}
#else
void setup() { Serial.begin(115200); setup_data(); }
void loop() { bench_once(); delay(2000); }
#endif
"""


def make_data():
    """the inputs and numpy results (integers, the firmware's order of operations does not matter)"""
    rng = np.random.default_rng(2028)
    q8, k8, v8 = rng.integers(-127, 128, HD), rng.integers(-127, 128, (T, HD)), rng.integers(-127, 128, (T, HD))
    pw = rng.integers(0, 1 << 15, T)
    qbits, kbits = rng.integers(0, 2, HD), rng.integers(0, 2, (T, HD))
    codes = rng.choice(np.array([-3, -1, 1, 3]), (T, HD))
    tab = np.round(32768 * 2.0 ** (-TAU * np.arange(65))).astype(np.int64)
    h = (qbits[None, :] != kbits).sum(1)
    w = tab[h - h.min()]
    words = lambda bits: (bits.reshape(-1, 32).astype(np.uint64) << np.arange(32, dtype=np.uint64)).sum(1)
    return dict(q8=q8, k8=k8, v8=v8, pw=pw, qb=words(qbits), kb=words(kbits.ravel()), v2=kv.pack(codes, 2),
                tab=tab, h=h, hmin=int(h.min()), w=w, exp_d8=k8 @ q8, exp_acc8=pw @ v8, exp_accb=w @ codes,
                exp_z=int(w.sum()))


def write_data(path, board):
    x = make_data()
    with open(path, "w") as f:
        f.write(f'#pragma once\n#define BOARD_NAME "{board}"\n#define HD {HD}\n#define T {T}\n'
                f"#define EXP_HMIN {x['hmin']}\n#define EXP_Z {x['exp_z']}u\n")
        for ctype, name, key, suf in (("int8_t", "Q8", "q8", ""), ("int8_t", "K8", "k8", ""), ("int8_t", "V8", "v8", ""),
                                      ("int32_t", "PW", "pw", ""), ("uint32_t", "QB", "qb", "u"),
                                      ("uint32_t", "KB", "kb", "u"), ("uint8_t", "V2", "v2", ""),
                                      ("uint16_t", "TAB", "tab", ""), ("uint8_t", "EXP_H", "h", ""),
                                      ("uint16_t", "EXP_W", "w", ""), ("int32_t", "EXP_D8", "exp_d8", ""),
                                      ("int32_t", "EXP_ACC8", "exp_acc8", ""), ("int32_t", "EXP_ACCB", "exp_accb", "")):
            f.write(c_array(ctype, name, x[key], suffix=suf))


def read_results(port, board):
    s = open_port(port, 115200, reset=RESET[board])
    rows, ready, t0 = [], False, time.time()
    while time.time() - t0 < 120:
        line = s.readline().decode(errors="replace").strip()
        if "BIT ATTN BENCH ready" in line:
            if f"({board})" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected {board}")
            ready, rows = True, []
        elif ready and line.startswith("R:"):
            op, t, ok = line[2:].split("|")
            rows.append((op, int(t), ok == "1"))
        elif ready and line == "DONE":
            s.close()
            return rows
        elif not ready and board == "s3" and time.time() - t0 > 20 and not line:
            s.write(b"\x55")
    s.close()
    sys.exit("[X] no complete results from the board in 120 s (a run that lost lines is not saved)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=sorted(TICKS), default="s3")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    project = os.path.join(HERE, "runs", f"bit_attn_bench_{a.board}")
    if not a.no_upload:
        kv.write_project(project, a.port, a.board, FIRMWARE, write_data, "bit_attn_bench")
        flash(project, verbose=False, what=a.board)
    rows = read_results(a.port, a.board)
    unit, per_us = TICKS[a.board]
    print(f"\n{a.board}, one head, {T} positions x {HD} (best of 7, per head)")
    print(f"  {'op':<17}{unit:>10}{'us':>9}  equal")
    for op, t, ok in rows:
        print(f"  {op:<17}{t:>10}{t / per_us:>9.1f}  {'OK' if ok else 'NO'}")
    if not all(r[2] for r in rows):
        sys.exit("[X] a run differs from numpy: the times are not trusted, nothing saved")
    path = os.path.join(HERE, "results", f"bit_attn_bench_{a.board}.json")
    with open(path, "w") as f:
        json.dump(dict(board=a.board, head_dim=HD, positions=T, table_tau=TAU, reps="best of 7", unit=unit,
                       date=time.strftime("%Y-%m-%d"), bytes_per_position={"int8_kv": 2 * HD, "bits_k_2bit_v": 24},
                       rows=[dict(op=op, **({unit: t}), us=round(t / per_us, 2), equal_numpy=ok) for op, t, ok in rows]),
                  f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
