"""
kv260k_bench.py -- the attention of the 260K story model, one method at a time: what does each cache trick cost?

../tiny_llm/README.md steps 7-9b train the 260K (8 query heads of width 8, 4 key / value heads, context 512) with
1-bit keys, an int8 window of the last 32 positions, a learned temperature, unit keys (no per-token scale), a
centered key (the mean added back as a score term) and 2-bit values. This bench times one query head of one decoding
step at the last position (512 cached positions, the worst case), every stage separately, so each method's own cost
shows (plain C on every board, internal RAM):
    s_i8          int8 query x 512 int8 keys, times the per-token int32 scale (tinylm.c, the base)
    s_k1          1-bit keys: a 256-entry table of the query's subset sums, one lookup per position, x scale
    s_k1_win      s_k1 before the window, the last 32 positions int8 (k1tr)
    s_k1u_win     unit keys (no per-token scale: one constant) + window
    s_k1c         s_k1 + the center term q . RoPE_j(mean) (4 rotation pairs per position)
    s_k1uc_win    unit keys + center + window (k1uctr)
    sm_base       tinylm.c's softmax: distance x qm >> 20, / att_div, exp table (no temperature)
    sm_temp       the tempered softmax (two-step division, x tq >> 26; tinylm.c --attn-temp)
    v_i8          the int8 value loop (16-bit weight halves, int32 sums)
    v2_win        2-bit values (tinylm.c's layout and decoding) before the window, int8 in it
The center term has NO integer model in intsim yet (README step 9): its form here (A_p cos + B_p sin per pair, one
rounding, x a constant) is one plausible way, timed to see its order of cost, not the exact model. The unit-key
constant and the center constant are runtime values (a model's would be). Random data (seed 2029); every result is
checked against numpy (512 scores, 512 weights + their sum, 8 sums), otherwise its time is not trusted.

    python kv260k_bench.py --board esp32 --port /dev/ttyUSB0
    python kv260k_bench.py --board pico2_arm --port /dev/ttyACM1        # Cortex-M33
    python kv260k_bench.py --board pico2_riscv --port /dev/ttyACM1      # Hazard3
    python kv260k_bench.py --board s3 --port /dev/ttyACM0

Times: best of 7, per query head (cycles at 240 MHz on the ESP32 boards; on the Pico 2 micros() over 16 calls, ns).
A whole token's attention = 5 layers x 8 heads x (scores + softmax + values); the table prints it from the head times.
Results: results/kv260k_bench_<board>.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import kv_attn_bench as kv
from kv_attn_bench import HERE, RESET, TICKS, c_array, flash, open_port

HD, T, WIN, EXP_N = 8, 512, 32, 2048
HEADS_PER_TOKEN = 5 * 8

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
static int8_t q[HD];
static int8_t k8[T * HD], k8w[WIN * HD], v8[T * HD], v8w[WIN * HD];
static uint8_t kb[T], vb[HD * (T / 4)];
static int32_t km[T], km1[T], km8w[WIN], vm[T], vm2[T], vm8w[WIN];
static int16_t cs[T][HD / 2], sn[T][HD / 2], mu[HD];
static uint16_t expt[EXP_N];
static int64_t u[T];
static uint16_t p[T];
static int64_t P;
static int64_t acc[HD];
static int8_t vd[T * HD];
static int32_t pl[T], ph[T], pvj[T];
static int16_t lut[256];
volatile int32_t ku_rt, kc_rt, qm_rt, ad_rt, tq_rt;     /* model constants: runtime values */

static int32_t make_lut(void) {      /* the subset sums of the 8 query values; returns sum q */
    lut[0] = 0;
    for (int k = 0; k < 8; k++)
        for (int b = 0; b < (1 << k); b++) lut[(1 << k) + b] = (int16_t)(lut[b] + q[k]);
    int32_t sq = 0;
    for (int i = 0; i < HD; i++) sq += q[i];
    return sq;
}
static void s_i8(void) {
    const int8_t* kj = k8;
    for (int j = 0; j < T; j++, kj += HD) {
        int32_t d = 0;
        for (int i = 0; i < HD; i++) d += q[i] * kj[i];
        u[j] = (int64_t)d * km[j];
    }
}
static void win_tail(void) {         /* the last WIN positions int8 */
    const int8_t* kj = k8w;
    for (int r = 0; r < WIN; r++, kj += HD) {
        int32_t d = 0;
        for (int i = 0; i < HD; i++) d += q[i] * kj[i];
        u[T - WIN + r] = (int64_t)d * km8w[r];
    }
}
static void k1_rng(int n) {
    const int32_t sq = make_lut();
    for (int j = 0; j < n; j++) u[j] = (int64_t)(2 * lut[kb[j]] - sq) * km1[j];
}
static void s_k1(void) { k1_rng(T); }
static void s_k1_win(void) { k1_rng(T - WIN); win_tail(); }
static void s_k1u_win(void) {
    const int32_t sq = make_lut(), ku = ku_rt;
    for (int j = 0; j < T - WIN; j++) u[j] = (int64_t)(2 * lut[kb[j]] - sq) * ku;
    win_tail();
}
/* center term: q . RoPE_j(mu) = sum_p cos_jp (q1 m1 + q2 m2) + sin_jp (q2 m1 - q1 m2) (mu int8, the pair sums >> 4),
   one rounding, x kc */
static int32_t A[HD / 2], B[HD / 2];
static void center_ab(void) {
    for (int p2 = 0; p2 < HD / 2; p2++) {
        const int32_t q1 = q[2 * p2], q2 = q[2 * p2 + 1], m1 = mu[2 * p2], m2 = mu[2 * p2 + 1];
        A[p2] = (q1 * m1 + q2 * m2) >> 4; B[p2] = (q2 * m1 - q1 * m2) >> 4;   /* 12 bits: int32 sums below */
    }
}
static inline int64_t center_term(int j, int32_t kc) {
    int32_t c = 0;
    for (int p2 = 0; p2 < HD / 2; p2++) c += A[p2] * cs[j][p2] + B[p2] * sn[j][p2];
    return (int64_t)((c + 8192) >> 14) * kc;
}
static void s_k1c(void) {
    const int32_t sq = make_lut(), kc = kc_rt;
    center_ab();
    for (int j = 0; j < T; j++) u[j] = (int64_t)(2 * lut[kb[j]] - sq) * km1[j] + center_term(j, kc);
}
static void s_k1uc_win(void) {
    const int32_t sq = make_lut(), ku = ku_rt, kc = kc_rt;
    center_ab();
    for (int j = 0; j < T - WIN; j++) u[j] = (int64_t)(2 * lut[kb[j]] - sq) * ku + center_term(j, kc);
    win_tail();
}
static int64_t umax_of(void) { int64_t m = INT64_MIN; for (int j = 0; j < T; j++) if (U[j] > m) m = U[j]; return m; }
static void sm_base(void) {          /* on the s_i8 scores U (bench_data.h) */
    const int64_t mx = umax_of();
    const int64_t qm = qm_rt;
    const uint32_t ad = (uint32_t)ad_rt;
    int64_t s = 0;
    for (int j = 0; j < T; j++) {
        const uint64_t n20 = (uint64_t)((mx - U[j]) * qm) >> 20;
        const uint32_t xi = n20 < (uint64_t)EXP_N * ad ? (uint32_t)n20 / ad : EXP_N;
        p[j] = xi < EXP_N ? expt[xi] : 0;
        s += p[j];
    }
    P = s;
}
static void sm_temp(void) {
    const int64_t mx = umax_of();
    const int64_t qm = qm_rt;
    const uint64_t ad = (uint64_t)ad_rt, tq = (uint64_t)tq_rt;
    const uint64_t cut = ((((uint64_t)1 << 37) + tq - 1) / tq) * ad;
    int64_t s = 0;
    for (int j = 0; j < T; j++) {
        const uint64_t n10 = (uint64_t)((mx - U[j]) * qm) >> 10;
        uint32_t xi = EXP_N;
        if (n10 < cut) {
            const uint32_t a = (uint32_t)(n10 >> 16), q1 = a / (uint32_t)ad;
            const uint32_t lo = ((a - q1 * (uint32_t)ad) << 16) | (uint32_t)(n10 & 0xFFFF);
            const uint64_t x0 = ((uint64_t)q1 << 16) + lo / (uint32_t)ad;
            xi = (uint32_t)((x0 * tq) >> 26);
        }
        p[j] = xi < EXP_N ? expt[xi] : 0;
        s += p[j];
    }
    P = s;
}
/* tinylm.c's value loop over n positions of vq (rows of HD), weights pv_j = PW_j x scale_j in two 16-bit halves */
static void values_rows(const int8_t* vq, const int32_t* scale, int n) {
    for (int j = 0; j < n; j++) {
        const int64_t pv = (int64_t)PW[j] * scale[j];
        pl[j] = (int32_t)(pv & 0xFFFF); ph[j] = (int32_t)(pv >> 16);
    }
    memset(acc, 0, sizeof acc);
    for (int j0 = 0; j0 < n; j0 += 256)          /* runs of 256 keys: the int32 sums do not overflow */
    for (int i = 0; i < HD; i += 4) {
        int32_t l0 = 0, l1 = 0, l2 = 0, l3 = 0, h0 = 0, h1 = 0, h2 = 0, h3 = 0;
        const int8_t* v = vq + j0 * HD + i;
        const int j1 = j0 + 256 < n ? j0 + 256 : n;
        for (int j = j0; j < j1; j++, v += HD) {
            const int32_t a = pl[j], b = ph[j];
            const int32_t v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
            l0 += a * v0; l1 += a * v1; l2 += a * v2; l3 += a * v3;
            h0 += b * v0; h1 += b * v1; h2 += b * v2; h3 += b * v3;
        }
        acc[i] += (int64_t)h0 * 65536 + l0; acc[i + 1] += (int64_t)h1 * 65536 + l1;
        acc[i + 2] += (int64_t)h2 * 65536 + l2; acc[i + 3] += (int64_t)h3 * 65536 + l3;
    }
}
static void v_i8(void) { values_rows(v8, vm, T); }
static void v2_win(void) {           /* tinylm.c low_bit_values (plain): decode before the window, copy the window */
    const int j0 = T - WIN;
    for (int j = 0; j < j0; j++) {
        for (int i = 0; i < HD; i++)
            vd[j * HD + i] = (int8_t)(2 * ((vb[i * (T / 4) + j / 4] >> (2 * (j % 4))) & 3) - 3);
        pvj[j] = vm2[j];
    }
    memcpy(vd + j0 * HD, v8w, WIN * HD);
    for (int r = 0; r < WIN; r++) pvj[j0 + r] = vm8w[r];
    values_rows(vd, pvj, T);
}

enum { S_I8, S_K1, S_K1_WIN, S_K1U_WIN, S_K1C, S_K1UC_WIN, SM_BASE, SM_TEMP, V_I8, V2_WIN, NOP };
static const char* NAME[NOP] = {"s_i8", "s_k1", "s_k1_win", "s_k1u_win", "s_k1c", "s_k1uc_win", "sm_base",
                                "sm_temp", "v_i8", "v2_win"};
static void (*const FN[NOP])(void) = {s_i8, s_k1, s_k1_win, s_k1u_win, s_k1c, s_k1uc_win, sm_base, sm_temp, v_i8,
                                      v2_win};
static const int64_t* const EXP_U[6] = {EU_I8, EU_K1, EU_K1_WIN, EU_K1U_WIN, EU_K1C, EU_K1UC_WIN};

static int same(int op) {
    if (op < SM_BASE) { for (int j = 0; j < T; j++) if (u[j] != EXP_U[op][j]) return 0; return 1; }
    if (op <= SM_TEMP) {
        const uint16_t* e = op == SM_BASE ? EP_BASE : EP_TEMP;
        for (int j = 0; j < T; j++) if (p[j] != e[j]) return 0;
        return P == (op == SM_BASE ? EPS_BASE : EPS_TEMP);
    }
    const int64_t* e = op == V_I8 ? EA_I8 : EA_V2;
    for (int i = 0; i < HD; i++) if (acc[i] != e[i]) return 0;
    return 1;
}

static void bench_once(void) {
    OUT("KV260K BENCH ready (%s)\n", BOARD_NAME);
    int n = 0;
    for (int op = 0; op < NOP; op++) {
        uint32_t b = 0xFFFFFFFFu;
        memset(u, 0, sizeof u); memset(p, 0, sizeof p); memset(acc, 0, sizeof acc); P = 0;
        for (int r = 0; r < REPS; r++) {
            const uint32_t t0 = NOW();
            for (int k = 0; k < INNER; k++) FN[op]();
            const uint32_t t = (uint32_t)((uint64_t)(NOW() - t0) * SCALE / INNER);
            if (t < b) b = t;
        }
        OUT("R:%s|%lu|%d\n", NAME[op], (unsigned long)b, same(op));
        n++;
    }
    OUT("DONE:%d\n", n);
}

static void setup_data(void) {
    memcpy(q, Q, HD); memcpy(k8, K8, sizeof k8); memcpy(k8w, K8W, sizeof k8w); memcpy(v8, V8, sizeof v8);
    memcpy(v8w, V8W, sizeof v8w); memcpy(kb, KB, sizeof kb); memcpy(vb, VB, sizeof vb);
    memcpy(km, KM, sizeof km); memcpy(km1, KM1, sizeof km1); memcpy(km8w, KM8W, sizeof km8w);
    memcpy(vm, VM, sizeof vm); memcpy(vm2, VM2, sizeof vm2); memcpy(vm8w, VM8W, sizeof vm8w);
    memcpy(cs, COS, sizeof cs); memcpy(sn, SIN, sizeof sn); memcpy(mu, MU, sizeof mu); memcpy(expt, EXPT, sizeof expt);
    ku_rt = KU; kc_rt = KC; qm_rt = QM; ad_rt = AD; tq_rt = TQ;
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


def att_div(hd):
    """intsim.att_div: the integer score divisor of a head width"""
    return (math.isqrt(16129 * 16129 * hd) + 4) >> 3


def make_data():
    rng = np.random.default_rng(2029)
    d = {}
    d["q"] = rng.integers(-127, 128, HD)
    d["k8"], d["v8"] = rng.integers(-127, 128, (T, HD)), rng.integers(-127, 128, (T, HD))
    d["k8w"], d["v8w"] = rng.integers(-127, 128, (WIN, HD)), rng.integers(-127, 128, (WIN, HD))
    d["kb"] = rng.integers(0, 256, T)
    d["vc"] = rng.integers(0, 4, (T, HD))                         # 2-bit codes u = c + 2, value 2u - 3
    d["km"], d["km1"], d["km8w"] = rng.integers(1, 1 << 20, T), rng.integers(1, 1 << 20, T), rng.integers(1, 1 << 20, WIN)
    d["vm"], d["vm2"], d["vm8w"] = rng.integers(1, 1 << 15, T), rng.integers(1, 1 << 15, T), rng.integers(1, 1 << 15, WIN)
    ang = np.arange(T)[:, None] * 10000.0 ** (-2 * np.arange(HD // 2) / HD)
    d["cos"], d["sin"] = np.round(np.cos(ang) * 16384).astype(np.int64), np.round(np.sin(ang) * 16384).astype(np.int64)
    d["mu"] = rng.integers(-127, 128, HD)                          # the key mean, int8
    d["expt"] = np.round(65535 * np.exp(-np.arange(EXP_N) / 128)).astype(np.int64)
    d["pw"] = rng.integers(0, 1 << 16, T)                         # softmax weights for the value ops
    d["us"] = rng.integers(0, 250_000_000, T)                     # softmax input: almost every index inside the table
    d["ku"], d["kc"], d["qm"], d["tq"] = int(rng.integers(1 << 10, 1 << 20)), int(rng.integers(1, 1 << 12)), \
        int(rng.integers(1 << 13, 1 << 15)), int(rng.integers(1 << 15, 1 << 17))
    d["ad"] = att_div(HD)
    return d


def expected(d):
    q, t0 = d["q"], T - WIN
    sign = np.where((d["kb"][:, None] >> np.arange(8)) & 1, 1, -1)   # bit k of the byte = channel k
    k1 = sign @ q
    e = {"EU_I8": (d["k8"] @ q) * d["km"], "EU_K1": k1 * d["km1"]}
    win = (d["k8w"] @ q) * d["km8w"]
    e["EU_K1_WIN"] = np.concatenate([(k1 * d["km1"])[:t0], win])
    e["EU_K1U_WIN"] = np.concatenate([(k1 * d["ku"])[:t0], win])
    q1, q2, m1, m2 = q[0::2], q[1::2], d["mu"][0::2], d["mu"][1::2]
    A, B = (q1 * m1 + q2 * m2) >> 4, (q2 * m1 - q1 * m2) >> 4
    c = ((d["cos"] @ A + d["sin"] @ B + 8192) >> 14) * d["kc"]
    e["EU_K1C"] = k1 * d["km1"] + c
    e["EU_K1UC_WIN"] = np.concatenate([(k1 * d["ku"] + c)[:t0], win])
    U = d["us"]
    dist = (U.max() - U) * d["qm"]
    assert dist.max() < 2 ** 62
    n20 = dist >> 20
    xi = np.where(n20 < EXP_N * d["ad"], n20 // d["ad"], EXP_N)
    e["EP_BASE"] = np.where(xi < EXP_N, d["expt"][np.minimum(xi, EXP_N - 1)], 0)
    n10, tq = dist >> 10, d["tq"]
    cut = (((1 << 37) + tq - 1) // tq) * d["ad"]
    xt = np.where(n10 < cut, ((n10 // d["ad"]) * tq) >> 26, EXP_N)
    e["EP_TEMP"] = np.where(xt < EXP_N, d["expt"][np.minimum(xt, EXP_N - 1)], 0)
    e["EPS_BASE"], e["EPS_TEMP"] = int(e["EP_BASE"].sum()), int(e["EP_TEMP"].sum())
    e["EA_I8"] = (d["pw"] * d["vm"]) @ d["v8"]
    v2 = np.concatenate([2 * d["vc"][:t0] - 3, d["v8w"]])
    s2 = np.concatenate([d["vm2"][:t0], d["vm8w"]])
    e["EA_V2"] = (d["pw"] * s2) @ v2
    assert (np.abs(np.concatenate([e[k] for k in e if k.startswith("EU")])) < 2 ** 62).all()
    assert ((d["pw"] * np.maximum(d["vm"].max(), d["vm2"].max())) < 2 ** 31).all()
    return e


def pack_vb(vc):
    """tinylm.c's 2-bit value layout: channel i, position j -> byte i * (T / 4) + j / 4, bits 2 (j % 4)"""
    vb = np.zeros((HD, T // 4), np.int64)
    for k in range(4):
        vb |= vc[k::4].T << (2 * k)
    return vb


def write_data(path, board):
    d = make_data()
    e = expected(d)
    with open(path, "w") as f:
        f.write(f"#pragma once\n#define BOARD_NAME \"{board}\"\n#define HD {HD}\n#define T {T}\n#define WIN {WIN}\n"
                f"#define EXP_N {EXP_N}\n#define KU {d['ku']}\n#define KC {d['kc']}\n#define QM {d['qm']}\n"
                f"#define AD {d['ad']}\n#define TQ {d['tq']}\n")
        f.write(c_array("int8_t", "Q", d["q"]))
        for name, ct in (("k8", "int8_t"), ("k8w", "int8_t"), ("v8", "int8_t"), ("v8w", "int8_t"), ("kb", "uint8_t"),
                         ("km", "int32_t"), ("km1", "int32_t"), ("km8w", "int32_t"), ("vm", "int32_t"),
                         ("vm2", "int32_t"), ("vm8w", "int32_t"), ("cos", "int16_t"), ("sin", "int16_t"),
                         ("mu", "int16_t"), ("expt", "uint16_t"), ("pw", "int32_t")):
            f.write(c_array(ct, name.upper(), d[name]))
        f.write(c_array("uint8_t", "VB", pack_vb(d["vc"])))
        f.write(c_array("int64_t", "U", d["us"], suffix="LL"))
        for k in ("EU_I8", "EU_K1", "EU_K1_WIN", "EU_K1U_WIN", "EU_K1C", "EU_K1UC_WIN", "EA_I8", "EA_V2"):
            f.write(c_array("int64_t", k, e[k], suffix="LL"))
        for k in ("EP_BASE", "EP_TEMP"):
            f.write(c_array("uint16_t", k, e[k]))
        f.write(f"#define EPS_BASE {e['EPS_BASE']}LL\n#define EPS_TEMP {e['EPS_TEMP']}LL\n")


def read_results(port, board):
    s = open_port(port, 115200, reset=RESET[board])
    rows, ready, t0 = [], False, time.time()
    while time.time() - t0 < 120:
        line = s.readline().decode(errors="replace").strip()
        if "KV260K BENCH ready" in line:
            if f"({board})" not in line:
                s.close()
                sys.exit(f"[X] the board runs another firmware: {line!r}, expected {board}")
            ready, rows = True, []
        elif ready and line.startswith("R:"):
            op, t, ok = line[2:].split("|")
            rows.append((op, int(t), ok == "1"))
        elif ready and line.startswith("DONE"):
            s.close()
            return kv.check_rows(rows, line, lambda r: r[0])
        elif not ready and board == "s3" and time.time() - t0 > 20 and not line:
            s.write(b"\x55")
    s.close()
    sys.exit("[X] no complete results from the board in 120 s (a run that lost lines is not saved)")


# the methods one at a time: (row, scores op, softmax op, values op); each row changes one thing from the one it names
METHODS = [("int8 (base)", "s_i8", "sm_base", "v_i8"),
           ("+ temperature", "s_i8", "sm_temp", "v_i8"),
           ("k1 (1-bit keys)", "s_k1", "sm_base", "v_i8"),
           ("k1c (k1 + center)", "s_k1c", "sm_base", "v_i8"),
           ("k1tr (k1 + window + temp)", "s_k1_win", "sm_temp", "v_i8"),
           ("k1uctr (unit + center + window + temp)", "s_k1uc_win", "sm_temp", "v_i8"),
           ("k1u + window + temp (no center)", "s_k1u_win", "sm_temp", "v_i8"),
           ("k1tr + 2-bit values", "s_k1_win", "sm_temp", "v2_win"),
           ("k1uctr + 2-bit values", "s_k1uc_win", "sm_temp", "v2_win")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=sorted(TICKS), default="esp32")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--no-upload", action="store_true", help="the firmware is already on the board")
    a = ap.parse_args()
    project = os.path.join(HERE, "runs", f"kv260k_bench_{a.board}")
    if not a.no_upload:
        kv.write_project(project, a.port, a.board, FIRMWARE, write_data, "kv260k_bench")
        flash(project, verbose=False, what=a.board)
    rows = read_results(a.port, a.board)
    unit, per_us = TICKS[a.board]
    print(f"\n{a.board}, one query head, {T} positions x {HD} (best of 7)")
    for op, t, ok in rows:
        print(f"  {op:<12}{t:>10} {unit}{t / per_us:>9.1f} us  {'OK' if ok else 'NO'}")
    if not all(r[2] for r in rows):
        sys.exit("[X] a run differs from numpy: the times are not trusted, nothing saved")
    us = {op: t / per_us for op, t, _ in rows}
    methods = []
    print(f"\n  {'method':<42}{'us / head':>10}{'ms / token':>11}")
    for name, s, sm, v in METHODS:
        h = us[s] + us[sm] + us[v]
        methods.append(dict(method=name, ops=[s, sm, v], us_per_head=round(h, 2),
                            ms_per_token=round(h * HEADS_PER_TOKEN / 1000, 3)))
        print(f"  {name:<42}{h:>10.1f}{h * HEADS_PER_TOKEN / 1000:>11.2f}")
    path = os.path.join(HERE, "results", f"kv260k_bench_{a.board}.json")
    with open(path, "w") as f:
        json.dump(dict(board=a.board, head_dim=HD, positions=T, window=WIN, heads_per_token=HEADS_PER_TOKEN,
                       reps="best of 7", unit=unit, date=time.strftime("%Y-%m-%d"),
                       rows=[dict(op=op, **{unit: t}, us=round(t / per_us, 2), equal_numpy=ok) for op, t, ok in rows],
                       methods=methods), f, indent=1)
    print(f"[*] -> {path}")


if __name__ == "__main__":
    main()
