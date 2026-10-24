r"""
patch_and_pie.py -- boolean engine v9: the AND + PIE arithmetic on the ESP32-S3 (symbex_bool.py, convul_symbex_v3/).

benchmarks/experiments/pie_conv_bench.py measured it on this network's three conv shapes: 1.36x, 1.44x and 1.49x faster
than XOR + lut16 (the same results). What changes:
  1. one header for every board: weight rows padded with zeros to 16 bytes (the PIE loads); the ESP32 and x86 keep
     XOR + popcount over the real words only, the padding is never read
  2. ESP32-S3: differences = popcount(window) + popcount(valid weights) - 2 * count(window AND weights), the AND
     counted 128 bits at a time with the PIE multiplier; the window is copied once per position to an aligned buffer
  3. x86 --and-c: the same AND arithmetic in plain C, to check it against the simulation on the PC (must be 100%)
  4. the engine version is saved with every chip measurement (chip_resultados.jsonl, "motor"): S3 latencies measured
     before v9 are not comparable with the new ones
Run from convul_symbex_v3/ (after patch_binary_stem.py and patch_termo_export.py). Backup: symbex_bool.py.bak3.

    python patch_and_pie.py
    python symbex_bool.py exportar --ancho 2.0 --stem termo --niveles 8
    python symbex_bool.py x86 --ancho 2.0 --stem termo --niveles 8            # XOR path: must still be 100%
    python symbex_bool.py x86 --ancho 2.0 --stem termo --niveles 8 --and-c    # AND path in C: must be 100%
    python symbex_bool.py chip --ancho 2.0 --stem termo --niveles 8 --placa s3 --n 1000
"""
import os
import shutil
import sys

CONV_BOOL = r'''static void conv_bool(const CapaBool* L, const uint32_t* in, uint32_t* out) {
    const int pin = (L->cin + 31) / 32, pout = (L->cout + 31) / 32, k = L->k, pad = k / 2;
    const int npal = k * k * pin, fila = k * pin, H = L->hin, s = L->s, st = L->st, kk = k * k;
    const int32_t nbits_lleno = kk * L->cin;
    static int32_t pw_total[MAX_COUT];                     /* popcount of each output's weights (valid ones: minus borders) */
    for (int o = 0; o < L->cout; o++) {
        int32_t t = 0;
        for (int q = 0; q < kk; q++) t += L->pcw[o * kk + q];
        pw_total[o] = t;
    }
    and_preparar();
    int invalido[16];
    for (int oy = 0; oy < L->hout; oy++) {
        for (int ox = 0; ox < L->hout; ox++) {
            const int y0 = oy * s - pad, x0 = ox * s - pad;
            uint32_t* o_px = &out[(oy * L->hout + ox) * pout];
            memset(o_px, 0, pout * 4);
            int nv = 0, ni = 0;
            if (y0 >= 0 && x0 >= 0 && y0 + k <= H && x0 + k <= H) {
                /* interior: the k rows of the window are k runs of `fila` words in the map */
                for (int kh = 0; kh < k; kh++) memcpy(&ven_and[kh * fila], &in[((y0 + kh) * H + x0) * pin], fila * 4);
                nv = kk;
            } else {
                for (int kh = 0; kh < k; kh++) for (int kw = 0; kw < k; kw++) {
                    const int y = y0 + kh, x = x0 + kw;
                    uint32_t* dst = &ven_and[(kh * k + kw) * pin];
                    if (y >= 0 && y < H && x >= 0 && x < H) {
                        memcpy(dst, &in[(y * H + x) * pin], pin * 4);
                        nv++;
                    } else {
                        memset(dst, 0, pin * 4);
                        invalido[ni++] = kh * k + kw;
                    }
                }
            }
            for (int i = npal; i < st; i++) ven_and[i] = 0;       /* padding to the 16-byte row: adds nothing */
            int32_t pcx = 0;
            for (int i = 0; i < npal; i++) pcx += PC(ven_and[i]);
            const int32_t nbits = nv * L->cin;
            for (int o = 0; o < L->cout; o++) {
                int32_t pw = pw_total[o];
                for (int q = 0; q < ni; q++) pw -= L->pcw[o * kk + invalido[q]];
                const int32_t d = pcx + pw - 2 * and_contar(ven_and, &L->w[o * st], st);
                if (nbits - 2 * d >= L->u[o]) o_px[o >> 5] |= 1u << (o & 31);
            }
        }
    }
    (void)nbits_lleno;
}
'''

AND_HELPERS = r'''#if defined(CONFIG_IDF_TARGET_ESP32S3) || defined(FORCE_AND_C)
#define CONV_AND 1
#endif

#ifdef CONV_AND
/* ---- AND arithmetic (v9): differences = popcount(window) + popcount(valid weights) - 2 * count(window AND weights) ----
 * ESP32-S3: the AND count with the PIE unit, 128 bits per AND, counted with the SIMD multiplier (ee.vmulas.u8.accx:
 * bit k of each of the 16 bytes, times 1, summed in ACCX); measured 1.36-1.49x faster than XOR + lut16 on this network's
 * layers (benchmarks/experiments/pie_conv_bench.py). FORCE_AND_C: the same arithmetic in plain C (to check it on a PC).
 * The window is copied once per position to an aligned buffer, padded with zeros to the weight row stride. */
static uint32_t ven_and[MAX_VENTANA + 4] __attribute__((aligned(16)));
#if defined(CONFIG_IDF_TARGET_ESP32S3) && !defined(FORCE_AND_C)
static const uint8_t AND_UNOS[16] __attribute__((aligned(16))) = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#define AND_PASO(k) "ssai " #k " \n\t ee.vsr.32 q3, q2 \n\t ee.andq q3, q3, q6 \n\t ee.vmulas.u8.accx q3, q7 \n\t"
#define ARITMETICA "AND + PIE (ee.vmulas, 128 bits)"
static inline void and_preparar(void) {                   /* q6 = q7 = 16 ones, once per layer */
    const uint8_t* a = AND_UNOS;
    const uint8_t* b = AND_UNOS;
    __asm__ __volatile__("ee.vld.128.ip q6, %0, 0 \n\t ee.vld.128.ip q7, %1, 0 \n\t" : "+r"(a), "+r"(b) : : "memory");
}
static inline int32_t and_contar(const uint32_t* a, const uint32_t* b, int st) {
    __asm__ __volatile__("ee.zero.accx");
    for (int g = 0; g < st / 4; g++) {
        __asm__ __volatile__("ee.vld.128.ip q0, %0, 16 \n\t ee.vld.128.ip q1, %1, 16 \n\t ee.andq q2, q0, q1 \n\t"
                             "ee.andq q3, q2, q6 \n\t ee.vmulas.u8.accx q3, q7 \n\t"
                             AND_PASO(1) AND_PASO(2) AND_PASO(3) AND_PASO(4) AND_PASO(5) AND_PASO(6) AND_PASO(7)
                             : "+r"(a), "+r"(b) : : "memory");
    }
    uint32_t v;
    __asm__ __volatile__("rur.accx_0 %0" : "=r"(v));
    return (int32_t)v;
}
#else
#define ARITMETICA "AND in C (FORCE_AND_C, to check on a PC)"
static inline void and_preparar(void) {}
static inline int32_t and_contar(const uint32_t* a, const uint32_t* b, int st) {
    int32_t c = 0;
    for (int i = 0; i < st; i++) c += PC(a[i] & b[i]);
    return c;
}
#endif
#else
#define ARITMETICA "XOR + popcount"
#endif

'''

EDITS = [
    # the engine needs the target (CONFIG_IDF_TARGET_ESP32S3) before the conv code
    (r'''#include "esp_timer.h"
static inline int64_t reloj_ns(void)''',
     r'''#include "esp_timer.h"
#include "sdkconfig.h"
static inline int64_t reloj_ns(void)'''),
    (r'''#define MOTOR_VERSION "motor booleano v8 (buffers a la medida + tablas de conteo 16/11/0)"''',
     r'''#define MOTOR_VERSION_BASE "motor booleano v9 (filas de pesos de 16 bytes; S3: AND + PIE)"'''),
    (r'''typedef struct { int cin, cout, k, s, hin, hout; const uint32_t* w; const int32_t* u; const int16_t* pcw; } CapaBool;''',
     r'''typedef struct { int cin, cout, k, s, hin, hout; const uint32_t* w; const int32_t* u; const int16_t* pcw; int st; } CapaBool;
/* st: words per weight row, padded with zeros to a multiple of 4 (16 bytes, for the PIE loads of the S3) */'''),
    # header: weight rows padded to 16 bytes, aligned arrays, the stride in each layer
    (r'''        pal = _empacar_px(bits, ppx).reshape(O, k * k * ppx)''',
     r'''        pal = _empacar_px(bits, ppx).reshape(O, k * k * ppx)
        st = -(-(k * k * ppx) // 4) * 4                    # row stride: multiple of 4 words (16 bytes, PIE loads)
        pal = np.pad(pal, ((0, 0), (0, st - k * k * ppx)))'''),
    (r'''pal=pal, umbral=c["umbral"], pcw=pcw))''',
     r'''pal=pal, umbral=c["umbral"], pcw=pcw, st=st))'''),
    (r'''            _arr(f, "uint32_t", f"W{i}", d["pal"], hexa)''',
     r'''            _arr(f, "uint32_t __attribute__((aligned(16)))", f"W{i}", d["pal"], hexa)'''),
    (r'''W{i}, U{i}, PCW{i}}},  /* {d['nombre']} */\n")''',
     r'''W{i}, U{i}, PCW{i}, {d['st']}}},  /* {d['nombre']} */\n")'''),
    # x86 --and-c
    (r'''    if subprocess.run(["gcc", "-O2", "-march=native", "-o", exe, src]).returncode != 0:''',
     r'''    if subprocess.run(["gcc", "-O2", "-march=native", *(["-DFORCE_AND_C"] if getattr(args, "and_c", False) else []),
                       "-o", exe, src]).returncode != 0:'''),
    (r'''    ap.add_argument("--no-subir", action="store_true")''',
     r'''    ap.add_argument("--no-subir", action="store_true")
    ap.add_argument("--and-c", action="store_true",
                    help="x86: run the AND arithmetic in plain C (the S3 path without PIE) to check it against the simulation")'''),
    # the engine version is saved with every measurement
    (r'''        t0, avisado, en_chip, nombres, perfiles = time.time(), False, None, [], []''',
     r'''        t0, avisado, en_chip, nombres, perfiles = time.time(), False, None, [], []
        motor = None'''),
    (r'''            if l.startswith("BOOL |"):
                print(f"  [chip] {l}")''',
     r'''            if l.startswith("BOOL |"):
                print(f"  [chip] {l}")
                if "motor booleano" in l:
                    motor = l.split("BOOL | ", 1)[1].strip()'''),
    (r'''        f.write(json.dumps(dict(placa=args.placa, modelo=carpeta(args.ancho), cod_ram=bool(args.cod_ram),''',
     r'''        f.write(json.dumps(dict(placa=args.placa, modelo=carpeta(args.ancho), motor=motor, cod_ram=bool(args.cod_ram),'''),
]


def main():
    path = "symbex_bool.py"
    if not os.path.exists(path):
        sys.exit("[X] symbex_bool.py not found: run this from convul_symbex_v3/")
    text = open(path).read()
    if "motor booleano v9" in text:
        sys.exit("[*] already patched (motor booleano v9)")
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"[X] the expected text is not in symbex_bool.py (or appears twice); nothing written:\n{old[:300]}")
        text = text.replace(old, new)
    # the conv: the whole function is replaced (XOR path kept, with the padded rows; AND path added)
    a = text.find("static void conv_bool(const CapaBool* L, const uint32_t* in, uint32_t* out) {")
    b = text.find("static void or_mapa(")
    if a < 0 or b < 0 or b < a or text.count("static void conv_bool(") != 1:
        sys.exit("[X] conv_bool / or_mapa not found as expected; nothing written")
    old_conv = text[a:b]
    xor_conv = (old_conv.replace("&L->w[o * npal], npal, d)", "&L->w[o * st], st, d)")
                        .replace("const uint32_t* w = &L->w[o * npal];", "const uint32_t* w = &L->w[o * st];")
                        .replace("const int npal = k * k * pin, fila = k * pin, H = L->hin, s = L->s;",
                                 "const int npal = k * k * pin, fila = k * pin, H = L->hin, s = L->s, st = L->st;"))
    for must in ("&L->w[o * st], st, d)", "const uint32_t* w = &L->w[o * st];", "s = L->s, st = L->st;"):
        if must not in xor_conv:
            sys.exit(f"[X] the XOR conv_bool does not look as expected ({must}); nothing written")
    if "npal]" in xor_conv.replace("ven[MAX_VENTANA]", ""):
        sys.exit("[X] the XOR conv_bool still indexes weights by npal somewhere; nothing written")
    text = text[:a] + AND_HELPERS + "#ifdef CONV_AND\n" + CONV_BOOL + "#else\n" + xor_conv + "#endif\n\n" + text[b:]
    # the version string printed by the chip names the arithmetic too
    text = text.replace('#define MOTOR_VERSION_BASE "motor booleano v9 (filas de pesos de 16 bytes; S3: AND + PIE)"',
                        '#define MOTOR_VERSION_BASE "motor booleano v9 (filas de pesos de 16 bytes; S3: AND + PIE)"\n'
                        '/* MOTOR_VERSION (with the arithmetic in use) is defined after the conv code */', 1)
    text = text.replace("static void or_mapa(", '#define MOTOR_VERSION MOTOR_VERSION_BASE " | " ARITMETICA\n\nstatic void or_mapa(', 1)
    shutil.copy(path, path + ".bak3")
    open(path, "w").write(text)
    print("[*] patched symbex_bool.py -> boolean engine v9 (backup symbex_bool.py.bak3). Now:")
    print("    python symbex_bool.py exportar --ancho 2.0 --stem termo --niveles 8")
    print("    python symbex_bool.py x86 --ancho 2.0 --stem termo --niveles 8")
    print("    python symbex_bool.py x86 --ancho 2.0 --stem termo --niveles 8 --and-c")
    print("    python symbex_bool.py chip --ancho 2.0 --stem termo --niveles 8 --placa s3 --n 1000")


if __name__ == "__main__":
    main()
