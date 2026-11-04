/*
 * symbex_bool_vertical.c -- the vertical engine of the fully binary networks (see symbex_bool_vertical.h and
 * docs/vertical.md). Every count is bit-sliced: a 128-bit "plane" holds one bit of 128 output pixels, and the counts
 * are kept as planes too (bit i of the count of all 128 pixels in plane i), added with carry-save adders.
 */
#if defined(__has_include)                       /* CONFIG_IDF_TARGET_ESP32S3 (ESP-IDF) lives in sdkconfig.h */
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif
#include <string.h>
#include "layers/conv/symbex_bool_vertical.h"
#include "hal/symbex_hal.h"                  /* SYMBEX_FAST_CODE */

#if defined(CONFIG_IDF_TARGET_ESP32S3) && !defined(SYMBEX_HAL_TEST_NO_PIE)
#define VERT_PIE 1
#endif

typedef struct { uint32_t w[4]; } __attribute__((aligned(16))) vplane_t;
#define NB 15                       /* planes of a count (counts < 32768) */
#define EP 12                       /* planes of the threshold operands (every value < 4096) */
#define MAX_KINDS 9                 /* border kinds of a 3 x 3 kernel: 3 row kinds x 3 column kinds */

/* the counting state of one call (on its stack, so that separate instances, tasks or cores never share it): the
   "eights" planes of a count and their next level (in the instance's work block); the portable code also keeps the
   running counters here (PIE: in q0..q2) */
struct VCount {
    vplane_t *ba, *bb;
#if !defined(VERT_PIE)
    vplane_t q[3];
#endif
};

/* ================= vertical counting: Harley-Seal carry-save adders on 128-bit planes =================
   The running count of 128 lanes is kept in three planes (ones, twos, fours: q0..q2); every 8 input planes add to
   them and give one "eights" plane (weight 8), stored; count_finish counts the eights the same way, level after
   level, and writes the count in binary, one plane per bit. */
#if defined(VERT_PIE)
/* q0 = ones, q1 = twos, q2 = fours (kept across calls); q3..q7 temporaries.
   CSA(a, b, c, t): a <- a ^ b ^ c (sum), b <- majority(a, b, c) (carry), 5 logic instructions. */
#define CSA(A_, B_, C_, T_) "ee.xorq " T_ ", " A_ ", " B_ " \n\t ee.andq " B_ ", " A_ ", " B_ " \n\t ee.xorq " A_ ", " T_ ", " C_ " \n\t" \
                            "ee.andq " T_ ", " T_ ", " C_ " \n\t ee.orq " B_ ", " B_ ", " T_ " \n\t"
#define HS8(L0, L1, L2, L3, L4, L5, L6, L7) L0 L1 CSA("q0","q3","q4","q5") L2 L3 CSA("q0","q4","q5","q6") CSA("q1","q3","q4","q5") \
    L4 L5 CSA("q0","q4","q5","q6") L6 L7 CSA("q0","q5","q6","q7") CSA("q1","q4","q5","q6") CSA("q2","q3","q4","q5") \
    "ee.vst.128.ip q3, %[o], 16 \n\t"
#define CL(Q) "ee.vld.128.ip " Q ", %[p], 16 \n\t"                                         /* contiguous load */

/* the planes of a list: the first at p, then p += l[0], l[1]...; 8 planes per group. The deltas are loaded two at a
   time ahead of the loads that use them (the scalar load latency hides behind the adders). */
SYMBEX_FAST_CODE static void hs_gather(struct VCount* S, const uint8_t* p, const uint16_t* l, int groups, vplane_t* o) {
    (void)S;
    uint32_t d0, d1;
    for (int g = 0; g < groups; g++)
        __asm__ __volatile__(
            "l16ui %[d0], %[l], 0 \n\t l16ui %[d1], %[l], 2 \n\t"
            "ee.vld.128.xp q3, %[p], %[d0] \n\t ee.vld.128.xp q4, %[p], %[d1] \n\t"
            "l16ui %[d0], %[l], 4 \n\t l16ui %[d1], %[l], 6 \n\t"
            CSA("q0","q3","q4","q5")
            "ee.vld.128.xp q4, %[p], %[d0] \n\t ee.vld.128.xp q5, %[p], %[d1] \n\t"
            "l16ui %[d0], %[l], 8 \n\t l16ui %[d1], %[l], 10 \n\t"
            CSA("q0","q4","q5","q6") CSA("q1","q3","q4","q5")
            "ee.vld.128.xp q4, %[p], %[d0] \n\t ee.vld.128.xp q5, %[p], %[d1] \n\t"
            "l16ui %[d0], %[l], 12 \n\t l16ui %[d1], %[l], 14 \n\t"
            CSA("q0","q4","q5","q6")
            "ee.vld.128.xp q5, %[p], %[d0] \n\t ee.vld.128.xp q6, %[p], %[d1] \n\t"
            "addi %[l], %[l], 16 \n\t"
            CSA("q0","q5","q6","q7") CSA("q1","q4","q5","q6") CSA("q2","q3","q4","q5")
            "ee.vst.128.ip q3, %[o], 16 \n\t"
            : [p] "+r"(p), [l] "+r"(l), [o] "+r"(o), [d0] "=&r"(d0), [d1] "=&r"(d1) : : "memory");
}
/* consecutive planes, 8 per group */
SYMBEX_FAST_CODE static void hs_contig(struct VCount* S, const vplane_t* p, int groups, vplane_t* o) {
    (void)S;
    for (int g = 0; g < groups; g++)
        __asm__ __volatile__(HS8(CL("q3"), CL("q4"), CL("q4"), CL("q5"), CL("q4"), CL("q5"), CL("q5"), CL("q6"))
                             : [p] "+r"(p), [o] "+r"(o) : : "memory");
}
/* q3 is a temporary everywhere it is called */
static inline void zero_planes(vplane_t* p, int count) {
    if (count <= 0) return;
    __asm__ __volatile__("ee.zero.q q3");
    while (count--) __asm__ __volatile__("ee.vst.128.ip q3, %0, 16" : "+r"(p) : : "memory");
}
static inline void zero3(struct VCount* S) { (void)S; __asm__ __volatile__("ee.zero.q q0 \n\t ee.zero.q q1 \n\t ee.zero.q q2"); }
static inline void store3(struct VCount* S, vplane_t* d) {
    (void)S;
    __asm__ __volatile__("ee.vst.128.ip q0, %0, 16 \n\t ee.vst.128.ip q1, %0, 16 \n\t ee.vst.128.ip q2, %0, 16" : "+r"(d) : : "memory");
}
/* V = 2 count + (2048 - pcx) + (2048 - E), 12 bit planes each (x = the count planes shifted by one): a carry-save
   add of the three operands bit by bit, then the carry chain of sum + carries; out = bit 12 of V.
   q0 = chain carry, q1 / q2 = the carry-save carry (alternating). */
#define E3(KIN, KOUT) "ee.vld.128.ip q3, %[a], 16 \n\t ee.vld.128.ip q4, %[b], 16 \n\t ee.vld.128.ip q5, %[x], 16 \n\t" \
    "ee.xorq q6, q3, q4 \n\t ee.andq " KOUT ", q3, q4 \n\t ee.andq q3, q6, q5 \n\t ee.xorq q6, q6, q5 \n\t" \
    "ee.orq " KOUT ", " KOUT ", q3 \n\t ee.xorq q7, q6, " KIN " \n\t ee.andq q6, q6, " KIN " \n\t" \
    "ee.andq q7, q7, q0 \n\t ee.orq q0, q6, q7 \n\t"
SYMBEX_FAST_CODE static void epilogue3(const vplane_t* a, const vplane_t* b, const vplane_t* x, vplane_t* out) {
    __asm__ __volatile__("ee.zero.q q0 \n\t ee.zero.q q1 \n\t"
                         E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1")
                         E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1") E3("q1","q2") E3("q2","q1")
                         "ee.xorq q2, q1, q0 \n\t ee.vst.128.ip q2, %[o], 0 \n\t"
                         : [a] "+r"(a), [b] "+r"(b), [x] "+r"(x), [o] "+r"(out) : : "memory");
}
#else
/* the same steps in portable C: the running counters in S->q[0..2], the same adder tree as the PIE code */
static inline void csa(vplane_t* a, vplane_t* b, const vplane_t* c) {
    for (int w = 0; w < 4; w++) {
        const uint32_t t = a->w[w] ^ b->w[w];
        b->w[w] = (a->w[w] & b->w[w]) | (t & c->w[w]);
        a->w[w] = t ^ c->w[w];
    }
}
SYMBEX_FAST_CODE static void hs8(vplane_t* Q, const vplane_t* x0, const vplane_t* x1, const vplane_t* x2, const vplane_t* x3,
                                 const vplane_t* x4, const vplane_t* x5, const vplane_t* x6, const vplane_t* x7, vplane_t* o) {
    vplane_t a = *x0, b = *x1, c, d, e, f;
    csa(&Q[0], &a, &b);                          /* a: twos */
    c = *x2; d = *x3;
    csa(&Q[0], &c, &d);                          /* c: twos */
    csa(&Q[1], &a, &c);                          /* a: fours */
    c = *x4; d = *x5;
    csa(&Q[0], &c, &d);
    e = *x6; f = *x7;
    csa(&Q[0], &e, &f);
    csa(&Q[1], &c, &e);                          /* c: fours */
    csa(&Q[2], &a, &c);                          /* a: eights */
    *o = a;
}
SYMBEX_FAST_CODE static void hs_gather(struct VCount* S, const uint8_t* p, const uint16_t* l, int groups, vplane_t* o) {
    for (int g = 0; g < groups; g++, l += 8, o++) {
        const vplane_t* x[8];
        for (int i = 0; i < 8; i++) { x[i] = (const vplane_t*)p; p += l[i]; }
        hs8(S->q, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], o);
    }
}
SYMBEX_FAST_CODE static void hs_contig(struct VCount* S, const vplane_t* p, int groups, vplane_t* o) {
    for (int g = 0; g < groups; g++, p += 8, o++) hs8(S->q, p, p + 1, p + 2, p + 3, p + 4, p + 5, p + 6, p + 7, o);
}
static inline void zero_planes(vplane_t* p, int count) { if (count > 0) memset(p, 0, (size_t)count * sizeof *p); }
static inline void zero3(struct VCount* S) { memset(S->q, 0, sizeof S->q); }
static inline void store3(struct VCount* S, vplane_t* d) { d[0] = S->q[0]; d[1] = S->q[1]; d[2] = S->q[2]; }
SYMBEX_FAST_CODE static void epilogue3(const vplane_t* a, const vplane_t* b, const vplane_t* x, vplane_t* out) {
    for (int w = 0; w < 4; w++) {
        uint32_t chain = 0, kin = 0;
        for (int i = 0; i < EP; i++) {
            const uint32_t A = a[i].w[w], B = b[i].w[w], X = x[i].w[w], ab = A ^ B, s = ab ^ X;
            const uint32_t kout = (A & B) | (ab & X);
            chain = (s & kin) | ((s ^ kin) & chain);
            kin = kout;
        }
        out->w[w] = kin ^ chain;
    }
}
#endif

/* level 1 already ran (its counters in q0..q2, n eights planes in S->ba); the eights are counted again level after
   level, each level giving 3 more bits: bits = the count in binary, one plane per bit; planes past the levels are
   cleared up to `need` (the planes the reader uses) */
SYMBEX_FAST_CODE static void count_finish(struct VCount* S, int n, vplane_t* bits, int need) {
    store3(S, bits);
    int lvl = 1;
    vplane_t *src = S->ba, *dst = S->bb;
    while (n > 0) {
        const int g = (n + 7) >> 3;
        zero_planes(src + n, g * 8 - n);
        zero3(S); hs_contig(S, src, g, dst); store3(S, bits + 3 * lvl); lvl++;
        if (n < 8) break;
        vplane_t* t = src; src = dst; dst = t; n = g;
    }
    zero_planes(bits + 3 * lvl, need - 3 * lvl);
}

/* ================= the plane list of one output channel =================
   The weights that are 1 select the planes to add: the list holds the byte offset of the first one (start) and the
   byte deltas between consecutive ones, padded to a multiple of 8 with the zero plane K. Built from the 1-bit weight
   row one byte (8 weights) at a time: a 256-entry table gives the deltas inside the byte. */
typedef struct { uint16_t d[7]; uint16_t first, last, n; } ListByte;
static void list_table(ListByte* lt) {
    for (int b = 0; b < 256; b++) {
        int c = 0, prev = 0;
        memset(&lt[b], 0, sizeof lt[b]);
        for (int i = 0; i < 8; i++)
            if ((b >> i) & 1) {
                if (c == 0) lt[b].first = (uint16_t)(i * 16); else lt[b].d[c - 1] = (uint16_t)((i - prev) * 16);
                prev = i; c++;
            }
        lt[b].n = (uint16_t)c; lt[b].last = (uint16_t)(prev * 16);
    }
}
/* q is the slot of the delta into the next weight that is 1 (d[-1] first: it receives the start); the 7 in-byte
   deltas are always copied (d has room); a zero byte writes slots that are overwritten later and keeps prev.
   K is a multiple of 8. Returns the number of weights that are 1. */
static int make_list(const uint32_t* row, int K, const ListByte* lt, uint16_t* d, uint32_t* start) {
    const uint8_t* rb = (const uint8_t*)row;
    const int bytes = K / 8;
    uint16_t* q = d - 1;
    uint32_t prev = 0;
    for (int j = 0; j < bytes; j++) {
        const ListByte* e = &lt[rb[j]];
        const uint32_t base = (uint32_t)j * 128, c = e->n;
        q[0] = (uint16_t)(base + e->first - prev);
        q[1] = e->d[0]; q[2] = e->d[1]; q[3] = e->d[2]; q[4] = e->d[3]; q[5] = e->d[4]; q[6] = e->d[5]; q[7] = e->d[6];
        q += c;
        const uint32_t last = base + e->last;
        prev = c ? last : prev;
    }
    const int n = (int)(q - (d - 1));
    *start = n ? d[-1] : 0;
    const int n8 = (n + 7) & ~7;
    if (n) {
        d[n - 1] = (uint16_t)(n8 > n ? (uint32_t)K * 16 - prev : 0);
        for (int i = n; i < n8; i++) d[i] = 0;
    }
    return n;
}

/* ================= shapes and work buffers ================= */
static int divides32(int w) { return w >= 1 && w <= 32 && (32 % w) == 0; }
static const char* check_conv(const struct SymbexBoolConv* L) {
    const int K = L->kernel * L->kernel * L->in_ch;
    if (L->kernel != 1 && L->kernel != 3) return "kernel other than 1 x 1 or 3 x 3";
    if (L->stride == 1) {
        if (L->out_h != L->in_h || L->out_w != L->in_w) return "stride 1 with an output size other than the input";
    } else if (L->stride == 2) {
        if (L->kernel != 3 || L->in_h != 2 * L->out_h || L->in_w != 2 * L->out_w) return "stride 2 other than 3 x 3, input = 2 x output";
    } else {
        return "stride other than 1 or 2";
    }
    if (!divides32(L->in_w) || !divides32(L->out_w)) return "map width that does not divide 32";
    if (((uint32_t)L->in_h * L->in_w) % 32 || ((uint32_t)L->out_h * L->out_w) % 32) return "map with h * w not a multiple of 32";
    if ((uint32_t)L->out_h * L->out_w > 256u * 128) return "output map above 32768 pixels (more than 256 tiles of 128)";
    if (K % 8 || K >= 2048) return "k * k * in_ch not a multiple of 8 or not below 2048";
    return NULL;
}
const char* symbex_bool_vertical_check(const struct SymbexBoolNet* net) {
    if (net->stem_i8) return "int8 stem (the vertical engine runs the thermometer stem only, for now)";
    if (3 * net->levels > 32) return "thermometer with more than 32 bits per pixel";
    if (!divides32(net->in_w) || ((uint32_t)net->in_h * net->in_w) % 32) return "input size (width must divide 32, h * w a multiple of 32)";
    const char* e = check_conv(&net->stem);
    if (e) return e;
    for (uint16_t i = 0; i < net->n_blocks; i++) {
        const struct SymbexBoolBlock* b = &net->blocks[i];
        if ((e = check_conv(&b->conv))) return e;
        if (b->shortcut && (e = check_conv(b->shortcut))) return e;
    }
    return NULL;
}

static uint32_t map_words(int ch, int h, int w) { return (uint32_t)ch * symbex_bool_vplane_stride(h, w) + 4; }
struct Sizes { uint32_t map, phases, xq, ap, km, bt, ba, list, tiles, hwc; };
static void conv_sizes(const struct SymbexBoolConv* L, struct Sizes* s) {
    const uint32_t nwo = (uint32_t)L->out_h * L->out_w / 32, TL = (nwo + 3) / 4;
    const uint32_t K = (uint32_t)L->kernel * L->kernel * L->in_ch, K8 = (K + 1 + 7) / 8 * 8;
    const uint32_t mo = map_words(L->out_ch, L->out_h, L->out_w), ph = L->stride == 2 ? 4 * map_words(L->in_ch, L->out_h, L->out_w) : 0;
    const uint32_t ba = ((K8 / 8 + 7) & ~7u) + 8;
#define UP(f, v) if ((v) > s->f) s->f = (v)
    UP(map, mo); UP(phases, ph); UP(xq, TL * K8); UP(ap, TL * EP); UP(km, TL * MAX_KINDS); UP(bt, TL * EP);
    UP(ba, ba); UP(list, K + 18); UP(tiles, TL);
}
static void net_sizes(const struct SymbexBoolNet* net, struct Sizes* s) {
    memset(s, 0, sizeof *s);
    UP(map, map_words(3 * net->levels, net->in_h, net->in_w));
    conv_sizes(&net->stem, s);
    const struct SymbexBoolConv* last = &net->stem;
    for (uint16_t i = 0; i < net->n_blocks; i++) {
        const struct SymbexBoolBlock* b = &net->blocks[i];
        conv_sizes(&b->conv, s);
        if (b->shortcut) {
            if (b->shortcut_pool) UP(map, map_words(b->conv.in_ch, b->conv.in_h / 2, b->conv.in_w / 2));
            conv_sizes(b->shortcut, s);
        }
        last = &b->conv;
    }
    const uint32_t ppx = (last->out_ch + 31) / 32;
    UP(hwc, (uint32_t)last->out_h * last->out_w * ppx + net->fc_row_words + 4);
#undef UP
}
/* the work block in order (16-byte pieces); with base NULL only the size */
static size_t carve(const struct SymbexBoolNet* net, uint8_t* base, struct SymbexBoolVertical* v) {
    struct Sizes s;
    net_sizes(net, &s);
    size_t off = 0;
#define TAKE(dst, type, bytes) do { if (base) (dst) = (type)(void*)(base + off); off += ((size_t)(bytes) + 15) & ~(size_t)15; } while (0)
    for (int i = 0; i < 4; i++) TAKE(v->maps[i], uint32_t*, s.map * 4);
    TAKE(v->xq, uint32_t*, s.xq * 16);
    const size_t shared = off;                   /* the phases are only read in step 2 of a conv, km and bt only from */
    TAKE(v->km, uint32_t*, s.km * 16);           /* step 4 on (and never with prepared data): they share the bytes */
    TAKE(v->bt, uint32_t*, s.bt * 16);
    if (base) v->phases = (uint32_t*)(void*)(base + shared);
    if (shared + (size_t)s.phases * 4 > off) off = (shared + (size_t)s.phases * 4 + 15) & ~(size_t)15;
    TAKE(v->ap, uint32_t*, s.ap * 16);
    TAKE(v->ba, uint32_t*, s.ba * 16);
    TAKE(v->bb, uint32_t*, s.ba * 16);
    TAKE(v->cb, uint32_t*, (NB + 1) * 16);
    TAKE(v->pxb, uint32_t*, NB * 16);
    TAKE(v->list, uint16_t*, s.list * 2 + 4);
    TAKE(v->tile_type, uint8_t*, s.tiles);
    TAKE(v->tile_rep, uint8_t*, s.tiles);
    TAKE(v->tile_kinds, uint8_t*, s.tiles * MAX_KINDS);
    TAKE(v->tile_nkinds, uint8_t*, s.tiles);
    TAKE(v->list_table, void*, 256 * sizeof(ListByte));
    TAKE(v->hwc, uint32_t*, s.hwc * 4);
#undef TAKE
    if (base) v->list += 2;                  /* list[-1] receives the start offset */
    return off;
}
size_t symbex_bool_vertical_work_bytes(const struct SymbexBoolNet* net) {
    if (symbex_bool_vertical_check(net)) return 0;
    struct SymbexBoolVertical v;
    return carve(net, NULL, &v);
}
int symbex_bool_vertical_init(const struct SymbexBoolNet* net, void* work, size_t bytes, struct SymbexBoolVertical* v) {
    const size_t need = symbex_bool_vertical_work_bytes(net);
    if (!need || bytes < need || ((uintptr_t)work & 15)) return -1;
    memset(work, 0, need);
    carve(net, (uint8_t*)work, v);
    list_table((ListByte*)v->list_table);
    v->net = net;
    v->prep = NULL;
    v->prep_compact = 0;
    return 0;
}

/* ================= the layers ================= */
#define PLANE(map, ps, c) ((map) + 4 + (size_t)(c) * (ps))

/* even bits to the low half, odd bits to the high half (Hacker's Delight unshuffle) */
static inline uint32_t unshuffle(uint32_t x) {
    uint32_t t;
    t = (x ^ (x >> 1)) & 0x22222222u; x ^= t ^ (t << 1);
    t = (x ^ (x >> 2)) & 0x0C0C0C0Cu; x ^= t ^ (t << 2);
    t = (x ^ (x >> 4)) & 0x00F000F0u; x ^= t ^ (t << 4);
    t = (x ^ (x >> 8)) & 0x0000FF00u; x ^= t ^ (t << 8);
    return x;
}
/* the 32 x 32 bit matrix a (row r = a[r], column c = bit c) transposed in place */
SYMBEX_FAST_CODE static void transpose32(uint32_t a[32]) {
    uint32_t m = 0x0000FFFFu;
    for (int j = 16; j; j >>= 1, m ^= m << j)
        for (int k = 0; k < 32; k = (k + j + 1) & ~j) {
            const uint32_t t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
        }
}

SYMBEX_FAST_CODE void symbex_bool_vthermometer(const uint8_t* img, uint16_t h, uint16_t w, uint8_t levels,
                                               uint32_t* out) {
    const int nch = 3 * levels, l2 = 2 * levels;
    const uint32_t ps = symbex_bool_vplane_stride(h, w), n = (uint32_t)h * w;
    uint32_t lut[256];                           /* pixel value -> its thermometer bits, as symbex_thermometer */
    {
        int k = 0;
        uint32_t bits = 0;
        for (int v = 0; v < 256; v++) {
            while (k < levels && v >= (k + 1) * 256 / (levels + 1)) bits |= 1UL << k++;
            lut[v] = bits;
        }
    }
    memset(out, 0, map_words(nch, h, w) * 4);
    uint32_t a[32];
    for (uint32_t b = 0; b < n / 32; b++) {      /* 32 pixels: their bits, transposed into one word per channel */
        for (int j = 0; j < 32; j++) {
            const uint8_t* px = &img[(b * 32 + j) * 3];
            a[j] = lut[px[0]] | (lut[px[1]] << levels) | (l2 < 32 ? lut[px[2]] << l2 : 0);
        }
        transpose32(a);
        for (int c = 0; c < nch; c++) PLANE(out, ps, c)[b] = a[c];
    }
}

SYMBEX_FAST_CODE void symbex_bool_vor_pool(const uint32_t* in, uint32_t* out, uint16_t ch, uint16_t h, uint16_t w) {
    const int oh = h / 2, ow = w / 2;
    const uint32_t psi = symbex_bool_vplane_stride(h, w), pso = symbex_bool_vplane_stride(oh, ow);
    const uint32_t rm = w == 32 ? ~0u : (1u << w) - 1, om = ow == 32 ? ~0u : (1u << ow) - 1;
    memset(out, 0, map_words(ch, oh, ow) * 4);
    for (int c = 0; c < ch; c++) {
        const uint32_t* s = PLANE(in, psi, c);
        uint32_t* d = PLANE(out, pso, c);
        for (int y = 0; y < oh; y++) {
            const int b0 = 2 * y * w, b1 = b0 + w, ob = y * ow;
            const uint32_t r = ((s[b0 >> 5] >> (b0 & 31)) | (s[b1 >> 5] >> (b1 & 31))) & rm;
            const uint32_t u = unshuffle(r);
            d[ob >> 5] |= ((u | (u >> 16)) & om) << (ob & 31);
        }
    }
}

SYMBEX_FAST_CODE void symbex_bool_vor(uint32_t* dst, const uint32_t* src, uint16_t ch, uint16_t h, uint16_t w) {
    const uint32_t n = map_words(ch, h, w);
    for (uint32_t i = 0; i < n; i++) dst[i] |= src[i];
}

/* out bit p = src bit p + off (0 outside the plane: the zero words around it), then the column mask; written into
   the tiles of the shifted planes (one plane every `step` words) */
SYMBEX_FAST_CODE static void shift_plane(const uint32_t* s, int off, uint32_t mask, uint32_t* d, int nwo, int step) {
    const int a = off >> 5, b = off & 31;
    s += a;
    const int full = nwo & ~3;                   /* whole tiles, 4 words each, then the words of a part tile */
    if (b == 0) {
        for (int i = 0; i < full; i += 4, s += 4, d += step) {
            d[0] = s[0] & mask; d[1] = s[1] & mask; d[2] = s[2] & mask; d[3] = s[3] & mask;
        }
        for (int i = 0; i < nwo - full; i++) d[i] = s[i] & mask;
    } else {
        const int c = 32 - b;
        for (int i = 0; i < full; i += 4, s += 4, d += step) {
            d[0] = ((s[0] >> b) | (s[1] << c)) & mask; d[1] = ((s[1] >> b) | (s[2] << c)) & mask;
            d[2] = ((s[2] >> b) | (s[3] << c)) & mask; d[3] = ((s[3] >> b) | (s[4] << c)) & mask;
        }
        for (int i = 0; i < nwo - full; i++) d[i] = ((s[i] >> b) | (s[i + 1] << c)) & mask;
    }
}
/* the lanes of a word whose column is 0 (sx = -1) or ow - 1 (sx = +1) cleared: their source is outside the image */
static uint32_t colmask(int sx, int ow) {
    if (sx == 0) return ~0u;
    uint32_t p = 0;
    for (int j = 0; j < 32; j += ow) p |= 1u << j;
    return sx < 0 ? ~p : ~(p << (ow - 1));
}
/* the kernel rows (columns) of output row oy (column ox) that fall inside the image, as a bit mask */
static uint32_t valid_mask(int o, int s, int pad, int k, int n) {
    uint32_t m = 0;
    for (int i = 0; i < k; i++) { const int y = o * s - pad + i; if (y >= 0 && y < n) m |= 1u << i; }
    return m;
}
static int find_or_add(uint32_t* vals, int* cnt, int* n, uint32_t v) {
    int i = 0;
    while (i < *n && vals[i] != v) i++;
    if (i == *n) { vals[i] = v; cnt[i] = 0; (*n)++; }
    cnt[i]++;
    return i;
}

/* the border kinds of a layer (row kind x column kind: the windows that keep the same kernel pixels inside the
   image) and its tile types (tiles with the same kind masks share their threshold planes). v->km[t * NK + kind] = the
   lanes of that kind in tile t (the lanes past the map take the base kind, the commonest); v->tile_type[t] = the type
   of tile t; tile_rep / tile_kinds / tile_nkinds = per type, a tile of it and its kinds other than the base. */
struct Kinds { uint32_t rv[3], cv[3]; int nr, nc, NK, KBASE, NTT, nvk[MAX_KINDS]; };
static void border_kinds(const struct SymbexBoolConv* L, struct SymbexBoolVertical* v, struct Kinds* kd) {
    const int kq = L->kernel, pad = kq / 2, sd = L->stride, H = L->in_h, W = L->in_w, OH = L->out_h, OW = L->out_w;
    const int nwo = OH * OW / 32, TL = (nwo + 3) / 4;
    vplane_t* KM = (vplane_t*)v->km;
    uint32_t *rv = kd->rv, *cv = kd->cv;
    int rc[3], cc[3], nr = 0, nc = 0;
    uint32_t cmask[3] = {0, 0, 0};
    for (int ox = 0; ox < OW; ox++) {
        const int k = find_or_add(cv, cc, &nc, valid_mask(ox, sd, pad, kq, W));
        for (int j = ox; j < 32; j += OW) cmask[k] |= 1u << j;
    }
    for (int oy = 0; oy < OH; oy++) find_or_add(rv, rc, &nr, valid_mask(oy, sd, pad, kq, H));
    const int NK = nr * nc;
    int rb = 0, cbk = 0;
    for (int i = 1; i < nr; i++) if (rc[i] > rc[rb]) rb = i;
    for (int i = 1; i < nc; i++) if (cc[i] > cc[cbk]) cbk = i;
    const int KBASE = rb * nc + cbk;
    zero_planes(KM, TL * NK);
    const int rows_per_word = 32 / OW;
    const uint32_t rowbits = OW == 32 ? ~0u : (1u << OW) - 1;
    for (int wd = 0; wd < 4 * TL; wd++) {
        const int t = wd / 4, j = wd % 4;
        if (wd >= nwo) { KM[t * NK + KBASE].w[j] = ~0u; continue; }
        uint32_t rbits[3] = {0, 0, 0};
        for (int r = 0; r < rows_per_word; r++) {
            const int oy = wd * rows_per_word + r;
            const uint32_t m = valid_mask(oy, sd, pad, kq, H);
            int k = 0;
            while (rv[k] != m) k++;
            rbits[k] |= rowbits << (r * OW);
        }
        for (int a = 0; a < nr; a++)
            for (int b = 0; b < nc; b++) KM[t * NK + a * nc + b].w[j] = rbits[a] & cmask[b];
    }
    uint8_t *TT = v->tile_type, *REP = v->tile_rep, *TK = v->tile_kinds, *TKN = v->tile_nkinds;
    int NTT = 0;
    for (int t = 0; t < TL; t++) {
        int ty = 0;
        while (ty < NTT && memcmp(&KM[REP[ty] * NK], &KM[t * NK], (size_t)NK * sizeof(vplane_t))) ty++;
        if (ty == NTT) {
            REP[NTT] = (uint8_t)t; TKN[NTT] = 0;            /* t < 256: check_conv caps the tiles */
            for (int k = 0; k < NK; k++) {
                const vplane_t* m = &KM[t * NK + k];
                if (k != KBASE && (m->w[0] | m->w[1] | m->w[2] | m->w[3])) TK[NTT * MAX_KINDS + TKN[NTT]++] = (uint8_t)k;
            }
            NTT++;
        }
        TT[t] = (uint8_t)ty;
    }
    for (int a = 0; a < nr; a++)
        for (int b = 0; b < nc; b++) kd->nvk[a * nc + b] = __builtin_popcount(rv[a]) * __builtin_popcount(cv[b]);
    kd->nr = nr; kd->nc = nc; kd->NK = NK; kd->KBASE = KBASE; kd->NTT = NTT;
}
/* the threshold of output channel o per border kind (kd from border_kinds): E = ceil((thr + 2 pw - nv * cin) / 2)
   (pw: weights that are 1 at the kernel pixels inside the image, nv: those pixels), clamped to [-K, K + 1], kept as
   B = 2048 - E */
static void kind_values(const struct SymbexBoolConv* L, int o, const struct Kinds* kd, uint32_t* bk) {
    const int kq = L->kernel, kk = kq * kq, cin = L->in_ch, K = kk * cin, nc = kd->nc;
    const int16_t* wc = L->weight_counts;
    for (int a = 0; a < kd->nr; a++)
        for (int b = 0; b < nc; b++) {
            int pw = 0;
            for (int kh = 0; kh < kq; kh++)
                if ((kd->rv[a] >> kh) & 1)
                    for (int kw = 0; kw < kq; kw++)
                        if ((kd->cv[b] >> kw) & 1) pw += wc[o * kk + kh * kq + kw];
            const int64_t R = (int64_t)L->thresholds[o] + 2 * pw - (int64_t)kd->nvk[a * nc + b] * cin;
            int64_t e = (R + 1) >> 1;
            if (e < -K) e = -K;
            if (e > K + 1) e = K + 1;
            bk[a * nc + b] = (uint32_t)(2048 - e);
        }
}
/* the threshold planes from the kind values bk: one set of EP planes per tile type, every lane holding the B of its
   kind (KM: the kind masks, tile type ty at KM + REP[ty] * NK, or KM + ty * NK when REP is NULL) */
static void kind_planes(const uint32_t* bk, int KBASE, int NK, int NTT, const vplane_t* KM, const uint8_t* REP,
                        const uint8_t* TK, const uint8_t* TKN, vplane_t* BT) {
    const uint32_t bb = bk[KBASE];
    for (int ty = 0; ty < NTT; ty++) {
        vplane_t* P = BT + ty * EP;
        for (int i = 0; i < EP; i++) {
            const uint32_t m = 0u - ((bb >> i) & 1);
            P[i].w[0] = m; P[i].w[1] = m; P[i].w[2] = m; P[i].w[3] = m;
        }
        for (int j = 0; j < TKN[ty]; j++) {
            const int k = TK[ty * MAX_KINDS + j];
            const vplane_t* m = &KM[(REP ? REP[ty] : ty) * NK + k];
            for (uint32_t d = bk[k] ^ bb; d; d &= d - 1) {
                vplane_t* q = &P[__builtin_ctz(d)];
                q->w[0] ^= m->w[0]; q->w[1] ^= m->w[1]; q->w[2] ^= m->w[2]; q->w[3] ^= m->w[3];
            }
        }
    }
}
/* the threshold planes of output channel o, one set of EP planes per tile type */
static void out_thresholds(const struct SymbexBoolConv* L, int o, const struct Kinds* kd,
                           const struct SymbexBoolVertical* v, vplane_t* BT) {
    uint32_t bk[MAX_KINDS];
    kind_values(L, o, kd, bk);
    kind_planes(bk, kd->KBASE, kd->NK, kd->NTT, (const vplane_t*)v->km, v->tile_rep, v->tile_kinds, v->tile_nkinds, BT);
}
/* the convs of a network in order (stem, then every block's conv and shortcut): L's position, -1 if not there */
static int conv_index(const struct SymbexBoolNet* net, const struct SymbexBoolConv* L) {
    if (!net) return -1;
    if (L == &net->stem) return 0;
    int i = 1;
    for (uint16_t b = 0; b < net->n_blocks; b++) {
        if (L == &net->blocks[b].conv) return i;
        i++;
        if (net->blocks[b].shortcut) {
            if (L == net->blocks[b].shortcut) return i;
            i++;
        }
    }
    return -1;
}
static int n_convs(const struct SymbexBoolNet* net) {
    int n = 1;
    for (uint16_t b = 0; b < net->n_blocks; b++) n += net->blocks[b].shortcut ? 2 : 1;
    return n;
}
static const struct SymbexBoolConv* conv_at(const struct SymbexBoolNet* net, int idx) {
    if (idx == 0) return &net->stem;
    int i = 1;
    for (uint16_t b = 0; b < net->n_blocks; b++) {
        if (i++ == idx) return &net->blocks[b].conv;
        if (net->blocks[b].shortcut && i++ == idx) return net->blocks[b].shortcut;
    }
    return NULL;
}

/* ================= the prepared data (optional) =================
   Every conv's plane lists and threshold planes, built once into RAM by symbex_bool_vertical_prepare, so that
   symbex_bool_vconv only reads them. Layout (offsets in bytes from the start, every piece 16-byte aligned):
       uint32 dir[3 * n_convs]      per conv: the offsets of its threshold planes, its lists and its tile types
       per conv: threshold planes   out_ch x tile types x EP planes (output channel o, tile type ty at (o * NTT + ty) * EP)
                                    compact (symbex_bool_vertical_prepare_compact): instead, NK, KBASE, NTT (uint16),
                                    the NK kind masks of every tile type, its kinds and their count (tile_kinds,
                                    tile_nkinds), then the NK kind values of every output channel (uint16): the
                                    planes are built per output channel while running, from about half the RAM
                 lists              per output channel: start, n, then the n8 = n rounded up to 8 list entries (uint16)
                 tile types         NTT, then the type of every tile (uint8), so the run needs no border_kinds */
static size_t prepare_walk(struct SymbexBoolVertical* v, uint8_t* base, int compact) {
    const struct SymbexBoolNet* net = v->net;
    const int nconv = n_convs(net);
    size_t off = ((size_t)12 * nconv + 15) & ~(size_t)15;
    const ListByte* lt = (const ListByte*)v->list_table;
    for (int ci = 0; ci < nconv; ci++) {
        const struct SymbexBoolConv* L = conv_at(net, ci);
        const int K = L->kernel * L->kernel * L->in_ch;
        struct Kinds kd;
        border_kinds(L, v, &kd);
        if (base) ((uint32_t*)base)[3 * ci] = (uint32_t)off;
        if (compact) {
            const size_t kmb = (size_t)kd.NTT * kd.NK * sizeof(vplane_t);
            if (base) {
                uint16_t* h = (uint16_t*)(base + off);
                h[0] = (uint16_t)kd.NK; h[1] = (uint16_t)kd.KBASE; h[2] = (uint16_t)kd.NTT;
                vplane_t* km = (vplane_t*)(base + off + 16);
                for (int ty = 0; ty < kd.NTT; ty++)
                    memcpy(km + ty * kd.NK, (const vplane_t*)v->km + v->tile_rep[ty] * kd.NK, (size_t)kd.NK * sizeof(vplane_t));
                uint8_t* tk = base + off + 16 + kmb;
                memcpy(tk, v->tile_kinds, (size_t)kd.NTT * MAX_KINDS);
                memcpy(tk + kd.NTT * MAX_KINDS, v->tile_nkinds, (size_t)kd.NTT);
                uint16_t* bv = (uint16_t*)(base + off + 16 + kmb + (((size_t)kd.NTT * (MAX_KINDS + 1) + 15) & ~(size_t)15));
                for (int o = 0; o < L->out_ch; o++) {
                    uint32_t bk[MAX_KINDS];
                    kind_values(L, o, &kd, bk);
                    for (int k = 0; k < kd.NK; k++) bv[o * kd.NK + k] = (uint16_t)bk[k];
                }
            }
            off += 16 + kmb + (((size_t)kd.NTT * (MAX_KINDS + 1) + 15) & ~(size_t)15)
                 + (((size_t)L->out_ch * kd.NK * 2 + 15) & ~(size_t)15);
        } else {
            for (int o = 0; base && o < L->out_ch; o++)
                out_thresholds(L, o, &kd, v, (vplane_t*)(base + off) + (size_t)o * kd.NTT * EP);
            off += (size_t)L->out_ch * kd.NTT * EP * sizeof(vplane_t);
        }
        if (base) ((uint32_t*)base)[3 * ci + 1] = (uint32_t)off;
        size_t lb = 0;
        for (int o = 0; o < L->out_ch; o++) {
            uint32_t start;
            const int n = make_list(L->weights + (size_t)o * L->row_words, K, lt, v->list, &start);
            const int n8 = (n + 7) & ~7;
            if (base) {
                uint16_t* d = (uint16_t*)(base + off + lb);
                d[0] = (uint16_t)start; d[1] = (uint16_t)n;
                memcpy(d + 2, v->list, (size_t)n8 * 2);
            }
            lb += (size_t)(2 + n8) * 2;
        }
        off += (lb + 15) & ~(size_t)15;
        const int TL = (L->out_h * L->out_w / 32 + 3) / 4;
        if (base) {
            ((uint32_t*)base)[3 * ci + 2] = (uint32_t)off;
            base[off] = (uint8_t)kd.NTT;
            memcpy(base + off + 1, v->tile_type, (size_t)TL);
        }
        off += ((size_t)1 + TL + 15) & ~(size_t)15;
    }
    return off;
}
static int prepare(struct SymbexBoolVertical* v, void* buf, size_t bytes, int compact) {
    if (!v || !v->net || ((uintptr_t)buf & 15)) return -1;
    v->prep = NULL;
    const size_t need = prepare_walk(v, NULL, compact);
    if (bytes < need) return -1;
    prepare_walk(v, (uint8_t*)buf, compact);
    v->prep = buf;
    v->prep_compact = compact;
    return 0;
}
size_t symbex_bool_vertical_prepared_bytes(struct SymbexBoolVertical* v) {
    return v && v->net ? prepare_walk(v, NULL, 0) : 0;
}
int symbex_bool_vertical_prepare(struct SymbexBoolVertical* v, void* buf, size_t bytes) {
    return prepare(v, buf, bytes, 0);
}
size_t symbex_bool_vertical_prepared_compact_bytes(struct SymbexBoolVertical* v) {
    return v && v->net ? prepare_walk(v, NULL, 1) : 0;
}
int symbex_bool_vertical_prepare_compact(struct SymbexBoolVertical* v, void* buf, size_t bytes) {
    return prepare(v, buf, bytes, 1);
}

/* step 2 of a conv for the words w0 .. w0 + nw - 1 of the output map (whole tiles from w0, a multiple of 4): the
   shifted plane of every (kernel pixel q, channel c) into plane q * cin + c of the tiles at XQ (K8 planes each) */
SYMBEX_FAST_CODE static void shift_all(const struct SymbexBoolConv* L, const uint32_t* src, uint32_t sps, int w0,
                                       int nw, vplane_t* XQ, int K8) {
    const int cin = L->in_ch, kq = L->kernel, pad = kq / 2, OW = L->out_w;
    for (int kh = 0; kh < kq; kh++)
        for (int kw = 0; kw < kq; kw++) {
            const int q = kh * kq + kw;
            int sy, sx, ph = 0;
            if (L->stride == 1) { sy = kh - pad; sx = kw - pad; }
            else { sy = (kh - pad) >> 1; sx = (kw - pad) >> 1; ph = ((kh - pad) & 1) * 2 + ((kw - pad) & 1); }
            const int off = sy * OW + sx;
            const uint32_t m = colmask(sx, OW);
            for (int c = 0; c < cin; c++)
                shift_plane(PLANE(src, sps, L->stride == 1 ? c : ph * cin + c) + w0, off, m, (uint32_t*)(XQ + q * cin + c),
                            nw, K8 * 4);
        }
}
/* step 3 for one tile: the bits of every window (pcx), kept as the planes of 2048 - pcx = ~pcx + 1 with bit 11
   flipped */
SYMBEX_FAST_CODE static void window_bits(struct VCount* S, const vplane_t* XQt, int K8, vplane_t* PXB, vplane_t* APt) {
    zero3(S); hs_contig(S, XQt, K8 / 8, S->ba); count_finish(S, K8 / 8, PXB, EP);
    for (int w = 0; w < 4; w++) {
        uint32_t c = ~0u;
        for (int i = 0; i < EP; i++) {
            const uint32_t a = ~PXB[i].w[w], s = a ^ c;
            c &= a;
            APt[i].w[w] = i == EP - 1 ? ~s : s;
        }
    }
}

SYMBEX_FAST_CODE void symbex_bool_vconv(const struct SymbexBoolConv* L, const uint32_t* in, uint32_t* out,
                                        struct SymbexBoolVertical* v) {
    const int cin = L->in_ch, cout = L->out_ch, kq = L->kernel, kk = kq * kq, sd = L->stride;
    const int H = L->in_h, W = L->in_w, OH = L->out_h, OW = L->out_w;
    const int nwo = OH * OW / 32, TL = (nwo + 3) / 4, K = kk * cin, K8 = (K + 1 + 7) / 8 * 8;
    const uint32_t psi = symbex_bool_vplane_stride(H, W), pso = symbex_bool_vplane_stride(OH, OW);
    vplane_t *XQ = (vplane_t*)v->xq, *AP = (vplane_t*)v->ap, *BT = (vplane_t*)v->bt;
    vplane_t *CB = (vplane_t*)v->cb, *PXB = (vplane_t*)v->pxb;
    struct VCount S = {.ba = (vplane_t*)v->ba, .bb = (vplane_t*)v->bb};

    /* 1. stride 2: the 4 phases (even / odd rows x columns) of every input plane, on the output grid */
    const uint32_t* src = in;
    uint32_t sps = psi;
    if (sd == 2) {
        const uint32_t om = OW == 32 ? ~0u : (1u << OW) - 1, rm = W == 32 ? ~0u : (1u << W) - 1;
        memset(v->phases, 0, (size_t)4 * map_words(cin, OH, OW) * 4);
        for (int c = 0; c < cin; c++) {
            const uint32_t* s = PLANE(in, psi, c);
            uint32_t* ph[4];
            for (int r = 0; r < 4; r++) ph[r] = PLANE(v->phases, pso, r * cin + c);
            for (int y = 0; y < H; y++) {
                const int bit = y * W, ob = (y >> 1) * OW;
                const uint32_t u = unshuffle((s[bit >> 5] >> (bit & 31)) & rm);
                uint32_t *ev = ph[(y & 1) * 2], *od = ph[(y & 1) * 2 + 1];
                ev[ob >> 5] |= (u & om) << (ob & 31);
                od[ob >> 5] |= ((u >> 16) & om) << (ob & 31);
            }
        }
        src = v->phases; sps = pso;
    }
    /* 2. the shifted plane of every (kernel pixel q, channel c): plane q * cin + c of every tile (the order of the
       weight row bits); planes K..K8-1 are 0 (K is the zero plane of the lists) */
    for (int t = 0; t < TL; t++) zero_planes(XQ + (size_t)t * K8 + K, K8 - K);
    shift_all(L, src, sps, 0, nwo, XQ, K8);
    /* 3. the bits of every window (pcx) */
    for (int t = 0; t < TL; t++) window_bits(&S, XQ + (size_t)t * K8, K8, PXB, AP + t * EP);
    const int pi = v->prep ? conv_index(v->net, L) : -1;
    if (pi >= 0) {
        /* 4-5 prepared: the tile types, lists and threshold planes read from the prepared block */
        const uint8_t* pb = (const uint8_t*)v->prep;
        const uint32_t* dir = (const uint32_t*)v->prep;
        const uint8_t* tt = pb + dir[3 * pi + 2];
        const int NTT = tt[0];
        const uint16_t* lp = (const uint16_t*)(pb + dir[3 * pi + 1]);
        const vplane_t* bt = (const vplane_t*)(pb + dir[3 * pi]);
        /* compact: the threshold planes of each output channel built here from its kind values */
        const uint16_t* h = (const uint16_t*)(pb + dir[3 * pi]);
        const int NK = h[0], KB = h[1];
        const vplane_t* kmr = (const vplane_t*)(pb + dir[3 * pi] + 16);
        const uint8_t* tk = (const uint8_t*)(kmr + NTT * NK);
        const uint16_t* bv = (const uint16_t*)(tk + ((NTT * (MAX_KINDS + 1) + 15) & ~15));
        for (int o = 0; o < cout; o++) {
            const int n = lp[1], n8 = (n + 7) & ~7;
            vplane_t* po = (vplane_t*)PLANE(out, pso, o);
            if (v->prep_compact) {
                uint32_t bk[MAX_KINDS];
                for (int k = 0; k < NK; k++) bk[k] = bv[o * NK + k];
                kind_planes(bk, KB, NK, NTT, kmr, NULL, tk, tk + NTT * MAX_KINDS, BT);
                bt = BT;
            }
            for (int t = 0; t < TL; t++) {
                zero3(&S);
                if (n8) hs_gather(&S, (const uint8_t*)(XQ + (size_t)t * K8) + lp[0], lp + 2, n8 / 8, S.ba);
                count_finish(&S, n >= 8 ? n8 / 8 : 0, CB + 1, EP - 1);
                epilogue3(AP + t * EP, bt + tt[1 + t] * EP, CB, po + t);
            }
            lp += 2 + n8;
            bt += NTT * EP;
        }
    } else {
    /* 4. border kinds and tile types (the shape only) */
    struct Kinds kd;
    border_kinds(L, v, &kd);

    /* 5. every output channel: its list and threshold planes per tile type, built here, then every tile: count,
       threshold */
    const ListByte* lt = (const ListByte*)v->list_table;
    const uint8_t* TT = v->tile_type;
    for (int o = 0; o < cout; o++) {
        uint32_t start;
        const int n = make_list(L->weights + (size_t)o * L->row_words, K, lt, v->list, &start), n8 = (n + 7) & ~7;
        out_thresholds(L, o, &kd, v, BT);
        vplane_t* po = (vplane_t*)PLANE(out, pso, o);
        for (int t = 0; t < TL; t++) {
            zero3(&S);
            if (n8) hs_gather(&S, (const uint8_t*)(XQ + (size_t)t * K8) + start, v->list, n8 / 8, S.ba);
            count_finish(&S, n >= 8 ? n8 / 8 : 0, CB + 1, EP - 1);   /* the epilogue reads CB[0..EP-1] */
            epilogue3(AP + t * EP, BT + TT[t] * EP, CB, po + t);
        }
    }
    }
    /* 6. the zero words of the output map: before plane 0, past the map in every plane and between the planes */
    out[0] = out[1] = out[2] = out[3] = 0;
    for (int o = 0; o < cout; o++) {
        uint32_t* p = PLANE(out, pso, o);
        for (uint32_t i = (uint32_t)nwo; i < pso; i++) p[i] = 0;
    }
}

SYMBEX_FAST_CODE int symbex_bool_vhead(const struct SymbexBoolNet* net, const uint32_t* map, uint16_t ch, uint16_t h,
                                       uint16_t w, struct SymbexBoolVertical* v) {
    const uint32_t ps = symbex_bool_vplane_stride(h, w), n = (uint32_t)h * w, ppx = (ch + 31) / 32;
    uint32_t* hwc = v->hwc;
    memset(hwc + (size_t)n * ppx, 0, (size_t)net->fc_row_words * 4);
    uint32_t a[32];                              /* planes -> HWC (pixel p, channel c: bit c % 32 of word p * ppx + c / 32): */
    for (uint32_t g = 0; g < ppx; g++)           /* 32 channels x 32 pixels at a time, transposed */
        for (uint32_t i = 0; i < n / 32; i++) {
            for (uint32_t j = 0; j < 32; j++) a[j] = g * 32 + j < ch ? PLANE(map, ps, g * 32 + j)[i] : 0;
            transpose32(a);
            uint32_t* dst = hwc + (size_t)i * 32 * ppx + g;
            for (int px = 0; px < 32; px++) dst[(size_t)px * ppx] = a[px];
        }
    if (net->cells) {
        symbex_bool_cell_head(hwc, net->cells_h, net->cells_w, (uint16_t)net->fc_bits, net->fc_weights, net->fc_row_words,
                              net->n_classes, net->fc_scale, net->fc_bias, net->scratch, net->cells);
        return symbex_bool_grid_checksum(net->cells, (uint32_t)net->cells_h * net->cells_w);
    }
    return symbex_bool_classify(hwc, net->fc_words, net->fc_bits, net->fc_weights, net->fc_row_words,
                                net->n_classes, net->fc_weight_counts, net->fc_scale, net->fc_bias, net->scratch);
}
