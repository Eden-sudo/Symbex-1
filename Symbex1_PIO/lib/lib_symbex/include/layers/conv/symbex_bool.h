/*
 * symbex_bool.h -- fully binary convolutional networks: every layer turns bits into bits, with no multiplication.
 *
 * Maps are binary and HWC: pixel (y, x) takes PPX = ceil(C / 32) words, channel c is bit c % 32 of word c / 32;
 * the bits past C are 0. A layer (symbex_bool_conv) is a binary convolution followed by one integer threshold per
 * output channel -- where the trained BatchNorm and sign were folded by the exporter:
 *     raw = (window bits equal to the weight bits) - (bits that differ), over the in-image part of the window
 *     out bit = raw >= threshold
 * The count uses the HAL's whole-layer primitives, in the form the HAL measured as faster on the chip:
 *     XOR form  differences = count(window XOR weights)       (symbex_xor_popcount_rows: pie_swar on the ESP32-S3)
 *     AND form  differences = popcount(window) + popcount(weights in the image) - 2 * count(window AND weights)
 *               (symbex_and_popcount_rows: with the 64 KB table, as on the ESP32)
 * The window is built once per position in an aligned buffer, PACKED: the in_ch bits of its k*k pixels one after the
 * other (with 16 channels, 144 bits fill 5 words instead of 9 half-empty ones; with in_ch a multiple of 32 it is the
 * plain copy of the map words), zero padded to the weight row stride. Positions outside the image are 0 there, and
 * their weights are taken out with the per-pixel weight counts.
 *
 * The other pieces of a network:
 *     symbex_thermometer    uint8 RGB -> bits: `levels` bits per color, bit k = pixel >= (k + 1) * 256 / (levels + 1)
 *     symbex_bool_stem_i8   OPTIONAL int8 first layer, in place of thermometer + binary stem: int8 weights on the
 *                           centered pixels (pixel - 128), one integer threshold per channel -> the same output bits.
 *                           More accurate (ResNet-8: +4.5 points); faster where the chip multiplies int8 in SIMD
 *                           (ESP32-S3 PIE, Cortex-M33 DSP), slower where it does not (ESP32, Hazard3): see
 *                           benchmarks/experiments/chip_kernels/int8_first_bench.py
 *     symbex_bool_or_pool   2x2 pooling of bits = OR of the 4 pixels (max of +-1)
 *     symbex_bool_or        OR of two maps (the shortcut of a block)
 *     symbex_bool_classify  binary weights on the final bits, an int32 scale and bias per class (fixed point,
 *                           int64 products), the first class with the largest logit
 *     symbex_bool_pool_head OPTIONAL classifier in its place: the count of ones per k x k window of every channel,
 *                           int8 weights on those counts, the same int32 scale and bias per class. More accurate
 *                           (ResNet-8 2x2 windows: +2-3 points, benchmarks/experiments/training_gap), 2 KB more flash
 *     symbex_bool_net_run   the whole network: thermometer, stem, blocks (conv OR shortcut), classifier. It is
 *                           static inline so that SYMBEX_MARK(k) of the firmware times every step (as in the MLPs).
 *
 * Weights, thresholds and counts are read directly (RAM or memory-mapped flash): not for AVR.
 */
#ifndef SYMBEX_BOOL_H
#define SYMBEX_BOOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYMBEX_BOOL_MAX_KK 9                 /* kernels up to 3 x 3 */
#define SYMBEX_BOOL_MAX_CELL_WORDS 16        /* per-cell head: up to 512 channels in the final map */

struct SymbexBoolConv {
    uint16_t in_ch, out_ch, kernel, stride;  /* padding = kernel / 2 (the output keeps ceil(in / stride)) */
    uint16_t in_h, in_w, out_h, out_w;
    uint16_t row_words;                      /* words per weight row: ceil(kernel^2 * in_ch / 32), rounded up to 4 */
    const uint32_t* weights;                 /* out_ch rows, PACKED: the in_ch bits of kernel pixel q = kh * k + kw
                                                start at bit q * in_ch of the row; zero padded */
    const int32_t* thresholds;               /* out bit = raw >= thresholds[o] */
    const int16_t* weight_counts;            /* popcount of the weights of output o at kernel pixel q: [o][kh * k + kw] */
};

/* the count form of the conv layers: 0 = what the HAL measured as faster (symbex_hal_prefers_xor), 1 = AND form
   (count(window AND weights) + popcounts), 2 = XOR form (count(window XOR weights) directly). Both give the same bits:
   the tests run every case in both; set it only to compare them. */
extern int symbex_bool_force_mode;

/* one layer. window: row_words words, 16-byte aligned; scratch: 2 * out_ch int32 */
void symbex_bool_conv(const struct SymbexBoolConv* layer, const uint32_t* in, uint32_t* out,
                      uint32_t* window, int32_t* scratch);

/* uint8 RGB image, HWC (h x w x 3) -> bits, HWC, 3 * levels channels (channel = color * levels + k) */
void symbex_thermometer(const uint8_t* img, uint16_t h, uint16_t w, uint8_t levels, uint32_t* out);

/* the optional int8 first layer (the CDT's boolean method with an int8 stem): a k x k conv (k odd, <= 3; stride 1 or
   2; padding k / 2) on the uint8 RGB image, in the centered values x = pixel - 128 (zero padding = x 0):
       dot_o = sum over the in-image window of x * weights[o]       out bit o = dot_o >= thresholds[o]
   The trained per-channel weight scale, BatchNorm and sign are folded by the exporter into that integer threshold. The
   products: symbex_s8_dot_rows (PIE, 16 per instruction) on the ESP32-S3, SMLAD (2 per instruction) where the
   compiler offers the Arm DSP extension (Cortex-M33 of the Pico 2), plain C (4 rows per pass) elsewhere. */
#define SYMBEX_BOOL_STEM_I8_WINDOW 32       /* bytes of the window buffer (k * k * 3 <= 27, zero padded) */
struct SymbexBoolStemI8 {
    uint16_t out_ch, kernel, stride;
    uint16_t in_h, in_w, out_h, out_w;
    uint16_t row_bytes;                      /* bytes per weight row: kernel^2 * 3 rounded up to 4 (28 for 3 x 3) */
    const int8_t* weights;                   /* out_ch rows [kh][kw][color], zero padded; 4-byte aligned */
    const int32_t* thresholds;               /* out bit = dot >= thresholds[o] */
};

/* window: SYMBEX_BOOL_STEM_I8_WINDOW bytes, 16-byte aligned; scratch: out_ch int32. out: HWC bits, as symbex_bool_conv */
void symbex_bool_stem_i8(const struct SymbexBoolStemI8* layer, const uint8_t* img, uint32_t* out, int8_t* window,
                         int32_t* scratch);

/* 2x2 OR pooling: (h x w x ch) -> (h/2 x w/2 x ch) */
void symbex_bool_or_pool(const uint32_t* in, uint32_t* out, uint16_t ch, uint16_t h, uint16_t w);

/* dst |= src over n words */
void symbex_bool_or(uint32_t* dst, const uint32_t* src, uint32_t n_words);

/* the class with the largest scale[o] * raw_o + bias[o] (int64), raw_o = n_bits - 2 * differences with row o.
   bits: n_words words (16-byte aligned for PIE); weights: n_classes rows of row_words words (zero padded);
   weight_counts[o] = popcount of row o; scratch: n_classes int32 */
int symbex_bool_classify(const uint32_t* bits, uint16_t n_words, uint32_t n_bits, const uint32_t* weights,
                         uint16_t row_words, uint16_t n_classes, const int32_t* weight_counts,
                         const int32_t* scale, const int32_t* bias, int32_t* scratch);

/* per-cell head (FOMO-style detection): the integer classifier applied at every cell of the final map (h x w cells,
   ch channels, PPX = ceil(ch / 32) words per cell): class of cell p = the first o with the largest
   scale[o] * raw_o + bias[o] (int64), raw_o = ch - 2 * count(cell XOR row o). weights: n_classes rows of row_words
   words (row_words >= PPX, zero padded); scratch: n_classes int32; out: h * w class indices (class 0 = background).
   ch <= 32 * SYMBEX_BOOL_MAX_CELL_WORDS. */
void symbex_bool_cell_head(const uint32_t* map, uint16_t h, uint16_t w, uint16_t ch, const uint32_t* weights,
                           uint16_t row_words, uint16_t n_classes, const int32_t* scale, const int32_t* bias,
                           int32_t* scratch, uint8_t* out);

/* the optional int8 classifier on pooled counts: the final map (in_h x in_w x channels, HWC bits) is cut into k x k
   windows; count[j][c] = ones of channel c in window j (0..k*k), windows row by row. Then
       dot_o = sum over j, c of weight[o][j][c] * count[j][c]     class = the first o with the largest
       scale[o] * dot_o + bias[o] (int64)
   The counts are built bit-sliced (n_planes = bit length of k * k words per 32 channels, window j at words
   j * PPX .. j * PPX + PPX - 1 of every plane, PPX = ceil(channels / 32)), and the weights are stored as their 8 bit
   planes (two's complement): dot_o = sum over count plane p, weight plane q of
   (q == 7 ? -128 : 1 << q) * 2^p * count(count plane p AND weight plane q), with the HAL's symbex_and_popcount_rows. */
#define SYMBEX_BOOL_POOL_MAX_PLANES 7        /* windows up to 8 x 8 (counts 0..64) */
struct SymbexBoolPoolHead {
    uint16_t window, channels, in_h, in_w;   /* in_h and in_w multiples of window */
    uint16_t n_classes;
    uint16_t row_words;                      /* words per plane row: (in_h / k) * (in_w / k) * PPX, rounded up to 4 */
    uint8_t n_planes;                        /* bit length of window^2 */
    const uint32_t* weight_planes;           /* 8 planes x n_classes rows x row_words: bit q of the int8 weight of
                                                (window j, channel c) at bit 32 * j * PPX + c of the row; zero padded */
    const int32_t *scale, *bias;
};

/* map: the final map (HWC bits); counts: n_planes * row_words words, 16-byte aligned; scratch: 2 * n_classes int32 */
int symbex_bool_pool_head(const struct SymbexBoolPoolHead* head, const uint32_t* map, uint32_t* counts,
                          int32_t* scratch);

/* a checksum of the class grid (FNV-1a over the bytes, 31 bits): what the chip reports per image for a detector, so
   that the exactness check against the simulation covers every cell */
static inline int symbex_bool_grid_checksum(const uint8_t* cells, uint32_t n) {
    uint32_t hsh = 2166136261u;
    for (uint32_t i = 0; i < n; i++) { hsh ^= cells[i]; hsh *= 16777619u; }
    return (int)(hsh & 0x7FFFFFFFu);
}

struct SymbexBoolBlock {
    struct SymbexBoolConv conv;              /* the block's conv: its output bits */
    const struct SymbexBoolConv* shortcut;   /* NULL: the block input itself; else a 1x1 conv (after an OR pool) */
    uint8_t shortcut_pool;                   /* 1: 2x2 OR pool before the shortcut conv (the conv has stride 2) */
};

struct SymbexBoolNet {
    uint16_t in_h, in_w;
    uint8_t levels;                          /* thermometer levels per color */
    struct SymbexBoolConv stem;
    const struct SymbexBoolBlock* blocks;
    uint16_t n_blocks;
    const uint32_t* fc_weights;              /* classes x fc_row_words, in the HWC order of the final map */
    uint16_t fc_row_words, n_classes;
    uint32_t fc_bits;                        /* real bits of the final map: h * w * channels */
    uint16_t fc_words;                       /* words of the final map: h * w * ceil(channels / 32) */
    const int32_t *fc_weight_counts, *fc_scale, *fc_bias;
    uint32_t* maps[4];                       /* four map buffers, each as large as the largest map (16-byte aligned) */
    uint32_t* window;                        /* the largest weight row (16-byte aligned) */
    int32_t* scratch;                        /* 2 * the largest out_ch (and >= n_classes) */
    /* detection (FOMO-style), last so that classifier headers leave them 0: with cells != NULL the fc_* arrays are a
       per-cell head (fc_row_words >= words per cell, fc_bits = channels of the final map) and the network writes the
       class of every cell of the final map to cells, returning symbex_bool_grid_checksum of that grid */
    uint8_t* cells;
    uint16_t cells_h, cells_w;
    /* optional int8 first layer (NULL: the thermometer stem; new fields go here, last, 0 = the behavior before them:
       core/symbex_format.h): with it the network runs
       symbex_bool_stem_i8 on the image instead of thermometer + stem (stem then only gives the shapes: in_ch 3,
       out_ch, kernel, stride, sizes; no weights). maps[0] is not used then. */
    const struct SymbexBoolStemI8* stem_i8;
    /* optional int8 classifier on pooled counts (NULL: the binary classifier fc_*; not with cells): with it the network
       runs symbex_bool_pool_head on the final map, with window as its counts buffer (the exporter sizes it) */
    const struct SymbexBoolPoolHead* pool_head;
};

#ifdef SYMBEX_FINE_PROFILE
/* fine profile (tools/fine_profile.py; -DSYMBEX_FINE_PROFILE): CPU cycles and calls of every part of the layers.
   Without the flag none of this exists and the code is exactly the normal one. */
enum { SYMBEX_PF_THERMOMETER, SYMBEX_PF_WINDOW, SYMBEX_PF_WINDOW_BORDER, SYMBEX_PF_WINDOW_POPCOUNT, SYMBEX_PF_COUNT,
       SYMBEX_PF_THRESHOLD, SYMBEX_PF_OR_POOL, SYMBEX_PF_OR, SYMBEX_PF_CLASSIFY, SYMBEX_PF_STEM_I8, SYMBEX_PF_N };
extern const char* const symbex_pf_names[SYMBEX_PF_N];
extern uint64_t symbex_pf_cycles[SYMBEX_PF_N];
extern uint32_t symbex_pf_calls[SYMBEX_PF_N];
extern uint32_t symbex_pf_counter_cost;      /* cycles an empty measurement carries (subtracted by the tool) */
void symbex_pf_reset(void);
#endif

#ifndef SYMBEX_MARK
#define SYMBEX_MARK(k)
#endif

/* the whole network on one uint8 RGB image (HWC). Steps for SYMBEX_MARK: 0 start, 1 thermometer + stem (or the int8
   stem), 2 + i after block i, 2 + n_blocks after the classifier. */
static inline int symbex_bool_net_run(const struct SymbexBoolNet* net, const uint8_t* img) {
    SYMBEX_MARK(0);
    uint32_t* const* m = net->maps;
    if (net->stem_i8) {
        symbex_bool_stem_i8(net->stem_i8, img, m[1], (int8_t*)net->window, net->scratch);
    } else {
        symbex_thermometer(img, net->in_h, net->in_w, net->levels, m[0]);
        symbex_bool_conv(&net->stem, m[0], m[1], net->window, net->scratch);
    }
    SYMBEX_MARK(1);
    int x = 1, y = 0, p = 2, s = 3;
    for (uint16_t i = 0; i < net->n_blocks; i++) {
        const struct SymbexBoolBlock* b = &net->blocks[i];
        symbex_bool_conv(&b->conv, m[x], m[y], net->window, net->scratch);
        const uint32_t n = (uint32_t)b->conv.out_h * b->conv.out_w * ((b->conv.out_ch + 31) / 32);
        if (b->shortcut) {
            const uint32_t* src = m[x];
            if (b->shortcut_pool) {
                symbex_bool_or_pool(m[x], m[p], b->conv.in_ch, b->conv.in_h, b->conv.in_w);
                src = m[p];
            }
            symbex_bool_conv(b->shortcut, src, m[s], net->window, net->scratch);
            symbex_bool_or(m[y], m[s], n);
        } else {
            symbex_bool_or(m[y], m[x], n);
        }
        const int t = x; x = y; y = t;
        SYMBEX_MARK(2 + i);
    }
    int c;
    if (net->pool_head) {
        c = symbex_bool_pool_head(net->pool_head, m[x], net->window, net->scratch);
    } else if (net->cells) {
        symbex_bool_cell_head(m[x], net->cells_h, net->cells_w, (uint16_t)net->fc_bits, net->fc_weights, net->fc_row_words,
                              net->n_classes, net->fc_scale, net->fc_bias, net->scratch, net->cells);
        c = symbex_bool_grid_checksum(net->cells, (uint32_t)net->cells_h * net->cells_w);
    } else {
        c = symbex_bool_classify(m[x], net->fc_words, net->fc_bits, net->fc_weights, net->fc_row_words,
                                 net->n_classes, net->fc_weight_counts, net->fc_scale, net->fc_bias, net->scratch);
    }
    SYMBEX_MARK(2 + net->n_blocks);
    return c;
}

#ifdef __cplusplus
}
#endif

#endif /* SYMBEX_BOOL_H */
