/*
 * symbex_bool_vertical.h -- the vertical (bit-sliced) engine of the fully binary networks (symbex_bool.h): the same
 * network, the same exported data, the same output bits, computed 128 output pixels at a time.
 *
 * The standard engine (symbex_bool_net_run) counts one window at a time: copy the window, count its bits against
 * every weight row. The vertical engine turns the loops around (docs/vertical.md has the derivation):
 *     maps       channel planes: bit p of plane c = pixel p (row-major) of channel c. Plane c starts at word
 *                4 + c * PS of the map buffer, PS = symbex_bool_vplane_stride(h, w); the 4 words before and after
 *                every plane and the bits past h * w are 0 (the zero padding of the convolution)
 *     planes     per layer, the input shifted by every kernel offset (kh - pad, kw - pad): K = k * k * cin planes,
 *                128 output pixels ("lanes") per 128-bit tile. Stride 2: the input split first into its 4 phases
 *     pcx        the bits of every window, counted vertically (Harley-Seal carry-save adders, 128 lanes at a time)
 *     count      per output channel, the planes of the weights that are 1 summed the same way (a list of plane
 *                offsets built on the chip from the 1-bit weights, one byte of weights per step; or built once
 *                into RAM by symbex_bool_vertical_prepare, optional: more RAM, faster)
 *     threshold  out bit = 2 count - pcx >= E: E is one value per output channel and border kind (the windows that
 *                lose the same kernel pixels at the image border), built on the chip from the exported thresholds
 *                and per-pixel weight counts; a bit-sliced 3-operand add gives the 128 output bits at once
 * The head (classifier, pooled int8 classifier or per-cell head) runs on the final map converted back to HWC:
 * symbex_bool_classify / symbex_bool_pool_head / symbex_bool_cell_head, the same functions as the standard engine.
 * An int8 stem (stem_i8) runs as in the standard engine (symbex_bool_stem_i8, into the HWC buffer), its output then
 * turned into channel planes (symbex_bool_vstem_i8): the blocks after it run vertically.
 *
 * Shapes it runs (symbex_bool_vertical_check tells which part of a network is not): kernel 1 or 3; stride 1 (same
 * size) or 2 (3 x 3, input = 2 x output); map widths that divide 32 (1, 2, 4, 8, 16, 32) and h * w a multiple of 32,
 * output maps up to 32768 pixels (256 tiles of 128);
 * k * k * in_ch a multiple of 8 and below 2048; thermometer with 3 * levels <= 32 bits (an int8 stem: its output
 * width divides 32 and h * w is a multiple of 32). Exact: every output bit equals
 * symbex_bool_net_run's (tests/run_tests.py, and chip_test.py --engine vertical on the board).
 * On the ESP32-S3 the counts run on PIE (128-bit logic); elsewhere the same steps in portable C.
 * Separate instances share nothing: the counting state of a call lives on its stack and in its own work block
 * (run_tests.py runs two instances at the same time in two threads). On the S3, two tasks on one core also need the
 * RTOS to save the PIE q registers on a task switch.
 */
#ifndef SYMBEX_BOOL_VERTICAL_H
#define SYMBEX_BOOL_VERTICAL_H

#include <stdint.h>
#include <stddef.h>
#include "layers/conv/symbex_bool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* the work buffers of the vertical engine, carved from one aligned block by symbex_bool_vertical_init */
struct SymbexBoolVertical {
    uint32_t* maps[4];                       /* channel-plane maps (the rotation of symbex_bool_net_run) */
    uint32_t* phases;                        /* stride 2: the 4 phases of every input plane */
    uint32_t *xq, *ap, *km, *bt;             /* shifted planes, 2048 - pcx, border-kind masks, threshold planes */
    uint32_t *ba, *bb, *cb, *pxb;            /* the counters' planes: eights, their next level, count and pcx bits */
    uint16_t* list;                          /* the plane list of one output channel (list[-1] = its start) */
    uint8_t *tile_type, *tile_rep, *tile_kinds, *tile_nkinds;
    void* list_table;                        /* 256 entries: one byte of weights -> its part of a list */
    uint32_t* hwc;                           /* the final map in HWC, for the head */
    const struct SymbexBoolNet* net;         /* the network of symbex_bool_vertical_init */
    void* prep;                              /* the prepared lists and thresholds (NULL: built while running) */
    int prep_compact;                        /* 1: prep holds the compact thresholds (prepare_compact) */
};

/* words between two planes of an h x w map (the plane data rounded up to a 128-bit tile, plus 4 zero words) */
static inline uint32_t symbex_bool_vplane_stride(uint32_t h, uint32_t w) { return ((h * w / 32 + 3) & ~3u) + 4; }

/* NULL if the vertical engine runs this network, else what it does not support (for a message) */
const char* symbex_bool_vertical_check(const struct SymbexBoolNet* net);

/* bytes of the work block (0: the network is not supported) */
size_t symbex_bool_vertical_work_bytes(const struct SymbexBoolNet* net);

/* carves `work` (16-byte aligned, `bytes` long) into v and zeroes it; 0 = ready, -1 = not supported or too small */
int symbex_bool_vertical_init(const struct SymbexBoolNet* net, void* work, size_t bytes, struct SymbexBoolVertical* v);

/* OPTIONAL, more RAM for more speed: every conv's plane lists and threshold planes, built once from the network's data
   (the same bits) into `buf` (16-byte aligned), so that each inference only reads them instead of building them.
   After symbex_bool_vertical_init. prepared_bytes: the bytes buf needs (they depend on how many weights are 1).
   prepare: 0 = done (v uses buf from now on; buf must stay alive), -1 = buf too small or not aligned. Without it the
   engine builds the lists and thresholds while it runs: less RAM, slower. */
size_t symbex_bool_vertical_prepared_bytes(struct SymbexBoolVertical* v);
int symbex_bool_vertical_prepare(struct SymbexBoolVertical* v, void* buf, size_t bytes);
/* OPTIONAL, the same with compact thresholds: the lists prepared as above, the thresholds kept as one value per output
   channel and border kind, their planes built while running. About half the RAM of prepare, a little slower (ResNet-8
   on the ESP32-S3: 90 KB instead of 179 KB, 7.7 ms instead of 6.4 ms; the default engine 10.3 ms). Same results. */
size_t symbex_bool_vertical_prepared_compact_bytes(struct SymbexBoolVertical* v);
int symbex_bool_vertical_prepare_compact(struct SymbexBoolVertical* v, void* buf, size_t bytes);

/* the pieces of a network, on channel-plane maps */
void symbex_bool_vthermometer(const uint8_t* img, uint16_t h, uint16_t w, uint8_t levels, uint32_t* out);
/* the int8 stem (net->stem_i8) on the image: symbex_bool_stem_i8 into v->hwc, then its channel planes into out */
void symbex_bool_vstem_i8(const struct SymbexBoolNet* net, const uint8_t* img, uint32_t* out,
                          struct SymbexBoolVertical* v);
void symbex_bool_vconv(const struct SymbexBoolConv* layer, const uint32_t* in, uint32_t* out,
                       struct SymbexBoolVertical* v);
void symbex_bool_vor_pool(const uint32_t* in, uint32_t* out, uint16_t ch, uint16_t h, uint16_t w);
void symbex_bool_vor(uint32_t* dst, const uint32_t* src, uint16_t ch, uint16_t h, uint16_t w);
/* the head on the final map: its class (classifier) or the grid checksum (per-cell head), as symbex_bool_net_run */
int symbex_bool_vhead(const struct SymbexBoolNet* net, const uint32_t* map, uint16_t ch, uint16_t h, uint16_t w,
                      struct SymbexBoolVertical* v);

/* the whole network on one uint8 RGB image (HWC), the same result as symbex_bool_net_run(net, img). v: from
   symbex_bool_vertical_init. Static inline, as symbex_bool_net_run, so that the firmware's SYMBEX_MARK(k) times the
   same steps: 0 start, 1 thermometer + stem (or the int8 stem), 2 + i after block i, 2 + n_blocks after the head. */
static inline int symbex_bool_net_run_vertical(const struct SymbexBoolNet* net, struct SymbexBoolVertical* v,
                                               const uint8_t* img) {
    SYMBEX_MARK(0);
    uint32_t* const* m = v->maps;
    if (net->stem_i8) {
        symbex_bool_vstem_i8(net, img, m[1], v);
    } else {
        symbex_bool_vthermometer(img, net->in_h, net->in_w, net->levels, m[0]);
        symbex_bool_vconv(&net->stem, m[0], m[1], v);
    }
    SYMBEX_MARK(1);
    int x = 1, y = 0, p = 2, s = 3;
    const struct SymbexBoolConv* last = &net->stem;
    for (uint16_t i = 0; i < net->n_blocks; i++) {
        const struct SymbexBoolBlock* b = &net->blocks[i];
        const struct SymbexBoolConv* c = &b->conv;
        symbex_bool_vconv(c, m[x], m[y], v);
        if (b->shortcut) {
            const uint32_t* src = m[x];
            if (b->shortcut_pool) {
                symbex_bool_vor_pool(m[x], m[p], c->in_ch, c->in_h, c->in_w);
                src = m[p];
            }
            symbex_bool_vconv(b->shortcut, src, m[s], v);
            symbex_bool_vor(m[y], m[s], c->out_ch, c->out_h, c->out_w);
        } else {
            symbex_bool_vor(m[y], m[x], c->out_ch, c->out_h, c->out_w);
        }
        const int t = x; x = y; y = t;
        last = c;
        SYMBEX_MARK(2 + i);
    }
    const int r = symbex_bool_vhead(net, m[x], last->out_ch, last->out_h, last->out_w, v);
    SYMBEX_MARK(2 + net->n_blocks);
    return r;
}

#ifdef __cplusplus
}
#endif

#endif /* SYMBEX_BOOL_VERTICAL_H */
