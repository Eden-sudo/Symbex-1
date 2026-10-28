/*
 * symbex_bool1d.h -- binary 1D convolutions for time series (sensor data): every weight and activation is one bit.
 *
 * Maps: ONE continuous bit stream, time-major: the bit of channel c at step t is stream bit t * ch + c (stream bit i =
 * bit i % 32 of word i / 32, least significant first). No padding between steps: any number of channels costs exactly
 * its bits. A buffer of n bits needs ceil(n / 32) + 1 words: reading a window may touch one word past the last one it
 * uses (those bits are masked out).
 *
 * Convolution: no padding (out = (in_len - kernel) / stride + 1 steps), any kernel, stride and channel count, followed
 * by an OR pool of `pool` consecutive outputs (pool = 1: none), fused: the outputs are ORed straight into the pooled map.
 * The bits are counted by the HAL row functions, the same as the 2D layers (symbex_bool.h), in the AND or the XOR form.
 *
 * The classifier is symbex_bool_classify (symbex_bool.h) on the final stream: its rows hold the weights in the same
 * time-major order.
 */
#ifndef SYMBEX_BOOL1D_H
#define SYMBEX_BOOL1D_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SymbexBool1D {
    uint16_t in_ch, out_ch, kernel, stride, pool;
    uint16_t in_len, out_len;               /* out_len = ((in_len - kernel) / stride + 1) / pool (rounded down) */
    uint16_t row_words;                     /* words per weight row: ceil(kernel * in_ch / 32), rounded up to 4 */
    const uint32_t* weights;                /* out_ch rows: row bit k * in_ch + c = the weight of step k, channel c;
                                               zero padded */
    const int32_t* thresholds;              /* out bit = raw >= thresholds[o]; binary input: raw = agreements -
                                               disagreements over the kernel * in_ch bits; uint8 input: raw =
                                               sum of x * (+1 / -1) */
    const int32_t* weight_counts;           /* popcount of row o (binary input, AND form); NULL for the uint8 layer */
};

/* binary input. window: 2 * row_words words, 16-byte aligned (two windows where the bits are counted inline);
   scratch: out_ch int32.
   out: ceil(out_len * out_ch / 32) + 1 words, overwritten. */
void symbex_bool1d_conv(const struct SymbexBool1D* layer, const uint32_t* in, uint32_t* out, uint32_t* window,
                        int32_t* scratch);

/* first layer: uint8 input, time-major (in_len x in_ch bytes), 1-bit weights; signed data enters offset by 128 (the
   offset folds into the thresholds). workspace: row_words * 32 bytes, 16-byte aligned; scratch: out_ch int32. */
void symbex_bool1d_conv_u8(const struct SymbexBool1D* layer, const uint8_t* in, uint32_t* out, uint8_t* workspace,
                           int32_t* scratch);

#ifdef __cplusplus
}
#endif

#endif
