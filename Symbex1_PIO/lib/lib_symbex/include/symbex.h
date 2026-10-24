/*
 * symbex.h -- the public API of SYMBEX-1 in one include.
 * #include "symbex.h" is all an application needs.
 *
 * Layout of the library:
 *   core/     types, shared constants and the network runners (dense and convolutional)
 *   hal/      the hardware interface (generic C, ESP32-S3 PIE): popcount, XNOR and AND counts
 *   ops/      bit operations, bit packing, 1-bit multiply-accumulate and the AND arithmetic
 *   layers/   dense (fully connected) and conv (convolution, BPR input encoding, binary pooling) layers
 */

#ifndef SYMBEX_H
#define SYMBEX_H

/* core */
#include "core/symbex_types.h"
#include "core/symbex_core.h"
#include "core/network/symbex_network_dense.h"
#include "core/network/symbex_network_conv.h"

/* hardware interface */
#include "hal/symbex_hal.h"

/* operations */
#include "ops/symbex_bitops.h"
#include "ops/symbex_bitpack.h"
#include "ops/symbex_mac_1bit.h"
#include "ops/symbex_and.h"

/* layers */
#include "layers/dense/symbex_dense.h"
#include "layers/dense/symbex_gated.h"
#include "layers/dense/symbex_ternary.h"
#include "layers/conv/symbex_conv2d.h"
#include "layers/conv/symbex_bpr.h"
#include "layers/conv/symbex_binary_pool.h"
/* pending: layers/conv/symbex_conv1d.h */

#endif /* SYMBEX_H */
