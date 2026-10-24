/*
 * symbex.h -- ARCHIVO PUENTE, tu API publica.
 * #include "symbex.h" -- eso es todo lo que hace falta.
 */

#ifndef SYMBEX_H
#define SYMBEX_H

#include "hal/symbex_hal.h"
#include "ops/symbex_mac_1bit.h"
#include "layers/spatial/symbex_conv2d.h"
#include "layers/vector/symbex_dense.h"
#include "layers/vector/symbex_gated.h"
#include "layers/vector/symbex_ternary.h"
// #include "layers/spatial/symbex_conv1d.h"  -- pendiente
// #include "layers/spatial/symbex_pool.h"    -- pendiente

#endif // SYMBEX_H
