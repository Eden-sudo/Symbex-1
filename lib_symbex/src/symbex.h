/**
 * @file symbex.h
 * @brief Master file for the SYMBEX-1 V2 library.
 *  
 * Ultra-lightweight pure C inference engine for Edge-AI architectures.
 * Supports Multi-Bit (Bitslice) and Conditional (Block-Gated) networks.
 */

#ifndef SYMBEX_H
#define SYMBEX_H

// 1. Data types and inert containers (AoS)
#include "symbex_types.h"

// 2. Hardware Abstraction and Math Core (XNOR + Popcount)
#include "symbex_core.h"

// 3. Linear Engine (Standard Feed-Forward)
#include "symbex_bitslice.h"

// 4. Conditional Engine (Block-Gated with Early Exit)
#include "symbex_gated.h"

#endif // SYMBEX_H
