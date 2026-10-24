"""Validators. `plain` (conv) checks the bit-serial engine's math bit-for-bit;
validator.py (one level up) still covers gated/bitslice. `ternary` checks the packed-word simulator of the
ternary dense engine against a numpy integer reference."""
from .plain import verificar_fidelidad_conv
from .ternary import verificar_bit_exacto
