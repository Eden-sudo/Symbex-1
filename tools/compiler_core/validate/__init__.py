"""Validators. `plain` (conv) checks the bit-serial engine's math bit-for-bit. `ternary` checks the packed-word
simulator of the ternary dense engine against a numpy integer reference."""
from .plain import check_conv_fidelity, verificar_fidelidad_conv
from .ternary import verify_bit_exact
