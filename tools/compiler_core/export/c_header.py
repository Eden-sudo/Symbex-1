"""
c_header.py -- the text every exported model_symbex.h shares: the format check and the struct initializers.

The rules that keep an exported header working when lib_symbex grows are in the library's core/symbex_format.h:
structs written by field name, -Wmissing-field-initializers silenced inside the header only, and a format number
checked against the library. Every exporter of the current methods (boolean, bool1d, dense, ternary) uses these
three functions, so the rules live in one place. words() and c_array() pack bits and write the arrays.
"""

import numpy as np

FORMAT = 1        # must match SYMBEX_FORMAT in lib_symbex/include/core/symbex_format.h when the CDT writes it
INT_MIN = -2 ** 31


def prologue():
    """C lines right after the include guard: the format check and the start of the silenced warning."""
    return ("#include \"core/symbex_format.h\"\n"
            f"#if SYMBEX_FORMAT < {FORMAT} || SYMBEX_FORMAT_MIN > {FORMAT}\n"
            f"#error \"model_symbex.h was exported in header format {FORMAT}, which this lib_symbex does not read: "
            "export it again with the CDT (or use the library it was exported for)\"\n"
            "#endif\n"
            "/* fields added to the library after this export start at 0 / NULL (core/symbex_format.h) */\n"
            "#pragma GCC diagnostic push\n"
            "#pragma GCC diagnostic ignored \"-Wmissing-field-initializers\"\n")


def epilogue():
    """C lines right before the include guard's #endif."""
    return "#pragma GCC diagnostic pop\n"


def struct(**fields):
    """A struct initializer by field name, in the order given (write it in declaration order: C++20 asks for it):
    struct(a=1, b="NULL") -> "{.a = 1, .b = NULL}". Values are C text (str) or numbers."""
    return "{" + ", ".join(f".{k} = {v}" for k, v in fields.items()) + "}"


def words(bits, n_words):
    """(..., n) 0/1 -> (..., n_words) uint32, bit i of word i // 32"""
    bits = np.asarray(bits, dtype=np.uint8)
    pad = np.zeros(bits.shape[:-1] + (n_words * 32,), dtype=np.uint8)
    pad[..., :bits.shape[-1]] = bits
    return np.packbits(pad, axis=-1, bitorder="little").view("<u4").reshape(bits.shape[:-1] + (n_words,))


def c_array(f, ctype, name, values, attr=""):
    """writes `static SYMBEX_ROM <ctype> <name>[n]<attr> = {...};` to the file f, 12 values per line"""
    values = np.asarray(values).ravel()
    f.write(f"static SYMBEX_ROM {ctype} {name}[{len(values)}]{attr} = {{")
    for i, v in enumerate(values):
        lit = f"0x{int(v):08X}u" if ctype == "uint32_t" else ("(-2147483647-1)" if int(v) == INT_MIN else str(int(v)))
        f.write(("\n    " if i % 12 == 0 else " ") + lit + ",")
    f.write("\n};\n")
