"""
c_header.py -- the text every exported model_symbex.h shares: the format check and the struct initializers.

The rules that keep an exported header working when lib_symbex grows are in the library's core/symbex_format.h:
structs written by field name, -Wmissing-field-initializers silenced inside the header only, and a format number
checked against the library. Every exporter of the current methods (boolean, bool1d, dense, ternary) uses these
three functions, so the rules live in one place.
"""

FORMAT = 1        # must match SYMBEX_FORMAT in lib_symbex/include/core/symbex_format.h when the CDT writes it


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
