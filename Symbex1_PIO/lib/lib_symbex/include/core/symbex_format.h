/*
 * symbex_format.h -- the format of the headers the CDT exports (model_symbex.h).
 *
 * An exported header is data for this library: it fills the library's structs (SymbexBoolNet, SymbexBool1D,
 * SymbexLayerData, ...). These rules keep a header working when the library grows, without exporting it again:
 *
 *   1. A new field of a struct that exported headers fill goes LAST, and 0 / NULL means the behavior before it
 *      existed (example: SymbexBoolNet.stem_i8 = NULL is the thermometer stem).
 *   2. The CDT writes every struct by field name (.field = value), so the order of the fields does not matter and a
 *      field a header does not name starts at 0 / NULL.
 *   3. The header silences -Wmissing-field-initializers inside itself only (push / pop): C++ warns about a field
 *      left out even with field names, and the firmware builds treat warnings as errors.
 *
 * A change that cannot follow those rules (a field removed, renamed, or given another meaning) raises SYMBEX_FORMAT
 * and, when older headers can no longer be read, SYMBEX_FORMAT_MIN: such a header then stops the build with an
 * #error that says to export it again with the CDT. The tests build headers frozen at earlier formats
 * (tests/headers/) against the current library, in C and C++, so a break shows up before it is pushed.
 *
 * Headers exported before this file existed (no format check, fields by position) must be exported once again.
 */

#ifndef SYMBEX_FORMAT_H
#define SYMBEX_FORMAT_H

#define SYMBEX_FORMAT 1          /* the format this library writes and reads */
#define SYMBEX_FORMAT_MIN 1      /* the oldest format it still reads */

#endif /* SYMBEX_FORMAT_H */
