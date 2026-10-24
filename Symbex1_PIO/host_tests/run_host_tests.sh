#!/bin/sh
# Compiles test_xnor_popcount.c against every HAL variant and runs it on the PC.
#   sh Symbex1_PIO/host_tests/run_host_tests.sh   (CC=clang / CC=gcc to choose the compiler)
set -eu
DIR=$(cd "$(dirname "$0")" && pwd)
LIB="$DIR/../lib/lib_symbex"
CC=${CC:-cc}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
# no -Werror: hal/generico has a pre-existing unused popcount_swar on the native path
FLAGS="-std=c11 -O2 -Wall -Wextra -I$LIB/include"
S3="-DCONFIG_IDF_TARGET_ESP32S3 -DSYMBEX_HAL_PRUEBA_SIN_PIE"

run() {
    name=$1; hal=$2; shift 2
    echo "== $name"
    $CC $FLAGS "$@" "$DIR/test_xnor_popcount.c" "$LIB/src/hal/$hal/symbex_hal_impl.c" -o "$OUT/$name"
    "$OUT/$name"
}

run generico_nativo  generico
run generico_swar    generico -DSYMBEX_POPCOUNT_FORZAR_SWAR
run generico_lut16   generico -DSYMBEX_POPCOUNT_FORZAR_LUT16
run s3_lut16_emu     esp32s3  $S3
run s3_xorq_swar_emu esp32s3  $S3 -DSYMBEX_POPCOUNT_SIN_LUT16
run s3_pie_vmulas_emu esp32s3 $S3 -DSYMBEX_USE_PIE -DSYMBEX_POPCOUNT_SIN_LUT16
run s3_pie_vmulas_emu_lut16 esp32s3 $S3 -DSYMBEX_USE_PIE
run s3_pie_vmulas_emu_vsr_logico esp32s3 $S3 -DSYMBEX_USE_PIE -DSYMBEX_PIE_EMU_VSR_LOGICO
echo "== all variants passed"
