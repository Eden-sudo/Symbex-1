import pytest
from tiny_runtime_support import compile_c, module, run


def test_rng(tmp_path):
    py = module("sample_int")
    exe = compile_c(tmp_path, r'''
#include <stdio.h>
#include <inttypes.h>
#include "tinylm_rng.h"
int main(void) {
    uint32_t state, zero = 0;
    if (tinylm_rng_next(NULL) != -1 || tinylm_rng_next(&zero) != -1 || zero) return 1;
    while (scanf("%" SCNu32, &state) == 1)
        for (int i = 0; i < 256; ++i) {
            if (tinylm_rng_next(&state)) return 2;
            printf("%" PRIu32 "\n", state);
        }
    return 0;
}
''')
    seeds = [1, 2, 0x80000000, 0xffffffff, 42]
    expected = []
    for state in seeds:
        for _ in range(256):
            state = py.xorshift32(state)
            expected.append(state)
    assert expected[:3] == [270369, 67634689, 2647435461]
    assert list(map(int, run(exe, "\n".join(map(str, seeds))).split())) == expected
    for seed in (0, -1, 1 << 32):
        with pytest.raises(ValueError):
            py.xorshift32(seed)
