import subprocess
from pathlib import Path


def test_random_shapes_exact_against_reference(tmp_path):
    source = tmp_path / "test.c"
    source.write_text(r'''
#include "tinylm_factor_popcount.h"
#include <string.h>
static uint32_t rng = 12345;
static uint32_t next(void) { rng = rng * 1664525u + 1013904223u; return rng; }
int main(void) {
    int8_t x[39], z[73], refz[73];
    uint8_t v[73*5], u[11*10], planes[8*10+2];
    int32_t m1[73], m2[11];
    int64_t y[11], refy[11];
    const int ranks[] = {1, 7, 8, 9, 31, 64, 73};
    const int shifts[] = {0, 1, 7, 20, 31, 63};
    const int32_t mult[] = {INT32_MIN, INT32_MAX, -3, 0, 1, 19};
    for (int k = 0; k < 300; ++k) {
        int n = 1 + (int)(next()%39), r = ranks[k%7], o = 1 + (int)(next()%11);
        int sh = shifts[k%6], vs = (n+7)/8, us = (r+7)/8;
        for (int i=0; i<n; ++i) x[i] = (int8_t)((int)(next()%256)-128);
        for (int i=0; i<r*vs; ++i) v[i] = (uint8_t)(next() >> 16);
        for (int i=0; i<o*us; ++i) u[i] = (uint8_t)(next() >> 16);
        for (int i=0; i<r; ++i) m1[i] = mult[next()%6];
        for (int i=0; i<o; ++i) m2[i] = mult[next()%6];
        memset(planes, 0xa5, sizeof(planes));
        if (tinylm_factor(x,v,u,m1,m2,n,r,o,sh,refz,refy)) return 1;
        if (tinylm_factor_popcount(x,v,u,m1,m2,n,r,o,sh,z,y,planes+1,(size_t)8*us)) return 2;
        if (memcmp(z,refz,(size_t)r) || memcmp(y,refy,(size_t)o*sizeof(*y))) return 3;
        if (planes[0]!=0xa5 || planes[8*us+1]!=0xa5) return 4;
        y[0] = 123;
        if (tinylm_factor_popcount(x,v,u,m1,m2,n,r,o,sh,z,y,planes,(size_t)8*us-1)!=-1 || y[0]!=123) return 5;
    }
    if (tinylm_factor_popcount(x,v,u,m1,m2,0,1,1,0,z,y,planes,sizeof(planes))!=-1) return 6;
    if (tinylm_factor_popcount(x,v,u,m1,m2,1,1,1,0,z,y,NULL,0)!=-1) return 7;
    return 0;
}
''')
    include = Path(__file__).resolve().parents[1] / "benchmarks/experiments/tiny_llm/c"
    executable = tmp_path / "test"
    subprocess.run(["/usr/bin/clang", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-fsanitize=undefined", "-fno-sanitize-recover=all", "-I", str(include), str(source),
                    "-o", str(executable)], check=True, capture_output=True, timeout=15)
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=5)
