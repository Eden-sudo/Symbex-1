import importlib.util
import itertools
import random
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "littlebit_int", ROOT / "benchmarks/experiments/tiny_llm/littlebit_int.py"
)
REF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REF)

DRIVER = r'''
#include "tinylm_factor.h"
#include "tinylm_factor.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#define READ(p, f) do { if (scanf(f, (p)) != 1) return 2; } while (0)
int main(void) {
    int n, r, o, sh, cases, k, i, t;
    int8_t a = 9, z = 42;
    uint8_t b = 1;
    int32_t m = 1;
    int64_t y = 43;
    READ(&cases, "%d");
#define INVALID(X,V,U,M,N,A,R,O,S,Z,Y) do { \
    if (tinylm_factor(X,V,U,M,N,A,R,O,S,Z,Y) != -1 || z != 42 || y != 43) return 3; \
} while (0)
    INVALID(NULL,&b,&b,&m,&m,1,1,1,0,&z,&y);
    INVALID(&a,NULL,&b,&m,&m,1,1,1,0,&z,&y);
    INVALID(&a,&b,NULL,&m,&m,1,1,1,0,&z,&y);
    INVALID(&a,&b,&b,NULL,&m,1,1,1,0,&z,&y);
    INVALID(&a,&b,&b,&m,NULL,1,1,1,0,&z,&y);
    INVALID(&a,&b,&b,&m,&m,1,1,1,0,NULL,&y);
    INVALID(&a,&b,&b,&m,&m,1,1,1,0,&z,NULL);
    INVALID(&a,&b,&b,&m,&m,0,1,1,0,&z,&y);
    INVALID(&a,&b,&b,&m,&m,-1,1,1,0,&z,&y);
    INVALID(&a,&b,&b,&m,&m,1,0,1,0,&z,&y);
    INVALID(&a,&b,&b,&m,&m,1,-1,1,0,&z,&y);
    INVALID(&a,&b,&b,&m,&m,1,1,0,0,&z,&y);
    INVALID(&a,&b,&b,&m,&m,1,1,-1,0,&z,&y);
    INVALID(&a,&b,&b,&m,&m,1,1,1,-1,&z,&y);
    INVALID(&a,&b,&b,&m,&m,1,1,1,64,&z,&y);
    INVALID(&a,&b,&b,&m,&m,INT32_MAX/128+1,1,1,0,&z,&y);
    INVALID(&a,&b,&b,&m,&m,1,INT32_MAX/127+1,1,0,&z,&y);
    for (k=0; k<cases; k++) {
        int8_t *x, *zz, *zz2; uint8_t *v, *u; int32_t *m1, *m2; int64_t *yy, *yy2;
        unsigned long long before = 0, after = 0;
        size_t vs, us;
        READ(&n,"%d"); READ(&r,"%d"); READ(&o,"%d"); READ(&sh,"%d");
        vs=(size_t)(n+7)/8; us=(size_t)(r+7)/8;
        x=malloc((size_t)n); v=malloc((size_t)r*vs); u=malloc((size_t)o*us);
        m1=malloc((size_t)r*sizeof(*m1)); m2=malloc((size_t)o*sizeof(*m2));
        zz=malloc((size_t)r+2); yy=malloc(((size_t)o+2)*sizeof(*yy));
        zz2=malloc((size_t)r); yy2=malloc((size_t)o*sizeof(*yy2));
        if (!x || !v || !u || !m1 || !m2 || !zz || !yy || !zz2 || !yy2) return 4;
        for(i=0;i<n;i++) { READ(&t,"%d"); x[i]=(int8_t)t; }
        for(i=0;i<r*(int)vs;i++) { READ(&t,"%d"); v[i]=(uint8_t)t; }
        for(i=0;i<o*(int)us;i++) { READ(&t,"%d"); u[i]=(uint8_t)t; }
        for(i=0;i<r;i++) READ(&m1[i],"%" SCNd32);
        for(i=0;i<o;i++) READ(&m2[i],"%" SCNd32);
        for(i=0;i<n;i++) before=before*31+(uint8_t)x[i];
        for(i=0;i<r*(int)vs;i++) before=before*31+v[i];
        for(i=0;i<o*(int)us;i++) before=before*31+u[i];
        for(i=0;i<r;i++) before=before*31+(uint32_t)m1[i];
        for(i=0;i<o;i++) before=before*31+(uint32_t)m2[i];
        zz[0]=42; zz[r+1]=43; yy[0]=44; yy[o+1]=45;
        t=tinylm_factor(x,v,u,m1,m2,n,r,o,sh,zz+1,yy+1);
        if (tinylm_factor(x,v,u,m1,m2,n,r,o,sh,zz2,yy2) != 0 ||
            memcmp(zz+1,zz2,(size_t)r) || memcmp(yy+1,yy2,(size_t)o*sizeof(*yy))) return 6;
        for(i=0;i<n;i++) after=after*31+(uint8_t)x[i];
        for(i=0;i<r*(int)vs;i++) after=after*31+v[i];
        for(i=0;i<o*(int)us;i++) after=after*31+u[i];
        for(i=0;i<r;i++) after=after*31+(uint32_t)m1[i];
        for(i=0;i<o;i++) after=after*31+(uint32_t)m2[i];
        if (before != after) return 7;
        if (zz[0]!=42 || zz[r+1]!=43 || yy[0]!=44 || yy[o+1]!=45) return 5;
        printf("%d",t);
        for(i=0;i<r;i++) printf(" %d",zz[i+1]);
        for(i=0;i<o;i++) printf(" %" PRId64,yy[i+1]);
        puts("");
        free(x); free(v); free(u); free(m1); free(m2); free(zz); free(yy); free(zz2); free(yy2);
    }
    return 0;
}
'''


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert compiler, "A C compiler (cc, clang, or gcc) is required"
    path = tmp_path_factory.mktemp("factor")
    source = path / "driver.c"
    source.write_text(DRIVER)
    result = subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra",
                             "-Werror", "-pedantic", "-I",
                             str(ROOT / "benchmarks/experiments/tiny_llm/c"),
                             str(source), "-o", str(path / "driver")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    return path / "driver"


def run_cases(driver, cases, tmp_path):
    lines = [str(len(cases))]
    for x, v, u, m1, m2, n, r, o, sh in cases:
        lines.append(" ".join(map(str, [n, r, o, sh, *x, *v, *u, *m1, *m2])))
    result = subprocess.run([str(driver)], input="\n".join(lines), text=True,
                            capture_output=True, cwd=tmp_path, timeout=20)
    assert result.returncode == 0, result.stderr
    rows = result.stdout.splitlines()
    assert len(rows) == len(cases)
    parsed = []
    for row, case in zip(rows, cases):
        values = list(map(int, row.split()))
        r, o = case[6:8]
        assert len(values) == 1 + r + o and values[0] == 0
        assert values[1+r:] == REF.factor_int(*case)
        parsed.append((values[1:1+r], values[1+r:]))
    return parsed


def test_random(driver, tmp_path):
    rng = random.Random(37291)
    cases = []
    for n, r, o, sh in itertools.product((13, 77, 384), (1, 7, 64),
                                         (1, 5), (0, 1, 7, 31, 62, 63)):
        for _ in range(2):
            mult = [-(2**31), 2**31-1, -3, 0, 1, 5]
            cases.append(([rng.randrange(-128, 128) for _ in range(n)],
                          [rng.randrange(256) for _ in range(r*((n+7)//8))],
                          [rng.randrange(256) for _ in range(o*((r+7)//8))],
                          [rng.choice(mult) for _ in range(r)],
                          [rng.choice(mult) for _ in range(o)], n, r, o, sh))
    run_cases(driver, cases, tmp_path)


def test_known(driver, tmp_path):
    cases = [([p], [1], [1], [1], [1], 1, 1, 1, 1)
             for p in (-5, -3, -1, 0, 1, 3, 5)]
    cases += [([x], [v], [1], [m], [2**31-1], 1, 1, 1, 0)
              for x, v, m in [(127, 1, 1), (-127, 1, 1), (127, 1, 2),
                              (-128, 1, 2), (-128, 0, 1), (127, 1, 2**31-1)]]
    result = run_cases(driver, cases, tmp_path)
    assert [z[0] for z, _ in result] == [-3, -2, -1, 0, 1, 2, 3, 127, -127, 127, -127, 127, 127]
    assert result[-1][1] == [127 * (2**31-1)]


def test_padding_and_orientation(driver, tmp_path):
    case = ([1, 2, 4, 8, 16, 32, 64, 1, 2], [0x55, 1, 0xaa, 0],
            [1, 2, 3], [1, 1], [1, 1, 1], 9, 2, 3, 0)
    x, v, u, m1, m2, n, r, o, sh = case
    flipped = (x, [v[0], v[1] ^ 254, v[2], v[3] ^ 254],
               [b ^ 252 for b in u], m1, m2, n, r, o, sh)
    result = run_cases(driver, [case, flipped, case], tmp_path)
    assert result[0] == result[1] == result[2]
    assert result[0] == ([44, -44], [88, -88, 0])


@pytest.mark.parametrize("index,value,name", [
    (5, 0, "n_in"), (5, 2**31, "n_in"), (6, 0, "rank"),
    (6, 2**31, "rank"), (7, -1, "n_out"), (8, -1, "sh1"),
    (8, 64, "sh1"), (8, 1.5, "sh1"), (0, [], "x"),
    (1, [], "v"), (2, [], "u"), (3, [], "m1"), (4, [], "m2"),
    (0, [128], "x"), (0, [0.5], "x"), (1, [256], "v"),
    (2, [-1], "u"), (3, [2**31], "m1"), (4, [-2**31-1], "m2"),
])
def test_invalid_python(index, value, name):
    args = [[1], [1], [1], [1], [1], 1, 1, 1, 0]
    args[index] = value
    with pytest.raises(ValueError, match=name):
        REF.factor_int(*args)
