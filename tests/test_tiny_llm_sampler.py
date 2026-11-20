import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "benchmarks/experiments/tiny_llm/c"
SPEC = importlib.util.spec_from_file_location(
    "tiny_llm_sample_int_test", ROOT / "benchmarks/experiments/tiny_llm/sample_int.py"
)
sampler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampler)

DRIVER = r'''
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinylm_sample.h"
#include "tinylm_sample.h"

int main(void) {
    int n, k, i, j, token, cap;
    uint32_t shift, seed, state;
    int64_t *scores, *copy;
    int *scratch;
    if (tinylm_sample_xorshift32(1) != 270369 ||
        tinylm_sample_xorshift32(270369) != 67634689 ||
        tinylm_sample_xorshift32(67634689) != 2647435461U) return 1;
    if (tinylm_sample_exp_q(0, 0) != 0) return 2;
    while (scanf("%d %d %" SCNu32 " %" SCNu32, &n, &k, &shift, &seed) == 4) {
        cap = k < n ? k : n;
        scores = malloc((size_t)n * sizeof(*scores));
        copy = malloc((size_t)n * sizeof(*copy));
        scratch = malloc((size_t)(cap + 2) * sizeof(*scratch));
        if (!scores || !copy || !scratch) return 3;
        for (i = 0; i < n; ++i) if (scanf("%" SCNd64, &scores[i]) != 1) return 4;
        memcpy(copy, scores, (size_t)n * sizeof(*copy));
        for (i = 0; i < cap + 2; ++i) scratch[i] = 123456;
        state = seed;
#define INVALID(S,N,K,H,P,B,L) do { \
    if (tinylm_sample_topk(S,N,K,H,P,B,L) != -1 || state != seed) return 5; \
    for (j = 0; j < cap + 2; ++j) if (scratch[j] != 123456) return 6; \
} while (0)
        INVALID(NULL,n,k,shift,&state,scratch+1,cap);
        INVALID(scores,0,k,shift,&state,scratch+1,cap);
        INVALID(scores,n,0,shift,&state,scratch+1,cap);
        INVALID(scores,n,-1,shift,&state,scratch+1,cap);
        INVALID(scores,n,k,0,&state,scratch+1,cap);
        INVALID(scores,n,k,shift,NULL,scratch+1,cap);
        INVALID(scores,n,k,shift,&state,NULL,cap);
        INVALID(scores,n,k,shift,&state,scratch+1,cap-1);
        INVALID(scores,n,k,shift,&state,scratch+1,-1);
        state = 0;
        if (tinylm_sample_topk(scores,n,k,shift,&state,scratch+1,cap) != -1 || state != 0) return 7;
        for (j = 0; j < cap + 2; ++j) if (scratch[j] != 123456) return 8;
        state = seed;
        for (i = 0; i < 200; ++i) {
            token = tinylm_sample_topk(scores,n,k,shift,&state,scratch+1,cap);
            if (token < 0 || scratch[0] != 123456 || scratch[cap+1] != 123456 ||
                memcmp(copy,scores,(size_t)n*sizeof(*copy))) return 9;
            printf("%d %" PRIu32 "\n", token, state);
        }
        free(scores); free(copy); free(scratch);
    }
    return 0;
}
'''


def test_cross_language(tmp_path):
    compiler = next((shutil.which(c) for c in ("cc", "clang", "gcc") if shutil.which(c)), None)
    if compiler is None:
        pytest.fail("integer sampler tests require cc, clang, or gcc on PATH")
    source = tmp_path / "driver.c"
    exe = tmp_path / "driver"
    source.write_text(DRIVER)
    result = subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                             "-I", str(HEADER), str(source), "-o", str(exe)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    cases = []
    for n in (7, 4096):
        for k in (1, 5, 40):
            for seed in (1, 0xffffffff):
                for mode in range(3):
                    scores = ([0] * n if mode == 0 else
                              [-(i % 9) for i in range(n)] if mode == 1 else
                              [((i * 31) % 37) - 20 for i in range(n)])
                    if mode == 2:
                        scores[0], scores[2], scores[-1] = (1 << 63) - 1, (1 << 63) - 1, -(1 << 63)
                    cases.append((scores, k, 3, seed))
    cases.extend([([0, 2, 1], 3, 1, 1), ([4], 1, 1, 1), ([0, -100], 2, 1, 1)])
    payload = ''.join(f'{len(s)} {k} {shift} {seed}\n' + ' '.join(map(str, s)) + '\n'
                      for s, k, shift, seed in cases)
    result = subprocess.run([str(exe)], input=payload, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr
    lines = iter(result.stdout.splitlines())
    for scores, k, shift, seed in cases:
        original = scores[:]
        state = seed
        top = set(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k])
        sequence = []
        for _ in range(200):
            token, state = sampler.sample_topk(scores, k, shift, state)
            assert (token, state) == tuple(map(int, next(lines).split()))
            assert token in top
            if k == 1:
                assert token == scores.index(max(scores))
                assert state != seed or len(sequence) > 0
            sequence.append((token, state))
        state = seed
        for expected in sequence:
            token, state = sampler.sample_topk(scores, k, shift, state)
            assert (token, state) == expected
        assert scores == original
    assert next(lines, None) is None


def test_helpers_and_oracles():
    state = 1
    for expected in (270369, 67634689, 2647435461):
        state = sampler.xorshift32(state)
        assert state == expected
    for shift in (1, 3, 0xffffffff):
        for distance, expected in ((0, 65536), (shift - 1, 65536),
                                   (shift, 32768), (2 * shift - 1, 32768),
                                   (16 * shift, 1), (17 * shift, 0),
                                   (0xffffffffffffffff, 0)):
            assert sampler.exp_q(distance, shift) == expected
    assert sampler.exp_q(0, 0) == 0
    assert sampler.sample_topk([0, 2, 1], 3, 1, 1) == (1, 270369)
    state = 1
    for expected in (4, 2, 1):
        token, state = sampler.sample_topk([0] * 5, 5, 1, state)
        assert token == expected
    assert all(sampler.sample_topk([0, -100], 2, 1, seed)[0] == 0
               for seed in range(1, 100))


def test_topk_delegates_to_shared_sampler(monkeypatch):
    calls = []

    def sample(*args):
        calls.append(args)
        return 1, 270369

    monkeypatch.setattr(sampler, "sample", sample)
    assert sampler.sample_topk([0, 2, 1], 3, 1, 1) == (1, 270369)
    assert calls == [([0, 2, 1], 3, 1, 65536, 1)]


def test_invalid_and_independent_streams():
    scores = [1, 2, 3]
    for args, error, name in [
        ((None, 1, 1, 1), TypeError, 'scores'), (([], 1, 1, 1), ValueError, 'scores'),
        (([True], 1, 1, 1), TypeError, 'scores'), (([1 << 63], 1, 1, 1), ValueError, 'scores'),
        (([-(1 << 63) - 1], 1, 1, 1), ValueError, 'scores'),
        ((scores, True, 1, 1), TypeError, 'k'), ((scores, 0, 1, 1), ValueError, 'k'),
        ((scores, 1, False, 1), TypeError, 'shift'), ((scores, 1, 0, 1), ValueError, 'shift'),
        ((scores, 1, 1, True), TypeError, 'state'), ((scores, 1, 1, 0), ValueError, 'state'),
    ]:
        with pytest.raises(error, match=name):
            sampler.sample_topk(*args)
        assert scores == [1, 2, 3]
    for args, error, name in [((-1, 1), ValueError, 'distance'),
                              ((1 << 64, 1), ValueError, 'distance'),
                              ((True, 1), TypeError, 'distance'),
                              ((0, 1 << 32), ValueError, 'shift')]:
        with pytest.raises(error, match=name):
            sampler.exp_q(*args)
    def run(seed):
        out = []
        for _ in range(50):
            token, seed = sampler.sample_topk(scores, 3, 1, seed)
            out.append((token, seed))
        return out
    a, b = 1, 999
    isolated_a, isolated_b = run(a), run(b)
    for ea, eb in zip(isolated_a, isolated_b):
        ta, a = sampler.sample_topk(scores, 3, 1, a)
        tb, b = sampler.sample_topk(scores, 3, 1, b)
        assert (ta, a) == ea and (tb, b) == eb
