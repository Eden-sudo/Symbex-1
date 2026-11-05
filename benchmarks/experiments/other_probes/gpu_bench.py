"""
gpu_bench.py -- quick speed test of every kind of arithmetic on the CUDA GPU, on the same matrix product.

Run:   python gpu_bench.py                  (a few seconds per size)
       python gpu_bench.py --sizes 1024,4096,8192

The same logical problem for every type: C (M x N) = A (M x K) * B (K x N), with M = N = K.
    fp32   plain FP32 (CUDA cores)                      cuBLAS
    tf32   FP32 inputs on the tensor cores (TF32)       cuBLAS
    fp16   half precision on the tensor cores           cuBLAS
    bf16   bfloat16 on the tensor cores                 cuBLAS
    int8   8-bit integers on the tensor cores           torch._int_mm (cuBLASLt)
    1bit   XOR + popcount (__popc), K bits packed in K / 32 words per row, on the CUDA cores, with a
           kernel written here in Triton (the binary result is checked against an exact reference)
Reported: median time and TOPS (multiply-adds counted as 2 operations; for 1 bit, one XOR + count per bit
pair counts the same, so the numbers compare the SAME problem).

Fairness, said up front: the five cuBLAS kernels are NVIDIA's, tuned for years and running on the tensor
cores; the 1-bit kernel is a straightforward Triton kernel on the CUDA cores (the 1-bit tensor core
instructions are not reachable from PyTorch or Triton). So this measures what each type gives you from
Python today, not the ceiling of binary arithmetic on this GPU.
Requires: PyTorch with CUDA; Triton (ships with PyTorch on Linux) for the 1-bit kernel.
"""
import argparse

import torch

try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra.cuda import libdevice as _ld          # Triton 3.x
    except ImportError:
        try:
            from triton.language.extra import libdevice as _ld            # some 3.x builds
        except ImportError:
            _ld = tl.math                                                  # Triton 2.x
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _bin_gemm(a_ptr, b_ptr, c_ptr, M, N, KW, K_BITS,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """C[m, n] = K_BITS - 2 * sum_k popcount(A[m, k] XOR B[n, k]); A: M x KW, B: N x KW int32 words."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k0 in range(0, KW, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(a_ptr + rm[:, None] * KW + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < KW), other=0)
            b = tl.load(b_ptr + rn[:, None] * KW + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < KW), other=0)
            x = a[:, None, :] ^ b[None, :, :]                      # BM x BN x BK
            acc += tl.sum(_ld.popc(x), axis=2)
        c = K_BITS - 2 * acc
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], c, mask=(rm[:, None] < M) & (rn[None, :] < N))

    def bin_gemm(a, b, k_bits, BM=32, BN=32, BK=8):
        M, KW = a.shape
        N = b.shape[0]
        c = torch.empty((M, N), dtype=torch.int32, device=a.device)
        _bin_gemm[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, KW, k_bits, BM=BM, BN=BN, BK=BK)
        return c


def medir(f, rep=20):
    """Median time in ms of f() on the GPU (CUDA events), after a warm-up."""
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    tiempos = []
    for _ in range(rep):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        f()
        e1.record()
        torch.cuda.synchronize()
        tiempos.append(e0.elapsed_time(e1))
    return sorted(tiempos)[len(tiempos) // 2]


def check_binary(dev):
    """The Triton kernel against an exact reference (+-1 matrices in FP32, exact for these sizes)."""
    M, N, KW = 64, 48, 16
    a = torch.randint(-2 ** 31, 2 ** 31, (M, KW), dtype=torch.int64, device=dev).to(torch.int32)
    b = torch.randint(-2 ** 31, 2 ** 31, (N, KW), dtype=torch.int64, device=dev).to(torch.int32)
    bits = torch.arange(32, device=dev)

    def a_pm1(m):
        return (((m.unsqueeze(-1).to(torch.int64) >> bits) & 1).reshape(m.shape[0], -1).float() * 2 - 1)
    ref = (a_pm1(a) @ a_pm1(b).T).round().to(torch.int32)
    return torch.equal(bin_gemm(a, b, KW * 32), ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", "--tamanos", dest="sizes", default="1024,4096",
                    help="M = N = K values, comma separated (multiples of 256); --tamanos is the old name")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("[X] No CUDA GPU visible to PyTorch")
    dev = torch.device("cuda")
    print(f"[*] {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}"
          + (f" | Triton {triton.__version__}" if HAS_TRITON else " | no Triton: the 1-bit kernel is skipped"))
    if HAS_TRITON:
        ok = check_binary(dev)
        print(f"[*] 1-bit kernel check against an exact reference: {'OK' if ok else '[X] WRONG'}")
        if not ok:
            raise SystemExit("[X] the 1-bit kernel gives wrong results: its timings would mean nothing")

    for n in (int(v) for v in args.sizes.split(",")):
        if n % 256:
            raise SystemExit(f"[X] {n}: sizes must be multiples of 256")
        ops = 2 * n ** 3
        res = {}
        a32, b32 = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
        torch.backends.cuda.matmul.allow_tf32 = False
        res["fp32"] = medir(lambda: a32 @ b32)
        torch.backends.cuda.matmul.allow_tf32 = True
        res["tf32"] = medir(lambda: a32 @ b32)
        torch.backends.cuda.matmul.allow_tf32 = False
        a16, b16 = a32.half(), b32.half()
        res["fp16"] = medir(lambda: a16 @ b16)
        ab, bb = a32.bfloat16(), b32.bfloat16()
        res["bf16"] = medir(lambda: ab @ bb)
        try:
            a8 = torch.randint(-128, 128, (n, n), dtype=torch.int8, device=dev)
            b8 = torch.randint(-128, 128, (n, n), dtype=torch.int8, device=dev)
            res["int8"] = medir(lambda: torch._int_mm(a8, b8))
        except (RuntimeError, AttributeError) as e:
            print(f"    [!] int8 skipped: {str(e)[:80]}")
        if HAS_TRITON:
            aw = torch.randint(-2 ** 31, 2 ** 31, (n, n // 32), dtype=torch.int64, device=dev).to(torch.int32)
            bw = torch.randint(-2 ** 31, 2 ** 31, (n, n // 32), dtype=torch.int64, device=dev).to(torch.int32)
            res["1bit"] = medir(lambda: bin_gemm(aw, bw, n))
        print(f"\n  M = N = K = {n}")
        print(f"  {'type':<6}{'ms':>10}{'TOPS':>9}{'vs fp32':>9}   memory of A + B")
        print("  " + "-" * 56)
        bytes_por = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "int8": 1, "1bit": 1 / 8}
        for t, ms in res.items():
            print(f"  {t:<6}{ms:>10.3f}{ops / ms / 1e9:>9.1f}{res['fp32'] / ms:>8.1f}x   "
                  f"{2 * n * n * bytes_por[t] / 2 ** 20:7.2f} MB")
    print("\n  TOPS = 2 * M * N * K / time. The cuBLAS types use the tensor cores (NVIDIA's tuned kernels);"
          "\n  the 1-bit type is a plain Triton kernel on the CUDA cores: see the note at the top of the file.")


if __name__ == "__main__":
    main()
