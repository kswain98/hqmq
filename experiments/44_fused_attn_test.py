"""Correctness + benchmark for the fused HQMQ-Attention Triton kernel.

Compares:
  1. PyTorch SDPA on fp16 K/V (baseline fp16 attention)
  2. fused_hqmq_attention_torch (reference: decode K/V then SDPA)
  3. fused_hqmq_attention_triton (fused decode + attention)

Reports max abs diff between (2) and (3) for correctness, and wall-clock
latency for all three at a Mistral-7B-like prefill workload.
"""

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quantizers.hqmq_attention import (
    _decode_kv_torch, fused_hqmq_attention_torch, fused_hqmq_attention_triton,
    TRITON_AVAILABLE,
)


def make_packed_kv(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype):
    """Build random packed-HQMQ K and V plus the joint codebook."""
    idx = torch.randint(0, K_size, (B, H_kv, T_kv, n_chunks), dtype=torch.int32, device=device)
    rq = torch.randint(1, r_qmax + 1, (B, H_kv, T_kv, n_chunks), dtype=torch.int32, device=device)
    rs = (torch.rand((B, H_kv, T_kv), dtype=dtype, device=device) + 0.5)
    joint = torch.randn((H_kv, K_size, 4), dtype=dtype, device=device)
    # Normalize each codeword to unit (HQMQ codewords lie on S^3)
    joint = joint / joint.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return idx, rq, rs, joint


def bench(fn, *args, warmup=5, repeats=20):
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main():
    # Mistral-7B-like prefill workload: B=1, H_q=32, H_kv=8 (GQA), T=2048, d_h=128
    # Reduce H to 4 and T to 256 for a fast correctness test
    B, H_q, H_kv = 1, 4, 1  # n_kv_groups = 4
    T_q, T_kv = 128, 128
    d_h = 32  # 8 chunks of 4
    n_chunks = d_h // 4
    K_size = 24 * 24  # s24 codebook
    r_qmax = 7  # 3-bit radius

    device = "cuda"
    dtype = torch.float32  # fp32 for tight numerical check

    print(f"=== Correctness test (B={B}, H_q={H_q}, H_kv={H_kv}, T={T_q}, d_h={d_h}, K={K_size}) ===")
    torch.manual_seed(0)
    Q = torch.randn((B, H_q, T_q, d_h), dtype=dtype, device=device) * 0.3
    K_idx, K_rq, K_rs, joint_K = make_packed_kv(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)
    V_idx, V_rq, V_rs, joint_V = make_packed_kv(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)

    O_ref = fused_hqmq_attention_torch(
        Q, K_idx, K_rq, K_rs, joint_K, V_idx, V_rq, V_rs, joint_V,
        r_qmax=r_qmax, causal=True,
    )
    print(f"  Reference O shape: {O_ref.shape}, dtype: {O_ref.dtype}")
    print(f"  Reference O[:3, :3, 0, :4]: {O_ref[0, 0, 0, :4]}")

    if not TRITON_AVAILABLE:
        print("  Triton not available — skipping fused-kernel test.")
        return

    O_triton = fused_hqmq_attention_triton(
        Q, K_idx, K_rq, K_rs, joint_K, V_idx, V_rq, V_rs, joint_V,
        r_qmax=r_qmax, causal=True, block_q=64, block_kv=64,
    )
    print(f"  Triton O shape: {O_triton.shape}, dtype: {O_triton.dtype}")
    print(f"  Triton O[:3, :3, 0, :4]: {O_triton[0, 0, 0, :4]}")

    diff = (O_ref - O_triton).abs()
    print(f"  Max abs diff: {diff.max().item():.4e}")
    print(f"  Mean abs diff: {diff.mean().item():.4e}")
    print(f"  Relative norm: {(diff.norm() / O_ref.norm()).item():.4e}")
    print()

    # Benchmark
    print(f"=== Benchmark: Mistral-like prefill (B=1, H_q=32, H_kv=8, T=2048, d_h=128) ===")
    B, H_q, H_kv = 1, 32, 8
    T_q, T_kv = 2048, 2048
    d_h = 128
    n_chunks = d_h // 4
    K_size = 24 * 192  # s192 codebook (largest in our paper)
    r_qmax = 15
    dtype = torch.float16

    Q = torch.randn((B, H_q, T_q, d_h), dtype=dtype, device=device)
    K_idx, K_rq, K_rs, joint_K = make_packed_kv(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)
    V_idx, V_rq, V_rs, joint_V = make_packed_kv(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)

    # 1. fp16 SDPA baseline (dense fp16 K/V)
    K_dense = _decode_kv_torch(K_idx, K_rq, K_rs, joint_K, r_qmax).repeat_interleave(H_q // H_kv, dim=1)
    V_dense = _decode_kv_torch(V_idx, V_rq, V_rs, joint_V, r_qmax).repeat_interleave(H_q // H_kv, dim=1)
    scale = 1.0 / math.sqrt(d_h)
    sdpa_fn = lambda Q_, K_, V_: F.scaled_dot_product_attention(Q_, K_, V_, is_causal=True, scale=scale)
    t_sdpa = bench(sdpa_fn, Q, K_dense, V_dense, warmup=5, repeats=20)
    print(f"  fp16 SDPA (dense K/V): {t_sdpa*1e3:.3f} ms/call")

    # 2. Reference: decode-then-SDPA
    t_ref = bench(fused_hqmq_attention_torch, Q, K_idx, K_rq, K_rs, joint_K,
                  V_idx, V_rq, V_rs, joint_V, r_qmax, warmup=3, repeats=10)
    print(f"  Decode-then-SDPA (fake-quant pipeline): {t_ref*1e3:.3f} ms/call")

    # 3. Fused Triton kernel
    if TRITON_AVAILABLE:
        try:
            # Warmup + correctness at small blocks
            O_ref_bench = fused_hqmq_attention_torch(
                Q, K_idx, K_rq, K_rs, joint_K, V_idx, V_rq, V_rs, joint_V, r_qmax, causal=True)
            O_tri = fused_hqmq_attention_triton(
                Q, K_idx, K_rq, K_rs, joint_K, V_idx, V_rq, V_rs, joint_V,
                r_qmax=r_qmax, causal=True, block_q=16, block_kv=16)
            print(f"  Triton vs Ref max abs diff at fp16 bench scale: {(O_ref_bench - O_tri).abs().max().item():.4e}")
            best_t = None
            best_cfg = None
            # Try block sizes x num_warps x num_stages combinations
            configs = []
            for bq in [128, 64, 32]:
                for bk in [128, 64, 32]:
                    for nw in [4, 8]:
                        for ns in [1, 2, 3]:
                            configs.append((bq, bk, nw, ns))
            for bq, bk, nw, ns in configs:
                try:
                    def fn():
                        return fused_hqmq_attention_triton(
                            Q, K_idx, K_rq, K_rs, joint_K, V_idx, V_rq, V_rs, joint_V,
                            r_qmax=r_qmax, causal=True, block_q=bq, block_kv=bk,
                            num_warps=nw, num_stages=ns)
                    _ = fn(); torch.cuda.synchronize()
                    t = bench(fn, warmup=3, repeats=10)
                    print(f"  bQ={bq:3d} bKV={bk:3d} warps={nw} stages={ns}: {t*1e3:.3f} ms/call")
                    if best_t is None or t < best_t:
                        best_t = t
                        best_cfg = (bq, bk, nw, ns)
                except Exception as e:
                    if "shared memory" not in str(e):
                        print(f"  bQ={bq} bKV={bk} warps={nw} stages={ns} FAILED: {str(e)[:120]}")
            if best_t is not None:
                print(f"  Best fused-kernel (prefill): {best_t*1e3:.3f} ms/call (cfg={best_cfg})")
                print(f"     vs fp16 SDPA: {t_sdpa/best_t:.2f}x")
                print(f"     vs fake-quant pipeline: {t_ref/best_t:.2f}x speedup")
        except Exception as e:
            print(f"  Triton kernel FAILED: {str(e)[:400]}")

    # ============================================================
    # Decode-step benchmark: T_q=1 (production inference setting)
    # ============================================================
    # Prefill at longer contexts
    for T_pref in [4096, 8192]:
        print()
        print(f"=== Prefill ({T_pref}-token, d_h=128, s192) ===")
        B, H_q, H_kv = 1, 32, 8
        T_q = T_kv = T_pref
        d_h = 128
        n_chunks = d_h // 4
        K_size = 24 * 192
        r_qmax = 15
        dtype = torch.float16

        Q = torch.randn((B, H_q, T_q, d_h), dtype=dtype, device=device)
        K_idx, K_rq, K_rs, joint_K = make_packed_kv(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)
        V_idx, V_rq, V_rs, joint_V = make_packed_kv(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)

        K_dense = _decode_kv_torch(K_idx, K_rq, K_rs, joint_K, r_qmax).repeat_interleave(H_q // H_kv, dim=1)
        V_dense = _decode_kv_torch(V_idx, V_rq, V_rs, joint_V, r_qmax).repeat_interleave(H_q // H_kv, dim=1)
        scale = 1.0 / math.sqrt(d_h)

        t_sdpa_pref = bench(sdpa_fn, Q, K_dense, V_dense, warmup=3, repeats=10)
        print(f"  fp16 SDPA (dense K/V):                  {t_sdpa_pref*1e3:.3f} ms/call")
        try:
            t_ref_pref = bench(fused_hqmq_attention_torch, Q, K_idx, K_rq, K_rs, joint_K,
                              V_idx, V_rq, V_rs, joint_V, r_qmax, warmup=2, repeats=5)
            print(f"  Decode-then-SDPA (fake-quant pipeline): {t_ref_pref*1e3:.3f} ms/call")
        except Exception as e:
            t_ref_pref = None
            print(f"  Decode-then-SDPA FAILED: {str(e)[:100]}")

        best_t = None
        best_cfg = None
        for bq, bk, nw, ns in [(128, 64, 4, 1), (128, 32, 4, 2), (64, 64, 4, 1)]:
            try:
                def fn():
                    return fused_hqmq_attention_triton(
                        Q, K_idx, K_rq, K_rs, joint_K, V_idx, V_rq, V_rs, joint_V,
                        r_qmax=r_qmax, causal=True, block_q=bq, block_kv=bk,
                        num_warps=nw, num_stages=ns)
                _ = fn(); torch.cuda.synchronize()
                t = bench(fn, warmup=2, repeats=5)
                if best_t is None or t < best_t:
                    best_t = t
                    best_cfg = (bq, bk, nw, ns)
            except Exception:
                pass
        if best_t is not None:
            print(f"  Fused HQMQ-Attention (best):            {best_t*1e3:.3f} ms/call (cfg={best_cfg})")
            print(f"     vs fp16 SDPA:           {t_sdpa_pref/best_t:.2f}x")
            if t_ref_pref:
                print(f"     vs fake-quant pipeline: {t_ref_pref/best_t:.2f}x speedup")

    for T_kv_test in [4096, 16384, 32768]:
        print()
        print(f"=== Decode-step (T_q=1, T_kv={T_kv_test}, d_h=128, s192) ===")
        B, H_q, H_kv = 1, 32, 8
        T_q, T_kv = 1, T_kv_test
        d_h = 128
        n_chunks = d_h // 4
        K_size = 24 * 192
        r_qmax = 15
        dtype = torch.float16

        Q = torch.randn((B, H_q, T_q, d_h), dtype=dtype, device=device)
        K_idx, K_rq, K_rs, joint_K = make_packed_kv(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)
        V_idx, V_rq, V_rs, joint_V = make_packed_kv(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)

        K_dense = _decode_kv_torch(K_idx, K_rq, K_rs, joint_K, r_qmax).repeat_interleave(H_q // H_kv, dim=1)
        V_dense = _decode_kv_torch(V_idx, V_rq, V_rs, joint_V, r_qmax).repeat_interleave(H_q // H_kv, dim=1)
        scale = 1.0 / math.sqrt(d_h)

        t_sdpa_dec = bench(sdpa_fn, Q, K_dense, V_dense, warmup=5, repeats=30)
        print(f"  fp16 SDPA (dense K/V):                  {t_sdpa_dec*1e3:.3f} ms/call")
        t_ref_dec = bench(fused_hqmq_attention_torch, Q, K_idx, K_rq, K_rs, joint_K,
                          V_idx, V_rq, V_rs, joint_V, r_qmax, warmup=3, repeats=20)
        print(f"  Decode-then-SDPA (fake-quant pipeline): {t_ref_dec*1e3:.3f} ms/call")

        if TRITON_AVAILABLE:
            best_t_dec = None
            best_cfg_dec = None
            for bq, bk, nw, ns in [(16, 128, 4, 1), (16, 64, 4, 1), (16, 64, 4, 2),
                                   (16, 32, 4, 2), (32, 64, 4, 1), (32, 32, 4, 2)]:
                try:
                    def fn():
                        return fused_hqmq_attention_triton(
                            Q, K_idx, K_rq, K_rs, joint_K, V_idx, V_rq, V_rs, joint_V,
                            r_qmax=r_qmax, causal=True, block_q=bq, block_kv=bk,
                            num_warps=nw, num_stages=ns)
                    _ = fn(); torch.cuda.synchronize()
                    t = bench(fn, warmup=3, repeats=20)
                    if best_t_dec is None or t < best_t_dec:
                        best_t_dec = t
                        best_cfg_dec = (bq, bk, nw, ns)
                except Exception as e:
                    pass
            if best_t_dec is not None:
                print(f"  Fused HQMQ-Attention (best):            {best_t_dec*1e3:.3f} ms/call (cfg={best_cfg_dec})")
                print(f"     vs fp16 SDPA:           {t_sdpa_dec/best_t_dec:.2f}x")
                print(f"     vs fake-quant pipeline: {t_ref_dec/best_t_dec:.2f}x speedup")


if __name__ == "__main__":
    main()
