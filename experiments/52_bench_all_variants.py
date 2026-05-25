"""Side-by-side benchmark of all HQMQ kernel variants on the local GPU.

Variants tested:
  1. fp16 SDPA (cuDNN FlashAttention)            — gold-standard baseline
  2. Decode-then-SDPA (fake-quant pipeline)      — current research baseline
  3. Ada-tuned fused HQMQ (hqmq_attention.py)    — paper's main kernel
  4. Hopper-tuned fused HQMQ + FP8 codebook      — auto-skipped on non-Hopper
  5. Blackwell-tuned fused HQMQ                  — auto-skipped on non-Blackwell

Run this on each new GPU to see which variant wins at which workload.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hqmq


def make_packed(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype):
    idx = torch.randint(0, K_size, (B, H_kv, T_kv, n_chunks), dtype=torch.int32, device=device)
    rq = torch.randint(1, r_qmax + 1, (B, H_kv, T_kv, n_chunks), dtype=torch.int32, device=device)
    rs = torch.rand((B, H_kv, T_kv), dtype=dtype, device=device) + 0.5
    joint = torch.randn((H_kv, K_size, 4), dtype=dtype, device=device)
    joint = joint / joint.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return idx, rq, rs, joint


def bench(fn, warmup=3, repeats=10):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def run_workload(label, B, H_q, H_kv, T_q, T_kv, d_h, S, r_qmax, dtype, device):
    n_chunks = d_h // 4
    K_size = 24 * S
    print(f"\n=== {label} ===")
    print(f"   B={B} H_q={H_q} H_kv={H_kv} T_q={T_q} T_kv={T_kv} d_h={d_h} S={S}")

    Q = torch.randn((B, H_q, T_q, d_h), dtype=dtype, device=device)
    Ki, Krq, Krs, Jk = make_packed(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)
    Vi, Vrq, Vrs, Jv = make_packed(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)

    K_dense = hqmq.fused_attention_reference.__module__  # just for import side-effects
    from src.quantizers.hqmq_attention import _decode_kv_torch
    K_full = _decode_kv_torch(Ki, Krq, Krs, Jk, r_qmax).repeat_interleave(H_q // H_kv, dim=1)
    V_full = _decode_kv_torch(Vi, Vrq, Vrs, Jv, r_qmax).repeat_interleave(H_q // H_kv, dim=1)
    scale = 1.0 / math.sqrt(d_h)

    rows = []

    # 1. fp16 SDPA baseline
    sdpa_fn = lambda: F.scaled_dot_product_attention(Q, K_full, V_full, is_causal=True, scale=scale)
    t = bench(sdpa_fn)
    rows.append(("fp16 SDPA (dense K/V)", t, "baseline"))

    # 2. Fake-quant pipeline
    fq_fn = lambda: hqmq.fused_attention_reference(
        Q, Ki, Krq, Krs, Jk, Vi, Vrq, Vrs, Jv, r_qmax=r_qmax, causal=True)
    t = bench(fq_fn)
    rows.append(("Fake-quant pipeline", t, "decode-then-SDPA"))

    # 3. Ada-tuned (default) — uses device-aware dispatch
    try:
        ada_fn = lambda: hqmq.fused_attention(
            Q, Ki, Krq, Krs, Jk, Vi, Vrq, Vrs, Jv, r_qmax=r_qmax, causal=True)
        # Warmup with the auto-pick, then measure
        _ = ada_fn(); torch.cuda.synchronize()
        t = bench(ada_fn)
        rows.append(("Fused HQMQ (auto-dispatch)", t, hqmq.device_info()["variant_auto_dispatched"]))
    except Exception as e:
        rows.append(("Fused HQMQ (auto-dispatch)", float("inf"), f"FAILED: {str(e)[:80]}"))

    # 4. Hopper FP8 — auto-skipped on non-Hopper (silently dispatches to fp16 kernel)
    if hqmq.kernels_hopper.hopper_available():
        try:
            hopper_fn = lambda: hqmq.kernels_hopper.fused_hqmq_attention_hopper(
                Q, Ki, Krq, Krs, Jk, Vi, Vrq, Vrs, Jv, r_qmax=r_qmax, causal=True,
                use_fp8_codebook=True,
            )
            _ = hopper_fn(); torch.cuda.synchronize()
            t = bench(hopper_fn)
            rows.append(("Fused HQMQ (Hopper FP8 codebook)", t, "fp8 codebook"))
        except Exception as e:
            rows.append(("Fused HQMQ (Hopper FP8 codebook)", float("inf"), f"FAILED: {str(e)[:80]}"))
    else:
        rows.append(("Fused HQMQ (Hopper FP8 codebook)", None, "skipped (need SM 9.0+)"))

    # 5. Blackwell — falls back to Hopper when not on Blackwell
    if hqmq.kernels_blackwell.blackwell_available():
        try:
            bw_fn = lambda: hqmq.kernels_blackwell.fused_hqmq_attention_blackwell(
                Q, Ki, Krq, Krs, Jk, Vi, Vrq, Vrs, Jv, r_qmax=r_qmax, causal=True,
                use_fp8_codebook=True,
            )
            _ = bw_fn(); torch.cuda.synchronize()
            t = bench(bw_fn)
            rows.append(("Fused HQMQ (Blackwell tcgen05+TMEM)", t, "tcgen05+TMEM"))
        except Exception as e:
            rows.append(("Fused HQMQ (Blackwell tcgen05+TMEM)", float("inf"), f"FAILED: {str(e)[:80]}"))
    else:
        rows.append(("Fused HQMQ (Blackwell tcgen05+TMEM)", None, "skipped (need SM 10.0+)"))

    print(f"   {'variant':<42s} {'ms/call':>10s}  notes")
    print(f"   {'-'*42}  {'-'*10}  {'-'*30}")
    sdpa_t = rows[0][1]
    fq_t = rows[1][1]
    for nm, t, note in rows:
        if t is None:
            print(f"   {nm:<42s} {'---':>10s}  {note}")
        elif math.isinf(t):
            print(f"   {nm:<42s} {'FAILED':>10s}  {note}")
        else:
            speedup_vs_fq = fq_t / t
            speedup_vs_sdpa = sdpa_t / t
            extra = f"{speedup_vs_sdpa:.2f}x SDPA, {speedup_vs_fq:.2f}x fq"
            print(f"   {nm:<42s} {t*1e3:>10.3f}  {note:<30s} {extra}")

    return rows


def main():
    p = argparse.ArgumentParser()
    args = p.parse_args()

    device = torch.device("cuda")
    info = hqmq.device_info()
    print(f"GPU: {info['gpu']}  (capability {info['compute_capability']})")
    print(f"Auto-dispatch variant: {info['variant_auto_dispatched']}")
    print(f"FP8 codebook supported: {info['fp8_codebook_supported']}")

    workloads = [
        # (label, B, H_q, H_kv, T_q, T_kv, d_h, S, r_qmax, dtype)
        ("Prefill T=2048 (Mistral-class GQA)",   1, 32, 8, 2048, 2048, 128, 192, 15, torch.float16),
        ("Decode T_kv=4k  (Mistral-class GQA)",  1, 32, 8, 1,    4096, 128, 192, 15, torch.float16),
        ("Decode T_kv=16k (Mistral-class GQA)",  1, 32, 8, 1,   16384, 128, 192, 15, torch.float16),
        ("Decode T_kv=32k (Mistral-class GQA)",  1, 32, 8, 1,   32768, 128, 192, 15, torch.float16),
    ]

    all_results = {}
    for w in workloads:
        rows = run_workload(*w, device=device)
        all_results[w[0]] = [
            {"variant": r[0], "ms": r[1], "note": r[2]} for r in rows
        ]

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"52_bench_all_{info['gpu'].replace(' ', '_')}_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "gpu": info["gpu"],
        "compute_capability": info["compute_capability"],
        "variant_auto_dispatched": info["variant_auto_dispatched"],
        "results": all_results,
    }, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
