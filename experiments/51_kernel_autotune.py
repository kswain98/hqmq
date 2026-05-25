"""Run the fused HQMQ-attention kernel across a sweep of block sizes / num_warps /
num_stages on the local GPU. Reports the best config per (workload, GPU).

Use this to re-tune for H100 / H200 / B200 by simply running it on the target
hardware. Compare its "best cfg" output to the defaults in
``hqmq_attention._pick_device_config`` and update if necessary.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quantizers.hqmq_attention import (
    fused_hqmq_attention_torch, fused_hqmq_attention_triton, TRITON_AVAILABLE,
    _decode_kv_torch, _pick_device_config,
)


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


def sweep(label, B, H_q, H_kv, T_q, T_kv, d_h, S, r_qmax, dtype, device):
    n_chunks = d_h // 4
    K_size = 24 * S
    Q = torch.randn((B, H_q, T_q, d_h), dtype=dtype, device=device)
    Ki, Krq, Krs, Jk = make_packed(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)
    Vi, Vrq, Vrs, Jv = make_packed(B, H_kv, T_kv, n_chunks, K_size, r_qmax, device, dtype)

    # The current default for this workload
    default_cfg = _pick_device_config(device, T_q)
    print(f"\n=== {label}: device default = {default_cfg} ===")

    # Sweep grid
    configs = []
    is_decode = T_q <= 4
    if is_decode:
        bq_list = [8, 16, 32]
        bk_list = [32, 64, 128, 256]
    else:
        bq_list = [64, 128, 256]
        bk_list = [32, 64, 128, 256]
    for bq in bq_list:
        for bk in bk_list:
            for nw in [4, 8]:
                for ns in [1, 2, 3, 4]:
                    configs.append((bq, bk, nw, ns))

    best_t = float("inf")
    best_cfg = None
    for bq, bk, nw, ns in configs:
        def fn():
            return fused_hqmq_attention_triton(
                Q, Ki, Krq, Krs, Jk, Vi, Vrq, Vrs, Jv,
                r_qmax=r_qmax, causal=True,
                block_q=bq, block_kv=bk, num_warps=nw, num_stages=ns,
            )
        try:
            _ = fn()
            torch.cuda.synchronize()
            t = bench(fn, warmup=2, repeats=8)
            if t < best_t:
                best_t = t
                best_cfg = (bq, bk, nw, ns)
        except Exception:
            continue

    if best_cfg is None:
        print(f"  All configs FAILED")
        return None
    print(f"  best = (BLOCK_Q={best_cfg[0]}, BLOCK_KV={best_cfg[1]}, "
          f"num_warps={best_cfg[2]}, num_stages={best_cfg[3]})  → {best_t*1e3:.3f} ms/call")
    if best_cfg != (default_cfg["block_q"], default_cfg["block_kv"],
                    default_cfg["num_warps"], default_cfg["num_stages"]):
        print(f"  ⚠ Different from device-default. Consider updating "
              f"_pick_device_config for this GPU.")
    return best_cfg, best_t


def main():
    p = argparse.ArgumentParser()
    args = p.parse_args()

    if not TRITON_AVAILABLE:
        print("Triton not available; exiting.")
        return

    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(device)
    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu_name}  (compute capability {cap[0]}.{cap[1]})")

    workloads = [
        ("Prefill T=2048 (Mistral GQA)",  1, 32, 8, 2048, 2048, 128, 192, 15, torch.float16),
        ("Prefill T=8192 (Mistral GQA)",  1, 32, 8, 8192, 8192, 128, 192, 15, torch.float16),
        ("Decode  T_kv=4k (Mistral GQA)", 1, 32, 8, 1, 4096,    128, 192, 15, torch.float16),
        ("Decode  T_kv=32k (Mistral GQA)",1, 32, 8, 1, 32768,   128, 192, 15, torch.float16),
    ]

    summary = {}
    for label, B, H_q, H_kv, T_q, T_kv, d_h, S, r_qmax, dtype in workloads:
        try:
            result = sweep(label, B, H_q, H_kv, T_q, T_kv, d_h, S, r_qmax, dtype, device)
            if result is not None:
                summary[label] = {"cfg": result[0], "ms": result[1] * 1e3}
        except Exception as e:
            print(f"  FAILED: {str(e)[:200]}")

    print("\n=== Summary on this GPU ===")
    for label, data in summary.items():
        print(f"  {label:<40s}  best={data['cfg']}  {data['ms']:.3f} ms")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"51_kernel_autotune_{gpu_name.replace(' ', '_')}_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "gpu": gpu_name, "compute_capability": list(cap), "results": summary,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
