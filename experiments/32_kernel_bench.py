"""Microbenchmark the HQMQ decode: PyTorch einsum vs Triton fused kernel.

Decode = reconstruction of (B, H, T, d_head) fp16 KV from packed codebook
indices, per-chunk radii, and the precomputed joint codebook. This isolates
the dequant cost from the attention math; for a real attention kernel the
two would be fused, eliminating the round-trip through fp16.

We benchmark a single (layer, kv) decode for a transformer-sized workload:
  B=1, H=8, T=4096, d_head=128, S=192 (24*S=4608 codewords)
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quantizers.hqmq_kernel import hqmq_decode_triton, TRITON_AVAILABLE


def pytorch_decode(idx, radius_q, radius_scale, joint, r_qmax):
    """Reference PyTorch implementation of the same decode."""
    B, H, T, n_chunks = idx.shape
    K = joint.shape[1]
    d_head = n_chunks * 4
    # Gather codewords per head
    joint_b = joint.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, H, 1, 1, K, 4)
    idx_gather = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 1, 4)  # (B,H,T,nc,1,4)
    joint_b_exp = joint_b.expand(B, H, T, n_chunks, K, 4)
    codeword = joint_b_exp.gather(dim=-2, index=idx_gather).squeeze(-2)  # (B,H,T,nc,4)
    # Dequant radius
    radius = (radius_q.to(joint.dtype) * radius_scale.unsqueeze(-1)) / r_qmax  # (B,H,T,nc)
    out = codeword * radius.unsqueeze(-1)
    return out.view(B, H, T, d_head)


def bench(fn, *args, warmup=3, repeats=50):
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--T", type=int, default=4096)
    p.add_argument("--d-head", type=int, default=128)
    p.add_argument("--S", type=int, default=192)
    p.add_argument("--r-bits", type=int, default=4)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()

    dev = "cuda"
    dtype = torch.float16
    n_chunks = args.d_head // 4
    K = 24 * args.S
    r_qmax = (1 << args.r_bits) - 1

    print(f"Config: B={args.B}, H={args.H}, T={args.T}, d_head={args.d_head}, n_chunks={n_chunks}, K={K}, r_bits={args.r_bits}")
    print(f"Triton available: {TRITON_AVAILABLE}")
    print()

    # Generate test inputs
    torch.manual_seed(0)
    idx = torch.randint(0, K, (args.B, args.H, args.T, n_chunks), dtype=torch.int32, device=dev)
    radius_q = torch.randint(0, r_qmax + 1, (args.B, args.H, args.T, n_chunks), dtype=torch.int32, device=dev)
    radius_scale = torch.rand((args.B, args.H, args.T), dtype=dtype, device=dev) + 0.5
    joint = torch.randn((args.H, K, 4), dtype=dtype, device=dev)

    # PyTorch reference
    t_pt = bench(pytorch_decode, idx, radius_q, radius_scale, joint, r_qmax, repeats=args.repeats)
    print(f"PyTorch decode: {t_pt*1e3:.3f} ms/call")

    if TRITON_AVAILABLE:
        # Triton kernel
        try:
            # Warmup + correctness sanity
            out_pt = pytorch_decode(idx, radius_q, radius_scale, joint, r_qmax)
            out_tt = hqmq_decode_triton(idx, radius_q, radius_scale, joint, r_qmax)
            max_err = (out_pt - out_tt).abs().max().item()
            print(f"Triton vs PyTorch max abs diff: {max_err:.4e}")

            t_tt = bench(hqmq_decode_triton, idx, radius_q, radius_scale, joint, r_qmax, repeats=args.repeats)
            print(f"Triton kernel: {t_tt*1e3:.3f} ms/call  ({t_pt/t_tt:.2f}× speedup)")
        except Exception as e:
            print(f"Triton failed: {str(e)[:300]}")

    # FLOP/byte rough sanity
    n_elem = args.B * args.H * args.T * args.d_head
    bytes_input = (idx.element_size() * idx.numel() + radius_q.element_size() * radius_q.numel()
                   + radius_scale.element_size() * radius_scale.numel()
                   + joint.element_size() * joint.numel())
    bytes_output = (args.B * args.H * args.T * args.d_head) * 2  # bf16
    print()
    print(f"Output size: {bytes_output/1e6:.2f} MB; total input: {bytes_input/1e6:.2f} MB")
    print(f"PyTorch effective bandwidth: {(bytes_input + bytes_output) / t_pt / 1e9:.1f} GB/s")


if __name__ == "__main__":
    main()
