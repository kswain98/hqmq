"""Triton-fused HQMQ decode kernel.

The fake-quantization decode in `hqmq.py` does codebook gather + radius
dequantize as separate PyTorch ops (einsum + gather + multiply). For
production-grade inference, a fused kernel doing:
    1) lookup q_p index (low 5 bits) → product factor from cached LUT
    2) lookup q_s index (high log2(S) bits) → product factor from per-(layer,head) codebook
    3) compute q_p · q_s (Hamilton product on 4-element registers)
    4) multiply by quantized radius (dequant from b_r bits to fp16)
    5) feed into attention compute

eliminates the round-trip through fp16 KV cache. We implement step (3)+(4) in
Triton, keeping the inputs/outputs in fp16/bf16 to interoperate cleanly with
the existing attention compute. The codebook indices and packed radius are
stored in a compact tensor, and we expand to fp16 inside the kernel.

Status: research prototype. Used in the wall-clock latency comparison in the
paper. Future work: fuse the entire attention compute (QK^T softmax + V
combine) with the dequant inside one kernel — this would amortize the
codebook lookup over the entire attention scan.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _hqmq_decode_kernel(
        # Outputs
        OUT_PTR,                # (B, H, T, n_chunks, 4) fp16/bf16
        # Inputs
        IDX_PTR,                # (B, H, T, n_chunks) int32 — codeword index in [0, 24*S)
        R_QUANT_PTR,            # (B, H, T, n_chunks) int32 — radius quantum in [0, 2^b_r)
        R_SCALE_PTR,            # (B, H, T) fp16 — per-token radius scale
        JOINT_PTR,              # (H, 24*S, 4) fp16/bf16 — precomputed q_p · q_s table per head
        # Strides
        out_b_stride, out_h_stride, out_t_stride, out_c_stride,
        idx_b_stride, idx_h_stride, idx_t_stride,
        rq_b_stride, rq_h_stride, rq_t_stride,
        rs_b_stride, rs_h_stride,
        j_h_stride, j_k_stride,
        # Sizes
        B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
        N_CHUNKS: tl.constexpr, K: tl.constexpr,
        R_QMAX: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """One kernel block per (batch, head, chunk, t_block). T tiled across program_id 2."""
        bh = tl.program_id(0)
        chunk = tl.program_id(1)
        t_block = tl.program_id(2)
        b = bh // H
        h = bh % H

        t_off = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_off < T

        # Load codeword indices
        idx_ptr = IDX_PTR + b * idx_b_stride + h * idx_h_stride + t_off * idx_t_stride + chunk
        idx = tl.load(idx_ptr, mask=t_mask, other=0).to(tl.int32)

        # Load radius quantum
        rq_ptr = R_QUANT_PTR + b * rq_b_stride + h * rq_h_stride + t_off * rq_t_stride + chunk
        rq = tl.load(rq_ptr, mask=t_mask, other=0).to(tl.float32)

        # Load per-token radius scale
        rs_ptr = R_SCALE_PTR + b * rs_b_stride + h * rs_h_stride + t_off
        rs = tl.load(rs_ptr, mask=t_mask, other=1.0).to(tl.float32)

        # Dequantize radius
        radius = rq * rs / R_QMAX

        # Load 4 components of joint codeword. JOINT[h, idx, :4]
        joint_base = JOINT_PTR + h * j_h_stride + idx * j_k_stride
        c0 = tl.load(joint_base + 0, mask=t_mask, other=0.0).to(tl.float32)
        c1 = tl.load(joint_base + 1, mask=t_mask, other=0.0).to(tl.float32)
        c2 = tl.load(joint_base + 2, mask=t_mask, other=0.0).to(tl.float32)
        c3 = tl.load(joint_base + 3, mask=t_mask, other=0.0).to(tl.float32)

        # Multiply by radius
        o0 = radius * c0
        o1 = radius * c1
        o2 = radius * c2
        o3 = radius * c3

        # Store back
        out_base = OUT_PTR + b * out_b_stride + h * out_h_stride + t_off * out_t_stride + chunk * out_c_stride
        tl.store(out_base + 0, o0, mask=t_mask)
        tl.store(out_base + 1, o1, mask=t_mask)
        tl.store(out_base + 2, o2, mask=t_mask)
        tl.store(out_base + 3, o3, mask=t_mask)


def hqmq_decode_triton(
    idx: torch.Tensor,         # (B, H, T, n_chunks) int32
    radius_q: torch.Tensor,    # (B, H, T, n_chunks) int (small)
    radius_scale: torch.Tensor,# (B, H, T) fp16/bf16
    joint: torch.Tensor,       # (H, K, 4) fp16/bf16
    r_qmax: int,
) -> torch.Tensor:
    """Reconstruct (B, H, T, d_head) fp16 KV by fused codebook lookup + radius dequant.

    d_head = n_chunks * 4.
    """
    assert TRITON_AVAILABLE, "Triton not available — install triton to use this kernel"
    B, H, T, n_chunks = idx.shape
    K = joint.shape[1]
    d_head = n_chunks * 4
    out = torch.empty((B, H, T, n_chunks, 4), dtype=joint.dtype, device=idx.device)

    BLOCK_T = 256
    n_t_blocks = (T + BLOCK_T - 1) // BLOCK_T
    grid = (B * H, n_chunks, n_t_blocks)
    _hqmq_decode_kernel[grid](
        out, idx, radius_q.to(torch.int32), radius_scale, joint,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2),
        radius_q.stride(0), radius_q.stride(1), radius_q.stride(2),
        radius_scale.stride(0), radius_scale.stride(1),
        joint.stride(0), joint.stride(1),
        B=B, H=H, T=T, N_CHUNKS=n_chunks, K=K,
        R_QMAX=r_qmax, BLOCK_T=BLOCK_T,
    )
    return out.view(B, H, T, d_head)
