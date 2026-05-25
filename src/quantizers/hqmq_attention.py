"""Fused HQMQ-Attention kernel: skip the round-trip through fp16 KV cache.

The current QuantizedCache pipeline dequantizes packed HQMQ K/V to fp16 before
attention. This kernel eliminates that round-trip: K and V are decoded inline
from their codebook indices + radii inside the attention compute, using
FlashAttention-style online softmax to avoid materializing the full QK^T.

INTERFACE
---------
Inputs (per attention layer per forward pass):
  Q:              (B, H_q, T_q, d_h)              fp16/bf16
  K_idx:          (B, H_kv, T_kv, n_chunks)       int32, codeword index in [0, 24*S)
  K_radius_q:     (B, H_kv, T_kv, n_chunks)       int8/int16, radius quantum in [0, 2^b_r)
  K_radius_scale: (B, H_kv, T_kv)                 fp16/bf16, per-token scale
  V_idx:          (B, H_kv, T_kv, n_chunks)       int32
  V_radius_q:     (B, H_kv, T_kv, n_chunks)       int8/int16
  V_radius_scale: (B, H_kv, T_kv)                 fp16/bf16
  joint_K:        (H_kv, K, 4)                    fp16/bf16, precomputed q_p · q_s table
  joint_V:        (H_kv, K, 4)                    fp16/bf16

  d_h = n_chunks * 4 (HQMQ chunk dim is 4)
  GQA: H_q = num_kv_groups * H_kv. Q heads are grouped, K/V heads broadcast.

Output:
  O: (B, H_q, T_q, d_h) fp16/bf16

Two implementations:
  1. `fused_hqmq_attention_torch` — PyTorch reference (correct, slow).
  2. `fused_hqmq_attention_triton` — Triton fused kernel (fast, target).
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _decode_kv_torch(idx: torch.Tensor, radius_q: torch.Tensor, radius_scale: torch.Tensor,
                     joint: torch.Tensor, r_qmax: int) -> torch.Tensor:
    """Reference decode of packed HQMQ K or V back to dense fp16 of shape (B, H, T, d_h).

    idx:           (B, H, T, n_chunks)      int32
    radius_q:      (B, H, T, n_chunks)      int
    radius_scale:  (B, H, T)                fp16/bf16
    joint:         (H, K, 4)                fp16/bf16 (precomputed q_p · q_s)
    r_qmax:        int  (e.g. 2^b_r - 1)

    Returns: (B, H, T, d_h) fp16/bf16  where d_h = n_chunks * 4.
    """
    B, H, T, n_chunks = idx.shape
    K_ = joint.shape[1]
    # Gather codeword per (b, h, t, c)
    # joint: (H, K, 4) -> broadcast to (B, H, T, n_chunks, 4) and index along K dim
    joint_exp = joint.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, H, 1, 1, K, 4)
    joint_b_exp = joint_exp.expand(B, H, T, n_chunks, K_, 4)
    idx_gather = idx.unsqueeze(-1).unsqueeze(-1).expand(B, H, T, n_chunks, 1, 4)
    codeword = joint_b_exp.gather(dim=-2, index=idx_gather.long()).squeeze(-2)  # (B,H,T,nc,4)

    # Dequantize radius
    radius = (radius_q.to(joint.dtype) * radius_scale.unsqueeze(-1)) / r_qmax  # (B,H,T,nc)

    # Multiply
    out = codeword * radius.unsqueeze(-1)  # (B, H, T, nc, 4)
    return out.view(B, H, T, n_chunks * 4)


def fused_hqmq_attention_torch(
    Q: torch.Tensor,
    K_idx: torch.Tensor, K_radius_q: torch.Tensor, K_radius_scale: torch.Tensor, joint_K: torch.Tensor,
    V_idx: torch.Tensor, V_radius_q: torch.Tensor, V_radius_scale: torch.Tensor, joint_V: torch.Tensor,
    r_qmax: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference fused-HQMQ attention via PyTorch SDPA.

    This is the non-fused-but-correct path: decode K and V to fp16, then run
    standard attention. Used as the correctness oracle for the Triton kernel.

    GQA: K/V heads are broadcast to Q heads (group size = H_q / H_kv).
    """
    B, H_q, T_q, d_h = Q.shape
    H_kv = K_idx.shape[1]
    assert H_q % H_kv == 0, f"H_q={H_q} must be divisible by H_kv={H_kv}"
    n_kv_groups = H_q // H_kv

    K = _decode_kv_torch(K_idx, K_radius_q, K_radius_scale, joint_K, r_qmax)  # (B, H_kv, T_kv, d_h)
    V = _decode_kv_torch(V_idx, V_radius_q, V_radius_scale, joint_V, r_qmax)  # (B, H_kv, T_kv, d_h)

    # Broadcast K/V heads to Q heads if GQA
    if n_kv_groups > 1:
        K = K.repeat_interleave(n_kv_groups, dim=1)
        V = V.repeat_interleave(n_kv_groups, dim=1)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_h)

    # Standard SDPA: attn_mask=None and is_causal=True triggers Flash backend on CUDA
    O = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=causal, scale=softmax_scale)
    return O


# =====================================================================
# Triton fused kernel
# =====================================================================
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    # --------------------------------------------------------------
    # Optimized kernel: fp16 tensor cores, vectorized loads, pipelining
    # --------------------------------------------------------------
    @triton.jit
    def _fused_hqmq_attn_kernel_opt(
        O_PTR,                # (B, H_q, T_q, D_HEAD)  fp16
        Q_PTR,                # (B, H_q, T_q, D_HEAD)  fp16
        K_IDX_PTR,            # (B, H_kv, T_kv, N_CHUNKS) int32
        K_RQ_PTR,             # (B, H_kv, T_kv, N_CHUNKS) int32
        K_RS_PTR,             # (B, H_kv, T_kv)         fp16
        JOINT_K_PTR,          # (H_kv, K_SIZE, 4)       fp16
        V_IDX_PTR,            # (B, H_kv, T_kv, N_CHUNKS) int32
        V_RQ_PTR,             # (B, H_kv, T_kv, N_CHUNKS) int32
        V_RS_PTR,             # (B, H_kv, T_kv)         fp16
        JOINT_V_PTR,          # (H_kv, K_SIZE, 4)       fp16
        q_b, q_h, q_t, q_d,
        ki_b, ki_h, ki_t,
        krq_b, krq_h, krq_t,
        krs_b, krs_h,
        jk_h, jk_k,
        vi_b, vi_h, vi_t,
        vrq_b, vrq_h, vrq_t,
        vrs_b, vrs_h,
        jv_h, jv_k,
        o_b, o_h, o_t, o_d,
        T_Q: tl.constexpr, T_KV: tl.constexpr,
        N_KV_GROUPS: tl.constexpr,
        N_CHUNKS: tl.constexpr,
        D_HEAD: tl.constexpr,
        K_SIZE: tl.constexpr,
        R_QMAX: tl.constexpr,
        SCALE: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_KV: tl.constexpr,
    ):
        """Optimized fused HQMQ-attention using fp16 tensor cores.

        Layout: 3D grid (b, h_q, q_block). For each tile we:
          1) Load Q tile (fp16) into registers.
          2) Loop kv tiles:
             a) Load K codeword indices and radii (int32 / int).
             b) Reconstruct K_tile (fp16) via codebook gather + radius mul.
             c) S = Q @ K_tile.T  (fp16 tensor-core dot, fp32 accumulate)
             d) Causal mask + online softmax update.
             e) Load V codebook indices and radii; reconstruct V_tile (fp16).
             f) acc += P @ V_tile  (fp16 tensor-core dot)
          3) Finalize: O = acc / l_i, store as fp16.

        Key opts vs v2:
         - fp16 throughout for matmuls (uses TC on Ada/Hopper).
         - Vectorized codeword load: gather all 4 components in one `tl.load` via
           offset broadcast (each chunk's 4 components are contiguous in `joint`).
         - No staged Q in shared mem; relies on register file.
         - `num_stages` set by autotune at call site for software pipelining.
        """
        b = tl.program_id(0)
        h_q = tl.program_id(1)
        q_block_id = tl.program_id(2)
        h_kv = h_q // N_KV_GROUPS

        # Token & dim offsets
        q_off = q_block_id * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_off < T_Q
        d_off = tl.arange(0, D_HEAD)
        chunk_for_d = d_off // 4
        comp_for_d = d_off % 4

        # Load Q tile and force fp16 for tensor-core matmul path
        q_ptrs = Q_PTR + b * q_b + h_q * q_h + q_off[:, None] * q_t + d_off[None, :] * q_d
        Q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float16)

        # Online softmax accumulators (fp32 for stability)
        m_i = tl.full([BLOCK_Q], value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)

        # Causal: only need to scan kv tiles up to the maximum q_off in this q tile
        if CAUSAL:
            kv_end = tl.minimum(T_KV, (q_block_id + 1) * BLOCK_Q)
        else:
            kv_end = T_KV

        for kv_start in tl.range(0, kv_end, BLOCK_KV):
            kv_off = kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = kv_off < T_KV

            # --- Load K codeword indices for entire (BLOCK_KV, D_HEAD) tile in one go ---
            # idx[kv, chunk] then expand to (BLOCK_KV, D_HEAD) via chunk_for_d
            kidx_for_d = tl.load(
                K_IDX_PTR + b * ki_b + h_kv * ki_h + kv_off[:, None] * ki_t + chunk_for_d[None, :],
                mask=kv_mask[:, None], other=0,
            ).to(tl.int32)
            jk_ptrs = JOINT_K_PTR + h_kv * jk_h + kidx_for_d * jk_k + comp_for_d[None, :]
            K_tile_raw = tl.load(jk_ptrs, mask=kv_mask[:, None], other=0.0)

            # Radius dequant per (kv, chunk) → per (kv, d). Use fp16 throughout
            # to keep SRAM footprint low. R_QMAX is constexpr int so we can hoist.
            krq_for_d = tl.load(
                K_RQ_PTR + b * krq_b + h_kv * krq_h + kv_off[:, None] * krq_t + chunk_for_d[None, :],
                mask=kv_mask[:, None], other=0,
            ).to(tl.float16)
            krs = tl.load(K_RS_PTR + b * krs_b + h_kv * krs_h + kv_off,
                          mask=kv_mask, other=1.0).to(tl.float16)
            # Fused multiply: K_tile = K_codeword * krq * krs / R_QMAX (one register pass)
            inv_rqmax: tl.constexpr = 1.0 / R_QMAX
            K_tile = (K_tile_raw.to(tl.float16) * krq_for_d) * (krs[:, None] * inv_rqmax)

            # QK^T via tensor cores (fp16 inputs, fp32 accumulate)
            S = tl.dot(Q, tl.trans(K_tile), out_dtype=tl.float32) * SCALE

            if CAUSAL:
                S = tl.where(q_off[:, None] >= kv_off[None, :], S, -1.0e6)
            S = tl.where(kv_mask[None, :], S, -1.0e6)

            # Online softmax
            m_ij = tl.max(S, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            P = tl.exp(S - m_new[:, None])
            l_ij = tl.sum(P, axis=1)
            l_new = alpha * l_i + l_ij

            # --- Load V codebook (same pattern) ---
            vidx_for_d = tl.load(
                V_IDX_PTR + b * vi_b + h_kv * vi_h + kv_off[:, None] * vi_t + chunk_for_d[None, :],
                mask=kv_mask[:, None], other=0,
            ).to(tl.int32)
            jv_ptrs = JOINT_V_PTR + h_kv * jv_h + vidx_for_d * jv_k + comp_for_d[None, :]
            V_tile_raw = tl.load(jv_ptrs, mask=kv_mask[:, None], other=0.0)

            vrq_for_d = tl.load(
                V_RQ_PTR + b * vrq_b + h_kv * vrq_h + kv_off[:, None] * vrq_t + chunk_for_d[None, :],
                mask=kv_mask[:, None], other=0,
            ).to(tl.float16)
            vrs = tl.load(V_RS_PTR + b * vrs_b + h_kv * vrs_h + kv_off,
                          mask=kv_mask, other=1.0).to(tl.float16)
            V_tile = (V_tile_raw.to(tl.float16) * vrq_for_d) * (vrs[:, None] * inv_rqmax)

            # Accumulator update with tensor-core dot
            acc = acc * alpha[:, None] + tl.dot(P.to(V_tile.dtype), V_tile, out_dtype=tl.float32)
            m_i = m_new
            l_i = l_new

        # Finalize and store. Cast to the output tensor's dtype (typically fp16/bf16/fp32).
        O = acc / l_i[:, None]  # fp32
        o_ptrs = O_PTR + b * o_b + h_q * o_h + q_off[:, None] * o_t + d_off[None, :] * o_d
        # Use tl.load on a single element to infer the output dtype, or just cast and trust:
        tl.store(o_ptrs, O.to(O_PTR.dtype.element_ty), mask=q_mask[:, None])


    @triton.jit
    def _fused_hqmq_attn_kernel_v2(
        # Outputs
        O_PTR,                # (B, H_q, T_q, D_HEAD)
        # Inputs
        Q_PTR,                # (B, H_q, T_q, D_HEAD)
        K_IDX_PTR,            # (B, H_kv, T_kv, N_CHUNKS) int32
        K_RQ_PTR,             # (B, H_kv, T_kv, N_CHUNKS) int32 (any int that fits)
        K_RS_PTR,             # (B, H_kv, T_kv)           fp
        JOINT_K_PTR,          # (H_kv, K_SIZE, 4)         fp
        V_IDX_PTR,            # (B, H_kv, T_kv, N_CHUNKS)
        V_RQ_PTR,             # (B, H_kv, T_kv, N_CHUNKS)
        V_RS_PTR,             # (B, H_kv, T_kv)
        JOINT_V_PTR,          # (H_kv, K_SIZE, 4)
        # Strides
        q_b, q_h, q_t, q_d,
        ki_b, ki_h, ki_t,
        krq_b, krq_h, krq_t,
        krs_b, krs_h,
        jk_h, jk_k,
        vi_b, vi_h, vi_t,
        vrq_b, vrq_h, vrq_t,
        vrs_b, vrs_h,
        jv_h, jv_k,
        o_b, o_h, o_t, o_d,
        # Sizes
        T_Q: tl.constexpr, T_KV: tl.constexpr,
        N_KV_GROUPS: tl.constexpr,
        N_CHUNKS: tl.constexpr,
        D_HEAD: tl.constexpr,
        K_SIZE: tl.constexpr,
        R_QMAX: tl.constexpr,
        SCALE: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_KV: tl.constexpr,
    ):
        """3D grid: (b, h_q, q_block). h_kv = h_q // N_KV_GROUPS."""
        b = tl.program_id(0)
        h_q = tl.program_id(1)
        q_block_id = tl.program_id(2)
        h_kv = h_q // N_KV_GROUPS

        # Q tile offsets
        q_off = q_block_id * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_off < T_Q

        # Load Q tile: (BLOCK_Q, D_HEAD)
        d_off = tl.arange(0, D_HEAD)
        q_ptrs = Q_PTR + b * q_b + h_q * q_h + q_off[:, None] * q_t + d_off[None, :] * q_d
        Q = tl.load(q_ptrs, mask=q_mask[:, None] & (d_off[None, :] < D_HEAD), other=0.0).to(tl.float32)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_Q], value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)

        chunk_for_d = d_off // 4
        comp_for_d = d_off % 4

        # Loop over KV tiles
        for kv_start in tl.range(0, T_KV, BLOCK_KV):
            kv_off = kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = kv_off < T_KV

            # --- Decode K tile inline ---
            # Gather codeword index per (kv, chunk_for_d) and expand to (BLOCK_KV, D_HEAD)
            # by broadcasting chunk_for_d across the d-axis; component is comp_for_d.
            kidx_for_d = tl.load(
                K_IDX_PTR + b * ki_b + h_kv * ki_h + kv_off[:, None] * ki_t + chunk_for_d[None, :],
                mask=kv_mask[:, None], other=0,
            ).to(tl.int32)
            jk_full = JOINT_K_PTR + h_kv * jk_h + kidx_for_d * jk_k + comp_for_d[None, :]
            K_tile = tl.load(jk_full, mask=kv_mask[:, None], other=0.0).to(tl.float32)

            krs = tl.load(K_RS_PTR + b * krs_b + h_kv * krs_h + kv_off,
                          mask=kv_mask, other=1.0).to(tl.float32)
            krq_for_d = tl.load(
                K_RQ_PTR + b * krq_b + h_kv * krq_h + kv_off[:, None] * krq_t + chunk_for_d[None, :],
                mask=kv_mask[:, None], other=0,
            ).to(tl.float32)
            K_radius_for_d = krq_for_d * krs[:, None] / R_QMAX
            K_tile = K_tile * K_radius_for_d  # (BLOCK_KV, D_HEAD)

            # QK^T: (BLOCK_Q, D_HEAD) @ (BLOCK_KV, D_HEAD).T -> (BLOCK_Q, BLOCK_KV)
            S = tl.dot(Q, tl.trans(K_tile)) * SCALE

            # Causal mask: q_off >= kv_off => keep, else -inf
            if CAUSAL:
                S = tl.where(q_off[:, None] >= kv_off[None, :], S, -1.0e6)
            # Mask out kv beyond T_KV
            S = tl.where(kv_mask[None, :], S, -1.0e6)

            # Online softmax update
            m_ij = tl.max(S, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            P = tl.exp(S - m_new[:, None])
            l_ij = tl.sum(P, axis=1)
            l_new = alpha * l_i + l_ij

            # --- Decode V tile inline (same pattern as K) ---
            vidx_for_d = tl.load(V_IDX_PTR + b * vi_b + h_kv * vi_h + kv_off[:, None] * vi_t + chunk_for_d[None, :],
                                  mask=kv_mask[:, None], other=0).to(tl.int32)
            jv_full = JOINT_V_PTR + h_kv * jv_h + vidx_for_d * jv_k + comp_for_d[None, :]
            V_tile = tl.load(jv_full, mask=kv_mask[:, None], other=0.0).to(tl.float32)

            vrq_for_d = tl.load(V_RQ_PTR + b * vrq_b + h_kv * vrq_h + kv_off[:, None] * vrq_t + chunk_for_d[None, :],
                                 mask=kv_mask[:, None], other=0).to(tl.float32)
            vrs = tl.load(V_RS_PTR + b * vrs_b + h_kv * vrs_h + kv_off, mask=kv_mask, other=1.0).to(tl.float32)
            V_radius_for_d = vrq_for_d * vrs[:, None] / R_QMAX
            V_tile = V_tile * V_radius_for_d  # (BLOCK_KV, D_HEAD)

            # Update accumulator
            acc = acc * alpha[:, None]
            acc = acc + tl.dot(P.to(V_tile.dtype), V_tile)
            m_i = m_new
            l_i = l_new

        # Finalize
        O = acc / l_i[:, None]

        # Store (cast to the output tensor's actual dtype: fp16/bf16/fp32)
        o_ptrs = O_PTR + b * o_b + h_q * o_h + q_off[:, None] * o_t + d_off[None, :] * o_d
        tl.store(o_ptrs, O.to(O_PTR.dtype.element_ty),
                 mask=q_mask[:, None] & (d_off[None, :] < D_HEAD))


def _pick_device_config(device: torch.device, T_q: int):
    """Pick block sizes / warp count based on GPU compute capability.

    Defaults are tuned for Ada (RTX 4090, SM 8.9, ~96 KB shared mem / SM).
    On Hopper (SM 9.0, ~228 KB / SM) we can use bigger BLOCK_KV tiles.
    On Blackwell (SM 10.0+, even more shared mem) we use the largest tiles.

    Decode workloads (T_q small) get smaller BLOCK_Q to avoid wasting
    parallelism on padded query rows.
    """
    if not torch.cuda.is_available():
        return dict(block_q=64, block_kv=32, num_warps=4, num_stages=2)
    cap = torch.cuda.get_device_capability(device)
    is_decode = T_q <= 4
    major = cap[0]
    if major >= 10:  # Blackwell (B100/B200/GB200) — sm_100/101/120
        if is_decode:
            return dict(block_q=16, block_kv=256, num_warps=8, num_stages=4)
        return dict(block_q=256, block_kv=256, num_warps=8, num_stages=4)
    if major >= 9:   # Hopper (H100/H200)
        if is_decode:
            return dict(block_q=16, block_kv=128, num_warps=8, num_stages=3)
        return dict(block_q=256, block_kv=128, num_warps=8, num_stages=3)
    # Ada (RTX 4090) and older — keep the empirically tuned defaults
    if is_decode:
        return dict(block_q=32, block_kv=32, num_warps=4, num_stages=2)
    return dict(block_q=128, block_kv=32, num_warps=4, num_stages=2)


def fused_hqmq_attention_triton(
    Q: torch.Tensor,
    K_idx: torch.Tensor, K_radius_q: torch.Tensor, K_radius_scale: torch.Tensor, joint_K: torch.Tensor,
    V_idx: torch.Tensor, V_radius_q: torch.Tensor, V_radius_scale: torch.Tensor, joint_V: torch.Tensor,
    r_qmax: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    block_q: Optional[int] = None,
    block_kv: Optional[int] = None,
    optimized: bool = True,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    use_fp8_codebook: bool = False,
) -> torch.Tensor:
    """Fused HQMQ-Attention via Triton. Decodes K and V inline; never materializes
    a dense fp16 KV cache. Returns O of shape (B, H_q, T_q, d_h).

    Performance: if any of block_q/block_kv/num_warps/num_stages is None, picks
    a device-aware default (Ada / Hopper / Blackwell). Setting them explicitly
    overrides the autopick.

    Set use_fp8_codebook=True on Hopper+ (SM 9.0+) to encode the joint codebook
    in FP8 (e4m3) instead of fp16. The call is then dispatched to the dedicated
    Hopper kernel (``src/quantizers/hqmq_attention_hopper.py``) which halves
    the codebook memory footprint and uses Hopper's tensor-core paths.

    On Blackwell (SM 10.0+), use the ``hqmq_attention_blackwell`` module
    directly for the design-stage TMEM/tcgen05 paths.
    """
    # If user explicitly asks for FP8 codebook and we're on Hopper+,
    # dispatch to the dedicated FP8 kernel.
    if use_fp8_codebook and Q.is_cuda:
        cap = torch.cuda.get_device_capability(Q.device)
        if cap[0] >= 9:
            from .hqmq_attention_hopper import fused_hqmq_attention_hopper
            return fused_hqmq_attention_hopper(
                Q, K_idx, K_radius_q, K_radius_scale, joint_K,
                V_idx, V_radius_q, V_radius_scale, joint_V,
                r_qmax=r_qmax, causal=causal, softmax_scale=softmax_scale,
                block_q=block_q or 256, block_kv=block_kv or 128,
                num_warps=num_warps or 8, num_stages=num_stages or 3,
                use_fp8_codebook=True,
            )
        # Not on Hopper — silently fall through to fp16 kernel
    assert TRITON_AVAILABLE, "Triton not installed"
    B, H_q, T_q, d_h = Q.shape
    H_kv = K_idx.shape[1]
    T_kv = K_idx.shape[2]
    n_chunks = K_idx.shape[3]
    assert d_h == n_chunks * 4, f"d_h={d_h} != n_chunks*4={n_chunks*4}"
    n_kv_groups = H_q // H_kv
    K_size = joint_K.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_h)

    # Pick device-aware defaults for any block sizes the caller didn't specify
    device_cfg = _pick_device_config(Q.device, T_q)
    if block_q is None: block_q = device_cfg["block_q"]
    if block_kv is None: block_kv = device_cfg["block_kv"]
    if num_warps is None: num_warps = device_cfg["num_warps"]
    if num_stages is None: num_stages = device_cfg["num_stages"]

    O = torch.empty_like(Q)

    grid = (B, H_q, (T_q + block_q - 1) // block_q)
    kernel = _fused_hqmq_attn_kernel_opt if optimized else _fused_hqmq_attn_kernel_v2
    kernel[grid](
        O, Q,
        K_idx, K_radius_q.to(torch.int32), K_radius_scale, joint_K,
        V_idx, V_radius_q.to(torch.int32), V_radius_scale, joint_V,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_idx.stride(0), K_idx.stride(1), K_idx.stride(2),
        K_radius_q.stride(0), K_radius_q.stride(1), K_radius_q.stride(2),
        K_radius_scale.stride(0), K_radius_scale.stride(1),
        joint_K.stride(0), joint_K.stride(1),
        V_idx.stride(0), V_idx.stride(1), V_idx.stride(2),
        V_radius_q.stride(0), V_radius_q.stride(1), V_radius_q.stride(2),
        V_radius_scale.stride(0), V_radius_scale.stride(1),
        joint_V.stride(0), joint_V.stride(1),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        T_Q=T_q, T_KV=T_kv,
        N_KV_GROUPS=n_kv_groups,
        N_CHUNKS=n_chunks,
        D_HEAD=d_h,
        K_SIZE=K_size,
        R_QMAX=r_qmax,
        SCALE=softmax_scale,
        CAUSAL=causal,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return O

