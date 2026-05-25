"""Blackwell-native HQMQ-Attention kernel (design notes + stub).

**Status: design-only, not validated.** Triton's Blackwell support is still
maturing as of mid-2026, and we don't have B100/B200 hardware for this work.
The kernel below is a forward port of the Hopper variant; the Blackwell-
specific instruction set (TMEM, tcgen05, native FP4 MMA, cluster mode) is
documented in comments but not yet wired into actual Triton intrinsics.

When Triton's Blackwell support is mature (`tl.tcgen05`, `tl.tmem.*`,
`tl.cluster`), this file is the natural place to add those paths.

What Blackwell can in principle do better than Hopper for HQMQ:

1. **Native FP4 tensor cores** (E2M1). HQMQ codewords lie on $S^3$ — only the
   relative direction matters, not the absolute magnitudes. FP4 has 1-bit
   mantissa (8 levels) which is below the angular resolution needed for
   $S^3$ vector quantization in our experiments (we measured ~10% ppl
   degradation when forcing FP4 codebook on Hopper-class hardware via
   simulation). Recommendation: stick with FP8 codebook on Blackwell;
   reserve FP4 for activation / intermediate accumulators only.

2. **TMEM** (tensor memory, separate from SRAM). Codebooks fit naturally in
   TMEM since they're (H_kv, K, 4) = O(50KB) constant tensors. Loading them
   once into TMEM per kernel launch and reusing for every KV iteration
   eliminates the scattered codebook gather entirely. Estimated speedup:
   2--3x on the codebook-gather phase. Requires `tl.tmem` intrinsics.

3. **5th-gen tensor cores** via `tcgen05.mma`. Larger tile shapes (256×256×16
   for FP16, 256×256×32 for FP8). Triton dispatches automatically when block
   sizes are large enough; we set BLOCK_KV=256 in the autotune sweep.

4. **Cluster mode** (multiple SMs cooperating via distributed shared memory).
   Useful only for very long contexts (T_kv > 256k) where a single SM can't
   hold a meaningful KV tile in SRAM. Not relevant for our 32k--128k target.

5. **HBM3e bandwidth** (~8 TB/s on B200 vs ~3 TB/s on H100). Compressed KV
   stays the dominant winner: a 5x-compressed cache means 5x less HBM traffic
   per decode step, and on B200 that's >5x decode latency win vs dense fp16
   reads (we project ~0.003 ms/step at 32k context, vs cuDNN's ~0.006 ms).

Until tested on B200 hardware, this module re-exports the Hopper kernel.
"""
from typing import Optional

import torch

from .hqmq_attention_hopper import fused_hqmq_attention_hopper


def blackwell_available() -> bool:
    """True if running on Blackwell (SM 10.0+) hardware."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


def fused_hqmq_attention_blackwell(
    Q: torch.Tensor,
    K_idx: torch.Tensor, K_radius_q: torch.Tensor, K_radius_scale: torch.Tensor, joint_K: torch.Tensor,
    V_idx: torch.Tensor, V_radius_q: torch.Tensor, V_radius_scale: torch.Tensor, joint_V: torch.Tensor,
    r_qmax: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    block_q: int = 256,
    block_kv: int = 256,
    num_warps: int = 8,
    num_stages: int = 4,
    use_fp8_codebook: bool = True,
) -> torch.Tensor:
    """Blackwell-tuned HQMQ-attention.

    Falls back to the Hopper kernel (which is also FP8-codebook-capable and
    uses tensor cores). The differences vs Hopper are entirely in block sizes
    (256x256 tiles) and num_stages (4 instead of 3) — values that the device-
    capability dispatch in `hqmq_attention.fused_hqmq_attention_triton` will
    pick automatically when running on Blackwell.

    The TMEM / cluster-mode / tcgen05 paths described in this module's
    docstring require Triton intrinsics that are not yet exposed at the
    Python level; we leave them as design targets for the production-grade
    kernel.
    """
    # Re-dispatch to the Hopper kernel with Blackwell-aggressive tiles.
    # On actual Blackwell hardware, Triton's compiler picks tcgen05 MMA
    # instructions automatically once block sizes are large enough.
    return fused_hqmq_attention_hopper(
        Q,
        K_idx, K_radius_q, K_radius_scale, joint_K,
        V_idx, V_radius_q, V_radius_scale, joint_V,
        r_qmax=r_qmax, causal=causal, softmax_scale=softmax_scale,
        block_q=block_q, block_kv=block_kv,
        num_warps=num_warps, num_stages=num_stages,
        use_fp8_codebook=use_fp8_codebook,
    )


# ============================================================
# Projected performance (NOT MEASURED — design estimates only)
# ============================================================
#
# Workload: Mistral-class GQA, T_q=1, T_kv=32k, d_h=128, s192 codebook, fp16
#
#   Hardware        cuDNN SDPA   Our fused (current)   Our fused (TMEM+FP8)
#   --------------- -----------  -------------------   --------------------
#   RTX 4090 (Ada)  0.014 ms     0.028 ms (measured)   n/a (no TMA/FP8)
#   H100/H200       ~0.006 ms    ~0.012 ms (proj.)     ~0.004 ms (proj., FP8)
#   B100/B200       ~0.004 ms    ~0.007 ms (proj.)     ~0.002 ms (proj., FP8+TMEM)
#
# The "TMEM+FP8" column for Blackwell represents the ceiling when:
#   - the joint codebook is loaded once into TMEM via tcgen05.cp
#   - the QK^T and PV matmuls use cluster-shared TMEM operands
#   - the codebook is stored in FP8 (E4M3)
#
# At those numbers, the fused HQMQ kernel BEATS cuDNN FlashAttention in
# absolute terms (since cuDNN reads 5x more KV bytes per decode step).
