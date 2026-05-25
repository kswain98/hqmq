"""HQMQ: Hurwitz Quaternion Multiplicative Quantization for KV cache.

Public API for using HQMQ in your own attention layer or inference engine.

Basic usage
-----------

    import hqmq
    import torch

    # 1) Construct a quantizer for a 32-layer GQA model with 8 KV heads
    q = hqmq.HQMQ(n_layers=32, n_heads=8, secondary_size=192, radius_bits=6)

    # 2) Wrap in outlier-extraction for outlier-heavy models (Qwen, gpt-oss)
    q = hqmq.with_outlier_extraction(q, C=3.0)   # Med3x

    # 3) Quantize K and V on cache writes (HuggingFace-compatible fake-quant)
    K_quantized = q.quantize_K(K, layer_idx=0)
    V_quantized = q.quantize_V(V, layer_idx=0)

    # 4) Effective bits per element (including outlier overhead)
    print(f"{q.bits_per_value():.2f} bits/element")


Fused attention kernel (production path)
----------------------------------------

    from hqmq import fused_attention

    O = fused_attention(
        Q,                              # (B, H_q, T_q, d_h) fp16
        K_idx, K_radius_q, K_radius_scale, joint_K,
        V_idx, V_radius_q, V_radius_scale, joint_V,
        r_qmax=63, causal=True,
        # Optional: use_fp8_codebook=True on Hopper+ (auto-dispatched)
    )


GPU-specific paths
------------------

The default `fused_attention` dispatches block sizes / num_warps / num_stages
based on `torch.cuda.get_device_capability()`. For explicit access to the
hardware-specific variants:

    from hqmq.kernels import (
        fused_attention_ada,        # RTX 4090 / A100 / older (fp16 path)
        fused_attention_hopper,     # H100 / H200 (FP8 codebook optional)
        fused_attention_blackwell,  # B100 / B200 (design-stage)
    )
"""

# Public types
from src.quantizers.hqmq import HQMQQuantizer as HQMQ
from src.quantizers.hqmq_padded import PaddedHQMQQuantizer as PaddedHQMQ
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer
from src.quantizers.outlier_generic import OutlierAwareGenericQuantizer

# Public functions
from src.quantizers.hqmq_attention import (
    fused_hqmq_attention_triton as fused_attention,
    fused_hqmq_attention_torch as fused_attention_reference,
)

# Hardware-specific variants
from src.quantizers import hqmq_attention as kernels_ada  # noqa: F401
from src.quantizers import hqmq_attention_hopper as kernels_hopper  # noqa: F401
from src.quantizers import hqmq_attention_blackwell as kernels_blackwell  # noqa: F401


def with_outlier_extraction(quantizer, C: float = 3.0, *, generic: bool = False):
    """Wrap a quantizer with median-multiplier outlier extraction.

    Chunks whose norm exceeds C * median are kept at fp16; the rest go
    through the inner quantizer. C=3 is the universal default validated
    across all five main-paper models with no per-model tuning.

    If `generic=False` (default), uses the HQMQ-specialized wrapper
    (`OutlierAwareHQMQQuantizer`) which exposes additional bit-accounting.
    If `generic=True`, uses the type-agnostic wrapper which works on
    any KVQuantizer (used for the int4+Med3x disentanglement ablation).
    """
    if generic:
        return OutlierAwareGenericQuantizer(
            quantizer, chunk_dim=4,
            threshold_mode="median_mult", median_mult=C,
        )
    return OutlierAwareHQMQQuantizer(
        quantizer,
        threshold_mode="median_mult", median_mult=C,
    )


def device_info() -> dict:
    """Inspect the GPU and report which kernel variant will be auto-dispatched."""
    import torch
    if not torch.cuda.is_available():
        return {"gpu": None, "variant": "cpu", "fp8_codebook": False}
    cap = torch.cuda.get_device_capability()
    gpu = torch.cuda.get_device_name()
    if cap[0] >= 10:
        variant, fp8 = "blackwell", True
    elif cap[0] >= 9:
        variant, fp8 = "hopper", True
    elif cap[0] >= 8:
        variant, fp8 = "ada", False
    else:
        variant, fp8 = "ada (older)", False
    return {
        "gpu": gpu,
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "variant_auto_dispatched": variant,
        "fp8_codebook_supported": fp8,
    }


__all__ = [
    "HQMQ", "PaddedHQMQ",
    "OutlierAwareHQMQQuantizer", "OutlierAwareGenericQuantizer",
    "fused_attention", "fused_attention_reference",
    "with_outlier_extraction", "device_info",
    "kernels_ada", "kernels_hopper", "kernels_blackwell",
]

__version__ = "0.1.0"
