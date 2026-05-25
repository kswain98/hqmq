"""Hardware-specific HQMQ kernel re-exports.

For explicit access to a particular variant (bypassing the auto-dispatch).
"""

from src.quantizers.hqmq_attention import (
    fused_hqmq_attention_triton as fused_attention_ada,
    fused_hqmq_attention_torch as fused_attention_reference,
)

from src.quantizers.hqmq_attention_hopper import (
    fused_hqmq_attention_hopper as fused_attention_hopper,
    encode_joint_fp8,
    hopper_available,
)

from src.quantizers.hqmq_attention_blackwell import (
    fused_hqmq_attention_blackwell as fused_attention_blackwell,
    blackwell_available,
)

# Standalone dequant kernel
from src.quantizers.hqmq_kernel import (
    hqmq_decode_triton as decode_dequant,
)

__all__ = [
    "fused_attention_ada",
    "fused_attention_hopper",
    "fused_attention_blackwell",
    "fused_attention_reference",
    "decode_dequant",
    "encode_joint_fp8",
    "hopper_available",
    "blackwell_available",
]
