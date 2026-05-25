"""HQMQ with d_head padding for models whose head dimension is not divisible by 4.

The motivating case is OpenAI's gpt-oss-20b ($d_h = 45$). We pad the head
dimension to the next multiple of 4 (e.g. 45 -> 48), apply HQMQ on the padded
tensor, and truncate back to the original $d_h$ on dequant. The padding
elements are stored at zero and never used downstream, so the only cost is the
wasted bit budget on the padded slots.

Bit accounting: per-element bits scale by $d_h^{\text{padded}} / d_h$. At
$d_h = 45$ padded to 48, the bit-budget overhead is $48 / 45 - 1 \approx 6.7\%$.
"""

import torch

from .base import KVQuantizer
from .hqmq import HQMQQuantizer


class PaddedHQMQQuantizer(KVQuantizer):
    """HQMQ wrapper that pads the head dimension to the next multiple of 4."""

    def __init__(self, n_layers: int, n_heads: int, d_head: int,
                 secondary_size: int = 24,
                 radius_bits: int = 4,
                 device: str = "cuda",
                 init: str = "random",
                 seed: int = 42):
        self.d_head_orig = d_head
        # Pad to next multiple of 4
        self.d_head_padded = ((d_head + 3) // 4) * 4
        self.pad_count = self.d_head_padded - d_head
        self.chunk_dim = 4
        self.radius_bits = radius_bits
        self.secondary_size = secondary_size

        # Construct the underlying HQMQ quantizer with the padded head dim
        # (HQMQQuantizer doesn't need d_head explicitly — it infers from input)
        self.inner = HQMQQuantizer(
            n_layers=n_layers, n_heads=n_heads,
            secondary_size=secondary_size, radius_bits=radius_bits,
            device=device, init=init, seed=seed,
        )

    def trainable_parameters(self):
        return self.inner.trainable_parameters()

    def set_training(self, training, tau=None):
        if hasattr(self.inner, "set_training"):
            self.inner.set_training(training, tau=tau)

    def _qdq(self, x: torch.Tensor, layer_idx: int, kv: int) -> torch.Tensor:
        B, H, T, d_h = x.shape
        if d_h == self.d_head_padded:
            # Already padded — just pass through
            x_padded = x
        else:
            assert d_h == self.d_head_orig, f"Expected d_head={self.d_head_orig}, got {d_h}"
            # Pad with zeros at the end
            pad_shape = (B, H, T, self.pad_count)
            zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, zeros], dim=-1)

        # Quantize-dequantize on the padded tensor
        if kv == 0:
            xq_padded = self.inner.quantize_K(x_padded, layer_idx)
        else:
            xq_padded = self.inner.quantize_V(x_padded, layer_idx)

        # Truncate back to the original head dim
        xq = xq_padded[..., :self.d_head_orig]
        return xq

    def quantize_K(self, k, layer_idx):
        return self._qdq(k, layer_idx, kv=0)

    def quantize_V(self, v, layer_idx):
        return self._qdq(v, layer_idx, kv=1)

    def bits_per_value(self):
        # Inner gives bits/element on the padded head; rescale by padded/orig ratio
        # to get bits/element on the original (truncated) head dimension.
        return self.inner.bits_per_value() * self.d_head_padded / self.d_head_orig

    @property
    def name(self):
        return f"PaddedHQMQ_d{self.d_head_orig}to{self.d_head_padded}_{self.inner.name if hasattr(self.inner, 'name') else 'hqmq'}"
