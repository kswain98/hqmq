"""KVQuantizer base class.

A KVQuantizer is a callable that takes a K or V tensor of shape
(batch, n_kv_heads, seq_len, head_dim) and returns a same-shape fake-quantized
tensor (quantize then dequantize to fp16/bf16). The `QuantizedCache` in
model_utils.py invokes these on every cache write.

Subclasses must implement `quantize_K`, `quantize_V`, and `bits_per_value`.
The two are split because some methods quantize K and V differently
(e.g. KIVI: per-channel K, per-token V).
"""

from abc import ABC, abstractmethod
import torch


class KVQuantizer(ABC):
    @abstractmethod
    def quantize_K(self, k: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """k shape: (B, n_kv_heads, T_new, d_head). Returns same shape."""
        ...

    @abstractmethod
    def quantize_V(self, v: torch.Tensor, layer_idx: int) -> torch.Tensor:
        ...

    @abstractmethod
    def bits_per_value(self) -> float:
        """Effective bits per K (or V) scalar, including amortized scale/index overhead."""
        ...

    @property
    def name(self) -> str:
        return type(self).__name__
