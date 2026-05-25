import torch

from .base import KVQuantizer


class NaivePerTokenIntQuantizer(KVQuantizer):
    """Per-token symmetric integer quantization for both K and V.

    For each (batch, head, token), computes the max absolute value across
    d_head, divides into 2^(bits-1)-1 levels. Symmetric (no zero-point).
    Reports `bits` bits per value plus the scale, amortized across d_head.
    """

    def __init__(self, bits: int = 4):
        assert 2 <= bits <= 8, "use bits in [2, 8]"
        self.bits = bits
        self._qmax = 2 ** (bits - 1) - 1

    def _qdq(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_head). Per (b, h, t) scale.
        x_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = x_max / self._qmax
        q = (x / scale).round().clamp(-self._qmax, self._qmax)
        return q * scale

    def quantize_K(self, k, layer_idx):
        return self._qdq(k)

    def quantize_V(self, v, layer_idx):
        return self._qdq(v)

    def bits_per_value(self):
        # bits per element + fp16 scale per row (d_head elements).
        # Amortized: bits + 16 / d_head. We don't know d_head here generically;
        # report just `bits` and let the experiment layer add overhead.
        return float(self.bits)
