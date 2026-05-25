"""Generic outlier-extraction wrapper around any KVQuantizer.

Mirrors OutlierAwareHQMQQuantizer but works on any inner quantizer (e.g.
NaivePerTokenIntQuantizer) so we can disentangle the contributions of
(a) outlier extraction and (b) the HQMQ codebook structure.

Threshold and bit-accounting logic match OutlierAwareHQMQQuantizer exactly;
only the inner type is generalized.
"""

import torch

from .base import KVQuantizer


class OutlierAwareGenericQuantizer(KVQuantizer):
    """Outlier-extraction wrapper around any inner KVQuantizer.

    Bit accounting (per chunk_dim = 4):
      effective bits/elem = (1 - p) * inner_bits_per_elem
                          + p * 16
                          + 1/4
    """

    def __init__(self, inner: KVQuantizer, chunk_dim: int = 4,
                 outlier_fraction: float = 0.03,
                 threshold_mode: str = "median_mult",
                 median_mult: float = 3.0):
        self.inner = inner
        self.chunk_dim = chunk_dim
        self.outlier_fraction = outlier_fraction
        self.threshold_mode = threshold_mode
        self.median_mult = median_mult
        self._observed_outlier_rate = 0.0
        self._n_observations = 0

    def trainable_parameters(self):
        return self.inner.trainable_parameters() if hasattr(self.inner, "trainable_parameters") else []

    def set_training(self, training, tau=None):
        if hasattr(self.inner, "set_training"):
            self.inner.set_training(training, tau=tau)

    def _qdq(self, x: torch.Tensor, layer_idx: int, kv: int) -> torch.Tensor:
        B, H, T, d_head = x.shape
        n_chunks = d_head // self.chunk_dim
        x_chunked = x.view(B, H, T, n_chunks, self.chunk_dim)
        chunk_norms = x_chunked.norm(dim=-1)  # (B, H, T, n_chunks)

        flat = chunk_norms.permute(1, 0, 2, 3).reshape(H, -1).float()
        if self.threshold_mode == "fraction":
            q_target = 1.0 - self.outlier_fraction
            threshold = torch.quantile(flat, q_target, dim=-1)
        elif self.threshold_mode == "median_mult":
            med = torch.quantile(flat, 0.5, dim=-1)
            threshold = med * self.median_mult
        else:
            raise ValueError(self.threshold_mode)
        outlier_mask = chunk_norms > threshold.view(1, H, 1, 1).to(chunk_norms.dtype)
        rate = outlier_mask.float().mean().item()
        self._observed_outlier_rate = (
            self._observed_outlier_rate * self._n_observations + rate
        ) / (self._n_observations + 1)
        self._n_observations += 1

        # Quantize everything via inner
        if kv == 0:
            inner_full = self.inner.quantize_K(x, layer_idx)
        else:
            inner_full = self.inner.quantize_V(x, layer_idx)
        inner_chunked = inner_full.view(B, H, T, n_chunks, self.chunk_dim)

        # For outlier positions, restore fp16 chunks
        result = torch.where(outlier_mask.unsqueeze(-1), x_chunked, inner_chunked)
        return result.view(B, H, T, d_head)

    def quantize_K(self, k, layer_idx):
        return self._qdq(k, layer_idx, kv=0)

    def quantize_V(self, v, layer_idx):
        return self._qdq(v, layer_idx, kv=1)

    def bits_per_value(self):
        if self._n_observations > 0:
            p = self._observed_outlier_rate
        else:
            p = self.outlier_fraction if self.threshold_mode == "fraction" else 0.03
        inner_bpv = self.inner.bits_per_value()
        outlier_bpv = 16.0
        meta_bpv = 1.0 / self.chunk_dim
        return (1 - p) * inner_bpv + p * outlier_bpv + meta_bpv

    @property
    def name(self):
        if self.threshold_mode == "fraction":
            return f"Outlier{int(self.outlier_fraction * 100)}p_{self.inner.__class__.__name__}"
        return f"OutlierMed{self.median_mult:.0f}_{self.inner.__class__.__name__}"
