"""Outlier-aware HQMQ wrapper.

Modern open LLMs (Qwen2.5, Llama-3) have a small fraction of K-chunks with
extreme magnitudes (sometimes 100×+ above the median). With uniform per-token
max scaling these outliers dominate the scale and zero out the bulk; without
them most chunks get norm-quantized to 0.

This wrapper handles them by *extracting* outliers: chunks above a per-batch
quantile threshold are kept at fp16, the rest are quantized via the inner
HQMQ. The threshold is chosen *per call* from the current batch's statistics
— so the calibration-free property is preserved.

Bit accounting (per chunk):
  - Non-outlier (fraction = 1 - p): inner HQMQ bits
  - Outlier (fraction = p): 64 bits (4 fp16 values)
  - Metadata (1 bit per chunk to mark outliers): 1 bit

Effective per-element bits = (1 - p) * inner_bits_per_elem
                             + p * 16  (fp16 elements)
                             + 1/4     (metadata, 1 bit per 4-element chunk)
"""

import torch

from .base import KVQuantizer
from .hqmq import HQMQQuantizer


class OutlierAwareHQMQQuantizer(KVQuantizer):
    """Outlier-extraction wrapper around HQMQQuantizer."""

    def __init__(self, inner: HQMQQuantizer, outlier_fraction: float = 0.01,
                 threshold_mode: str = "fraction",  # "fraction" or "median_mult"
                 median_mult: float = 20.0):
        assert isinstance(inner, HQMQQuantizer), "wrapper expects HQMQQuantizer"
        self.inner = inner
        self.outlier_fraction = outlier_fraction
        self.threshold_mode = threshold_mode
        self.median_mult = median_mult
        self.chunk_dim = inner.chunk_dim
        # Track average outlier rate (for bits-per-value reporting)
        self._observed_outlier_rate = 0.0
        self._n_observations = 0

    def trainable_parameters(self):
        return self.inner.trainable_parameters() if hasattr(self.inner, "trainable_parameters") else []

    def set_training(self, training, tau=None):
        if hasattr(self.inner, "set_training"):
            self.inner.set_training(training, tau=tau)

    def _refresh_rotations(self):
        if hasattr(self.inner, "_refresh_rotations"):
            self.inner._refresh_rotations()

    def _qdq(self, x: torch.Tensor, layer_idx: int, kv: int) -> torch.Tensor:
        B, H, T, d_head = x.shape
        n_chunks = d_head // self.chunk_dim
        x_chunked = x.view(B, H, T, n_chunks, self.chunk_dim)
        chunk_norms = x_chunked.norm(dim=-1)  # (B, H, T, n_chunks)

        # Threshold per (head)
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
        # Track empirical outlier rate
        rate = outlier_mask.float().mean().item()
        self._observed_outlier_rate = (self._observed_outlier_rate * self._n_observations + rate) / (self._n_observations + 1)
        self._n_observations += 1

        # Quantize everything via inner
        if kv == 0:
            hqmq_full = self.inner.quantize_K(x, layer_idx)
        else:
            hqmq_full = self.inner.quantize_V(x, layer_idx)
        hqmq_chunked = hqmq_full.view(B, H, T, n_chunks, self.chunk_dim)

        # For outlier positions, restore original fp16 chunks
        result = torch.where(outlier_mask.unsqueeze(-1), x_chunked, hqmq_chunked)
        return result.view(B, H, T, d_head)

    def quantize_K(self, k, layer_idx):
        return self._qdq(k, layer_idx, kv=0)

    def quantize_V(self, v, layer_idx):
        return self._qdq(v, layer_idx, kv=1)

    def bits_per_value(self):
        # If we've observed actual outlier rate, use it; else use the nominal fraction.
        # 0.03 fallback matches the empirical rate seen at median_mult=3 on modern LLMs
        # and is consistent with OutlierAwareGenericQuantizer.
        if self._n_observations > 0:
            p = self._observed_outlier_rate
        else:
            p = self.outlier_fraction if self.threshold_mode == "fraction" else 0.03
        inner_bpv = self.inner.bits_per_value()
        outlier_bpv = 16
        meta_bpv = 1 / self.chunk_dim
        return (1 - p) * inner_bpv + p * outlier_bpv + meta_bpv

    @property
    def name(self):
        if self.threshold_mode == "fraction":
            p_pct = int(self.outlier_fraction * 100)
            return f"Outlier{p_pct}p_{self.inner.name}"
        else:
            return f"OutlierMed{self.median_mult:.0f}_{self.inner.name}"
