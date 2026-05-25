"""Flat (non-multiplicative) spherical VQ — baseline for the HQMQ ablation.

Stores `codebook_size` independent random unit quaternions per (layer, head, K/V),
without any algebraic composition rule. Same encode/decode interface as HQMQ:
look up the nearest unit-quaternion codeword by inner product, store its index
plus a quantized radius.

At matched `codebook_size`, this isolates the contribution of quaternion
multiplication in HQMQ: if HQMQ ≈ FlatSphericalQuantizer at same bits, then
the multiplicative structure is a *storage* trick (24× fewer params); if HQMQ
exceeds it, the structure is a quality contribution too.
"""

import math
from typing import Optional

import torch

from .base import KVQuantizer


class FlatSphericalQuantizer(KVQuantizer):
    """Direct comparison-counterpart to HQMQQuantizer: codebook of size K (random
    unit quaternions) per (layer, head, K|V), no multiplicative structure."""

    def __init__(self, n_layers: int, n_heads: int,
                 codebook_size: int = 576,    # matched to HQMQ s24 (24*24)
                 radius_bits: int = 4,
                 device: str = "cuda",
                 seed: int = 42):
        self.chunk_dim = 4
        self.codebook_size = codebook_size
        self.radius_bits = radius_bits
        self.training = False
        self.tau = 1.0

        g = torch.Generator().manual_seed(seed)
        raw = torch.randn(n_layers, n_heads, 2, codebook_size, 4, generator=g)
        # Normalize once to unit quaternions; codebook is fixed (no learning).
        codebook = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self._codebook = codebook.to(device=device, dtype=torch.float32)  # (L, H, 2, K, 4)

    def trainable_parameters(self):
        return []

    def set_training(self, training: bool, tau: Optional[float] = None):
        self.training = training
        if tau is not None:
            self.tau = tau

    def _refresh_rotations(self):
        pass

    def _qdq(self, x, layer_idx, kv):
        B, H, T, d_head = x.shape
        if d_head % 4 != 0:
            raise ValueError(f"d_head={d_head} not divisible by 4")
        n_chunks = d_head // 4
        x_chunked = x.view(B, H, T, n_chunks, 4)

        radius = x_chunked.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        direction = x_chunked / radius

        codebook = self._codebook[layer_idx, :, kv].to(x.dtype)   # (H, K, 4)
        sims = torch.einsum("bhtnc,hkc->bhtnk", direction, codebook)
        idx = sims.argmax(dim=-1)
        idx_gather = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 1, 4)
        cb_b = codebook.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(B, H, T, n_chunks, codebook.shape[1], 4)
        direction_q = cb_b.gather(dim=-2, index=idx_gather).squeeze(-2)

        if self.radius_bits >= 16:
            radius_q = radius
        else:
            r_max = radius.amax(dim=-2, keepdim=True).clamp(min=1e-8)
            qmax = (1 << self.radius_bits) - 1
            scale = r_max / qmax
            radius_q = (radius / scale).round().clamp(0, qmax) * scale

        out = radius_q * direction_q
        return out.view(B, H, T, d_head)

    def quantize_K(self, k, layer_idx):
        return self._qdq(k, layer_idx, kv=0)

    def quantize_V(self, v, layer_idx):
        return self._qdq(v, layer_idx, kv=1)

    def bits_per_value(self):
        return (math.log2(self.codebook_size) + self.radius_bits) / 4.0

    @property
    def name(self) -> str:
        return f"FlatVQ_k{self.codebook_size}_r{self.radius_bits}"
