"""Additive Vector Quantization — CommVQ-style baseline for HQMQ comparison.

For each 4-element chunk, approximate as the sum of `n_codebooks` codewords,
one from each codebook, via greedy residual search:

    x ≈ c_1[i_1] + c_2[i_2] + ... + c_K[i_K]

Bits per chunk: K * log2(codebook_size). For matched bits with HQMQ s24
(log2(576) = 9.17 bits), use K=2, codebook_size=24 (2 * 4.58 = 9.17 bits).

Each codebook is initialized as random vectors (no training). With training
(k-means + refinement) this would become full CommVQ. Without training, it
serves as the *uncalibrated additive baseline* — the ablation that shows
HQMQ's structural advantage over flat additive composition.
"""

import math

import torch

from .base import KVQuantizer


class AdditiveVQQuantizer(KVQuantizer):
    """Greedy additive VQ baseline. No training — codebooks are fixed random."""

    def __init__(self, n_layers: int, n_heads: int,
                 n_codebooks: int = 2,
                 codebook_size: int = 24,
                 radius_bits: int = 4,
                 device: str = "cuda",
                 seed: int = 42):
        self.chunk_dim = 4
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.radius_bits = radius_bits

        g = torch.Generator().manual_seed(seed)
        # codebooks: (n_codebooks, n_layers, n_heads, 2, codebook_size, 4)
        # Initialized small so initial residuals make sense; we scale to match data norm.
        raw = torch.randn(n_codebooks, n_layers, n_heads, 2, codebook_size, 4, generator=g) * 0.1
        self._codebooks = raw.to(device=device, dtype=torch.float32)

    def trainable_parameters(self):
        return []

    def _refresh_rotations(self):
        pass

    def _qdq(self, x, layer_idx, kv):
        B, H, T, d_head = x.shape
        n_chunks = d_head // 4
        x_chunked = x.view(B, H, T, n_chunks, 4)

        approx = torch.zeros_like(x_chunked)
        for k in range(self.n_codebooks):
            cb = self._codebooks[k, layer_idx, :, kv].to(x.dtype)        # (H, K, 4)
            residual = x_chunked - approx
            sims = torch.einsum("bhtnc,hkc->bhtnk", residual, cb)
            idx = sims.argmax(dim=-1)                                    # (B, H, T, n_chunks)
            idx_g = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 1, 4)
            cb_b = cb.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(B, H, T, n_chunks, cb.shape[1], 4)
            chosen = cb_b.gather(dim=-2, index=idx_g).squeeze(-2)
            approx = approx + chosen

        # Apply per-chunk radius correction (optional but helps; matches HQMQ's structure)
        if self.radius_bits >= 16:
            return approx.view(B, H, T, d_head)

        # Per-chunk radius from input; we apply scalar quantization to the residual norm.
        # In additive VQ, we don't separately track radius — the codebooks should absorb it.
        # For fairness with HQMQ, we add a per-token-max scale on the approx itself.
        # (radius_bits >= 16 already returned early above, so we always quantize here.)
        x_norm = x_chunked.norm(dim=-1, keepdim=True)
        approx_norm = approx.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        r_max = x_norm.amax(dim=-2, keepdim=True).clamp(min=1e-8)
        qmax = (1 << self.radius_bits) - 1
        s = r_max / qmax
        x_norm_q = (x_norm / s).round().clamp(0, qmax) * s
        scale = x_norm_q / approx_norm
        approx = approx * scale

        return approx.view(B, H, T, d_head)

    def quantize_K(self, k, layer_idx):
        return self._qdq(k, layer_idx, kv=0)

    def quantize_V(self, v, layer_idx):
        return self._qdq(v, layer_idx, kv=1)

    def bits_per_value(self):
        # Per chunk: n_codebooks * log2(codebook_size) bits for the indices + radius_bits
        return (self.n_codebooks * math.log2(self.codebook_size) + self.radius_bits) / 4.0

    @property
    def name(self) -> str:
        return f"AdditiveVQ_n{self.n_codebooks}_k{self.codebook_size}_r{self.radius_bits}"
