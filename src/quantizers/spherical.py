"""Spherical-codebook product quantization for KV cache.

Chunks each (batch, head, token, d_head) vector along d_head into groups of
`chunk_dim` (default 4). Per chunk: extract radius and unit direction.
Direction is quantized to the nearest vertex of a fixed optimal spherical
code (24-cell on S^3 for chunk_dim=4, cross-polytope for chunk_dim=2 etc).
Radius is scalar-quantized.

This is the structural piece of the proposed method. The "output-aware"
training of bit allocation and any learned per-head rotation gets layered
on top later — see SphericalQuantizerOutputAware (TODO).
"""

from itertools import product
import math

import torch

from .base import KVQuantizer


def _make_24cell_codebook(dtype=torch.float32) -> torch.Tensor:
    """24-cell vertices on S^3: optimal kissing-number arrangement of 24 points.

    Two orbits:
      - 8 unit-axis vectors: (±1, 0, 0, 0) and permutations
      - 16 sign vectors: (±0.5, ±0.5, ±0.5, ±0.5)

    Returns shape (24, 4). All rows have unit L2 norm.
    """
    cb = []
    for i in range(4):
        for s in (+1.0, -1.0):
            v = [0.0] * 4
            v[i] = s
            cb.append(v)
    for signs in product((+1.0, -1.0), repeat=4):
        cb.append([s * 0.5 for s in signs])
    t = torch.tensor(cb, dtype=dtype)
    assert t.shape == (24, 4)
    norms = t.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    return t


def _make_cross_polytope_codebook(dim: int, dtype=torch.float32) -> torch.Tensor:
    """Cross-polytope (±e_i): 2*dim vectors on S^{dim-1}. Simplest spherical code."""
    cb = []
    for i in range(dim):
        for s in (+1.0, -1.0):
            v = [0.0] * dim
            v[i] = s
            cb.append(v)
    return torch.tensor(cb, dtype=dtype)


def _make_e8_codebook(dtype=torch.float32) -> torch.Tensor:
    """E8 lattice minimum-norm vectors (240 vectors on S^7).

    Two orbits, after normalizing to unit L2:
      - 112 of form (±1/√2, ±1/√2, 0, 0, 0, 0, 0, 0) with all 8C2=28 position
        pairs × 4 sign combos = 112.
      - 128 of form (±1/(2√2)) × 8 entries, with an EVEN number of minus signs
        (this is the half-integer orbit; 2^7 = 128).

    Minimum angular distance ≈ acos(1/2) = 60°, optimal kissing in 8D.
    Result is log2(240) ≈ 7.91 bits per chunk.
    """
    from itertools import combinations, product as iproduct
    import math as _m

    cb = []
    # Orbit 1: ±1/√2 at two positions
    norm1 = 1.0 / _m.sqrt(2.0)
    for i, j in combinations(range(8), 2):
        for s1, s2 in iproduct((+1.0, -1.0), repeat=2):
            v = [0.0] * 8
            v[i] = s1 * norm1
            v[j] = s2 * norm1
            cb.append(v)
    # Orbit 2: all ±1/(2√2), even number of minuses
    norm2 = 1.0 / (2.0 * _m.sqrt(2.0))
    for signs in iproduct((+1.0, -1.0), repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            cb.append([s * norm2 for s in signs])

    t = torch.tensor(cb, dtype=dtype)
    assert t.shape == (240, 8), f"expected 240 codewords, got {t.shape}"
    norms = t.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    return t


def make_spherical_codebook(chunk_dim: int, kind: str = "auto") -> torch.Tensor:
    """Return a spherical codebook for the given chunk dimension.

    chunk_dim=2 / "cross":   4-vertex cross-polytope (2 bits)
    chunk_dim=4 / "24cell":  24-cell (log2(24) ≈ 4.585 bits)
    chunk_dim=8 / "e8":      E8 root system, 240 unit vectors (log2(240) ≈ 7.91 bits)
    chunk_dim=8 / "cross":   16-vertex cross-polytope (4 bits) — baseline only
    Other chunk_dim: cross-polytope.
    """
    if kind == "auto":
        kind = {4: "24cell", 8: "e8"}.get(chunk_dim, "cross")
    if kind == "24cell":
        assert chunk_dim == 4
        return _make_24cell_codebook()
    if kind == "e8":
        assert chunk_dim == 8
        return _make_e8_codebook()
    if kind == "cross":
        return _make_cross_polytope_codebook(chunk_dim)
    raise ValueError(f"unknown codebook kind={kind}")


class SphericalProductQuantizerJL(KVQuantizer):
    """Spherical codebook quantization + JL residual correction.

    Primary: 24-cell direction + scalar radius (same as SphericalProductQuantizer).
    Residual: Johnson-Lindenstrauss projection to jl_dim, 1-bit sign quantization.
    Total per chunk: log2(|codebook|) + radius_bits + jl_dim bits (+overhead).

    The residual sign-scale is computed from the per-chunk residual norm
    (currently free; production version would calibrate a global constant
    or use per-token scale to reduce overhead).
    """

    def __init__(self, chunk_dim: int = 4, radius_bits: int = 4, jl_dim: int = 4,
                 codebook: str = "auto", seed: int = 42, training: bool = False, tau: float = 1.0):
        self.chunk_dim = chunk_dim
        self.radius_bits = radius_bits
        self.jl_dim = jl_dim
        self._codebook = make_spherical_codebook(chunk_dim, kind=codebook)
        self._codebook_size = self._codebook.shape[0]
        # Orthonormal-ish JL projection
        g = torch.Generator().manual_seed(seed)
        M = torch.randn(max(jl_dim, chunk_dim), chunk_dim, generator=g)
        Q, _ = torch.linalg.qr(M)
        self._R = Q[:jl_dim]  # (jl_dim, chunk_dim) orthonormal rows
        self._state_cache: dict = {}
        # Training-time soft-codebook switches (STE-to-hard for forward; smooth grad in backward)
        self.training = training
        self.tau = tau

    def set_training(self, training: bool, tau: float = None):
        self.training = training
        if tau is not None:
            self.tau = tau

    def _get_state(self, ref: torch.Tensor):
        key = (ref.device, ref.dtype)
        if key not in self._state_cache:
            self._state_cache[key] = (
                self._codebook.to(device=ref.device, dtype=ref.dtype),
                self._R.to(device=ref.device, dtype=ref.dtype),
            )
        return self._state_cache[key]

    def _qdq(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, d_head = x.shape
        if d_head % self.chunk_dim != 0:
            raise ValueError(f"d_head={d_head} not divisible by chunk_dim={self.chunk_dim}")
        n_chunks = d_head // self.chunk_dim
        x_chunked = x.view(B, H, T, n_chunks, self.chunk_dim)
        codebook, R = self._get_state(x)

        # Primary spherical quantization
        radius = x_chunked.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        direction = x_chunked / radius
        sims = torch.einsum("bhtnc,kc->bhtnk", direction, codebook)
        # Hard argmax for the actual selected codeword (used in forward path)
        idx = sims.argmax(dim=-1)
        direction_q_hard = codebook[idx]
        if self.training:
            # Soft weighted-avg for gradient propagation; STE-to-hard for forward value.
            weights = torch.softmax(sims / self.tau, dim=-1)
            direction_q_soft = torch.einsum("bhtnk,kc->bhtnc", weights, codebook)
            direction_q = direction_q_soft + (direction_q_hard - direction_q_soft).detach()
        else:
            direction_q = direction_q_hard

        if self.radius_bits >= 16:
            radius_q = radius
        else:
            r_max = radius.amax(dim=-2, keepdim=True).clamp(min=1e-8)
            qmax = (1 << self.radius_bits) - 1
            scale = r_max / qmax
            radius_q_hard = (radius / scale).round().clamp(0, qmax) * scale
            if self.training:
                # Soft round: linear interp via STE (smooth backward, hard forward)
                radius_q = radius + (radius_q_hard - radius).detach()
            else:
                radius_q = radius_q_hard

        primary = radius_q * direction_q  # (B, H, T, n_chunks, chunk_dim)

        # JL residual refinement
        if self.jl_dim > 0:
            residual = x_chunked - primary
            y = torch.einsum("bhtnc,jc->bhtnj", residual, R)
            signs_hard = torch.sign(y)
            if self.training:
                # Soft sign via tanh; STE-to-hard for forward.
                signs_soft = torch.tanh(y / self.tau)
                signs = signs_soft + (signs_hard - signs_soft).detach()
            else:
                signs = signs_hard
            res_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            alpha = res_norm / (self.jl_dim ** 0.5)
            y_q = signs * alpha
            residual_q = torch.einsum("bhtnj,jc->bhtnc", y_q, R)
        else:
            residual_q = 0.0

        out = primary + residual_q
        return out.view(B, H, T, d_head)

    def quantize_K(self, k, layer_idx):
        return self._qdq(k)

    def quantize_V(self, v, layer_idx):
        return self._qdq(v)

    def bits_per_value(self):
        bits_per_chunk = math.log2(self._codebook_size) + self.radius_bits + self.jl_dim
        return bits_per_chunk / self.chunk_dim


class SphericalProductQuantizer(KVQuantizer):
    """Product spherical-code quantization for K and V.

    Algorithm per chunk:
      1. Split x_chunk (shape ..., chunk_dim) into radius r = ||x_chunk||
         and direction u = x_chunk / r.
      2. Quantize u to nearest codebook vertex c (argmax dot product).
      3. Quantize r to `radius_bits` uniform-symmetric levels per-token.
      4. Reconstruct as r_q * c.

    Per-element bit budget:
      log2(|codebook|) / chunk_dim    (direction)
      + radius_bits / chunk_dim       (radius)
      + 16 / d_head                   (per-token scale overhead, amortized)
    """

    def __init__(self, chunk_dim: int = 4, radius_bits: int = 8, codebook: str = "auto"):
        self.chunk_dim = chunk_dim
        self.radius_bits = radius_bits
        self._codebook = make_spherical_codebook(chunk_dim, kind=codebook)
        self._codebook_size = self._codebook.shape[0]
        self._codebook_cache: dict = {}  # (device, dtype) → tensor

    def _get_codebook(self, ref: torch.Tensor) -> torch.Tensor:
        key = (ref.device, ref.dtype)
        if key not in self._codebook_cache:
            self._codebook_cache[key] = self._codebook.to(device=ref.device, dtype=ref.dtype)
        return self._codebook_cache[key]

    def _qdq(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_head). Chunk last dim into d_head / chunk_dim groups.
        B, H, T, d_head = x.shape
        if d_head % self.chunk_dim != 0:
            raise ValueError(f"d_head={d_head} not divisible by chunk_dim={self.chunk_dim}")
        n_chunks = d_head // self.chunk_dim
        x_chunked = x.view(B, H, T, n_chunks, self.chunk_dim)

        # Direction quantization via codebook NN search
        radius = x_chunked.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, H, T, n_chunks, 1)
        direction = x_chunked / radius                                 # (B, H, T, n_chunks, chunk_dim)
        codebook = self._get_codebook(direction)                       # (K, chunk_dim)
        # Similarities (closest codeword by inner product since codewords are unit-norm)
        sims = torch.einsum("bhtnc,kc->bhtnk", direction, codebook)    # (B, H, T, n_chunks, K)
        idx = sims.argmax(dim=-1)                                      # (B, H, T, n_chunks)
        direction_q = codebook[idx]                                    # (B, H, T, n_chunks, chunk_dim)

        # Radius quantization (per-token symmetric, applied across all chunks of a token)
        if self.radius_bits >= 16:
            radius_q = radius
        else:
            r_max = radius.amax(dim=-2, keepdim=True).clamp(min=1e-8)  # (B, H, T, 1, 1)
            qmax = (1 << self.radius_bits) - 1
            scale = r_max / qmax
            radius_q = (radius / scale).round().clamp(0, qmax) * scale

        out = radius_q * direction_q                                   # (B, H, T, n_chunks, chunk_dim)
        return out.view(B, H, T, d_head)

    def quantize_K(self, k, layer_idx):
        return self._qdq(k)

    def quantize_V(self, v, layer_idx):
        return self._qdq(v)

    def bits_per_value(self):
        bits_direction = math.log2(self._codebook_size)
        bits_per_chunk = bits_direction + self.radius_bits
        return bits_per_chunk / self.chunk_dim
